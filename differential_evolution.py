"""Vectorised differential evolution on the GPU (or CPU) with PyTorch.

DE/rand/1/bin, optionally split into independent sub-populations ("islands").

The optimiser is *population-vectorised*: the whole population lives in one
tensor ``P`` of shape ``(pop, *dim)`` and every step evaluates all candidates at
once. Your objective ``f`` therefore receives the **whole population** and must
return one cost per individual (shape ``(pop,)``). Pass
``f_for_individuals=True`` if you'd rather write ``f`` one individual at a time
(convenient, but much slower -- it falls back to a Python loop).

:func:`optimize` is the convenient entry point; :class:`DifferentialEvolver` is
the object underneath if you want to drive the generations yourself.

Nothing here knows about your problem domain; it is a generic black-box
minimiser. Keep it that way.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional, Sequence, Union

import torch

from progress import log, progress_bar

Tensor = torch.Tensor
Number = Union[int, float]
# A schedule is anything tofunc() understands: a constant, a (lo, hi) range that
# is resampled every call, or a zero-arg callable returning a float.
Schedule = Union[Number, Sequence[Number], slice, Callable[[], float]]

# The 'multinomial' partner sampler draws 3 distinct partners from an island,
# excluding the individual itself -- so an island needs at least 4 members.
MIN_BLOCK_FOR_MULTINOMIAL = 4


# --------------------------------------------------------------------------- #
# small helpers (these used to live in helpers.py)
# --------------------------------------------------------------------------- #
def randfloat(a: float, b: float) -> float:
    """Uniform sample from ``[a, b)``."""
    return a + (b - a) * random.random()


def tofunc(x: Schedule) -> Callable[[], float]:
    """Turn a constant / ``(lo, hi)`` range / callable into a zero-arg sampler.

    Used for the mutation and crossover schedules so they can be either fixed
    (``0.8``) or jittered each generation (``(0.3, 0.9)`` resamples uniformly).
    """
    if isinstance(x, (int, float)):
        return lambda: float(x)
    if isinstance(x, slice):
        return lambda: randfloat(x.start, x.stop)
    if isinstance(x, (tuple, list)):
        return lambda: randfloat(x[0], x[1])
    return x  # type: ignore[return-value]


def _get_block(k: int, i: int, j: int) -> Tensor:
    # A k x k all-ones-off-diagonal block, padded with i zero-blocks before and
    # j after (horizontally). Used to build get_block_eye.
    a = 1 - torch.eye(k)
    z = torch.zeros_like(a)
    return torch.cat([z] * i + [a] + [z] * j, dim=1)


def get_block_eye(k: int, n: int) -> Tensor:
    """Block-diagonal "everyone-but-myself" mask for ``n`` blocks of size ``k``.

    Row ``r`` has 1s exactly on the individuals that share ``r``'s island
    (excluding ``r`` itself), so ``torch.multinomial`` over a row samples a
    partner from the same island and never ``r``. Only used by the
    ``'multinomial'`` index-sampling path -- note this is a dense
    ``(k*n) x (k*n)`` matrix, so it is quadratic in the total population size in
    both time and memory.
    """
    return torch.cat([_get_block(k, i, n - i - 1) for i in range(n)], dim=0)


def individual2population(f: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Lift a per-individual function to a per-population one.

    Convenient but slow (a Python loop over the population) -- prefer writing
    ``f`` so it operates on the whole batch at once.
    """
    return lambda P: torch.stack([f(p) for p in P])


def pick_device(
    device: Optional[Union[str, torch.device]] = None,
    use_cuda: Optional[bool] = None,
) -> torch.device:
    """Resolve a device: explicit ``device`` wins, then the legacy ``use_cuda``
    flag, else CUDA if available and CPU otherwise."""
    if device is not None:
        return torch.device(device)
    if use_cuda is not None:
        return torch.device("cuda" if use_cuda else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# the optimiser
# --------------------------------------------------------------------------- #
class DifferentialEvolver:
    """A population of candidate solutions evolved by DE/rand/1/bin.

    Parameters
    ----------
    f:
        Objective. Receives the whole population tensor ``(pop, *dim)`` and
        returns a cost per individual ``(pop,)``. Lower is better (unless
        ``maximize``).
    initial_pop:
        Starting population tensor ``(pop, *dim)``. If ``None``, one is sampled
        from ``randn(pop_size * num_populations, *dim)``.
    pop_size, dim:
        Size of each island and the shape of one individual. Ignored when
        ``initial_pop`` is given.
    num_populations:
        Split the population into this many independent islands; partners for
        the DE update are drawn only from within an island. ``shuffle()``
        reshuffles individuals across islands (call it occasionally so the
        islands can share progress).
    proj_to_domain:
        Map any candidate back into the feasible set (e.g. clamp coordinates).
        Applied to the initial population and to every candidate.
    maximize:
        Maximise ``f`` instead of minimising it.
    device:
        Where to run: ``'cuda'``, ``'cpu'``, a ``torch.device``, or ``None`` to
        auto-detect (CUDA when available).
    use_cuda:
        Deprecated alias kept for backwards compatibility; ``device`` wins.
    dtype:
        dtype for the population. Everything downstream (the objective via
        ``x.dtype``, the crossover mask via ``.to(P)``, the mutation arithmetic)
        inherits it from ``P``, so this pins the precision of the whole engine.
        Defaults to ``torch.double``: DE compares candidate costs that can sit a
        hair apart, and float32 runs out of digits sooner than you'd like. Pass
        ``torch.float`` for speed/memory, or ``None`` to keep whatever
        ``initial_pop`` already is.
    prob_choosing_method:
        How DE partner indices are drawn: ``'randint'`` (fast, O(pop) per step,
        allows the occasional self/duplicate pick), ``'multinomial'``
        (within-island, guaranteed no self-pick, but O(pop^2) per step in time
        *and* memory -- see :func:`get_block_eye`, and it needs at least
        ``MIN_BLOCK_FOR_MULTINOMIAL`` individuals per island), or
        ``'automatic'`` (``'multinomial'`` only for small islands, 4 to 7, where
        avoiding self-picks matters more than the cost; ``'randint'``
        otherwise).
    chromosome_replacement_dimension:
        Granularity of the binomial crossover. ``None`` => every scalar can be
        swapped independently; ``k`` => crossover decisions are shared across
        the trailing ``len(dim) - k`` axes (e.g. for points of shape ``(n, d)``,
        ``1`` swaps whole points at a time); ``0`` => the whole individual is
        swapped or not (rarely what you want).
    """

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        initial_pop: Optional[Tensor] = None,
        pop_size: int = 50,
        dim: Union[int, Sequence[int]] = (1,),
        num_populations: int = 1,
        proj_to_domain: Callable[[Tensor], Tensor] = lambda x: x,
        f_for_individuals: bool = False,
        proj_for_individuals: Optional[bool] = None,
        maximize: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        use_cuda: Optional[bool] = None,
        dtype: Optional[torch.dtype] = torch.double,
        prob_choosing_method: str = "automatic",
        chromosome_replacement_dimension: Optional[int] = None,
    ):
        if isinstance(dim, int):
            dim = (dim,)

        dev = pick_device(device, use_cuda)

        if initial_pop is None:
            P = torch.randn(pop_size * num_populations, *dim, dtype=dtype)
        else:
            P = initial_pop
        P = P.to(device=dev) if dtype is None else P.to(device=dev, dtype=dtype)

        self.pop_size, *self.dim = P.shape
        self.num_populations = num_populations
        assert self.pop_size % self.num_populations == 0, (
            "population size must be divisible by num_populations"
        )
        block_size = self.pop_size // self.num_populations

        if proj_for_individuals is None:
            proj_for_individuals = f_for_individuals
        if f_for_individuals:
            f = individual2population(f)
        if proj_for_individuals:
            proj_to_domain = individual2population(proj_to_domain)

        P = proj_to_domain(P)

        self.use_randint = prob_choosing_method in ("randint", "random", "rand_int")
        if prob_choosing_method in ("automatic", "auto", None):
            # Two reasons to take the randint path. Above ~8 per island it is
            # simply the better trade: the multinomial path costs O(pop^2) per
            # generation (it samples from a full pop x pop matrix to get its
            # no-self-pick guarantee) against O(pop) for randint, so for a
            # population of a few thousand it is ~1000x slower per step for a
            # guarantee DE barely needs. Below 4 it is the only path that works
            # at all -- see the assert below.
            self.use_randint = block_size >= 8 or block_size < MIN_BLOCK_FOR_MULTINOMIAL

        if self.use_randint:
            n, s, b = self.pop_size, self.num_populations, block_size
            if s == 1:
                self._rand_indices = lambda: torch.randint(0, n, (3, n), device=P.device)
            else:
                # Row i's island is i // b, and that island's rows start at
                # (i // b) * b -- hence the "* b". (Without it, island j drew its
                # partners from rows [j, j+b), so every island but #0 overlapped
                # its neighbours and none of them were actually isolated.)
                offsets = (
                    torch.arange(s, device=P.device).repeat_interleave(b)[None].contiguous() * b
                )
                self._rand_indices = lambda: offsets + torch.randint(0, b, (3, n), device=P.device)
        else:
            # Each row of idx_prob has block_size - 1 nonzero entries (everyone
            # in the island but me) and we draw 3 of them without replacement.
            # With fewer than 4 per island torch.multinomial either raises or --
            # at exactly 3 -- silently returns the zero-probability entry, i.e.
            # the individual itself, quietly breaking the one guarantee this
            # path exists to provide.
            assert block_size >= MIN_BLOCK_FOR_MULTINOMIAL, (
                f"prob_choosing_method='multinomial' needs at least "
                f"{MIN_BLOCK_FOR_MULTINOMIAL} individuals per island to draw 3 "
                f"distinct partners, got {block_size}. Use "
                f"prob_choosing_method='randint' or a bigger population."
            )
            self.idx_prob = get_block_eye(block_size, self.num_populations).to(P)

        self.f = f if not maximize else (lambda x: -f(x))
        self.cost = self._eval(P)
        self.P = P
        self.proj_to_domain = proj_to_domain
        self.maximize = maximize

        # broadcast shapes: _dims_1 selects whole individuals; _crossp_dims is
        # the shape of the crossover mask under chromosome_replacement_dimension.
        self._dims_1 = tuple([self.pop_size] + [1 for _ in self.dim])
        crp = chromosome_replacement_dimension
        if crp is None:
            crp = len(self.dim)
        self._crossp_dims = tuple(
            [self.pop_size] + [d for d in self.dim[:crp]] + [1 for _ in self.dim[crp:]]
        )

    def _eval(self, P: Tensor) -> Tensor:
        """Evaluate the objective over a whole population, as a flat ``(pop,)``.

        ``f`` is allowed to return ``(pop,)`` or ``(pop, 1)``. Anything else is
        almost always the same mistake -- reducing over too few axes, so that a
        population of matrices comes back with one cost per *row* instead of per
        individual -- and it is worth naming, because the alternative is a
        baffling reshape error several lines later.
        """
        cost = self.f(P).reshape(-1)
        assert cost.numel() == self.pop_size, (
            f"objective returned {cost.numel()} costs for a population of "
            f"{self.pop_size}; it must return one cost per individual, i.e. shape "
            f"({self.pop_size},). Reduce over every axis but the first, e.g. "
            f"f(P).sum(dim=tuple(range(1, P.dim())))."
        )
        return cost

    def _cross_pollination(self, crossp: float) -> Tensor:
        return (torch.rand(self._crossp_dims, device=self.P.device) < crossp).to(self.P)

    def _get_ABC(self) -> Tensor:
        I = self._rand_indices() if self.use_randint else torch.multinomial(self.idx_prob, 3).T
        return self.P[I]

    def shuffle(self) -> None:
        """Reshuffle individuals across islands (mixes the sub-populations)."""
        I = torch.randperm(self.P.shape[0], device=self.P.device)
        self.P = self.P[I]
        self.cost = self.cost[I]

    def step(self, mut: float = 0.8, crossp: float = 0.7) -> None:
        """One DE generation: mutate, crossover, project, greedily select."""
        A, B, C = self._get_ABC()
        mutants = A + mut * (B - C)

        T = self._cross_pollination(crossp)
        candidates = self.proj_to_domain(T * mutants + (1 - T) * self.P)
        f_candidates = self._eval(candidates)

        should_replace = f_candidates <= self.cost
        self.cost = torch.where(should_replace, f_candidates, self.cost)

        S = should_replace.to(self.P).view(*self._dims_1)  # broadcast over the chromosome
        self.P = S * candidates + (1 - S) * self.P

    def best(self) -> tuple[float, Tensor]:
        """Return ``(best_cost, best_individual)`` (cost un-negated if maximising)."""
        best_cost, best_index = torch.min(self.cost, dim=0)
        if self.maximize:
            best_cost = -best_cost
        return best_cost.item(), self.P[best_index]


def _total_of(pbar: Any, epochs: Any) -> Optional[int]:
    """Best current guess at the number of generations, or ``None`` if unknown.

    Re-read every generation rather than cached, because ``epochs`` may be a
    ``timer.Timer`` whose length is an estimate that sharpens as it runs. Asking
    ``epochs`` directly beats reading ``pbar.total``: fastprogress throttles its
    redraws, so the bar's copy of the total can be many generations stale.
    """
    if hasattr(epochs, "__len__"):
        return len(epochs)
    return getattr(pbar, "total", None)


def optimize(
    f: Callable[[Tensor], Tensor],
    initial_pop: Optional[Tensor] = None,
    pop_size: int = 20,
    dim: Union[int, Sequence[int]] = (1,),
    num_populations: int = 1,
    shuffles: int = 0,
    mut: Schedule = 0.8,
    crossp: Schedule = 0.7,
    epochs: Union[int, range] = 1000,
    proj_to_domain: Callable[[Tensor], Tensor] = lambda x: x,
    f_for_individuals: bool = False,
    proj_for_individuals: Optional[bool] = None,
    maximize: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    use_cuda: Optional[bool] = None,
    dtype: Optional[torch.dtype] = torch.double,
    prob_choosing_method: str = "automatic",
    chromosome_replacement_dimension: Optional[int] = 1,
    break_at_cost: Optional[float] = None,
    on_interrupt: Optional[Callable[[], None]] = None,
    mb: Optional[Any] = None,
) -> tuple[float, Tensor]:
    """Run differential evolution and return ``(best_cost, best_individual)``.

    ``mut``/``crossp`` may be constants or ``(lo, hi)`` ranges (resampled each
    generation). ``epochs`` may be an int, a range, or any sized iterable --
    including a ``timer.Timer``, to run for a wall-clock budget instead of a
    generation count. ``shuffles`` evenly spaces sub-population reshuffles
    across the run.

    ``break_at_cost`` stops early once the best cost *reaches or passes* it (use
    ``0`` to stop as soon as a perfect solution turns up); when maximising, it
    stops once the best cost is at least ``break_at_cost``.

    ``Ctrl-C`` returns the best found so far instead of propagating, and calls
    ``on_interrupt`` if given -- useful when a caller is driving many of these
    in a loop and needs to know to stop rather than move on to the next one.

    Pass ``mb`` to nest the progress bar under a ``progress.master_bar``.
    """
    if num_populations == 1:
        shuffles = 0  # nothing to mix

    D = DifferentialEvolver(
        f=f,
        initial_pop=initial_pop,
        pop_size=pop_size,
        dim=dim,
        num_populations=num_populations,
        proj_to_domain=proj_to_domain,
        f_for_individuals=f_for_individuals,
        proj_for_individuals=proj_for_individuals,
        maximize=maximize,
        device=device,
        use_cuda=use_cuda,
        dtype=dtype,
        prob_choosing_method=prob_choosing_method,
        chromosome_replacement_dimension=chromosome_replacement_dimension,
    )

    if isinstance(epochs, int):
        epochs = range(epochs)
    mut_fn, crossp_fn = tofunc(mut), tofunc(crossp)

    pbar = progress_bar(epochs, parent=mb)

    try:
        i = 0
        shuffles_so_far = 0
        for _ in pbar:
            D.step(mut=mut_fn(), crossp=crossp_fn())
            i += 1

            total = _total_of(pbar, epochs)
            if shuffles and total and i / total > (shuffles_so_far + 1) / (shuffles + 1):
                shuffles_so_far += 1
                D.shuffle()

            best_cost, _ = D.best()
            if hasattr(pbar, "comment"):
                pbar.comment = f"| best cost = {best_cost:.4f}"
            if break_at_cost is not None:
                # Costs rarely land exactly on the target, so this is a >=/<=
                # test, not the == it used to be.
                reached = best_cost >= break_at_cost if maximize else best_cost <= break_at_cost
                if reached:
                    break
    except KeyboardInterrupt:
        if on_interrupt is not None:
            on_interrupt()
        log(mb, "Interrupting! Returning best found so far.")

    return D.best()
