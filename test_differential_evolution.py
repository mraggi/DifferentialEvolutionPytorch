"""Behavioural tests for the optimiser. Run with ``python test_differential_evolution.py``.

Plain asserts and a ``check()`` counter -- no pytest, no dependency beyond what
the library itself needs. Exits nonzero if anything fails, so it drops into CI
as-is. Runs twice over if you care: once with ``fastprogress`` installed and
once without, since the two take different paths through ``progress.py``.

The centrepiece is the island-isolation group. ``num_populations`` promises that
DE partners are drawn only from within a sub-population, and for years that
promise was quietly broken on the ``randint`` path (a missing ``* b`` in the
island base offset made island j draw from rows ``[j, j+b)`` instead of
``[j*b, (j+1)*b)``, so every island but #0 overlapped its neighbours). It cost
nothing visible -- the optimiser still converged -- which is exactly why it
needs a test.
"""

import sys

import torch

from differential_evolution import (
    MIN_BLOCK_FOR_MULTINOMIAL,
    DifferentialEvolver,
    _total_of,
    get_block_eye,
    optimize,
    pick_device,
    tofunc,
)
from progress import HAVE_FASTPROGRESS, NoBar, log, master_bar, progress_bar
from timer import Timer

torch.manual_seed(0)

fails = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        fails.append(name)


def sphere(P):
    """Classic DE test objective: minimised (0) at the origin."""
    return (P**2).sum(dim=tuple(range(1, P.dim())))


print(f"--- torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
      f"fastprogress={HAVE_FASTPROGRESS} ---")

# ---- 1. sphere function converges -------------------------------------------
c, x = optimize(sphere, pop_size=60, dim=(5,), epochs=300, device="cpu")
check("converges on sphere", c < 1e-6, f"(cost={c:.3e})")
check("dtype defaults to double", x.dtype == torch.float64, f"({x.dtype})")

# ---- 2. dtype override -------------------------------------------------------
c, x = optimize(sphere, pop_size=40, dim=(3,), epochs=50, device="cpu", dtype=torch.float32)
check("dtype=float32 honoured", x.dtype == torch.float32, f"({x.dtype})")
ip = torch.rand(40, 3, dtype=torch.float16)
c, x = optimize(sphere, initial_pop=ip, epochs=10, device="cpu", dtype=None)
check("dtype=None keeps initial_pop dtype", x.dtype == torch.float16, f"({x.dtype})")

# ---- 3. ISLAND ISOLATION (the bug) ------------------------------------------
# Island k is seeded far from the origin at distance 10*(k+1). DE partners are
# A + mut*(B-C) drawn from within an island, so with isolated islands each stays
# in its own neighbourhood. If the islands leak into each other, the population
# collapses toward whichever island is winning.
n_isl, blk, d = 4, 16, 2
pop = torch.zeros(n_isl*blk, d, dtype=torch.double)
for k in range(n_isl):
    pop[k*blk:(k+1)*blk] = 100.0*(k+1) + 0.01*torch.randn(blk, d, dtype=torch.double)
D = DifferentialEvolver(lambda P: (P**2).sum(dim=1), initial_pop=pop.clone(),
                        num_populations=n_isl, device="cpu", prob_choosing_method="randint")
for _ in range(30):
    D.step(mut=0.5, crossp=0.9)
means = [D.P[k*blk:(k+1)*blk].mean().item() for k in range(n_isl)]
# each island must still sit near its own seed (partners never came from elsewhere)
ok = all(abs(means[k] - 100.0*(k+1)) < 50.0 for k in range(n_isl))
check("islands stay isolated (randint path)", ok, f"means={[round(m,1) for m in means]} want~[100,200,300,400]")

# same test through the multinomial path, which was always correct
D2 = DifferentialEvolver(lambda P: (P**2).sum(dim=1), initial_pop=pop.clone(),
                         num_populations=n_isl, device="cpu", prob_choosing_method="multinomial")
for _ in range(30):
    D2.step(mut=0.5, crossp=0.9)
means2 = [D2.P[k*blk:(k+1)*blk].mean().item() for k in range(n_isl)]
check("islands stay isolated (multinomial path)",
      all(abs(means2[k] - 100.0*(k+1)) < 50.0 for k in range(n_isl)),
      f"means={[round(m,1) for m in means2]}")

# index-level check: every sampled partner index is inside the caller's island
D3 = DifferentialEvolver(sphere, pop_size=8, dim=(2,), num_populations=5, device="cpu",
                         prob_choosing_method="randint")
b = D3.pop_size // D3.num_populations
bad = 0
for _ in range(200):
    I = D3._rand_indices()               # (3, pop)
    for row in range(D3.pop_size):
        if any((I[j, row].item() // b) != (row // b) for j in range(3)): bad += 1
check("sampled partners never cross islands", bad == 0, f"({bad} crossings in 200*8*3 draws)")

# ---- 4. automatic threshold picks randint for reasonable islands ------------
D4 = DifferentialEvolver(sphere, pop_size=64, dim=(2,), num_populations=4, device="cpu")
check("automatic -> randint for block_size 64", D4.use_randint is True)
D5 = DifferentialEvolver(sphere, pop_size=4, dim=(2,), num_populations=2, device="cpu")
check("automatic -> multinomial for block_size 4", D5.use_randint is False)

# ---- 5. break_at_cost --------------------------------------------------------
c, _ = optimize(sphere, pop_size=40, dim=(3,), epochs=100000, device="cpu", break_at_cost=1e-3)
check("break_at_cost stops early (<= not ==)", c <= 1e-3, f"(cost={c:.3e})")

# ---- 6. maximize -------------------------------------------------------------
def neg_sphere(P): return -(P**2).sum(dim=tuple(range(1, P.dim())))
c, x = optimize(neg_sphere, pop_size=60, dim=(4,), epochs=300, device="cpu", maximize=True)
check("maximize returns un-negated cost", c < 0 and c > -1e-6, f"(cost={c:.3e})")
check("maximize agrees with f", abs(c - neg_sphere(x[None])[0].item()) < 1e-12)
c, _ = optimize(neg_sphere, pop_size=40, dim=(3,), epochs=100000, device="cpu",
                maximize=True, break_at_cost=-1e-3)
check("break_at_cost flips direction when maximising", c >= -1e-3, f"(cost={c:.3e})")

# ---- 7. epochs forms ---------------------------------------------------------
for name, ep in [("int", 20), ("range", range(20)), ("list", list(range(20))), ("Timer", Timer(1))]:
    try:
        optimize(sphere, pop_size=20, dim=(3,), epochs=ep, device="cpu"); ok = True
    except Exception as e:
        ok = False; print("   ", e)
    check(f"epochs as {name}", ok)

# ---- 8. shuffles with variable-size epochs (used to divide by pbar.total) ----
try:
    optimize(sphere, pop_size=20, dim=(3,), num_populations=4, shuffles=3, epochs=Timer(1), device="cpu")
    ok = True
except Exception as e:
    ok = False; print("   ", e)
check("shuffles + Timer epochs", ok)

# ---- 9. legacy API surface (Examples.ipynb) ----------------------------------
def matrix_cost(M):
    return torch.abs(torch.mean(torch.mean(M, dim=1), dim=1))
legacy = [
  dict(pop_size=100, dim=(20,20), epochs=Timer(1), use_cuda=True, prob_choosing_method='multinomial'),
  dict(pop_size=100, dim=(20,20), epochs=Timer(1), use_cuda=True, mut=(0,1), prob_choosing_method='randint'),
  dict(pop_size=100, dim=(20,20), num_populations=5, epochs=Timer(1), use_cuda=True, mut=(0,1), prob_choosing_method='multinomial'),
  dict(pop_size=100, dim=(20,20), num_populations=5, epochs=Timer(1), use_cuda=True, mut=(0,1), prob_choosing_method='randint'),
  dict(pop_size=6000, dim=(20,20), epochs=Timer(1), use_cuda=True),
  dict(pop_size=60, dim=(20,20), shuffles=7, epochs=Timer(1), use_cuda=False),
  dict(pop_size=60, dim=(5,5), epochs=100),
]
for i, kw in enumerate(legacy):
    try:
        optimize(matrix_cost, **kw); ok = True
    except Exception as e:
        ok = False; print("   ", type(e).__name__, e)
    check(f"legacy Examples.ipynb call #{i}", ok)
try:
    optimize(matrix_cost, initial_pop=torch.rand(600,50,50), epochs=Timer(1)); ok = True
except Exception as e:
    ok = False; print("   ", e)
check("legacy initial_pop call", ok)

# ---- 10. use_cuda / device resolution ---------------------------------------
check("use_cuda=True -> cuda", pick_device(None, True).type == "cuda")
check("use_cuda=False -> cpu", pick_device(None, False).type == "cpu")
check("device beats use_cuda", pick_device("cpu", True).type == "cpu")
_, x = optimize(sphere, pop_size=20, dim=(3,), epochs=5, use_cuda=False)
check("use_cuda=False lands on cpu", x.device.type == "cpu")

# ---- 11. edge cases ----------------------------------------------------------
try:  # tiny islands now route to randint instead of exploding in multinomial
    c, x = optimize(sphere, pop_size=1, dim=(2,), epochs=5, device="cpu"); ok = True
except Exception as e:
    ok = False; print("   ", type(e).__name__, e)
check("pop_size=1 does not crash (reshape vs squeeze)", ok)
for b in range(1, MIN_BLOCK_FOR_MULTINOMIAL):
    D = DifferentialEvolver(sphere, pop_size=b, dim=(2,), num_populations=1, device="cpu")
    check(f"automatic avoids impossible multinomial at block_size={b}", D.use_randint is True)
try:  # explicit multinomial on a too-small island must say why, not RuntimeError
    DifferentialEvolver(sphere, pop_size=MIN_BLOCK_FOR_MULTINOMIAL - 1, dim=(2,),
                        num_populations=1, device="cpu", prob_choosing_method="multinomial")
    ok = False
except AssertionError as e:
    ok = f"at least {MIN_BLOCK_FOR_MULTINOMIAL}" in str(e)
check("explicit multinomial on tiny island asserts clearly", ok)
try:  # f returning shape (pop, 1)
    optimize(lambda P: (P**2).sum(dim=1, keepdim=True), pop_size=20, dim=(3,), epochs=5, device="cpu"); ok = True
except Exception as e:
    ok = False; print("   ", type(e).__name__, e)
check("f returning (pop,1) still works", ok)
try:  # the classic mistake: reducing over too few axes, one cost per row
    optimize(lambda P: (P**2).sum(dim=1), pop_size=30, dim=(3, 4), epochs=5, device="cpu")
    ok = False
except AssertionError as e:
    ok = "30" in str(e) and "120" in str(e)
    print("   ", e)
check("wrong-shaped objective is named, not a reshape error", ok)
try:
    optimize(lambda p: (p**2).sum(), pop_size=20, dim=(3,), epochs=5, device="cpu", f_for_individuals=True); ok = True
except Exception as e:
    ok = False; print("   ", type(e).__name__, e)
check("f_for_individuals", ok)
try:
    optimize(sphere, pop_size=20, dim=(3,), epochs=5, device="cpu",
             proj_to_domain=lambda P: P.clamp(-1, 1)); ok = True
except Exception as e:
    ok = False; print("   ", type(e).__name__, e)
check("proj_to_domain", ok)
try:  # pop_size is PER ISLAND, so only initial_pop can be indivisible
    optimize(sphere, initial_pop=torch.rand(7, 2), num_populations=2, epochs=5, device="cpu")
    ok = False
except AssertionError:
    ok = True
check("indivisible initial_pop asserts", ok)
for crd in (None, 0, 1, 2):
    try:
        optimize(sphere, pop_size=20, dim=(3,4), epochs=5, device="cpu",
                 chromosome_replacement_dimension=crd); ok = True
    except Exception as e:
        ok = False; print("   ", e)
    check(f"chromosome_replacement_dimension={crd}", ok)

# ---- 12. cuda actually runs --------------------------------------------------
if torch.cuda.is_available():
    c, x = optimize(sphere, pop_size=256, dim=(10,), num_populations=4, shuffles=2,
                    epochs=1000, device="cuda")
    check("cuda run converges", c < 1e-8 and x.device.type == "cuda", f"(cost={c:.3e}, {x.device})")

# ---- 13. tofunc --------------------------------------------------------------
check("tofunc const", tofunc(0.8)() == 0.8)
check("tofunc int", tofunc(1)() == 1.0)
check("tofunc tuple in range", 0.0 <= tofunc((0., 1.))() <= 1.0)
check("tofunc slice in range", 2.0 <= tofunc(slice(2., 3.))() <= 3.0)
check("tofunc callable", tofunc(lambda: 0.5)() == 0.5)

# ---- 14. get_block_eye -------------------------------------------------------
M = get_block_eye(3, 2)
expect = torch.tensor([[0.,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,0,0,0],
                       [0,0,0,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
check("get_block_eye shape/values", torch.equal(M, expect))


# ---- 15. progress bars ------------------------------------------------------
# variable-size total: a Timer's length grows as it runs, and the bar must follow
t = Timer(1)
pb = progress_bar(t)
totals = []
for _ in pb:
    totals.append(pb.total)
# fastprogress throttles redraws, so the bar's total only refreshes now and
# then -- it must move, but it is not expected to be exact on the last step.
check("bar total tracks a Timer's growing estimate",
      len(set(totals)) > 1 and totals[-1] > totals[0],
      f"(first={totals[0]}, last={totals[-1]}, n_distinct={len(set(totals))})")

# ...which is why optimize asks the iterable, not the bar, for the live total.
# Timer's own __len__ is time-dependent (two calls a microsecond apart already
# disagree), so use a deterministic variable-size stand-in to compare against.

class GrowingGen:
    """Yields 200 items; its advertised length climbs 100 -> 300 as it goes."""
    def __init__(self): self.i = 0
    def __len__(self): return 100 + self.i
    def __iter__(self):
        for self.i in range(200):
            yield self.i

g = GrowingGen()
pbg = progress_bar(g)
stale_bar = mismatched = 0
for _ in pbg:
    if _total_of(pbg, g) != len(g): mismatched += 1
    if getattr(pbg, "total", None) != len(g): stale_bar += 1
check("_total_of is always the live estimate", mismatched == 0, f"({mismatched} mismatches / 200)")
check("...and the bar's own total does go stale (why we don't use it)",
      stale_bar > 0 or not HAVE_FASTPROGRESS, f"({stale_bar}/200 stale)")

# a generator with no __len__ at all must not crash optimize
def gen_epochs(n):
    for i in range(n): yield i
try:
    optimize(lambda P: (P**2).sum(1), pop_size=20, dim=(3,), num_populations=2,
             shuffles=2, epochs=gen_epochs(20), device="cpu")
    ok = True
except Exception as e:
    ok = False; print("   ", type(e).__name__, e)
check("epochs with no __len__ (total unknown)", ok)

# fixed-size iterables keep a constant, correct total
pb = progress_bar(range(7))
seen = [pb.total for _ in pb]
check("bar total constant for range", set(seen) == {7}, f"({sorted(set(seen))})")

# nesting: a child bar under a master bar, the way search loops use it
try:
    mb = master_bar(range(3))
    for _ in mb:
        c, _ = optimize(lambda P: (P**2).sum(1), pop_size=20, dim=(3,), epochs=10,
                        device="cpu", mb=mb)
        log(mb, f"  inner best {c:.3e}")
    ok = True
except Exception as e:
    ok = False; print("   ", type(e).__name__, e)
check("optimize nests under master_bar", ok)

# NoBar is always available and silences even real fastprogress children
try:
    nb = NoBar(range(4))
    check("NoBar total", nb.total == 4)
    check("NoBar iterates", list(nb) == [0, 1, 2, 3])
    check("NoBar int gen", list(NoBar(3)) == [0, 1, 2])
    check("NoBar Timer total is live", NoBar(Timer(1)).total is not None)
    nb.comment = "x"
    c, _ = optimize(lambda P: (P**2).sum(1), pop_size=20, dim=(3,), epochs=10,
                    device="cpu", mb=nb)
    ok = True
except Exception as e:
    ok = False; print("   ", type(e).__name__, e)
check("optimize under a NoBar parent", ok)
check("log falls back to print", log(None, "  (this line came from log(None, ...))") is None)

# ---- 16. interrupt hook -----------------------------------------------------
called = []
class Boom:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __iter__(self):
        for i in range(self.n):
            if i == 3: raise KeyboardInterrupt
            yield i
c, x = optimize(lambda P: (P**2).sum(1), pop_size=20, dim=(3,), epochs=Boom(50),
                device="cpu", on_interrupt=lambda: called.append(1))
check("Ctrl-C calls on_interrupt", called == [1])
check("Ctrl-C still returns a best", torch.is_tensor(x) and c >= 0, f"(cost={c:.3e})")


print()
print("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
