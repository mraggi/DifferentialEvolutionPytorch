# DifferentialEvolutionPytorch

Differential evolution (DE/rand/1/bin) in PyTorch, vectorised over the whole population so it runs a few thousand candidates per generation on the GPU as easily as on the CPU.

The population lives in a single tensor of shape `(pop, *dim)`, and your objective is called **once per generation with the entire population**, returning one cost per individual. That is the whole trick: if you can write your cost as batched tensor ops, DE costs you about as much as one forward pass per generation.

## Install

```
pip install torch
pip install fastprogress   # optional, for progress bars
```

`fastprogress` is genuinely optional — without it the bars turn into silent no-ops and everything else behaves identically.

## Usage

```python
import torch
from differential_evolution import optimize

def sphere(P):                 # P is (pop, 10); return one cost per individual
    return (P**2).sum(dim=1)

cost, x = optimize(sphere, pop_size=100, dim=(10,), epochs=1000)
print(cost, x.shape)           # ~1e-20, torch.Size([10])
```

`optimize` returns `(best_cost, best_individual)` and picks CUDA automatically when it is available.

### Writing the objective

`f` receives the whole population and must return a 1-D tensor of `pop` costs. Individuals may have any shape — `dim=(20, 20)` gives you a population of matrices — so reduce over every axis but the first:

```python
def matrix_cost(M):            # M is (pop, 20, 20)
    return M.mean(dim=(1, 2)).abs()
```

If batching is awkward, `f_for_individuals=True` lets you write `f` one individual at a time. It is much slower (a Python loop per generation), so reach for it only when the vectorised form is genuinely hard.

### Useful options

| Option | What it does |
| --- | --- |
| `epochs` | An int, a `range`, or any iterable — including `timer.Timer(30)` to run for 30 seconds instead of a fixed generation count. |
| `break_at_cost` | Stop as soon as the best cost reaches this (`0` for "stop at a perfect solution"). |
| `proj_to_domain` | Projection back into the feasible set, applied to every candidate — e.g. `lambda P: P.clamp(-1, 1)`. |
| `mut`, `crossp` | Constants (`0.8`), or `(lo, hi)` ranges resampled every generation (`mut=(0, 1)` works well). |
| `maximize` | Maximise instead of minimise. |
| `device` | `'cuda'`, `'cpu'`, or `None` to auto-detect. |
| `dtype` | Defaults to `torch.double`; pass `torch.float` for speed, or `None` to keep `initial_pop`'s dtype. |
| `initial_pop` | Start from a population you built yourself instead of `randn`. |
| `num_populations` | Evolve independent islands (see below). |
| `chromosome_replacement_dimension` | Granularity of the crossover — for a population of points shaped `(n, d)`, `1` swaps whole points rather than individual coordinates. |

### Sub-populations (islands)

`num_populations=k` splits the population into `k` islands that evolve independently: DE draws each candidate's partners only from its own island, so the islands explore separately instead of all collapsing into the same basin. `shuffles=n` redistributes individuals across islands `n` times over the run, letting good solutions spread.

```python
cost, x = optimize(sphere, pop_size=64, dim=(10,), num_populations=8, shuffles=3, epochs=2000)
```

`pop_size` is the size of **each** island, so the above evolves 512 individuals in 8 islands of 64.

### Progress bars

`optimize` draws a bar per run. To nest many runs under one master bar:

```python
from progress import master_bar, log

mb = master_bar(problems)
for problem in mb:
    cost, x = optimize(problem.f, dim=problem.dim, epochs=500, mb=mb)
    log(mb, f"solved with cost {cost}")     # prints above the bar instead of garbling it
```

Use `progress.NoBar` in place of `master_bar` to silence bars entirely, even when `fastprogress` is installed.

## Tests

```
python test_differential_evolution.py
```

No pytest needed; it exits nonzero on failure. Worth running once with `fastprogress` installed and once without, since the two take different paths through `progress.py`.

## Files

| File | |
| --- | --- |
| `differential_evolution.py` | The optimiser: `optimize()` and the `DifferentialEvolver` class underneath it. |
| `progress.py` | Progress bars — fastprogress when installed, silent no-ops otherwise. |
| `timer.py` | `Timer(seconds)`, an iterable you can pass as `epochs` to run for a wall-clock budget. |
| `test_differential_evolution.py` | Behavioural tests. |
| `Examples.ipynb` | Assorted example calls. |
