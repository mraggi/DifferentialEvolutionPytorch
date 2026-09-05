"""Progress-bar plumbing: fastprogress if installed, else a silent no-op shim.

``differential_evolution`` wants three things from a progress bar that plain
``fastprogress`` doesn't quite give on its own:

* **Optional.** The optimiser is useful without ``fastprogress`` installed, so
  every bar here degrades to a shim that iterates and draws nothing.
* **Variable size.** ``epochs`` may be a ``timer.Timer``, whose ``__len__`` is a
  running *estimate* that changes as the loop runs. A real fastprogress bar
  samples ``len(gen)`` once, so :class:`VariableSizeProgressBar` re-reads it on
  every update (this is what the old ``progress_bar.py`` did).
* **Nestable.** ``master_bar`` over many runs, with each run's generations as a
  child bar underneath.

Use :func:`log` instead of a bare ``print()`` for anything emitted while a bar
is live: with fastprogress that routes through ``bar.write()`` (which prints
cleanly above the bar instead of garbling its in-place redraw), and without it
it is just ``print()``.

Callers that want to *suppress* bars entirely (e.g. a batch driver whose own
output would fight with them) can use :class:`NoBar` in place of
:func:`master_bar` / :func:`progress_bar`.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


class NoBar:
    """Silent stand-in for a fastprogress bar: iterates, tracks ``.comment``,
    and ``.write()`` falls back to ``print()``. Always available, so callers can
    opt out of progress bars even when fastprogress *is* installed.

    ``total`` is a property rather than a stored value so a variable-size
    iterable (``timer.Timer``) reports its current estimate, matching
    :class:`VariableSizeProgressBar`.

    Implements enough of the fastprogress parent/child protocol (``add_child``,
    ``child``, ``progress``, ``main_bar``) that a real fastprogress child bar
    created with ``parent=NoBar(...)`` lands here without error and stays
    invisible.
    """

    def __init__(self, gen: Any, parent: Optional[Any] = None):
        self.gen = range(gen) if isinstance(gen, int) else gen
        self.comment = ""
        self.main_bar = self

    @property
    def total(self) -> Optional[int]:
        return len(self.gen) if hasattr(self.gen, "__len__") else None

    def __iter__(self):
        return iter(self.gen)

    def write(self, line: str, table: bool = False) -> None:
        print(line)

    def add_child(self, child: Any) -> None:
        """Accept a child bar (fastprogress protocol) but leave it invisible."""

    def child(self, gen: Any) -> "NoBar":
        """Create a child bar (fastprogress protocol)."""
        return NoBar(gen, parent=self)

    def progress(self, gen: Any) -> "NoBar":
        """Create a child progress bar (fastprogress protocol)."""
        return NoBar(gen, parent=self)


try:
    from fastprogress.fastprogress import master_bar as _real_master_bar
    from fastprogress.fastprogress import progress_bar as _real_progress_bar

    HAVE_FASTPROGRESS = True

    class VariableSizeProgressBar(_real_progress_bar):
        """A fastprogress bar whose total is re-read from the iterable.

        ``epochs=Timer(5)`` doesn't know how many generations it will yield
        until it has run for a while; its ``__len__`` improves as it goes.
        Refreshing ``total`` on every update keeps the ETA honest instead of
        pinning it to the first (wild) guess.
        """

        def on_update(self, val, text):
            if hasattr(self.gen, "__len__"):
                self.total = len(self.gen)
            super().on_update(val, text)

except Exception:  # pragma: no cover - only when fastprogress is missing
    HAVE_FASTPROGRESS = False
    VariableSizeProgressBar = NoBar  # type: ignore[assignment,misc]


def master_bar(gen: Iterable) -> Any:
    """Return a master progress bar over ``gen`` (fastprogress if available)."""
    if HAVE_FASTPROGRESS:
        return _real_master_bar(gen)
    return NoBar(gen)


def progress_bar(gen: Iterable, parent: Optional[Any] = None) -> Any:
    """Return a progress bar over ``gen``, optionally nested under ``parent``.

    ``gen`` needn't be sized: fastprogress infers the total from ``len(gen)``
    and raises a ``TypeError`` on a plain generator, so unsized iterables are
    passed through with ``total='noinfer'`` -- no ETA, but they run.
    """
    if HAVE_FASTPROGRESS:
        total = None if hasattr(gen, "__len__") else "noinfer"
        return VariableSizeProgressBar(gen, total=total, parent=parent)
    return NoBar(gen, parent)


def log(bar: Optional[Any], msg: str) -> None:
    """Print ``msg`` without garbling ``bar`` if it is a live fastprogress bar.

    ``bar`` may be ``None`` (plain ``print``), one of this module's bars, or a
    real fastprogress bar -- anything with ``.write()`` is routed through it.
    """
    if bar is not None and hasattr(bar, "write"):
        bar.write(msg)
    else:
        print(msg)
