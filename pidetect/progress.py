"""progress.py : tells you where a long run is, without drowning the log

* track(iterable, ...)  wraps a loop and prints one status line (with %, elapsed and an ETA) only when the loop is
                        slow: the first line after 15 s, then one every 30 s, and a closing line if it printed any.
                        Fast loops stay silent, so the log of a run that takes hours stays readable.
* SeedProgress          one line when a seed starts and one when it ends, with the elapsed time and an ETA for the
                        remaining seeds.
Plain lines (no carriage returns, no extra package), so they also work when the output is piped to a file.
"""
from __future__ import annotations

import time
from typing import Iterable, Iterator, Optional, Sequence, TypeVar

T = TypeVar("T")
_STAGE = ""


def stamp() -> str:
    return time.strftime("[%H:%M:%S] ")


def fmt_dur(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def set_stage(label: str) -> None:
    """The label shown in front of every status line, e.g. 'h1 seed 2/5'"""
    global _STAGE
    _STAGE = label


def track(iterable: Iterable[T], total: Optional[int] = None, desc: str = "", first_after: float = 15.0,
          every: float = 30.0) -> Iterator[T]:
    if total is None:
        try:
            total = len(iterable)                                     # type: ignore[arg-type]
        except TypeError:
            total = None
    t0 = last = time.time()
    printed = False
    n = 0
    for item in iterable:
        yield item
        n += 1
        now = time.time()
        if now - last >= (every if printed else first_after):
            last, printed = now, True
            el = now - t0
            if total:
                eta = el / n * (total - n)
                msg = f"{n}/{total} ({100 * n // total}%) | elapsed {fmt_dur(el)}, ETA {fmt_dur(eta)}"
            else:
                msg = f"{n} done | elapsed {fmt_dur(el)}"
            print(f"{stamp()}   {(_STAGE + ' | ') if _STAGE else ''}{desc}: {msg}", flush=True)
    if printed and total:
        print(f"{stamp()}   {(_STAGE + ' | ') if _STAGE else ''}{desc}: finished ({total} steps) in "
              f"{fmt_dur(time.time() - t0)}", flush=True)


class SeedProgress:
    def __init__(self, name: str, seeds: Sequence[int]):
        self.name, self.seeds, self.t0, self.i = name, list(seeds), time.time(), 0

    def start(self, seed: int) -> None:
        self.i += 1
        self.t_seed = time.time()
        set_stage(f"{self.name} seed {seed} ({self.i}/{len(self.seeds)})")
        print(f"{stamp()}=== {self.name}: seed {seed} ({self.i}/{len(self.seeds)}) started", flush=True)

    def done(self, seed: int) -> None:
        el = time.time() - self.t0
        left = len(self.seeds) - self.i
        eta = (el / self.i) * left
        print(f"{stamp()}=== {self.name}: seed {seed} ({self.i}/{len(self.seeds)}) finished in "
              f"{fmt_dur(time.time() - self.t_seed)} | elapsed {fmt_dur(el)}"
              + (f", about {fmt_dur(eta)} left for the remaining {left} seed(s)" if left else ", all seeds done"),
              flush=True)
        set_stage("")
