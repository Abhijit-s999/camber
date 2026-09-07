"""A cross-process semaphore, because requests-per-minute was the wrong model.

Measured against NVIDIA NIM on 2026-09-07: two simultaneous requests to the
same model returned 200 and an instant 429 -- with the account nowhere near
40 requests per minute. The binding constraint is how many requests are *in
flight*, not how many were started in the last minute. A token bucket cannot
express that: it counts starts and forgets them, while a slow request occupies
capacity for its entire duration, and these requests run for minutes.

Slots are files. Holding a slot means holding a non-blocking exclusive flock on
one of them for the whole request, so a crashed process releases its slot when
the kernel closes its file descriptors -- which a counter in a file would not.
"""

from __future__ import annotations

import fcntl
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .config import state_dir


class ConcurrencyTimeout(RuntimeError):
    """No slot came free within the caller's patience."""


@dataclass
class ConcurrencySlots:
    key: str
    limit: int

    @property
    def _dir(self) -> Path:
        return state_dir() / "slots"

    def _slot_path(self, index: int) -> Path:
        return self._dir / f"{self.key}.{index}.lock"

    def _try_take(self) -> int | None:
        """Grab any free slot, or None. Returns the open fd on success."""
        self._dir.mkdir(parents=True, exist_ok=True)
        # Randomised order so several waiters do not all fight over slot 0.
        order = list(range(self.limit))
        random.shuffle(order)
        for i in order:
            fd = os.open(self._slot_path(i), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            return fd
        return None

    @contextmanager
    def hold(self, timeout: float | None = None, poll: float = 0.25):
        if self.limit <= 0:  # unlimited
            yield
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        fd = None
        while True:
            fd = self._try_take()
            if fd is not None:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise ConcurrencyTimeout(
                    f"no free slot for {self.key!r} "
                    f"(limit {self.limit}) within {timeout:.1f}s"
                )
            time.sleep(poll * (0.75 + random.random() / 2))
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def in_use(self) -> int:
        """How many slots are currently held. Racy by nature; for display."""
        if self.limit <= 0:
            return 0
        used = 0
        for i in range(self.limit):
            path = self._slot_path(i)
            if not path.exists():
                continue
            fd = os.open(path, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                used += 1
            finally:
                os.close(fd)
        return used


_slots: dict[str, ConcurrencySlots] = {}


def slots_for(key: str, limit: int) -> ConcurrencySlots:
    if key not in _slots:
        _slots[key] = ConcurrencySlots(key=key, limit=limit)
    return _slots[key]


def _reset_slots() -> None:
    _slots.clear()
