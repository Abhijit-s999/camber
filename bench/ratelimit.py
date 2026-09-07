"""A token bucket that is shared across processes, not just across threads.

The MCP server, the overnight daemon and an interactive CLI call can all be
running at once, all against the same NVIDIA account. An in-process limiter
would let each of them independently believe it owns the full 40 rpm, and the
account would return 429 to whichever arrived third. So the bucket lives in a
file and is guarded by flock: whoever holds the lock owns the arithmetic.

The lock is never held across a sleep. A waiter computes how long it must wait,
releases, sleeps, and re-competes for the tokens -- otherwise one blocked caller
would serialise every other process behind it.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .config import state_dir


class RateLimitTimeout(RuntimeError):
    """Waited longer than the caller was willing to wait for a token."""


@dataclass
class TokenBucket:
    key: str
    rpm: int | None
    #: Burst allowance. Defaults to a full minute's worth.
    capacity: float | None = None

    def __post_init__(self) -> None:
        if self.capacity is None and self.rpm is not None:
            self.capacity = float(self.rpm)

    @property
    def _dir(self) -> Path:
        return state_dir() / "buckets"

    @property
    def _path(self) -> Path:
        return self._dir / f"{self.key}.json"

    @property
    def unmetered(self) -> bool:
        return self.rpm is None

    def _read(self, fd: int) -> tuple[float, float]:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 4096).decode() or "{}"
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            state = {}
        return (
            float(state.get("tokens", self.capacity or 0.0)),
            float(state.get("updated", time.monotonic())),
        )

    def _write(self, fd: int, tokens: float, updated: float) -> None:
        payload = json.dumps({"tokens": tokens, "updated": updated}).encode()
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, payload)
        os.fsync(fd)

    def _try_consume(self, cost: float) -> float:
        """Take `cost` tokens, or report how many seconds short we are."""
        self._dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            now = time.monotonic()
            tokens, updated = self._read(fd)

            # A reboot resets monotonic time; treat a future timestamp as a
            # fresh bucket rather than crediting an enormous refill.
            elapsed = now - updated
            if elapsed < 0:
                tokens, elapsed = float(self.capacity or 0.0), 0.0

            rate = self.rpm / 60.0  # type: ignore[operator]
            tokens = min(float(self.capacity or 0.0), tokens + elapsed * rate)

            if tokens >= cost:
                self._write(fd, tokens - cost, now)
                return 0.0
            self._write(fd, tokens, now)
            return (cost - tokens) / rate
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def acquire(self, cost: float = 1.0, timeout: float | None = None) -> float:
        """Block until `cost` tokens are available. Returns seconds waited."""
        if self.unmetered:
            return 0.0
        waited = 0.0
        while True:
            deficit = self._try_consume(cost)
            if deficit <= 0.0:
                return waited
            if timeout is not None and waited + deficit > timeout:
                raise RateLimitTimeout(
                    f"bucket {self.key!r}: would wait {waited + deficit:.1f}s, "
                    f"limit was {timeout:.1f}s"
                )
            # Small pad so we do not wake a hair early and spin.
            time.sleep(deficit + 0.01)
            waited += deficit

    def peek(self) -> float:
        """Tokens currently available, without consuming any.

        Refills under the lock so the answer reflects elapsed time rather than
        whatever was last written.
        """
        if self.unmetered:
            return float("inf")
        self._dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            now = time.monotonic()
            tokens, updated = self._read(fd)
            elapsed = max(0.0, now - updated)
            rate = self.rpm / 60.0  # type: ignore[operator]
            tokens = min(float(self.capacity or 0.0), tokens + elapsed * rate)
            self._write(fd, tokens, now)
            return tokens
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


_buckets: dict[str, TokenBucket] = {}


def bucket_for(bucket_key: str, rpm: int | None) -> TokenBucket:
    if bucket_key not in _buckets:
        _buckets[bucket_key] = TokenBucket(key=bucket_key, rpm=rpm)
    return _buckets[bucket_key]


def _reset_buckets() -> None:
    """Test hook. The bucket objects are memoised; their state files are not."""
    _buckets.clear()
