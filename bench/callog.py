"""Append-only record of every model call the bench makes.

One line of JSON per call, flock-guarded so concurrent processes interleave
cleanly. This is the provenance trail: which model, against which commit, was
asked what, and what it cost. When a physics module later turns out to be
wrong, this is how you find out what produced it.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import ROOT, state_dir


@lru_cache(maxsize=1)
def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class CallRecord:
    role: str
    model: str
    provider: str
    prompt_hash: str
    cached: bool
    ok: bool
    latency_s: float
    waited_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    ts: float = field(default_factory=time.time)
    git: str = field(default_factory=git_sha)


class CallLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or state_dir() / "calls.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: CallRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode())
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]

    def summary(self) -> dict:
        records = self.read()
        if not records:
            return {"calls": 0}
        by_model: dict[str, dict] = {}
        for r in records:
            m = by_model.setdefault(
                r["model"], {"calls": 0, "cached": 0, "failed": 0, "in": 0, "out": 0}
            )
            m["calls"] += 1
            m["cached"] += int(r.get("cached", False))
            m["failed"] += int(not r.get("ok", True))
            m["in"] += r.get("prompt_tokens") or 0
            m["out"] += r.get("completion_tokens") or 0
        return {"calls": len(records), "by_model": by_model}
