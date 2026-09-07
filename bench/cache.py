"""Response cache keyed by the exact request that produced it.

Two reasons this exists, and the second matters more than the first:

* Quota. Re-running a campaign should not re-ask questions already answered.
* Reproducibility. A build that consults a language model is only defensible if
  you can show what was asked and what came back. The cache plus the call log
  are the provenance trail -- the same role DECISIONS.md plays for the design.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .config import state_dir

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key      TEXT PRIMARY KEY,
    model    TEXT NOT NULL,
    created  REAL NOT NULL,
    request  TEXT NOT NULL,
    payload  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS responses_model ON responses(model);
"""


def request_key(model_id: str, messages: list[dict], **params: Any) -> str:
    """Stable hash of everything that could change the answer."""
    canonical = json.dumps(
        {"model": model_id, "messages": messages, "params": params},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class Cache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or state_dir() / "cache.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30.0)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> dict | None:
        row = self._conn.execute(
            "SELECT payload FROM responses WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model_id: str, request: dict, payload: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, model, created, request, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                key,
                model_id,
                time.time(),
                json.dumps(request, ensure_ascii=False),
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT model, COUNT(*) FROM responses GROUP BY model"
        ).fetchall()
        return {model: count for model, count in rows}

    def close(self) -> None:
        self._conn.close()
