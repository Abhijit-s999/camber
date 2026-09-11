"""A project is a folder: the working vehicle, its history, its results.

    ~/Camber/<name>/
        vehicle.json      the design being edited
        project.json      name, created, template
        history/          one snapshot per edit, for undo
        results/          frames, aero tables, videos, optimiser fronts
        jobs/             one record and one log per background job

Plain files on purpose. A user can open the folder, read the vehicle as JSON,
copy a project by copying a directory, and nothing is lost if the app is closed
mid-edit: every change is written through.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from ..spec import io

HOME = Path(os.environ.get("CAMBER_HOME", Path.home() / "Camber"))
EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"
TEMPLATES = {"racer5": "racer5.toml", "quad5": "quad5.toml"}
HISTORY = 200


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    if not s:
        raise ValueError("a project needs a name")
    return s[:64]


class Project:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not (self.root / "vehicle.json").exists():
            raise FileNotFoundError(f"{self.root} is not a Camber project")
        self.meta = json.loads((self.root / "project.json").read_text())
        self.vehicle = io.load(self.root / "vehicle.json")
        self._cursor = self._latest()

    # --- lifecycle -------------------------------------------------------
    @classmethod
    def create(cls, name: str, template: str = "racer5",
               home: Path | None = None) -> "Project":
        home = Path(home or HOME)
        root = home / slug(name)
        if root.exists():
            raise FileExistsError(f"a project called {root.name!r} already exists")
        if template not in TEMPLATES:
            raise ValueError(f"unknown template {template!r}; "
                             f"choose from {', '.join(TEMPLATES)}")
        for d in ("history", "results", "jobs"):
            (root / d).mkdir(parents=True, exist_ok=True)
        v = io.load(EXAMPLES / TEMPLATES[template])
        v.name = slug(name)
        io.save_json(v, root / "vehicle.json")
        (root / "project.json").write_text(json.dumps(
            {"name": name, "template": template, "created": time.time()}, indent=1))
        p = cls(root)
        p._snapshot("created from " + template)
        return p

    @staticmethod
    def list(home: Path | None = None) -> list[dict]:
        home = Path(home or HOME)
        out = []
        for d in sorted(home.glob("*/project.json")):
            meta = json.loads(d.read_text())
            out.append({"id": d.parent.name, "name": meta.get("name", d.parent.name),
                        "modified": (d.parent / "vehicle.json").stat().st_mtime})
        return sorted(out, key=lambda r: -r["modified"])

    @property
    def id(self) -> str:
        return self.root.name

    # --- editing, with undo ---------------------------------------------
    def commit(self, label: str) -> None:
        """Write the working vehicle through, and snapshot it for undo."""
        io.save_json(self.vehicle, self.root / "vehicle.json")
        self._snapshot(label)

    def _snaps(self) -> list[Path]:
        return sorted((self.root / "history").glob("*.json"))

    def _latest(self) -> int:
        return len(self._snaps()) - 1

    def _snapshot(self, label: str) -> None:
        snaps = self._snaps()
        # Undo then edit discards the redo branch, as every editor does.
        for s in snaps[self._cursor + 1:] if hasattr(self, "_cursor") else []:
            s.unlink()
        snaps = self._snaps()
        n = (int(snaps[-1].stem.split("-")[0]) + 1) if snaps else 0
        path = self.root / "history" / f"{n:06d}-{slug(label)[:40]}.json"
        io.save_json(self.vehicle, path)
        for old in self._snaps()[:-HISTORY]:
            old.unlink()
        self._cursor = self._latest()

    def undo(self) -> str | None:
        if self._cursor <= 0:
            return None
        self._cursor -= 1
        return self._restore()

    def redo(self) -> str | None:
        if self._cursor >= self._latest():
            return None
        self._cursor += 1
        return self._restore()

    def _restore(self) -> str:
        snap = self._snaps()[self._cursor]
        self.vehicle = io.load(snap)
        io.save_json(self.vehicle, self.root / "vehicle.json")
        return snap.stem.split("-", 1)[1] if "-" in snap.stem else snap.stem

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < self._latest()

    # --- files -----------------------------------------------------------
    def path(self, rel: str) -> Path:
        """A file inside the project; refuses anything that escapes it."""
        p = (self.root / rel).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise PermissionError(f"{rel!r} is outside the project")
        return p

    def results(self) -> list[dict]:
        out = []
        for f in sorted((self.root / "results").rglob("*")):
            if f.is_file() and f.suffix in (".stl", ".mp4", ".json", ".png"):
                out.append({"path": str(f.relative_to(self.root)),
                            "size": f.stat().st_size, "modified": f.stat().st_mtime})
        return out

    def duplicate(self, name: str) -> "Project":
        dst = self.root.parent / slug(name)
        if dst.exists():
            raise FileExistsError(f"a project called {dst.name!r} already exists")
        shutil.copytree(self.root, dst, ignore=shutil.ignore_patterns("jobs"))
        (dst / "jobs").mkdir()
        meta = json.loads((dst / "project.json").read_text())
        meta.update(name=name, created=time.time())
        (dst / "project.json").write_text(json.dumps(meta, indent=1))
        return Project(dst)
