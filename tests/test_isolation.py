"""The product must not import its own build tooling.

This is docs/DECISIONS.md AI-04 expressed as a test. If it ever fails, someone
has made NVIDIA a runtime dependency of Camber, which breaks both the offline
guarantee and the open-source claim.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, stays inside the package
                continue
            if node.module:
                found.add(node.module.split(".")[0])
    return found


def test_camber_never_imports_bench():
    offenders = []
    for path in (ROOT / "camber").rglob("*.py"):
        if "bench" in _imports(path):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        f"camber must not import bench (AI-04). Offending files: {offenders}"
    )


def test_camber_has_no_network_client_dependency():
    """The product's hard dependencies must not include an HTTP client."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime = pyproject["project"]["dependencies"]
    banned = {"httpx", "requests", "aiohttp", "urllib3", "openai", "anthropic", "mcp"}
    found = {d.split("[")[0].split(">")[0].split("=")[0].strip() for d in runtime} & banned
    assert not found, f"network clients must stay out of runtime deps (AI-04): {found}"


def test_bench_is_not_packaged():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "bench" not in packages, "the bench must never ship (AI-04)"
