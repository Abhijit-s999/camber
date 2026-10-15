"""Fetch source datasets onto the user's machine.

This is what Camber ships instead of the data: a builder. See sources.py for
why -- no source grants redistribution rights, so the catalogue is assembled
locally from the originals.

That costs a network connection on first run, which is the one place AI-07's
"works with nothing installed and no network" does not hold. The distinction
worth keeping is between *setup* and *runtime*: fetching is a one-off step the
user initiates, and everything afterwards -- searching the catalogue, running
the optimiser, evaluating physics -- is offline and deterministic.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .sources import SOURCES, Source, md5_of

from ..paths import raw as _raw

DEFAULT_ROOT = _raw()


class FetchError(RuntimeError):
    pass


@dataclass
class FetchResult:
    source: str
    path: Path | None
    bytes: int
    cached: bool
    verified: bool | None
    message: str = ""


def fetch(key: str, root: Path = DEFAULT_ROOT, force: bool = False,
          progress=None) -> FetchResult:
    """Download one source, verifying its checksum when one is published."""
    if key not in SOURCES:
        raise KeyError(f"unknown source {key!r}; have {sorted(SOURCES)}")
    source = SOURCES[key]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = source.cache_path(root)

    if target.exists() and not force:
        ok = md5_of(target) == source.md5 if source.md5 else None
        message = "already present"
        if ok is not False and unpack(source, target, root):
            message += ", extracted"
        return FetchResult(key, target, target.stat().st_size, True, ok, message)

    if not source.url.endswith((".zip", ".tar.gz", ".tgz", ".csv", ".json")):
        return FetchResult(
            key, None, 0, False, None,
            f"{source.name} has no bulk download; see {source.url}. Camber "
            f"cannot scrape it on your behalf, and its terms do not permit "
            f"redistribution, so this source must be gathered manually.")

    try:
        with urllib.request.urlopen(source.url, timeout=120) as response, \
                target.open("wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"could not fetch {source.name}: {exc}") from exc

    verified = None
    if source.md5:
        verified = md5_of(target) == source.md5
        if not verified:
            raise FetchError(
                f"{source.name} downloaded but its checksum does not match the "
                f"published value. Refusing to use it: silently proceeding with "
                f"altered source data would poison every result downstream.")
    unpack(source, target, root, force=force)
    return FetchResult(key, target, target.stat().st_size, False, verified)


def unpack(source: Source, archive: Path, root: Path, force: bool = False) -> bool:
    """Extract an archive source next to it, once. True if it extracted.

    Every reader takes the extracted tree (`paths.propdb()` for UIUC), so a
    fetch that left only the zip behind would leave the user one undocumented
    step from a working catalogue.
    """
    if not source.unpacks_to or not str(archive).endswith(".zip"):
        return False
    dest = Path(root) / source.unpacks_to
    if dest.exists() and not force:
        return False
    base = Path(root).resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            # An entry that resolves outside the root is a path-traversal
            # attempt or a broken archive; either way it is not extracted.
            if not (base / name).resolve().is_relative_to(base):
                raise FetchError(f"{source.name}: archive entry {name!r} "
                                 f"would land outside {root}; refusing to extract")
        zf.extractall(base)
    if not dest.exists():
        raise FetchError(f"{source.name}: extracted, but {source.unpacks_to}/ "
                         f"is not in the archive")
    return True


def fetch_all(root: Path = DEFAULT_ROOT, force: bool = False) -> list[FetchResult]:
    out: list[FetchResult] = []
    for key in SOURCES:
        try:
            out.append(fetch(key, root, force))
        except FetchError as exc:
            out.append(FetchResult(key, None, 0, False, False, str(exc)))
    return out
