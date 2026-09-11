"""Where Camber's data lives, independent of the working directory.

Two kinds, kept apart because one is read-only once installed:

* **shipped** -- files that come with Camber: the fitted interference model,
  the validation summary. In a checkout that is `data/` in the repository; in
  an installed app it is next to the package.
* **user** -- what `camber catalogue fetch` downloads. Third-party data is never
  redistributed (LIC-01), so it has to land somewhere writable: the checkout's
  `data/` in development, `$XDG_DATA_HOME/camber` once installed.

Every default used to be `Path("data/...")`, relative to wherever Camber was
run from. The app runs jobs from a project folder, where that path does not
exist -- the interference estimate would have been skipped without a word.
"""

from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_REPO_DATA = _PKG.parent / "data"


def shipped() -> Path:
    env = os.environ.get("CAMBER_DATA")
    if env:
        return Path(env)
    for p in (_REPO_DATA, _PKG / "data"):
        if p.exists():
            return p
    return _PKG / "data"


def user() -> Path:
    env = os.environ.get("CAMBER_USER_DATA")
    if env:
        return Path(env)
    if _REPO_DATA.exists() and os.access(_REPO_DATA, os.W_OK):
        return _REPO_DATA
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "camber"


def interference_model() -> Path:
    return shipped() / "interference.json"


def raw() -> Path:
    return user() / "raw"


def propdb() -> Path:
    return raw() / "UIUC-propDB"
