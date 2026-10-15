"""Where Camber's data lives, independent of the working directory.

Two kinds, kept apart because one is read-only once installed:

* **shipped** -- files that come with Camber: the fitted interference model,
  the validation summary. In a checkout that is `data/` in the repository; in
  an installed app it is next to the package.
* **user** -- what `camber catalogue fetch` downloads. Third-party data is never
  redistributed (LIC-01), so it has to land somewhere writable: the checkout's
  `data/` in development, `$XDG_DATA_HOME/camber` once installed.

The app's starting templates, the example pilots and the OpenFOAM wrapper
follow the same rule as shipped data: the checkout's copy in development, the
copy the wheel puts inside the package once installed (PKG-02).

Every default used to be `Path("data/...")`, relative to wherever Camber was
run from. The app runs jobs from a project folder, where that path does not
exist -- the interference estimate would have been skipped without a word.
"""

from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
_REPO_DATA = _REPO / "data"


def _bundled(relative: str) -> Path:
    """The checkout's copy of a shipped file if there is one, else the wheel's."""
    checkout = _REPO / relative
    return checkout if checkout.exists() else _PKG / relative


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


def examples() -> Path:
    """The example vehicles the app starts new projects from."""
    return _bundled("examples")


def pilots() -> Path:
    return examples() / "pilots"


def foam_env() -> Path:
    """The one entry point to OpenFOAM (see scripts/foam-env.sh)."""
    return _bundled("scripts/foam-env.sh")
