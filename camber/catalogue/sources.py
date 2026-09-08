"""Where component data comes from, and why Camber fetches it rather than ships it.

**No third-party measurements are redistributed with Camber.** Every source
below was checked (2026-09-09) and none of them grants redistribution rights:

* UIUC Propeller Database -- "(c) 1994-2026 UIUC Applied Aerodynamics Group",
  a requested citation, and no licence. Under copyright the default is
  reserved.
* APC performance files -- downloadable, with no licence statement anywhere.
* Tyto Robotics database -- "(c) 2026"; the terms of service are silent on
  reuse, and silence is not permission.
* ThrustCurve.org -- licences are declared **per file** and range from public
  domain to "other (restricted) with explicit copyright", so no blanket
  redistribution is possible even in principle.

Raw measurements may not be copyrightable everywhere, but a *compiled* database
attracts protection for its selection and arrangement in most jurisdictions,
and more strongly in some than in US law. So the catalogue is built on the
user's machine from the original sources, and what Camber ships is the builder.

Each source keeps its citation, and honouring UIUC's requested form is both
free and correct.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    url: str
    #: What the site actually says about reuse, as checked. Not a summary of
    #: what would be convenient.
    terms: str
    redistributable: bool
    citation: str = ""
    md5: str = ""

    def cache_path(self, root: Path) -> Path:
        return Path(root) / f"{self.key}{Path(self.url).suffix or '.dat'}"


SOURCES: dict[str, Source] = {
    "uiuc": Source(
        key="uiuc",
        name="UIUC Propeller Data Site, Volumes 1-4",
        url="https://m-selig.ae.illinois.edu/props/download/UIUC-propDB.zip",
        terms="(c) 1994-2026 UIUC Applied Aerodynamics Group. A citation is "
              "requested; no licence is granted and no redistribution right is "
              "given.",
        redistributable=False,
        citation="J.B. Brandt, R.W. Deters, G.K. Ananda, O.D. Dantsker and "
                 "M.S. Selig, UIUC Propeller Database, Vols 1-4, University of "
                 "Illinois at Urbana-Champaign, Department of Aerospace "
                 "Engineering, https://m-selig.ae.illinois.edu/props/propDB.html",
        md5="a41e484f1fd0fb6ff80b76e27410808b",
    ),
    "apc": Source(
        key="apc",
        name="APC propeller performance files",
        url="https://www.apcprop.com/technical-information/file-downloads/",
        terms="No licence statement is published with the data files.",
        redistributable=False,
        citation="APC Propellers, performance data, https://www.apcprop.com/",
    ),
    "tyto": Source(
        key="tyto",
        name="Tyto Robotics motor, propeller and ESC database",
        url="https://database.tytorobotics.com/",
        terms="(c) 2026 Tyto Robotics Inc. The terms of service do not address "
              "reuse or redistribution.",
        redistributable=False,
        citation="Tyto Robotics, drone motor and propeller database, "
                 "https://database.tytorobotics.com/",
    ),
    "thrustcurve": Source(
        key="thrustcurve",
        name="ThrustCurve.org certified motor data",
        url="https://www.thrustcurve.org/",
        terms="Licences are declared per file and range from public domain to "
              "explicitly restricted. No blanket redistribution is possible.",
        redistributable=False,
        citation="J. Coker, ThrustCurve.org, https://www.thrustcurve.org/",
    ),
}


def redistributable() -> list[Source]:
    """Sources that may legally ship with Camber. Currently none."""
    return [s for s in SOURCES.values() if s.redistributable]


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def notice() -> str:
    """Attribution text, for NOTICE and for `camber catalogue sources`."""
    lines = [
        "Camber builds its component catalogue from third-party datasets.",
        "None of them is redistributed with Camber; each is fetched from its",
        "origin on the user's machine. Cite them if you publish results.",
        "",
    ]
    for s in SOURCES.values():
        lines += [f"{s.name}", f"  {s.url}", f"  terms: {s.terms}"]
        if s.citation:
            lines.append(f"  cite:  {s.citation}")
        lines.append("")
    return "\n".join(lines)
