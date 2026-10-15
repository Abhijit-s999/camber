"""Where component data comes from, and why Camber fetches it rather than ships it.

**No third-party measurements are redistributed with Camber.** The one source
it reads today, the UIUC Propeller Database, says "(c) 1994-2026 UIUC Applied
Aerodynamics Group", requests a citation, and grants no licence. Under
copyright the default is reserved.

Raw measurements may not be copyrightable everywhere, but a *compiled* database
attracts protection for its selection and arrangement in most jurisdictions,
and more strongly in some than in US law. So the catalogue is built on the
user's machine from the original source, and what Camber ships is the builder.

APC, Tyto Robotics and ThrustCurve.org were checked as well (2026-09-09, see
DECISIONS.md LIC-01) and none of them grants redistribution either. They are
not listed here because nothing reads them yet; they come back with component
selection.

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
    #: The directory an archive extracts to, when the source is one.
    unpacks_to: str = ""

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
        unpacks_to="UIUC-propDB",
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
