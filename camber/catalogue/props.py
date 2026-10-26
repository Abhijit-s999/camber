"""The propellers Camber can choose from: the ones the UIUC tunnel measured.

The archive on disk (`camber catalogue fetch`, LIC-01) is the catalogue. A
propeller is a candidate when it has static measurements, a size that parses,
and someone sells it. The archive also holds 3D-printed research designs
(DA4002, DA4022, DA4052, NR640) and some micro propellers named in
millimetres; a bill of materials naming one of the first is one nobody can
fill, and the second sit below the smallest motor in the table.

Each candidate is flown on its own measurements (`MeasuredPropeller`), not on
one blade-element map scaled across sizes, which is what OPT-05 did: the
Reynolds error that carried disappears for anything chosen here (CAT-02).

The masses come from `prop_masses.toml`, published weights with their pages.
A propeller without one gets an estimate from a power law through the
published ones, and says so (CAT-04).
"""

from __future__ import annotations

import functools
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..spec.units import INCH
from . import uiuc


@dataclass(frozen=True)
class Family:
    maker: str
    line: str
    retail: bool = True
    blades: int = 2
    #: Maximum rpm times diameter in inches, where the maker publishes one.
    rpm_inch: float | None = None
    rpm_source: str = ""


#: APC publishes a safe maximum rpm per line, as a constant over the diameter
#: in inches. Free Flight and Carbon Fiber have none that maps cleanly, and no
#: other maker in the archive publishes one (CAT-03).
APC_RPM = "https://www.apcprop.com/technical-information/rpm-limits/"

#: Family codes as the archive's own section headings name them.
FAMILIES: dict[str, Family] = {
    "apce": Family("APC", "Thin Electric", rpm_inch=150_000, rpm_source=APC_RPM),
    "apcsf": Family("APC", "Slow Flyer", rpm_inch=65_000, rpm_source=APC_RPM),
    # "Sport Propellers (for IC Engines) on Electric Motors": 225,000.
    "apcsp": Family("APC", "Sport", rpm_inch=225_000, rpm_source=APC_RPM),
    "apcff": Family("APC", "Free Flight"),
    "apc29ff": Family("APC", "Free Flight (29)"),
    "apccf": Family("APC", "Carbon Fiber"),
    "gwsdd": Family("GWS", "Direct Drive"),
    "gwssf": Family("GWS", "Slow Flyer"),
    "grcp": Family("Graupner", "CAM Prop"),
    "grcsp": Family("Graupner", "CAM Slim"),
    "grsn": Family("Graupner", "Super Nylon"),
    "ma": Family("Master Airscrew", "Classic"),
    "mae": Family("Master Airscrew", "Electric"),
    "magf": Family("Master Airscrew", "G/F"),
    "mas": Family("Master Airscrew", "Scimitar"),
    "kyosho": Family("Kyosho", ""),
    "kavfk": Family("Kavon", "FK"),
    "zin": Family("Zingali", ""),
    "rusp": Family("Rev Up", "Special Prop Series"),
    "ance": Family("Aero-naut", "CAM Carbon Electric"),
    "ancf": Family("Aero-naut", "CAM Carbon folding"),
    "mi": Family("Micro Invent", "2-blade"),
    "mit": Family("Micro Invent", "3-blade", blades=3),
    "da4002": Family("UIUC", "DA4002 research design", retail=False),
    "da4022": Family("UIUC", "DA4022 research design", retail=False),
    "da4052": Family("UIUC", "DA4052 research design", retail=False),
    "nr640": Family("UIUC", "NR640 research design", retail=False),
}

#: Above this a propeller is not something the table can stand for. The
#: archive's largest real propellers are 21 in; its sizes in millimetres
#: parse as 57 to 140 "inches".
MAX_DIAMETER_IN = 30.0

MASSES = Path(__file__).with_name("prop_masses.toml")


@dataclass(frozen=True)
class CatalogueProp:
    key: str
    family: Family
    diameter_m: float
    pitch_m: float
    blades: int
    mass_kg: float
    #: The page the weight came from, or how it was estimated.
    mass_source: str
    measured: object       # physics.propulsion.MeasuredPropeller

    @property
    def diameter_in(self) -> float:
        return self.diameter_m / INCH

    @property
    def pitch_in(self) -> float:
        return self.pitch_m / INCH

    @property
    def label(self) -> str:
        return describe(self.key)

    @property
    def max_rpm(self) -> float | None:
        f = self.family
        return f.rpm_inch / self.diameter_in if f.rpm_inch else None


def describe(key: str) -> str:
    """A propeller's name from its key alone, without loading the archive."""
    try:
        code, dia, pitch, _ = uiuc.parse_name(key)
    except uiuc.UiucError:
        return key
    f = FAMILIES.get(code)
    if f is None:
        return key
    line = f" {f.line}" if f.line else ""
    return f"{f.maker}{line} {dia / INCH:g}x{pitch / INCH:g}"


def published_masses(path: Path = MASSES) -> dict[str, tuple[float, str]]:
    """UIUC key -> (kg, URL) for every propeller with a published weight."""
    if not path.exists():
        return {}
    rows = tomllib.loads(path.read_text())["prop"]
    return {r["uiuc_key"]: (r["mass_g"] / 1000.0, r["url"]) for r in rows}


def mass_law(masses: dict[str, tuple[float, str]]) -> tuple[float, float, dict[str, float]]:
    """mass = a * D**b, D in inches, through the published weights.

    One exponent for every propeller, fitted in log space because the weights
    span three decades. The scale is per family wherever a family has two
    published weights, because construction dominates it: a 5-inch Micro
    Invent weighs 0.35 g and a 5-inch APC Sport ten times that. One law for
    all scattered by a factor of 1.7; per family, 1.3 left out one at a time
    (CAT-04).

    Returns (a for families without enough data, b, {family: a}).
    """
    rows = []
    for key, (kg, _) in masses.items():
        try:
            code, dia, _, _ = uiuc.parse_name(key)
        except uiuc.UiucError:
            continue
        rows.append((code, math.log(dia / INCH), math.log(kg)))
    if len(rows) < 3:
        raise ValueError("too few published propeller weights to estimate the rest")
    b, log_a = np.polyfit([r[1] for r in rows], [r[2] for r in rows], 1)
    scale: dict[str, list[float]] = {}
    for code, log_d, log_m in rows:
        scale.setdefault(code, []).append(log_m - b * log_d)
    per_family = {code: math.exp(float(np.mean(v))) for code, v in scale.items() if len(v) >= 2}
    return math.exp(log_a), b, per_family


def _static(runs: list[uiuc.StaticRun]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One static sweep: the one reaching the highest rpm.

    Several propellers were measured in two tunnels. Mixing the sweeps would
    zigzag between them; the one reaching highest is the nearer to how a
    multirotor turns it.
    """
    run = max(runs, key=lambda r: (max(p.rpm for p in r.points), len(r.points)))
    pts = sorted(run.points, key=lambda p: p.rpm)
    return (np.array([p.rpm for p in pts]), np.array([p.ct for p in pts]),
            np.array([p.cp for p in pts]))


def _runs(runs: list[uiuc.DynamicRun]) -> tuple:
    out = []
    for r in runs:
        pts = sorted(r.points, key=lambda p: p.advance_ratio)
        if len(pts) < 2:
            continue
        out.append((r.rpm, np.array([p.advance_ratio for p in pts]),
                    np.array([p.ct for p in pts]), np.array([p.cp for p in pts])))
    return tuple(out)


def build(db: uiuc.Database, masses: dict[str, tuple[float, str]] | None = None,
          retail_only: bool = True) -> list[CatalogueProp]:
    """Every choosable propeller in an archive, smallest first."""
    # Here rather than at the top: the propulsion module needs scipy, and
    # naming a propeller (`describe`) must not (AI-07).
    from ..physics.propulsion import MeasuredPropeller

    masses = published_masses() if masses is None else masses
    a_all, b, a_family = mass_law(masses)
    out: list[CatalogueProp] = []
    for key, static in db.static.items():
        try:
            code, dia, pitch, blades = uiuc.parse_name(key)
        except uiuc.UiucError:
            continue
        family = FAMILIES.get(code)
        if family is None or (retail_only and not family.retail):
            continue
        if dia / INCH > MAX_DIAMETER_IN or pitch <= 0:
            continue
        blades = max(blades, family.blades)
        rpm, ct, cp = _static(static)
        if (ct <= 0).any() or (cp <= 0).any():
            continue
        if key in masses:
            mass, source = masses[key]
        else:
            a = a_family.get(code, a_all)
            mass = a * (dia / INCH) ** b
            whose = f"{family.maker} {family.line}".strip() if code in a_family else "all"
            source = (f"estimated from diameter: {a * 1000:.3g} g x D^{b:.2f}, the law "
                      f"through {len(masses)} published weights scaled to {whose} propellers")
        out.append(CatalogueProp(
            key, family, dia, pitch, blades, mass, source,
            MeasuredPropeller(dia, rpm, ct, cp, _runs(db.runs_for(key)), key)))
    return sorted(out, key=lambda p: (p.diameter_m, p.pitch_m, p.key))


@functools.lru_cache(maxsize=4)
def load(root: str | None = None, retail_only: bool = True) -> tuple[CatalogueProp, ...]:
    """The propeller catalogue from the fetched archive. Cached per root."""
    from ..paths import propdb
    return tuple(build(uiuc.load(root or propdb()), retail_only=retail_only))
