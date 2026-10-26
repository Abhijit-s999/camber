"""Reader for the UIUC Propeller Data Site.

The database holds wind-tunnel measurements for several hundred small
propellers across four volumes (Brandt 2005-2008, Deters 2009-2015, Dantsker
2020, Dantsker 2021-2022). It is the ground truth this project's blade-element
model is validated against: 123 propellers carry *both* blade geometry and
measured performance, which is what makes the comparison falsifiable rather
than decorative.

**Licensing.** The archive is downloaded for local use and is git-ignored. It
is not redistributed with Camber, and whether a derived catalogue may be
shipped is still open -- see DECISIONS.md LIC-01.

**File formats**, three of them, all whitespace-delimited with a header line:

* ``<key>_geom.txt``            -- ``r/R  c/R  beta`` (beta in degrees)
* ``<key>_static_<tunnel>.txt`` -- ``RPM  CT  CP``
* ``<key>_<tunnel>_<rpm>.txt``  -- ``J  CT  CP  eta``

**Coefficient convention** is the propeller one, with *n* in revolutions per
second and *D* the diameter -- not the rotor-disc convention used in helicopter
work, which normalises on tip speed and disc area and differs by a factor of
pi^3/8. Mixing the two is the classic way to be wrong by 3.9x:

    J  = V / (n D)        CT = T / (rho n^2 D^4)        CP = P / (rho n^3 D^5)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..spec.units import INCH

#: The propeller map scaled across radii (OPT-05) when nobody names one: the
#: one the README, `camber endurance` and the app's Optimise step all use.
REFERENCE_PROP = "gwsdd_5x4.3"

#: `apce_11x5.5` -> family `apce`, 11 in diameter, 5.5 in pitch. A trailing
#: `_3b` names a three-blade variant of a two-blade design.
_KEY = re.compile(r"^(?P<family>[a-z0-9\-]+?)_(?P<dia>\d+(?:\.\d+)?)x"
                  r"(?P<pitch>\d+(?:\.\d+)?)(?:_(?P<blades>\d)b)?$")

#: Families whose file names drop the decimal point: `apce_14x85` is the 14 x
#: 8.5 and `ancf_125x6` the 12.5 x 6, by the archive's own headings. Only
#: these two do it; elsewhere a large number is a size in millimetres
#: (`ef_130x70`), so the rule is per family rather than a guess from size.
_POINT_DROPPED = {"apce", "ancf"}


class UiucError(ValueError):
    pass


@dataclass(frozen=True)
class BladeStation:
    """One radial station of the blade."""

    r_over_R: float
    chord_over_R: float
    twist_rad: float


@dataclass(frozen=True)
class PropGeometry:
    key: str
    family: str
    diameter_m: float
    pitch_m: float
    stations: tuple[BladeStation, ...]

    @property
    def radius_m(self) -> float:
        return self.diameter_m / 2.0

    @property
    def pitch_to_diameter(self) -> float:
        return self.pitch_m / self.diameter_m


@dataclass(frozen=True)
class PerfPoint:
    advance_ratio: float
    ct: float
    cp: float
    efficiency: float


@dataclass(frozen=True)
class DynamicRun:
    """A sweep of advance ratio at one fixed RPM."""

    key: str
    tunnel_id: str
    rpm: float
    points: tuple[PerfPoint, ...]

    @property
    def n_rev_s(self) -> float:
        return self.rpm / 60.0


@dataclass(frozen=True)
class StaticPoint:
    rpm: float
    ct: float
    cp: float


@dataclass(frozen=True)
class StaticRun:
    key: str
    tunnel_id: str
    points: tuple[StaticPoint, ...]


def parse_key(key: str) -> tuple[str, float, float]:
    """`apce_10x5` -> ('apce', 0.254 m, 0.127 m). Inches in, metres out."""
    family, diameter, pitch, _ = parse_name(key)
    return family, diameter, pitch


def parse_name(key: str) -> tuple[str, float, float, int]:
    """Family, diameter (m), pitch (m) and blade count from a propeller key."""
    m = _KEY.match(key)
    if not m:
        raise UiucError(
            f"cannot parse propeller key {key!r}; expected <family>_<dia>x<pitch>"
        )
    dia, pitch = float(m["dia"]), float(m["pitch"])
    if m["family"] in _POINT_DROPPED:
        if dia >= 100:
            dia /= 10
        # 12x13 is a real pitch; 12x65 is 6.5. No propeller here has a pitch
        # above twice its diameter.
        if pitch > 2 * dia and "." not in m["pitch"]:
            pitch /= 10
    return m["family"], dia * INCH, pitch * INCH, int(m["blades"] or 2)


def _rows(path: Path, expected: int) -> list[list[float]]:
    """Numeric rows, skipping the header and any malformed line.

    Some files in the archive carry a stray blank or a repeated header between
    blocks. Skipping non-numeric lines is safe here because every data line is
    entirely numeric -- there is no in-band text to lose.
    """
    out: list[list[float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != expected:
            continue
        try:
            out.append([float(p) for p in parts])
        except ValueError:
            continue
    if not out:
        raise UiucError(f"{path.name}: no numeric rows with {expected} columns")
    return out


def read_geometry(path: Path) -> PropGeometry:
    key = path.stem[: -len("_geom")]
    family, diameter, pitch = parse_key(key)
    stations = tuple(
        BladeStation(r, c, beta * 3.141592653589793 / 180.0)
        for r, c, beta in _rows(path, 3)
    )
    if len(stations) < 3:
        raise UiucError(f"{path.name}: only {len(stations)} blade stations")
    return PropGeometry(key, family, diameter, pitch, stations)


def read_dynamic(path: Path) -> DynamicRun:
    stem = path.stem
    key, tunnel_id, rpm_text = stem.rsplit("_", 2)
    try:
        rpm = float(rpm_text)
    except ValueError as exc:
        raise UiucError(f"{path.name}: trailing token {rpm_text!r} is not an RPM") from exc
    points = tuple(PerfPoint(j, ct, cp, eta) for j, ct, cp, eta in _rows(path, 4))
    return DynamicRun(key, tunnel_id, rpm, points)


def read_static(path: Path) -> StaticRun:
    key, tunnel_id = path.stem.split("_static_")
    points = tuple(StaticPoint(rpm, ct, cp) for rpm, ct, cp in _rows(path, 3))
    return StaticRun(key, tunnel_id, points)


@dataclass
class Database:
    geometry: dict[str, PropGeometry]
    dynamic: dict[str, list[DynamicRun]]
    static: dict[str, list[StaticRun]]

    @property
    def validation_keys(self) -> list[str]:
        """Propellers with both blade geometry and measured performance.

        These, and only these, can be used to check a blade-element model:
        without geometry there is nothing to predict from, and without
        measurements nothing to check against.
        """
        return sorted(set(self.geometry) & set(self.dynamic))

    def runs_for(self, key: str) -> list[DynamicRun]:
        return self.dynamic.get(key, [])


def load(root: Path | str) -> Database:
    """Read every volume under an extracted UIUC-propDB directory."""
    root = Path(root)
    if not root.exists():
        raise UiucError(
            f"{root} not found. Fetch the propeller data first:\n"
            f"  camber catalogue fetch"
        )

    geometry: dict[str, PropGeometry] = {}
    dynamic: dict[str, list[DynamicRun]] = {}
    static: dict[str, list[StaticRun]] = {}

    for path in sorted(root.glob("volume-*/data/*.txt")):
        try:
            if path.stem.endswith("_geom"):
                g = read_geometry(path)
                geometry[g.key] = g
            elif "_static_" in path.stem:
                s = read_static(path)
                static.setdefault(s.key, []).append(s)
            else:
                d = read_dynamic(path)
                dynamic.setdefault(d.key, []).append(d)
        except (UiucError, ValueError):
            # A handful of files are notes or use a variant naming scheme.
            # Skipping them loses a little data; guessing at them would put
            # unvalidated numbers into the ground truth, which is worse.
            continue

    return Database(geometry, dynamic, static)
