"""The motors Camber can choose from: a table compiled from makers' datasheets.

Unlike the propellers, this table ships with Camber (LIC-03). Each row is one
maker's published specification for its own product, with the page it came
from and the text quoted, so a reviewer can check a number without trusting
whoever typed it (AI-06). The table is Camber's own selection; no one else's
database was copied to make it.

The physics needs three numbers, Kv, winding resistance and no-load current
(see physics/motor.py), and the constraints three more: mass, the maker's
current rating and the cell range. A motor whose page lacks one of those is
left out rather than completed by guesswork.
"""

from __future__ import annotations

import functools
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..physics.battery import CELL_FULL_V

TABLE = Path(__file__).with_name("motors.toml")

_ID = re.compile(r"^[a-z0-9][a-z0-9\-]*$")
REQUIRED = ("id", "maker", "model", "kv", "resistance_mohm", "no_load_current_a",
            "mass_g", "max_current_a", "cells_max", "url", "quote")


class MotorTableError(ValueError):
    pass


@dataclass(frozen=True)
class CatalogueMotor:
    id: str
    maker: str
    model: str
    kv: float
    resistance_ohm: float
    no_load_current_a: float
    mass_kg: float
    max_current_a: float
    cells_min: int
    cells_max: int
    url: str
    stator: str = ""
    #: How long the maker says the motor takes `max_current_a`; None when the
    #: page calls it continuous or does not say.
    max_current_s: float | None = None
    max_power_w: float | None = None
    props_in: tuple[float, ...] = ()
    quote: str = ""
    notes: str = ""

    @property
    def label(self) -> str:
        return f"{self.maker} {self.model} {self.kv:g}Kv"

    @property
    def motor(self):
        """The physics model of it, which needs scipy (the `physics` extra);
        the table itself loads with numpy alone (AI-07)."""
        from ..physics.motor import Motor
        return Motor(self.id, kv_rpm_v=self.kv, resistance_ohm=self.resistance_ohm,
                     no_load_current_a=self.no_load_current_a, mass_kg=self.mass_kg,
                     max_current_a=self.max_current_a,
                     max_voltage_v=self.cells_max * CELL_FULL_V, source=self.url)


def parse(rows: list[dict]) -> list[CatalogueMotor]:
    """Rows of the table, checked. Raises on the first bad one, by id."""
    out: list[CatalogueMotor] = []
    seen: set[str] = set()
    for i, r in enumerate(rows):
        name = r.get("id", f"row {i + 1}")
        missing = [k for k in REQUIRED if r.get(k) in (None, "")]
        if missing:
            raise MotorTableError(f"{name}: missing {', '.join(missing)}")
        if not _ID.match(r["id"]) or r["id"] in seen:
            raise MotorTableError(f"{name}: id must be unique and [a-z0-9-]")
        seen.add(r["id"])
        if not str(r["url"]).startswith("https://"):
            raise MotorTableError(f"{name}: url must be an https page")
        for k in ("kv", "resistance_mohm", "no_load_current_a", "mass_g", "max_current_a"):
            if not float(r[k]) > 0:
                raise MotorTableError(f"{name}: {k} must be positive")
        # Some makers state only a maximum ("Max Lipo Cell 4S"); that sets
        # no floor, so none is invented.
        cells_min = r.get("cells_min", 1)
        if not 1 <= cells_min <= r["cells_max"] <= 14:
            raise MotorTableError(f"{name}: cell range {cells_min}-{r['cells_max']}")
        out.append(CatalogueMotor(
            id=r["id"], maker=r["maker"], model=r["model"], kv=float(r["kv"]),
            resistance_ohm=float(r["resistance_mohm"]) / 1000.0,
            no_load_current_a=float(r["no_load_current_a"]),
            mass_kg=float(r["mass_g"]) / 1000.0,
            max_current_a=float(r["max_current_a"]),
            cells_min=int(cells_min), cells_max=int(r["cells_max"]),
            url=r["url"], stator=str(r.get("stator", "")),
            max_current_s=r.get("max_current_duration_s"),
            max_power_w=r.get("max_power_w"),
            props_in=tuple(float(d) for d in r.get("props_in", ())),
            quote=r["quote"], notes=r.get("notes", "")))
    return out


@functools.lru_cache(maxsize=2)
def load(path: str | None = None) -> tuple[CatalogueMotor, ...]:
    """The shipped table, or another in the same format."""
    p = Path(path) if path else TABLE
    try:
        rows = tomllib.loads(p.read_text())["motor"]
    except (OSError, tomllib.TOMLDecodeError, KeyError) as exc:
        raise MotorTableError(f"{p}: {exc}") from exc
    return tuple(parse(rows))
