"""PHYS-09: the motor model and the measured propellers, against makers' thrust stands.

Propellers were validated against UIUC (PHYS-01..08); motors never were. Some
makers publish a test table per motor -- volts, throttle, amps, grams, rpm --
run on a propeller that UIUC also measured. Those rows are a held-out set for
the coupled model the optimiser ranks on: the motor from the shipped table,
the propeller from the tunnel, nothing fitted to the rows themselves.

The acceptance band was fixed before any row was compared (AI-05): **15% on
thrust and on current.** Two checks were planned per row:

* **at the measured rpm** -- thrust from the tunnel's coefficients, current
  from the motor model carrying the propeller's torque. This separates the
  propeller data from the motor's speed constant.
* **at full throttle, from volts alone** -- the whole coupled solve. No source
  prints throttle or the voltage under load, so this one never runs; the
  makers' round pack voltages are not what the motor saw.

**What it found (PHYS-09):** on the four motors whose charts name the
propeller's line and were run at full throttle (Cobra x3, Scorpion; 114 rows),
106 thrust and 108 current comparisons fall inside the band. Every thrust miss
is one of two propellers, the same way on every motor that ran it: APC 8x6E
reads 17-21% low and 12x8E 23-27% high, so it is the tunnel's sample against
the makers', not the motor. The SunnySky sweeps (41 rows) do not pass: their
page names "APC1047" with no line, flagged before the comparison, and runs a
9x4.7 to 1.6x APC's limit for the Slow Flyer it was taken to be. They are
reported, not gated. The test below holds the recorded result as a floor.
"""

import functools
import math
import statistics
import tomllib
from pathlib import Path

import pytest

from camber.catalogue import motors, props

ROOT = Path(__file__).resolve().parent.parent
PROPDB = ROOT / "data" / "raw" / "UIUC-propDB"
ROWS = Path(__file__).with_name("data") / "motor_tests.toml"
BAND = 0.15
G0 = 9.80665

pytestmark = pytest.mark.skipif(not PROPDB.exists(),
                                reason="UIUC data not fetched; `camber catalogue fetch`")


def rows():
    return tomllib.loads(ROWS.read_text())["test"]


@functools.lru_cache(maxsize=1)
def motor_table():
    """The shipped catalogue, plus motors kept here for validation only."""
    data = tomllib.loads(ROWS.read_text())
    extra = []
    for r in data.get("motor", []):
        r = dict(r)
        r.setdefault("cells_max", 14)      # no cell range published; unused here
        extra.append(r)
    return {m.id: m for m in (*motors.load(), *motors.parse(extra))}


def parts(row):
    m = motor_table()[row["motor_id"]]
    p = {x.key: x for x in props.load(str(PROPDB))}[row["prop_uiuc"]]
    return m, p


def at_rpm(row):
    """Model thrust (g) and supply current (A) at the row's measured rpm.

    The stands' "Amps" is the DC current from the supply (their Watts column
    is volts times amps), not the motor's: at part throttle the ESC draws
    throttle times the motor current (PHYS-10). The first run of this check
    compared motor current with it and read +290% on the lightest rows. A
    full-throttle row needs no correction; it ran from a sagging pack, so its
    printed voltage is not the supply's and the two currents are the same.
    """
    m, p = parts(row)
    omega = row["rpm"] * 2 * math.pi / 60
    thrust, torque = p.measured.thrust_torque(omega, 0.0)
    motor_current = torque / m.motor.kt + m.no_load_current_a
    motor_volts = omega / m.motor.kv_rad + motor_current * m.resistance_ohm
    duty = 1.0 if row.get("full_throttle") else min(1.0, motor_volts / row["voltage_v"])
    return thrust / G0 * 1000, motor_current * duty


def test_there_is_a_held_out_set_worth_the_name():
    rs = rows()
    assert len(rs) >= 10
    assert len({r["motor_id"] for r in rs}) >= 3
    for r in rs:
        assert r["url"].startswith("https://") and r["quote"]


def test_every_row_names_parts_camber_has():
    for r in rows():
        parts(r)


def errors(full_throttle: bool) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"thrust": [], "current": []}
    for r in rows():
        if bool(r.get("full_throttle")) != full_throttle:
            continue
        thrust, current = at_rpm(r)
        out["thrust"].append(thrust / r["thrust_g"] - 1)
        out["current"].append(current / r["current_a"] - 1)
    return out


@pytest.mark.parametrize("quantity", ["thrust", "current"])
def test_on_the_named_propellers_the_model_is_inside_the_band(quantity):
    e = errors(full_throttle=True)[quantity]
    inside = sum(abs(x) <= BAND for x in e)
    assert len(e) >= 100
    assert inside / len(e) >= 0.90, f"{inside}/{len(e)} inside +/-{BAND:.0%}"
    assert abs(statistics.median(e)) <= 0.10


def test_the_misses_are_two_propellers_not_the_motors():
    """Every thrust miss on the named propellers is the 8x6E or the 12x8E."""
    missed = set()
    for r in rows():
        if r.get("full_throttle"):
            thrust, _ = at_rpm(r)
            if abs(thrust / r["thrust_g"] - 1) > BAND:
                missed.add(r["prop_uiuc"])
    assert missed <= {"apce_8x6", "apce_12x8"}, missed
