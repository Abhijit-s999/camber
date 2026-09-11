"""Measure a vehicle across its own flight envelope, and tabulate the result.

For every test point from `envelope`, one steady RANS case: the airframe in a
freestream arriving at the test point's attitude, with the rotors as actuator
discs carrying the thrust that flight condition needs. The body force and
moment come back from OpenFOAM's `forces` object, are moved into FLU axes and
about the centre of gravity, and become coefficients in an `AeroTable` the
simulator flies with.

The rotors are included because they change the answer. A body in a rotor
wash is not a body in clean air: at hover the wash is the whole download, and
in forward flight it sweeps back over the airframe at an angle set by the
ratio of flight speed to induced velocity. Testing the bare airframe would
measure a vehicle that does not exist.

What is measured is the force on the *airframe* -- the `body` patch. The rotor
thrust itself is the simulator's to apply, from the same allocation matrix as
everywhere else, so nothing is counted twice.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..control import solve_hover
from ..sim import frames
from .case import CaseBuilder, CaseConfig
from .envelope import TestPoint, envelope
from . import run as foamrun

RHO = 1.225


def reference(vehicle) -> tuple[float, float]:
    """(area, length) the coefficients are quoted against.

    Area is the airframe's footprint, the same `planform_area()` sum used for
    blockage and for the drag estimate; length is the rotor span. Any fixed pair
    would do -- they cancel between the CFD and the simulator -- but these two
    mean something to a reader.
    """
    from .case import rotor_span
    return (sum(b.shape.planform_area() for b in vehicle.all_bodies()),
            rotor_span(vehicle))


def thrust_split(vehicle, total: float) -> dict[str, float]:
    """Per-rotor thrust for a total, in the vehicle's own hover-trim ratio.

    Not an equal split: the CG is not under the rotor centroid, so level
    flight needs the same uneven distribution hover does.
    """
    h = solve_hover(vehicle)
    if not h.feasible:
        raise ValueError(f"{vehicle.name} cannot hover: {h.reason}")
    share = np.asarray(h.thrusts) / float(np.sum(h.thrusts))
    return {t.name: float(total * s) for t, s in zip(vehicle.thrusters, share)}


def run_point(vehicle, point: TestPoint, stl: Path, root: Path,
              config: CaseConfig | None = None) -> dict:
    """One case, meshed and solved; returns the measured body force/moment."""
    base = config or CaseConfig()
    cfg = CaseConfig(**{**asdict(base), "u_inf": tuple(point.relative_wind_ned())})
    thrusts = thrust_split(vehicle, point.thrust) if point.thrust > 0 else \
        {t.name: 0.0 for t in vehicle.thrusters}
    case = CaseBuilder(vehicle, thrusts, cfg).build(root / point.name, stl)
    t0 = time.time()
    foamrun.mesh(case)
    solved = foamrun.solve(case)
    f = foamrun.read_body_force(case)
    m = foamrun.read_body_moment(case)
    if f is None or m is None:
        raise RuntimeError(f"{point.name}: no forces written ({solved})")
    return {"point": asdict(point), "cells": foamrun.cell_count(case),
            "seconds": round(time.time() - t0, 1),
            "force_ned": [f["fx"], f["fy"], f["fz"]],
            "moment_ned_origin": [m["mx"], m["my"], m["mz"]]}


def tabulate(vehicle, results: list[dict]) -> dict:
    """Measured forces -> coefficients in FLU axes, moments about the CG."""
    area, length = reference(vehicle)
    cg = np.asarray(vehicle.mass_properties().cg.as_list())
    rows = []
    for r in results:
        p = r["point"]
        q = 0.5 * RHO * p["speed"] ** 2
        F = np.asarray(r["force_ned"])
        # The forces object reports moments about (0, 0, 0); the simulator
        # applies them about the CG. M_cg = M_o - r_cg x F.
        M = np.asarray(r["moment_ned_origin"]) - np.cross(cg, F)
        Ff, Mf = frames.vec(F), frames.vec(M)
        rows.append((p["alpha_deg"], Ff[0] / (q * area), Ff[2] / (q * area),
                     Mf[1] / (q * area * length)))
    rows.sort()
    a, cx, cz, cm = (list(x) for x in zip(*rows))
    return {"vehicle": vehicle.name, "alpha_deg": a, "CX": cx, "CZ": cz, "Cm": cm,
            "area": area, "length": length,
            "speeds": [r["point"]["speed"] for r in results],
            "cases": results}


def sweep(vehicle, stl: Path, root: Path, *, n: int = 6,
          config: CaseConfig | None = None, points=None, log=print) -> dict:
    """Run the envelope and return the table (also written to root/aero.json)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    points = points or envelope(vehicle, n=n)
    results = []
    for p in points:
        log(f"  {p.name}: alpha {p.alpha_deg:+.1f} deg at {p.speed:.1f} m/s, "
            f"thrust {p.thrust:.2f} N")
        results.append(run_point(vehicle, p, stl, root, config))
        r = results[-1]
        log(f"      {r['cells']:,} cells, {r['seconds']:.0f} s, "
            f"body force NED ({', '.join(f'{x:+.3f}' for x in r['force_ned'])}) N")
    table = tabulate(vehicle, results)
    (root / "aero.json").write_text(json.dumps(table, indent=1))
    return table
