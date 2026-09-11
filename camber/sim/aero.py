"""Aerodynamic forces on the airframe body, in flight.

Two sources, in order of preference: a table measured by CFD across the
attitudes and speeds the vehicle actually flies (`AeroTable`, produced by
`camber.cfd.sweep`), or an estimate from the vehicle's own extents
(`BodyDrag.estimate`). The estimate exists so the simulator flies before any
CFD has run; it is not the thing to trust at racing speed, where drag on a
nose-down airframe is most of what limits top speed.

Rotor-plane effects in forward flight -- H-force, blade flapping, the inflow
change that makes a rotor lose thrust as it translates -- are not modelled
here. They are real, and at 30 m/s they are not small; they are a separate
problem from body drag and are not smuggled into it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import frames

RHO = 1.225


class BodyDrag:
    """Quadratic drag, independently per body axis: F_i = -1/2 rho CdA_i |v_i| v_i."""

    def __init__(self, cda, rho: float = RHO) -> None:
        self.cda = np.asarray(cda, dtype=float)
        if self.cda.shape != (3,) or (self.cda < 0).any():
            raise ValueError("CdA needs three non-negative components")
        self.rho = rho

    @classmethod
    def estimate(cls, vehicle, cd: float = 1.05) -> "BodyDrag":
        """Projected areas of every body along each FLU axis, times a bluff Cd.

        Bounding-box areas for the frontal and side projections, the real
        footprint for the plan -- the same `planform_area()` the interference
        model uses, so a generated frame is not credited with the drag of its
        bounding box.
        """
        ax = ay = az = 0.0
        for b in vehicle.all_bodies():
            h = frames.NED_TO_FLU @ np.asarray(b.shape.half_extents().as_list())
            hx, hy, hz = np.abs(h)
            ax += 4.0 * hy * hz
            ay += 4.0 * hx * hz
            az += b.shape.planform_area()
        return cls(cd * np.array([ax, ay, az]))

    def force(self, v_air_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(force, moment) on the body in FLU axes, for air-relative velocity."""
        v = np.asarray(v_air_body, dtype=float)
        return -0.5 * self.rho * self.cda * np.abs(v) * v, np.zeros(3)


class AeroTable:
    """Body force and moment coefficients measured by CFD, against angle of attack.

    Angle of attack is the aircraft convention, alpha = atan2(w, u) with w the
    downward body velocity -- in FLU axes, atan2(-v_z, v_x). Positive is nose up
    into the wind, air striking the underside: a plane climbing, or a wing
    lifting. A multirotor flying forward pitches nose *down*, so the air
    strikes its top and alpha is negative -- a racing quad at speed lives
    around -40 to -60 degrees. One convention for both kinds of vehicle, so a
    table measured for either means the same thing.

    CX and CZ are along FLU +x (forward) and +z (up); Cm is about FLU +y,
    which points left, so positive Cm is nose DOWN. Coefficients are referenced
    to `area` and dynamic pressure, moments also to `length`, and interpolated
    linearly; outside the measured range the nearest value is held, and
    `extrapolated` counts how often, so it can be reported rather than hidden.
    """

    def __init__(self, alpha_deg, cx, cz, cm, *, area: float, length: float,
                 speeds=None, rho: float = RHO, source: str = "") -> None:
        self.alpha = np.radians(np.asarray(alpha_deg, dtype=float))
        order = np.argsort(self.alpha)
        self.alpha = self.alpha[order]
        self.cx = np.asarray(cx, float)[order]
        self.cz = np.asarray(cz, float)[order]
        self.cm = np.asarray(cm, float)[order]
        self.area, self.length, self.rho = float(area), float(length), rho
        self.speeds = list(speeds) if speeds is not None else []
        self.source = source
        self.extrapolated = 0

    @classmethod
    def load(cls, path: str | Path) -> "AeroTable":
        d = json.loads(Path(path).read_text())
        return cls(d["alpha_deg"], d["CX"], d["CZ"], d["Cm"], area=d["area"],
                   length=d["length"], speeds=d.get("speeds"),
                   source=str(path))

    def force(self, v_air_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(v_air_body, dtype=float)
        speed = float(np.linalg.norm(v))
        if speed < 1e-6:
            return np.zeros(3), np.zeros(3)
        alpha = math.atan2(-v[2], v[0])              # see class docstring
        if alpha < self.alpha[0] or alpha > self.alpha[-1]:
            self.extrapolated += 1
        q = 0.5 * self.rho * speed * speed
        cx = np.interp(alpha, self.alpha, self.cx)
        cz = np.interp(alpha, self.alpha, self.cz)
        cm = np.interp(alpha, self.alpha, self.cm)
        # Side force: not swept (beta = 0 only), so drag-like and symmetric.
        cy = -abs(cx) * (v[1] / speed)
        f = q * self.area * np.array([cx, cy, cz])
        m = np.array([0.0, q * self.area * self.length * cm, 0.0])
        return f, m
