"""Measurements a builder actually checks before cutting carbon.

All in metres, NED body frame; the app shows millimetres.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from ..spec.vehicle import Rotor


def distance(vehicle, a: str, b: str) -> dict:
    """Centre-to-centre distance between two named parts, and its components."""
    pa = np.asarray(_find(vehicle, a).position.as_list())
    pb = np.asarray(_find(vehicle, b).position.as_list())
    d = pb - pa
    return {"a": a, "b": b, "distance": float(np.linalg.norm(d)),
            "dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2]),
            "plan": float(np.hypot(d[0], d[1]))}


def _find(vehicle, name):
    for p in [*vehicle.all_bodies(), *vehicle.thrusters]:
        if p.name == name:
            return p
    raise KeyError(f"no part named {name!r}")


def wheelbase(vehicle) -> float:
    """Diagonal motor-to-motor distance -- the number a frame is sold by
    ("5-inch, 220 mm"). The largest plan distance between any two rotors."""
    rotors = [t for t in vehicle.thrusters if isinstance(t, Rotor)]
    best = 0.0
    for a, b in itertools.combinations(rotors, 2):
        best = max(best, math.hypot(a.position.x - b.position.x,
                                    a.position.y - b.position.y))
    return best


def _box_gap(rotor, body) -> tuple[float, float]:
    """(plan gap, vertical gap) between a prop disc and a body's box.

    Plan gap: distance from the disc edge to the nearest point of the body's
    footprint, negative when the disc overhangs it. Vertical gap: distance
    between the disc plane and the nearer face of the box, negative when the
    disc plane passes through it. The two only collide if both are negative.
    """
    h = np.abs(body.shape.half_extents().as_list())
    c = np.asarray(body.position.as_list())
    r = np.asarray(rotor.position.as_list())
    dx = max(abs(r[0] - c[0]) - h[0], 0.0)
    dy = max(abs(r[1] - c[1]) - h[1], 0.0)
    plan = math.hypot(dx, dy) - rotor.radius
    vertical = abs(r[2] - c[2]) - h[2]
    return plan, vertical


def clearances(vehicle, warn: float = 0.004) -> dict:
    """Prop tip to prop tip, and prop to body. The first thing that goes wrong
    when parts are placed by eye is a prop that clips an arm or its neighbour.

    A pair is a *conflict* when the gap is below `warn` in plan and the discs
    (or the disc and the box) overlap vertically too. Rotors at different
    heights -- a coaxial or stacked layout -- can overlap in plan legitimately,
    so vertical separation is part of the verdict, not ignored.
    """
    rotors = [t for t in vehicle.thrusters if isinstance(t, Rotor)]
    tips, bodies = [], []
    for a, b in itertools.combinations(rotors, 2):
        centre = math.hypot(a.position.x - b.position.x, a.position.y - b.position.y)
        gap = centre - a.radius - b.radius
        dz = abs(a.position.z - b.position.z)
        tips.append({"a": a.name, "b": b.name, "gap": gap, "height_apart": dz,
                     "conflict": gap < warn and dz < 0.01})
    for r in rotors:
        for body in vehicle.all_bodies():
            plan, vertical = _box_gap(r, body)
            bodies.append({"rotor": r.name, "body": body.name, "plan_gap": plan,
                           "vertical_gap": vertical,
                           "conflict": plan < warn and vertical < warn})
    return {"tips": tips, "bodies": bodies,
            "min_tip_gap": min((t["gap"] for t in tips), default=float("inf")),
            "conflicts": [t for t in tips + bodies if t["conflict"]]}


def balance(vehicle) -> dict:
    """Where the weight is against where the thrust is, and what that costs.

    The thrust centroid is the thrust-weighted average of the rotor positions
    at equal thrust; the CG offset from it is what every rotor has to make up
    with an uneven split. The split itself comes from `solve_hover`, so the
    number shown is the one the optimiser and simulator use.
    """
    from ..control import G, controllability, solve_hover

    mp = vehicle.mass_properties()
    cg = np.asarray(mp.cg.as_list())
    rotors = [t for t in vehicle.thrusters if isinstance(t, Rotor)]
    out = {"mass": mp.mass, "cg": cg.tolist()}
    if not rotors:
        return out
    centroid = np.mean([t.position.as_list() for t in rotors], axis=0)
    offset = cg[:2] - centroid[:2]
    out.update({"thrust_centroid": centroid.tolist(),
                "cg_offset": offset.tolist(),
                "cg_offset_norm": float(np.linalg.norm(offset))})
    try:
        total = sum(t.max_thrust() for t in rotors)
        out["thrust_to_weight"] = total / (mp.mass * G)
    except ValueError:
        out["thrust_to_weight"] = None
    h = solve_hover(vehicle)
    if h.feasible:
        split = np.asarray(h.thrusts) / float(np.sum(h.thrusts))
        out["hover_split"] = {t.name: float(s) for t, s in zip(rotors, split)}
        out["hover_imbalance"] = float(split.max() / split.min() - 1.0)
        out["hover_utilisation"] = float(np.max(h.utilisation))
    else:
        out["hover_split"] = None
        out["hover_reason"] = h.reason
    c = controllability(vehicle)
    out["controllable"] = list(c.controllable_axes)
    out["control_margin"] = c.margin
    return out
