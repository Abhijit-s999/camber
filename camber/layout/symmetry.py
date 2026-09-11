"""Mirror symmetry: finding twins, measuring how far off they are, fixing it.

The symptom this exists for: parts dragged into place by eye in a 3D view look
symmetric and are not. A vehicle a few millimetres off symmetric flies -- the
mixer trims it out -- but it hovers on an uneven thrust split, spends control
authority holding itself level, and the structure the optimiser grows for it
is lopsided for no reason. Measuring beats eyeballing.

Planes are named by the two axes they contain, in the NED body frame:

  "xz"  the centreline plane: left/right mirror (y -> -y)
  "yz"  the lateral plane:    front/back mirror (x -> -x)

**Spin is checked by balance, not by pairs.** A reflection reverses
handedness, so the strict mirror image of a clockwise rotor turns
counter-clockwise -- and on an X-quad that is the rule. But a plus-quad's left
and right arms *both* spin clockwise, correctly, with front and rear both the
other way. "Mirrored pairs spin opposite" is sufficient for yaw balance and not
necessary, and a first version that enforced it flagged the standard plus-quad
as a mistake. What hover actually needs is for the reaction torques to cancel
at equal thrust: as many rotors each way, torque coefficient for torque
coefficient. That is what is reported, and what `symmetrise` repairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..spec.units import Vec3
from ..spec.vehicle import Rotor, Spin

FLIP = {"xz": np.array([1.0, -1.0, 1.0]), "yz": np.array([-1.0, 1.0, 1.0])}
AXIS = {"xz": 1, "yz": 0}          # the coordinate the plane zeroes


def _check(plane: str) -> None:
    if plane not in FLIP:
        raise ValueError(f"plane must be 'xz' (left/right) or 'yz' "
                         f"(front/back), got {plane!r}")


def mirror(p, plane: str) -> np.ndarray:
    _check(plane)
    return np.asarray(p.as_list() if hasattr(p, "as_list") else p, float) * FLIP[plane]


def _parts(vehicle):
    """Everything placeable, as (name, part). Frame members are the structure,
    not payload, and are left alone."""
    return [(b.name, b) for b in vehicle.bodies] + \
           [(t.name, t) for t in vehicle.thrusters]


def _kind(part) -> tuple:
    """What has to match for two parts to be twins: type and size and mass."""
    if isinstance(part, Rotor):
        return ("rotor", round(part.radius, 4))
    size = tuple(np.round(np.abs(part.shape.half_extents().as_list()), 4))
    return (type(part).__name__, size, round(part.mass, 4))


@dataclass
class Pairing:
    plane: str
    pairs: list = field(default_factory=list)       # (a, b): b is a's twin
    centred: list = field(default_factory=list)     # on the plane
    unmatched: list = field(default_factory=list)   # no twin, not on the plane

    def twin(self, name: str) -> str | None:
        for a, b in self.pairs:
            if name == a:
                return b
            if name == b:
                return a
        return None


def find_pairs(vehicle, plane: str = "xz", *, on_plane: float = 0.004,
               reach: float = 0.35) -> Pairing:
    """Match every part with its mirror twin.

    A part within `on_plane` metres of the plane is taken to sit *on* it -- a
    battery on the centreline. Otherwise it is paired, greedily by mismatch,
    with the same kind of part (same type, size and mass) nearest its mirror
    image, provided that is within `reach` of the part's own distance from the
    plane -- close enough that it is plausibly meant as its twin rather than a
    different part of the same size.
    """
    _check(plane)
    ax = AXIS[plane]
    parts = _parts(vehicle)
    out = Pairing(plane)
    loose = []
    for name, part in parts:
        p = np.asarray(part.position.as_list())
        if abs(p[ax]) <= on_plane:
            out.centred.append(name)
        else:
            loose.append((name, part, p))
    candidates = []
    for i, (na, pa_, pa) in enumerate(loose):
        for nb, pb_, pb in loose[i + 1:]:
            if _kind(pa_) != _kind(pb_):
                continue
            if np.sign(pa[ax]) == np.sign(pb[ax]):
                continue                           # same side: not twins
            miss = float(np.linalg.norm(mirror(pa, plane) - pb))
            if miss <= reach * max(abs(pa[ax]), abs(pb[ax]), 0.01):
                candidates.append((miss, na, nb))
    used = set()
    for miss, na, nb in sorted(candidates):
        if na in used or nb in used:
            continue
        out.pairs.append((na, nb))
        used |= {na, nb}
    out.unmatched = [n for n, _, _ in loose if n not in used]
    return out


def symmetry_report(vehicle, plane: str = "xz", pairing: Pairing | None = None
                    ) -> dict:
    """How far from symmetric, in metres, part by part.

    For a pair, the distance between one part and the mirror image of its twin;
    for a centred part, its distance from the plane. `yaw_balance` is the net
    reaction torque at equal thrust as a fraction of the total -- zero when the
    spins cancel, whatever the pattern.
    """
    pairing = pairing or find_pairs(vehicle, plane)
    pos = {n: np.asarray(p.position.as_list()) for n, p in _parts(vehicle)}
    part = dict(_parts(vehicle))
    rows = []
    for a, b in pairing.pairs:
        miss = float(np.linalg.norm(mirror(pos[a], plane) - pos[b]))
        mirrored = None
        if isinstance(part[a], Rotor) and isinstance(part[b], Rotor):
            mirrored = part[a].spin != part[b].spin
        rows.append({"a": a, "b": b, "error": miss, "mirrored_spin": mirrored})
    ax = AXIS[plane]
    for n in pairing.centred:
        rows.append({"a": n, "b": None, "error": abs(float(pos[n][ax])),
                     "mirrored_spin": None})
    worst = max((r["error"] for r in rows), default=0.0)
    return {"plane": plane, "rows": rows, "max_error": worst,
            "yaw_balance": yaw_balance(vehicle),
            "unmatched": list(pairing.unmatched)}


def yaw_balance(vehicle) -> float:
    """Net reaction torque at equal thrust, over the total: 0 is balanced."""
    rotors = [t for t in vehicle.thrusters if isinstance(t, Rotor)]
    total = sum(t.torque_coefficient for t in rotors)
    if total == 0:
        return 0.0
    return float(sum(t.spin.value * t.torque_coefficient for t in rotors) / total)


def symmetrise(vehicle, plane: str = "xz", pairing: Pairing | None = None,
               *, fix_spin: bool = True) -> list[str]:
    """Snap the vehicle to exact symmetry, in place. Returns what changed.

    Each pair moves to the *average* of where it is and where its twin says it
    should be, so neither side is privileged and a pair that was 4 mm apart
    each gives 2 mm. Centred parts are projected onto the plane. If `fix_spin`
    and the spins do not balance, rotor pairs are given opposite spins -- the
    mirror rule, which balances every pair; a pattern that already balances,
    such as a plus-quad's, is left alone.
    """
    pairing = pairing or find_pairs(vehicle, plane)
    part = dict(_parts(vehicle))
    changed = []

    def place(name, p):
        obj = part[name]
        obj.placement = type(obj.placement)(Vec3(*map(float, p)),
                                            obj.placement.rotation)

    for a, b in pairing.pairs:
        pa = np.asarray(part[a].position.as_list())
        pb = np.asarray(part[b].position.as_list())
        target = 0.5 * (pa + mirror(pb, plane))
        if np.linalg.norm(target - pa) > 1e-12 or \
                np.linalg.norm(mirror(target, plane) - pb) > 1e-12:
            changed.append(f"{a}/{b}")
        place(a, target)
        place(b, mirror(target, plane))
    if fix_spin and abs(yaw_balance(vehicle)) > 1e-12:
        for a, b in pairing.pairs:
            if isinstance(part[a], Rotor) and isinstance(part[b], Rotor) \
                    and part[a].spin == part[b].spin:
                part[b].spin = Spin.CW if part[a].spin == Spin.CCW else Spin.CCW
                changed.append(f"{b} spin")
    ax = AXIS[plane]
    for n in pairing.centred:
        p = np.asarray(part[n].position.as_list())
        if p[ax] != 0.0:
            p[ax] = 0.0
            place(n, p)
            changed.append(n)
    return changed


def move_with_twin(vehicle, name: str, position, planes=("xz",)) -> list[str]:
    """Move one part and keep its twins mirrored: the linked drag.

    With two planes (4-way symmetry) a part has up to three twins -- its
    left/right image, front/back image, and the image through both -- and all
    of them move. A part on a plane stays on it.
    """
    part = dict(_parts(vehicle))
    p = np.asarray(position, dtype=float)
    moved = [name]
    pairings = [find_pairs(vehicle, pl) for pl in planes]
    for pl, pr in zip(planes, pairings):
        if name in pr.centred:
            p[AXIS[pl]] = 0.0

    def place(n, q):
        o = part[n]
        o.placement = type(o.placement)(Vec3(*map(float, q)), o.placement.rotation)

    place(name, p)
    for pl, pr in zip(planes, pairings):
        t = pr.twin(name)
        if t:
            place(t, mirror(p, pl))
            moved.append(t)
    if len(planes) == 2:                            # the diagonal image
        t1 = pairings[0].twin(name)
        t2 = pairings[1].twin(t1) if t1 else None
        if t2 and t2 not in moved:
            place(t2, mirror(mirror(p, planes[0]), planes[1]))
            moved.append(t2)
    return moved
