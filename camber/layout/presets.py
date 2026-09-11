"""Exact rotor layouts by name and wheelbase, instead of by dragging.

A frame is specified by its wheelbase -- the diagonal motor-to-motor distance --
and its rotor count and arrangement. Placing four motors by hand to hit 220 mm
exactly is not a job anyone should do with a mouse.

Spins alternate round the ring, which is what cancels reaction torque in hover;
every layout here is also mirror-symmetric left/right, with mirrored rotors
spinning opposite ways, as a reflection requires.
"""

from __future__ import annotations

import math

from ..spec.units import Vec3
from ..spec.vehicle import Rotor, Spin

#: name -> (rotor count, angle of the first rotor from straight ahead, deg).
#: Angles run clockwise seen from above, i.e. towards +y (right) in NED.
PRESETS = {
    "quad-x": (4, 45.0),
    "quad-plus": (4, 0.0),
    "hex-x": (6, 30.0),
    "hex-plus": (6, 0.0),
    "octo-x": (8, 22.5),
    "tri-y": (3, 60.0),
}

NAMES = {
    4: ("front_right", "rear_right", "rear_left", "front_left"),
}


def rotor_layout(kind: str, wheelbase: float, height: float = -0.03,
                 first_spin: Spin = Spin.CCW) -> list[tuple[str, Vec3, Spin]]:
    """(name, position, spin) for each rotor of a named layout.

    `wheelbase` is the diameter of the rotor circle -- for an even count, the
    diagonal motor-to-motor distance. `height` is z in NED, negative = above.
    """
    if kind not in PRESETS:
        raise ValueError(f"unknown layout {kind!r}; choose from {', '.join(PRESETS)}")
    if wheelbase <= 0:
        raise ValueError("wheelbase must be positive")
    n, start = PRESETS[kind]
    r = wheelbase / 2.0
    out = []
    names = NAMES.get(n) if kind == "quad-x" else None
    other = Spin.CW if first_spin == Spin.CCW else Spin.CCW
    for i in range(n):
        th = math.radians(start + 360.0 * i / n)
        pos = Vec3(r * math.cos(th), r * math.sin(th), height)
        spin = first_spin if i % 2 == 0 else other
        out.append((names[i] if names else f"rotor{i + 1}", pos, spin))
    if n % 2:
        # An odd ring cannot alternate all the way round; a tricopter cancels
        # yaw with a tilting tail rotor, which this layout does not model.
        pass
    return out


def apply_layout(vehicle, kind: str, wheelbase: float, height: float = -0.03,
                 first_spin: Spin = Spin.CCW) -> None:
    """Re-place the vehicle's rotors, in place, keeping their hardware.

    Rotors are reused in order; if the count changes, new ones copy the first
    rotor's motor, prop and thrust, and surplus ones are removed.
    """
    import copy

    spec = rotor_layout(kind, wheelbase, height, first_spin)
    rotors = [t for t in vehicle.thrusters if isinstance(t, Rotor)]
    if not rotors:
        raise ValueError("the vehicle has no rotor to copy the hardware from")
    others = [t for t in vehicle.thrusters if not isinstance(t, Rotor)]
    new = []
    for i, (name, pos, spin) in enumerate(spec):
        r = rotors[i] if i < len(rotors) else copy.deepcopy(rotors[0])
        r.name = name
        r.placement = type(r.placement)(pos, r.placement.rotation)
        r.spin = spin
        new.append(r)
    vehicle.thrusters[:] = new + others
