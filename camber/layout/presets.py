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

#: name -> (rotor count, angle of the first rotor from straight ahead deg,
#: cant deg). Angles run clockwise seen from above, towards +y (right) in NED.
#:
#: Cant is the tilt of each motor away from vertical, tangent to the ring, with
#: its direction tied to the rotor's spin. Even rings leave it at zero: their
#: rotors alternate spin all the way round, so paired reaction torques give
#: yaw. An odd ring cannot alternate evenly -- a five-rotor ring has three of
#: one sense and two of the other -- so its reaction torques do not cancel in
#: hover and it has almost no clean yaw. Canting the motors makes yaw out of
#: differential *thrust* instead, which is how real penta-copters fly.
PRESETS = {
    "quad-x": (4, 45.0, 0.0),
    "quad-plus": (4, 0.0, 0.0),
    "penta-x": (5, 0.0, 10.0),
    "hex-x": (6, 30.0, 0.0),
    "hex-plus": (6, 0.0, 0.0),
    "hepta-x": (7, 0.0, 8.0),
    "octo-x": (8, 22.5, 0.0),
    # No tricopter: three fixed rotors give only three thrust degrees of
    # freedom and flight needs four (thrust, roll, pitch, yaw). A real one adds
    # a tilting tail servo, and cant alone cannot make the fourth axis. It
    # returns with thrust vectoring (SIM-07, parked).
}

NAMES = {
    4: ("front_right", "rear_right", "rear_left", "front_left"),
}


#: Straight up in NED (z points down), the default thrust direction.
UP = Vec3(0.0, 0.0, -1.0)


def rotor_layout(kind: str, wheelbase: float, height: float = -0.03,
                 first_spin: Spin = Spin.CCW
                 ) -> list[tuple[str, Vec3, Spin, Vec3]]:
    """(name, position, spin, thrust_axis) for each rotor of a named layout.

    `wheelbase` is the diameter of the rotor circle -- for an even count, the
    diagonal motor-to-motor distance. `height` is z in NED, negative = above.
    The thrust axis is straight up for an even ring, and canted tangentially
    for an odd one (see PRESETS) so it can command yaw.
    """
    if kind not in PRESETS:
        raise ValueError(f"unknown layout {kind!r}; choose from {', '.join(PRESETS)}")
    if wheelbase <= 0:
        raise ValueError("wheelbase must be positive")
    n, start, cant = PRESETS[kind]
    r = wheelbase / 2.0
    out = []
    names = NAMES.get(n) if kind == "quad-x" else None
    other = Spin.CW if first_spin == Spin.CCW else Spin.CCW
    c = math.radians(cant)
    for i in range(n):
        th = math.radians(start + 360.0 * i / n)
        pos = Vec3(r * math.cos(th), r * math.sin(th), height)
        spin = first_spin if i % 2 == 0 else other
        # Tangent to the ring, in the direction the rotor's own wash swirls, so
        # its thrust-vector yaw and its reaction-torque yaw pull the same way.
        tangent = Vec3(-math.sin(th), math.cos(th), 0.0) * float(spin.value)
        axis = (UP + tangent * math.tan(c)).unit() if cant else UP
        out.append((names[i] if names else f"rotor{i + 1}", pos, spin, axis))
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
    for i, (name, pos, spin, axis) in enumerate(spec):
        r = rotors[i] if i < len(rotors) else copy.deepcopy(rotors[0])
        r.name = name
        r.thrust_axis = axis
        r.placement = type(r.placement)(pos, r.placement.rotation)
        r.spin = spin
        new.append(r)
    vehicle.thrusters[:] = new + others
