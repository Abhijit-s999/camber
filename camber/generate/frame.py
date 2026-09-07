"""Generate an airframe from rotor positions and the payloads it must carry.

Up to here Camber has chosen components and placed them on a frame the user
supplied. This is the first module that produces structure of its own: a centre
plate sized to hold the payloads and arms out to each rotor, each arm sized by
the load cases in camber/structure.py and checked against the chosen process in
camber/manufacture.py.

The design is deliberately conservative in what it invents. An arm is a
straight tube from the plate edge to the rotor hub, because that is what almost
every real multirotor is and because a shape the structural model can size
exactly is worth more than a shape it can only approximate. Topology
optimisation would produce something lighter and something no closed-form beam
model can check -- which would mean trusting a generated airframe on a
generated analysis, with nothing measured anywhere in the chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..manufacture import Feature, Process
from ..spec.shapes import Box, Cylinder
from ..spec.units import Vec3
from ..spec.vehicle import Body, Placement, Rotor, Vehicle
from ..structure import (
    Material, MATERIALS, Section, analyse_arm, arm_mass, load_cases, size_arm,
)


@dataclass
class FrameSpec:
    """What the frame must satisfy, as opposed to what it looks like."""

    material: str = "cf_tube"
    plate_material: str = "cf_tube"
    #: Clearance from the payload envelope to the plate edge.
    plate_margin: float = 0.008
    plate_thickness: float = 0.002
    #: Landing severity the arms must survive, in g.
    landing_g: float = 6.0
    #: Which process the result must be makeable by (MFG-01).
    process: str = "cnc-3axis"
    section_kind: str = "tube"


@dataclass
class GeneratedFrame:
    plate: Body
    arms: list[Body]
    section: Section
    arm_length: float
    governing_case: str
    margin: float
    total_mass: float
    features: list[Feature] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def members(self) -> list[Body]:
        return [self.plate, *self.arms]


def _payload_envelope(vehicle: Vehicle) -> tuple[float, float]:
    """Half-extents in x and y that the centre plate must cover."""
    hx = hy = 0.0
    for b in vehicle.bodies:
        h = b.shape.half_extents()
        hx = max(hx, abs(b.position.x) + h.x)
        hy = max(hy, abs(b.position.y) + h.y)
    return hx, hy


def generate_frame(vehicle: Vehicle, spec: FrameSpec | None = None,
                   process: Process | None = None) -> GeneratedFrame:
    """Synthesise a plate and arms that carry this vehicle's rotors.

    Raises `ValueError` when no stock section survives the load cases, rather
    than returning the strongest available and hoping. A frame that cannot be
    built from purchasable material is a real answer, and quietly returning an
    under-strength one is how a generated design gets flown.
    """
    from ..manufacture import get_process

    spec = spec or FrameSpec()
    process = process or get_process(spec.process)
    material = MATERIALS[spec.material]
    plate_material = MATERIALS[spec.plate_material]

    rotors = [t for t in vehicle.thrusters if isinstance(t, Rotor)]
    if not rotors:
        raise ValueError("cannot generate a frame with no rotors to carry")

    hx, hy = _payload_envelope(vehicle)
    plate_x = 2.0 * (hx + spec.plate_margin)
    plate_y = 2.0 * (hy + spec.plate_margin)

    # Arms run from the plate edge to the rotor hub. Using the reach from the
    # origin would overstate the cantilever and oversize every arm.
    reaches = []
    for r in rotors:
        radial = math.hypot(r.position.x, r.position.y)
        if radial <= 1e-9:
            raise ValueError(f"rotor {r.name!r} sits on the axis; no arm to size")
        ux, uy = r.position.x / radial, r.position.y / radial
        edge = min(plate_x / 2 / abs(ux) if abs(ux) > 1e-9 else math.inf,
                   plate_y / 2 / abs(uy) if abs(uy) > 1e-9 else math.inf)
        reaches.append(max(radial - edge, 0.02))
    arm_length = max(reaches)

    mass = vehicle.mass_properties().mass
    thrusts = []
    for r in rotors:
        try:
            thrusts.append(r.max_thrust())
        except ValueError:
            thrusts.append(mass * 9.80665 / len(rotors) * 2.5)
    max_thrust = max(thrusts)

    loads = load_cases(mass * 9.80665 / len(rotors), max_thrust, mass,
                       len(rotors), spec.landing_g)
    sized = size_arm(arm_length, material, loads, kind=spec.section_kind)
    if sized is None:
        raise ValueError(
            f"no stock {spec.section_kind} in {material.name} survives a "
            f"{arm_length*1000:.0f} mm arm at {max_thrust:.1f} N. Shorten the "
            f"arms, choose a stronger material, or accept a lower landing g."
        )
    section, per_arm = sized
    worst = min((analyse_arm(arm_length, section, material, l) for l in loads),
                key=lambda r: r.margin)

    plate = Body(
        "plate",
        mass=plate_x * plate_y * spec.plate_thickness * plate_material.density,
        shape=Box(plate_x, plate_y, spec.plate_thickness),
        placement=Placement(Vec3(0.0, 0.0, 0.0)),
    )

    arms: list[Body] = []
    for r in rotors:
        radial = math.hypot(r.position.x, r.position.y)
        midpoint = Vec3(r.position.x * (1 - arm_length / (2 * radial)),
                        r.position.y * (1 - arm_length / (2 * radial)),
                        r.position.z / 2.0)
        arms.append(Body(
            f"arm_{r.name}", mass=per_arm,
            shape=Cylinder(radius=section.a / 2, length=arm_length),
            placement=Placement(midpoint),
        ))

    features = [
        Feature("plate", "wall", size=spec.plate_thickness),
        Feature("arm clamp corner", "corner", radius=0.0025),
        Feature("battery pocket", "pocket", size=min(plate_x, plate_y) * 0.6,
                depth=spec.plate_thickness),
    ]
    notes = [f"{v.rule}: {v.detail}" for v in process.check_all(features)]

    return GeneratedFrame(
        plate=plate, arms=arms, section=section, arm_length=arm_length,
        governing_case=worst.case, margin=worst.margin,
        total_mass=plate.mass + per_arm * len(arms),
        features=features, notes=notes,
    )
