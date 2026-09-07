"""The vehicle specification: Camber's single source of truth.

Everything downstream derives from these objects -- the CAD solid, the mesh for
CFD, the mass properties, the camera occlusion tests, the structural model and
the control allocation matrix. That is why the schema is phase one and the
physics is phase two: a private copy of the geometry growing inside each module
is the failure mode this package exists to prevent.

See docs/DECISIONS.md ARCH-01 and ARCH-02.
"""

from .vehicle import (
    Battery,
    Body,
    Camera,
    Frame,
    Mission,
    Placement,
    Rotor,
    Thruster,
    Vehicle,
)
from .constraints import (
    CGBox,
    Constraint,
    Fixed,
    MinDistance,
    Relative,
    Symmetry,
    ViewCone,
)
from .units import Vec3

__all__ = [
    "Battery", "Body", "Camera", "Frame", "Mission", "Placement", "Rotor",
    "Thruster", "Vehicle", "CGBox", "Constraint", "Fixed", "MinDistance",
    "Relative", "Symmetry", "ViewCone", "Vec3",
]
