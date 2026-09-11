"""Build a CAD assembly from a spec, and write it out for a machine shop.

The kernel earns its place here and nowhere else. Mass properties are analytic
(see camber/spec/shapes.py) because closed forms are exact and fast; B-rep
solids are needed for STEP output, booleans and imported geometry, and for
nothing else. Keeping the boundary sharp is what lets the optimiser run
millions of evaluations without OCCT in the loop.

**Units.** Camber is SI, so metres; STEP and every CAM package expect
millimetres. The conversion happens here, once, at the boundary, and is the
only place in the codebase where a length is not in metres.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..spec.shapes import Box, Cylinder, PointMass, Shape, Sphere
from ..spec.vehicle import Body, Vehicle

#: The one unit conversion in the product. See the module docstring.
M_TO_MM = 1000.0


class CadUnavailable(RuntimeError):
    """build123d is not installed.

    Not fatal, and deliberately not a hard dependency: everything except CAD
    export works without it (AI-07), and OCCT is a ~200 MB wheel that no one
    running an optimisation sweep on a headless box needs.
    """


def _b3():
    try:
        import build123d
    except ModuleNotFoundError as exc:
        raise CadUnavailable(
            "CAD export needs build123d: uv pip install build123d\n"
            "(Everything else in Camber works without it.)"
        ) from exc
    return build123d


def _solid(shape: Shape, b3):
    """One primitive, centred on its own origin, in millimetres."""
    if isinstance(shape, Box):
        return b3.Box(shape.lx * M_TO_MM, shape.ly * M_TO_MM, shape.lz * M_TO_MM)
    if isinstance(shape, Cylinder):
        return b3.Cylinder(shape.radius * M_TO_MM, shape.length * M_TO_MM)
    if isinstance(shape, Sphere):
        return b3.Sphere(shape.radius * M_TO_MM)
    if isinstance(shape, PointMass):
        return None
    raise TypeError(f"no CAD representation for {type(shape).__name__}")


def _placed(body: Body, b3):
    solid = _solid(body.shape, b3)
    if solid is None:
        return None
    axis, angle = body.placement.rotation.as_axis_angle()
    location = b3.Location(
        tuple(v * M_TO_MM for v in body.position),
        tuple(axis),
        math.degrees(angle),
    )
    return location * solid


def build(vehicle: Vehicle):
    """A build123d Compound of every solid body, in millimetres.

    Thrusters are not included: a propeller disc is a swept volume, not a part,
    and putting one in a file destined for a CNC machine invites someone to try
    to cut it.
    """
    b3 = _b3()
    solids = [s for s in (_placed(b, b3) for b in vehicle.all_bodies())
              if s is not None]
    if not solids:
        raise ValueError(
            f"{vehicle.name!r} has no bodies with extent; nothing to export "
            f"(point masses have no geometry)"
        )
    return b3.Compound(children=solids)


def export(vehicle: Vehicle, path: str | Path, *, metres: bool = False) -> Path:
    """Write STEP for CNC, or STL for printing, chosen by file extension.

    `metres=True` emits the geometry at true SI scale instead of millimetres.
    CAM wants millimetres; a CFD case is built in metres, and handing it the
    millimetre file makes a 180 mm airframe 180 m across, which silently
    swallows the whole domain. snappyHexMesh then reports success having
    refined nothing, and the forces function object finds no patch to
    integrate. See DECISIONS.md GEOM-04 and CFD-02.
    """
    b3 = _b3()
    path = Path(path)
    assembly = build(vehicle)
    if metres:
        assembly = assembly.scale(1.0 / M_TO_MM)
    else:
        # Manufacturing output goes Z-up; the model is NED, z down. A half-turn
        # about x, not a mirror in z -- a mirror would machine the airframe's
        # reflection. The CFD path (`metres=True`) stays in NED, because the
        # CFD case is built in NED. See topology.ned_to_z_up.
        assembly = assembly.rotate(b3.Axis.X, 180)

    suffix = path.suffix.lower()
    if suffix == ".step" or suffix == ".stp":
        # Unit.MM matches the millimetre geometry built above; a mismatch here
        # produces a part that is correct in shape and wrong by 1000x in size,
        # which is a mistake a machinist discovers expensively.
        b3.export_step(assembly, str(path), unit=b3.Unit.MM)
    elif suffix == ".stl":
        b3.export_stl(assembly, str(path))
    else:
        raise ValueError(
            f"unsupported export format {suffix!r}; use .step (CNC, exact "
            f"B-rep) or .stl (printing, tessellated)"
        )
    return path


def bounding_box_mm(vehicle: Vehicle) -> tuple[float, float, float]:
    """Overall envelope, for stock sizing and build-volume checks."""
    bb = build(vehicle).bounding_box()
    return (bb.size.X, bb.size.Y, bb.size.Z)
