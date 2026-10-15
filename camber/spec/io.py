"""Read and write vehicle specifications.

TOML and JSON only, both from the standard library. A spec format is the API
users actually programme against, so it should not drag a parser dependency
behind it -- and `AI-07` says the product must work with nothing installed.

Every unknown key is an error rather than a shrug. A spec with `mas = 0.25`
should stop, not silently build a vehicle with a default mass and produce a
confidently wrong endurance figure.
"""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path
from typing import Any

from .constraints import (
    CGBox, Constraint, Fixed, MinDistance, Relative, Symmetry, ViewCone,
)
from .rotation import Rotation
from .shapes import Box, Cylinder, PointMass, Shape, Sphere, Tabulated
from .units import DEG, Vec3
from .vehicle import (
    Battery, Body, Camera, Frame, Mission, Objective, Placement, Rotor, Spin,
    Thruster, Vehicle,
)


class SpecError(ValueError):
    """The specification is malformed. The message says where."""


def _require(d: dict, key: str, where: str):
    if key not in d:
        raise SpecError(f"{where}: missing required key {key!r}")
    return d[key]


def _reject_unknown(d: dict, known: set[str], where: str) -> None:
    extra = set(d) - known
    if extra:
        raise SpecError(
            f"{where}: unknown key(s) {sorted(extra)}. Known keys are "
            f"{sorted(known)}. (A typo here would otherwise be silently "
            f"replaced by a default.)"
        )


def _shape(d: dict | None, where: str) -> Shape:
    if d is None:
        return PointMass()
    kind = _require(d, "kind", where)
    rest = {k: v for k, v in d.items() if k != "kind"}
    try:
        if kind == "box":
            _reject_unknown(rest, {"lx", "ly", "lz"}, f"{where}.shape")
            return Box(**rest)
        if kind == "cylinder":
            _reject_unknown(rest, {"radius", "length"}, f"{where}.shape")
            return Cylinder(**rest)
        if kind == "sphere":
            _reject_unknown(rest, {"radius"}, f"{where}.shape")
            return Sphere(**rest)
        if kind == "point":
            return PointMass()
        if kind == "tabulated":
            _reject_unknown(rest, {"volume_m3", "inertia_tensor", "extents",
                                   "planform", "source"}, f"{where}.shape")
            return Tabulated(**rest)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{where}.shape ({kind}): {exc}") from exc
    raise SpecError(f"{where}.shape: unknown kind {kind!r}; "
                    f"expected box, cylinder, sphere, point or tabulated")


def _placement(d: dict, where: str) -> Placement:
    position = Vec3.parse(d.get("position", [0.0, 0.0, 0.0]))
    rot = d.get("rotation_deg")
    if rot is None:
        rotation = Rotation.identity()
    else:
        r = Vec3.parse(rot)
        rotation = Rotation.from_euler(r.x * DEG, r.y * DEG, r.z * DEG)
    return Placement(position, rotation)


_BODY_COMMON = {"name", "type", "mass", "shape", "position", "rotation_deg"}


_BODY_EXTRA = {
    "body": set(),
    "battery": {"capacity_ah", "nominal_voltage", "max_c_rate"},
    "camera": {"view_direction", "horizontal_fov_deg"},
}


def _body(d: dict, index: int) -> Body:
    where = f"bodies[{index}]"
    kind = d.get("type", "body")
    if kind not in _BODY_EXTRA:
        raise SpecError(f"{where}: unknown body type {kind!r}; "
                        f"expected body, battery or camera")
    _reject_unknown(d, _BODY_COMMON | _BODY_EXTRA[kind], where)

    name = _require(d, "name", where)
    mass = float(_require(d, "mass", where))
    shape = _shape(d.get("shape"), where)
    placement = _placement(d, where)

    try:
        if kind == "body":
            return Body(name, mass, shape, placement)
        if kind == "battery":
            return Battery(
                name, mass, shape, placement,
                capacity_ah=float(_require(d, "capacity_ah", where)),
                nominal_voltage=float(_require(d, "nominal_voltage", where)),
                max_c_rate=float(d.get("max_c_rate", 0.0)),
            )
        if kind == "camera":
            return Camera(
                name, mass, shape, placement,
                view_direction=Vec3.parse(d.get("view_direction", [1.0, 0.0, 0.0])),
                horizontal_fov=float(d.get("horizontal_fov_deg", 0.0)) * DEG,
            )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SpecError):
            raise
        raise SpecError(f"{where}: {exc}") from exc
    raise AssertionError(f"unreachable: body kind {kind!r}")


def _thruster(d: dict, index: int) -> Thruster:
    where = f"thrusters[{index}]"
    kind = d.get("type", "rotor")
    if kind != "rotor":
        raise SpecError(
            f"{where}: thruster type {kind!r} is not implemented yet. "
            f"v1 validates on electric multirotor (SCOPE-01); rocket, edf and "
            f"control_surface come later."
        )
    _reject_unknown(d, {
        "name", "type", "mass", "position", "rotation_deg", "radius", "pitch",
        "blades", "ducted", "spin", "motor_sku", "prop_sku", "static_thrust_n",
        "torque_coefficient",
    }, where)
    spin_raw = str(d.get("spin", "ccw")).lower()
    if spin_raw not in ("cw", "ccw"):
        raise SpecError(f"{where}.spin: expected 'cw' or 'ccw', got {spin_raw!r}")

    try:
        return Rotor(
            name=_require(d, "name", where),
            mass=float(_require(d, "mass", where)),
            placement=_placement(d, where),
            radius=float(_require(d, "radius", where)),
            pitch=float(d.get("pitch", 0.0)),
            blades=int(d.get("blades", 2)),
            ducted=bool(d.get("ducted", False)),
            spin=Spin.CW if spin_raw == "cw" else Spin.CCW,
            motor_sku=d.get("motor_sku"),
            prop_sku=d.get("prop_sku"),
            static_thrust_n=(None if d.get("static_thrust_n") is None
                             else float(d["static_thrust_n"])),
            torque_coefficient=float(d.get("torque_coefficient", 0.016)),
        )
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{where}: {exc}") from exc


def _constraint(d: dict, index: int) -> Constraint:
    where = f"constraints[{index}]"
    kind = _require(d, "type", where)
    try:
        if kind == "fixed":
            return Fixed(d["body"], Vec3.parse(d["position"]))
        if kind == "relative":
            return Relative(d["a"], d["b"], Vec3.parse(d["offset"]))
        if kind == "min_distance":
            return MinDistance(d["a"], d["b"], float(d["distance"]))
        if kind == "cg_box":
            return CGBox(Vec3.parse(d["lower"]), Vec3.parse(d["upper"]))
        if kind == "symmetry":
            return Symmetry(d["a"], d["b"], d.get("plane", "xz"))
        if kind == "view_cone":
            return ViewCone(
                d["camera"], float(d["half_angle_deg"]) * DEG,
                tuple(d.get("obstructed_by", [])),
            )
    except KeyError as exc:
        raise SpecError(f"{where} ({kind}): missing key {exc}") from exc
    raise SpecError(f"{where}: unknown constraint type {kind!r}")


def from_dict(data: dict[str, Any]) -> Vehicle:
    _reject_unknown(data, {
        "name", "mission", "bodies", "thrusters", "frame", "constraints",
    }, "<root>")

    m = data.get("mission", {})
    _reject_unknown(m, {
        "objective", "target_range", "min_endurance", "payload_mass",
    }, "mission")
    obj_raw = m.get("objective", "max_endurance")
    try:
        objective = Objective(obj_raw)
    except ValueError as exc:
        raise SpecError(
            f"mission.objective: unknown objective {obj_raw!r}; expected one of "
            f"{[o.value for o in Objective]}"
        ) from exc

    vehicle = Vehicle(
        name=_require(data, "name", "<root>"),
        bodies=[_body(b, i) for i, b in enumerate(data.get("bodies", []))],
        thrusters=[_thruster(t, i) for i, t in enumerate(data.get("thrusters", []))],
        frame=Frame([_body(b, i) for i, b in enumerate(data.get("frame", []))]),
        constraints=[_constraint(c, i)
                     for i, c in enumerate(data.get("constraints", []))],
        mission=Mission(
            objective=objective,
            target_range=m.get("target_range"),
            min_endurance=m.get("min_endurance"),
            payload_mass=float(m.get("payload_mass", 0.0)),
        ),
    )
    _check_references(vehicle)
    return vehicle


def _check_references(v: Vehicle) -> None:
    """Every name a constraint mentions must exist.

    Caught here rather than at first evaluation, because a typo in a constraint
    would otherwise surface as a KeyError from deep inside the optimiser after
    an hour of searching.
    """
    known = {b.name for b in v.all_bodies()} | {t.name for t in v.thrusters}
    for i, c in enumerate(v.constraints):
        for ref in c.bodies():
            if ref not in known:
                raise SpecError(
                    f"constraints[{i}] ({type(c).__name__}) refers to "
                    f"{ref!r}, which is not a body or thruster. Known names: "
                    f"{sorted(known)}"
                )
    seen: set[str] = set()
    for name in [b.name for b in v.all_bodies()] + [t.name for t in v.thrusters]:
        if name in seen:
            raise SpecError(f"duplicate name {name!r}")
        seen.add(name)


def load(path: str | Path) -> Vehicle:
    path = Path(path)
    text = path.read_text()
    try:
        data = (json.loads(text) if path.suffix == ".json"
                else tomllib.loads(text))
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SpecError(f"{path}: {exc}") from exc
    return _anchor_sources(from_dict(data), path.parent, absolute=True)


def _anchor_sources(v: Vehicle, base: Path, *, absolute: bool) -> Vehicle:
    """Resolve generated-frame mesh paths against the spec file's own folder.

    A `Tabulated` shape records the STL it was measured from. Taken literally,
    a relative path resolves against wherever Camber happens to be run from, so
    a spec that works from the repository root silently loses its frame mesh
    one directory down -- the simulator draws a bounding box and the CFD tier
    raises. On load a relative path is anchored to the spec's folder; on save it
    is written relative to the new file's folder, so the pair can move together.
    """
    import dataclasses
    import os

    from .shapes import Tabulated

    for b in v.all_bodies():
        s = b.shape
        if not isinstance(s, Tabulated) or not s.source:
            continue
        src = Path(s.source)
        if absolute and not src.is_absolute():
            src = (base / src).resolve()
        elif not absolute and src.is_absolute():
            src = Path(os.path.relpath(src, base.resolve()))
        b.shape = dataclasses.replace(s, source=str(src))
    return v


def to_dict(v: Vehicle) -> dict[str, Any]:
    """Round-trippable dictionary form."""
    def shape_of(s: Shape) -> dict | None:
        if isinstance(s, Box):
            return {"kind": "box", "lx": s.lx, "ly": s.ly, "lz": s.lz}
        if isinstance(s, Cylinder):
            return {"kind": "cylinder", "radius": s.radius, "length": s.length}
        if isinstance(s, Sphere):
            return {"kind": "sphere", "radius": s.radius}
        if isinstance(s, Tabulated):
            return {"kind": "tabulated", "volume_m3": s.volume_m3,
                    "inertia_tensor": [list(r) for r in s.inertia_tensor],
                    "extents": list(s.extents), "planform": s.planform,
                    "source": s.source}
        if isinstance(s, PointMass):
            return {"kind": "point"}
        # This used to fall through to {"kind": "point"} for anything it did
        # not recognise, so saving a shape it had not been taught about kept
        # the mass and silently deleted the geometry and the inertia -- a
        # generated airframe would have come back as a point.
        raise TypeError(f"cannot serialise shape {type(s).__name__}")

    def body_of(b: Body) -> dict:
        d: dict[str, Any] = {"name": b.name, "mass": b.mass,
                             "shape": shape_of(b.shape),
                             "position": b.position.as_list()}
        roll, pitch, yaw = b.placement.rotation.as_euler()
        if any(abs(a) > 1e-12 for a in (roll, pitch, yaw)):
            d["rotation_deg"] = [roll / DEG, pitch / DEG, yaw / DEG]
        if isinstance(b, Battery):
            d.update(type="battery", capacity_ah=b.capacity_ah,
                     nominal_voltage=b.nominal_voltage, max_c_rate=b.max_c_rate)
        elif isinstance(b, Camera):
            d.update(type="camera",
                     view_direction=b.view_direction.as_list(),
                     horizontal_fov_deg=b.horizontal_fov / DEG)
        else:
            d["type"] = "body"
        return d

    out: dict[str, Any] = {
        "name": v.name,
        "mission": {"objective": v.mission.objective.value,
                    "payload_mass": v.mission.payload_mass},
        "bodies": [body_of(b) for b in v.bodies],
        "thrusters": [],
    }
    if v.mission.target_range is not None:
        out["mission"]["target_range"] = v.mission.target_range
    if v.mission.min_endurance is not None:
        out["mission"]["min_endurance"] = v.mission.min_endurance
    for t in v.thrusters:
        d = {"name": t.name, "type": "rotor", "mass": t.mass,
             "position": t.position.as_list()}
        if isinstance(t, Rotor):
            d.update(radius=t.radius, pitch=t.pitch, blades=t.blades,
                     ducted=t.ducted,
                     spin="cw" if t.spin is Spin.CW else "ccw",
                     torque_coefficient=t.torque_coefficient)
            if t.static_thrust_n is not None:
                d["static_thrust_n"] = t.static_thrust_n
            if t.motor_sku:
                d["motor_sku"] = t.motor_sku
            if t.prop_sku:
                d["prop_sku"] = t.prop_sku
        out["thrusters"].append(d)
    if v.frame.members:
        out["frame"] = [body_of(b) for b in v.frame.members]
    if v.constraints:
        out["constraints"] = [_constraint_dict(c) for c in v.constraints]
    return out


def _constraint_dict(c: Constraint) -> dict:
    """The inverse of `_constraint`, key for key.

    This did not exist, and `to_dict` simply omitted constraints -- so every
    JSON save silently deleted the CG box, the camera's view cone and every
    minimum distance a user had declared. The optimiser's `--out`, the framed
    vehicle `camber generate --into` writes, and every app project (stored as
    JSON) all lost them, and nothing complained: a vehicle with no constraints
    is a perfectly valid vehicle.
    """
    v = lambda x: [float(x.x), float(x.y), float(x.z)]      # noqa: E731
    if isinstance(c, Fixed):
        return {"type": "fixed", "body": c.body, "position": v(c.position)}
    if isinstance(c, Relative):
        return {"type": "relative", "a": c.a, "b": c.b, "offset": v(c.offset)}
    if isinstance(c, MinDistance):
        return {"type": "min_distance", "a": c.a, "b": c.b, "distance": c.distance}
    if isinstance(c, CGBox):
        return {"type": "cg_box", "lower": v(c.lower), "upper": v(c.upper)}
    if isinstance(c, Symmetry):
        return {"type": "symmetry", "a": c.a, "b": c.b, "plane": c.plane}
    if isinstance(c, ViewCone):
        return {"type": "view_cone", "camera": c.camera,
                "half_angle_deg": c.half_angle / DEG,
                "obstructed_by": list(c.obstructed_by)}
    raise TypeError(f"cannot serialise constraint {type(c).__name__}")


def save_json(v: Vehicle, path: str | Path) -> None:
    import copy
    path = Path(path)
    portable = _anchor_sources(copy.deepcopy(v), path.parent, absolute=False)
    path.write_text(json.dumps(to_dict(portable), indent=2) + "\n")
