"""What the UI can do, as plain functions over a Project.

Units and axes are converted here and nowhere else. Camber works in metres in
the NED body frame (GEOM-02: z points DOWN). People read millimetres, and
"up" as positive, so the UI's frame is (forward, right, up) in mm: NED with
the vertical negated, then scaled. Every position crossing this boundary goes
through `to_ui` / `from_ui`.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from ..layout import (balance, clearances, find_pairs, symmetrise,
                      symmetry_report, wheelbase)
from ..layout.presets import PRESETS, apply_layout
from ..layout.symmetry import move_with_twin
from ..spec import io
from ..spec.shapes import Box, Cylinder, PointMass, Sphere, Tabulated
from ..spec.units import Vec3
from ..spec.vehicle import Battery, Body, Camera, Placement, Rotor, Spin

MM = 1000.0
UI = np.array([1.0, 1.0, -1.0])            # NED -> (forward, right, up)
LINKS = {"off": (), "lr": ("xz",), "4way": ("xz", "yz")}


def to_ui(v) -> list[float]:
    a = np.asarray(v.as_list() if hasattr(v, "as_list") else v, dtype=float)
    return [round(float(x), 4) for x in a * UI * MM]


def from_ui(p) -> Vec3:
    a = np.asarray(p, dtype=float) / MM * UI
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("a position is three finite numbers, in mm")
    return Vec3(*map(float, a))


def _three_quat(R_ned: np.ndarray) -> list[float]:
    """A part's orientation as a three.js quaternion (x, y, z, w).

    three.js is y-up with forward along -z; the map from NED is the rotation
    M = [[0,1,0],[0,0,-1],[-1,0,0]], and an orientation conjugates by it.
    """
    M = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
    R = M @ np.asarray(R_ned, float) @ M.T
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        return [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                (R[1, 0] - R[0, 1]) / (4 * w), w]
    # 180-degree case: pick the largest diagonal.
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
    q[i] = s / 4.0
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    return q


def _kind(p) -> str:
    if isinstance(p, Rotor):
        return "rotor"
    if isinstance(p, Battery):
        return "battery"
    if isinstance(p, Camera):
        return "camera"
    return "body"


def _shape(s) -> dict:
    if isinstance(s, Box):
        return {"type": "box", "size": [s.lx * MM, s.ly * MM, s.lz * MM]}
    if isinstance(s, Cylinder):
        return {"type": "cylinder", "radius": s.radius * MM, "length": s.length * MM}
    if isinstance(s, Sphere):
        return {"type": "sphere", "radius": s.radius * MM}
    if isinstance(s, Tabulated):
        return {"type": "mesh", "extents": [2 * e * MM for e in s.extents]}
    return {"type": "point"}


def vehicle_view(project) -> dict:
    """Everything the editor draws and every check it shows."""
    v = project.vehicle
    pairs = {pl: find_pairs(v, pl) for pl in ("xz", "yz")}
    parts = []
    for p, where in [*[(b, "payload") for b in v.bodies],
                     *[(f, "frame") for f in v.frame.members],
                     *[(t, "rotor") for t in v.thrusters]]:
        d = {"name": p.name, "kind": "frame" if where == "frame" else _kind(p),
             "mass_g": round(p.mass * 1000.0, 3), "pos": to_ui(p.position),
             "quat": _three_quat(p.placement.rotation.matrix),
             "editable": where != "frame",
             "twin": {pl: pairs[pl].twin(p.name) for pl in pairs},
             "centred": {pl: p.name in pairs[pl].centred for pl in pairs}}
        if isinstance(p, Rotor):
            d.update(radius=p.radius * MM, spin="cw" if p.spin == Spin.CW else "ccw",
                     axis=[float(x) for x in (np.array(p.direction().as_list()))],
                     thrust_n=p.static_thrust_n)
        else:
            d["shape"] = _shape(p.shape)
        if isinstance(p, Battery):
            d.update(capacity_ah=p.capacity_ah, voltage=p.nominal_voltage)
        if isinstance(p, Camera):
            b = np.array(p.boresight().as_list())
            d.update(boresight=[float(b[1]), float(-b[2]), float(-b[0])],   # three.js
                     fov_deg=math.degrees(p.horizontal_fov))
        parts.append(d)

    sym = {}
    for pl in ("xz", "yz"):
        r = symmetry_report(v, pl, pairs[pl])
        sym[pl] = {"max_error_mm": r["max_error"] * MM,
                   "rows": [{"a": x["a"], "b": x["b"], "error_mm": x["error"] * MM}
                            for x in r["rows"] if x["error"] > 1e-7],
                   "unmatched": r["unmatched"]}
    from ..layout import yaw_balance
    bal = balance(v)
    clr = clearances(v)
    out = {
        "id": project.id, "name": project.meta.get("name", v.name),
        "mass_g": bal["mass"] * 1000.0, "cg": to_ui(bal["cg"]),
        "parts": parts,
        "wheelbase_mm": wheelbase(v) * MM,
        "symmetry": sym, "yaw_balance": yaw_balance(v),
        "clearance": {"min_tip_gap_mm": (clr["min_tip_gap"] * MM
                                         if math.isfinite(clr["min_tip_gap"]) else None),
                      "conflicts": [_conflict(c) for c in clr["conflicts"]]},
        "balance": {"thrust_to_weight": bal.get("thrust_to_weight"),
                    "cg_offset_mm": [x * MM for x in bal.get("cg_offset", [0, 0])],
                    "hover_imbalance": bal.get("hover_imbalance"),
                    "hover_split": bal.get("hover_split"),
                    "hover_reason": bal.get("hover_reason"),
                    "controllable": bal.get("controllable", []),
                    "thrust_centroid": (to_ui(bal["thrust_centroid"])
                                        if "thrust_centroid" in bal else None)},
        "mission": v.mission.objective.value,
        "has_generated_frame": any(isinstance(b.shape, Tabulated)
                                   for b in v.all_bodies()),
        "can_undo": project.can_undo, "can_redo": project.can_redo,
        "presets": list(PRESETS),
        "aero_table": project.meta.get("aero_table"),
        "constraints": _constraints(v),
        "wash": _wash(v),
    }
    return out


def _constraints(v) -> list[dict]:
    """Every constraint the design declares, checked now, in words."""
    from ..spec.constraints import (CGBox, Fixed, MinDistance, Relative,
                                    Symmetry, ViewCone)
    pos = v.part_positions()
    pos["__cg__"] = v.mass_properties().cg
    out = []
    for c in v.constraints:
        try:
            breach = float(c.violation(pos, v))
        except (KeyError, ValueError) as exc:
            out.append({"label": type(c).__name__, "ok": False,
                        "note": f"refers to a part that is not there ({exc})"})
            continue
        nice = lambda n: n.replace("_", " ")                   # noqa: E731
        if isinstance(c, CGBox):
            label, note = "CG inside its box", f"{breach * MM:.1f} mm outside"
        elif isinstance(c, ViewCone):
            label = f"{nice(c.camera).capitalize()} can see past " + \
                ", ".join(map(nice, c.obstructed_by))
            note = f"blocked by {math.degrees(breach):.1f} degrees"
        elif isinstance(c, MinDistance):
            label = f"{nice(c.a).capitalize()} at least {c.distance * MM:.0f} mm from {nice(c.b)}"
            note = f"{breach * MM:.1f} mm too close"
        elif isinstance(c, Fixed):
            label, note = f"{nice(c.body).capitalize()} stays put", f"{breach * MM:.1f} mm away"
        elif isinstance(c, Symmetry):
            label, note = f"{nice(c.a).capitalize()} mirrors {nice(c.b)}", f"{breach * MM:.1f} mm off"
        elif isinstance(c, Relative):
            label, note = f"{nice(c.a).capitalize()} tied to {nice(c.b)}", f"{breach * MM:.1f} mm off"
        else:
            label, note = type(c).__name__, f"breach {breach:.3g}"
        out.append({"label": label, "ok": breach <= 1e-9, "note": "" if breach <= 1e-9 else note})
    return out


def _wash(v) -> dict | None:
    """How much of the airframe sits in the rotor wash, and what it costs."""
    try:
        from ..cfd.screen import screen
        sc = screen(v, samples=6000)
    except Exception:                                        # noqa: BLE001
        return None
    out = {"overlap": sc.overlap, "loss": None, "sigma": None, "outside": []}
    from ..paths import interference_model
    path = interference_model()
    if path.exists():
        from ..calibrate import load_model
        global _MODEL
        if "_MODEL" not in globals():
            _MODEL = load_model(path)
        loss, sigma = _MODEL.correction(v)
        out.update(loss=loss, sigma=sigma, outside=_MODEL.outside(v))
    return out


def _conflict(c: dict) -> dict:
    if "rotor" in c:
        return {"a": c["rotor"], "b": c["body"], "what": "prop touches body",
                "gap_mm": c["plan_gap"] * MM}
    return {"a": c["a"], "b": c["b"], "what": "props overlap", "gap_mm": c["gap"] * MM}


# --- edits: each one validates, applies, commits a labelled undo step ---------

def _part(v, name):
    for p in [*v.bodies, *v.thrusters, *v.frame.members]:
        if p.name == name:
            return p
    raise KeyError(f"no part called {name!r}")


def move(project, name: str, pos, link: str = "off") -> list[str]:
    if link not in LINKS:
        raise ValueError(f"link must be one of {', '.join(LINKS)}")
    p = _part(project.vehicle, name)
    if p in project.vehicle.frame.members:
        raise ValueError(f"{name} is the airframe and is placed by the structure step")
    moved = move_with_twin(project.vehicle, name, from_ui(pos).as_list(),
                           planes=LINKS[link])
    project.commit(f"move {name}")
    return moved


def make_symmetric(project, link: str = "lr") -> list[str]:
    changed = []
    for pl in LINKS.get(link, ()):
        changed += symmetrise(project.vehicle, pl)
    project.commit("make symmetric")
    return changed


def layout(project, kind: str, wheelbase_mm: float, height_mm: float) -> None:
    if not 20.0 <= float(wheelbase_mm) <= 5000.0:
        raise ValueError("wheelbase should be between 20 mm and 5 m")
    apply_layout(project.vehicle, kind, float(wheelbase_mm) / MM,
                 height=-float(height_mm) / MM)
    project.commit(f"{kind} layout")


def update(project, name: str, fields: dict) -> None:
    p = _part(project.vehicle, name)
    if "mass_g" in fields:
        m = float(fields["mass_g"])
        if not 0.0 < m < 1e6:
            raise ValueError("mass must be positive, in grams")
        p.mass = m / 1000.0
    if "size_mm" in fields and isinstance(getattr(p, "shape", None), Box):
        lx, ly, lz = (float(x) / MM for x in fields["size_mm"])
        if min(lx, ly, lz) <= 0:
            raise ValueError("every dimension must be positive")
        p.shape = Box(lx, ly, lz)
    if isinstance(p, Rotor):
        if "spin" in fields:
            p.spin = Spin.CW if fields["spin"] == "cw" else Spin.CCW
        if "radius_mm" in fields:
            r = float(fields["radius_mm"]) / MM
            if r <= 0:
                raise ValueError("prop radius must be positive")
            p.radius = r
        if "thrust_n" in fields:
            t = float(fields["thrust_n"])
            if t <= 0:
                raise ValueError("thrust must be positive")
            p.static_thrust_n = t
    if "name" in fields and fields["name"] != name:
        new = str(fields["name"]).strip()
        if not new or any(q.name == new for q in
                          [*project.vehicle.all_bodies(), *project.vehicle.thrusters]):
            raise ValueError(f"{new!r} is empty or already used")
        p.name = new
    project.commit(f"edit {name}")


def add(project, kind: str, name: str, mass_g: float, size_mm, pos) -> None:
    v = project.vehicle
    name = str(name).strip()
    if not name or any(q.name == name for q in [*v.all_bodies(), *v.thrusters]):
        raise ValueError(f"{name!r} is empty or already used")
    size = [float(x) / MM for x in size_mm]
    shape = Box(*size)
    placement = Placement(from_ui(pos))
    mass = float(mass_g) / 1000.0
    if kind == "battery":
        part = Battery(name, mass=mass, shape=shape, placement=placement,
                       capacity_ah=1.3, nominal_voltage=14.8, max_c_rate=75.0)
    elif kind == "camera":
        part = Camera(name, mass=mass, shape=shape, placement=placement,
                      view_direction=Vec3(1.0, 0.0, 0.0),
                      horizontal_fov=math.radians(120))
    else:
        part = Body(name, mass=mass, shape=shape, placement=placement)
    v.bodies.append(part)
    project.commit(f"add {name}")


def remove(project, name: str) -> None:
    v = project.vehicle
    p = _part(v, name)
    if isinstance(p, Rotor):
        if len(v.thrusters) <= 3:
            raise ValueError("a multirotor needs at least three rotors")
        v.thrusters.remove(p)
    elif p in v.frame.members:
        v.frame.members.remove(p)
    else:
        v.bodies.remove(p)
    project.commit(f"remove {name}")


def apply_vehicle(project, path: Path, label: str) -> None:
    """Adopt a vehicle a job produced (an optimiser pick, a framed vehicle)."""
    new = io.load(path)
    new.name = project.vehicle.name
    project.vehicle = new
    project.commit(label)


def apply_front_point(project, front: Path, index: int) -> None:
    data = json.loads(Path(front).read_text())
    points = data["points"]
    if not 0 <= index < len(points):
        raise IndexError("no such design on the front")
    new = io.from_dict(points[index]["vehicle"])
    new.name = project.vehicle.name
    project.vehicle = new
    project.commit(f"optimiser design {index + 1}")


def frame_mesh_three(project) -> bytes | None:
    """The generated frame as binary STL, in three.js coordinates and mm.

    Converted here so the browser never has to know about NED or Z-up, and
    decimated so a 600k-facet print mesh does not have to cross the wire.
    """
    from ..generate.topology import decimate
    from ..geometry.surface import frame_mesh_ned

    for b in project.vehicle.all_bodies():
        if isinstance(b.shape, Tabulated):
            m = decimate(frame_mesh_ned(b.shape), target_faces=150_000)
            M = np.eye(4)
            M[:3, :3] = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0],
                                  [-1.0, 0.0, 0.0]]) * MM
            m = m.copy()
            m.apply_transform(M)
            return m.export(file_type="stl")
    return None
