"""A local, single-user web front end for the vehicle model.

Binds to loopback only. There is no authentication, no CSRF protection and no
multi-user state, because there is exactly one user on one machine and adding
those would be security theatre around a tool that never leaves localhost. It
refuses to bind anywhere else for the same reason.

The API is deliberately thin: the browser owns the *view*, and every number it
displays comes from the same Python that the CLI and the optimiser use. There
is no second implementation of mass properties in JavaScript, which is the
usual way a GUI and its engine drift apart.
"""

from __future__ import annotations

import json
import math
import threading
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..calibrate import design_features
from ..cfd.screen import screen
from ..control import G, controllability, solve_hover
from ..spec import io
from ..spec.units import Vec3
from ..spec.vehicle import Battery, Camera, Rotor, Vehicle

STATIC = Path(__file__).resolve().parent / "static"


def _shape_json(body) -> dict:
    from ..spec.shapes import Box, Cylinder, Sphere
    s = body.shape
    if isinstance(s, Box):
        return {"kind": "box", "lx": s.lx, "ly": s.ly, "lz": s.lz}
    if isinstance(s, Cylinder):
        return {"kind": "cylinder", "radius": s.radius, "length": s.length}
    if isinstance(s, Sphere):
        return {"kind": "sphere", "radius": s.radius}
    return {"kind": "point"}


def vehicle_json(v: Vehicle) -> dict:
    """Everything the viewport needs to draw, plus everything it must not
    compute itself."""
    mp = v.mass_properties()
    c = controllability(v)
    h = solve_hover(v)
    sc = screen(v)

    bodies = []
    for b in v.all_bodies():
        kind = ("battery" if isinstance(b, Battery)
                else "camera" if isinstance(b, Camera) else "body")
        entry = {
            "name": b.name, "kind": kind, "mass": b.mass,
            "position": b.position.as_list(), "shape": _shape_json(b),
            "frame": b in v.frame.members,
        }
        if isinstance(b, Camera):
            entry["boresight"] = b.boresight().as_list()
            entry["fov"] = b.horizontal_fov
        if isinstance(b, Battery):
            entry["energy_wh"] = b.energy_wh
        bodies.append(entry)

    thrusters = []
    for i, t in enumerate(v.thrusters):
        thrusters.append({
            "name": t.name,
            "position": t.position.as_list(),
            "direction": t.direction().as_list(),
            "radius": getattr(t, "radius", 0.0),
            "spin": getattr(getattr(t, "spin", None), "name", ""),
            "thrust": (float(h.thrusts[i]) if h.feasible and h.thrusts is not None
                       else None),
            "utilisation": (float(h.utilisation[i])
                            if h.utilisation is not None
                            and np.isfinite(h.utilisation[i]) else None),
        })

    positions = v.part_positions()
    constraints = []
    for con in v.constraints:
        try:
            amount = float(con.violation(positions, v))
        except Exception:  # noqa: BLE001
            amount = float("nan")
        constraints.append({
            "type": type(con).__name__,
            "bodies": list(con.bodies()),
            "violation": None if math.isnan(amount) else amount,
            "ok": amount <= 1e-9,
        })

    return {
        "name": v.name,
        "mission": v.mission.objective.value,
        "bodies": bodies,
        "thrusters": thrusters,
        "constraints": constraints,
        "mass": {
            "total_kg": mp.mass,
            "cg": mp.cg.as_list(),
            "inertia_diag": [float(x) for x in np.diag(mp.inertia)],
            "physical": bool(mp.is_physical()),
            "energy_wh": v.total_energy_wh,
        },
        "control": {
            "rank": c.rank,
            "commandable": list(c.controllable_axes),
            "uncommandable": list(c.uncontrollable_axes),
            "margin": c.margin,
        },
        "hover": {
            "feasible": bool(h.feasible),
            "weight_n": h.weight_n,
            "reason": h.reason,
            "max_utilisation": (float(np.nanmax(h.utilisation))
                                if h.utilisation is not None
                                and np.any(np.isfinite(h.utilisation)) else None),
        },
        "screen": {
            "overlap": sc.overlap,
            "disc_loading": sc.disc_loading,
            "worth_solving": sc.worth_solving,
            "reason": sc.reason,
        },
        "features": [float(x) for x in design_features(v)],
    }


class Session:
    """The one vehicle being edited, guarded for concurrent requests."""

    def __init__(self, vehicle: Vehicle, source: Path | None = None) -> None:
        self.vehicle = vehicle
        self.source = source
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return vehicle_json(self.vehicle)

    def move(self, name: str, position: list[float]) -> dict:
        """Reposition one part. The only mutation the viewport can make.

        Rotors move too: dragging an arm outward is the single most instructive
        thing to do in this tool, because hover trim, controllability margin
        and wake overlap all respond at once.
        """
        with self._lock:
            target = self.vehicle.part(name)
            target.placement = type(target.placement)(
                Vec3(*position), target.placement.rotation)
            return vehicle_json(self.vehicle)

    def save(self, path: Path) -> None:
        with self._lock:
            io.save_json(self.vehicle, path)


class Handler(BaseHTTPRequestHandler):
    session: Session = None            # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:      # quiet; this is a local tool
        pass

    def _send(self, payload: dict | bytes, status: int = 200,
              content_type: str = "application/json") -> None:
        body = (payload if isinstance(payload, bytes)
                else json.dumps(payload).encode())
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._send((STATIC / "index.html").read_bytes(), 200,
                       "text/html; charset=utf-8")
            return
        if route == "/favicon.ico":
            # An empty 204 rather than a 404: the browser asks on every load
            # and a missing favicon is not an error worth logging.
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if route == "/api/vehicle":
            self._send(self.session.snapshot())
            return
        static = STATIC / route.lstrip("/")
        if static.is_file() and STATIC in static.resolve().parents:
            kind = ("application/javascript" if static.suffix == ".js"
                    else "text/css" if static.suffix == ".css" else "text/plain")
            self._send(static.read_bytes(), 200, kind)
            return
        self._send({"error": f"no route {route}"}, 404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send({"error": f"bad JSON: {exc}"}, 400)
            return

        if route == "/api/move":
            try:
                self._send(self.session.move(payload["name"], payload["position"]))
            except KeyError as exc:
                self._send({"error": f"unknown part {exc}"}, 404)
            except (TypeError, ValueError) as exc:
                self._send({"error": str(exc)}, 400)
            return
        if route == "/api/save":
            target = Path(payload.get("path", "vehicle.json"))
            self.session.save(target)
            self._send({"saved": str(target)})
            return
        self._send({"error": f"no route {route}"}, 404)


def serve(spec: Path, port: int = 5173, open_browser: bool = True) -> None:
    """Run the viewport. Loopback only -- see the module docstring."""
    vehicle = io.load(spec)
    Handler.session = Session(vehicle, spec)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"{vehicle.name}: {url}   (ctrl-c to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
