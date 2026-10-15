"""The app's HTTP server: static UI, a JSON API, and project files.

Standard library only (UI-02), bound to loopback. Unlike the old viewport it
is *not* unauthenticated: this server starts processes -- CFD, topology runs,
the simulator, user pilot scripts -- and any web page open in any browser on
this machine can send requests to 127.0.0.1. So every API call must carry a
token generated at launch, and the Host header must be the loopback address,
which is what stops a DNS-rebinding page from reaching it by a hostname.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import paths
from . import api
from .jobs import JobManager, child_env
from .project import HOME, TEMPLATES, Project

STATIC = Path(__file__).resolve().parent / "static"
PILOTS = paths.pilots()


class App:
    """Open projects, their job managers, and the launch token."""

    def __init__(self, home: Path | None = None) -> None:
        self.home = Path(home or HOME)
        self.home.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(24)
        self._open: dict[str, Project] = {}
        self._flights: dict[str, object] = {}
        self._lock = threading.RLock()

    def project(self, pid: str) -> Project:
        with self._lock:
            if pid not in self._open:
                root = (self.home / pid).resolve()
                if self.home.resolve() not in root.parents:
                    raise PermissionError("no such project")
                self._open[pid] = Project(root)
            return self._open[pid]

    def jobs(self, pid: str) -> JobManager:
        return JobManager(self.project(pid))

    def flight(self, pid: str, start: dict | None = None):
        """The live flight for a project, started on demand. Starting a new
        one ends the old one: two simulations of the same vehicle running at
        once would both be real, and only one of them would be watched."""
        from .live import Flight, aero_for
        with self._lock:
            if start is not None:
                old = self._flights.pop(pid, None)
                if old is not None:
                    old.stop()
                proj = self.project(pid)
                f = Flight(proj.vehicle,
                           pilot=str(start.get("pilot", "manual")),
                           mode=str(start.get("mode", "assist")),
                           course=bool(start.get("course", True)),
                           speed=float(start.get("speed", 14.0)),
                           aero=aero_for(proj, bool(start.get("aero"))),
                           pilot_factory=_project_pilot(proj))
                self._flights[pid] = f.start()
            if pid not in self._flights:
                raise KeyError("no flight is running")
            return self._flights[pid]

    def land(self, pid: str | None = None) -> None:
        """Stop one flight, or all of them when the app closes."""
        with self._lock:
            ids = [pid] if pid else list(self._flights)
            for i in ids:
                f = self._flights.pop(i, None)
                if f is not None:
                    f.stop()

    def create(self, name: str, template: str) -> Project:
        p = Project.create(name, template, home=self.home)
        pilots = p.root / "pilots"
        pilots.mkdir(exist_ok=True)
        for f in PILOTS.glob("*.py"):
            shutil.copy(f, pilots / f.name)
        return p

    def import_spec(self, path: Path, name: str | None = None) -> Project:
        """A new project from a vehicle file, carrying its meshes with it.

        A generated frame's mesh is copied into the project and repointed, so
        the project is a self-contained folder: moving or deleting the file it
        was imported from must not leave the airframe drawn as a box.
        """
        import dataclasses

        from ..spec import io
        from ..spec.shapes import Tabulated
        v = io.load(path)
        p = self.create(name or v.name, "racer5")
        v.name = p.vehicle.name
        meshes = p.root / "results" / "imported"
        for b in v.all_bodies():
            if isinstance(b.shape, Tabulated) and b.shape.source \
                    and Path(b.shape.source).exists():
                meshes.mkdir(parents=True, exist_ok=True)
                dst = meshes / f"{b.name}{Path(b.shape.source).suffix}"
                shutil.copy(b.shape.source, dst)
                b.shape = dataclasses.replace(b.shape, source=str(dst))
        p.vehicle = v
        p.commit(f"imported {Path(path).name}")
        return p


def _pilots(project: Project) -> list[dict]:
    builtin = [{"id": "gates", "name": "Gate race", "about": "Laps an eight-gate oval"},
               {"id": "hover", "name": "Hover", "about": "Climbs 1.5 m and holds"}]
    d = project.root / "pilots"
    mine = [{"id": f"pilots/{f.name}", "name": f.stem.replace("_", " ").capitalize(),
             "about": (f.read_text().split('"""')[1].strip().splitlines()[0]
                       if '"""' in f.read_text() else "")}
            for f in sorted(d.glob("*.py"))] if d.exists() else []
    return builtin + mine


PILOT_TEMPLATE = '''"""{title}: describe what this pilot does in one line.

    camber sim vehicle.json --pilot pilots/{file} --viewer

`update(state, sim)` runs every 2 ms. Set a target and the cascade flies there,
or override it completely and return one motor command per rotor in [0, 1].
state: position, velocity, quaternion, rates, specific_force, thrust (x fwd,
y left, z up). See docs/FLYING.md for the layers you can take over.
"""

import numpy as np

from camber.sim.flight import Cascade


class Pilot(Cascade):
    def update(self, state, sim):
        t = sim.t
        self.target = np.array([2.0 * np.cos(0.5 * t), 2.0 * np.sin(0.5 * t), 1.5])
        return super().update(state, sim)
'''


def _project_pilot(proj: Project):
    """Load a pilot from the project's own `pilots/` folder, by path."""
    def factory(kind: str, course, speed: float):
        import importlib.util
        f = proj.path(kind)
        if not f.is_file() or f.suffix != ".py":
            raise ValueError(f"no such pilot: {kind}")
        spec = importlib.util.spec_from_file_location(f"camber_pilot_{f.stem}", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, "Pilot", None)
        if cls is None:
            raise ValueError(f"{f.name} defines no Pilot class")
        try:
            return cls(course, speed=speed)              # a racing pilot
        except TypeError:
            return cls()
    return factory


def _open_path(path: Path) -> None:
    """Hand a file or folder to the desktop: the user's editor, file manager."""
    opener = shutil.which("xdg-open")
    if opener:
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)


class Handler(BaseHTTPRequestHandler):
    app: App = None                     # set by serve()
    server_version = "Camber"
    # The flight view sends sticks and reads telemetry every animation frame.
    # On HTTP/1.0 that is a new TCP connection 60 times a second, and a long
    # flight leaves tens of thousands of sockets in TIME_WAIT. Every response
    # here carries a Content-Length, so keep-alive is safe.
    protocol_version = "HTTP/1.1"
    # And TCP_NODELAY with it. Small request, small response, one after the
    # other, is exactly the pattern where Nagle's algorithm waits for the
    # other end's delayed ACK: measured at 41 ms a round trip, which would
    # have held the flight view to 24 Hz and blamed it on the physics.
    disable_nagle_algorithm = True

    def log_message(self, *args) -> None:
        pass

    # --- plumbing ------------------------------------------------------------
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def _authorised(self) -> bool:
        return secrets.compare_digest(self.headers.get("X-Camber-Token", ""),
                                      self.app.token)

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=_jsonable).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _file(self, path: Path, download: str | None = None) -> None:
        """Serve a file, with Range support: <video> seeks by range requests
        and will not scrub without them."""
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            ctype = "text/javascript"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def _body(self) -> dict:
        """Read the request body -- always, even when the request is about to
        be refused. On a kept-alive connection an unread body is still sitting
        in the socket when the next request arrives, and the server parses
        `{"do": "start"}` as a request line ("Bad HTTP/0.9 request type").
        A refusal must not poison the connection it was sent on."""
        n = int(self.headers.get("Content-Length") or 0)
        if n > 4_000_000:
            self.close_connection = True          # too big to drain; hang up
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def _drain(self) -> None:
        try:
            self._body()
        except (ValueError, json.JSONDecodeError):
            pass

    # --- routing ---------------------------------------------------------------
    def do_GET(self) -> None:
        self._drain()                             # a GET may still carry one
        if not self._host_ok():
            return self._error(403, "bad host")
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            return self._static(url.path)
        if not self._authorised() and "t" not in parse_qs(url.query):
            return self._error(401, "missing token")
        if "t" in parse_qs(url.query) and not secrets.compare_digest(
                parse_qs(url.query)["t"][0], self.app.token):
            return self._error(401, "bad token")
        self._dispatch("GET", url.path, parse_qs(url.query), {})

    def do_POST(self) -> None:
        try:
            body = self._body()                   # first, whatever happens next
        except (ValueError, json.JSONDecodeError) as exc:
            return self._error(400, str(exc))
        if not self._host_ok():
            return self._error(403, "bad host")
        if not self._authorised():
            return self._error(401, "missing token")
        self._dispatch("POST", urlparse(self.path).path, {}, body)

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (STATIC / rel).resolve()
        if STATIC.resolve() not in f.parents or not f.is_file():
            return self._error(404, "not found")
        self._file(f)

    def _dispatch(self, method: str, path: str, query: dict, body: dict) -> None:
        try:
            result = self._route(method, path, query, body)
            if result is not None:
                self._json(result)
        except (KeyError, FileNotFoundError) as exc:
            self._error(404, str(exc).strip("'\""))
        except (ValueError, IndexError, FileExistsError, RuntimeError,
                PermissionError) as exc:
            self._error(400, str(exc))
        except Exception as exc:                        # noqa: BLE001
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")

    def _route(self, method, path, q, b):
        app = self.app
        parts = [p for p in path.split("/") if p][1:]         # drop "api"
        if parts == ["projects"]:
            if method == "GET":
                return {"projects": Project.list(app.home),
                        "templates": list(TEMPLATES), "home": str(app.home)}
            if b.get("import"):
                p = app.import_spec(Path(b["import"]).expanduser(), b.get("name"))
            else:
                p = app.create(b.get("name", ""), b.get("template", "racer5"))
            return {"id": p.id}
        if not parts or parts[0] != "p" or len(parts) < 3:
            raise KeyError("no such endpoint")
        pid, action, rest = parts[1], parts[2], parts[3:]
        proj = app.project(pid)

        if method == "GET":
            if action == "vehicle":
                return api.vehicle_view(proj)
            if action == "frame.stl":
                data = api.frame_mesh_three(proj)
                if data is None:
                    raise KeyError("this vehicle has no generated frame")
                self.send_response(200)
                self.send_header("Content-Type", "model/stl")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return None
            if action == "jobs":
                if rest and rest[-1] == "log":
                    return {"log": app.jobs(pid).log(rest[0], tail=200)}
                return {"jobs": [j.public() for j in app.jobs(pid).list()]}
            if action == "file":
                f = proj.path(q["path"][0])
                if not f.is_file():
                    raise FileNotFoundError(q["path"][0])
                dl = f.name if q.get("download") else None
                self._file(f, download=dl)
                return None
            if action == "front":
                job = app.jobs(pid).get(q["job"][0])
                return json.loads(proj.path(job.outputs["front"]).read_text())
            if action == "json":
                return json.loads(proj.path(q["path"][0]).read_text())
            if action == "envelope":
                from ..cfd.envelope import envelope, kind_of
                return {"kind": kind_of(proj.vehicle),
                        "points": [{"name": p.name, "alpha_deg": p.alpha_deg,
                                    "speed": p.speed, "thrust_n": p.thrust, "why": p.why}
                                   for p in envelope(proj.vehicle, n=int(q.get("n", ["6"])[0]))]}
            if action == "pilots":
                return {"pilots": _pilots(proj)}
            if action == "live":
                f = app.flight(pid)
                f.touch()
                return f.snapshot()
            if action == "export":
                return self._export(proj, q.get("what", ["vehicle_json"])[0])
            raise KeyError(f"no GET {action}")

        # --- POST: edits ---------------------------------------------------------
        if action == "move":
            api.move(proj, b["name"], b["pos"], b.get("link", "off"))
        elif action == "symmetrise":
            api.make_symmetric(proj, b.get("link", "lr"))
        elif action == "layout":
            api.layout(proj, b["kind"], b["wheelbase_mm"], b.get("height_mm", 30.0))
        elif action == "update":
            api.update(proj, b["name"], b.get("fields", {}))
        elif action == "add":
            api.add(proj, b.get("kind", "body"), b["name"], b["mass_g"],
                    b["size_mm"], b.get("pos", [0, 0, 0]))
        elif action == "remove":
            api.remove(proj, b["name"])
        elif action == "undo":
            proj.undo()
        elif action == "redo":
            proj.redo()
        elif action == "jobs":
            if rest and rest[-1] == "cancel":
                return app.jobs(pid).cancel(rest[0]).public()
            return app.jobs(pid).start(b["kind"], b.get("params", {})).public()
        elif action == "apply":
            job = app.jobs(pid).get(b["job"])
            if job.kind == "optimise":
                api.apply_front_point(proj, proj.path(job.outputs["front"]),
                                      int(b.get("index", 0)))
            elif job.kind == "generate":
                api.apply_vehicle(proj, proj.path(job.outputs["vehicle"]), "grown frame")
            elif job.kind == "aero":
                proj.meta["aero_table"] = job.outputs["table"]
                (proj.root / "project.json").write_text(json.dumps(proj.meta, indent=1))
            else:
                raise ValueError(f"nothing to apply from a {job.kind} job")
        elif action == "fly":
            self._fly(proj, b)
            return {"ok": True}
        elif action == "live":
            return self._live(pid, b)
        elif action == "pilot":
            return self._new_pilot(proj, b.get("name", "my pilot"))
        elif action == "open":
            target = proj.path(b.get("path", "."))
            _open_path(target)
            return {"ok": True}
        else:
            raise KeyError(f"no POST {action}")
        return api.vehicle_view(proj)

    def _live(self, pid: str, b: dict) -> dict:
        """Fly here, in the app: one round trip carries the sticks in and the
        telemetry out, so there is no second channel to keep in step."""
        app = self.app
        what = str(b.get("do", "poll"))
        if what == "start":
            f = app.flight(pid, start=b)
            return {"scene": f.scene(), "state": f.snapshot()}
        if what == "stop":
            app.land(pid)
            return {"stopped": True}
        f = app.flight(pid)
        f.touch()
        if what == "input":
            s = b.get("sticks") or [0.0, 0.0, 0.0, 0.5]
            f.set_sticks(*[float(x) for x in s[:4]])
        elif what == "mode":
            f.set_mode(str(b.get("mode", "assist")))
        elif what == "arm":
            f.arm(bool(b.get("on", True)))
        elif what == "reset":
            f.reset()
        elif what != "poll":
            raise ValueError(f"no such flight action: {what}")
        return {"state": f.snapshot()}

    def _fly(self, proj: Project, b: dict) -> None:
        """Open MuJoCo's own window, flying the current design. Not a job: it
        is interactive, and lives as long as the user keeps it open."""
        from ..spec import io
        spec = proj.root / "results" / "live-vehicle.json"
        io.save_json(proj.vehicle, spec)
        pilot = str(b.get("pilot", "gates"))
        if pilot not in ("gates", "hover"):
            pilot = str(proj.path(pilot))
        argv = [sys.executable, "-m", "camber.cli", "sim", str(spec), "--pilot", pilot,
                "--viewer", "--duration", str(float(b.get("duration", 120))),
                "--speed", str(float(b.get("speed", 14)))]
        if b.get("aero") and proj.meta.get("aero_table"):
            argv += ["--aero", str(proj.path(proj.meta["aero_table"]))]
        subprocess.Popen(argv, cwd=proj.root, env=child_env(), start_new_session=True,
                         stdout=open(proj.root / "results" / "live.log", "w"),
                         stderr=subprocess.STDOUT)

    def _new_pilot(self, proj: Project, name: str) -> dict:
        stem = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "pilot"
        d = proj.root / "pilots"
        d.mkdir(exist_ok=True)
        f = d / f"{stem}.py"
        n = 2
        while f.exists():
            f = d / f"{stem}_{n}.py"
            n += 1
        f.write_text(PILOT_TEMPLATE.format(title=name.strip() or "My pilot", file=f.name))
        _open_path(f)
        return {"id": f"pilots/{f.name}", "path": str(f)}

    def _export(self, proj: Project, what: str):
        from ..geometry import cad
        from ..spec import io
        out = proj.root / "results" / "export"
        out.mkdir(parents=True, exist_ok=True)
        name = proj.vehicle.name
        if what == "vehicle_json":
            f = out / f"{name}.json"
            io.save_json(proj.vehicle, f)
        elif what == "stl":
            f = out / f"{name}.stl"
            cad.export(proj.vehicle, f)                      # mm, Z-up
        elif what == "step":
            f = out / f"{name}.step"
            cad.export(proj.vehicle, f)
        else:
            raise ValueError(f"cannot export {what!r}")
        self._file(f, download=f.name)
        return None


def _jsonable(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def serve(port: int = 0, home: Path | None = None) -> tuple[ThreadingHTTPServer, App]:
    """Start the server on a free loopback port; returns it and the App."""
    app = App(home)
    Handler.app = app
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    return srv, app
