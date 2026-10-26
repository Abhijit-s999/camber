"""The app, driven the way the UI drives it: over HTTP, with the token."""

import http.client
import json
import threading
import time

import pytest

from camber.app.project import Project
from camber.app.server import serve


@pytest.fixture
def server(tmp_path):
    srv, app = serve(0, home=tmp_path / "Camber")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1], app
    srv.shutdown()


def call(port, token, method, path, body=None, host="127.0.0.1"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    headers = {"Host": f"{host}:{port}", "Content-Type": "application/json"}
    if token:
        headers["X-Camber-Token"] = token
    c.request(method, path, json.dumps(body) if body is not None else None, headers)
    r = c.getresponse()
    data = r.read()
    try:
        return r.status, json.loads(data)
    except ValueError:              # binary bodies (an STL) are not JSON
        return r.status, data


@pytest.fixture
def project(server):
    port, app = server
    s, r = call(port, app.token, "POST", "/api/projects",
                {"name": "Test racer", "template": "racer5"})
    assert s == 200
    return port, app, r["id"]


class TestSecurity:
    def test_the_api_needs_the_token(self, server):
        port, app = server
        assert call(port, None, "GET", "/api/projects")[0] == 401
        assert call(port, "wrong", "POST", "/api/projects", {"name": "x"})[0] == 401

    def test_a_rebinding_hostname_is_refused(self, server):
        """A page that points evil.example at 127.0.0.1 still sends its own
        name in the Host header, and that is what gives it away."""
        port, app = server
        assert call(port, app.token, "GET", "/api/projects",
                    host="evil.example")[0] == 403

    def test_files_cannot_escape_the_project(self, project):
        port, app, pid = project
        s, _ = call(port, app.token, "GET", f"/api/p/{pid}/file?path=../../etc/passwd")
        assert s == 400

    def test_the_ui_loads_without_a_token(self, server):
        port, _ = server
        s, body = call(port, None, "GET", "/")
        assert s == 200 and b"<html" in body.lower()


class TestEditing:
    def test_the_vehicle_view_has_what_the_editor_needs(self, project):
        port, app, pid = project
        s, v = call(port, app.token, "GET", f"/api/p/{pid}/vehicle")
        assert s == 200
        names = {p["name"] for p in v["parts"]}
        assert {"front_right", "battery", "chassis"} <= names
        fr = [p for p in v["parts"] if p["name"] == "front_right"][0]
        assert fr["pos"] == [88.0, 88.0, 42.0]            # mm, forward/right/up
        assert fr["twin"]["xz"] == "front_left"
        assert v["symmetry"]["xz"]["max_error_mm"] == pytest.approx(0.0)

    def test_a_linked_drag_keeps_it_symmetric(self, project):
        port, app, pid = project
        s, v = call(port, app.token, "POST", f"/api/p/{pid}/move",
                    {"name": "front_right", "pos": [95, 80, 42], "link": "lr"})
        assert s == 200
        fl = [p for p in v["parts"] if p["name"] == "front_left"][0]
        assert fl["pos"] == [95.0, -80.0, 42.0]
        assert v["symmetry"]["xz"]["max_error_mm"] == pytest.approx(0.0)

    def test_an_unlinked_drag_is_measured_and_can_be_fixed(self, project):
        port, app, pid = project
        _, v = call(port, app.token, "POST", f"/api/p/{pid}/move",
                    {"name": "front_right", "pos": [91, 88, 42], "link": "off"})
        assert v["symmetry"]["xz"]["max_error_mm"] == pytest.approx(3.0)
        _, v = call(port, app.token, "POST", f"/api/p/{pid}/symmetrise", {"link": "lr"})
        assert v["symmetry"]["xz"]["max_error_mm"] == pytest.approx(0.0, abs=1e-6)

    def test_undo_and_redo(self, project):
        port, app, pid = project
        _, v = call(port, app.token, "GET", f"/api/p/{pid}/vehicle")
        start = [p for p in v["parts"] if p["name"] == "battery"][0]["pos"]
        call(port, app.token, "POST", f"/api/p/{pid}/move",
             {"name": "battery", "pos": [10, 0, -20], "link": "lr"})
        _, v = call(port, app.token, "POST", f"/api/p/{pid}/undo", {})
        bat = [p for p in v["parts"] if p["name"] == "battery"][0]
        assert bat["pos"] == pytest.approx(start)
        _, v = call(port, app.token, "POST", f"/api/p/{pid}/redo", {})
        bat = [p for p in v["parts"] if p["name"] == "battery"][0]
        assert bat["pos"][0] == pytest.approx(10.0)

    def test_a_preset_places_rotors_exactly(self, project):
        port, app, pid = project
        _, v = call(port, app.token, "POST", f"/api/p/{pid}/layout",
                    {"kind": "hex-x", "wheelbase_mm": 300, "height_mm": 40})
        rotors = [p for p in v["parts"] if p["kind"] == "rotor"]
        assert len(rotors) == 6
        assert v["wheelbase_mm"] == pytest.approx(300.0)
        assert v["yaw_balance"] == 0

    def test_add_edit_remove(self, project):
        port, app, pid = project
        s, v = call(port, app.token, "POST", f"/api/p/{pid}/add",
                    {"kind": "body", "name": "gps", "mass_g": 12,
                     "size_mm": [25, 25, 8], "pos": [0, 0, 30]})
        assert s == 200 and any(p["name"] == "gps" for p in v["parts"])
        _, v = call(port, app.token, "POST", f"/api/p/{pid}/update",
                    {"name": "gps", "fields": {"mass_g": 15}})
        assert [p for p in v["parts"] if p["name"] == "gps"][0]["mass_g"] == 15
        _, v = call(port, app.token, "POST", f"/api/p/{pid}/remove", {"name": "gps"})
        assert not any(p["name"] == "gps" for p in v["parts"])

    def test_nonsense_is_a_400_not_a_crash(self, project):
        port, app, pid = project
        assert call(port, app.token, "POST", f"/api/p/{pid}/update",
                    {"name": "battery", "fields": {"mass_g": -5}})[0] == 400
        assert call(port, app.token, "POST", f"/api/p/{pid}/move",
                    {"name": "nobody", "pos": [0, 0, 0]})[0] == 404

    def test_the_airframe_cannot_be_dragged(self, project):
        port, app, pid = project
        s, r = call(port, app.token, "POST", f"/api/p/{pid}/move",
                    {"name": "chassis", "pos": [5, 0, 0]})
        assert s == 400 and "structure step" in r["error"]


class TestProject:
    def test_edits_are_written_through(self, project):
        port, app, pid = project
        call(port, app.token, "POST", f"/api/p/{pid}/move",
             {"name": "battery", "pos": [12, 0, -20], "link": "lr"})
        again = Project(app.home / pid)
        assert again.vehicle.body("battery").position.x == pytest.approx(0.012)

    def test_new_projects_come_with_example_pilots(self, project):
        port, app, pid = project
        _, r = call(port, app.token, "GET", f"/api/p/{pid}/pilots")
        ids = [p["id"] for p in r["pilots"]]
        assert "gates" in ids and "pilots/figure_eight.py" in ids

    def test_the_aero_plan_comes_from_the_vehicle(self, project):
        port, app, pid = project
        _, r = call(port, app.token, "GET", f"/api/p/{pid}/envelope")
        assert r["kind"] == "multirotor" and len(r["points"]) == 6
        assert min(p["alpha_deg"] for p in r["points"]) < -40


class TestJobs:
    def test_an_optimiser_job_runs_and_its_pick_applies(self, project):
        port, app, pid = project
        s, job = call(port, app.token, "POST", f"/api/p/{pid}/jobs",
                      {"kind": "optimise",          # the app's default budget
                       "params": {"population": 40, "generations": 30}})
        assert s == 200 and job["status"] == "running"
        deadline = time.time() + 120
        while time.time() < deadline:
            _, r = call(port, app.token, "GET", f"/api/p/{pid}/jobs")
            j = [x for x in r["jobs"] if x["id"] == job["id"]][0]
            if j["status"] != "running":
                break
            time.sleep(0.5)
        assert j["status"] == "done", j["detail"]
        _, front = call(port, app.token, "GET", f"/api/p/{pid}/front?job={job['id']}")
        assert front["points"]
        s, v = call(port, app.token, "POST", f"/api/p/{pid}/apply",
                    {"job": job["id"], "index": 0})
        assert s == 200 and v["can_undo"]

    def test_a_grown_frame_can_be_used_by_the_steps_after_it(self, project):
        """SPEC-03: "Use this frame" adopted a vehicle whose mesh path had
        doubled, so the frame vanished from the view and Aerodynamics failed."""
        pytest.importorskip("skimage")
        pytest.importorskip("trimesh")
        port, app, pid = project
        _, job = call(port, app.token, "POST", f"/api/p/{pid}/jobs",
                      {"kind": "generate", "params": {
                          "mode": "mass", "mass": 40, "element": 0.008,
                          "thickness": 0.024, "iterations": 4, "min_feature": 16}})
        deadline = time.time() + 300
        while time.time() < deadline:
            _, r = call(port, app.token, "GET", f"/api/p/{pid}/jobs")
            j = [x for x in r["jobs"] if x["id"] == job["id"]][0]
            if j["status"] != "running":
                break
            time.sleep(0.5)
        assert j["status"] == "done", j["detail"]
        s, v = call(port, app.token, "POST", f"/api/p/{pid}/apply", {"job": job["id"]})
        assert s == 200 and v["has_generated_frame"]
        s, data = call(port, app.token, "GET", f"/api/p/{pid}/frame.stl")
        assert s == 200 and len(data) > 1000
        from camber.geometry.surface import airframe_surface
        assert airframe_surface(app.project(pid).vehicle).is_watertight

    def test_a_job_can_be_cancelled(self, project):
        port, app, pid = project
        _, job = call(port, app.token, "POST", f"/api/p/{pid}/jobs",
                      {"kind": "optimise", "params": {"population": 60, "generations": 400}})
        s, j = call(port, app.token, "POST", f"/api/p/{pid}/jobs/{job['id']}/cancel", {})
        assert s == 200 and j["status"] == "cancelled"


class TestImport:
    def test_a_framed_vehicle_brings_its_mesh(self, server):
        """The project is a folder: its airframe must not depend on the file
        it was imported from staying where it was."""
        import pathlib
        port, app = server
        spec = pathlib.Path(__file__).resolve().parent.parent / "examples" / "racer5_framed.json"
        p = app.import_spec(spec, "Imported")
        frame = [b for b in p.vehicle.all_bodies() if b.name == "frame"][0]
        src = pathlib.Path(frame.shape.source)
        assert src.exists() and p.root.resolve() in src.resolve().parents
        s, data = call(port, app.token, "GET", f"/api/p/{p.id}/frame.stl")
        assert s == 200 and len(data) > 10_000


class TestOptimiseRanksOnTheValidatedModel:
    """UI-08: the app's chart used the momentum-theory placeholder even with the
    propeller data on disk, and showed a 5-inch racer flying for 100 minutes.
    Since CAT-01 the CLI chooses the parts itself whenever the data is there,
    so the job passes no propeller unless the user narrows the choice."""

    OUT = {"front": "front.json", "best": "best.json"}

    def test_by_default_the_catalogue_chooses(self, tmp_path, monkeypatch):
        from camber.app.jobs import _args
        (tmp_path / "raw" / "UIUC-propDB").mkdir(parents=True)
        monkeypatch.setenv("CAMBER_USER_DATA", str(tmp_path))
        a = _args("optimise", "v.json", {}, self.OUT)
        assert "--prop" not in a and "--motor" not in a

    def test_a_narrowed_choice_is_passed_on(self):
        from camber.app.jobs import _args
        a = _args("optimise", "v.json", {"prop": "gwsdd_5x4.3,gwsdd_5x3",
                                          "motor": "emax-e32207-2400", "cells": "4-6"},
                  self.OUT)
        assert a[a.index("--prop") + 1] == "gwsdd_5x4.3,gwsdd_5x3"
        assert a[a.index("--motor") + 1] == "emax-e32207-2400"
        assert a[a.index("--cells") + 1] == "4-6"

    def test_without_it_the_placeholder_runs_and_the_front_says_so(self, tmp_path,
                                                                  monkeypatch):
        from camber.app.jobs import _args
        from camber.cli import main
        monkeypatch.setenv("CAMBER_USER_DATA", str(tmp_path))
        a = _args("optimise", "examples/quad5.toml",
                  {"population": 24, "generations": 10}, self.OUT)
        assert "--prop" not in a
        monkeypatch.chdir(tmp_path)
        assert main(a) == 0
        front = json.loads((tmp_path / "front.json").read_text())
        assert front["endurance_model"].startswith("momentum")

    @pytest.mark.parametrize("params", [{"prop": "x; rm -rf ~"}, {"motor": "a b"},
                                        {"cells": "4; reboot"}])
    def test_a_part_cannot_carry_anything_else(self, params):
        from camber.app.jobs import _args
        with pytest.raises(ValueError):
            _args("optimise", "v.json", params, self.OUT)
