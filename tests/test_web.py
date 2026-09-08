"""The local viewport's API.

The browser owns the view and nothing else. Every number it shows comes from
the same Python the CLI and optimiser use, so these tests mostly check that the
API cannot quietly grow a second implementation of the engine.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from camber.spec import io
from camber.web.server import Handler, Session, vehicle_json

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "quad5.toml"


@pytest.fixture
def vehicle():
    return io.load(EXAMPLE)


@pytest.fixture
def server(vehicle):
    Handler.session = Session(vehicle)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return r.status, json.loads(r.read())


def post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestPayload:
    def test_carries_everything_the_view_must_not_compute(self, vehicle):
        d = vehicle_json(vehicle)
        for key in ("mass", "control", "hover", "screen", "constraints",
                    "bodies", "thrusters", "features"):
            assert key in d

    def test_numbers_match_the_engine_exactly(self, vehicle):
        """No second implementation in JavaScript."""
        from camber.control import controllability, solve_hover
        d = vehicle_json(vehicle)
        assert d["mass"]["total_kg"] == pytest.approx(
            vehicle.mass_properties().mass)
        assert d["control"]["rank"] == controllability(vehicle).rank
        assert d["hover"]["feasible"] == solve_hover(vehicle).feasible

    def test_is_json_serialisable(self, vehicle):
        """numpy scalars leak into JSON and break the client silently."""
        json.dumps(vehicle_json(vehicle))

    def test_frame_members_are_marked(self, vehicle):
        from camber.generate import generate_frame
        vehicle.frame.members = generate_frame(vehicle).members
        d = vehicle_json(vehicle)
        assert any(b["frame"] for b in d["bodies"])
        assert any(not b["frame"] for b in d["bodies"])

    def test_camera_carries_its_boresight(self, vehicle):
        d = vehicle_json(vehicle)
        cam = next(b for b in d["bodies"] if b["kind"] == "camera")
        assert cam["boresight"][2] < 0, "z is down, so a raised camera looks up"


class TestRoutes:
    def test_serves_the_page(self, server):
        with urllib.request.urlopen(server + "/", timeout=5) as r:
            assert r.status == 200 and b"<!doctype html>" in r.read()[:40]

    def test_vehicle_endpoint(self, server):
        status, d = get(server, "/api/vehicle")
        assert status == 200 and d["name"] == "quad5"

    def test_favicon_is_not_an_error(self, server):
        """The browser asks on every load; a 404 is noise, not information."""
        with urllib.request.urlopen(server + "/favicon.ico", timeout=5) as r:
            assert r.status == 204

    def test_unknown_route(self, server):
        try:
            get(server, "/api/nope")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404


class TestMove:
    def test_moving_a_rotor_redistributes_hover_thrust(self, server):
        _, before = get(server, "/api/vehicle")
        status, after = post(server, "/api/move",
                             {"name": "front_right",
                              "position": [0.16, 0.16, -0.045]})
        assert status == 200
        b = [t["thrust"] for t in before["thrusters"]]
        a = [t["thrust"] for t in after["thrusters"]]
        assert a != b, "moving an arm must change the trim"
        assert a[0] < b[0], "a rotor moved outboard carries less of the weight"

    def test_moving_a_payload_shifts_the_cg(self, server):
        _, before = get(server, "/api/vehicle")
        _, after = post(server, "/api/move",
                        {"name": "battery", "position": [0.05, 0.0, 0.018]})
        assert after["mass"]["cg"][0] > before["mass"]["cg"][0]
        assert after["mass"]["total_kg"] == pytest.approx(
            before["mass"]["total_kg"]), "moving mass must not create it"

    def test_move_updates_the_interference_screen(self, server):
        _, before = get(server, "/api/vehicle")
        _, after = post(server, "/api/move",
                        {"name": "front_right", "position": [0.03, 0.03, -0.045]})
        assert after["screen"]["overlap"] > before["screen"]["overlap"]

    def test_unknown_part_is_404_not_a_crash(self, server):
        status, body = post(server, "/api/move",
                            {"name": "ghost", "position": [0, 0, 0]})
        assert status == 404 and "ghost" in body["error"]

    def test_malformed_position_is_400(self, server):
        status, _ = post(server, "/api/move",
                         {"name": "battery", "position": ["a", "b", "c"]})
        assert status == 400


class TestSession:
    def test_move_mutates_the_session_vehicle(self, vehicle):
        s = Session(vehicle)
        s.move("battery", [0.04, 0.0, 0.018])
        assert vehicle.body("battery").position.x == pytest.approx(0.04)

    def test_save_round_trips(self, vehicle, tmp_path):
        s = Session(vehicle)
        s.move("battery", [0.04, 0.0, 0.018])
        out = tmp_path / "v.json"
        s.save(out)
        assert io.load(out).body("battery").position.x == pytest.approx(0.04)
