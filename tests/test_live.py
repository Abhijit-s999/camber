"""Flying in the app: a simulation paced to the clock, steered over HTTP.

The physics is tested elsewhere. What matters here is that the flight starts,
takes sticks, reports what it is doing, and stops -- including when nobody is
watching any more.
"""

import threading
import time

import pytest

mujoco = pytest.importorskip("mujoco")

from camber.app import live as live_mod                    # noqa: E402
from camber.app.live import Flight                         # noqa: E402
from camber.app.server import serve                        # noqa: E402
from camber.spec import io                                 # noqa: E402
from tests.test_app import call                            # noqa: E402

import pathlib                                             # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def racer():
    return io.load(ROOT / "examples" / "racer5.toml")


@pytest.fixture
def project(tmp_path):
    """A served project, as the app serves it."""
    srv, app = serve(0, home=tmp_path / "Camber")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s, r = call(srv.server_address[1], app.token, "POST", "/api/projects",
                {"name": "Live test", "template": "racer5"})
    yield srv.server_address[1], app, r["id"]
    app.land()
    srv.shutdown()


@pytest.fixture
def flight(racer):
    f = Flight(racer, pilot="manual", mode="assist")
    yield f
    f.stop()


class TestPacing:
    def test_it_flies_in_real_time(self, flight):
        """Not as fast as it can: a second of simulation per second, so what
        you see is what the vehicle would do."""
        flight.start()
        time.sleep(1.0)
        flight.touch()
        t = flight.snapshot()["t"]
        assert 0.7 < t < 1.3

    def test_it_reports_rate_and_headroom_apart(self, flight):
        flight.start()
        time.sleep(0.8)
        flight.touch()
        s = flight.snapshot()
        assert 0.8 < s["rate"] < 1.2            # kept up with the clock
        assert s["headroom"] > 1.0              # and had time to spare

    def test_falling_behind_drops_time_rather_than_banking_it(self, flight):
        """If the machine stalls, the flight rejoins the present. Banking the
        debt would make it sprint through the backlog in fast-forward."""
        flight.start()
        time.sleep(0.3)
        with flight.lock:                       # hold the simulation still
            time.sleep(0.8)
        time.sleep(0.3)
        flight.touch()
        assert flight.snapshot()["t"] < 1.1     # not the full 1.4 s of wall clock


class TestSticks:
    def test_sticks_reach_the_vehicle(self, flight):
        flight.start()
        flight.set_sticks(0.0, 0.0, 0.0, 1.0)   # full throttle
        time.sleep(1.2)
        flight.touch()
        assert flight.snapshot()["altitude"] > 1.0

    def test_disarming_stops_the_motors(self, flight):
        flight.start()
        flight.set_sticks(0.0, 0.0, 0.0, 1.0)
        time.sleep(0.4)
        flight.arm(False)
        time.sleep(0.2)
        flight.touch()
        s = flight.snapshot()
        assert not s["armed"] and sum(s["thrust"]) < 1e-6

    def test_modes_switch_in_flight(self, flight):
        flight.start()
        flight.set_mode("acro")
        assert flight.snapshot()["mode"] == "acro"
        with pytest.raises(ValueError):
            flight.set_mode("hover-please")

    def test_reset_puts_it_back(self, flight):
        flight.start()
        flight.set_sticks(0.0, 1.0, 0.0, 0.9)
        time.sleep(1.0)
        flight.reset()
        s = flight.snapshot()
        assert s["t"] < 0.1 and abs(s["pos"][0]) < 0.1 and s["gates_passed"] == 0


class TestWatching:
    def test_an_unwatched_flight_shuts_itself_down(self, racer, monkeypatch):
        """Nobody is going to notice a forgotten tab; the machine will."""
        monkeypatch.setattr(live_mod, "IDLE_STOP", 0.3)
        f = Flight(racer, pilot="hover")
        f.start()
        time.sleep(0.15)
        assert f.running
        time.sleep(0.6)
        assert not f.running

    def test_the_racing_pilot_flies_itself(self, racer):
        f = Flight(racer, pilot="gates", speed=14.0)
        try:
            f.start()
            for _ in range(40):
                time.sleep(0.1)
                f.touch()
                if f.snapshot()["gates_passed"]:
                    break
            s = f.snapshot()
            assert s["gates_passed"] >= 1 and s["gates_total"] == 8
            assert s["mode"] is None and s["sticks"] is None
        finally:
            f.stop()


class TestOverHTTP:
    def test_a_flight_starts_flies_and_stops(self, project):
        port, app, pid = project
        s, r = call(port, app.token, "POST", f"/api/p/{pid}/live",
                    {"do": "start", "pilot": "manual", "mode": "assist"})
        assert s == 200 and len(r["scene"]["gates"]) == 8
        assert r["state"]["running"] and r["state"]["mode"] == "assist"
        for _ in range(20):                     # as the browser does, every frame
            s, r = call(port, app.token, "POST", f"/api/p/{pid}/live",
                        {"do": "input", "sticks": [0, 0, 0, 1.0]})
            assert s == 200
            time.sleep(0.05)
        assert r["state"]["altitude"] > 0.3
        s, r = call(port, app.token, "POST", f"/api/p/{pid}/live", {"do": "stop"})
        assert s == 200 and r["stopped"]
        s, r = call(port, app.token, "GET", f"/api/p/{pid}/live")
        assert s == 404

    def test_starting_again_replaces_the_first_flight(self, project):
        port, app, pid = project
        call(port, app.token, "POST", f"/api/p/{pid}/live", {"do": "start"})
        first = app.flight(pid)
        call(port, app.token, "POST", f"/api/p/{pid}/live", {"do": "start"})
        assert app.flight(pid) is not first and not first.running

    def test_it_needs_the_token(self, project):
        port, app, pid = project
        s, _ = call(port, None, "POST", f"/api/p/{pid}/live", {"do": "start"})
        assert s == 401

    def test_a_project_pilot_file_can_fly(self, project):
        port, app, pid = project
        s, r = call(port, app.token, "POST", f"/api/p/{pid}/live",
                    {"do": "start", "pilot": "pilots/figure_eight.py"})
        assert s == 200 and r["state"]["running"]
        time.sleep(0.4)
        s, r = call(port, app.token, "POST", f"/api/p/{pid}/live", {"do": "poll"})
        assert s == 200 and r["state"]["t"] > 0.1

    def test_a_missing_pilot_is_an_error_not_a_crash(self, project):
        port, app, pid = project
        s, r = call(port, app.token, "POST", f"/api/p/{pid}/live",
                    {"do": "start", "pilot": "pilots/nope.py"})
        assert s == 400 and "nope" in r["error"]


class TestKeepAlive:
    """The flight view sends 60 requests a second down one connection."""

    def test_a_refused_request_does_not_poison_the_connection(self, project):
        import http.client
        import json as _json
        port, app, pid = project
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        def send(token, body):
            h = {"Host": f"127.0.0.1:{port}", "Content-Type": "application/json"}
            if token:
                h["X-Camber-Token"] = token
            c.request("POST", f"/api/p/{pid}/live", _json.dumps(body), h)
            r = c.getresponse()
            return r.status, _json.loads(r.read() or b"{}")
        # Refused: body never reaches a handler...
        assert send(None, {"do": "start", "pilot": "manual"})[0] == 401
        # ...but the next request on the same socket must still be understood.
        s, r = send(app.token, {"do": "start", "pilot": "manual"})
        assert s == 200 and r["state"]["running"]
        s, r = send(app.token, {"do": "input", "sticks": [0, 0, 0, 0.6]})
        assert s == 200 and r["state"]["running"]
        c.close()

    def test_many_requests_share_one_connection(self, project):
        import http.client
        import json as _json
        port, app, pid = project
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        h = {"Host": f"127.0.0.1:{port}", "X-Camber-Token": app.token,
             "Content-Type": "application/json"}
        c.request("POST", f"/api/p/{pid}/live", _json.dumps({"do": "start"}), h)
        assert c.getresponse().read()
        sock = c.sock
        for _ in range(50):
            c.request("POST", f"/api/p/{pid}/live",
                      _json.dumps({"do": "input", "sticks": [0, 0, 0, 0.5]}), h)
            r = c.getresponse()
            r.read()                              # each body, or the next
            assert r.status == 200                # request has one queued ahead
        assert c.sock is sock                     # never reconnected
        c.close()
