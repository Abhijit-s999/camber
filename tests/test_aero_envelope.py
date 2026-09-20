"""Generative aero testing: the conditions come from the vehicle."""

import math
import pathlib

import numpy as np
import pytest

from camber.cfd.envelope import envelope, kind_of, trim_speed
from camber.spec import io

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def racer():
    return io.load(ROOT / "examples" / "racer5.toml")


@pytest.fixture
def cruiser():
    return io.load(ROOT / "examples" / "quad5.toml")


class TestEnvelope:
    def test_a_racer_tilts_further_than_an_endurance_quad(self, racer, cruiser):
        assert (min(p.alpha_deg for p in envelope(racer))
                < min(p.alpha_deg for p in envelope(cruiser)))

    def test_the_tilt_is_capped_by_thrust_to_weight(self, racer):
        """The steepest tilt at which height can be held with 15% in reserve
        for control -- a rotor at 100% has nothing left to steer with."""
        for t in racer.thrusters:
            t.static_thrust_n = 1.5
        mg = racer.mass_properties().mass * 9.80665
        limit = math.degrees(math.acos(mg / (0.85 * 4 * 1.5)))
        assert -min(p.alpha_deg for p in envelope(racer)) <= limit + 1e-9

    def test_each_point_is_a_level_flight_trim(self, racer):
        from camber.sim.aero import BodyDrag
        d = BodyDrag.estimate(racer)
        mg = racer.mass_properties().mass * 9.80665
        for p in envelope(racer):
            tilt = math.radians(-p.alpha_deg)
            assert p.thrust * math.cos(tilt) == pytest.approx(mg, rel=1e-3)
            a = math.radians(p.alpha_deg)
            K = d.cda[0] * abs(math.cos(a)) ** 3 + d.cda[2] * abs(math.sin(a)) ** 3
            drag = 0.5 * 1.225 * p.speed ** 2 * K
            assert p.thrust * math.sin(tilt) == pytest.approx(drag, rel=2e-3)

    def test_the_relative_wind_comes_from_ahead_and_onto_the_top(self, racer):
        """The old hover-only case set the freestream to (+V, 0, 0), air
        arriving from behind. Flying forward nose-down, the air comes from ahead
        (-x) and strikes the top, moving toward +z in NED."""
        p = [p for p in envelope(racer)][-1]
        u = p.relative_wind_ned()
        assert u[0] < 0 and u[2] > 0
        assert np.linalg.norm(u) == pytest.approx(p.speed)

    def test_a_vehicle_without_rotors_gets_an_alpha_sweep(self, racer):
        racer.thrusters.clear()
        assert kind_of(racer) == "fixed_wing"
        alphas = [p.alpha_deg for p in envelope(racer, cruise=18.0)]
        assert min(alphas) < 0 < max(alphas) and max(alphas) >= 12


class TestTabulation:
    def test_moments_are_moved_to_the_cg(self, racer):
        """OpenFOAM reports moments about (0,0,0); the simulator applies them
        about the CG. A pure force through the CG must tabulate as zero Cm."""
        from camber.cfd.sweep import tabulate
        cg = np.asarray(racer.mass_properties().cg.as_list())
        F = np.array([-1.0, 0.0, 4.0])
        m_origin = np.cross(cg, F)                  # a force acting at the CG
        res = [{"point": {"alpha_deg": -30.0, "speed": 20.0},
                "force_ned": F.tolist(), "moment_ned_origin": m_origin.tolist()}]
        t = tabulate(racer, res)
        assert t["Cm"][0] == pytest.approx(0.0, abs=1e-12)

    def test_coefficients_are_in_flu_axes(self, racer):
        from camber.cfd.sweep import tabulate
        res = [{"point": {"alpha_deg": -30.0, "speed": 20.0},
                "force_ned": [0.0, 0.0, 5.0], "moment_ned_origin": [0, 0, 0]}]
        t = tabulate(racer, res)
        assert t["CZ"][0] < 0          # +z NED (down) is -z FLU


class TestFarField:
    def _case(self, tmp_path, racer, u):
        from camber.cfd.case import CaseBuilder, CaseConfig
        stl = tmp_path / "b.stl"
        stl.write_text("solid b\nendsolid b\n")
        thrusts = {t.name: 1.0 for t in racer.thrusters}
        return CaseBuilder(racer, thrusts, CaseConfig(u_inf=u)).build(tmp_path / "c", stl)

    def test_forward_flight_uses_freestream_boundaries(self, tmp_path, racer):
        """totalPressure on every face is right for hover and wrong in a
        freestream: it pinned static pressure differently at inflow and
        outflow, stalled the residual at 0.19 and inflated the body force
        2.7x on the first forward-flight case."""
        case = self._case(tmp_path, racer, (-20.0, 0.0, 10.0))
        assert "freestreamVelocity" in (case / "0" / "U").read_text()
        assert "freestreamPressure" in (case / "0" / "p").read_text()

    def test_hover_keeps_the_open_boundaries(self, tmp_path, racer):
        case = self._case(tmp_path, racer, (0.0, 0.0, 0.0))
        assert "totalPressure" in (case / "0" / "p").read_text()


class TestTableOrder:
    def test_speeds_stay_with_their_angles(self, racer):
        """Rows sort by angle; the speed column must sort with them."""
        from camber.cfd.sweep import tabulate
        res = [{"point": {"alpha_deg": a, "speed": v},
                "force_ned": [0.0, 0.0, 1.0], "moment_ned_origin": [0, 0, 0]}
               for a, v in [(-10.0, 16.8), (-65.0, 30.6), (-40.0, 25.0)]]
        t = tabulate(racer, res)
        assert list(zip(t["alpha_deg"], t["speeds"])) == [(-65.0, 30.6), (-40.0, 25.0),
                                                           (-10.0, 16.8)]
