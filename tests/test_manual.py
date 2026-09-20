"""Flying it by hand: what each stick is supposed to do, checked by doing it.

The conventions are the pilot's -- roll right positive, pitch forward (nose
down) positive, yaw right positive -- and getting one of them backwards makes
a vehicle that flies into things, so each is checked against the direction the
vehicle actually moves.
"""

import math
import pathlib

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from camber.sim import Simulation                          # noqa: E402
from camber.sim.manual import MODES, Manual, Sticks, expo  # noqa: E402
from camber.spec import io                                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def racer():
    return io.load(ROOT / "examples" / "racer5.toml")


def fly(vehicle, sticks, seconds=2.0, mode="assist", start=12.0, **kw):
    """Hold the sticks there for a while; hand back the state it ends in."""
    sim = Simulation(vehicle)
    pilot = Manual(mode, **kw)
    state = sim.reset(position=(0.0, 0.0, start))
    pilot.reset(sim)
    pilot.sticks = Sticks(*sticks) if not isinstance(sticks, Sticks) else sticks
    for _ in range(int(seconds / sim.dt)):
        state = sim.step(pilot.update(state, sim))
    return state, sim, pilot


class TestSticks:
    def test_centre_sticks_hold_position(self, racer):
        state, _, _ = fly(racer, (0, 0, 0, 0.5), seconds=4.0, start=12.0)
        assert np.linalg.norm(state.position - np.array([0, 0, 12.0])) < 0.15

    def test_pitch_forward_goes_forward(self, racer):
        state, _, _ = fly(racer, (0, 1, 0, 0.5))
        assert state.position[0] > 3.0 and abs(state.position[1]) < 0.5

    def test_pitch_back_goes_back(self, racer):
        state, _, _ = fly(racer, (0, -1, 0, 0.5))
        assert state.position[0] < -3.0

    def test_roll_right_goes_right(self, racer):
        """Right is -y: the world is x forward, y LEFT, z up."""
        state, _, _ = fly(racer, (1, 0, 0, 0.5))
        assert state.position[1] < -3.0 and abs(state.position[0]) < 0.5

    def test_throttle_up_climbs_and_down_descends(self, racer):
        up, _, _ = fly(racer, (0, 0, 0, 1.0))
        down, _, _ = fly(racer, (0, 0, 0, 0.0))
        assert up.position[2] > 13.0 > down.position[2]

    def test_yaw_right_turns_right(self, racer):
        """Half a second only: at 180 deg/s, two seconds is a full turn and
        the heading comes back to where it started."""
        state, _, _ = fly(racer, (0, 0, 1, 0.5), seconds=0.5)
        assert math.radians(-110) < state.euler[2] < math.radians(-40)

    def test_it_stops_when_the_sticks_centre(self, racer):
        sim = Simulation(racer)
        pilot = Manual("assist")
        state = sim.reset(position=(0.0, 0.0, 12.0))
        pilot.reset(sim)
        pilot.sticks = Sticks(0, 1, 0, 0.5)
        for _ in range(int(2.0 / sim.dt)):
            state = sim.step(pilot.update(state, sim))
        moving = float(np.linalg.norm(state.velocity))
        pilot.sticks = Sticks(0, 0, 0, 0.5)
        for _ in range(int(4.0 / sim.dt)):
            state = sim.step(pilot.update(state, sim))
        stopped = state.position.copy()
        for _ in range(int(3.0 / sim.dt)):      # and it stays where it stopped
            state = sim.step(pilot.update(state, sim))
        assert moving > 5.0
        assert float(np.linalg.norm(state.velocity)) < 0.3
        assert float(np.linalg.norm(state.position - stopped)) < 1.0


class TestModes:
    def test_angle_mode_banks_and_levels(self, racer):
        state, sim, pilot = fly(racer, (1, 0, 0, 0.55), seconds=1.0, mode="angle")
        assert math.degrees(state.euler[0]) > 25        # rolled right, near the 35 limit
        pilot.sticks = Sticks(0, 0, 0, 0.55)            # let go
        for _ in range(int(1.0 / sim.dt)):
            state = sim.step(pilot.update(state, sim))
        assert abs(math.degrees(state.euler[0])) < 3    # and it levels itself

    def test_acro_mode_holds_a_roll_rate(self, racer):
        """Averaged once it has settled: the first tenth of a second is the
        loop getting there, and the stick asks for a rate, not an angle."""
        sim = Simulation(racer)
        pilot = Manual("acro")
        state = sim.reset(position=(0.0, 0.0, 30.0))
        pilot.reset(sim)
        pilot.sticks = Sticks(1, 0, 0, 0.55)
        rates = []
        for i in range(int(1.0 / sim.dt)):
            state = sim.step(pilot.update(state, sim))
            if sim.t > 0.4:
                rates.append(float(state.rates[0]))
        assert abs(np.mean(rates) - pilot.max_stick_rate) < 0.1 * pilot.max_stick_rate

    def test_acro_does_not_level_itself(self, racer):
        """The whole point of acro: let go inverted and it stays inverted."""
        sim = Simulation(racer)
        pilot = Manual("acro")
        state = sim.reset(position=(0.0, 0.0, 12.0))
        pilot.reset(sim)
        pilot.sticks = Sticks(1, 0, 0, 0.55)
        while abs(state.euler[0]) < math.radians(170) and sim.t < 3.0:
            state = sim.step(pilot.update(state, sim))
        pilot.sticks = Sticks(0, 0, 0, 0.3)
        for _ in range(int(0.4 / sim.dt)):
            state = sim.step(pilot.update(state, sim))
        assert state.rotation[2, 2] < -0.5              # still upside down

    def test_assist_flies_gently(self, racer):
        assert Manual("assist").max_tilt < Manual("acro").max_tilt


class TestThrottleCurve:
    def test_mid_stick_is_a_hover(self, racer):
        sim = Simulation(racer)
        pilot = Manual("angle")
        pilot.reset(sim)
        assert pilot.collective(0.5) == pytest.approx(sim.weight)

    def test_the_ends_are_nothing_and_everything(self, racer):
        sim = Simulation(racer)
        pilot = Manual("angle")
        pilot.reset(sim)
        assert pilot.collective(0.0) == pytest.approx(0.0)
        assert pilot.collective(1.0) == pytest.approx(float(np.sum(sim.max_thrust)))

    def test_a_heavy_vehicle_hovers_at_mid_stick_too(self, racer):
        """The curve is what makes one stick fly any vehicle: a 11:1 racer
        would otherwise hover at 8.5% of a linear stick."""
        sim = Simulation(racer)
        pilot = Manual("angle")
        pilot.reset(sim)
        pilot.hover, pilot.max_total = 8.0, 10.0          # a barely-flying lifter
        assert pilot.collective(0.5) == pytest.approx(8.0)
        assert pilot.collective(1.0) == pytest.approx(10.0)


class TestSafety:
    def test_disarmed_motors_are_off(self, racer):
        sim = Simulation(racer)
        pilot = Manual("acro")
        state = sim.reset()
        pilot.reset(sim)
        pilot.armed = False
        pilot.sticks = Sticks(1, 1, 1, 1.0)
        for _ in range(50):
            state = sim.step(pilot.update(state, sim))
        assert float(np.sum(state.thrust)) < 1e-6

    def test_sticks_are_clipped(self):
        s = Sticks(5.0, -9.0, 0.5, 40.0).clipped()
        assert (s.roll, s.pitch, s.yaw, s.throttle) == (1.0, -1.0, 0.5, 1.0)

    def test_expo_softens_the_middle_and_keeps_the_ends(self):
        assert abs(expo(0.5)) < 0.5 and expo(1.0) == pytest.approx(1.0)
        assert expo(0.0) == 0.0 and expo(-1.0) == pytest.approx(-1.0)

    def test_every_mode_flies(self, racer):
        for mode in MODES:
            state, _, _ = fly(racer, (0, 0, 0, 0.5), seconds=1.5, mode=mode)
            assert state.position[2] > 2.0, f"{mode} fell out of the sky"

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError):
            Manual("sport")
