"""The simulator flies the vehicle Camber designed -- checked against physics
that can be done by hand, not against itself."""

import math
import pathlib

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from camber.control import solve_hover                        # noqa: E402
from camber.sim import Simulation, frames                     # noqa: E402
from camber.sim.sim import Command                            # noqa: E402
from camber.sim.aero import AeroTable, BodyDrag               # noqa: E402
from camber.sim.course import Course, Gate                    # noqa: E402
from camber.sim.flight import Cascade, GateRacer, Mixer       # noqa: E402
from camber.sim.motors import MotorBank                       # noqa: E402
from camber.spec import io                                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def racer():
    return io.load(ROOT / "examples" / "racer5.toml")


@pytest.fixture
def still(racer):
    """No aerodynamics: tests about rigid-body physics should not also be
    tests of a drag estimate."""
    return Simulation(racer, aero=BodyDrag([0.0, 0.0, 0.0]))


class TestFrames:
    def test_ned_to_flu_is_a_rotation_not_a_mirror(self):
        assert np.linalg.det(frames.NED_TO_FLU) == pytest.approx(1.0)

    def test_it_is_its_own_inverse(self):
        assert np.allclose(frames.NED_TO_FLU @ frames.NED_TO_FLU, np.eye(3))

    def test_up_is_up(self):
        assert frames.vec([0.0, 0.0, -1.0]) == pytest.approx([0, 0, 1])

    def test_inertia_keeps_its_eigenvalues(self):
        I = np.array([[3.0, 0.2, -0.1], [0.2, 4.0, 0.3], [-0.1, 0.3, 5.0]])
        assert np.allclose(np.linalg.eigvalsh(frames.tensor(I)),
                           np.linalg.eigvalsh(I))


class TestTheModelIsCambers:
    def test_mass_is_exact(self, racer, still):
        b = still.model.body("airframe").id
        assert still.model.body_mass[b] == pytest.approx(
            racer.mass_properties().mass, rel=1e-12)

    def test_inertia_is_exact(self, still):
        b = still.model.body("airframe").id
        assert np.allclose(np.sort(still.model.body_inertia[b]),
                           np.sort(np.linalg.eigvalsh(still.inertia)), rtol=1e-9)

    def test_the_allocation_matrix_is_the_optimisers(self, racer, still):
        from camber.control import allocation_matrix
        assert np.allclose(still.allocation,
                           frames.wrench(allocation_matrix(racer)))


class TestRigidBodyPhysics:
    def test_free_fall_matches_the_integrators_own_sum(self, still):
        """Not 1/2 g t^2: MuJoCo integrates in steps, and after N steps of dt
        the drop is g dt^2 N(N-1)/2. Matching that to the micron is the proof
        gravity and units are right; matching 1/2 g t^2 to 0.4% is not."""
        s0 = still.reset(position=(0, 0, 5))
        n = 250
        for _ in range(n):
            s = still.step(np.zeros(4))
        want = 9.80665 * still.dt ** 2 * n * (n - 1) / 2
        assert s0.position[2] - s.position[2] == pytest.approx(want, rel=1e-6)

    def test_cambers_hover_trim_holds_still(self, racer, still):
        """The agreement that matters: the optimiser's static trim, flown
        dynamically, must not move. Equal thrust on every rotor does NOT hover
        this vehicle -- its CG is 1.8 mm aft -- and a test that assumed it would
        mistook correct physics for a bug."""
        u = MotorBank.command_for(solve_hover(racer).thrusts, still.max_thrust)
        still.reset(position=(0, 0, 5), spin=u)
        for _ in range(1000):
            s = still.step(u)
        assert np.abs(s.velocity).max() < 1e-4
        assert np.abs(s.rates).max() < 1e-4

    def test_yaw_goes_the_way_the_allocation_says(self, still):
        uh = math.sqrt(still.weight / still.max_thrust.sum())
        still.reset(position=(0, 0, 5), spin=uh)
        u = np.full(4, uh)
        u[[0, 2]] += 0.03
        u[[1, 3]] -= 0.03
        predicted = still.allocation[5] @ (still.max_thrust * u ** 2)
        for _ in range(150):
            s = still.step(u)
        assert np.sign(s.rates[2]) == np.sign(predicted) != 0


class TestMotors:
    def test_thrust_goes_as_speed_squared(self):
        m = MotorBank([10.0])
        m.reset(0.5)
        assert m.step([0.5], 0.001)[0] == pytest.approx(2.5)

    def test_the_lag_is_the_exact_discrete_solution(self):
        """Stable for any dt, including one longer than tau -- where an Euler
        step would overshoot."""
        m = MotorBank([1.0], tau=0.03)
        m.step([1.0], 0.09)
        assert m.speed[0] == pytest.approx(1.0 - math.exp(-3.0))

    def test_command_for_inverts_the_square(self):
        t = np.array([0.25, 4.0])
        assert MotorBank.command_for(t, np.array([1.0, 16.0])) == pytest.approx([0.5, 0.5])


class TestTiltRotors:
    def test_a_tilt_servo_has_lag_and_changes_the_real_wrench(self, racer):
        """The allocation must follow the physical servo, not a static spec."""
        from camber.spec.units import Vec3
        from camber.spec.vehicle import TiltRotor

        old = racer.thrusters[0]
        racer.thrusters[0] = TiltRotor(
            name=old.name, mass=old.mass, placement=old.placement,
            thrust_axis=old.thrust_axis, radius=old.radius, pitch=old.pitch,
            blades=old.blades, ducted=old.ducted, spin=old.spin,
            static_thrust_n=old.static_thrust_n,
            torque_coefficient=old.torque_coefficient,
            tilt_axis=Vec3(0, 1, 0), tilt_min=0, tilt_max=np.pi / 2,
            tilt_tau=0.04,
        )
        sim = Simulation(racer, aero=BodyDrag([0, 0, 0]), latency=0)
        before = sim.allocation.copy()
        for _ in range(120):
            sim.step(Command(np.ones(4), tilts=np.array([np.pi / 2])))
        assert sim.tilts.angle[0] == pytest.approx(np.pi / 2, abs=0.01)
        assert sim.allocation[0, 0] < -0.99
        assert not np.allclose(sim.allocation, before)


class TestMixer:
    def test_a_reachable_wrench_is_produced_exactly(self, still):
        mix = Mixer(still.allocation, still.max_thrust)
        want = np.array([still.weight, 0.01, -0.02, 0.003])
        t = mix.thrusts(want[0], want[1:])
        assert still.allocation[[2, 3, 4, 5]] @ t == pytest.approx(want, abs=1e-9)

    def test_it_gives_up_yaw_before_roll(self, still):
        mix = Mixer(still.allocation, still.max_thrust)
        t = mix.thrusts(0.9 * still.max_thrust.sum(), np.array([0.3, 0.0, 0.5]))
        got = still.allocation[[3, 4, 5]] @ t
        assert got[0] == pytest.approx(0.3, rel=0.05)       # roll kept
        assert abs(got[2]) < 0.5                            # yaw shed
        assert (t >= -1e-9).all() and (t <= still.max_thrust + 1e-9).all()


class TestAero:
    def test_drag_opposes_motion(self):
        d = BodyDrag([0.01, 0.02, 0.03])
        f, _ = d.force(np.array([10.0, -5.0, 2.0]))
        assert (np.sign(f) == -np.sign([10.0, -5.0, 2.0])).all()

    def test_the_estimate_uses_the_footprint_not_the_bounding_box(self, racer):
        d = BodyDrag.estimate(racer)
        assert d.cda[2] == pytest.approx(
            1.05 * sum(b.shape.planform_area() for b in racer.all_bodies()))

    def test_forward_flight_nose_down_is_negative_alpha(self):
        """Aircraft convention: a multirotor pitched forward meets the air on
        its top. v in FLU body axes has +z when the nose is down."""
        seen = []

        class Probe(AeroTable):
            def force(self, v):
                seen.append(math.degrees(math.atan2(-v[2], v[0])))
                return super().force(v)

        t = Probe([-60, 0, 20], [0, 0, 0], [0, 0, 0], [0, 0, 0], area=1, length=1)
        th = math.radians(40)
        t.force(20 * np.array([math.cos(th), 0.0, math.sin(th)]))
        assert seen[0] == pytest.approx(-40.0)

    def test_extrapolation_is_counted_not_hidden(self):
        t = AeroTable([-10, 10], [0, 0], [0, 0], [0, 0], area=1, length=1)
        t.force(np.array([10.0, 0.0, 30.0]))                 # ~ -72 deg
        assert t.extrapolated == 1


class TestFlying:
    def test_it_holds_a_point(self, racer):
        sim = Simulation(racer)
        c = Cascade()
        s = sim.reset()
        c.reset(sim)
        c.target = np.array([0.0, 0.0, 1.0])
        errs = []
        for i in range(2500):
            s = sim.step(c.update(s, sim))
            if i > 1500:
                errs.append(np.linalg.norm(s.position - c.target))
        assert max(errs) < 0.1 and not sim.crashed

    def test_it_flies_the_gate_course(self, racer):
        from camber.sim.run import fly
        course = Course.oval()
        sim = Simulation(racer, course=course)
        pilot = GateRacer(course, speed=9.0)
        log = fly(sim, pilot, 12.0)
        assert len(pilot.passed) == len(course.gates)
        assert not log.crashed

    def test_a_gate_is_crossed_only_through_its_opening(self):
        g = Gate([0, 0, 1.5], [1, 0, 0])
        assert g.crossed(np.array([0.1, 0.2, 1.4]))
        assert not g.crossed(np.array([-0.1, 0.0, 1.5]))      # not yet
        assert not g.crossed(np.array([0.1, 2.0, 1.5]))       # round the side


class TestCrashIsContact:
    """Upside down is a flip, not a crash. An earlier rule said otherwise and
    declared every flip a crash at 90 degrees of roll."""

    def test_inverted_in_the_air_is_not_a_crash(self, racer):
        sim = Simulation(racer)
        sim.reset(position=(0, 0, 5), quaternion=(0.0, 1.0, 0.0, 0.0))
        sim.step(np.zeros(4))
        assert not sim.crashed

    def test_hitting_the_floor_fast_is(self, racer):
        sim = Simulation(racer)
        sim.reset(position=(0, 0, 0.3), velocity=(0, 0, -8.0))
        for _ in range(100):
            sim.step(np.zeros(4))
            if sim.crashed:
                break
        assert sim.crashed


class TestPilots:
    @pytest.mark.parametrize("name", ["figure_eight.py", "flip.py"])
    def test_the_example_pilots_fly_without_crashing(self, racer, name):
        from camber.cli import _load_pilot
        from camber.sim.run import fly
        pilot = _load_pilot(str(ROOT / "examples" / "pilots" / name), None)
        log = fly(Simulation(racer), pilot, 10.0)
        assert not log.crashed

    def test_the_flip_actually_goes_inverted(self, racer):
        from camber.cli import _load_pilot
        from camber.sim.run import fly
        pilot = _load_pilot(str(ROOT / "examples" / "pilots" / "flip.py"), None)
        log = fly(Simulation(racer), pilot, 10.0)
        roll = np.degrees(np.abs(np.asarray(log.euler)[:, 0]))
        assert roll.max() > 170 and pilot.phase == "recover"

    def test_an_unknown_pilot_is_a_clear_error(self):
        from camber.cli import _load_pilot
        with pytest.raises(SystemExit, match="no pilot"):
            _load_pilot("nonexistent.py", None)

    def test_the_cli_flies_the_course(self):
        from camber.cli import main
        assert main(["sim", str(ROOT / "examples" / "racer5.toml"),
                     "--pilot", "gates", "--duration", "10"]) == 0


class SteadyLoad:
    """A constant force and moment in body axes: stands in for the aerodynamic
    load at speed, without needing to reach speed to feel it."""

    def __init__(self, force=(0.0, 0.0, 0.0), moment=(0.0, 0.0, 0.0)) -> None:
        self.f, self.m = np.asarray(force, float), np.asarray(moment, float)

    def force(self, v_air_body):
        return self.f, self.m


class TestSteadyLoads:
    """What a controller must do that a P and D loop alone cannot: hold
    position and attitude against a load that does not go away."""

    def _height_error(self, racer, down, **gains):
        from camber.sim.flight import Hover
        sim = Simulation(racer, aero=SteadyLoad(force=(0.0, 0.0, -down)))
        pilot = Hover(height=2.0, **gains)
        state = sim.reset()
        pilot.reset(sim)
        for _ in range(6000):                             # 12 s
            state = sim.step(pilot.update(state, sim))
        return abs(state.position[2] - 2.0)

    def test_it_holds_height_against_a_steady_down_load(self, racer):
        """Aerodynamic down-load at speed is a steady disturbance. Held off by
        the velocity integrator; without one the loop can only trade height
        for force, and the racer flew into the ground from 3 m."""
        assert self._height_error(racer, 4.0) < 0.05

    def test_without_the_integrator_it_sags(self, racer):
        held = self._height_error(racer, 4.0)
        sagged = self._height_error(racer, 4.0, vel_int=0.0)
        assert sagged > 20 * held and sagged > 1.0

    def test_it_holds_attitude_against_a_steady_moment(self, racer):
        """0.14 N m is what the measured CFD table puts on the racer flat out."""
        from camber.sim.flight import Hover
        sim = Simulation(racer, aero=SteadyLoad(moment=(0.0, 0.14, 0.0)))
        pilot = Hover(height=2.0)
        state = sim.reset()
        pilot.reset(sim)
        for _ in range(6000):
            state = sim.step(pilot.update(state, sim))
        tilt = math.degrees(math.acos(np.clip(state.rotation[2, 2], -1.0, 1.0)))
        assert tilt < 3.0

    def test_the_rate_integrator_is_sized_from_control_authority(self, racer):
        """Its clamp used to be the constant 0.5, which on this vehicle caps
        the moment it can hold at 0.005 N m -- far under the aerodynamic one."""
        sim = Simulation(racer)
        pilot = Cascade()
        pilot.reset(sim)
        holdable = pilot._int_clamp * pilot.kint * np.diag(pilot.inertia)
        authority = 0.5 * np.abs(np.asarray(sim.allocation)[3:, :]) @ sim.max_thrust
        assert np.all(holdable > 0.1)                     # N m, all three axes
        assert np.all(holdable < authority)               # and within what it has


class TestCollective:
    def test_thrust_survives_a_reversal(self, racer):
        """Swinging from hard one way to hard the other puts more than 90
        degrees between the desired force and the body's z axis. Projecting
        gave zero thrust -- and zero moment with it, since rotors cannot spin
        backwards -- so the vehicle fell through the turn."""
        sim = Simulation(racer)
        pilot = Cascade()
        state = sim.reset()
        pilot.reset(sim)
        pilot.target = np.array([0.0, 0.0, 6.0])
        lowest = np.inf
        for i in range(6000):
            if sim.t > 2.0:                               # flip the demand
                side = 30.0 if int(sim.t - 2.0) % 2 == 0 else -30.0
                pilot.target = np.array([side, 0.0, 6.0])
            state = sim.step(pilot.update(state, sim))
            if sim.t > 2.0:
                lowest = min(lowest, float(state.thrust.sum()))
        assert lowest > 0.2 * sim.weight


class TestRacingLine:
    def test_it_slows_for_the_hairpins(self):
        """The oval's ends are 3 m radius and its sides are not. A pilot that
        flew every gate at the set speed lapped in the same 8 s whether set to
        9 m/s or 14, and crashed at 18."""
        pilot = GateRacer(Course.oval(), speed=40.0)
        speeds = [pilot.corner_speed(i) for i in range(len(pilot.course.gates))]
        assert min(speeds) < 0.6 * max(speeds)
        assert max(speeds) < 40.0                         # the course, not the setting

    def test_the_gate_line_does_not_dip(self, racer):
        """Projected along a 3D direction, the line through a gate sinks
        whenever the vehicle is above it -- and the height loop follows."""
        course = Course.oval()
        sim = Simulation(racer, course=course)
        pilot = GateRacer(course, speed=12.0)
        state = sim.reset(position=(0.0, 0.0, 6.0))       # well above the gates
        pilot.reset(sim)
        pilot.update(state, sim)
        assert pilot.target[2] == pytest.approx(course.gates[pilot.gate].centre[2])


class TestSpinningMass:
    """The rotors are not massless discs that make thrust appear. They store
    angular momentum, and the airframe pays for every change to it."""

    def test_rotor_speed_matches_what_these_props_turn_at(self, racer):
        """A 5-inch three-blade rated 12.3 N: high twenties of thousands rpm."""
        sim = Simulation(racer)
        rpm = sim.motors.max_rate * 60 / (2 * math.pi)
        assert (25_000 < rpm).all() and (rpm < 32_000).all()

    def test_propeller_inertia_is_a_rod_of_its_own_diameter(self):
        assert MotorBank.disc_inertia(0.0043, 0.0635) == pytest.approx(
            0.0043 * 0.127 ** 2 / 12)

    def test_spinning_the_rotors_up_turns_the_airframe_the_other_way(self, racer):
        """Angular momentum, conserved. In free fall with no aerodynamics and
        thrust that cannot torque the body about yaw except through drag, the
        rotors' momentum has to come from somewhere -- and it comes out of the
        airframe. The sum stays put to within a percent."""
        sim = Simulation(racer, aero=BodyDrag([0.0, 0.0, 0.0]), latency=0.0)
        for t in sim.vehicle.thrusters:                  # all one way: no cancelling
            pass
        sim.motors.spin = np.ones(4)
        sim.allocation = sim.allocation * 0.0            # no thrust, no drag torque
        sim.reset(position=(0.0, 0.0, 50.0))
        for _ in range(400):
            state = sim.step(np.full(4, 0.6))
        h_rotor = float(np.sum(sim.motors.angular_momentum))
        h_body = float((sim.inertia @ state.rates)[2])
        assert h_rotor > 0.01                            # the rotors spun up
        assert abs(h_body + h_rotor) < 0.01 * h_rotor    # and the body answered

    def test_gyroscopic_precession_couples_pitch_into_roll(self, racer):
        """Spin a rotor and turn its axis: it pushes back at right angles.
        On a quad the pairs cancel in hover; on anything whose rotors do not
        cancel -- three rotors, five, a tilting VTOL -- this is always there."""
        sim = Simulation(racer, aero=BodyDrag([0.0, 0.0, 0.0]), latency=0.0)
        sim.motors.spin = np.ones(4)                     # unbalanced, deliberately
        sim.reset(position=(0.0, 0.0, 50.0))
        sim.motors.reset(0.8)
        sim.data.qvel[4] = 2.0                           # pitch rate, body y
        h = float(np.sum(sim.motors.angular_momentum))
        moment = sim._rotor_moments(np.eye(3))
        assert h > 0.0
        assert moment[0] == pytest.approx(-2.0 * h, rel=1e-6)   # into roll
        assert abs(moment[1]) < 1e-9

    def test_a_balanced_quad_has_no_net_gyroscopic_moment(self, racer):
        sim = Simulation(racer, aero=BodyDrag([0.0, 0.0, 0.0]))
        sim.reset(position=(0.0, 0.0, 50.0))
        sim.motors.reset(0.5)                            # all four the same speed
        sim.data.qvel[4] = 2.0
        assert np.abs(sim._rotor_moments(np.eye(3))).max() < 1e-9


class TestLatency:
    """Nothing acts the instant it is asked to."""

    def test_a_command_takes_the_stated_time_to_arrive(self, racer):
        sim = Simulation(racer, latency=0.02)
        sim.reset()
        seen = []
        for _ in range(30):
            state = sim.step(np.ones(4))
            seen.append(float(state.thrust.sum()))
        assert seen[0] == 0.0 and seen[9] == 0.0          # 20 ms at 2 ms a step
        assert seen[12] > 0.0

    def test_zero_latency_acts_at_once(self, racer):
        sim = Simulation(racer, latency=0.0)
        sim.reset()
        assert sim.step(np.ones(4)).thrust.sum() > 0.0

    def test_it_is_a_delay_not_a_filter(self, racer):
        """The command comes out the far end unchanged, only later: the whole
        sequence, shifted by exactly the latency, not smeared by it."""
        sim = Simulation(racer, latency=0.01, motor_tau=1e-6)
        sim.reset()
        sent, arrived = [], []
        for i in range(20):
            u = 0.2 + 0.03 * i
            sent.append(u)
            sim.step(np.full(4, u))
            arrived.append(float(sim.motors.speed[0]))
        shift = int(round(0.01 / sim.dt))                 # five steps
        assert arrived[shift:] == pytest.approx(sent[:-shift], rel=1e-3)
