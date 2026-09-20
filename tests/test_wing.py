"""Lifting and control surfaces, checked against the aerodynamics by hand."""

import math

import numpy as np
import pytest

from camber.sim.wing import RHO, Surface, SurfaceSet


def wing():
    return Surface("wing", centre=[0.0, 0.0, 0.0], area=0.3, chord=0.2)


class TestLift:
    def test_no_lift_at_the_zero_lift_angle(self):
        f, _ = wing().wrench([20.0, 0.0, 0.0], [0, 0, 0], 0.0)
        assert abs(f[2]) < 1e-6

    def test_lift_matches_thin_wing_theory_below_stall(self):
        """L = 1/2 rho V^2 S CLalpha alpha, to a percent."""
        V, a = 20.0, math.radians(5.0)
        vb = [V * math.cos(a), 0.0, -V * math.sin(a)]
        f, _ = wing().wrench(vb, [0, 0, 0], 0.0)
        q = 0.5 * RHO * V * V * 0.3
        assert f[2] == pytest.approx(q * 5.0 * a, rel=0.02)

    def test_lift_is_perpendicular_to_the_flow(self):
        V, a = 20.0, math.radians(6.0)
        vb = np.array([V * math.cos(a), 0.0, -V * math.sin(a)])
        f, _ = wing().wrench(vb, [0, 0, 0], 0.0)
        lift = f - np.dot(f, vb / np.linalg.norm(vb)) * vb / np.linalg.norm(vb)
        # The lift component is orthogonal to the flow by construction; check
        # the force has almost no component *along* the flow beyond drag.
        along = float(np.dot(f, vb) / np.linalg.norm(vb))
        assert along < 0.0                                   # net drag opposes flow
        assert np.linalg.norm(lift) > 10 * abs(along)        # lift dominates

    def test_it_stalls(self):
        w = wing()
        below, _ = w.wrench(_flow(20, 14), [0, 0, 0], 0.0)
        past, _ = w.wrench(_flow(20, 25), [0, 0, 0], 0.0)
        assert past[2] < below[2]                            # lift falls past stall

    def test_drag_rises_with_lift(self):
        w = wing()
        f0, _ = w.wrench([20.0, 0.0, 0.0], [0, 0, 0], 0.0)
        f1, _ = w.wrench(_flow(20, 8), [0, 0, 0], 0.0)
        assert abs(f1[0]) > abs(f0[0])                       # induced drag


class TestControl:
    def test_elevator_makes_a_pitching_moment(self):
        e = Surface("elev", centre=[-0.5, 0, 0], area=0.05, cl_delta=3.0, channel=0)
        s = SurfaceSet([e])
        s.set_controls([1.0])
        _, up = s.wrench([20.0, 0, 0], [0, 0, 0])
        s.set_controls([-1.0])
        _, dn = s.wrench([20.0, 0, 0], [0, 0, 0])
        assert up[1] * dn[1] < 0                             # opposite signs
        assert abs(up[1]) > 1.0                              # and substantial

    def test_ailerons_make_a_clean_roll(self):
        r = Surface("ar", centre=[0, -0.8, 0], area=0.03, cl_delta=3.0, channel=1)
        l = Surface("al", centre=[0, 0.8, 0], area=0.03, cl_delta=3.0, channel=1,
                    channel_sign=-1.0)
        s = SurfaceSet([r, l])
        s.set_controls([0.0, 1.0])
        f, m = s.wrench([20.0, 0, 0], [0, 0, 0])
        assert abs(m[0]) > 1.0                               # roll
        assert abs(m[1]) < 1e-6 and abs(m[2]) < 1e-6         # no pitch or yaw
        assert abs(f[2]) < 1e-6                              # no net lift change

    def test_deflection_is_clamped(self):
        e = Surface("elev", centre=[-0.5, 0, 0], area=0.05, cl_delta=3.0, channel=0)
        s = SurfaceSet([e], max_deflection=math.radians(20))
        s.set_controls([5.0])                                # asked for 5x full
        assert s.deflections[0] == pytest.approx(math.radians(20))


class TestDamping:
    def test_a_tail_damps_pitch_rate(self):
        """omega x r changes the angle a tail meets the air: a nose-up pitch
        rate makes the tail push the nose back down. This is what stops a
        plane porpoising, and it needs no deflection."""
        t = Surface("tail", centre=[-0.6, 0, 0], area=0.08, channel=0)
        s = SurfaceSet([t])
        _, m = s.wrench([20.0, 0, 0], [0.0, 1.0, 0.0])       # nose-up pitch rate
        assert m[1] < -0.1                                   # opposes it


def _flow(v, alpha_deg):
    a = math.radians(alpha_deg)
    return [v * math.cos(a), 0.0, -v * math.sin(a)]


# --- a plane that actually flies, for the controller tests ------------------

def build_plane():
    """A 1.25 kg pusher aeroplane: fuselage, wing, tail surfaces, one motor.

    Small enough to state in a test, complete enough to fly: a cambered main
    wing, an elevator, a rudder, and two ailerons on their own channels."""
    import math as _m

    from camber.sim.aero import BodyDrag
    from camber.spec.shapes import Box
    from camber.spec.units import Vec3
    from camber.spec.vehicle import Body, Rotor, Spin, Vehicle

    fuse = Body("fuselage", 0.8, Box(0.6, 0.08, 0.08))
    wing_struct = Body("wing_struct", 0.4, Box(0.12, 1.0, 0.02))
    motor = Rotor("motor", 0.05, radius=0.10, blades=2, thrust_axis=Vec3(1, 0, 0),
                  static_thrust_n=8.0, spin=Spin.CCW)
    plane = Vehicle("plane1", bodies=[fuse, wing_struct], thrusters=[motor])
    surfaces = [
        Surface("wing", centre=[0, 0, 0], area=0.30, chord=0.25,
                alpha_0=_m.radians(-2)),
        Surface("htail", centre=[-0.55, 0, 0], area=0.06, chord=0.15,
                cl_delta=3.5, channel=0),
        Surface("vtail", centre=[-0.55, 0, 0], area=0.04, chord=0.15,
                span_axis=[0, 0, 1], normal=[0, 1, 0], cl_delta=3.5, channel=2),
        Surface("ail_r", centre=[0, -0.45, 0], area=0.02, cl_delta=3.0,
                channel=1, channel_sign=1.0),
        Surface("ail_l", centre=[0, 0.45, 0], area=0.02, cl_delta=3.0,
                channel=1, channel_sign=-1.0),
    ]
    aero = SurfaceSet(surfaces, body_drag=BodyDrag([0.02, 0.1, 0.1]))
    return plane, aero


class TestPlaneFlies:
    def _fly(self, seconds=24.0, turn_to=None, turn_at=8.0):
        from camber.sim import Simulation
        from camber.sim.flight import FixedWing
        plane, aero = build_plane()
        sim = Simulation(plane, aero=aero, latency=0.006)
        ap = FixedWing(speed=18.0, altitude=50.0)
        s = sim.reset(position=(0, 0, 50), velocity=(18, 0, 0))
        ap.reset(sim)
        for _ in range(int(seconds / sim.dt)):
            if turn_to is not None and sim.t > turn_at:
                ap.heading = turn_to
            s = sim.step(ap.update(s, sim))
        return s, sim, ap

    def test_it_holds_altitude_and_speed(self):
        s, sim, _ = self._fly(seconds=12.0)
        assert not sim.crashed
        assert abs(s.position[2] - 50.0) < 3.0
        assert abs(float(np.linalg.norm(s.velocity)) - 18.0) < 2.0

    def test_it_holds_wings_level_against_prop_torque(self):
        """A single propeller rolls the airframe; the roll loop must beat it."""
        s, _, _ = self._fly(seconds=8.0)
        assert abs(math.degrees(s.euler[0])) < 5.0

    def test_it_turns_to_a_commanded_heading(self):
        import math as _m
        s, sim, _ = self._fly(seconds=24.0, turn_to=_m.radians(90))
        heading = math.degrees(math.atan2(s.velocity[1], s.velocity[0]))
        assert not sim.crashed
        assert abs(heading - 90.0) < 6.0
        assert abs(math.degrees(s.euler[0])) < 6.0        # rolled back level

    def test_it_banks_into_the_turn(self):
        """Mid-turn it should be banked, not skidding round flat."""
        from camber.sim import Simulation
        from camber.sim.flight import FixedWing
        plane, aero = build_plane()
        sim = Simulation(plane, aero=aero, latency=0.006)
        ap = FixedWing(speed=18.0, altitude=50.0)
        s = sim.reset(position=(0, 0, 50), velocity=(18, 0, 0))
        ap.reset(sim)
        banks = []
        for _ in range(int(16 / sim.dt)):
            if sim.t > 6.0:
                ap.heading = math.radians(90)
            s = sim.step(ap.update(s, sim))
            if 6.5 < sim.t < 9.0:
                banks.append(abs(math.degrees(s.euler[0])))
        assert max(banks) > 15.0
