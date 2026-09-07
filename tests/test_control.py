"""Control allocation, checked against results a textbook would give."""

import math

import numpy as np
import pytest

from camber.control import G, allocation_matrix, controllability, solve_hover
from camber.spec import Body, Placement, Vec3, Vehicle
from camber.spec.rotation import Rotation
from camber.spec.shapes import Box
from camber.spec.vehicle import Rotor, Spin

ARM = 0.125
X_LAYOUT = [(ARM, ARM, Spin.CW), (ARM, -ARM, Spin.CCW),
            (-ARM, -ARM, Spin.CW), (-ARM, ARM, Spin.CCW)]


def quad(layout=X_LAYOUT, cant=0.0, thrust=8.0, body_mass=0.25):
    v = Vehicle(name="quad")
    v.bodies.append(Body("stack", body_mass, Box(0.04, 0.04, 0.02)))
    for i, (x, y, spin) in enumerate(layout):
        # Cant each rotor outward about the arm direction, for full actuation.
        rot = Rotation.identity()
        if cant:
            axis = Vec3(-y, x, 0.0)
            rot = Rotation.from_axis_angle(axis, cant)
        v.thrusters.append(Rotor(f"m{i}", mass=0.035, radius=0.0635,
                                 placement=Placement(Vec3(x, y, -0.01), rot),
                                 spin=spin, static_thrust_n=thrust))
    return v


class TestAllocation:
    def test_matrix_is_six_by_n(self):
        assert allocation_matrix(quad()).shape == (6, 4)

    def test_flat_rotors_all_push_along_minus_z(self):
        a = allocation_matrix(quad())
        assert np.allclose(a[0:3, :], np.tile([[0], [0], [-1]], 4), atol=1e-12)

    def test_no_thrusters_is_an_error(self):
        v = Vehicle(name="brick")
        v.bodies.append(Body("mass", 1.0))
        with pytest.raises(ValueError, match="no thrusters"):
            allocation_matrix(v)

    def test_diagonal_rotors_have_opposite_roll_arms(self):
        a = allocation_matrix(quad())
        assert a[3, 0] == pytest.approx(-a[3, 2])  # Mx of opposing arms


class TestControllability:
    def test_flat_quad_is_underactuated(self):
        """The textbook result: a flat quadrotor must tilt to translate."""
        c = controllability(quad())
        assert c.rank == 4
        assert set(c.uncontrollable_axes) == {"Fx", "Fy"}
        assert not c.is_fully_controllable

    def test_four_rotors_can_never_be_fully_actuated(self):
        """rank(A) <= min(6, N). Canting cannot manufacture a fifth dimension.

        A canted quad gives every axis a non-zero *projection* onto the range
        while the range stays 4-dimensional -- which is why controllability
        tests range membership rather than projection.
        """
        c = controllability(quad(cant=math.radians(20)))
        assert c.rank == 4
        assert not c.is_fully_controllable

    def test_canted_hexrotor_is_fully_actuated(self):
        """Six canted rotors do span the wrench space -- the known result for
        fully-actuated multirotors, and the reason they are built with six."""
        layout = []
        for k in range(6):
            a = k * math.pi / 3
            layout.append((ARM * math.cos(a), ARM * math.sin(a),
                           Spin.CW if k % 2 else Spin.CCW))
        c = controllability(quad(layout, cant=math.radians(25)))
        assert c.rank == 6
        assert c.is_fully_controllable
        assert c.margin > 0

    def test_rotors_coincident_at_the_cg_cannot_roll_or_pitch(self):
        """The design an unconstrained optimiser converges on if you let it:
        everything stacked at the CG is light, efficient, and cannot roll."""
        c = controllability(quad([(0.0, 0.0, Spin.CW), (0.0, 0.0, Spin.CCW),
                                  (0.0, 0.0, Spin.CW), (0.0, 0.0, Spin.CCW)]))
        assert c.rank < 4
        assert {"Mx", "My"} <= set(c.uncontrollable_axes)

    def test_margin_shrinks_as_the_arms_shorten(self):
        """Continuous, so an optimiser can be steered away from the cliff."""
        wide = controllability(quad())
        narrow_layout = [(x / 10, y / 10, s) for x, y, s in X_LAYOUT]
        narrow = controllability(quad(narrow_layout))
        assert narrow.margin < wide.margin


class TestHover:
    def test_symmetric_quad_splits_weight_evenly(self):
        v = quad()
        h = solve_hover(v)
        assert h.feasible
        expected = v.mass_properties().mass * G / 4
        assert h.thrusts == pytest.approx([expected] * 4)
        assert h.total_thrust_n == pytest.approx(v.mass_properties().mass * G)

    def test_yaw_torques_cancel_with_alternating_spins(self):
        a = allocation_matrix(quad())
        h = solve_hover(quad())
        assert float(a[5] @ h.thrusts) == pytest.approx(0.0, abs=1e-12)

    def test_all_rotors_spinning_the_same_way_cannot_hold_heading(self):
        """Four identical spins produce net yaw torque that nothing cancels."""
        same = [(x, y, Spin.CCW) for x, y, _ in X_LAYOUT]
        h = solve_hover(quad(same))
        assert not h.feasible
        assert "trim" in h.reason

    def test_offset_payload_shifts_the_load_onto_the_near_rotors(self):
        v = quad()
        v.bodies.append(Body("camera", 0.15, Box(0.03, 0.03, 0.03),
                             Placement(Vec3(0.18, 0.0, 0.0))))
        h = solve_hover(v)
        assert h.feasible
        front = h.thrusts[0] + h.thrusts[1]   # rotors at +x
        rear = h.thrusts[2] + h.thrusts[3]
        assert front > rear

    def test_infeasible_when_rotors_cannot_lift_the_weight(self):
        h = solve_hover(quad(thrust=0.5, body_mass=2.0))
        assert not h.feasible
        assert "rated thrust" in h.reason

    def test_utilisation_is_reported(self):
        h = solve_hover(quad())
        assert 0.0 < float(np.nanmax(h.utilisation)) < 1.0

    def test_unselected_rotor_refuses_to_invent_a_thrust(self):
        """AI-05: no invented constants. Geometry does not imply thrust."""
        v = quad()
        v.thrusters[0].static_thrust_n = None
        with pytest.raises(ValueError, match="does not guess"):
            v.thrusters[0].max_thrust()
