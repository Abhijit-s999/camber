"""Vectors, rotations and analytic inertia, checked against closed forms."""

import math

import numpy as np
import pytest

from camber.spec.rotation import Rotation
from camber.spec.shapes import Box, Cylinder, PointMass, Sphere
from camber.spec.units import DOWN, UP, Vec3


class TestVec3:
    def test_arithmetic(self):
        a, b = Vec3(1, 2, 3), Vec3(4, 5, 6)
        assert (a + b).as_list() == [5, 7, 9]
        assert (b - a).as_list() == [3, 3, 3]
        assert (a * 2).as_list() == [2, 4, 6]
        assert a.dot(b) == 32
        assert a.cross(b).as_list() == [-3, 6, -3]

    def test_cross_is_antisymmetric(self):
        a, b = Vec3(1, 2, 3), Vec3(4, 5, 6)
        assert a.cross(b).as_list() == [-v for v in b.cross(a)]

    def test_unit_and_norm(self):
        assert Vec3(3, 4, 0).norm == 5.0
        assert Vec3(3, 4, 0).unit().norm == pytest.approx(1.0)
        with pytest.raises(ValueError, match="zero vector"):
            Vec3().unit()

    def test_parse_accepts_the_forms_a_spec_file_produces(self):
        assert Vec3.parse([1, 2, 3]) == Vec3(1, 2, 3)
        assert Vec3.parse({"x": 1, "z": 2}) == Vec3(1, 0, 2)
        assert Vec3.parse(Vec3(1, 2, 3)) == Vec3(1, 2, 3)
        with pytest.raises(ValueError, match="expected 3"):
            Vec3.parse([1, 2])

    def test_up_is_negative_z(self):
        """NED body frame: z is down, so lift is negative z."""
        assert UP.z == -1.0 and DOWN.z == 1.0


class TestRotation:
    def test_identity_is_a_noop(self):
        v = Vec3(1, 2, 3)
        assert Rotation.identity().apply(v).as_list() == pytest.approx([1, 2, 3])

    def test_rejects_a_reflection(self):
        """det = -1 mirrors the part; it is not a rotation."""
        with pytest.raises(ValueError, match="determinant"):
            Rotation(np.diag([1.0, 1.0, -1.0]))

    def test_rejects_non_orthonormal(self):
        with pytest.raises(ValueError, match="orthonormal"):
            Rotation(np.diag([2.0, 1.0, 1.0]))

    def test_yaw_90_takes_x_to_y(self):
        r = Rotation.from_euler(yaw=math.pi / 2)
        assert r.apply(Vec3(1, 0, 0)).as_list() == pytest.approx([0, 1, 0], abs=1e-12)

    def test_euler_roundtrip(self):
        roll, pitch, yaw = 0.3, -0.4, 1.2
        got = Rotation.from_euler(roll, pitch, yaw).as_euler()
        assert got == pytest.approx((roll, pitch, yaw))

    def test_inverse_and_compose(self):
        r = Rotation.from_euler(0.2, 0.3, 0.4)
        assert np.allclose(r.compose(r.inverse()).matrix, np.eye(3))

    def test_aligning_points_source_at_target(self):
        r = Rotation.aligning(Vec3(0, 0, -1), Vec3(1, 0, 0))
        assert r.apply(Vec3(0, 0, -1)).as_list() == pytest.approx([1, 0, 0], abs=1e-12)

    def test_aligning_handles_antiparallel(self):
        """The 180-degree case has no unique axis and is where naive code fails."""
        r = Rotation.aligning(Vec3(0, 0, -1), Vec3(0, 0, 1))
        assert r.apply(Vec3(0, 0, -1)).as_list() == pytest.approx([0, 0, 1], abs=1e-12)

    def test_aligning_identical_vectors_is_identity(self):
        r = Rotation.aligning(Vec3(1, 0, 0), Vec3(1, 0, 0))
        assert np.allclose(r.matrix, np.eye(3))


class TestShapes:
    def test_box_matches_closed_form(self):
        b, m = Box(0.2, 0.1, 0.05), 2.0
        i = b.inertia(m)
        assert i[0, 0] == pytest.approx(m / 12 * (0.1**2 + 0.05**2))
        assert i[1, 1] == pytest.approx(m / 12 * (0.2**2 + 0.05**2))
        assert i[2, 2] == pytest.approx(m / 12 * (0.2**2 + 0.1**2))
        assert b.volume == pytest.approx(0.2 * 0.1 * 0.05)

    def test_cylinder_matches_closed_form(self):
        c, m = Cylinder(0.03, 0.10), 0.5
        i = c.inertia(m)
        assert i[2, 2] == pytest.approx(0.5 * m * 0.03**2)
        assert i[0, 0] == pytest.approx(m / 12 * (3 * 0.03**2 + 0.10**2))
        assert i[0, 0] == pytest.approx(i[1, 1])

    def test_sphere_matches_closed_form(self):
        s, m = Sphere(0.04), 1.5
        assert s.inertia(m)[0, 0] == pytest.approx(0.4 * m * 0.04**2)
        assert s.volume == pytest.approx(4 / 3 * math.pi * 0.04**3)

    def test_thin_rod_limit(self):
        """A long thin cylinder must approach the classic mL^2/12 rod."""
        c, m, L = Cylinder(1e-5, 1.0), 1.0, 1.0
        assert c.inertia(m)[0, 0] == pytest.approx(m * L**2 / 12, rel=1e-6)

    def test_point_mass_has_no_inertia_of_its_own(self):
        assert np.allclose(PointMass().inertia(5.0), np.zeros((3, 3)))
        assert PointMass().volume == 0.0

    @pytest.mark.parametrize("bad", [
        lambda: Box(0, 1, 1), lambda: Box(1, -1, 1),
        lambda: Cylinder(0, 1), lambda: Cylinder(1, 0), lambda: Sphere(-1),
    ])
    def test_degenerate_dimensions_are_rejected(self, bad):
        with pytest.raises(ValueError, match="positive"):
            bad()
