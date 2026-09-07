"""Assembly mass properties: the parallel-axis theorem, checked by hand."""

import numpy as np
import pytest

from camber.massprops import MassProperties, Part, combine
from camber.spec.shapes import Box, Sphere
from camber.spec.rotation import Rotation
from camber.spec.units import Vec3

I0 = np.zeros((3, 3))
E = np.eye(3)


def test_single_part_at_origin_is_its_own_inertia():
    b, m = Box(0.2, 0.1, 0.05), 2.0
    mp = combine([Part(m, Vec3(), b.inertia(m), E)])
    assert mp.mass == pytest.approx(m)
    assert np.allclose(mp.inertia, b.inertia(m))


def test_cg_is_the_mass_weighted_mean():
    mp = combine([Part(1.0, Vec3(1, 0, 0), I0, E),
                  Part(3.0, Vec3(-1, 0, 0), I0, E)])
    assert mp.cg.x == pytest.approx((1 * 1 + 3 * -1) / 4)


def test_dumbbell_matches_hand_arithmetic():
    """Two point masses at +/- d on y: Ixx = Izz = 2md^2, Iyy = 0."""
    d, m = 0.3, 1.0
    mp = combine([Part(m, Vec3(0, d, 0), I0, E), Part(m, Vec3(0, -d, 0), I0, E)])
    assert mp.inertia[0, 0] == pytest.approx(2 * m * d**2)
    assert mp.inertia[2, 2] == pytest.approx(2 * m * d**2)
    assert mp.inertia[1, 1] == pytest.approx(0.0, abs=1e-15)


def test_offset_body_picks_up_the_parallel_axis_term():
    b, m, d = Box(0.1, 0.1, 0.1), 1.0, 0.5
    at_origin = combine([Part(m, Vec3(), b.inertia(m), E)])
    offset = combine([Part(m, Vec3(d, 0, 0), b.inertia(m), E)])
    # Shifting the reference back to the part's centroid recovers the original.
    assert np.allclose(offset.inertia, at_origin.inertia)
    assert offset.cg.x == pytest.approx(d)


def test_rotating_a_symmetric_body_changes_nothing():
    s, m = Sphere(0.1), 2.0
    r = Rotation.from_euler(0.3, 0.5, 0.9).matrix
    a = combine([Part(m, Vec3(), s.inertia(m), E)])
    b = combine([Part(m, Vec3(), s.inertia(m), r)])
    assert np.allclose(a.inertia, b.inertia)


def test_rotating_an_asymmetric_body_permutes_its_axes():
    """A box yawed 90 degrees must swap its x and y moments."""
    b, m = Box(0.3, 0.1, 0.05), 1.0
    upright = combine([Part(m, Vec3(), b.inertia(m), E)])
    yawed = combine([Part(m, Vec3(), b.inertia(m),
                          Rotation.from_euler(yaw=np.pi / 2).matrix)])
    assert yawed.inertia[0, 0] == pytest.approx(upright.inertia[1, 1])
    assert yawed.inertia[1, 1] == pytest.approx(upright.inertia[0, 0])
    assert yawed.inertia[2, 2] == pytest.approx(upright.inertia[2, 2])


def test_inertia_tensor_is_symmetric():
    mp = combine([Part(1.0, Vec3(0.1, 0.2, 0.3), Box(0.1, 0.2, 0.3).inertia(1.0),
                       Rotation.from_euler(0.4, 0.5, 0.6).matrix),
                  Part(2.0, Vec3(-0.2, 0.1, 0.0), I0, np.eye(3))])
    assert np.allclose(mp.inertia, mp.inertia.T, atol=1e-18)


def test_triangle_inequality_holds_for_real_bodies():
    mp = combine([Part(1.0, Vec3(0.1, 0, 0), Box(0.1, 0.2, 0.3).inertia(1.0), E),
                  Part(0.5, Vec3(-0.2, 0.1, 0.05), Sphere(0.05).inertia(0.5), E)])
    assert mp.is_physical()


def test_impossible_tensor_is_caught():
    """I1 + I2 < I3 cannot arise from any positive mass distribution."""
    bogus = MassProperties(1.0, Vec3(), np.diag([0.001, 0.001, 1.0]))
    assert not bogus.is_physical()


def test_inertia_about_another_point():
    m, d = 2.0, 0.4
    mp = combine([Part(m, Vec3(), I0, E)])
    shifted = mp.inertia_about(Vec3(d, 0, 0))
    assert shifted[1, 1] == pytest.approx(m * d**2)
    assert shifted[0, 0] == pytest.approx(0.0, abs=1e-15)


def test_empty_assembly_is_empty_not_an_error():
    assert combine([]).mass == 0.0


def test_zero_mass_assembly_is_rejected():
    with pytest.raises(ValueError, match="zero or negative total mass"):
        combine([Part(0.0, Vec3(), I0, E)])
