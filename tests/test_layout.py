"""Symmetry, measurement and presets for the layout editor."""

import copy
import math
import pathlib

import numpy as np
import pytest

from camber.layout import (balance, clearances, distance, find_pairs, mirror,
                           rotor_layout, symmetrise, symmetry_report, wheelbase)
from camber.layout.presets import apply_layout
from camber.layout.symmetry import move_with_twin
from camber.spec import io
from camber.spec.units import Vec3
from camber.spec.vehicle import Spin

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def racer():
    return io.load(ROOT / "examples" / "racer5.toml")


def _nudge(v, name, dx=0.0, dy=0.0, dz=0.0):
    for p in [*v.bodies, *v.thrusters]:
        if p.name == name:
            q = p.position
            p.placement = type(p.placement)(Vec3(q.x + dx, q.y + dy, q.z + dz),
                                            p.placement.rotation)


class TestMirror:
    def test_left_right_flips_y(self):
        assert mirror([0.1, 0.2, -0.03], "xz").tolist() == [0.1, -0.2, -0.03]

    def test_front_back_flips_x(self):
        assert mirror([0.1, 0.2, -0.03], "yz").tolist() == [-0.1, 0.2, -0.03]

    def test_an_unknown_plane_is_refused(self):
        with pytest.raises(ValueError, match="left/right"):
            mirror([0, 0, 0], "xy")


class TestPairing:
    def test_the_racer_pairs_its_rotors_and_centres_its_payload(self, racer):
        p = find_pairs(racer, "xz")
        assert sorted(map(sorted, p.pairs)) == [["front_left", "front_right"],
                                                 ["rear_left", "rear_right"]]
        assert set(p.centred) == {"stack", "battery", "camera", "vtx"}
        assert p.unmatched == []

    def test_twins_must_be_the_same_kind_of_part(self, racer):
        """A 9 g camera is not the mirror twin of a 135 g battery just because
        it happens to sit near its image."""
        _nudge(racer, "camera", dy=0.03)
        _nudge(racer, "vtx", dy=-0.03)
        p = find_pairs(racer, "xz")
        assert ("camera", "vtx") not in p.pairs and ("vtx", "camera") not in p.pairs

    def test_parts_on_the_same_side_are_never_twins(self, racer):
        p = find_pairs(racer, "xz")
        for a, b in p.pairs:
            pa = [x for x in racer.thrusters if x.name == a][0].position.y
            pb = [x for x in racer.thrusters if x.name == b][0].position.y
            assert pa * pb < 0


class TestReport:
    def test_the_racer_is_exactly_symmetric(self, racer):
        assert symmetry_report(racer, "xz")["max_error"] == pytest.approx(0.0)

    def test_a_nudge_is_measured_to_the_millimetre(self, racer):
        _nudge(racer, "front_left", dx=0.003)
        r = symmetry_report(racer, "xz")
        assert r["max_error"] == pytest.approx(0.003, abs=1e-9)

    def test_an_off_centre_battery_shows_its_offset(self, racer):
        _nudge(racer, "battery", dy=0.002)
        r = symmetry_report(racer, "xz")
        row = [x for x in r["rows"] if x["a"] == "battery"][0]
        assert row["error"] == pytest.approx(0.002)

    def test_an_unbalanced_spin_count_is_flagged(self, racer):
        """Three rotors one way and one the other cannot cancel yaw at equal
        thrust; the vehicle has to hover lopsided to hold heading."""
        fl = [t for t in racer.thrusters if t.name == "front_left"][0]
        fr = [t for t in racer.thrusters if t.name == "front_right"][0]
        fl.spin = fr.spin
        assert symmetry_report(racer, "xz")["yaw_balance"] != 0

    def test_a_plus_quad_is_not_flagged(self, racer):
        """Left and right both clockwise, front and rear both counter: the
        standard plus layout, balanced, and not a strict mirror. A first
        version demanded mirrored pairs spin opposite and called it wrong."""
        apply_layout(racer, "quad-plus", 0.22)
        r = symmetry_report(racer, "xz")
        assert r["yaw_balance"] == 0 and r["max_error"] < 1e-9


class TestSymmetrise:
    def test_it_splits_the_difference(self, racer):
        """Neither side is privileged: a 4 mm miss becomes 2 mm each way."""
        fr = [t for t in racer.thrusters if t.name == "front_right"][0]
        before = fr.position.x
        _nudge(racer, "front_left", dx=0.004)
        symmetrise(racer, "xz")
        assert fr.position.x == pytest.approx(before + 0.002)
        assert symmetry_report(racer, "xz")["max_error"] == pytest.approx(0.0, abs=1e-12)

    def test_it_puts_centred_parts_on_the_plane(self, racer):
        _nudge(racer, "battery", dy=0.003)
        symmetrise(racer, "xz")
        b = [x for x in racer.bodies if x.name == "battery"][0]
        assert b.position.y == 0.0

    def test_it_leaves_a_balanced_pattern_alone(self, racer):
        apply_layout(racer, "quad-plus", 0.22)
        spins = [t.spin for t in racer.thrusters]
        symmetrise(racer, "xz")
        assert [t.spin for t in racer.thrusters] == spins

    def test_it_fixes_rotor_spin(self, racer):
        fl = [t for t in racer.thrusters if t.name == "front_left"][0]
        fr = [t for t in racer.thrusters if t.name == "front_right"][0]
        fl.spin = fr.spin
        symmetrise(racer, "xz")
        assert fl.spin != fr.spin

    def test_a_symmetric_quad_does_not_yaw_in_hover(self, racer):
        """The point of fixing spin: after symmetrising, the allocation's yaw
        row cancels at the hover trim."""
        from camber.control import allocation_matrix, solve_hover
        fl = [t for t in racer.thrusters if t.name == "front_left"][0]
        fr = [t for t in racer.thrusters if t.name == "front_right"][0]
        fl.spin = fr.spin
        symmetrise(racer, "xz")
        h = solve_hover(racer)
        assert abs(allocation_matrix(racer)[5] @ h.thrusts) < 1e-9


class TestLinkedDrag:
    def test_moving_one_rotor_moves_its_twin(self, racer):
        moved = move_with_twin(racer, "front_right", [0.10, 0.09, -0.042])
        fl = [t for t in racer.thrusters if t.name == "front_left"][0]
        assert "front_left" in moved
        assert fl.position.as_list() == pytest.approx([0.10, -0.09, -0.042])

    def test_a_centred_part_stays_on_the_centreline(self, racer):
        move_with_twin(racer, "battery", [0.01, 0.02, 0.02])
        b = [x for x in racer.bodies if x.name == "battery"][0]
        assert b.position.y == 0.0 and b.position.x == pytest.approx(0.01)

    def test_four_way_moves_all_four(self, racer):
        moved = move_with_twin(racer, "front_right", [0.1, 0.1, -0.04],
                               planes=("xz", "yz"))
        assert set(moved) == {"front_right", "front_left", "rear_right", "rear_left"}
        rl = [t for t in racer.thrusters if t.name == "rear_left"][0]
        assert rl.position.as_list() == pytest.approx([-0.1, -0.1, -0.04])


class TestMeasure:
    def test_distance_between_parts(self, racer):
        d = distance(racer, "front_right", "front_left")
        assert d["distance"] == pytest.approx(0.176)
        assert d["dy"] == pytest.approx(-0.176)

    def test_wheelbase_is_the_motor_diagonal(self, racer):
        assert wheelbase(racer) == pytest.approx(math.hypot(0.176, 0.176))

    def test_tip_gap(self, racer):
        c = clearances(racer)
        r = racer.thrusters[0].radius
        assert c["min_tip_gap"] == pytest.approx(0.176 - 2 * r)

    def test_overlapping_props_are_a_conflict(self, racer):
        for t in racer.thrusters:
            t.radius = 0.095                       # 7.5-inch props on a 5-inch frame
        assert any("a" in c for c in clearances(racer)["conflicts"])

    def test_stacked_rotors_are_not(self, racer):
        """Coaxial layouts overlap in plan on purpose."""
        a, b = racer.thrusters[0], racer.thrusters[3]
        b.placement = type(b.placement)(Vec3(a.position.x, a.position.y, a.position.z + 0.05),
                                        b.placement.rotation)
        tips = [t for t in clearances(racer)["tips"]
                if {t["a"], t["b"]} == {a.name, b.name}][0]
        assert tips["gap"] < 0 and not tips["conflict"]


class TestBalance:
    def test_the_racer_is_aft_heavy_and_says_by_how_much(self, racer):
        b = balance(racer)
        assert b["cg_offset"][0] == pytest.approx(-0.00177, abs=5e-5)
        assert b["hover_imbalance"] == pytest.approx(0.0411, abs=2e-3)

    def test_a_centred_cg_splits_thrust_evenly(self, racer):
        for p in racer.bodies:
            q = p.position
            p.placement = type(p.placement)(Vec3(0.0, 0.0, q.z), p.placement.rotation)
        racer.frame.members[0].placement = type(racer.frame.members[0].placement)(
            Vec3(0.0, 0.0, racer.frame.members[0].position.z),
            racer.frame.members[0].placement.rotation)
        assert balance(racer)["hover_imbalance"] == pytest.approx(0.0, abs=1e-9)


class TestPresets:
    @pytest.mark.parametrize("kind,n", [("quad-x", 4), ("quad-plus", 4),
                                        ("penta-x", 5), ("hex-x", 6),
                                        ("hepta-x", 7), ("octo-x", 8)])
    def test_counts_and_wheelbase(self, kind, n):
        spec = rotor_layout(kind, 0.25)
        assert len(spec) == n
        radii = [math.hypot(p.x, p.y) for _, p, _, _ in spec]
        assert radii == pytest.approx([0.125] * n)

    @pytest.mark.parametrize("kind", ["quad-x", "quad-plus", "hex-x", "octo-x"])
    def test_every_even_layout_cancels_yaw_and_is_mirror_symmetric(self, racer, kind):
        apply_layout(racer, kind, 0.22, height=-0.042)
        rep = symmetry_report(racer, "xz")
        assert rep["max_error"] < 1e-9 and rep["yaw_balance"] == 0
        from camber.control import allocation_matrix
        n = len(racer.thrusters)
        assert abs(allocation_matrix(racer)[5] @ np.ones(n)) < 1e-12

    def test_applying_a_layout_keeps_the_hardware(self, racer):
        thrust = racer.thrusters[0].static_thrust_n
        apply_layout(racer, "hex-x", 0.3)
        assert len(racer.thrusters) == 6
        assert all(t.static_thrust_n == thrust for t in racer.thrusters)

    def test_quad_x_uses_the_names_people_know(self):
        assert [n for n, _, _, _ in rotor_layout("quad-x", 0.2)] == [
            "front_right", "rear_right", "rear_left", "front_left"]

    def test_even_rings_point_straight_up(self):
        for _, _, _, axis in rotor_layout("hex-x", 0.25):
            assert axis.z == pytest.approx(-1.0)              # UP in NED

    def test_odd_rings_are_canted_for_yaw(self):
        """An uncanted odd ring cannot alternate spin evenly, so it has almost
        no yaw. The cant tilts each motor tangentially to make yaw from thrust."""
        axes = [axis for _, _, _, axis in rotor_layout("penta-x", 0.25)]
        assert all(axis.z < -0.98 for axis in axes)          # still mostly up
        assert all(math.hypot(axis.x, axis.y) > 0.05 for axis in axes)  # but tilted

    @pytest.mark.parametrize("kind", ["quad-x", "penta-x", "hex-x", "hepta-x",
                                      "octo-x"])
    def test_four_flight_axes_are_reachable(self, racer, kind):
        """Thrust, roll, pitch and yaw, independently. This is what a
        multirotor flight controller needs -- lateral force is not commandable
        on any of them, and is not supposed to be. There is no tricopter
        preset: three fixed rotors give only three thrust degrees of freedom."""
        from camber.control import allocation_matrix
        apply_layout(racer, kind, 0.22)
        flight = allocation_matrix(racer)[[2, 3, 4, 5]]       # Fz, Mx, My, Mz
        assert np.linalg.matrix_rank(flight, tol=1e-6) == 4
