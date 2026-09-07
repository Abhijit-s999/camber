"""Physics checked against closed forms, conservation laws, and real data."""

import math
from pathlib import Path

import numpy as np
import pytest

from camber.physics.airfoil import build_polar, viterna
from camber.physics.battery import CELL_CUTOFF_V, Pack
from camber.physics.bemt import (
    RHO_AIR, BladeDefinition, analyse, prandtl_tip_loss,
)
from camber.physics.motor import Motor, match_speed
from camber.physics.propulsion import PropellerMap, solve_operating_point

UIUC_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "UIUC-propDB"
needs_uiuc = pytest.mark.skipif(
    not UIUC_ROOT.exists(),
    reason="UIUC database not downloaded (see camber/catalogue/uiuc.py)",
)


def simple_blade(radius=0.127, blades=2, twist_deg=15.0, chord_ratio=0.12, n=20):
    x = np.linspace(0.15, 1.0, n)
    return BladeDefinition(radius, blades, x,
                           np.full(n, chord_ratio),
                           np.radians(np.full(n, twist_deg)))


class TestAirfoil:
    def test_drag_is_never_negative_anywhere_on_the_circle(self):
        """Viterna's drag term goes negative past 90 degrees if left alone,
        which would let a BEMT solve extract energy from nothing."""
        p = build_polar()
        _, cd = p(np.linspace(-np.pi, np.pi, 3601), np.full(3601, 1e5))
        assert cd.min() > 0.0

    def test_lift_stays_bounded_through_the_viterna_singularity(self):
        """cos^2(a)/sin(a) is singular at 0 and pi; unguarded it returns 1e5."""
        p = build_polar()
        cl, _ = p(np.linspace(-np.pi, np.pi, 3601), np.full(3601, 1e5))
        assert np.abs(cl).max() < 3.0

    def test_flat_plate_limit_at_ninety_degrees(self):
        p = build_polar()
        cl, cd = p(np.pi / 2, 1e5)
        assert abs(float(cl)) < 0.05          # no lift edge-on
        assert 1.0 < float(cd) < 1.5          # normal flat plate

    def test_attached_range_is_sane(self):
        p = build_polar()
        cl, cd = p(np.radians([0.0, 5.0, 10.0]), np.full(3, 1e5))
        assert cl[0] < cl[1] < cl[2]          # lift curve rises pre-stall
        assert cd.max() < 0.1                 # attached drag stays small

    def test_polar_is_periodic(self):
        p = build_polar()
        a, b = p(np.radians(170.0), 1e5), p(np.radians(-190.0), 1e5)
        assert float(a[0]) == pytest.approx(float(b[0]), abs=1e-6)


class TestTipLoss:
    def test_goes_to_zero_at_the_tip(self):
        assert prandtl_tip_loss(math.radians(10), 0.999, 2) < 0.1

    def test_near_one_in_midspan(self):
        assert prandtl_tip_loss(math.radians(10), 0.6, 2) > 0.8

    def test_more_blades_means_less_tip_loss(self):
        """A tip vortex per blade; more blades, each less loaded."""
        assert (prandtl_tip_loss(math.radians(10), 0.9, 4)
                > prandtl_tip_loss(math.radians(10), 0.9, 2))

    def test_bounded(self):
        for phi in np.radians([1, 5, 30, 80]):
            for x in (0.2, 0.5, 0.95):
                assert 0.0 < prandtl_tip_loss(phi, x, 2) <= 1.0


class TestBemt:
    def test_produces_thrust_and_converges(self):
        r = analyse(simple_blade(), v_inf=3.0, rpm=5000)
        assert r.converged
        assert r.thrust_n > 0 and r.torque_nm > 0

    def test_thrust_falls_as_the_aircraft_speeds_up(self):
        blade = simple_blade()
        t = [analyse(blade, v, 5000).thrust_n for v in (2.0, 6.0, 10.0)]
        assert t[0] > t[1] > t[2]

    def test_thrust_rises_with_rpm(self):
        blade = simple_blade()
        assert analyse(blade, 3.0, 6000).thrust_n > analyse(blade, 3.0, 4000).thrust_n

    def test_ct_rises_with_rpm_at_fixed_advance_ratio(self):
        """Small propellers are NOT Reynolds-independent, and the data says so.

        CT and CP remove rpm by construction, so a textbook reading expects
        them to be invariant at fixed J. For propellers this size they are not:
        chord Reynolds numbers sit around 10^5, where laminar separation costs
        real performance, and raising rpm buys it back.

        The bound below is anchored to measurement, not taste. UIUC runs of the
        same propeller at several rpm give, for a doubling of rpm at fixed J,
        roughly +25% CT (apce_10x5) to +42% (apce_11x7). The model reproduces
        the direction and the order of magnitude while over-stating it by
        about a factor of two -- a known limitation, recorded as PHYS-04.
        """
        blade = simple_blade()
        a = analyse(blade, 4.0, 4000)
        b = analyse(blade, 8.0, 8000)   # same J, double the rpm
        assert a.advance_ratio == pytest.approx(b.advance_ratio, rel=1e-9)

        growth = (b.ct - a.ct) / a.ct
        assert growth > 0.10, (
            f"CT grew only {growth:+.0%} for double the rpm; measurement shows "
            f"+25% to +42%, so the model has lost its Reynolds sensitivity"
        )
        assert growth < 1.20, (
            f"CT grew {growth:+.0%} for double the rpm, far beyond the +25% to "
            f"+42% seen in the tunnel data"
        )

    def test_efficiency_is_physical(self):
        blade = simple_blade()
        for v in (2.0, 5.0, 8.0):
            r = analyse(blade, v, 5000)
            assert 0.0 <= r.efficiency < 1.0, "propulsive efficiency cannot exceed 1"

    def test_induced_velocity_agrees_with_momentum_theory(self):
        """The whole-disc check: T = 2 rho A V_a v_a must roughly hold.

        This is what caught a wrong induction ceiling: a clamp that looked
        harmless truncated the induced velocity at low advance ratio.
        """
        blade = simple_blade()
        v_inf = 3.0
        r = analyse(blade, v_inf, 5000)
        area = math.pi * blade.radius_m ** 2
        # Solve T = 2 rho A (V + v) v for the mean induced velocity.
        c = -r.thrust_n / (2 * RHO_AIR * area)
        v_induced = (-v_inf + math.sqrt(v_inf ** 2 - 4 * c)) / 2
        mean_a = float(np.mean([s.a for s in r.stations]))
        implied = mean_a * v_inf
        assert 0.3 < implied / v_induced < 3.0, (
            f"blade-element induced velocity {implied:.2f} m/s is inconsistent "
            f"with momentum theory's {v_induced:.2f} m/s"
        )

    def test_resample_preserves_the_blade(self):
        b = simple_blade(n=8).resample(30)
        assert len(b.r_over_R) == 30
        assert b.r_over_R[0] >= 0.15 and b.r_over_R[-1] <= 1.0
        assert np.all(np.diff(b.r_over_R) > 0)

    def test_rejects_bad_blades(self):
        with pytest.raises(ValueError, match="at least 2 blades"):
            BladeDefinition(0.1, 1, [0.5], [0.1], [0.2])
        with pytest.raises(ValueError, match="same length"):
            BladeDefinition(0.1, 2, [0.5, 0.6], [0.1], [0.2])
        with pytest.raises(ValueError, match="rpm must be positive"):
            analyse(simple_blade(), 3.0, 0.0)


class TestMotor:
    def test_kt_and_kv_are_the_same_constant_in_si(self):
        m = Motor("m", kv_rpm_v=1000, resistance_ohm=0.1, no_load_current_a=0.5)
        assert m.kt == pytest.approx(1.0 / m.kv_rad)
        assert m.kv_rad == pytest.approx(1000 * 2 * math.pi / 60)

    def test_no_load_speed(self):
        m = Motor("m", kv_rpm_v=1000, resistance_ohm=0.1, no_load_current_a=0.0)
        omega = 14.8 * m.kv_rad
        assert m.current_for(omega, 14.8) == pytest.approx(0.0, abs=1e-9)

    def test_stall_current_is_voltage_over_resistance(self):
        m = Motor("m", kv_rpm_v=1000, resistance_ohm=0.1, no_load_current_a=0.0)
        assert m.current_for(0.0, 10.0) == pytest.approx(100.0)

    def test_efficiency_below_one_and_peaks_in_between(self):
        m = Motor("m", kv_rpm_v=1000, resistance_ohm=0.1, no_load_current_a=0.5)
        effs = [m.operating_point(w, 14.8).efficiency
                for w in np.linspace(100, 14.8 * m.kv_rad * 0.99, 40)]
        assert max(effs) < 1.0
        assert 0 < np.argmax(effs) < 39, "peak efficiency should be interior"

    def test_matches_a_quadratic_load(self):
        m = Motor("m", kv_rpm_v=1000, resistance_ohm=0.1, no_load_current_a=0.5)
        k = 2e-7
        w = match_speed(m, 14.8, lambda w: k * w * w)
        assert m.torque(w, 14.8) == pytest.approx(k * w * w, rel=1e-6)

    def test_rejects_a_load_it_cannot_start(self):
        m = Motor("m", kv_rpm_v=1000, resistance_ohm=0.1, no_load_current_a=0.5)
        with pytest.raises(ValueError, match="cannot turn this load"):
            match_speed(m, 14.8, lambda w: 1e6)


class TestBattery:
    def test_voltage_sags_under_load(self):
        p = Pack("p", cells=4, capacity_ah=1.5, resistance_ohm=0.02, max_c_rate=75)
        assert p.terminal_voltage(50.0) < p.terminal_voltage(5.0)

    def test_endurance_falls_with_current(self):
        p = Pack("p", cells=4, capacity_ah=1.5, resistance_ohm=0.02, max_c_rate=75)
        assert p.endurance_s(5.0) > p.endurance_s(20.0) > p.endurance_s(60.0)

    def test_low_draw_matches_simple_capacity_division(self):
        p = Pack("p", cells=4, capacity_ah=2.0, resistance_ohm=0.01, max_c_rate=50)
        expected = 2.0 * 0.80 / 4.0 * 3600.0
        assert p.endurance_s(4.0) == pytest.approx(expected, rel=0.02)

    def test_high_draw_is_sag_limited_not_capacity_limited(self):
        """The failure a pure Ah/A division misses entirely."""
        p = Pack("weak", cells=4, capacity_ah=1.0, resistance_ohm=0.08, max_c_rate=100)
        assert p.sag_limited(80.0)
        assert not p.sag_limited(3.0)

    def test_open_circuit_voltage_is_monotonic(self):
        p = Pack("p", cells=4, capacity_ah=1.5, resistance_ohm=0.02, max_c_rate=75)
        v = [p.open_circuit_voltage(s) for s in np.linspace(0, 1, 50)]
        assert all(b >= a - 1e-9 for a, b in zip(v, v[1:]))
        assert p.open_circuit_voltage(1.0) == pytest.approx(p.full_voltage, rel=0.02)

    def test_cutoff_is_per_cell(self):
        p = Pack("p", cells=6, capacity_ah=1.5, resistance_ohm=0.02, max_c_rate=75)
        assert p.cutoff_voltage == pytest.approx(6 * CELL_CUTOFF_V)

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError):
            Pack("p", cells=0, capacity_ah=1.0, resistance_ohm=0.01, max_c_rate=10)
        p = Pack("p", cells=4, capacity_ah=1.5, resistance_ohm=0.02, max_c_rate=75)
        with pytest.raises(ValueError, match="current must be positive"):
            p.endurance_s(0.0)


@pytest.fixture(scope="module")
def prop_map():
    return PropellerMap.from_blade(simple_blade(), name="test")


class TestPropulsionCoupling:

    def test_map_reproduces_direct_bemt(self, prop_map):
        """The map is a speed optimisation, not a different model."""
        blade = simple_blade()
        for j in (0.2, 0.4, 0.6):
            rpm = 5000.0
            direct = analyse(blade, j * (rpm / 60) * 2 * blade.radius_m, rpm)
            ct, _ = prop_map.coefficients(j)
            assert ct == pytest.approx(direct.ct, rel=0.05)

    def test_map_is_far_faster_than_bemt(self, prop_map):
        import time
        t0 = time.perf_counter()
        for _ in range(2000):
            prop_map.thrust_torque(500.0, 3.0)
        per_call = (time.perf_counter() - t0) / 2000
        assert per_call < 1e-4, "map lookup must be microseconds, not milliseconds"

    def test_coupled_point_is_self_consistent(self, prop_map):
        m = Motor("m", kv_rpm_v=1000, resistance_ohm=0.12, no_load_current_a=0.9)
        op = solve_operating_point(m, prop_map, 14.8, v_inf=0.0)
        omega = op.rpm * 2 * math.pi / 60
        assert m.torque(omega, 14.8) == pytest.approx(op.torque_nm, rel=1e-4)
        assert op.rpm < 14.8 * m.kv_rpm_v, "loaded speed must be below no-load"

    def test_thrust_falls_with_airspeed_at_fixed_voltage(self, prop_map):
        m = Motor("m", kv_rpm_v=1000, resistance_ohm=0.12, no_load_current_a=0.9)
        t = [solve_operating_point(m, prop_map, 14.8, v).thrust_n for v in (0.0, 5.0, 10.0)]
        assert t[0] > t[1] > t[2]

    def test_motor_efficiency_is_plausible(self, prop_map):
        m = Motor("m", kv_rpm_v=1000, resistance_ohm=0.12, no_load_current_a=0.9)
        op = solve_operating_point(m, prop_map, 14.8, 0.0)
        assert 0.4 < op.motor_efficiency < 0.95


@needs_uiuc
class TestAgainstWindTunnelData:
    """The gate from AI-05: measured data, not tests written beside the code."""

    def test_geometry_parses_to_the_right_scale(self):
        from camber.catalogue import uiuc
        db = uiuc.load(UIUC_ROOT)
        g = db.geometry["apce_10x5"]
        assert g.diameter_m == pytest.approx(0.254, abs=1e-4)   # 10 inches
        # Implied pitch at 75% radius should be near the nameplate value.
        s = min(g.stations, key=lambda s: abs(s.r_over_R - 0.75))
        implied = 2 * math.pi * s.r_over_R * g.radius_m * math.tan(s.twist_rad)
        assert 0.8 < implied / g.pitch_m < 1.3

    def test_prediction_error_stays_within_the_documented_bound(self):
        """Regression gate. Loosen this only with a reason recorded in
        DECISIONS.md, never to make a failing build pass."""
        from camber.physics.validate import validate
        rep = validate(root=UIUC_ROOT, airfoil="e63", stations=20, max_props=25)
        s = rep.summary()["ct_scaled"]
        assert abs(s["median"]) < 0.15, f"CT bias drifted: {s['median']:+.1%}"
        assert s["mean_abs"] < 0.25, f"CT scatter grew: {s['mean_abs']:.1%}"
        assert s["within_20pct"] > 0.60
