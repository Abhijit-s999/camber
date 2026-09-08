"""The geometric screen that decides whether a CFD run is worth doing."""

import math
from pathlib import Path

import pytest

from camber.cfd.screen import SKIP_THRESHOLD, estimated_loss, screen, wake_overlap
from camber.spec import io
from camber.spec.units import Vec3

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "quad5.toml"


@pytest.fixture
def vehicle():
    return io.load(EXAMPLE)


def move_rotors(v, radial):
    import math
    for t in v.thrusters:
        r = math.hypot(t.position.x, t.position.y)
        k = radial / r
        t.placement = type(t.placement)(
            Vec3(t.position.x * k, t.position.y * k, t.position.z),
            t.placement.rotation)
    return v


class TestOverlap:
    def test_rotors_far_out_overlap_nothing(self, vehicle):
        overlap, _ = wake_overlap(move_rotors(vehicle, 0.60))
        assert overlap < 0.01

    def test_rotors_over_the_body_overlap_a_lot(self, vehicle):
        overlap, _ = wake_overlap(move_rotors(vehicle, 0.05))
        assert overlap > 0.5

    def test_overlap_falls_monotonically_with_arm_length(self, vehicle):
        import copy
        vals = [wake_overlap(move_rotors(copy.deepcopy(vehicle), r))[0]
                for r in (0.06, 0.10, 0.14, 0.20, 0.30)]
        assert all(a >= b - 0.02 for a, b in zip(vals, vals[1:])), vals

    def test_bigger_discs_overlap_more(self, vehicle):
        import copy
        small = copy.deepcopy(vehicle); big = copy.deepcopy(vehicle)
        for t in small.thrusters: t.radius = 0.045
        for t in big.thrusters: t.radius = 0.090
        assert wake_overlap(big)[0] > wake_overlap(small)[0]

    def test_disc_loading_is_weight_over_disc_area(self, vehicle):
        _, loading = wake_overlap(vehicle)
        area = sum(math.pi * t.radius**2 for t in vehicle.thrusters)
        expected = vehicle.mass_properties().mass * 9.80665 / area
        assert loading == pytest.approx(expected, rel=1e-6)

    def test_no_rotors_is_zero_not_a_crash(self, vehicle):
        vehicle.thrusters.clear()
        assert wake_overlap(vehicle) == (0.0, 0.0)

    def test_deterministic(self, vehicle):
        assert wake_overlap(vehicle, seed=3) == wake_overlap(vehicle, seed=3)


class TestScreen:
    def test_baseline_is_screened_out(self, vehicle):
        """The example's rotors sit clear of the airframe."""
        s = screen(vehicle)
        assert not s.worth_solving and "under a disc" in s.reason

    def test_overlapping_layout_is_worth_solving(self, vehicle):
        s = screen(move_rotors(vehicle, 0.05))
        assert s.worth_solving and s.reason == ""

    def test_score_combines_overlap_and_loading(self, vehicle):
        s = screen(vehicle)
        assert s.score == pytest.approx(s.overlap * s.disc_loading)

    def test_estimated_loss_refuses_to_guess_for_real_cases(self, vehicle):
        with pytest.raises(ValueError, match="needs a CFD run"):
            estimated_loss(move_rotors(vehicle, 0.05))

    def test_estimated_loss_is_zero_for_screened_layouts(self, vehicle):
        assert estimated_loss(vehicle) == 0.0

    def test_threshold_is_conservative(self):
        """Biased toward running: a wasted run costs a minute, a missed
        interference case poisons the calibration."""
        assert SKIP_THRESHOLD <= 0.05


class TestAgainstLabelledCases:
    """The screen's whole justification: it must not miss a real case."""

    def test_misses_nothing_on_the_labelled_campaign(self, vehicle):
        from camber.cfd.campaign import Campaign, build_variant, usable
        root = Path(__file__).resolve().parent.parent / "cases" / "campaign40"
        if not (root / "campaign.json").exists():
            pytest.skip("campaign40 results not present")
        cases = [r for r in usable(Campaign(vehicle, root).load()) if not r.screened]
        if len(cases) < 10:
            pytest.skip("too few labelled cases")
        missed = skipped = 0
        for r in cases:
            s = screen(build_variant(vehicle, r.arm, r.radius), samples=8000)
            if not s.worth_solving:
                skipped += 1
                if r.loss >= 0.01:
                    missed += 1
        assert missed == 0, f"screen skipped {missed} cases with real interference"
        assert skipped >= len(cases) // 3, "screen should save a real fraction of runs"
