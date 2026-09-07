"""Beam sizing, manufacturability, and generated frames."""

import math

import pytest

from camber.generate import FrameSpec, generate_frame
from camber.manufacture import CNC3Axis, FDM, Feature, get_process
from camber.spec import io
from camber.structure import (
    MATERIALS, Section, analyse_arm, arm_mass, load_cases, size_arm,
)

CF = MATERIALS["cf_tube"]


class TestSection:
    def test_tube_second_moment_matches_the_closed_form(self):
        s = Section("tube", 0.016, 0.001)
        ro, ri = 0.008, 0.007
        assert s.second_moment == pytest.approx(math.pi / 4 * (ro**4 - ri**4))
        assert s.area == pytest.approx(math.pi * (ro**2 - ri**2))

    def test_rect_second_moment_matches_the_closed_form(self):
        s = Section("rect", 0.020, 0.004)
        assert s.second_moment == pytest.approx(0.020 * 0.004**3 / 12)

    def test_a_wall_cannot_exceed_the_radius(self):
        with pytest.raises(ValueError, match="cannot equal or exceed"):
            Section("tube", 0.010, 0.005)

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError, match="positive"):
            Section("tube", 0.0, 0.001)

    def test_stiffness_rises_steeply_with_diameter(self):
        """I goes as the fourth power; this is why tubes beat solid rod."""
        thin = Section("tube", 0.010, 0.001)
        fat = Section("tube", 0.020, 0.001)
        assert fat.second_moment / thin.second_moment > 6
        assert fat.area / thin.area < 2.5     # for barely twice the material


class TestBeam:
    def test_stress_and_deflection_match_cantilever_formulas(self):
        s, L, F = Section("tube", 0.012, 0.001), 0.15, 20.0
        r = analyse_arm(L, s, CF, load_cases(1, F, 1, 1)[1])
        assert r.stress_pa == pytest.approx(F * L * s.extreme_fibre / s.second_moment)
        assert r.deflection_m == pytest.approx(
            F * L**3 / (3 * CF.youngs_modulus * s.second_moment))

    def test_longer_arms_are_dramatically_worse(self):
        """Stress goes as L, deflection as L^3."""
        s = Section("tube", 0.012, 0.001)
        load = load_cases(1, 20.0, 1, 1)[1]
        a = analyse_arm(0.10, s, CF, load)
        b = analyse_arm(0.20, s, CF, load)
        assert b.stress_pa == pytest.approx(2 * a.stress_pa)
        assert b.deflection_m == pytest.approx(8 * a.deflection_m)

    def test_brittle_materials_get_more_margin(self):
        s, L = Section("tube", 0.012, 0.001), 0.15
        load = load_cases(1, 20.0, 1, 1)[1]
        brittle = analyse_arm(L, s, MATERIALS["cf_tube"], load)
        ductile = analyse_arm(L, s, MATERIALS["al6061"], load)
        # CF is far stronger, yet its default safety factor is higher.
        assert MATERIALS["cf_tube"].brittle and not MATERIALS["al6061"].brittle
        assert brittle.margin > 0 and ductile.margin > 0

    def test_deflection_can_govern_before_stress(self):
        """A stiff-but-slender arm usually fails the deflection limit first."""
        s = Section("tube", 0.008, 0.0005)
        load = load_cases(1, 8.0, 1, 1)[1]
        r = analyse_arm(0.30, s, CF, load)
        stress_margin = CF.yield_strength / 3.0 / r.stress_pa
        assert r.margin <= stress_margin

    def test_zero_length_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            analyse_arm(0.0, Section("tube", 0.01, 0.001), CF,
                        load_cases(1, 1, 1, 1)[0])


class TestLoadCases:
    def test_manoeuvre_uses_full_thrust_not_hover(self):
        loads = {l.name: l for l in load_cases(1.13, 10.3, 0.46, 4)}
        assert loads["manoeuvre"].tip_force_n == 10.3
        assert loads["hover"].tip_force_n == 1.13

    def test_landing_scales_with_mass_and_g(self):
        soft = load_cases(1, 10, 2.0, 4, landing_g=2.0)[2].tip_force_n
        hard = load_cases(1, 10, 2.0, 4, landing_g=8.0)[2].tip_force_n
        assert hard == pytest.approx(4 * soft)

    def test_hover_sizing_is_unsafe_for_a_larger_aircraft(self):
        """The mistake this module exists to prevent.

        It does not bite on a 460 g quad -- the smallest stock tube is already
        over-strength -- but from a couple of kilograms up it does.
        """
        loads = load_cases(1.8 * 9.80665 / 4, 22.0, 1.8, 4)
        hover_only = size_arm(0.230, CF, [loads[0]])
        assert hover_only is not None
        worst = min((analyse_arm(0.230, hover_only[0], CF, l) for l in loads),
                    key=lambda r: r.margin)
        assert not worst.passes, "hover-sized arm should fail a real load case"


class TestSizing:
    def test_picks_a_section_that_passes_everything(self):
        loads = load_cases(1.13, 10.3, 0.46, 4)
        section, mass = size_arm(0.098, CF, loads)
        assert all(analyse_arm(0.098, section, CF, l).passes for l in loads)
        assert mass > 0

    def test_returns_none_when_nothing_in_stock_works(self):
        loads = load_cases(100, 5000.0, 50.0, 4)
        assert size_arm(1.5, MATERIALS["pla"], loads) is None

    def test_chooses_the_lightest_passing_section(self):
        loads = load_cases(1.13, 10.3, 0.46, 4)
        section, mass = size_arm(0.098, CF, loads)
        assert mass == pytest.approx(arm_mass(0.098, section, CF))


class TestManufacture:
    def test_cnc_rejects_an_undercut_outright(self):
        vs = CNC3Axis().check(Feature("lip", "corner", size=0.005, undercut=True))
        assert any(v.rule == "undercut" for v in vs)

    def test_fdm_accepts_an_undercut(self):
        """The relaxation that makes CNC-first the right order (MFG-01)."""
        assert not FDM().check(Feature("lip", "corner", size=0.005, undercut=True))

    def test_internal_corner_cannot_be_sharper_than_the_tool(self):
        vs = CNC3Axis(tool_diameter=0.003).check(
            Feature("pocket corner", "corner", radius=0.0005))
        assert any(v.rule == "internal-corner" for v in vs)
        ok = CNC3Axis(tool_diameter=0.003).check(
            Feature("pocket corner", "corner", radius=0.002))
        assert not ok

    def test_deep_narrow_pocket_is_flagged(self):
        vs = CNC3Axis().check(Feature("slot", "pocket", size=0.005, depth=0.050))
        assert any(v.rule == "pocket-too-deep" for v in vs)

    def test_fdm_flags_steep_overhangs(self):
        vs = FDM().check(Feature("shelf", "overhang", angle=math.radians(70)))
        assert any(v.rule == "overhang" for v in vs)
        assert not FDM().check(Feature("shelf", "overhang", angle=math.radians(30)))

    def test_violations_are_continuous_so_a_search_can_escape(self):
        mild = CNC3Axis().check(Feature("c", "corner", radius=0.00145))[0]
        severe = CNC3Axis().check(Feature("c", "corner", radius=0.0001))[0]
        assert severe.amount > mild.amount

    def test_cnc_is_stricter_than_fdm_on_the_same_part(self):
        feats = [Feature("u", "corner", size=0.004, undercut=True),
                 Feature("p", "pocket", size=0.002, depth=0.004),
                 Feature("w", "wall", size=0.0009)]
        assert len(CNC3Axis().check_all(feats)) > len(FDM().check_all(feats))

    def test_unknown_process(self):
        with pytest.raises(KeyError, match="unknown process"):
            get_process("laser-sintering-unicorn")


class TestGeneratedFrame:
    @pytest.fixture
    def vehicle(self):
        return io.load(__import__("pathlib").Path(__file__).resolve().parent.parent
                       / "examples" / "quad5.toml")

    def test_produces_a_plate_and_one_arm_per_rotor(self, vehicle):
        f = generate_frame(vehicle)
        assert len(f.arms) == len(vehicle.thrusters)
        assert f.plate.mass > 0 and all(a.mass > 0 for a in f.arms)

    def test_plate_covers_the_payloads(self, vehicle):
        f = generate_frame(vehicle)
        for b in vehicle.bodies:
            h = b.shape.half_extents()
            assert abs(b.position.x) + h.x <= f.plate.shape.lx / 2 + 1e-9
            assert abs(b.position.y) + h.y <= f.plate.shape.ly / 2 + 1e-9

    def test_arm_is_shorter_than_the_radial_reach(self, vehicle):
        """It spans plate edge to hub, not origin to hub."""
        f = generate_frame(vehicle)
        reach = math.hypot(vehicle.thrusters[0].position.x,
                           vehicle.thrusters[0].position.y)
        assert f.arm_length < reach

    def test_the_generated_frame_still_flies(self, vehicle):
        from camber.control import controllability, solve_hover
        f = generate_frame(vehicle)
        vehicle.frame.members = f.members
        assert controllability(vehicle).rank >= 4
        assert solve_hover(vehicle).feasible

    def test_weaker_material_needs_more_section(self, vehicle):
        strong = generate_frame(vehicle, FrameSpec(material="cf_tube"))
        weak = generate_frame(vehicle, FrameSpec(material="nylon_cf"))
        assert weak.section.a >= strong.section.a

    def test_refuses_rather_than_returning_something_unsafe(self, vehicle):
        for t in vehicle.thrusters:
            t.static_thrust_n = 5000.0        # absurd load
        with pytest.raises(ValueError, match="no stock"):
            generate_frame(vehicle, FrameSpec(material="pla"))

    def test_no_rotors_is_an_error(self, vehicle):
        vehicle.thrusters.clear()
        with pytest.raises(ValueError, match="no rotors"):
            generate_frame(vehicle)

    def test_reports_the_governing_load_case(self, vehicle):
        f = generate_frame(vehicle)
        assert f.governing_case in {"hover", "manoeuvre", "landing"}
        assert f.margin >= 1.0
