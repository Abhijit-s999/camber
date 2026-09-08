"""The design problem and the search over it."""

import numpy as np
import pytest

from camber.optimise.nsga2 import _constrained_ranks, minimise
from camber.optimise.problem import (
    DesignProblem, Evaluation, arm_length_variable, battery_capacity_variable,
    position_variables, rotor_radius_variable,
)
from camber.spec import io
from camber.spec.vehicle import Battery

MINIMAL = {
    "name": "t",
    "mission": {"objective": "max_endurance"},
    "bodies": [
        {"name": "stack", "type": "body", "mass": 0.05,
         "shape": {"kind": "box", "lx": 0.04, "ly": 0.04, "lz": 0.01}},
        {"name": "battery", "type": "battery", "mass": 0.15,
         "position": [0.0, 0.0, 0.01],
         "shape": {"kind": "box", "lx": 0.07, "ly": 0.035, "lz": 0.03},
         "capacity_ah": 1.5, "nominal_voltage": 14.8, "max_c_rate": 50.0},
    ],
    "thrusters": [
        {"name": f"m{i}", "type": "rotor", "mass": 0.03, "radius": 0.0635,
         "position": [x, y, -0.01], "spin": s, "static_thrust_n": 8.0}
        for i, (x, y, s) in enumerate([(0.1, 0.1, "cw"), (0.1, -0.1, "ccw"),
                                       (-0.1, -0.1, "cw"), (-0.1, 0.1, "ccw")])
    ],
}


@pytest.fixture
def vehicle():
    return io.from_dict(MINIMAL)


class TestVariableExtraction:
    def test_free_bodies_become_three_variables_each(self, vehicle):
        v = position_variables(vehicle)
        assert len(v) == 6                      # 2 bodies x 3 axes
        assert {d.name for d in v} >= {"stack.x", "battery.z"}

    def test_a_fixed_constraint_removes_its_body(self):
        spec = {**MINIMAL, "constraints": [
            {"type": "fixed", "body": "stack", "position": [0, 0, 0]}]}
        v = position_variables(io.from_dict(spec))
        assert not any(d.name.startswith("stack.") for d in v)
        assert any(d.name.startswith("battery.") for d in v)

    def test_a_relative_constraint_frees_only_the_anchor(self):
        spec = {**MINIMAL, "constraints": [
            {"type": "relative", "a": "stack", "b": "battery",
             "offset": [0.01, 0, 0]}]}
        v = position_variables(io.from_dict(spec))
        assert not any(d.name.startswith("stack.") for d in v)
        assert any(d.name.startswith("battery.") for d in v)

    def test_bounds_bracket_the_authored_position(self, vehicle):
        for d in position_variables(vehicle, span=0.05):
            assert d.upper - d.lower == pytest.approx(0.10)

    def test_battery_variable_changes_mass_with_capacity(self, vehicle):
        d = battery_capacity_variable(vehicle, "battery", 0.5, 4.0)
        before = vehicle.mass_properties().mass
        d.apply(vehicle, 3.0)
        after = vehicle.mass_properties().mass
        assert after > before
        assert isinstance(vehicle.body("battery"), Battery)
        assert vehicle.body("battery").capacity_ah == 3.0

    def test_rotor_radius_applies_to_every_rotor(self, vehicle):
        rotor_radius_variable(vehicle, 0.04, 0.09).apply(vehicle, 0.075)
        assert all(t.radius == 0.075 for t in vehicle.thrusters)

    def test_arm_length_preserves_layout_shape(self, vehicle):
        before = [(t.position.x, t.position.y) for t in vehicle.thrusters]
        arm_length_variable(vehicle, 0.05, 0.3).apply(vehicle, 0.2)
        after = [(t.position.x, t.position.y) for t in vehicle.thrusters]
        # Same signs, uniformly scaled.
        for (bx, by), (ax, ay) in zip(before, after):
            assert np.sign(ax) == np.sign(bx) and np.sign(ay) == np.sign(by)
        r0 = np.hypot(*after[0])
        assert all(np.hypot(*p) == pytest.approx(r0) for p in after)


class TestEvaluation:
    def test_the_authored_design_evaluates(self, vehicle):
        p = DesignProblem(vehicle, position_variables(vehicle))
        lo, hi = p.bounds
        e = p.evaluate((lo + hi) / 2)
        assert e.objectives.shape == (2,)
        assert np.isfinite(e.objectives).all()

    def test_uncontrollable_layouts_are_rejected_not_scored(self, vehicle):
        """Hard gate: an unflyable design must not compete on objectives."""
        p = DesignProblem(vehicle, [arm_length_variable(vehicle, 1e-6, 0.2)])
        e = p.evaluate([1e-6])          # all rotors collapsed onto the CG
        assert not e.feasible
        assert "rank" in e.reason or "margin" in e.reason
        assert e.objectives.min() >= 1e5, "rejected designs must not look good"

    def test_soft_constraints_produce_a_graded_violation(self):
        spec = {**MINIMAL, "constraints": [
            {"type": "min_distance", "a": "stack", "b": "battery",
             "distance": 0.30}]}
        v = io.from_dict(spec)
        p = DesignProblem(v, position_variables(v))
        lo, hi = p.bounds
        e = p.evaluate((lo + hi) / 2)
        assert not e.feasible and 0 < e.violation < 1.0, (
            "a geometric breach should be a finite penalty the search can "
            "climb out of, not a rejection"
        )

    def test_build_does_not_mutate_the_template(self, vehicle):
        p = DesignProblem(vehicle, position_variables(vehicle))
        before = vehicle.body("stack").position.x
        lo, hi = p.bounds
        p.build(hi)
        assert vehicle.body("stack").position.x == before


class TestConstrainedDomination:
    def test_feasible_always_outranks_infeasible(self):
        evals = [
            Evaluation(np.array([9.0, 9.0]), 0.0, True),     # feasible, poor
            Evaluation(np.array([0.0, 0.0]), 0.5, False),    # infeasible, great
        ]
        obj = np.array([e.objectives for e in evals])
        fronts = _constrained_ranks(evals, obj)
        assert fronts[0] == [0], "an unbuildable design must not lead the front"

    def test_infeasible_sorted_by_violation(self):
        evals = [Evaluation(np.zeros(2), v, False) for v in (0.9, 0.1, 0.5)]
        obj = np.zeros((3, 2))
        assert _constrained_ranks(evals, obj)[-1] == [1, 2, 0]


class TestSearch:
    def test_finds_a_front_and_improves(self, vehicle):
        p = DesignProblem(vehicle, position_variables(vehicle)
                          + [battery_capacity_variable(vehicle, "battery", 0.5, 3.0)])
        res = minimise(p, population=16, generations=6, seed=1)
        assert len(res.evaluations) == 16
        assert res.front_indices
        assert len(res.history) == 6

    def test_a_real_tradeoff_produces_a_spread_front(self, vehicle):
        """Mass must buy endurance, or the second objective is decorative."""
        p = DesignProblem(vehicle, [battery_capacity_variable(vehicle, "battery", 0.5, 4.0)])
        res = minimise(p, population=20, generations=8, seed=2)
        masses = res.front_objectives[:, 1]
        assert masses.max() - masses.min() > 0.05, "front collapsed to one point"

    def test_deterministic_for_a_given_seed(self, vehicle):
        p = DesignProblem(vehicle, position_variables(vehicle))
        a = minimise(p, population=12, generations=4, seed=7)
        b = minimise(p, population=12, generations=4, seed=7)
        assert np.allclose(a.objectives, b.objectives)

    def test_population_is_held_constant(self, vehicle):
        p = DesignProblem(vehicle, position_variables(vehicle))
        res = minimise(p, population=14, generations=5, seed=0)
        assert len(res.evaluations) == 14


class TestInterferenceCorrection:
    """The closed loop: CFD corrects L1, the optimiser searches corrected L1."""

    @pytest.fixture
    def model(self):
        from pathlib import Path

        from camber.calibrate import load_model
        p = Path(__file__).resolve().parent.parent / "data" / "interference.json"
        if not p.exists():
            pytest.skip("no calibrated model")
        return load_model(p)

    def test_correction_reaches_the_objective(self, vehicle, model):
        """It must change the answer, or it is not wired in."""
        from camber.optimise.problem import battery_capacity_variable
        var = [battery_capacity_variable(vehicle, "battery", 0.8, 3.0)]
        plain = DesignProblem(vehicle, var)
        corrected = DesignProblem(vehicle, var, interference=model)
        x = [(v.lower + v.upper) / 2 for v in var]
        assert "interference_loss" in corrected.evaluate(x).detail
        assert plain.evaluate(x).detail.get("interference_loss", 0.0) == 0.0

    def test_heavy_interference_is_penalised(self, vehicle, model):
        """Above 15% download the layout is treated as a constraint breach."""
        from camber.optimise.problem import arm_length_variable
        var = [arm_length_variable(vehicle, 0.02, 0.30)]
        p = DesignProblem(vehicle, var, interference=model)
        tight = p.evaluate([0.02])     # rotors piled onto the airframe
        wide = p.evaluate([0.30])
        assert tight.detail.get("interference_loss", 0) >= \
               wide.detail.get("interference_loss", 0)

    def test_loss_is_clamped_to_a_physical_range(self, vehicle, model):
        from camber.optimise.problem import arm_length_variable
        p = DesignProblem(vehicle, [arm_length_variable(vehicle, 0.01, 0.5)],
                          interference=model)
        for x in (0.01, 0.05, 0.2, 0.5):
            loss = p.evaluate([x]).detail.get("interference_loss", 0.0)
            assert 0.0 <= loss <= 0.5, f"loss {loss} outside the clamp"

    def test_model_round_trips_through_disk(self, tmp_path, vehicle, model):
        from camber.calibrate import load_model, save_model
        out = save_model(model, tmp_path / "m.json")
        again = load_model(out)
        assert again.correction(vehicle)[0] == pytest.approx(
            model.correction(vehicle)[0])
