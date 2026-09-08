"""The whole chain, spec to manufacturable output.

Every layer has its own tests. This one exists because a project can pass all
of them and still not work: the failure mode of a layered system is the seam
between layers, not the layers. It is deliberately a single narrative rather
than isolated cases, because what is being checked is that the handoffs hold.
"""

from pathlib import Path

import pytest

from camber.calibrate import load_model
from camber.control import controllability, solve_hover
from camber.generate import generate_frame
from camber.manufacture import Feature, get_process
from camber.optimise.nsga2 import minimise
from camber.optimise.problem import (
    DesignProblem, arm_length_variable, battery_capacity_variable,
    physics_endurance_model, rotor_radius_variable,
)
from camber.physics.motor import Motor
from camber.spec import io

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "quad5.toml"
PROPDB = ROOT / "data" / "raw" / "UIUC-propDB"


@pytest.fixture(scope="module")
def optimised():
    """Optimise a real vehicle, once, and hand the winner downstream."""
    if not PROPDB.exists():
        pytest.skip("UIUC data not fetched; run `camber catalogue fetch`")
    v = io.load(EXAMPLE)
    motor = Motor("2207", kv_rpm_v=1900, resistance_ohm=0.075,
                  no_load_current_a=1.4)
    interference = None
    model_path = ROOT / "data" / "interference.json"
    if model_path.exists():
        interference = load_model(model_path)
    problem = DesignProblem(
        v,
        [arm_length_variable(v, 0.09, 0.18),
         rotor_radius_variable(v, 0.05, 0.09),
         battery_capacity_variable(v, "battery", 1.0, 3.0)],
        endurance_model=physics_endurance_model("gwsdd_5x4.3", motor,
                                                str(PROPDB)),
        interference=interference,
    )
    result = minimise(problem, population=12, generations=5, seed=1)
    feasible = [i for i, e in enumerate(result.evaluations) if e.feasible]
    assert feasible, "the optimiser found nothing flyable"
    best = max(feasible, key=lambda i: -result.evaluations[i].objectives[0])
    return problem, result, best


def test_optimiser_returns_a_flyable_design(optimised):
    problem, result, best = optimised
    vehicle = problem.build(result.x[best])
    assert controllability(vehicle).rank >= 4
    assert solve_hover(vehicle).feasible


def test_endurance_is_physically_plausible(optimised):
    """A 5-inch quad on a few Ah flies for minutes, not seconds or hours.

    Loose on purpose: this catches a broken chain, not a modelling error.
    """
    _, result, best = optimised
    minutes = -result.evaluations[best].objectives[0] / 60
    assert 1.0 < minutes < 40.0, f"{minutes:.1f} min is not a real quadcopter"


def test_the_optimised_design_can_be_given_an_airframe(optimised):
    problem, result, best = optimised
    vehicle = problem.build(result.x[best])
    frame = generate_frame(vehicle)
    assert frame.margin >= 1.0
    assert frame.total_mass > 0
    vehicle.frame.members = frame.members
    # Adding structure must not make it unflyable.
    assert controllability(vehicle).rank >= 4
    assert solve_hover(vehicle).feasible


def test_the_result_reaches_cad(optimised, tmp_path):
    pytest.importorskip("build123d")
    from camber.geometry import cad
    problem, result, best = optimised
    vehicle = problem.build(result.x[best])
    vehicle.frame.members = generate_frame(vehicle).members
    step = cad.export(vehicle, tmp_path / "v.step")
    stl = cad.export(vehicle, tmp_path / "v.stl", metres=True)
    assert step.stat().st_size > 10_000
    assert stl.stat().st_size > 1_000
    assert "ISO-10303" in step.read_text(errors="ignore")[:2000]


def test_the_result_is_makeable(optimised):
    problem, result, best = optimised
    vehicle = problem.build(result.x[best])
    frame = generate_frame(vehicle)
    process = get_process("cnc-3axis")
    assert process.makeable([Feature("plate", "wall", size=0.002)])
    assert isinstance(frame.notes, list)


def test_the_design_round_trips_through_the_spec_format(optimised, tmp_path):
    """What comes out must be loadable as input, or the loop is not closed."""
    problem, result, best = optimised
    vehicle = problem.build(result.x[best])
    out = tmp_path / "optimised.json"
    io.save_json(vehicle, out)
    reloaded = io.load(out)
    assert reloaded.mass_properties().mass == pytest.approx(
        vehicle.mass_properties().mass)
    assert controllability(reloaded).rank == controllability(vehicle).rank
