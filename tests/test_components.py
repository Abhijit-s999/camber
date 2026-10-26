"""Component selection: the motor, propeller and cell count are chosen (CAT-01).

Until this existed the optimiser searched positions and one shared radius, and
the motor was three CLI flags. These tests hold the pieces that make the
choice honest: a motor table whose every number can be found on the page it
cites, propellers flown on their own measurements, categorical genes that
really explore, and limits that reject a pairing rather than hide it.
"""

import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from camber.catalogue import motors as motor_table
from camber.catalogue import props as prop_table
from camber.catalogue import uiuc
from camber.optimise.components import ComponentSpace
from camber.optimise.nsga2 import _categorical, minimise
from camber.optimise.problem import (DesignVariable, Evaluation,
                                     prop_overlap)
from camber.physics.battery import CELL_NOMINAL_V, Pack
from camber.physics.motor import Motor
from camber.physics.propulsion import (MeasuredPropeller, solve_operating_point,
                                       thrust_for)
from camber.spec import io
from camber.spec.units import Vec3

ROOT = Path(__file__).resolve().parent.parent
PROPDB = ROOT / "data" / "raw" / "UIUC-propDB"
needs_propdb = pytest.mark.skipif(not PROPDB.exists(),
                                  reason="UIUC data not fetched; `camber catalogue fetch`")


def flat_prop(diameter=0.127, ct=0.11, cp=0.05) -> MeasuredPropeller:
    """A propeller whose coefficients do not move: easy to reason about."""
    rpm = np.array([1000.0, 10000.0])
    return MeasuredPropeller(diameter, rpm, np.full(2, ct), np.full(2, cp), (), "flat")


def motor_row(**kw) -> motor_table.CatalogueMotor:
    base = dict(id="test-2207-1950", maker="Test", model="2207", kv=1950.0,
                resistance_ohm=0.06, no_load_current_a=1.3, mass_kg=0.032,
                max_current_a=45.0, cells_min=3, cells_max=6, url="https://example.org/m")
    base.update(kw)
    return motor_table.CatalogueMotor(**base)


def prop_row(key="gwsdd_5x4.3", diameter=0.127, mass=0.004, **kw) -> prop_table.CatalogueProp:
    return prop_table.CatalogueProp(
        key, prop_table.FAMILIES["gwsdd"], diameter, 0.109, 2, mass,
        "https://example.org/p", flat_prop(diameter, **kw))


# -- the motor table -------------------------------------------------------

class TestTheMotorTable:
    @property
    def rows(self):
        return tomllib.loads(motor_table.TABLE.read_text())["motor"]

    def test_it_loads_and_is_big_enough_to_choose_from(self):
        table = motor_table.load()
        assert len(table) >= 30
        assert len({m.maker for m in table}) >= 5

    def test_it_spans_small_quads_to_heavy_lifters(self):
        kv = [m.kv for m in motor_table.load()]
        assert min(kv) < 700 and max(kv) > 3000

    def test_every_number_is_on_the_page_it_cites(self):
        """AI-06: each value can be checked against the quoted spec block
        without trusting whoever typed it into the table."""
        for r in self.rows:
            q = r["quote"]
            for k in ("kv", "no_load_current_a", "mass_g", "max_current_a"):
                assert f"{r[k]:g}" in q, f"{r['id']}: {k} = {r[k]:g} is not in its quote"
            rm = r["resistance_mohm"]
            assert f"{rm:g}" in q or f"{rm / 1000:g}" in q, \
                f"{r['id']}: resistance {rm:g} mOhm is not in its quote"
            assert r["url"].startswith("https://")

    def test_a_row_missing_what_the_physics_needs_is_refused(self):
        row = dict(self.rows[0])
        del row["no_load_current_a"]
        with pytest.raises(motor_table.MotorTableError, match="no_load_current_a"):
            motor_table.parse([row])

    def test_the_resistance_is_converted_to_ohms(self):
        r = self.rows[0]
        m = next(x for x in motor_table.load() if x.id == r["id"])
        assert m.resistance_ohm == pytest.approx(r["resistance_mohm"] / 1000)


# -- propellers --------------------------------------------------------------

class TestPropellerNames:
    @pytest.mark.parametrize("key, dia, pitch, blades", [
        ("apce_10x5", 10, 5, 2),
        ("apce_14x85", 14, 8.5, 2),        # the archive dropped the point
        ("ancf_125x6", 12.5, 6, 2),
        ("ancf_12x13", 12, 13, 2),         # a real pitch above the diameter
        ("da4022_5x3.75_3b", 5, 3.75, 3),
        ("ef_130x70", 130, 70, 2),         # millimetres: left alone, filtered later
    ])
    def test_sizes_parse_as_the_archive_means_them(self, key, dia, pitch, blades):
        _, d, p, b = uiuc.parse_name(key)
        assert d / 0.0254 == pytest.approx(dia)
        assert p / 0.0254 == pytest.approx(pitch)
        assert b == blades

    def test_a_name_without_the_archive(self):
        assert prop_table.describe("apcsf_10x4.7") == "APC Slow Flyer 10x4.7"


class TestMeasuredPropeller:
    def test_thrust_is_the_measured_coefficient(self):
        p = flat_prop()
        omega = 2 * math.pi * 5000 / 60
        thrust, torque = p.thrust_torque(omega, 0.0)
        n = 5000 / 60
        assert thrust == pytest.approx(0.11 * 1.225 * n ** 2 * 0.127 ** 4)
        assert torque * omega == pytest.approx(0.05 * 1.225 * n ** 3 * 0.127 ** 5)

    def test_beyond_the_measured_rpm_the_end_value_is_held(self):
        rpm = np.array([2000.0, 8000.0])
        p = MeasuredPropeller(0.127, rpm, np.array([0.10, 0.12]), np.array([0.05, 0.06]))
        assert p.coefficients(30000.0) == pytest.approx((0.12, 0.06))
        assert p.coefficients(100.0) == pytest.approx((0.10, 0.05))

    def test_forward_flight_starts_from_static(self):
        run = (5000.0, np.array([0.2, 0.6]), np.array([0.08, 0.02]), np.array([0.05, 0.03]))
        p = MeasuredPropeller(0.127, np.array([1000.0, 9000.0]), np.full(2, 0.1),
                              np.full(2, 0.05), (run,))
        assert p.coefficients(5000, 0.0)[0] == pytest.approx(0.1)
        assert p.coefficients(5000, 0.1)[0] == pytest.approx(0.09)
        assert p.coefficients(5000, 0.6)[0] == pytest.approx(0.02)


@needs_propdb
class TestThePropellerCatalogue:
    @property
    def props(self):
        return prop_table.load(str(PROPDB))

    def test_only_what_can_be_bought(self):
        assert self.props
        assert not [p for p in self.props if not p.family.retail]

    def test_no_size_from_millimetres(self):
        assert max(p.diameter_in for p in self.props) <= prop_table.MAX_DIAMETER_IN

    def test_every_propeller_has_a_mass_and_says_where_from(self):
        for p in self.props:
            assert 0.0001 < p.mass_kg < 0.3, p.key      # 0.2 g to 110 g, published
            assert p.mass_source

    def test_three_blade_micro_invent_is_three_blades(self):
        mit = [p for p in self.props if p.key.startswith("mit_")]
        assert mit and all(p.blades == 3 for p in mit)

    def test_published_weights_are_used_as_published(self):
        published = prop_table.published_masses()
        for p in self.props:
            if p.key in published:
                assert p.mass_kg == published[p.key][0]
                assert p.mass_source.startswith("https://")


# -- the categorical gene (OPT-08) -------------------------------------------

class TestCategoricalGenes:
    def test_crossover_copies_a_parent_and_never_averages(self):
        rng = np.random.default_rng(0)
        parents = np.array([[3.0, 0.5], [9.0, 0.7]] * 50)
        mask = np.array([True, False])
        lo, hi = np.array([0.0, 0.0]), np.array([29.0, 1.0])
        kids = _categorical(parents, parents.copy(), mask, lo, hi, rng, prob=0.0)
        assert set(kids[:, 0]) == {3.0, 9.0}

    def test_mutation_reaches_both_ends_of_the_list(self):
        rng = np.random.default_rng(1)
        x = np.full((4000, 1), 14.0)
        out = _categorical(x, x.copy(), np.array([True]), np.array([0.0]),
                           np.array([29.0]), rng, prob=1.0)
        counts = np.bincount(out[:, 0].astype(int), minlength=30)
        assert counts.min() > 0.5 * counts.mean()      # 0 and 29 included

    def test_a_search_visits_every_option(self):
        """The acceptance test for OPT-08: every index reachable
        within a modest budget, on a problem where none is special."""
        seen = set()

        class Toy:
            n_var = 1
            bounds = (np.array([0.0]), np.array([29.0]))
            categorical = np.array([True])

            def evaluate(self, x):
                i = int(round(x[0]))
                seen.add(i)
                return Evaluation(np.array([float(i), float(-i)]), 0.0, True)

        minimise(Toy(), population=16, generations=25, seed=3)
        assert seen == set(range(30))

    def test_an_option_at_the_end_of_the_list_is_found(self):
        """SBX put children between their parents, so index 0 and the last
        index were the hardest to reach: the bias OPT-08 removes."""
        class Toy:
            n_var = 1
            bounds = (np.array([0.0]), np.array([39.0]))
            categorical = np.array([True])

            def evaluate(self, x):
                i = int(round(x[0]))
                return Evaluation(np.array([float(i != 39), 0.0]), 0.0, True)

        r = minimise(Toy(), population=8, generations=30, seed=0)
        assert 39 in {int(round(v)) for v in r.x[:, 0]}

    def test_a_categorical_variable_rounds(self):
        v = DesignVariable("motor", 0, 9, lambda *_: None, categorical=True)
        assert v.integral and v.clip(3.4) == 3


# -- the pack's current (PHYS-10) ---------------------------------------------

def test_the_pack_supplies_throttle_times_the_motor_current():
    """At part throttle the ESC gives the motor a fraction of the pack's
    voltage, and power is conserved, so the pack sees that fraction of the
    motor's current. Counting motor current as pack current halved every
    hover endurance."""
    motor = Motor("m", kv_rpm_v=1950, resistance_ohm=0.06, no_load_current_a=1.3)
    pack = Pack("p", cells=4, capacity_ah=1.5, resistance_ohm=0.018, max_c_rate=75)
    op = thrust_for(motor, flat_prop(), pack, 2.0)
    v_pack = pack.terminal_voltage(0.0, 1.0)
    throttle = op.voltage_v / v_pack
    assert 0.1 < throttle < 0.9
    assert op.supply_current(v_pack) == pytest.approx(throttle * op.current_a)


# -- choosing, on a real vehicle ---------------------------------------------

def racer():
    return io.load(ROOT / "examples" / "racer5.toml")


class TestResolve:
    def test_a_choice_sets_the_rotor_from_the_parts(self):
        space = ComponentSpace([motor_row()], [prop_row()], cells=(4, 4))
        v = racer()
        space.pin(v)
        c = space.resolve(v)
        expected = solve_operating_point(motor_row().motor, flat_prop(), 4 * CELL_NOMINAL_V)
        for t in v.thrusters:
            assert t.motor_sku == "test-2207-1950" and t.prop_sku == "gwsdd_5x4.3"
            assert t.mass == pytest.approx(0.036)
            assert t.radius == pytest.approx(0.0635)
            assert t.static_thrust_n == pytest.approx(expected.thrust_n)
        assert c.violations == {}

    def test_overcurrent_is_a_violation_with_its_size(self):
        space = ComponentSpace([motor_row(max_current_a=5.0)], [prop_row()], cells=(4, 4))
        v = racer()
        space.pin(v)
        c = space.resolve(v)
        assert c.violations["motor current"] == pytest.approx(c.full_current_a / 5.0 - 1)

    def test_a_pack_outside_the_motors_rating_is_a_violation(self):
        space = ComponentSpace([motor_row(cells_min=5, cells_max=6)], [prop_row()],
                               cells=(3, 3))
        v = racer()
        space.pin(v)
        assert space.resolve(v).violations["motor cells"] == 2

    def test_a_tip_past_the_compressibility_limit_is_a_violation(self):
        space = ComponentSpace([motor_row(kv=9000, max_current_a=1e9)],
                               [prop_row(diameter=0.254, mass=0.02, ct=0.02, cp=0.001)],
                               cells=(6, 6))
        v = racer()
        space.pin(v)
        assert "tip speed" in space.resolve(v).violations

    def test_the_cell_count_moves_the_pack_with_it(self):
        space = ComponentSpace([motor_row()], [prop_row()], cells=(3, 6))
        v = racer()
        cells = next(x for x in space.variables(v) if x.name == "battery.cells")
        cells.apply(v, 6)
        pack = v.batteries[0]
        assert pack.nominal_voltage == pytest.approx(6 * CELL_NOMINAL_V)
        assert pack.mass == pytest.approx(pack.capacity_ah * pack.nominal_voltage / 180)


def test_neighbouring_discs_may_not_overlap():
    v = racer()
    assert prop_overlap(v) == 0.0
    for t in v.thrusters:
        p = t.position
        t.placement = type(t.placement)(Vec3(p.x * 0.6, p.y * 0.6, p.z), t.placement.rotation)
    assert prop_overlap(v) > 0
    # Stacked rotors may overlap in plan.
    for i, t in enumerate(v.thrusters):
        p = t.position
        t.placement = type(t.placement)(Vec3(p.x, p.y, p.z - 0.02 * (i % 2)),
                                        t.placement.rotation)
    assert prop_overlap(v) < 1e-9


# -- the whole thing, from the command line --------------------------------

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """`camber optimise` with nothing but the spec, as the README runs it."""
    from camber.cli import main
    out = tmp_path_factory.mktemp("opt")
    code = main(["optimise", str(ROOT / "examples" / "quad5.toml"),
                 "--propdb", str(PROPDB), "--population", "32",
                 "--generations", "20", "--seed", "1",
                 "--front-json", str(out / "front.json"),
                 "--out", str(out / "best.json")])
    return code, out


@needs_propdb
class TestTheOptimiserChooses:
    def test_every_design_on_the_front_names_its_parts(self, run):
        code, out = run
        assert code == 0
        front = json.loads((out / "front.json").read_text())
        assert front["endurance_model"] == "catalogue"
        ids = {m.id for m in motor_table.load()}
        for p in front["points"]:
            c = p["components"]
            assert c["motor"]["id"] in ids and c["motor"]["url"].startswith("https://")
            assert c["prop"]["key"] and 2 <= c["cells"] <= 6

    def test_the_saved_design_flies_the_chosen_hardware(self, run):
        _, out = run
        v = io.load(out / "best.json")
        t = v.thrusters[0]
        motor = next(m for m in motor_table.load() if m.id == t.motor_sku)
        prop = next(p for p in prop_table.load(str(PROPDB)) if p.key == t.prop_sku)
        cells = round(v.batteries[0].nominal_voltage / CELL_NOMINAL_V)
        op = solve_operating_point(motor.motor, prop.measured, cells * CELL_NOMINAL_V)
        assert t.static_thrust_n == pytest.approx(op.thrust_n, rel=1e-6)
        assert t.radius == pytest.approx(prop.diameter_m / 2)
        assert t.mass == pytest.approx(motor.mass_kg + prop.mass_kg)

    def test_a_part_that_is_not_in_the_table_is_an_error(self, capsys):
        from camber.cli import main
        assert main(["optimise", str(ROOT / "examples" / "quad5.toml"),
                     "--propdb", str(PROPDB), "--motor", "no-such-motor"]) == 2
        assert "catalogue motors" in capsys.readouterr().err

    def test_pinning_the_parts_still_searches_the_layout(self, tmp_path):
        from camber.cli import main
        assert main(["optimise", str(ROOT / "examples" / "racer5.toml"),
                     "--propdb", str(PROPDB), "--motor", "emax-e32207-2400",
                     "--prop", "gwsdd_5x4.3", "--cells", "4",
                     "--population", "16", "--generations", "6",
                     "--out", str(tmp_path / "b.json")]) == 0
        t = io.load(tmp_path / "b.json").thrusters[0]
        assert (t.motor_sku, t.prop_sku) == ("emax-e32207-2400", "gwsdd_5x4.3")


def test_the_screen_starts_the_search_from_the_best_pairings():
    """CAT-05: every part combination is tried at the author's layout before
    the genetic search begins, so which motor it ends on does not depend on
    the seed. The best pairing here is the one with the right motor for the
    prop, and it must be the first row."""
    from camber.optimise.components import screen
    from camber.optimise.problem import DesignProblem
    good = motor_row(id="good")
    weak = motor_row(id="weak", kv=300.0)          # too slow to lift the racer
    hot = motor_row(id="hot", max_current_a=3.0)   # over its rating
    space = ComponentSpace([weak, hot, good], [prop_row()], cells=(4, 4))
    v = racer()
    problem = DesignProblem(v, space.variables(v), endurance_model=space.endurance_model(),
                            components=space)
    rows = screen(problem, keep=2)
    assert len(rows) == 2
    assert problem.resolve(rows[0])[1].motor.id == "good"


def test_the_core_still_runs_on_numpy_alone(tmp_path):
    """AI-07: `inspect` and `optimise` run with numpy and nothing else. The
    motor and propeller models need scipy, so without it `inspect` still names
    the chosen parts and `optimise` falls back, saying why, not a traceback."""
    import subprocess
    import sys
    v = racer()
    for t in v.thrusters:
        t.motor_sku, t.prop_sku = "emax-e32207-2400", "gwsdd_5x4.3"
    spec = tmp_path / "chosen.json"
    io.save_json(v, spec)
    block = ("import sys, importlib.abc\n"
             "class No(importlib.abc.MetaPathFinder):\n"
             "    def find_spec(self, name, path, target=None):\n"
             "        if name.split('.')[0] == 'scipy':\n"
             "            raise ImportError('no scipy here')\n"
             "sys.meta_path.insert(0, No())\n"
             "from camber.cli import main\n")
    import os
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    run = lambda *argv: subprocess.run(  # noqa: E731
        [sys.executable, "-c", block + f"sys.exit(main({list(argv)!r}))"],
        capture_output=True, text=True, cwd=tmp_path, env=env)
    r = run("inspect", str(spec))
    assert "Traceback" not in r.stderr, r.stderr
    assert "EMAX E3 Series 2207 2400Kv" in r.stdout
    assert "GWS Direct Drive 5x4.3" in r.stdout
    r = run("optimise", str(spec), "--population", "8", "--generations", "2")
    assert "Traceback" not in r.stderr, r.stderr
    assert r.returncode == 0 and "physics extra" in r.stderr
