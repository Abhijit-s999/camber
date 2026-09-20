"""Loading specs, and refusing to load broken ones."""

import math
from pathlib import Path

import numpy as np
import pytest

from camber.spec import io
from camber.spec.units import DEG, Vec3
from camber.spec.vehicle import Battery, Camera, Objective, Spin, TiltRotor

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "quad5.toml"

MINIMAL = {
    "name": "t",
    "bodies": [{"name": "b", "type": "body", "mass": 1.0,
                "shape": {"kind": "box", "lx": 0.1, "ly": 0.1, "lz": 0.1}}],
    "thrusters": [{"name": "m0", "type": "rotor", "mass": 0.03,
                   "radius": 0.05, "position": [0.1, 0.1, 0]}],
}


class TestExample:
    def test_loads(self):
        v = io.load(EXAMPLE)
        assert v.name == "quad5"
        assert len(v.thrusters) == 4
        assert v.mission.objective is Objective.MAX_ENDURANCE

    def test_battery_and_camera_are_typed(self):
        v = io.load(EXAMPLE)
        assert isinstance(v.body("battery"), Battery)
        assert isinstance(v.body("camera"), Camera)
        assert v.total_energy_wh == pytest.approx(1.5 * 14.8)

    def test_degrees_are_converted_to_radians(self):
        v = io.load(EXAMPLE)
        cam = v.body("camera")
        assert cam.horizontal_fov == pytest.approx(120.0 * DEG)
        # +25 deg pitch tilts the boresight up, which is -z.
        assert cam.boresight().z < 0

    def test_spins_alternate(self):
        v = io.load(EXAMPLE)
        spins = [t.spin for t in v.thrusters]
        assert spins.count(Spin.CW) == 2 and spins.count(Spin.CCW) == 2

    def test_mass_matches_the_sum_of_its_parts(self):
        v = io.load(EXAMPLE)
        parts = sum(b.mass for b in v.all_bodies()) + sum(t.mass for t in v.thrusters)
        assert v.mass_properties().mass == pytest.approx(parts)

    def test_tilt_rotor_round_trips_with_its_hard_stops(self):
        raw = {**MINIMAL, "thrusters": [{
            **MINIMAL["thrusters"][0], "type": "tilt_rotor",
            "tilt_axis": [0, 1, 0], "tilt_min_deg": -5,
            "tilt_max_deg": 90, "tilt_tau": 0.08,
        }]}
        t = io.from_dict(raw).thrusters[0]
        assert isinstance(t, TiltRotor)
        assert t.direction_at(math.pi / 2).as_list() == pytest.approx([-1, 0, 0])
        again = io.from_dict(io.to_dict(io.from_dict(raw))).thrusters[0]
        assert (again.tilt_min, again.tilt_max, again.tilt_tau) == pytest.approx(
            (t.tilt_min, t.tilt_max, t.tilt_tau))


class TestRejection:
    def test_typo_is_an_error_not_a_default(self):
        """`mas` must stop the load, not silently produce a default mass."""
        bad = {**MINIMAL, "bodies": [{"name": "b", "type": "body", "mas": 1.0}]}
        with pytest.raises(io.SpecError, match="unknown key"):
            io.from_dict(bad)

    def test_missing_required_key(self):
        bad = {**MINIMAL, "bodies": [{"name": "b", "type": "body"}]}
        with pytest.raises(io.SpecError, match="missing required key 'mass'"):
            io.from_dict(bad)

    def test_unknown_body_type(self):
        bad = {**MINIMAL, "bodies": [{"name": "b", "type": "wing", "mass": 1.0}]}
        with pytest.raises(io.SpecError, match="unknown body type"):
            io.from_dict(bad)

    def test_unimplemented_thruster_says_so_clearly(self):
        bad = {**MINIMAL, "thrusters": [{"name": "r", "type": "rocket", "mass": 0.1}]}
        with pytest.raises(io.SpecError, match="not implemented yet"):
            io.from_dict(bad)

    def test_constraint_referring_to_a_missing_body(self):
        bad = {**MINIMAL, "constraints": [
            {"type": "min_distance", "a": "b", "b": "ghost", "distance": 0.1}]}
        with pytest.raises(io.SpecError, match="'ghost'"):
            io.from_dict(bad)

    def test_constraint_may_reference_a_thruster(self):
        """A camera is most often obstructed by a propeller."""
        ok = {**MINIMAL, "constraints": [
            {"type": "min_distance", "a": "b", "b": "m0", "distance": 0.01}]}
        assert io.from_dict(ok).constraints

    def test_duplicate_names(self):
        bad = {**MINIMAL, "bodies": [
            {"name": "b", "type": "body", "mass": 1.0},
            {"name": "b", "type": "body", "mass": 2.0}]}
        with pytest.raises(io.SpecError, match="duplicate name"):
            io.from_dict(bad)

    def test_unknown_objective_lists_the_valid_ones(self):
        bad = {**MINIMAL, "mission": {"objective": "go_fast_ish"}}
        with pytest.raises(io.SpecError, match="max_endurance"):
            io.from_dict(bad)

    def test_bad_spin(self):
        bad = {**MINIMAL, "thrusters": [
            {**MINIMAL["thrusters"][0], "spin": "clockwise"}]}
        with pytest.raises(io.SpecError, match="expected 'cw' or 'ccw'"):
            io.from_dict(bad)

    def test_degenerate_shape_is_reported_with_its_location(self):
        bad = {**MINIMAL, "bodies": [{"name": "b", "type": "body", "mass": 1.0,
               "shape": {"kind": "box", "lx": 0.0, "ly": 1.0, "lz": 1.0}}]}
        with pytest.raises(io.SpecError, match=r"bodies\[0\]"):
            io.from_dict(bad)

    def test_negative_mass(self):
        bad = {**MINIMAL, "bodies": [{"name": "b", "type": "body", "mass": -1.0}]}
        with pytest.raises(io.SpecError, match="non-positive mass"):
            io.from_dict(bad)


class TestRoundTrip:
    def test_dict_roundtrip_preserves_mass_properties(self, tmp_path):
        original = io.load(EXAMPLE)
        restored = io.from_dict(io.to_dict(original))
        a, b = original.mass_properties(), restored.mass_properties()
        assert b.mass == pytest.approx(a.mass)
        assert b.cg.as_list() == pytest.approx(a.cg.as_list())
        assert np.allclose(a.inertia, b.inertia)

    def test_json_roundtrip(self, tmp_path):
        original = io.load(EXAMPLE)
        out = tmp_path / "v.json"
        io.save_json(original, out)
        restored = io.load(out)
        assert restored.name == original.name
        assert restored.mass_properties().mass == pytest.approx(
            original.mass_properties().mass)

    def test_rotation_survives_the_roundtrip(self):
        original = io.load(EXAMPLE)
        restored = io.from_dict(io.to_dict(original))
        assert restored.body("camera").boresight().as_list() == pytest.approx(
            original.body("camera").boresight().as_list())


class TestNothingIsLostOnSave:
    """to_dict silently omitted constraints: every JSON save deleted the CG
    box, the camera view cone and the minimum distances, and a vehicle with no
    constraints is perfectly valid, so nothing complained. The fix is a round
    trip of the whole vehicle, not of the one field that was caught."""

    @pytest.mark.parametrize("spec", ["racer5.toml", "quad5.toml"])
    def test_the_whole_vehicle_round_trips(self, spec):
        import pathlib
        v = io.load(pathlib.Path(__file__).resolve().parent.parent / "examples" / spec)
        d = io.to_dict(v)
        # Saving what was loaded from a save changes nothing, field for field.
        assert io.to_dict(io.from_dict(d)) == d
        # And the TOML's top-level sections all made it into the document.
        import tomllib
        raw = tomllib.loads((pathlib.Path(__file__).resolve().parent.parent
                             / "examples" / spec).read_text())
        assert set(raw) <= set(d), f"lost on save: {set(raw) - set(d)}"
        back = io.from_dict(d)
        assert np.allclose(back.mass_properties().inertia, v.mass_properties().inertia)

    @pytest.mark.parametrize("spec", ["racer5.toml", "quad5.toml"])
    def test_constraints_survive_a_json_file(self, spec, tmp_path):
        import pathlib
        v = io.load(pathlib.Path(__file__).resolve().parent.parent / "examples" / spec)
        assert v.constraints, "the example should declare some"
        io.save_json(v, tmp_path / "v.json")
        assert io.load(tmp_path / "v.json").constraints == v.constraints
