"""The generated frame, fed back into the vehicle it was generated for.

Until this existed `camber generate` wrote an STL and the vehicle model went on
believing in its placeholder -- on the racing quad an 85 g box -- so hover,
thrust-to-weight, inertia and interference all described an airframe nobody was
going to build.
"""

import copy
import pathlib

import numpy as np
import pytest

from camber.spec import io
from camber.spec.shapes import Box, Cylinder, Sphere, Tabulated

ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestTabulatedShape:
    def test_it_reports_what_it_was_given(self):
        t = Tabulated(2e-5, np.diag([1.0, 2.0, 3.0]), (0.1, 0.1, 0.006), 0.012)
        assert t.volume == 2e-5
        assert t.planform_area() == 0.012
        assert np.allclose(t.inertia(2.0), np.diag([2.0, 4.0, 6.0]))

    def test_an_asymmetric_tensor_is_refused(self):
        with pytest.raises(ValueError, match="symmetric"):
            Tabulated(1e-5, [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]], (0.1,) * 3, 0.01)

    def test_a_tensor_with_negative_inertia_is_refused(self):
        with pytest.raises(ValueError, match="semi-definite"):
            Tabulated(1e-5, np.diag([1.0, -1.0, 1.0]), (0.1,) * 3, 0.01)

    def test_zero_volume_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            Tabulated(0.0, np.eye(3), (0.1,) * 3, 0.01)

    def test_it_does_not_shadow_the_inertia_method(self):
        """A field called `inertia` would hide Shape.inertia(mass), which the
        mass-properties code calls; and dataclasses read the inherited method
        as the field's default, which is how the collision announced itself."""
        t = Tabulated(1e-5, np.eye(3), (0.1,) * 3, 0.01)
        assert callable(t.inertia)


class TestTheCalibrationIsUntouched:
    """Blockage is the interference model's first feature and it was trained
    on bounding rectangles. Every existing shape must report exactly what it
    always did, or the 25 CFD cases describe a different quantity."""

    @pytest.mark.parametrize("shape", [Box(0.15, 0.07, 0.024),
                                       Cylinder(0.004, 0.09), Sphere(0.01)])
    def test_the_default_footprint_is_the_old_formula(self, shape):
        h = shape.half_extents()
        assert shape.planform_area() == 4.0 * h.x * h.y

    @pytest.mark.parametrize("spec", ["racer5.toml", "quad5.toml"])
    def test_the_examples_features_are_unchanged(self, spec):
        from camber.calibrate import design_features
        v = io.load(ROOT / "examples" / spec)
        old = sum(b.shape.half_extents().x * b.shape.half_extents().y * 4.0
                  for b in v.all_bodies())
        disc = sum(np.pi * r.radius ** 2 for r in v.thrusters)
        assert design_features(v)[0] == pytest.approx(old / disc, rel=1e-12)


class TestSerialisation:
    def test_a_tabulated_body_round_trips(self, tmp_path, monkeypatch):
        from camber.spec.units import Vec3
        from camber.spec.vehicle import Body, Placement
        v = io.load(ROOT / "examples" / "racer5.toml")
        # In memory a relative source is relative to the working directory,
        # which is where every reader opens it (SPEC-03), so say which one.
        monkeypatch.chdir(tmp_path)
        v.frame.members = [Body(
            "frame", mass=0.05, placement=Placement(Vec3(0.0, 0.0, -0.001)),
            shape=Tabulated(3.2e-5, np.diag([4e-3, 4e-3, 8e-3]),
                            (0.1, 0.1, 0.006), 0.0119, "frame.stl"))]
        io.save_json(v, tmp_path / "v.json")
        back = io.load(tmp_path / "v.json")
        import dataclasses
        a, b = v.frame.members[0].shape, back.frame.members[0].shape
        # The source comes back anchored to the spec's folder, by design.
        assert dataclasses.replace(b, source="") == dataclasses.replace(a, source="")
        assert pathlib.Path(b.source) == (tmp_path / "frame.stl").resolve()
        assert np.allclose(back.mass_properties().inertia,
                           v.mass_properties().inertia)

    def test_an_unknown_shape_is_an_error_not_a_point_mass(self):
        """Saving used to fall through to {"kind": "point"} for anything it
        did not recognise: the mass survived, the geometry and inertia were
        silently deleted."""
        from dataclasses import dataclass

        from camber.spec.shapes import Shape
        from camber.spec.units import Vec3
        from camber.spec.vehicle import Body

        @dataclass(frozen=True)
        class Blob(Shape):
            @property
            def volume(self):
                return 1e-6

            def inertia_unit_mass(self):
                return np.eye(3) * 1e-6

            def half_extents(self):
                return Vec3(0.01, 0.01, 0.01)

        v = io.load(ROOT / "examples" / "racer5.toml")
        v.frame.members = [Body("blob", mass=0.01, shape=Blob())]
        with pytest.raises(TypeError, match="cannot serialise"):
            io.to_dict(v)


class TestFrameBody:
    """Mass properties measured from a mesh must agree with the closed form."""

    @pytest.fixture
    def slab(self):
        pytest.importorskip("trimesh")
        import trimesh
        m = trimesh.creation.box((0.12, 0.06, 0.01))
        m.apply_translation([0.003, -0.002, 0.001])
        return m

    def test_mass_is_volume_times_density(self, slab):
        from camber.generate import topology as T
        b = T.frame_body(slab, planform=0.0072, density=1550.0)
        assert b.mass == pytest.approx(0.12 * 0.06 * 0.01 * 1550.0)

    def test_inertia_matches_the_box_formula(self, slab):
        from camber.generate import topology as T
        b = T.frame_body(slab, planform=0.0072, density=1550.0)
        want = Box(0.12, 0.06, 0.01).inertia_unit_mass()
        assert np.allclose(b.shape.inertia_unit_mass(), want, rtol=1e-6)

    def test_it_sits_where_the_mesh_does(self, slab):
        from camber.generate import topology as T
        b = T.frame_body(slab, planform=0.0072, density=1550.0)
        assert [b.position.x, b.position.y, b.position.z] == pytest.approx(
            [0.003, -0.002, 0.001])

    def test_an_open_mesh_is_refused(self, slab):
        from camber.generate import topology as T
        slab.update_faces(np.arange(len(slab.faces) - 2))
        with pytest.raises(ValueError, match="open mesh"):
            T.frame_body(slab, planform=0.0072, density=1550.0)

    def test_the_footprint_is_the_shadow_not_the_box(self):
        """An X of two diagonal bars covers far less than its bounding box."""
        from camber.generate import topology as T
        d = T.Domain(20, 20, 3, 0.01)
        rho = np.zeros(d.shape)
        i = np.arange(20)
        rho[i, i, :] = 1.0
        rho[i, 19 - i, :] = 1.0
        area = T.footprint(T.Result(density=rho), d)
        assert area == pytest.approx(40 * 0.01 ** 2)
        assert area < 0.2 * (20 * 0.01) ** 2


class TestEnvelope:
    @pytest.fixture
    def model(self):
        from camber.calibrate import load_model
        path = ROOT / "data" / "interference.json"
        if not path.exists():
            pytest.skip("no fitted interference model")
        return load_model(path)

    def test_the_racing_quad_is_inside(self, model):
        assert model.outside(io.load(ROOT / "examples" / "racer5.toml")) == []

    def test_a_layout_far_outside_is_named(self, model):
        """Blockage of a box the size of the whole frame: the case the old
        bounding-box definition would have produced for a generated X."""
        v = io.load(ROOT / "examples" / "racer5.toml")
        v.frame.members[0].shape = Box(0.40, 0.40, 0.02)
        assert "blockage" in model.outside(v)


class TestTheLoopEndToEnd:
    def test_generate_into_replaces_the_placeholder(self, tmp_path):
        pytest.importorskip("skimage")
        pytest.importorskip("trimesh")
        from camber.cli import main
        out = tmp_path / "v.json"
        assert main(["generate", "examples/racer5.toml", "--mass", "40",
                     "--element", "0.008", "--thickness", "0.024",
                     "--iterations", "4", "--min-feature", "16",
                     "--out", str(tmp_path / "f.stl"), "--into", str(out)]) == 0
        before = io.load(ROOT / "examples" / "racer5.toml")
        after = io.load(out)
        assert [b.name for b in after.frame.members] == ["frame"]
        assert isinstance(after.frame.members[0].shape, Tabulated)
        chassis = sum(b.mass for b in before.frame.members)
        frame = after.frame.members[0].mass
        assert after.mass_properties().mass == pytest.approx(
            before.mass_properties().mass - chassis + frame)


class TestPortableSources:
    """A framed spec and its mesh must be able to move together."""

    def test_a_relative_source_resolves_against_the_spec_not_the_cwd(
            self, tmp_path, monkeypatch):
        pytest.importorskip("trimesh")
        import trimesh
        from camber.spec.units import Vec3
        from camber.spec.vehicle import Body, Placement
        d = tmp_path / "vehicles"
        d.mkdir()
        trimesh.creation.box((0.1, 0.1, 0.01)).export(d / "frame.stl")
        v = io.load(ROOT / "examples" / "racer5.toml")
        v.frame.members = [Body("frame", mass=0.05, placement=Placement(Vec3(0, 0, 0)),
                                shape=Tabulated(1e-4, np.eye(3) * 1e-3, (0.05, 0.05, 0.005),
                                                0.01, str(d / "frame.stl")))]
        io.save_json(v, d / "v.json")
        written = __import__("json").loads((d / "v.json").read_text())
        assert written["frame"][0]["shape"]["source"] == "frame.stl"
        monkeypatch.chdir(tmp_path)                       # run from elsewhere
        back = io.load(d / "v.json")
        assert pathlib.Path(back.frame.members[0].shape.source).exists()

    def test_a_frame_grown_inside_a_project_can_be_loaded_back(self, tmp_path,
                                                               monkeypatch):
        """SPEC-03: the app runs `generate` from the project folder with
        relative paths. The frame's source was saved as written, relative to
        the project, then anchored to the JSON's own folder on load: the path
        doubled, "Use this frame" showed no frame, and Aerodynamics failed."""
        pytest.importorskip("skimage")
        pytest.importorskip("trimesh")
        from camber.cli import main
        job = pathlib.Path("results") / "generate-job"
        (tmp_path / job).mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        assert main(["generate", str(ROOT / "examples" / "racer5.toml"), "--mass", "40",
                     "--element", "0.008", "--thickness", "0.024",
                     "--iterations", "4", "--min-feature", "16",
                     "--out", str(job / "frame.stl"), "--into", str(job / "framed.json")]) == 0
        written = __import__("json").loads((job / "framed.json").read_text())
        assert written["frame"][0]["shape"]["source"] == "frame.stl"
        monkeypatch.chdir(ROOT)                           # and from anywhere else
        back = io.load(tmp_path / job / "framed.json")
        assert pathlib.Path(back.frame.members[0].shape.source).exists()
