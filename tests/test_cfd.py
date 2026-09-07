"""Case generation and the calibration loop.

The OpenFOAM run itself is not exercised here -- it needs the solver and takes
minutes. What is tested is everything that decides whether the run will be
meaningful: units, mesh density, zone placement, and the correction model.
"""

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from camber.calibrate import (
    GaussianProcess, InterferenceModel, design_features, loss_fraction,
)
from camber.cfd.case import CaseBuilder, CaseConfig, discs_from, rotor_span
from camber.spec import io

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "quad5.toml"


def _stl_extent(path: Path) -> float:
    """Largest vertex coordinate magnitude in a binary STL.

    build123d writes binary by default: an 80-byte header, a uint32 triangle
    count, then 50 bytes per facet (12 float32 plus a 2-byte attribute).
    """
    import struct

    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    biggest = 0.0
    for i in range(count):
        off = 84 + i * 50 + 12          # skip the facet normal
        for v in struct.unpack("<9f", data[off:off + 36]):
            biggest = max(biggest, abs(v))
    return biggest


@pytest.fixture
def vehicle():
    return io.load(EXAMPLE)


@pytest.fixture
def builder(vehicle):
    return CaseBuilder(vehicle, {t.name: 1.128 for t in vehicle.thrusters},
                       CaseConfig())


class TestDomain:
    def test_span_includes_the_rotor_tip(self, vehicle):
        span = rotor_span(vehicle)
        arm = math.hypot(vehicle.thrusters[0].position.x,
                         vehicle.thrusters[0].position.y)
        assert span > arm, "the domain must contain the disc, not just the hub"

    def test_wake_side_of_the_domain_is_longer(self, builder):
        """z is down, so the wake goes to +z and needs the room."""
        lo, hi = builder.bounds()
        assert hi.z > abs(lo.z)

    def test_discs_need_a_thrust_each(self, vehicle):
        with pytest.raises(KeyError, match="no thrust given"):
            discs_from(vehicle, {}, 0.01)


class TestMeshSizing:
    def test_discs_resolve_across_many_cells(self, builder):
        """A momentum source spread over a few cells is a point force.

        Two earlier versions got this wrong in opposite directions: one
        divided the base cell by a further four and left three cells per disc;
        the other made the disc so thick it swallowed the airframe.
        """
        assert builder.cells_per_disc() > 50
        assert builder.cells_across_disc() >= 3, (
            "the disc needs cells through its thickness, not just across its "
            "face, or the source is a single sheet of cells"
        )

    def test_disc_never_intersects_the_airframe(self, vehicle):
        """A disc overlapping the body measures its own source, not a wake."""
        from camber.cfd.case import DiscGeometryError
        import copy
        from camber.spec.units import Vec3
        bad = copy.deepcopy(vehicle)
        for t in bad.thrusters:          # drop the rotors into the airframe
            t.placement = type(t.placement)(
                Vec3(t.position.x, t.position.y, 0.0), t.placement.rotation)
        with pytest.raises(DiscGeometryError, match="intersect"):
            CaseBuilder(bad, {t.name: 1.1 for t in bad.thrusters}, CaseConfig())

    def test_momentum_source_pushes_air_down(self, builder):
        """The source is the *reaction* to thrust: lift up, wake down.

        Getting this backwards blows the wake up through the airframe and
        lifts it, which reads as a large negative download -- impossible, and
        invisible to any check comparing only wake speed. See CFD-06.
        """
        import re
        vec = re.search(r"U   \( \(([^)]*)\)", builder.fv_options()).group(1)
        z = float(vec.split()[2])
        assert z > 0, "z is down, so the wake must have positive z"
        assert builder.discs[0].direction.z < 0, "thrust itself points up"

    def test_background_is_coarser_than_the_base_cell(self, builder):
        assert builder.background_cell > builder.cell
        assert builder.background_cell == pytest.approx(
            builder.cell * 2 ** builder.config.refinement_levels)

    def test_background_stays_affordable(self, builder):
        """Uniformly fine over the whole domain would be ~4M cells."""
        assert builder.background_cells() < 400_000

    def test_near_field_box_contains_the_vehicle_and_some_wake(self, builder):
        lo, hi = builder.near_field()
        assert hi.x > builder.span and hi.z > builder.span


class TestCaseFiles:
    def test_writes_a_runnable_case(self, builder, vehicle, tmp_path):
        stl = tmp_path / "body.stl"
        pytest.importorskip("build123d")
        from camber.geometry import cad
        cad.export(vehicle, stl, metres=True)
        case = builder.build(tmp_path / "case", stl)
        for f in ("system/blockMeshDict", "system/controlDict",
                  "system/fvSchemes", "system/fvSolution",
                  "system/snappyHexMeshDict", "system/topoSetDict",
                  "constant/fvOptions", "constant/transportProperties",
                  "0/U", "0/p", "0/k", "0/omega"):
            assert (case / f).exists(), f"missing {f}"
        assert (case / "constant/triSurface/body.stl").exists()

    def test_stl_is_exported_in_metres_for_cfd(self, vehicle, tmp_path):
        """The bug that made snappy refine nothing: a 180 mm body became 180 m.

        See DECISIONS.md CFD-02.
        """
        pytest.importorskip("build123d")
        from camber.geometry import cad
        mm = cad.export(vehicle, tmp_path / "mm.stl")
        m = cad.export(vehicle, tmp_path / "m.stl", metres=True)
        assert _stl_extent(mm) / _stl_extent(m) == pytest.approx(1000.0, rel=1e-3)

    def test_every_rotor_gets_a_momentum_source(self, builder):
        text = builder.fv_options()
        for t in builder.vehicle.thrusters:
            assert f"{t.name}Source" in text
            assert f"{t.name}Zone" in text
        assert "vectorSemiImplicitSource" in text

    def test_thrust_direction_is_upward_in_hover(self, builder):
        """z is down, so lift is a negative z force."""
        assert builder.discs[0].direction.z < 0

    def test_forces_object_watches_the_body_patch(self, builder):
        text = builder.controls()
        assert "patches         (body)" in text
        assert "type            forces" in text


class TestGaussianProcess:
    def test_interpolates_its_training_data(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, (12, 3))
        y = 0.1 * x[:, 0] + 0.05 * x[:, 1] ** 2
        gp = GaussianProcess(length_scale=0.8, variance=0.05, noise=1e-8).fit(x, y)
        mean, _ = gp.predict(x)
        assert np.abs(mean - y).max() < 1e-4

    def test_uncertainty_is_small_at_data_and_large_away_from_it(self):
        """The property that makes uncertainty sampling worth anything."""
        x = np.linspace(0, 1, 8)[:, None]
        gp = GaussianProcess(length_scale=0.5, variance=0.04, noise=1e-6).fit(x, x.ravel())
        _, near = gp.predict(x)
        _, far = gp.predict(np.array([[50.0]]))
        assert near.mean() < 0.01
        assert float(far[0]) == pytest.approx(math.sqrt(0.04), rel=0.05)

    def test_handles_a_constant_input_column(self):
        """A variable that never varied must not divide by zero."""
        x = np.column_stack([np.linspace(0, 1, 6), np.full(6, 3.0)])
        gp = GaussianProcess().fit(x, np.linspace(0, 1, 6))
        mean, _ = gp.predict(x)
        assert np.isfinite(mean).all()

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="against"):
            GaussianProcess().fit(np.zeros((3, 2)), np.zeros(4))

    def test_predicting_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError, match="fit the model"):
            GaussianProcess().predict(np.zeros((1, 2)))


class TestInterferenceModel:
    def test_features_are_dimensionless_where_they_should_be(self, vehicle):
        f = design_features(vehicle)
        assert f.shape == (4,)
        assert f[0] > 0 and f[1] > 0 and f[3] > 0

    def test_features_respond_to_layout(self, vehicle):
        """Spreading the arms must change the spacing feature."""
        from camber.optimise.problem import arm_length_variable
        before = design_features(vehicle)[1]
        arm_length_variable(vehicle, 0.05, 0.4).apply(vehicle, 0.30)
        assert design_features(vehicle)[1] > before

    def test_uncalibrated_is_an_explicit_no_op(self, vehicle):
        model = InterferenceModel()
        assert model.correction(vehicle) == (0.0, 0.0)
        assert model.installed_thrust(vehicle, 10.0) == 10.0

    def test_a_fitted_model_reduces_installed_thrust(self, vehicle):
        feats = np.array([design_features(vehicle) for _ in range(5)])
        feats += np.random.default_rng(1).normal(0, 0.01, feats.shape)
        model = InterferenceModel().fit(feats, np.full(5, 0.07))
        loss, _ = model.correction(vehicle)
        assert loss == pytest.approx(0.07, abs=0.02)
        assert model.installed_thrust(vehicle, 10.0) == pytest.approx(9.3, abs=0.2)

    def test_correction_is_clamped_outside_the_training_range(self, vehicle):
        """An extrapolating fit must not invent thrust or destroy all of it."""
        feats = np.array([design_features(vehicle) for _ in range(4)])
        feats += np.random.default_rng(2).normal(0, 0.01, feats.shape)
        model = InterferenceModel().fit(feats, np.full(4, 5.0))   # absurd
        assert model.installed_thrust(vehicle, 10.0) >= 4.0

    def test_next_case_prefers_the_least_known_layout(self, vehicle):
        import copy
        from camber.optimise.problem import arm_length_variable
        candidates = []
        for arm in (0.10, 0.12, 0.35):
            v = copy.deepcopy(vehicle)
            arm_length_variable(v, 0.05, 0.5).apply(v, arm)
            candidates.append(v)
        feats = np.array([design_features(c) for c in candidates[:2]])
        model = InterferenceModel().fit(feats, np.array([0.06, 0.065]))
        # The 0.35 m layout is far from both training points.
        assert model.next_case(candidates, n=1) == [2]

    def test_loss_fraction_sign_convention(self):
        """z is down: a positive body z-force is download, so a loss."""
        assert loss_fraction(10.0, 0.5) == pytest.approx(0.05)
        assert loss_fraction(10.0, -0.5) == pytest.approx(-0.05)
        with pytest.raises(ValueError):
            loss_fraction(0.0, 1.0)
