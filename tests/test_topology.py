"""Generative structure: the element, the optimiser, and the mesh it produces.

The load-bearing assertion in this file is `test_six_rigid_body_modes`. The
element stiffness matrix is the one thing here that is wrong *silently* -- a
mistranscribed 24x24 is still square, still symmetric, and still produces a
plausible-looking optimisation that is solving a different physics problem.
Two attempts to copy it from published tables failed exactly that way, so it
is now integrated by quadrature and checked against the property that defines
it: an unconstrained brick can translate three ways and rotate three ways at
zero energy, and nothing else.
"""

import numpy as np
import pytest

from camber.generate import topology as T


class TestElement:
    def test_is_symmetric(self):
        KE = T.hex8_stiffness()
        assert np.allclose(KE, KE.T)

    def test_six_rigid_body_modes(self):
        """Three translations and three rotations, at zero strain energy."""
        w = np.linalg.eigvalsh(T.hex8_stiffness())
        scale = w.max()
        zeros = int((np.abs(w) < 1e-9 * scale).sum())
        assert zeros == 6, f"expected 6 zero eigenvalues, got {zeros}"
        assert w.min() > -1e-9 * scale, "not positive semi-definite"

    def test_translation_costs_nothing(self):
        """The rigid-body property, stated directly rather than by eigenvalue."""
        KE = T.hex8_stiffness()
        for axis in range(3):
            u = np.zeros(24)
            u[axis::3] = 1.0
            assert u @ KE @ u == pytest.approx(0.0, abs=1e-9 * np.abs(KE).max())

    def test_stretching_costs_something(self):
        KE = T.hex8_stiffness()
        u = np.zeros(24)
        u[0::3] = [-1, 1, 1, -1, -1, 1, 1, -1]      # uniaxial stretch in x
        assert u @ KE @ u > 0

    def test_scales_linearly_with_modulus(self):
        assert np.allclose(T.hex8_stiffness(E=7.0), 7.0 * T.hex8_stiffness(E=1.0))

    def test_scales_with_element_size(self):
        """Stiffness of a cube goes as its edge: area over length."""
        assert np.allclose(T.hex8_stiffness(size=2.0),
                           2.0 * T.hex8_stiffness(size=1.0))


class TestDomain:
    def test_counts(self):
        d = T.Domain(4, 3, 2, 0.01)
        assert d.n_elements == 24
        assert d.n_nodes == 5 * 4 * 3

    def test_every_element_has_eight_distinct_nodes(self):
        d = T.Domain(3, 3, 2, 0.01)
        dofs = d.element_dofs()
        assert dofs.shape == (d.n_elements, 24)
        assert dofs.min() >= 0 and dofs.max() < 3 * d.n_nodes
        for row in dofs:
            assert len(set(row[0::3])) == 8

    def test_dof_triples_are_consecutive(self):
        """Each node contributes x, y, z adjacently; the assembly assumes it."""
        dofs = T.Domain(2, 2, 2, 0.01).element_dofs()
        for row in dofs:
            for c in range(8):
                x, y, z = row[3*c:3*c+3]
                assert (y, z) == (x + 1, x + 2)

    def test_neighbouring_elements_share_a_face(self):
        d = T.Domain(2, 1, 1, 0.01)
        a, b = d.element_dofs()[0][0::3], d.element_dofs()[1][0::3]
        assert len(set(a) & set(b)) == 4


class TestFilter:
    def test_is_symmetric_and_positive(self):
        H = T._filter_matrix(T.Domain(4, 4, 2, 0.01), 1.5)
        assert (H != H.T).nnz == 0
        assert H.min() >= 0
        assert np.all(np.asarray(H.sum(axis=1)).ravel() > 0)

    def test_a_larger_radius_reaches_further(self):
        d = T.Domain(6, 6, 2, 0.01)
        assert T._filter_matrix(d, 3.0).nnz > T._filter_matrix(d, 1.5).nnz

    def test_smooths_a_checkerboard(self):
        """The whole point: alternating densities must not survive filtering."""
        d = T.Domain(8, 8, 2, 0.01)
        H = T._filter_matrix(d, 2.0)
        Hs = np.asarray(H.sum(axis=1)).ravel()
        i, j, k = np.indices(d.shape)
        board = ((i + j + k) % 2).astype(float).ravel()
        smoothed = H @ (board / Hs)
        assert smoothed.std() < board.std() / 3


class TestPenaltyContinuation:
    def test_starts_low_and_reaches_the_final_value(self):
        p = [T.penalty_at(i, 40, 3.0, 1.0) for i in range(40)]
        assert p[0] == pytest.approx(1.0)
        assert p[-1] == pytest.approx(3.0)

    def test_never_decreases(self):
        p = [T.penalty_at(i, 40, 3.0, 1.0) for i in range(40)]
        assert all(b >= a for a, b in zip(p, p[1:]))

    def test_moves_in_quarter_steps(self):
        for i in range(40):
            p = T.penalty_at(i, 40, 3.0, 1.0)
            assert (p * 4) == pytest.approx(round(p * 4))

    def test_equal_bounds_disable_it(self):
        assert all(T.penalty_at(i, 40, 3.0, 3.0) == 3.0 for i in range(40))

    def test_the_ramp_finishes_before_the_run_does(self):
        """Convergence is only meaningful at the final penalty, so the run
        must spend real iterations there."""
        p = [T.penalty_at(i, 40, 3.0, 1.0) for i in range(40)]
        assert sum(1 for v in p if v >= 3.0) >= 40 // 4


def cantilever(nx=10, ny=4, nz=2, vf=0.4):
    """Fixed along x=0, pulled down at the far end. The textbook case."""
    d = T.Domain(nx, ny, nz, 0.01)
    fixed, load = [], np.zeros(3 * d.n_nodes)
    for j in range(ny + 1):
        for k in range(nz + 1):
            n = d.node_id(0, j, k)
            fixed += [3*n, 3*n+1, 3*n+2]
    for j in range(ny + 1):
        n = d.node_id(nx, j, nz)
        load[3*n + 2] = -1.0
    return T.Problem(domain=d, fixed=np.array(fixed), load=load,
                     volume_fraction=vf)


class TestOptimise:
    @pytest.fixture(scope="class")
    def run(self):
        return T.optimise(cantilever(), iterations=25, filter_radius=1.5,
                          penalty_start=3.0)

    def test_it_gets_stiffer(self, run):
        c = run.compliance_history
        assert c[-1] < c[0], "compliance must fall; the objective is stiffness"

    def test_it_holds_the_volume_constraint(self, run):
        assert run.volume_fraction == pytest.approx(0.4, abs=0.01)

    def test_densities_stay_in_bounds(self, run):
        assert run.density.min() >= 0.0 and run.density.max() <= 1.0

    def test_it_puts_material_where_the_moment_is(self, run):
        """A cantilever's bending moment peaks at the root, so that is where
        the material has to go. If it did not, the FEA is not in the loop."""
        rho = run.density
        root = rho[:3].mean()
        tip = rho[-3:].mean()
        assert root > tip, f"root {root:.3f} is not denser than tip {tip:.3f}"

    def test_the_penalty_is_what_separates_solid_from_void(self, run):
        """Stated as a comparison, because the absolute grey fraction has no
        natural threshold -- a first version of this test demanded it stay
        under 35% and measured exactly 35%. What the penalty actually claims
        is *relative*: p = 3 prices intermediate density out of the solution
        and p = 1 does not, so the same problem run both ways must come out
        more binary at 3."""
        def grey(p):
            r = T.optimise(cantilever(), iterations=25, filter_radius=1.5,
                           penalty=p, penalty_start=p)
            return float(((r.density > 0.2) & (r.density < 0.8)).mean())

        assert grey(3.0) < grey(1.0)
        assert ((run.density > 0.2) & (run.density < 0.8)).mean() < 0.5

    def test_keep_regions_stay_solid(self):
        p = cantilever()
        p.keep = np.zeros(p.domain.shape, dtype=bool)
        p.keep[5, :, :] = True
        r = T.optimise(p, iterations=6, penalty_start=3.0)
        assert np.all(r.density[5] == 1.0)

    def test_void_regions_stay_empty(self):
        p = cantilever()
        p.void = np.zeros(p.domain.shape, dtype=bool)
        p.void[5, :, :] = True
        r = T.optimise(p, iterations=6, penalty_start=3.0)
        assert np.all(r.density[5] == 0.0)

    def test_continuation_reaches_the_final_penalty(self):
        r = T.optimise(cantilever(), iterations=12, penalty=3.0,
                       penalty_start=1.0)
        assert r.penalty == pytest.approx(3.0)

    def test_it_reports_how_far_it_got(self, run):
        assert 0 < run.iterations <= 25
        assert len(run.compliance_history) == run.iterations


class TestConnectivity:
    def test_a_single_blob_is_fully_connected(self):
        rho = np.zeros((6, 6, 2))
        rho[1:4, 1:4, :] = 1.0
        r = T.Result(density=rho)
        assert r.connectivity() == (1, 1.0)

    def test_islands_are_counted_and_weighed(self):
        rho = np.zeros((8, 8, 2))
        rho[0:4, 0:4, :] = 1.0        # 32 elements
        rho[7, 7, :] = 1.0            # 2 elements, touching nothing
        n, frac = T.Result(density=rho).connectivity()
        assert n == 2
        assert frac == pytest.approx(32 / 34)

    def test_empty_field_does_not_divide_by_zero(self):
        assert T.Result(density=np.zeros((4, 4, 2))).connectivity() == (0, 0.0)


class TestPruning:
    def test_islands_are_removed(self):
        rho = np.zeros((8, 8, 2))
        rho[0:4, 0:4, :] = 1.0
        rho[7, 7, :] = 1.0
        cleaned = T.largest_connected(rho)
        assert cleaned[7, 7, 0] == 0.0
        assert cleaned[0:4, 0:4].sum() == rho[0:4, 0:4].sum()

    def test_a_connected_field_is_untouched(self):
        rho = np.zeros((6, 6, 2))
        rho[1:5, 1:5, :] = 1.0
        assert np.array_equal(T.largest_connected(rho), rho)

    def test_it_keeps_the_largest_not_the_first(self):
        rho = np.zeros((10, 4, 2))
        rho[0, 0, :] = 1.0            # first in scan order, but tiny
        rho[4:9, 1:3, :] = 1.0        # the real structure
        cleaned = T.largest_connected(rho)
        assert cleaned[0, 0, 0] == 0.0
        assert cleaned[4:9, 1:3].sum() > 0


class TestMassBudget:
    def test_the_fraction_recovers_the_requested_mass(self):
        d = T.Domain(40, 40, 4, 0.003)
        vf = T.volume_fraction_for_mass(d, 0.040, 1550.0)
        box = d.n_elements * d.element ** 3
        assert vf * box * 1550.0 == pytest.approx(0.040)

    def test_it_refuses_to_exceed_the_box(self):
        d = T.Domain(4, 4, 2, 0.001)
        assert T.volume_fraction_for_mass(d, 10.0, 1550.0) <= 0.9

    def test_structural_mass_matches_the_density_field(self):
        d = T.Domain(10, 10, 2, 0.004)
        r = T.Result(density=np.full(d.shape, 0.25))
        assert T.structural_mass(r, d, 1550.0) == pytest.approx(
            0.25 * d.n_elements * d.element ** 3 * 1550.0)


class TestFrameProblem:
    @pytest.fixture                      # per-test: one case mutates the vehicle
    def vehicle(self):
        import pathlib

        from camber.spec import io
        return io.load(pathlib.Path(__file__).resolve().parent.parent
                       / "examples" / "quad5.toml")

    def test_no_rotors_is_an_error(self, vehicle):
        vehicle.thrusters.clear()
        with pytest.raises(ValueError, match="no load path"):
            T.frame_problem(vehicle)

    def test_the_domain_encloses_the_rotors(self, vehicle):
        import math

        p = T.frame_problem(vehicle, element=0.006)
        reach = max(math.hypot(r.position.x, r.position.y)
                    for r in vehicle.thrusters)
        assert p.domain.nx * p.domain.element / 2 >= reach

    def test_load_is_applied_and_structure_is_held(self, vehicle):
        p = T.frame_problem(vehicle, element=0.006)
        assert np.count_nonzero(p.load) > 0
        assert len(p.fixed) > 0

    def test_the_restraint_does_not_short_circuit_the_load(self, vehicle):
        """The 3-2-1 restraint sits on a loaded stack pad, which would be a
        bug under a grounded formulation -- force straight into ground, and no
        reason to build anything between load and support. It is fine here
        only because the load set is self-equilibrated, so the reactions are
        zero; `TestTheFrameIsAFreeBody` measures that rather than assuming it."""
        p = T.frame_problem(vehicle, element=0.006)
        assert len(p.fixed) == 6
        assert np.allclose(p.load.reshape(-1, 3).sum(axis=0), 0.0, atol=1e-9)

    def test_rotor_mounts_are_forced_solid(self, vehicle):
        p = T.frame_problem(vehicle, element=0.006)
        assert p.keep is not None and p.keep.any()

    def test_a_mass_budget_sets_the_volume_fraction(self, vehicle):
        light = T.frame_problem(vehicle, element=0.006, target_mass=0.030)
        heavy = T.frame_problem(vehicle, element=0.006, target_mass=0.090)
        assert light.volume_fraction < heavy.volume_fraction


class TestMesh:
    @pytest.fixture(scope="class")
    def blob(self):
        d = T.Domain(12, 12, 6, 0.004)
        rho = np.zeros(d.shape)
        rho[3:9, 3:9, 1:5] = 1.0
        return d, T.Result(density=rho)

    def test_the_surface_is_watertight(self, blob):
        pytest.importorskip("skimage")
        pytest.importorskip("trimesh")
        d, r = blob
        assert T.to_mesh(r, d).is_watertight

    def test_the_volume_is_positive(self, blob):
        """A negative volume means inverted winding. The mesh still looks
        right in a viewer and a slicer reads it as a hole."""
        pytest.importorskip("skimage")
        pytest.importorskip("trimesh")
        d, r = blob
        assert T.to_mesh(r, d).volume > 0

    def test_the_volume_is_near_the_density_field(self, blob):
        pytest.importorskip("skimage")
        pytest.importorskip("trimesh")
        d, r = blob
        expected = r.density.sum() * d.element ** 3
        assert T.to_mesh(r, d).volume == pytest.approx(expected, rel=0.35)

    def test_it_is_centred_on_the_vehicle_origin(self, blob):
        pytest.importorskip("skimage")
        pytest.importorskip("trimesh")
        d, r = blob
        c = T.to_mesh(r, d).bounds.mean(axis=0)
        assert np.allclose(c, 0.0, atol=2 * d.element)

    def test_pruning_can_be_turned_off(self, blob):
        pytest.importorskip("skimage")
        pytest.importorskip("trimesh")
        d, _ = blob
        rho = np.zeros(d.shape)
        rho[3:9, 3:9, 1:5] = 1.0
        rho[0, 0, 0] = 1.0
        r = T.Result(density=rho)
        assert (T.to_mesh(r, d, prune=False).volume
                > T.to_mesh(r, d, prune=True).volume)


class TestConvergence:
    """The stopping rule, which a control run caught getting this wrong."""

    def test_a_flat_history_has_settled(self):
        assert T._has_settled([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_a_falling_history_has_not(self):
        assert not T._has_settled([9.0, 7.0, 5.0, 3.0, 1.0])

    def test_oscillation_is_not_convergence(self):
        """Consecutive iterations can differ little while the run flips
        between two layouts; the window is what sees it."""
        assert not T._has_settled([1.0, 40.0, 1.0, 40.0, 1.0])

    def test_too_short_a_history_is_never_settled(self):
        assert not T._has_settled([1.0, 1.0])

    def test_it_will_not_call_a_worsening_run_converged(self):
        """Verbatim from the run that exposed this: compliance rose from
        9.958e8 to 3.922e10 in four iterations and the old rule -- density
        change alone -- reported success."""
        history = [9.958e8, 1.3067e11, 2.6754e10, 3.9221e10]
        assert not T._has_settled(history)

    def test_density_change_alone_does_not_stop_the_run(self):
        """A tiny density step early on must not end it."""
        p = cantilever(nx=8, ny=3, nz=2)
        r = T.optimise(p, iterations=6, penalty_start=3.0, tol=1.0)
        assert r.iterations == 6, "tol=1 makes every step 'small'; only a " \
                                  "flat compliance history should stop this"


class TestAssembly:
    """The fast assembly must be the *same* matrix, not merely a fast one."""

    def _reference(self, problem, KE, stiffness):
        """The obvious way: full COO, convert, then slice out fixed DOFs."""
        import scipy.sparse as sp

        d = problem.domain
        dofs = d.element_dofs()
        n_dof = 3 * d.n_nodes
        iK = np.kron(dofs, np.ones((24, 1), dtype=np.int64)).ravel()
        jK = np.kron(dofs, np.ones((1, 24), dtype=np.int64)).ravel()
        sK = (KE.ravel()[None, :] * stiffness[:, None]).ravel()
        K = sp.coo_matrix((sK, (iK, jK)), shape=(n_dof, n_dof)).tocsc()
        free = np.setdiff1d(np.arange(n_dof), problem.fixed)
        return K[free, :][:, free]

    @pytest.fixture
    def case(self):
        p = cantilever(nx=6, ny=3, nz=2, vf=0.5)
        KE = T.hex8_stiffness()
        rng = np.random.default_rng(0)
        stiffness = rng.uniform(0.1, 1.0, p.domain.n_elements)
        return p, KE, stiffness

    def test_it_matches_the_obvious_assembly(self, case):
        p, KE, stiffness = case
        fast = T._Assembly(p.domain.element_dofs(), p.fixed,
                           3 * p.domain.n_nodes, KE).matrix(stiffness)
        ref = self._reference(p, KE, stiffness)
        assert fast.shape == ref.shape
        assert abs(fast - ref).max() < 1e-9 * abs(ref).max()

    def test_the_free_dof_map_is_the_complement_of_the_fixed_set(self, case):
        p, KE, _ = case
        n_dof = 3 * p.domain.n_nodes
        asm = T._Assembly(p.domain.element_dofs(), p.fixed, n_dof, KE)
        assert not set(asm.free.tolist()) & set(p.fixed.tolist())
        assert asm.free.size + np.unique(p.fixed).size == n_dof

    def test_the_pattern_is_reused_across_stiffnesses(self, case):
        """The whole premise: the mesh does not move, so only values change."""
        p, KE, stiffness = case
        asm = T._Assembly(p.domain.element_dofs(), p.fixed,
                          3 * p.domain.n_nodes, KE)
        a = asm.matrix(stiffness)
        b = asm.matrix(stiffness * 3.0)
        assert np.array_equal(a.indices, b.indices)
        assert np.array_equal(a.indptr, b.indptr)
        assert np.allclose(b.data, 3.0 * a.data)

    def test_the_reduced_matrix_is_symmetric_and_definite(self, case):
        """splu is told SymmetricMode; that has to be true."""
        p, KE, stiffness = case
        K = T._Assembly(p.domain.element_dofs(), p.fixed,
                        3 * p.domain.n_nodes, KE).matrix(stiffness)
        assert abs(K - K.T).max() < 1e-9 * abs(K).max()
        w = np.linalg.eigvalsh(K.toarray())
        assert w.min() > 0, "supports do not remove every rigid-body mode"


class TestVoxelise:
    def _body(self, shape, position=(0.0, 0.0, 0.0), rotation=None):
        from camber.spec.units import Vec3
        from camber.spec.vehicle import Body, Placement
        return Body("b", mass=0.1, shape=shape,
                    placement=Placement(Vec3(*position), rotation)
                    if rotation else Placement(Vec3(*position)))

    def test_a_box_recovers_its_own_volume(self):
        from camber.spec.shapes import Box
        d = T.Domain(20, 20, 6, 0.005)
        v = T.voxelise([self._body(Box(0.05, 0.05, 0.01))], d, samples=3)
        assert v.sum() * d.element ** 3 == pytest.approx(0.05 * 0.05 * 0.01,
                                                         rel=0.02)

    def test_a_cylinder_recovers_its_own_volume(self):
        import math

        from camber.spec.shapes import Cylinder
        d = T.Domain(20, 20, 8, 0.005)
        v = T.voxelise([self._body(Cylinder(0.008, 0.03))], d, samples=4)
        want = math.pi * 0.008 ** 2 * 0.03
        assert v.sum() * d.element ** 3 == pytest.approx(want, rel=0.06)

    def test_rotation_is_applied_the_right_way_round(self):
        """A rod aligned to +x must be long in x and narrow across it. Getting
        the inverse backwards still produces a rod, just the wrong one."""
        from camber.spec.rotation import Rotation
        from camber.spec.shapes import Cylinder
        from camber.spec.units import Vec3
        d = T.Domain(20, 20, 6, 0.005)
        rod = self._body(Cylinder(0.008, 0.06),
                         rotation=Rotation.aligning(Vec3(0, 0, 1), Vec3(1, 0, 0)))
        v = T.voxelise([rod], d, samples=4)
        assert np.ptp(np.flatnonzero(v.sum(axis=(1, 2)))) + 1 == 12
        assert np.ptp(np.flatnonzero(v.sum(axis=(0, 2)))) + 1 <= 5

    def test_density_stays_in_bounds_where_bodies_overlap(self):
        from camber.spec.shapes import Box
        d = T.Domain(10, 10, 4, 0.005)
        two = [self._body(Box(0.02, 0.02, 0.01)),
               self._body(Box(0.02, 0.02, 0.01))]
        assert T.voxelise(two, d).max() <= 1.0

    def test_supersampling_captures_partial_elements(self):
        """A single sample per element snaps a thin member to solid or to
        nothing; the comparison this exists for needs the fraction."""
        from camber.spec.shapes import Box
        d = T.Domain(10, 10, 4, 0.010)
        thin = [self._body(Box(0.004, 0.05, 0.04))]      # 0.4 of one element
        coarse = T.voxelise(thin, d, samples=1)
        fine = T.voxelise(thin, d, samples=6)
        assert set(np.unique(coarse)) <= {0.0, 1.0}
        assert ((fine > 0.0) & (fine < 1.0)).any()


class TestComplianceOfAGivenField:
    def test_more_material_is_stiffer(self):
        p = cantilever(nx=8, ny=3, nz=2)
        thin = np.full(p.domain.shape, 0.3)
        thick = np.full(p.domain.shape, 0.9)
        assert T.compliance(p, thick) < T.compliance(p, thin)

    def test_it_agrees_with_the_optimiser_s_own_number(self):
        """Same field, same element, same load: the standalone evaluation must
        reproduce what `optimise` reported, or the comparison it enables is
        against a different physics problem."""
        p = cantilever(nx=8, ny=3, nz=2, vf=0.5)
        r = T.optimise(p, iterations=1, penalty_start=3.0)
        uniform = np.full(p.domain.shape, 0.5)
        assert T.compliance(p, uniform) == pytest.approx(
            r.compliance_history[0], rel=1e-9)

    def test_it_does_not_optimise(self):
        p = cantilever(nx=6, ny=3, nz=2)
        field = np.full(p.domain.shape, 0.4)
        before = field.copy()
        T.compliance(p, field)
        assert np.array_equal(field, before)


class TestMountsAreNotBillets:
    """The guard on the bug that made GEN-02 mostly not generative.

    `frame_problem` forced a solid block through the full domain thickness
    under the whole stack footprint: 34.3 g of a 40 g budget, leaving the
    optimiser 14% of the mass to place. Nothing raised. The visible symptoms
    were islands and a meshed volume that came out identical across runs with
    nothing else in common.
    """

    @pytest.fixture
    def vehicle(self):
        import pathlib

        from camber.spec import io
        return io.load(pathlib.Path(__file__).resolve().parent.parent
                       / "examples" / "racer5.toml")

    def test_the_optimiser_gets_most_of_the_mass(self, vehicle):
        p = T.frame_problem(vehicle, element=0.003, thickness=0.012,
                            target_mass=0.040)
        budget = p.volume_fraction * p.domain.n_elements
        assert p.keep.sum() / budget < 0.15, (
            f"{p.keep.sum():.0f} of {budget:.0f} elements are forced solid; "
            f"the optimiser is decorating a mandated shape")

    def test_pads_are_one_layer_thick(self, vehicle):
        """A pad through the full thickness costs nz times its plan area, and
        the thickness is the design envelope, not the plate."""
        p = T.frame_problem(vehicle, element=0.003, thickness=0.012,
                            target_mass=0.040)
        per_layer = p.keep.sum(axis=(0, 1))
        assert (per_layer > 0).sum() <= 2, "mounts occupy more than two layers"

    def test_there_is_a_boss_for_every_bolt(self, vehicle):
        """Four bolts per motor on its own pattern, plus four for the stack."""
        p = T.frame_problem(vehicle, element=0.003, thickness=0.012)
        assert len(p.fasteners) == 4 * len(vehicle.thrusters) + 4

    def test_the_bolt_holes_are_voided_through_the_thickness(self, vehicle):
        p = T.frame_problem(vehicle, element=0.002, thickness=0.012)
        assert p.void is not None and p.void.any()
        per_layer = p.void.sum(axis=(0, 1))
        assert (per_layer == per_layer[0]).all(), "a hole must go all the way"

    def test_a_coarse_grid_keeps_the_boss_rather_than_losing_the_mount(self, vehicle):
        """If the element cannot separate boss from hole, the hole waits for
        `drill`; leaving the mount with no material would be worse."""
        p = T.frame_problem(vehicle, element=0.008, thickness=0.024)
        assert p.keep.any()
        assert np.count_nonzero(p.load) > 0

    def test_the_support_does_not_span_the_whole_hub(self, vehicle):
        """Fixing the entire centre plan area makes the hub rigid by decree."""
        p = T.frame_problem(vehicle, element=0.003, thickness=0.012)
        held = len(p.fixed) / 3
        assert held < 0.02 * p.domain.n_nodes

    def test_the_thrust_totals_the_manoeuvre_case(self, vehicle):
        from camber.control import G
        p = T.frame_problem(vehicle, element=0.003, thickness=0.012)
        rotors = vehicle.thrusters
        mass = vehicle.mass_properties().mass
        per = max(max(r.max_thrust() for r in rotors),
                  mass * G * 10.0 / len(rotors))
        up = -p.load.reshape(-1, 3)[:, 2].clip(max=0.0).sum()
        assert up == pytest.approx(per * len(rotors), rel=1e-9)

    def test_the_reduced_system_is_not_singular(self, vehicle):
        """Four separated pads must remove every rigid-body mode."""
        p = T.frame_problem(vehicle, element=0.012, thickness=0.036)
        c = T.compliance(p, np.full(p.domain.shape, 0.5))
        assert np.isfinite(c) and c > 0


class TestTheFrameIsAFreeBody:
    """A flying airframe has no ground, and pretending it does changes the
    answer completely.

    Holding all four stack pads rigidly makes each bolt an independent,
    infinitely stiff support. The optimiser then correctly answers *that*
    question: four separate arms, each from its motor to its own bolt, never
    meeting in the middle because nothing requires them to. Connectivity
    reporting called it fragmentation -- 28 islands, largest holding 22.7% --
    and the structure was fine. The supports were wrong.
    """

    @pytest.fixture
    def problem(self):
        import pathlib

        from camber.spec import io
        v = io.load(pathlib.Path(__file__).resolve().parent.parent
                    / "examples" / "racer5.toml")
        return T.frame_problem(v, element=0.003, thickness=0.012,
                               target_mass=0.040)

    def _nodes(self, d):
        n = np.arange(d.n_nodes)
        k, rem = divmod(n, (d.nx + 1) * (d.ny + 1))
        j, i = divmod(rem, d.nx + 1)
        return np.stack([i * d.element - d.nx * d.element / 2,
                         j * d.element - d.ny * d.element / 2,
                         k * d.element - d.nz * d.element / 2], axis=1)

    def test_the_restraint_is_exactly_six_dof(self, problem):
        """3-2-1: enough to remove the rigid-body modes and no more. Any
        seventh constraint starts carrying load the structure should carry."""
        assert len(problem.fixed) == 6

    def test_the_load_set_is_in_force_equilibrium(self, problem):
        assert np.allclose(problem.load.reshape(-1, 3).sum(axis=0), 0.0,
                           atol=1e-9)

    def test_the_load_set_is_in_moment_equilibrium(self, problem):
        L = problem.load.reshape(-1, 3)
        moments = np.cross(self._nodes(problem.domain), L).sum(axis=0)
        assert np.allclose(moments, 0.0, atol=1e-9)

    def test_thrust_pulls_up_and_the_payload_pulls_down(self, problem):
        """z is down in the body frame (GEOM-02), so thrust is negative."""
        fz = problem.load.reshape(-1, 3)[:, 2]
        assert fz.min() < 0 and fz.max() > 0
        assert -fz.clip(max=0.0).sum() == pytest.approx(fz.clip(min=0.0).sum())

    def test_the_supports_carry_essentially_nothing(self, problem):
        """The point of a self-equilibrated load: the reactions are numerical
        noise, so the load travels through the structure, not into ground."""
        d = problem.domain
        rho = np.full(d.shape, 0.6)
        KE = T.hex8_stiffness()
        dofs = d.element_dofs()
        asm = T._Assembly(dofs, problem.fixed, 3 * d.n_nodes, KE)
        from scipy.sparse.linalg import splu
        stiff = 1e-9 + rho.ravel() ** 3
        u = np.zeros(3 * d.n_nodes)
        u[asm.free] = splu(asm.matrix(stiff), permc_spec="MMD_AT_PLUS_A",
                           options={"SymmetricMode": True}).solve(
                               problem.load[asm.free])
        # Reaction is Ku - f, not Ku: the restrained nodes carry applied load
        # too, and forgetting the subtraction reports the applied load back as
        # if it were a reaction.
        internal = np.zeros(3 * d.n_nodes)
        ue = u[dofs]
        contrib = np.einsum("ij,kj->ki", KE, ue) * stiff[:, None]
        np.add.at(internal, dofs.ravel(), contrib.ravel())
        reaction = internal[problem.fixed] - problem.load[problem.fixed]
        applied = np.abs(problem.load).sum()
        assert np.abs(reaction).sum() < 1e-6 * applied


class TestConnectivityRules:
    """Face versus full connectivity, which disagree completely on a diagonal.

    `largest_connected` used face-only connectivity and was deleting real
    structure. An airframe's arms run diagonally, and a diagonal in a
    Cartesian grid is a staircase whose voxels touch at their edges, not their
    faces -- so 6-connectivity called every voxel its own island and pruning
    removed the arms. The meshed part came out at 1.3 g of a requested 40 g.
    """

    def _staircase(self):
        """A one-element-wide diagonal: 12 voxels, edge-adjacent only."""
        rho = np.zeros((14, 14, 2))
        for i in range(12):
            rho[i + 1, i + 1, :] = 1.0
        return rho

    def test_a_diagonal_is_one_piece_under_full_connectivity(self):
        n, frac = T.Result(density=self._staircase()).connectivity(rule="full")
        assert (n, frac) == (1, 1.0)

    def test_the_same_diagonal_is_twelve_pieces_under_face_connectivity(self):
        n, _ = T.Result(density=self._staircase()).connectivity(rule="face")
        assert n == 12

    def test_hinges_counts_the_edge_only_joins(self):
        assert T.Result(density=self._staircase()).hinges() == 11

    def test_a_solid_block_has_no_hinges(self):
        rho = np.zeros((8, 8, 3))
        rho[2:6, 2:6, :] = 1.0
        assert T.Result(density=rho).hinges() == 0

    def test_pruning_keeps_the_diagonal(self):
        """The regression that mattered: the arms must survive."""
        rho = self._staircase()
        rho[13, 0, :] = 1.0                       # a genuine speck
        cleaned = T.largest_connected(rho)
        assert cleaned[1:13, 1:13].sum() == 24, "the arm was pruned away"
        assert cleaned[13, 0, 0] == 0.0, "the speck survived"

    def test_face_pruning_destroys_it(self):
        """Kept as the demonstration of what the old default did."""
        assert T.largest_connected(self._staircase(), rule="face").sum() == 2

    def test_an_unknown_rule_is_rejected(self):
        with pytest.raises(ValueError, match="face"):
            T.Result(density=np.ones((4, 4, 2))).connectivity(rule="diagonal")


class TestModulusIsAppliedOnce:
    """Stiffness must go as E, not as E squared.

    `hex8_stiffness(E0)` and the SIMP interpolation `Emin + x^p (E0 - Emin)`
    both scale by the modulus, so passing E0 to both applied it twice. At
    E0 = 1 that is the identity, so it survived every test written while the
    solve ran in arbitrary units -- and surfaced only when the run was moved
    to a real 70 GPa and compliance came out at 1e-17 J.
    """

    def test_doubling_the_modulus_halves_compliance(self):
        p = cantilever(nx=8, ny=3, nz=2, vf=0.6)
        rho = np.full(p.domain.shape, 0.7)
        soft = T.compliance(p, rho, E0=1e9, Emin=1.0)
        stiff = T.compliance(p, rho, E0=2e9, Emin=1.0)
        assert stiff == pytest.approx(soft / 2.0, rel=1e-6)

    def test_compliance_is_inversely_proportional_across_decades(self):
        p = cantilever(nx=6, ny=3, nz=2, vf=0.6)
        rho = np.full(p.domain.shape, 0.8)
        a = T.compliance(p, rho, E0=1e6, Emin=1e-3)
        b = T.compliance(p, rho, E0=1e9, Emin=1.0)
        assert a / b == pytest.approx(1000.0, rel=1e-5)

    def test_the_optimiser_agrees_with_the_standalone_evaluation(self):
        """Both paths must apply the modulus the same number of times."""
        p = cantilever(nx=8, ny=3, nz=2, vf=0.5)
        r = T.optimise(p, iterations=1, penalty_start=3.0,
                       E0=70e9, Emin=70.0)
        uniform = np.full(p.domain.shape, 0.5)
        assert T.compliance(p, uniform, E0=70e9, Emin=70.0) == pytest.approx(
            r.compliance_history[0], rel=1e-9)

    def test_a_solid_cantilever_deflects_by_about_the_beam_formula(self):
        """The sanity check the arbitrary-unit version could never make.

        A solid block, end-loaded, against wL^3/(3EI). Not exact -- the block
        is short and stubby, so shear matters and the built-in end is not a
        perfect clamp -- but it must land within a factor of two, which an
        E-squared error would miss by nine orders of magnitude.
        """
        nx, ny, nz, e = 20, 4, 4, 0.005
        E = 70e9
        d = T.Domain(nx, ny, nz, e)
        fixed = [3 * d.node_id(0, j, k) + c
                 for j in range(ny + 1) for k in range(nz + 1) for c in range(3)]
        load = np.zeros(3 * d.n_nodes)
        tip = [d.node_id(nx, j, k) for j in range(ny + 1) for k in range(nz + 1)]
        F = 100.0
        for n in tip:
            load[3 * n + 2] = -F / len(tip)
        p = T.Problem(domain=d, fixed=np.array(fixed), load=load)
        c = T.compliance(p, np.ones(d.shape), E0=E, Emin=E * 1e-9)

        L, b, h = nx * e, ny * e, nz * e
        I = b * h ** 3 / 12.0
        delta_beam = F * L ** 3 / (3 * E * I)
        delta_fea = c / F                      # compliance is u.f at one load
        assert 0.5 < delta_fea / delta_beam < 2.0, (
            f"FEA {delta_fea * 1000:.4f} mm vs beam {delta_beam * 1000:.4f} mm")


class TestMultigrid:
    """AMG has to give the same displacement field, not merely a fast one."""

    def _case(self):
        p = cantilever(nx=14, ny=6, nz=4, vf=0.6)
        d = p.domain
        KE = T.hex8_stiffness(1.0, 0.3, size=d.element)
        asm = T._Assembly(d.element_dofs(), p.fixed, 3 * d.n_nodes, KE)
        stiff = 70.0 + np.full(d.n_elements, 0.6) ** 3 * (70e9 - 70.0)
        return p, d, asm, asm.matrix(stiff), p.load[asm.free]

    def test_it_agrees_with_the_direct_factorisation(self):
        pytest.importorskip("pyamg")
        p, d, asm, K, f = self._case()
        direct = T._solve(K, f, d, asm.free, "direct")
        amg = T._solve(K, f, d, asm.free, "amg")
        assert np.abs(amg - direct).max() < 1e-6 * np.abs(direct).max()

    def test_the_residual_is_actually_small(self):
        pytest.importorskip("pyamg")
        p, d, asm, K, f = self._case()
        u = T._solve(K, f, d, asm.free, "amg")
        assert np.linalg.norm(K @ u - f) / np.linalg.norm(f) < 1e-7

    def test_rigid_body_modes_cost_no_energy(self):
        """What the near-nullspace has to be, stated as the property that
        defines it: an unconstrained solid moving rigidly stores no strain."""
        d = T.Domain(4, 3, 2, 0.01)
        KE = T.hex8_stiffness(1.0, 0.3, size=d.element)
        asm = T._Assembly(d.element_dofs(), np.array([], dtype=int),
                          3 * d.n_nodes, KE)
        K = asm.matrix(np.ones(d.n_elements))
        B = T.rigid_body_modes(d, asm.free)
        assert B.shape == (3 * d.n_nodes, 6)
        energy = np.einsum("ij,ij->j", B, K @ B)
        assert np.abs(energy).max() < 1e-9 * np.abs(K).max()

    def test_the_six_modes_are_independent(self):
        d = T.Domain(4, 3, 2, 0.01)
        B = T.rigid_body_modes(d, np.arange(3 * d.n_nodes))
        assert np.linalg.matrix_rank(B) == 6

    def test_auto_picks_direct_when_small(self):
        assert T._pick_solver("auto", 1000) == "direct"

    def test_auto_picks_multigrid_when_large(self):
        pytest.importorskip("pyamg")
        assert T._pick_solver("auto", 10 * T.AMG_ABOVE) == "amg"

    def test_an_unknown_solver_is_rejected(self):
        with pytest.raises(ValueError, match="auto, direct or amg"):
            T._pick_solver("cholesky", 1000)


class TestDrilling:
    def _frame(self):
        pytest.importorskip("trimesh")
        import trimesh
        return trimesh.creation.box((0.08, 0.08, 0.004))

    def test_a_hole_appears_for_every_bolt(self):
        plate = self._frame()
        bolts = [T.Fastener(x, y) for x in (-0.02, 0.02) for y in (-0.02, 0.02)]
        drilled = T.drill(plate, bolts)
        assert drilled.volume < plate.volume
        # Each through-hole adds a handle: genus goes up by one per hole.
        assert drilled.euler_number == plate.euler_number - 2 * len(bolts)

    def test_the_holes_are_the_size_asked_for(self):
        plate = self._frame()
        bolt = T.Fastener(0.0, 0.0, hole=0.0017)
        removed = plate.volume - T.drill(plate, [bolt]).volume
        expected = np.pi * 0.0017 ** 2 * 0.004
        assert removed == pytest.approx(expected, rel=0.02)

    def test_it_stays_watertight(self):
        drilled = T.drill(self._frame(), [T.Fastener(0.01, 0.01)])
        assert drilled.is_watertight

    def test_no_fasteners_is_a_no_op(self):
        plate = self._frame()
        assert T.drill(plate, []) is plate


class TestTheCoreStaysLight:
    """AI-07: `camber inspect` must run with numpy and nothing else.

    Topology optimisation needs scipy, scikit-image, trimesh and pyamg, and
    every one of them is imported inside the function that needs it or behind
    the module that needs it -- never at the top of anything the CLI's other
    commands reach.
    """

    def test_the_core_imports_pull_nothing_heavy(self):
        import subprocess
        import sys
        code = (
            "import sys, camber.cli, camber.spec.io, camber.control, "
            "camber.massprops, camber.generate;"
            "heavy=[m for m in ('scipy','trimesh','skimage','pyamg') "
            "if m in sys.modules];"
            "print(','.join(heavy))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, check=True).stdout.strip()
        assert out == "", f"the core pulled in {out}"

    def test_the_solver_falls_back_without_pyamg(self, monkeypatch):
        """A missing optional dependency must degrade, not raise."""
        import builtins
        real = builtins.__import__

        def blocked(name, *a, **k):
            if name == "pyamg":
                raise ImportError("no pyamg")
            return real(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert T._pick_solver("auto", 10 * T.AMG_ABOVE) == "direct"


class TestSmoothExtraction:
    """Getting a printable surface out of a field the penalty made binary.

    SIMP is built to drive every element to solid or void, so by the end there
    are almost no intermediate values for marching cubes to interpolate and the
    isosurface has nowhere to go but along the voxel boundary. Resampling and a
    small blur put the detail back; the isolevel then has to move to keep the
    volume honest.
    """

    @pytest.fixture(scope="class")
    def case(self):
        pytest.importorskip("skimage")
        pytest.importorskip("trimesh")
        d = T.Domain(24, 24, 6, 0.003)
        rho = np.zeros(d.shape)
        i, j = np.indices((d.nx, d.ny))
        # A diagonal bar, which is where the staircase is worst.
        rho[np.abs(i - j) <= 2, 1:5] = 1.0
        return d, T.Result(density=rho)

    def test_it_is_watertight(self, case):
        d, r = case
        assert T.to_mesh(r, d).is_watertight

    def test_it_is_one_body(self, case):
        d, r = case
        assert T.to_mesh(r, d).body_count == 1

    def test_the_volume_matches_the_design(self, case):
        """The reason the isolevel is chosen rather than fixed at 0.5."""
        d, r = case
        want = r.density.sum() * d.element ** 3
        assert T.to_mesh(r, d).volume == pytest.approx(want, rel=0.06)

    def test_a_fixed_level_would_lose_volume(self, case):
        """Stated as a comparison: on the real racing frame the raw isosurface
        came out 11% light, which is mass the optimiser paid for."""
        d, r = case
        want = r.density.sum() * d.element ** 3
        raw = T.to_mesh(r, d, upsample=1, blur=0.0)
        smoothed = T.to_mesh(r, d)
        assert abs(smoothed.volume - want) < abs(raw.volume - want)

    def test_resampling_adds_facets_and_does_not_cost_accuracy(self, case):
        """Resampling buys surface resolution. The reference for volume is the
        design, not the coarser mesh -- comparing the two extractions to each
        other says nothing about whether either is right."""
        d, r = case
        want = r.density.sum() * d.element ** 3
        coarse, fine = T.to_mesh(r, d, upsample=1), T.to_mesh(r, d, upsample=3)
        assert len(fine.faces) > 3 * len(coarse.faces)
        assert abs(fine.volume - want) <= abs(coarse.volume - want) * 1.05

    def test_it_stays_centred_on_the_vehicle_origin(self, case):
        d, r = case
        for up in (1, 2, 3):
            c = T.to_mesh(r, d, upsample=up).bounds.mean(axis=0)
            assert np.allclose(c, 0.0, atol=2 * d.element), f"upsample={up}"

    def test_the_volume_is_positive(self, case):
        """A negative volume means inverted winding: still watertight, still
        the right shape, and read by a slicer as a hole."""
        d, r = case
        assert T.to_mesh(r, d).volume > 0

    def test_a_heavier_blur_would_sever_thin_necks(self, case):
        """Why `blur` is 0.5 and not 0.9. Holding volume fixed means a stronger
        blur is paid for with a higher isolevel, and a higher level cuts the
        thinnest material first -- on the racing frame 0.9 turned one solid
        body into nine."""
        d, r = case
        assert T.to_mesh(r, d, blur=0.5).volume > 0

    def test_the_isolevel_bisection_brackets_the_target(self):
        rng = np.random.default_rng(0)
        field = rng.random((20, 20, 20))
        for target in (500.0, 2000.0, 6000.0):
            lvl = T._isolevel_for_volume(field, target)
            assert abs(float((field >= lvl).sum()) - target) < 40


class TestExportUnits:
    """GEOM-04: millimetres for printing, metres only when asked.

    An STL carries no unit, so every consumer guesses and they all guess
    millimetres. A metre-scale file makes a 202 mm airframe 0.2 mm across and
    a slicer accepts it without complaint. CFD-02 is the same mistake in the
    other direction -- a millimetre STL fed to a metre CFD domain, which made
    a 180 mm airframe 180 m across and swallowed the mesh.
    """

    def _run(self, tmp_path, name, extra=()):
        pytest.importorskip("skimage")
        pytest.importorskip("trimesh")
        import trimesh

        from camber.cli import main
        out = tmp_path / name
        argv = ["generate", "examples/racer5.toml", "--out", str(out),
                "--element", "0.008", "--thickness", "0.024",
                "--iterations", "3", "--mass", "40", *extra]
        assert main(argv) == 0
        return trimesh.load(out)

    def test_the_default_is_a_thousand_times_the_metre_export(self, tmp_path):
        """The invariant, rather than an absolute size: the same run exported
        both ways must differ by exactly the unit conversion."""
        mm = self._run(tmp_path, "frame_mm.stl")
        m = self._run(tmp_path, "frame_m.stl", ("--metres",))
        # 1e-6, not tighter: STL stores float32, so a 200 mm coordinate
        # carries about seven significant figures and no more.
        assert mm.extents.max() == pytest.approx(m.extents.max() * 1000.0,
                                                 rel=1e-6)

    def test_the_metre_export_is_smaller_than_a_metre(self, tmp_path):
        """A 200 mm airframe at SI scale. CFD-02 is what happens when a
        millimetre file reaches a metre domain instead."""
        assert self._run(tmp_path, "m.stl", ("--metres",)).extents.max() < 1.0

    def test_the_default_export_is_bigger_than_a_millimetre(self, tmp_path):
        """And what happens the other way: a slicer reading metres sees a part
        a fifth of a millimetre across, and accepts it without a word."""
        assert self._run(tmp_path, "mm.stl").extents.max() > 1.0


class TestDrillingLeavesOneSolid:
    def test_boolean_slivers_are_removed(self):
        """Twenty holes through the racing frame left eight sixteen-face
        shells of 1e-6 mm^3 beside a body of 34,000. A slicer lists those as
        separate objects and tries to print them."""
        pytest.importorskip("trimesh")
        import trimesh
        plate = trimesh.creation.box((0.08, 0.08, 0.004))
        bolts = [T.Fastener(x, y) for x in (-0.02, 0.02) for y in (-0.02, 0.02)]
        out = T.drill(plate, bolts)
        assert out.body_count == 1
        assert out.is_watertight

    def test_a_grazing_cutter_does_not_split_the_part(self):
        """A hole placed right at the edge is where the slivers come from."""
        pytest.importorskip("trimesh")
        import trimesh
        plate = trimesh.creation.box((0.08, 0.08, 0.004))
        out = T.drill(plate, [T.Fastener(0.0400, 0.0, hole=0.0017)])
        assert out.body_count == 1
