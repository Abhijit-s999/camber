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

    def test_load_and_support_do_not_overlap(self, vehicle):
        """Loading a fixed dof puts force straight into ground and the
        optimiser sees no reason to build anything between them."""
        p = T.frame_problem(vehicle, element=0.006)
        loaded = set(np.nonzero(p.load)[0].tolist())
        assert not (loaded & set(p.fixed.tolist()))

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
