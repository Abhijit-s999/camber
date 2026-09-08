"""Generative structure: SIMP topology optimisation with FEA in the loop.

This grows an airframe rather than sizing one. The design domain is filled with
material, a load path is imposed, and material that is not carrying load is
removed until only the structure that does the work remains. The organic,
branching shapes that result are not styling -- they are the load paths made
visible.

**The method.** SIMP, Solid Isotropic Material with Penalisation, which is the
standard approach:

1. Voxelise the design domain. Every element gets a density ``rho`` in [0, 1].
2. Its stiffness is ``E0 * rho^p``. The penalty ``p = 3`` makes intermediate
   densities structurally inefficient, so the optimiser is pushed towards
   solid-or-void rather than a grey mush that cannot be manufactured.
3. Solve ``K u = f`` for the displacement under load.
4. Compliance ``uT K u`` is the objective -- minimising it maximises stiffness.
   Its sensitivity has a closed form, which is what makes this tractable at
   tens of thousands of elements.
5. Filter the sensitivities over a radius. Without this the solution develops
   checkerboard patterns that are a numerical artefact of the element
   formulation rather than real structure.
6. Update densities by optimality criteria, holding the volume fraction fixed.

**Why this needed FEA and the beam model would not do.** GEN-01 rejected
topology optimisation on the grounds that no closed-form model can check the
result. That was the right objection and the wrong conclusion: the answer is to
put a real structural solve in the loop, which is what this is. The compliance
and the displacement field come from the same FEA that decides where material
goes, so the shape and its analysis are consistent by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def hex8_stiffness(E: float = 1.0, nu: float = 0.3, size: float = 1.0) -> np.ndarray:
    """Stiffness of an 8-node hexahedral element, 24 x 24, by quadrature.

    Integrated with 2x2x2 Gauss points rather than copied from a published
    closed-form table. The tables exist and are faster to type, but they are
    1-indexed MATLAB and transcribing them wrongly is *silent*: the matrix
    still builds, is still 24x24 and is still symmetric -- it is simply a
    different element, and the only way to notice is to check its eigenvalues.
    Two attempts at transcription produced matrices with negative eigenvalues
    before this was written instead.

    Quadrature is self-checking: the result must be positive semi-definite with
    exactly six zero eigenvalues, the rigid-body modes of an unconstrained
    brick, and `test_topology.py` asserts precisely that.
    """
    # Isotropic elasticity matrix.
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu

    # Corner order must match Domain.element_dofs().
    corners = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], dtype=float)
    g = 1.0 / math.sqrt(3.0)
    gauss = [(a, b, c) for a in (-g, g) for b in (-g, g) for c in (-g, g)]

    KE = np.zeros((24, 24))
    half = size / 2.0                      # Jacobian of a cube of edge `size`
    detJ = half ** 3
    for xi, eta, zeta in gauss:
        # dN/d(xi,eta,zeta) for the trilinear brick.
        dN = np.empty((3, 8))
        for i, (a, b, c) in enumerate(corners):
            dN[0, i] = a * (1 + b * eta) * (1 + c * zeta) / 8.0
            dN[1, i] = b * (1 + a * xi) * (1 + c * zeta) / 8.0
            dN[2, i] = c * (1 + a * xi) * (1 + b * eta) / 8.0
        dNdx = dN / half                   # J = half * I for a cube

        B = np.zeros((6, 24))
        for i in range(8):
            x, y, z = dNdx[:, i]
            B[0, 3*i]   = x
            B[1, 3*i+1] = y
            B[2, 3*i+2] = z
            B[3, 3*i]   = y; B[3, 3*i+1] = x
            B[4, 3*i+1] = z; B[4, 3*i+2] = y
            B[5, 3*i]   = z; B[5, 3*i+2] = x
        KE += B.T @ D @ B * detJ
    return KE


@dataclass
class Domain:
    """The box material may occupy, discretised into elements."""

    nx: int
    ny: int
    nz: int
    #: Physical size of one cubic element, metres.
    element: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def n_elements(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    def node_id(self, i: int, j: int, k: int) -> int:
        return (k * (self.ny + 1) + j) * (self.nx + 1) + i

    def element_dofs(self) -> np.ndarray:
        """(n_elements, 24) global DOF indices, in the element's node order."""
        ex, ey, ez = np.meshgrid(np.arange(self.nx), np.arange(self.ny),
                                 np.arange(self.nz), indexing="ij")
        ex, ey, ez = ex.ravel(), ey.ravel(), ez.ravel()
        corners = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),
                   (0,0,1),(1,0,1),(1,1,1),(0,1,1)]
        dofs = np.empty((self.n_elements, 24), dtype=np.int64)
        for c, (di, dj, dk) in enumerate(corners):
            n = self.node_id(ex + di, ey + dj, ez + dk)
            dofs[:, 3*c:3*c+3] = np.stack([3*n, 3*n+1, 3*n+2], axis=1)
        return dofs


@dataclass
class Problem:
    """Loads, supports, and which elements the optimiser may not touch."""

    domain: Domain
    #: Fixed DOFs (indices into the global displacement vector).
    fixed: np.ndarray
    #: Applied loads, same indexing.
    load: np.ndarray
    #: Elements forced solid (mounts, bearing surfaces) and forced void.
    keep: np.ndarray | None = None
    void: np.ndarray | None = None
    volume_fraction: float = 0.3


@dataclass
class Result:
    density: np.ndarray                  # (nx, ny, nz) in [0, 1]
    compliance_history: list[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    #: Penalty the run finished at, which continuation makes worth recording.
    penalty: float = 3.0

    @property
    def volume_fraction(self) -> float:
        return float(self.density.mean())

    def connectivity(self, threshold: float = 0.4) -> tuple[int, float]:
        """(number of solid islands, fraction of material in the largest).

        Reported rather than left to be discovered downstream. A run that
        scatters a third of its volume budget into islands has not failed
        loudly -- compliance still fell, the field is still 96% binary -- and
        the only symptom is that the part you finally mesh weighs a third less
        than you asked for. The first run of this optimiser did exactly that.
        """
        from scipy import ndimage

        solid = self.density >= threshold
        if not solid.any():
            return 0, 0.0
        labels, n = ndimage.label(solid)
        if n <= 1:
            return n, 1.0
        mass = ndimage.sum(self.density, labels, range(1, n + 1))
        return n, float(mass.max() / mass.sum())


def _filter_matrix(domain: Domain, radius: float) -> sp.csr_matrix:
    """Cone-weighted neighbourhood averaging over elements.

    Sensitivity filtering is not cosmetic. Without it the checkerboard pattern
    -- alternating solid and void elements -- appears and is *stiffer* in the
    discrete model than real material would be, so the optimiser drives towards
    a structure that exists only as an artefact of the element formulation.
    """
    nx, ny, nz = domain.shape
    r = int(math.ceil(radius)) - 1
    idx = np.arange(domain.n_elements).reshape(nx, ny, nz)

    # One vectorised pass per offset rather than a quadruple Python loop over
    # every (element, neighbour) pair. The frame domain has 33,124 elements and
    # a radius-2.4 stencil reaches about a hundred of them each, so the loop
    # form is three million Python iterations of `math.dist` before the first
    # solve; there are only a hundred distinct offsets.
    rows, cols, vals = [], [], []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            for dk in range(-r, r + 1):
                w = radius - math.sqrt(di*di + dj*dj + dk*dk)
                if w <= 0:
                    continue
                a = idx[max(di, 0):nx + min(di, 0),
                        max(dj, 0):ny + min(dj, 0),
                        max(dk, 0):nz + min(dk, 0)]
                b = idx[max(-di, 0):nx + min(-di, 0),
                        max(-dj, 0):ny + min(-dj, 0),
                        max(-dk, 0):nz + min(-dk, 0)]
                rows.append(a.ravel()); cols.append(b.ravel())
                vals.append(np.full(a.size, w))
    H = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(domain.n_elements, domain.n_elements)).tocsr()
    return H


def penalty_at(iteration: int, iterations: int, final: float,
               start: float, ramp: float = 0.5, step: float = 0.25) -> float:
    """Penalty for a given iteration under continuation.

    Going straight to ``p = 3`` on a low volume fraction is what scatters the
    material. At ``p = 3`` an element at density 0.3 is worth 2.7% of solid
    stiffness, so joining two members through a thin grey neck is priced almost
    as dearly as leaving them separate, and the optimiser is free to park mass
    in islands that carry nothing. At ``p = 1`` the problem is convex and grey
    is honestly priced, so a connected layout forms first; raising ``p`` in
    quarter steps then hardens that layout instead of choosing a new one.

    The ramp occupies the first ``ramp`` of the iteration budget, and the
    remainder runs at the final penalty -- which is where the convergence test
    means anything, since every penalty increase moves the field on purpose.
    """
    if final <= start:
        return final
    span = max(1, int(round(ramp * iterations)))
    if iteration >= span:
        return final
    raw = start + (final - start) * (iteration / span)
    return min(final, start + step * math.floor((raw - start) / step + 1e-9))


class _Assembly:
    """Precomputed sparsity structure of K restricted to the free DOFs.

    Every SIMP iteration assembles the same matrix with different numbers in
    it: the mesh does not move, so the pattern of non-zeros never changes and
    only the values do. Rebuilding it from scratch each time was costing more
    than the linear solve.

    The naive loop did three avoidable things per iteration. It built a COO
    matrix of 24 x 24 x n_elements entries -- 19 million for the racing frame
    -- and converted it, which sorts all 19 million and sums duplicates. Then
    it took ``K[free, :][:, free]``, two fancy-index passes over a 127,000
    square sparse matrix, each of which allocates a full copy. Doing that
    eighty times is eighty times the same sort for the same answer.

    Here the sort happens once. Fixed DOFs are dropped at the index level
    rather than sliced out afterwards, the duplicate-summing is a
    ``reduceat`` over a precomputed ordering, and the CSC arrays are handed to
    scipy already built.
    """

    def __init__(self, dofs: np.ndarray, fixed: np.ndarray, n_dof: int,
                 KE: np.ndarray) -> None:
        self.free = np.setdiff1d(np.arange(n_dof), fixed)
        nf = self.free.size
        remap = np.full(n_dof, -1, dtype=np.int64)
        remap[self.free] = np.arange(nf)

        r = remap[dofs]                                     # (n_el, 24)
        rows = np.repeat(r, 24, axis=1).ravel()
        cols = np.tile(r, (1, 24)).ravel()
        # A row or column belonging to a fixed DOF contributes nothing to the
        # reduced system; dropping it here is what avoids the slicing.
        sel = np.flatnonzero((rows >= 0) & (cols >= 0))
        rows, cols = rows[sel], cols[sel]

        # Column-major keys give CSC ordering directly.
        keys = cols.astype(np.int64) * nf + rows
        order = np.argsort(keys, kind="stable")
        sel, keys = sel[order], keys[order]
        first = np.empty(keys.size, dtype=bool)
        first[0] = True
        np.not_equal(keys[1:], keys[:-1], out=first[1:])
        self.starts = np.flatnonzero(first)

        uniq = keys[self.starts]
        self.indices = (uniq % nf).astype(np.int32)
        counts = np.bincount((uniq // nf).astype(np.int64), minlength=nf)
        self.indptr = np.empty(nf + 1, dtype=np.int32)
        self.indptr[0] = 0
        np.cumsum(counts, out=self.indptr[1:])
        self.shape = (nf, nf)

        # Per-entry element index and the KE value it carries, so a rebuild is
        # one multiply over a gather rather than an outer product.
        self.element = (sel // 576).astype(np.int32)
        self.ke = KE.ravel()[sel % 576]

    def matrix(self, stiffness: np.ndarray) -> sp.csc_matrix:
        data = np.add.reduceat(self.ke * stiffness[self.element], self.starts)
        return sp.csc_matrix((data, self.indices, self.indptr), shape=self.shape)


def _has_settled(history: list[float], window: int = 5,
                 tol: float = 1e-3) -> bool:
    """True when compliance has gone flat over the last `window` iterations.

    The usual SIMP stopping rule is "the largest density change fell below
    1%", and on its own it is wrong at low volume fractions. Running this
    optimiser without continuation stopped after four iterations by exactly
    that rule, reporting ``converged=True`` with compliance forty times
    *worse* than the uniform field it started from. Nothing was converged: a
    few hundred solid elements spread over thirty thousand can rearrange a
    load path -- connect it or sever it -- while no single element moves much,
    so the density criterion goes quiet precisely when compliance is at its
    most violent.

    So both must hold: the field has stopped moving *and* the objective has
    stopped changing. The window catches oscillation, where consecutive
    iterations differ little but the run is flipping between two layouts.
    """
    if len(history) < window:
        return False
    tail = history[-window:]
    lo, hi = min(tail), max(tail)
    return lo > 0.0 and (hi - lo) / lo < tol


def optimise(problem: Problem, iterations: int = 40, penalty: float = 3.0,
             filter_radius: float = 1.5, move: float = 0.2,
             E0: float = 1.0, Emin: float = 1e-9, nu: float = 0.3,
             tol: float = 0.01, verbose: bool = False,
             penalty_start: float = 1.0, on_iteration=None) -> Result:
    """Run SIMP until the density field stops moving.

    ``penalty_start`` below ``penalty`` enables continuation; set the two equal
    to go straight to the final penalty.
    """
    d = problem.domain
    KE = hex8_stiffness(E0, nu)
    dofs = d.element_dofs()
    n_dof = 3 * d.n_nodes

    H = _filter_matrix(d, filter_radius)
    Hs = np.asarray(H.sum(axis=1)).ravel()

    x = np.full(d.n_elements, problem.volume_fraction)
    if problem.keep is not None:
        x[problem.keep.ravel()] = 1.0
    if problem.void is not None:
        x[problem.void.ravel()] = 0.0

    asm = _Assembly(dofs, problem.fixed, n_dof, KE)
    free = asm.free
    f_free = problem.load[free]

    result = Result(density=x.reshape(d.shape))
    previous = x.copy()

    p_final = penalty
    for it in range(iterations):
        penalty = penalty_at(it, iterations, p_final, penalty_start)
        stiff = Emin + x ** penalty * (E0 - Emin)
        u = np.zeros(n_dof)
        try:
            # SymmetricMode with minimum-degree ordering on A + A^T: K is
            # symmetric positive definite, and telling SuperLU so cuts the
            # fill-in that dominates a 3D elasticity factorisation.
            lu = splu(asm.matrix(stiff), permc_spec="MMD_AT_PLUS_A",
                      options={"SymmetricMode": True})
            u[free] = lu.solve(f_free)
        except Exception:                                   # noqa: BLE001
            break

        ue = u[dofs]
        ce = np.einsum("ij,jk,ik->i", ue, KE, ue)
        compliance = float((stiff * ce).sum())
        result.compliance_history.append(compliance)

        dc = -penalty * x ** (penalty - 1.0) * (E0 - Emin) * ce
        dv = np.ones(d.n_elements)
        dc = H @ (dc / Hs)
        dv = H @ (dv / Hs)

        # Optimality criteria: bisect the Lagrange multiplier until the volume
        # constraint is met exactly.
        lo, hi = 1e-9, 1e9
        target = problem.volume_fraction * d.n_elements
        xnew = x
        while (hi - lo) / (lo + hi) > 1e-6:
            mid = 0.5 * (lo + hi)
            xnew = np.clip(
                np.clip(x * np.sqrt(np.maximum(-dc / (dv * mid), 0.0)),
                        x - move, x + move), 0.0, 1.0)
            if problem.keep is not None:
                xnew[problem.keep.ravel()] = 1.0
            if problem.void is not None:
                xnew[problem.void.ravel()] = 0.0
            if xnew.sum() > target:
                lo = mid
            else:
                hi = mid

        change = float(np.abs(xnew - previous).max())
        previous, x = xnew.copy(), xnew
        result.density = x.reshape(d.shape)
        result.iterations = it + 1
        result.penalty = penalty
        if verbose:
            print(f"  it {it+1:3d}  p {penalty:4.2f}  "
                  f"compliance {compliance:10.4e}  "
                  f"vol {x.mean():.3f}  change {change:.4f}", flush=True)
        if on_iteration:
            on_iteration(it, compliance, x.reshape(d.shape))
        # Only meaningful once the ramp is done: each penalty step moves the
        # field deliberately, so "it stopped moving" mid-ramp says nothing.
        if (change < tol and penalty >= p_final
                and _has_settled(result.compliance_history)):
            result.converged = True
            break

    return result


def compliance(problem: Problem, density: np.ndarray, penalty: float = 3.0,
               E0: float = 1.0, Emin: float = 1e-9, nu: float = 0.3) -> float:
    """Compliance of a given density field under the problem's load.

    One FEA evaluation, no optimisation. This is what makes "the grown frame
    is stiffer" a measurement rather than a claim: voxelise the straight-tube
    frame into the same domain, apply the same load through the same element,
    and read off the same number the optimiser was minimising.
    """
    d = problem.domain
    KE = hex8_stiffness(E0, nu)
    dofs = d.element_dofs()
    asm = _Assembly(dofs, problem.fixed, 3 * d.n_nodes, KE)
    x = np.clip(density.ravel(), 0.0, 1.0)
    stiff = Emin + x ** penalty * (E0 - Emin)
    u = np.zeros(3 * d.n_nodes)
    u[asm.free] = splu(asm.matrix(stiff), permc_spec="MMD_AT_PLUS_A",
                       options={"SymmetricMode": True}).solve(
                           problem.load[asm.free])
    ue = u[dofs]
    return float((stiff * np.einsum("ij,jk,ik->i", ue, KE, ue)).sum())


def voxelise(bodies, domain: Domain, samples: int = 2) -> np.ndarray:
    """Rasterise placed Box and Cylinder bodies into the design domain.

    `samples` supersampling points per axis within each element, so a tube
    narrower than an element still registers as partial density rather than
    vanishing or snapping to solid. Without it a 8 mm arm in a 3 mm grid
    aliases badly enough to change the comparison it exists to make.
    """
    from ..spec.shapes import Box, Cylinder

    nx, ny, nz = domain.shape
    e = domain.element
    offs = (np.arange(samples) + 0.5) / samples - 0.5
    gi, gj, gk, si, sj, sk = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), offs, offs, offs,
        indexing="ij")
    # Element centres, in vehicle coordinates (the domain is origin-centred).
    px = (gi + 0.5 + si) * e - nx * e / 2.0
    py = (gj + 0.5 + sj) * e - ny * e / 2.0
    pz = (gk + 0.5 + sk) * e - nz * e / 2.0
    pts = np.stack([px.ravel(), py.ravel(), pz.ravel()], axis=1)

    inside = np.zeros(pts.shape[0], dtype=bool)
    for b in bodies:
        origin = np.array([b.position.x, b.position.y, b.position.z])
        local = pts - origin
        rot = getattr(b.placement, "rotation", None)
        if rot is not None:
            # Rotation.matrix maps body axes into parent, so the inverse --
            # right-multiplying the row vectors -- takes points the other way.
            local = local @ np.asarray(rot.matrix)
        if isinstance(b.shape, Box):
            h = np.array([b.shape.lx, b.shape.ly, b.shape.lz]) / 2.0
            inside |= np.all(np.abs(local) <= h, axis=1)
        elif isinstance(b.shape, Cylinder):
            radial = np.hypot(local[:, 0], local[:, 1])
            inside |= ((radial <= b.shape.radius)
                       & (np.abs(local[:, 2]) <= b.shape.length / 2.0))
    return inside.reshape(nx, ny, nz, -1).mean(axis=3)


def volume_fraction_for_mass(domain: Domain, target_mass: float,
                             material_density: float) -> float:
    """Volume fraction that yields a given structural mass.

    Setting the fraction directly is a guess about how much of a box the
    structure should fill, which is not a quantity anyone has an opinion about.
    Setting a target mass is: a racing frame is 30 to 50 grams, and the
    optimiser's job is to spend that budget where it buys the most stiffness.
    A first attempt at 28% of a 273 x 273 x 14 mm box came out at 453 g of
    carbon, which is a frame nobody would fly.
    """
    box = domain.n_elements * domain.element ** 3
    return float(np.clip(target_mass / (material_density * box), 0.01, 0.9))


def frame_problem(vehicle, domain_margin: float = 0.012, thickness: float = 0.014,
                  element: float = 0.0035, volume_fraction: float = 0.28,
                  landing_g: float = 10.0, target_mass: float | None = None,
                  material_density: float = 1550.0,
                  pad: float = 0.009, stack_pattern: float = 0.0305) -> Problem:
    """Pose the airframe as a topology problem, from the vehicle itself.

    The design domain is the plan-view box enclosing the rotors, given a
    thickness the structure may occupy. Material is anchored where the payload
    bolts on and loaded where the rotors pull, and the optimiser decides
    everything in between.

    **Mounts are bolt pads, not billets.** An earlier version forced a solid
    block through the full domain thickness under the whole stack footprint,
    39 x 39 x 12 mm. That is 34.3 g of carbon out of a 40 g budget: the
    optimiser was choosing where to put 14% of the mass and the other 86% was
    a mandated brick. It produced exactly the symptoms you would expect and
    none that name the cause -- material stranded in islands, because what was
    left could not span anything; and a meshed part whose facet count and
    volume came out *identical* across runs that had nothing else in common,
    because the block dominated both.

    What a frame actually needs is material at the fasteners. A flight stack
    bolts on a 30.5 mm square of M3, a motor on its own boss, and each of those
    is a pad at one face -- so `keep` is five small pads on one layer each,
    about 3 g, and the remaining 37 g is the optimiser's to spend.

    Loads are the manoeuvre case from STR-01, one arm at full thrust, applied
    simultaneously at all four mounts. That is conservative for a frame: it is
    the symmetric load the plate must carry, and the asymmetric roll case is
    less severe on the centre structure.
    """
    from ..control import G
    from ..spec.vehicle import Rotor

    rotors = [t for t in vehicle.thrusters if isinstance(t, Rotor)]
    if not rotors:
        raise ValueError("no rotors: there is no load path to optimise")

    reach = max(math.hypot(r.position.x, r.position.y) for r in rotors)
    half = reach + domain_margin
    nx = ny = max(8, int(round(2 * half / element)))
    nz = max(3, int(round(thickness / element)))
    domain = Domain(nx, ny, nz, element)
    span = max(1, int(round(pad / element)) // 2)      # pad half-width, elements

    def cell(x: float, y: float) -> tuple[int, int]:
        return (int(np.clip((x + half) / element, 0, nx - 1)),
                int(np.clip((y + half) / element, 0, ny - 1)))

    def pad_cells(x: float, y: float):
        i, j = cell(x, y)
        return (range(max(i - span, 0), min(i + span + 1, nx)),
                range(max(j - span, 0), min(j + span + 1, ny)))

    n_dof = 3 * domain.n_nodes
    load = np.zeros(n_dof)
    keep = np.zeros(domain.shape, dtype=bool)
    fixed: list[int] = []

    mass = vehicle.mass_properties().mass
    per_rotor = max((r.max_thrust() for r in rotors), default=mass * G / len(rotors))
    tip_load = max(per_rotor, mass * G * landing_g / len(rotors))

    # Motor bosses: one pad each at the top face, where the load enters.
    top = nz - 1
    for r in rotors:
        cols, rows = pad_cells(r.position.x, r.position.y)
        cols, rows = list(cols), list(rows)
        for i in cols:
            for j in rows:
                keep[i, j, top] = True
        nodes = [domain.node_id(i, j, nz)
                 for i in cols + [cols[-1] + 1]
                 for j in rows + [rows[-1] + 1]]
        for n in nodes:
            load[3 * n + 2] -= tip_load / len(nodes)   # z is down; thrust is -z

    # Flight stack: four M3 pads on a square pattern, at the bottom face. The
    # payload's weight enters here -- it does not leave here.
    b = stack_pattern / 2.0
    corners = [(-b, -b), (b, -b), (-b, b), (b, b)]
    pad_nodes = []
    for sx, sy in corners:
        cols, rows = pad_cells(sx, sy)
        cols, rows = list(cols), list(rows)
        for i in cols:
            for j in rows:
                keep[i, j, 0] = True
        nodes = [domain.node_id(i, j, 0)
                 for i in cols + [cols[-1] + 1]
                 for j in rows + [rows[-1] + 1]]
        pad_nodes.append(nodes)
        for n in nodes:
            load[3 * n + 2] += tip_load * len(rotors) / (4 * len(nodes))

    # A flying airframe is a free body. Holding all four stack pads rigidly --
    # which is what this did first -- makes each bolt an independent, infinitely
    # stiff ground, and the optimiser correctly answers that question: four
    # separate arms, each running from its motor to its own bolt, never meeting
    # in the middle because nothing requires them to. It looks like the
    # structure fragmented. It did not; the supports were wrong.
    #
    # Thrust up at the motors and the payload's weight down at the stack pads
    # is a self-equilibrated load set, so all the restraint has to do is remove
    # the six rigid-body modes. 3-2-1: one node pinned, a second held against
    # two axes, a third against one. The reactions come out at zero, and the
    # load has to travel from the payload out to the rotors through whatever
    # the optimiser builds -- which is the actual structural problem.
    a, bb, c = (pad_nodes[0][0], pad_nodes[1][0], pad_nodes[2][0])
    fixed += [3 * a, 3 * a + 1, 3 * a + 2,      # x, y, z
              3 * bb + 1, 3 * bb + 2,           # y, z   (offset from a in x)
              3 * c + 2]                        # z      (off the a-b line)

    if target_mass is not None:
        volume_fraction = volume_fraction_for_mass(domain, target_mass,
                                                   material_density)
    return Problem(domain=domain, fixed=np.unique(fixed), load=load,
                   keep=keep, volume_fraction=volume_fraction)


def largest_connected(density: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """Keep only material connected to the main structure.

    SIMP leaves isolated specks: elements that finished above the threshold but
    touch nothing. They are a numerical residue, not structure -- they carry no
    load, because a disconnected island cannot be in a load path -- and they
    cannot be manufactured, since a printer has nothing to build them on and a
    mill has nothing to hold.

    Removing them changes no physics. It is the same argument as the
    sensitivity filter: what the optimiser produced is not automatically
    something that can exist.
    """
    from scipy import ndimage

    solid = density >= threshold
    labels, n = ndimage.label(solid)
    if n <= 1:
        return density
    sizes = ndimage.sum(solid, labels, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    cleaned = density.copy()
    cleaned[(labels != keep) & solid] = 0.0
    return cleaned


def to_mesh(result: Result, domain: Domain, threshold: float = 0.5,
            smooth: bool = True, prune: bool = True):
    """Turn the density field into a watertight surface.

    Marching cubes on the density field, at a threshold chosen so the enclosed
    volume matches the optimiser's volume constraint. Taking the isosurface of
    the *continuous* field rather than the voxel boundary is what produces the
    smooth branching shapes: SIMP's intermediate densities carry sub-element
    information about where the true boundary lies, and a voxel staircase
    throws that away.
    """
    from skimage import measure
    import trimesh

    field = (largest_connected(result.density, threshold) if prune
             else result.density)
    rho = np.pad(field, 1, mode="constant", constant_values=0.0)
    # `descent` puts the normals on the outside of the solid. With the default
    # the mesh comes out inside-out: still watertight, still the right shape,
    # and with a *negative* enclosed volume -- which a slicer reads as a void,
    # so the part either fails to print or prints as its own negative.
    verts, faces, normals, _ = measure.marching_cubes(
        rho, level=threshold, gradient_direction="descent")
    # Undo the pad, scale voxels to metres, and centre on the vehicle origin.
    verts = (verts - 1.0) * domain.element
    verts[:, 0] -= domain.nx * domain.element / 2.0
    verts[:, 1] -= domain.ny * domain.element / 2.0
    verts[:, 2] -= domain.nz * domain.element / 2.0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    # trimesh 5 renamed these; keep working on either.
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
    else:                                                # pragma: no cover
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
    if smooth:
        trimesh.smoothing.filter_taubin(mesh, iterations=8)
    # Belt and braces: a negative volume means the winding is still inverted.
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    return mesh


def structural_mass(result: Result, domain: Domain, density: float) -> float:
    """Mass of the generated structure, from the density field directly."""
    volume = float(result.density.sum()) * domain.element ** 3
    return volume * density
