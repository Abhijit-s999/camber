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

    def node_coordinates(self) -> np.ndarray:
        """(n_nodes, 3) node positions in metres, origin at the domain centre."""
        n = np.arange(self.n_nodes)
        k, rem = np.divmod(n, (self.nx + 1) * (self.ny + 1))
        j, i = np.divmod(rem, self.nx + 1)
        return np.stack([i * self.element - self.nx * self.element / 2.0,
                         j * self.element - self.ny * self.element / 2.0,
                         k * self.element - self.nz * self.element / 2.0],
                        axis=1)

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
class Fastener:
    """A bolt the airframe has to be built around.

    Load does not enter an airframe at a point; it enters through fasteners, on
    a bolt circle, and the material has to reach every one of them. Modelling a
    motor as a single loaded pad gets a load path that runs to the middle of
    the boss and stops -- which is what produced arms tapering to a point with
    the mount floating unattached beyond them.
    """

    x: float
    y: float
    #: Clearance radius of the hole itself. M3 through a printed or milled part
    #: is 1.7 mm, not 1.5: a nominal-diameter hole does not admit the bolt.
    hole: float = 0.0017
    #: Radius of solid material required around it, so there is metal to bear
    #: on and enough edge distance not to tear out.
    boss: float = 0.0045

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "hole": self.hole, "boss": self.boss}


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
    #: With `material_price` unset this is a hard constraint the optimiser must
    #: meet exactly. With a price set the volume is free and this is only where
    #: the field starts.
    volume_fraction: float = 0.3
    #: Exchange rate between stiffness and material, or None for a fixed
    #: volume. See `optimise` -- 1.0 weights them equally, below 1.0 says
    #: material is cheaper than compliance.
    material_price: float | None = None
    #: Compliance the price is quoted against, joules: the strain energy the
    #: structure would store at the largest deflection anyone would accept.
    compliance_scale: float = 1.0
    #: Mass the price is quoted against, kg. The aircraft's own all-up mass,
    #: because a gram of airframe costs the same as a gram of anything else.
    mass_scale: float = 1.0
    #: kg per unit density per element, so mass follows from the field.
    element_mass: float = 1.0
    #: Where the bolts go, so the mesh can be drilled exactly afterwards.
    fasteners: list = field(default_factory=list)


def _structure(rule: str):
    """Neighbourhood for labelling: face-only (6) or face/edge/corner (26)."""
    from scipy import ndimage

    if rule not in ("face", "full"):
        raise ValueError(f"rule must be 'face' or 'full', got {rule!r}")
    return ndimage.generate_binary_structure(3, 1 if rule == "face" else 3)


@dataclass
class Result:
    density: np.ndarray                  # (nx, ny, nz) in [0, 1]
    compliance_history: list[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    #: Penalty the run finished at, which continuation makes worth recording.
    penalty: float = 3.0
    #: Projection sharpness it finished at, likewise.
    beta: float = 0.0

    @property
    def volume_fraction(self) -> float:
        return float(self.density.mean())

    def connectivity(self, threshold: float = 0.4,
                     rule: str = "full") -> tuple[int, float]:
        """(number of solid islands, fraction of material in the largest).

        Reported rather than left to be discovered downstream. A run that
        scatters a third of its volume budget into islands has not failed
        loudly -- compliance still fell, the field is still 96% binary -- and
        the only symptom is that the part you finally mesh weighs a third less
        than you asked for. The first run of this optimiser did exactly that.

        `rule` picks what counts as a join, and the two answers mean different
        things. "full" is 26-connectivity: voxels that share a face, an edge or
        a corner. That is what the finite-element model sees, because elements
        meeting at an edge share two nodes and transmit load through them, so
        it is the right rule for *is this one structure*. "face" is
        6-connectivity, which is the right rule for *could this be made*: a
        real part cannot carry load through a corner.

        On a diagonal arm the two disagree completely. A one-element-wide
        diagonal is a staircase whose voxels touch only at edges, so face
        connectivity reports one island per voxel -- 96 of them on a frame that
        is topologically a single piece. Use `hinges()` for the gap between
        them, which is the number that actually names the defect.
        """
        from scipy import ndimage

        solid = self.density >= threshold
        if not solid.any():
            return 0, 0.0
        labels, n = ndimage.label(solid, structure=_structure(rule))
        if n <= 1:
            return n, 1.0
        mass = ndimage.sum(self.density, labels, range(1, n + 1))
        return n, float(mass.max() / mass.sum())

    def thin_fraction(self, diameter: float, threshold: float = 0.5) -> float:
        """Fraction of the material narrower than `diameter` cells.

        Morphological opening -- erode, then dilate back -- removes exactly the
        solid that a ball of that diameter cannot fit inside, so what the
        opening takes away is the stringy material, measured. It is the number
        to look at instead of the render: a render cannot tell a 1.4 mm web
        from a 1.6 mm one, and the difference is the whole question.

        The ball is built by diameter, not radius. A ball centred on a voxel
        can only have an odd diameter -- 1, 3, 5 cells -- so asking for
        "narrower than 2 cells" that way silently tests "narrower than 3",
        which on a 3 mm feature and a 1.5 mm grid condemned every member that
        met the spec. An even diameter centres the ball on a cell corner.

        Material at the domain faces is judged too: outside is air, not a wall
        a thin skin can lean on.
        """
        from scipy import ndimage

        solid = self.density >= threshold
        if not solid.any():
            return 0.0
        n = max(1, int(round(diameter)))
        c = (n - 1) / 2.0
        g = np.arange(n) - c
        ball = (g[:, None, None] ** 2 + g[None, :, None] ** 2
                + g[None, None, :] ** 2) <= (n / 2.0) ** 2 + 1e-9
        opened = ndimage.binary_opening(solid, structure=ball)
        return float((solid & ~opened).sum() / solid.sum())

    def hinges(self, threshold: float = 0.4) -> int:
        """How many places the structure holds together only at an edge.

        The count of face-connected pieces inside the largest fully-connected
        one, less itself. Zero means every join is a shared face and the part
        can be made. A large number means the optimiser has exploited the
        element formulation: nodes shared at an edge carry load in the discrete
        model that a corner cannot carry in carbon.
        """
        from scipy import ndimage

        solid = self.density >= threshold
        if not solid.any():
            return 0
        full, n = ndimage.label(solid, structure=_structure("full"))
        if n == 0:
            return 0
        sizes = ndimage.sum(solid, full, range(1, n + 1))
        biggest = full == (int(np.argmax(sizes)) + 1)
        _, pieces = ndimage.label(biggest, structure=_structure("face"))
        return max(0, pieces - 1)


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


def project(x: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    """Smoothed Heaviside: push a filtered field back towards solid or void.

    A density filter is what gives a design a length scale -- no member can be
    thinner than the filter reaches -- but it also leaves a grey band at every
    boundary as wide as the filter. Projection puts the edge back: above `eta`
    towards 1, below towards 0, with `beta` setting how sharply. At beta near 0
    it is the identity; at large beta it is a step, and the length scale
    survives because what gets projected is the *filtered* field.
    """
    if beta <= 1e-6:
        return x.copy()
    t = math.tanh(beta * eta)
    return (t + np.tanh(beta * (x - eta))) / (t + math.tanh(beta * (1.0 - eta)))


def project_gradient(x: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    """d(project)/dx, which the chain rule needs and is easy to leave out."""
    if beta <= 1e-6:
        return np.ones_like(x)
    t = math.tanh(beta * eta)
    return (beta * (1.0 - np.tanh(beta * (x - eta)) ** 2)
            / (t + math.tanh(beta * (1.0 - eta))))


def should_sharpen(change: float, since: int, every: int,
                   settled: float = 0.02) -> bool:
    """Whether to double the projection sharpness now.

    On a schedule alone this oscillates. Every doubling of `beta` multiplies
    the projection gradient, so the optimality-criteria step saturates at the
    move limit and the design flips between two states: measured on a test
    cantilever, compliance alternating between 4.09e7 and 1.89e12 on
    consecutive iterations, forever. Sharpening only once the field has settled
    -- or after `every` iterations if it never does -- lets each level of
    sharpness be absorbed before the next arrives.

    This is the same lesson as the penalty ramp in GEN-03, and the same lesson
    as GEN-04's stopping rule: a schedule that ignores what the run is actually
    doing will happily drive it somewhere useless.
    """
    return change < settled or since >= every


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
        order = np.argsort(keys, kind="stable").astype(np.int64)
        sel, keys = sel[order], keys[order]
        del order
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

        # Per-entry element index and *which* entry of KE it carries. Storing
        # the index rather than the value matters at fine grids: 265,000
        # elements is 152 million entries, and int16 indices into a 576-entry
        # matrix cost 300 MB where float64 values cost 1.2 GB.
        self.element = (sel // 576).astype(np.int32)
        self.ke_index = (sel % 576).astype(np.int16)
        self.ke_flat = KE.ravel()

    def matrix(self, stiffness: np.ndarray) -> sp.csc_matrix:
        vals = np.take(self.ke_flat, self.ke_index)
        vals *= np.take(stiffness, self.element)
        data = np.add.reduceat(vals, self.starts)
        return sp.csc_matrix((data, self.indices, self.indptr), shape=self.shape)


#: Above this many unknowns the direct factorisation's fill-in costs more than
#: multigrid setup does. Measured, not guessed: see `_solve`.
AMG_ABOVE = 150_000


def _pick_solver(requested: str, n_free: int) -> str:
    if requested in ("direct", "amg"):
        return requested
    if requested != "auto":
        raise ValueError(f"solver must be auto, direct or amg, got {requested!r}")
    if n_free < AMG_ABOVE:
        return "direct"
    try:
        import pyamg                                        # noqa: F401
    except ImportError:
        return "direct"
    return "amg"


def rigid_body_modes(domain: Domain, free: np.ndarray) -> np.ndarray:
    """The six zero-energy modes of an unconstrained solid, at the free DOFs.

    Algebraic multigrid on elasticity is only as good as what it is told about
    the near-nullspace. Aggregation coarsens by grouping strongly-connected
    unknowns, and if it does not know that a rigid rotation costs no energy it
    builds coarse spaces that cannot represent one -- so the smoother has to
    resolve bending on the finest grid and convergence collapses to something
    barely better than plain CG. Handing over the three translations and three
    rotations is what makes the method work here.
    """
    xyz = domain.node_coordinates()
    n = xyz.shape[0]
    B = np.zeros((3 * n, 6))
    for a in range(3):
        B[a::3, a] = 1.0                                  # translations
    X, Y, Z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    B[0::3, 3], B[1::3, 3] = -Y, X                        # about z
    B[1::3, 4], B[2::3, 4] = -Z, Y                        # about x
    B[0::3, 5], B[2::3, 5] = Z, -X                        # about y
    return B[free]


def _solve(K, rhs, domain, free, mode, tol=1e-8, previous=None):
    """Solve K u = f, directly or by algebraic multigrid.

    The direct factorisation is exact and its fill-in is brutal: measured on
    this problem, 45 s at 2.5 mm elements and 376 s at 2.0 mm, for 1.8 times
    the unknowns. That curve is what puts fine grids out of reach, not the
    physics -- so above a few hundred thousand degrees of freedom the solve
    switches to conjugate gradients preconditioned by smoothed-aggregation
    multigrid, which is linear in the unknowns instead.
    """
    if mode == "direct":
        return splu(K, permc_spec="MMD_AT_PLUS_A",
                    options={"SymmetricMode": True}).solve(rhs)
    import pyamg

    ml = pyamg.smoothed_aggregation_solver(
        K.tocsr(), B=rigid_body_modes(domain, free), smooth="energy",
        strength=("symmetric", {"theta": 0.0}), max_coarse=500)
    # Warm-started from the previous iteration's displacement: consecutive SIMP
    # designs differ by at most the move limit, so the old field is a good
    # guess and typically saves a third of the cycles.
    return ml.solve(rhs, x0=previous, tol=tol, accel="cg", maxiter=200)


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
             penalty_start: float = 1.0,
             beta: float = 16.0, eta: float = 0.5, robust: float = 0.2,
             outside_void: bool = True,
             solver: str = "auto", on_iteration=None) -> Result:
    """Run SIMP until the density field stops moving.

    ``penalty_start`` below ``penalty`` enables continuation; set the two equal
    to go straight to the final penalty.

    If ``problem.material_price`` is set the volume constraint is dropped and
    material is charged for instead, so the mass is chosen rather than given.

    ``robust`` is the erosion margin: the analysis runs on the design projected
    at ``eta + robust``, so a member has to survive being thinned before it
    counts as structure. 0 turns it off. See `physical` for why.
    """
    d = problem.domain
    # Unit modulus, real element size. Two scale errors lived here and both
    # were invisible while the solve ran in arbitrary units and the volume was
    # constrained, because both are *global* factors on compliance and the
    # optimal shape does not depend on them. Passing E0 to `hex8_stiffness` as
    # well as to the SIMP interpolation applies the modulus twice, so stiffness
    # went as E0 squared -- the identity at E0 = 1. And leaving `size` at its
    # 1.0 default builds the matrix for a one-metre cube: at 5 mm elements that
    # is a factor of 200, which showed up as a cantilever 205 times stiffer
    # than wL^3/3EI.
    KE = hex8_stiffness(1.0, nu, size=d.element)
    dofs = d.element_dofs()
    n_dof = 3 * d.n_nodes

    H = _filter_matrix(d, filter_radius)
    if outside_void:
        # Normalise by the weight of the *whole* stencil, as if the domain
        # carried on, so neighbours outside it count as void. The usual
        # normalisation -- by the weights of the neighbours that exist -- is a
        # mirror boundary: an element on the top face sees its own layer
        # reflected above it, and a one-layer skin reads to the filter as
        # twice as thick as it is. Erosion then never bites at the faces,
        # which is exactly where a plate in bending puts its skins, and the
        # stringy one-element flanges survive every length-scale control.
        # The faces of this domain are the surfaces of the airframe; outside
        # them is air.
        r = int(math.ceil(filter_radius)) - 1
        g = np.arange(-r, r + 1)
        dist = np.sqrt(g[:, None, None] ** 2 + g[None, :, None] ** 2
                       + g[None, None, :] ** 2)
        Hs = np.full(d.n_elements, float(np.clip(filter_radius - dist, 0, None).sum()))
    else:
        Hs = np.asarray(H.sum(axis=1)).ravel()

    x = np.full(d.n_elements, problem.volume_fraction)
    keep_i = problem.keep.ravel() if problem.keep is not None else None
    void_i = problem.void.ravel() if problem.void is not None else None

    def physical(design: np.ndarray, sharpness: float,
                 level: float = eta) -> np.ndarray:
        """Design variables -> the density the structure actually has.

        **The filter was half-applied.** Sensitivities were transformed by
        `H @ (dc / Hs)`, which is the chain rule *for a density filter*, while
        the density field itself was never filtered. The optimiser was told
        about a smoothing that was not happening: it could still place a
        one-element web, and the sensitivity it read for doing so was the one
        it would have had if webs were smoothed. That is where the stringy
        single-element structure came from.

        `level` is the projection threshold. At `eta` this is the blueprint,
        the part that gets built. At `eta + robust` it is the same design
        *eroded*: every boundary pulled inward, so anything thinner than the
        erosion disappears. That second one is what the finite elements see.
        """
        rho = (H @ design) / Hs
        rho = project(rho, sharpness, level)
        if keep_i is not None:
            rho[keep_i] = 1.0
        if void_i is not None:
            rho[void_i] = 0.0
        return rho

    if keep_i is not None:
        x[keep_i] = 1.0
    if void_i is not None:
        x[void_i] = 0.0

    asm = _Assembly(dofs, problem.fixed, n_dof, KE)
    free = asm.free
    f_free = problem.load[free]
    how = _pick_solver(solver, free.size)
    warm = None
    if verbose:
        print(f"  solver: {how} ({free.size:,} free dof)", flush=True)

    result = Result(density=x.reshape(d.shape))
    previous = x.copy()
    c_ref = None                       # set from the first solve; see below

    p_final = penalty
    sharp, since_sharpen = 1.0, 0
    sharpen_every = max(15, iterations // 6)
    for it in range(iterations):
        penalty = penalty_at(it, iterations, p_final, penalty_start)
        xf = (H @ x) / Hs                    # filtered
        # Minimum thickness, enforced rather than hoped for. Projecting at
        # eta = 0.5 does not guarantee a length scale: a member thinner than
        # the filter can still project to solid, and on the racing frame some
        # did -- stringy webs survived GEN-14. So the stiffness is computed on
        # the *eroded* design. A web too thin to survive erosion carries no
        # load there, its sensitivity says so, and the optimiser either thickens
        # it or deletes it. For compliance the eroded design is always the
        # weakest of the eroded/blueprint/dilated set, so this is the full
        # robust formulation at the cost of one solve, not three.
        level_fe = min(eta + robust, 0.95)
        xp = physical(x, sharp, level_fe)    # eroded: what is analysed
        stiff = Emin + xp ** penalty * (E0 - Emin)
        u = np.zeros(n_dof)
        try:
            u[free] = _solve(asm.matrix(stiff), f_free, d, free, how,
                             previous=warm)
            warm = u[free].copy()
        except Exception:                                   # noqa: BLE001
            break

        ue = u[dofs]
        ce = np.einsum("ij,jk,ik->i", ue, KE, ue)
        compliance = float((stiff * ce).sum())
        result.compliance_history.append(compliance)
        # Anchor the exchange rate to the starting design, once. Re-normalising
        # every iteration would move the target the optimiser is walking
        # towards, and the run would chase its own tail.
        if c_ref is None:
            c_ref = compliance if compliance > 0 else 1.0

        # Chain rule all the way back to the design variables: through the
        # projection, then through the filter. Dropping the projection term is
        # the classic way to get a run that oscillates and never settles.
        # Stiffness is the eroded design's; material is the blueprint's, because
        # that is what gets built and weighed. Each chains through its own
        # projection threshold.
        dc = -penalty * xp ** (penalty - 1.0) * (E0 - Emin) * ce
        dc = H @ (dc * project_gradient(xf, sharp, level_fe) / Hs)
        dv = H @ (project_gradient(xf, sharp, eta) / Hs)

        # Every doubling of `beta` multiplies the projection gradient, so a
        # fixed move limit lets the optimality-criteria step saturate and the
        # design flips between two states. Shrinking the step as the projection
        # sharpens keeps each iteration inside the range where the linearised
        # sensitivity is still a good guide.
        move_now = move / math.sqrt(sharp)

        def bounded(step: np.ndarray) -> np.ndarray:
            step = np.clip(np.clip(step, x - move_now, x + move_now), 0.0, 1.0)
            if problem.keep is not None:
                step[problem.keep.ravel()] = 1.0
            if problem.void is not None:
                step[problem.void.ravel()] = 0.0
            return step

        if problem.material_price is None:
            # Optimality criteria: bisect the Lagrange multiplier until the
            # volume constraint is met exactly.
            lo, hi = 1e-9, 1e9
            target = problem.volume_fraction * d.n_elements
            xnew = x
            while (hi - lo) / (lo + hi) > 1e-6:
                mid = 0.5 * (lo + hi)
                xnew = bounded(x * np.sqrt(np.maximum(-dc / (dv * mid), 0.0)))
                # The constraint is on the material that exists, not on the
                # design variables that generate it.
                if physical(xnew, sharp).sum() > target:
                    lo = mid
                else:
                    hi = mid
        else:
            # Priced material. The multiplier is not solved for -- it *is* the
            # price -- so the mass is an outcome rather than an input:
            #
            #   J = c / c_scale  +  w * m / m_scale
            #
            # Both terms are dimensionless and, crucially, both are O(1) at a
            # design anyone would build, because the scales are physical: the
            # strain energy at the largest deflection the airframe may have,
            # and the mass of the aircraft it has to lift. That is what makes
            # w readable. An earlier version normalised by the box volume and
            # the starting compliance, which is dimensionally fine and
            # practically useless -- with the solve running at E0 = 1 the
            # compliance was in no unit at all, and w = 4 produced a 591 g
            # frame while w = 0.5 produced 1151 g. Numbers with no scale cannot
            # be quoted to anyone.
            #
            # Stationarity for an interior element is
            #     (-dc_e) / c_scale == w * element_mass / m_scale
            # so the ratio below is 1 at the optimum, and the square-root
            # damping is the same as the constrained update's.
            cost = (problem.material_price * problem.element_mass
                    / problem.mass_scale)
            ratio = (-dc / problem.compliance_scale) / (cost * dv)
            xnew = bounded(x * np.sqrt(np.maximum(ratio, 0.0)))

        change = float(np.abs(xnew - previous).max())
        previous, x = xnew.copy(), xnew

        since_sharpen += 1
        if sharp < beta and should_sharpen(change, since_sharpen,
                                           sharpen_every):
            sharp, since_sharpen = min(beta, sharp * 2.0), 0
        # The physical field is what was analysed and what gets meshed; the
        # design variables are an intermediate nobody should see.
        result.density = physical(x, sharp).reshape(d.shape)
        result.iterations = it + 1
        result.penalty = penalty
        result.beta = sharp
        if verbose:
            print(f"  it {it+1:3d}  p {penalty:4.2f}  b {sharp:5.1f}  "
                  f"compliance {compliance:10.4e}  "
                  f"vol {result.density.mean():.3f}  change {change:.4f}",
                  flush=True)
        if on_iteration:
            on_iteration(it, compliance, result.density)
        # Only meaningful once the ramp is done: each penalty step moves the
        # field deliberately, so "it stopped moving" mid-ramp says nothing.
        if (change < tol and penalty >= p_final and sharp >= beta
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
    KE = hex8_stiffness(1.0, nu, size=d.element)   # see optimise()
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
                  stack_pattern: float = 0.0305, motor_pattern: float = 0.016,
                  bolt_hole: float = 0.0017, boss: float = 0.0045,
                  material_price: float | None = None,
                  max_deflection: float = 0.001) -> Problem:
    """Pose the airframe as a topology problem, from the vehicle itself.

    The design domain is the plan-view box enclosing the rotors, given a
    thickness the structure may occupy. The optimiser decides everything except
    the fasteners, and the fasteners are the whole interface: a motor bolts to
    a 16 mm square of M3, a flight stack to a 30.5 mm square of M3, and the
    airframe's job is to carry load between those two bolt circles.

    **Mounts are bolt circles, not pads.** Each bolt contributes a solid boss --
    material to bear on, with edge distance -- and a voided hole through it, so
    the optimiser is not asked to discover that a bolt needs a hole. Load enters
    at the bosses, four per motor, which is physically where thrust reaches the
    frame. An earlier version applied thrust at a single 9 mm pad per motor and
    got exactly what it asked for: arms that taper to a point at the middle of
    the mount, with the mount itself a separate floating body.

    **Mounts are not billets either.** The version before that forced a solid
    block through the full domain thickness under the whole stack footprint,
    39 x 39 x 12 mm: 34.3 g of a 40 g budget, leaving the optimiser 14% of the
    mass to place.

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
    nx = ny = max(8, int(round(2 * (reach + domain_margin) / element)))
    nz = max(3, int(round(thickness / element)))
    domain = Domain(nx, ny, nz, element)

    # The mapping has to use the grid that exists, not the half-span that chose
    # it. `nx = round(2 * half / element)` moves the edge by up to half an
    # element, and mapping with the pre-rounding half-span puts the vehicle
    # origin off the grid centre -- on the racing frame by 1.05 mm, enough that
    # one diagonal came out a cell longer than the other and the optimiser made
    # visibly different choices on the two.
    cx = (np.arange(nx) + 0.5) * element - nx * element / 2.0
    cy = (np.arange(ny) + 0.5) * element - ny * element / 2.0
    gx, gy = np.meshgrid(cx, cy, indexing="ij")

    def disc(at_x: float, at_y: float, radius: float) -> np.ndarray:
        """Plan cells whose centres fall inside a circle.

        By distance, so it is mirror-symmetric by construction and the feature
        is the size asked for. Counting elements out from a centre cell can
        only make odd multiples of the element, which turned a 9 mm pad into
        16.5 mm on a 5.5 mm grid.
        """
        inside = np.hypot(gx - at_x, gy - at_y) <= radius + 1e-12
        if not inside.any():                     # feature finer than the grid
            d2 = np.hypot(gx - at_x, gy - at_y)
            inside = d2 <= d2.min() + 1e-12
        return inside

    n_dof = 3 * domain.n_nodes
    load = np.zeros(n_dof)
    keep = np.zeros(domain.shape, dtype=bool)
    void = np.zeros(domain.shape, dtype=bool)
    fasteners: list[Fastener] = []

    mass = vehicle.mass_properties().mass
    per_rotor = max((r.max_thrust() for r in rotors), default=mass * G / len(rotors))
    tip_load = max(per_rotor, mass * G * landing_g / len(rotors))

    def mount(at_x, at_y, layer, force_z, hold=False):
        """One bolt: a solid boss, a hole through it, and its share of load."""
        fasteners.append(Fastener(at_x, at_y, bolt_hole, boss))
        ring = disc(at_x, at_y, boss)
        hole = disc(at_x, at_y, bolt_hole)
        solid = ring & ~hole
        if not solid.any():
            # The grid cannot separate a 9 mm boss from a 3.4 mm hole, so
            # voiding the hole would leave the mount with no material at all
            # and nowhere to apply load. Keep the boss whole and let the hole
            # come from `drill`, which is exact and does not care about the
            # grid. The void mask is a hint to the optimiser, not the geometry.
            solid, hole = ring, np.zeros_like(hole)
        keep[solid, layer] = True
        void[hole, :] = True                     # the hole goes all the way
        cols, rows = np.where(solid)
        nodes = sorted({domain.node_id(i + di, j + dj, layer + dk)
                        for i, j in zip(cols, rows)
                        for di in (0, 1) for dj in (0, 1) for dk in (0, 1)})
        for n in nodes:
            load[3 * n + 2] += force_z / len(nodes)
        return nodes

    # Which face is which, from the vehicle rather than from a comment. Layer
    # 0 is the most negative z, and in NED (GEOM-02) negative z is UP -- so
    # layer 0 is the top face. The motor bosses belong on the face the rotors
    # are on, because that is where each motor's base bears. This used to say
    # "top face" and use layer nz-1, which is the *bottom*: the reinforcement
    # sat on the side facing the ground and the face the motor actually bolts
    # to had no material guaranteed at all. The renders looked right only
    # because the STL was written in NED and viewed Z-up, which flipped the
    # frame over -- one convention error hiding another.
    rotor_side = 0 if np.mean([r.position.z for r in rotors]) < 0 else nz - 1

    # Motor bosses: four bolts each, on the rotor face, where thrust enters.
    half_m = motor_pattern / 2.0
    for r in rotors:
        for sx in (-half_m, half_m):
            for sy in (-half_m, half_m):
                mount(r.position.x + sx, r.position.y + sy, rotor_side,
                      -tip_load / 4.0)           # z is down; thrust is -z

    # Flight stack: four bolts on the same face. A racing stack sits on the
    # plate on standoffs, on the prop side, so its weight bears on that face.
    # It enters here and does not leave -- see the restraint below.
    half_s = stack_pattern / 2.0
    stack_nodes = []
    for sx in (-half_s, half_s):
        for sy in (-half_s, half_s):
            stack_nodes.append(
                mount(sx, sy, rotor_side, tip_load * len(rotors) / 4.0))

    # A flying airframe is a free body. Holding all four stack bolts rigidly --
    # which is what this did first -- makes each an independent, infinitely
    # stiff ground, and the optimiser correctly answers that question: four
    # separate arms, each running from its motor to its own bolt, never meeting
    # in the middle because nothing requires them to. It looks like the
    # structure fragmented. It did not; the supports were wrong.
    #
    # Thrust up at the motors and the payload's weight down at the stack is a
    # self-equilibrated load set, so all the restraint has to do is remove the
    # six rigid-body modes. 3-2-1: one node pinned, a second held against two
    # axes, a third against one. The reactions come out at zero, and the load
    # has to travel from the payload out to the rotors through whatever the
    # optimiser builds -- which is the actual structural problem.
    a, b, c = stack_nodes[0][0], stack_nodes[2][0], stack_nodes[1][0]
    fixed = [3 * a, 3 * a + 1, 3 * a + 2,        # x, y, z
             3 * b + 1, 3 * b + 2,               # y, z   (offset from a in x)
             3 * c + 2]                          # z      (off the a-b line)

    if target_mass is not None:
        volume_fraction = volume_fraction_for_mass(domain, target_mass,
                                                   material_density)
    # Compliance is u.f, so the strain energy at the deflection limit is the
    # applied load times that deflection -- the point at which the airframe is
    # as floppy as anyone would accept. Material is priced against the mass of
    # the aircraft, since a gram of frame costs what any other gram costs.
    applied = float(np.abs(load).sum()) / 2.0        # halved: it is a couple
    return Problem(domain=domain, fixed=np.unique(fixed), load=load,
                   keep=keep, void=void, volume_fraction=volume_fraction,
                   material_price=material_price,
                   compliance_scale=applied * max_deflection,
                   mass_scale=mass,
                   element_mass=element ** 3 * material_density,
                   fasteners=fasteners)


def largest_connected(density: np.ndarray, threshold: float = 0.4,
                      rule: str = "full") -> np.ndarray:
    """Keep only material connected to the main structure.

    SIMP leaves isolated specks: elements that finished above the threshold but
    touch nothing. They are a numerical residue, not structure -- they carry no
    load, because a disconnected island cannot be in a load path -- and they
    cannot be manufactured, since a printer has nothing to build them on and a
    mill has nothing to hold.

    **This used face-only connectivity and was deleting real structure.** An
    airframe's arms run diagonally, and a diagonal in a Cartesian grid is a
    staircase whose voxels touch at their edges rather than their faces. Under
    6-connectivity every voxel of such an arm is its own island, so pruning
    removed the arms and kept whatever happened to be blockiest -- which is why
    the meshed part kept coming out at a fraction of the mass budget, 1.3 g of
    a requested 40 g in the worst case. The default is now 26-connectivity,
    which is what the element formulation itself uses.

    That does not make an edge-only join manufacturable; it makes it a
    different problem, counted by `Result.hinges()` and reported separately.
    Meshing bridges the staircase on its own, because marching cubes takes the
    isosurface of the continuous field rather than the voxel boundary.
    """
    from scipy import ndimage

    solid = density >= threshold
    labels, n = ndimage.label(solid, structure=_structure(rule))
    if n <= 1:
        return density
    sizes = ndimage.sum(solid, labels, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    cleaned = density.copy()
    cleaned[(labels != keep) & solid] = 0.0
    return cleaned


def _isolevel_for_volume(field: np.ndarray, target_cells: float,
                         lo: float = 0.05, hi: float = 0.95) -> float:
    """Isolevel enclosing a given number of cells, by bisection.

    Meshing at a fixed 0.5 is only volume-preserving if the field is symmetric
    about it, and a resampled, blurred field is not: blurring moves material
    outward across the boundary, and a fixed level then returns a part heavier
    than the one the optimiser designed. Choosing the level to hit the design
    volume keeps the mass honest, which is the whole point of pricing it.
    """
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float((field >= mid).sum()) > target_cells:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def to_mesh(result: Result, domain: Domain, threshold: float = 0.5,
            smooth: bool = True, prune: bool = True,
            upsample: int = 2, blur: float = 0.5, taubin: int = 40):
    """Turn the density field into a watertight surface.

    Marching cubes takes the isosurface of a *continuous* field, so in
    principle the boundary carries sub-element information and the result is
    smooth. In practice SIMP is built to destroy exactly that: the penalty
    exists to drive every element to solid or void, and a field that is 99.3%
    binary has no intermediate values left to interpolate. The isosurface then
    has nowhere to go but along the voxel boundary, and the part comes out
    faceted at the element scale -- visibly so, in a way that reads as a
    rendering artefact and is really the geometry.

    Three steps, in this order, fix it without touching the optimisation:

    1. **Resample.** Cubic-spline zoom by `upsample` puts the isosurface on a
       finer lattice than the elements. This does not invent detail; it stops
       the mesh quantising the detail that is there.
    2. **Blur.** A small Gaussian restores the intermediate band the penalty
       removed. `blur` is in upsampled cells, so it is a fixed fraction of an
       element whatever the grid.
    3. **Re-level.** Blurring moves material outward across the boundary, so a
       fixed 0.5 would return a heavier part than was designed. The level is
       chosen instead to enclose the design volume.

    `blur` is deliberately small. Holding the volume fixed means a stronger
    blur has to be paid for with a higher isolevel, and a higher level cuts the
    thinnest necks: at 0.9 this part went from one solid body to nine, purely
    from meshing. 0.5 rounds the staircase and keeps the topology.

    Taubin smoothing then runs on the mesh itself. It is used rather than
    Laplacian because Laplacian shrinks a part toward its centroid a little
    with every pass, and this is a part whose mass is the answer -- measured
    here, 80 passes of Taubin cost 1% of volume while taking 2% off the surface
    area, so the passes are close to free and the default is generous.

    Set `upsample=1, blur=0` for the raw voxel isosurface.
    """
    from scipy import ndimage
    from skimage import measure
    import trimesh

    field = (largest_connected(result.density, threshold) if prune
             else result.density)
    target = float(field.sum())                  # design volume, in elements
    # The margin has to outlast the blur. One element of padding is enough for
    # a raw isosurface and not for a blurred one: the Gaussian pushes material
    # out past the boundary, the surface reaches the edge of the array, and
    # marching cubes returns an open sheet -- which then reports a negative
    # volume and is not watertight, so the inversion guard below cannot even
    # tell it is inside-out. Three sigma of margin, in elements, plus one.
    margin = 1 + int(math.ceil(3.0 * blur / max(upsample, 1)))
    rho = np.pad(field, margin, mode="constant", constant_values=0.0)

    if upsample > 1:
        rho = ndimage.zoom(rho, upsample, order=3, mode="nearest")
        rho = np.clip(rho, 0.0, 1.0)
    if blur > 0:
        rho = ndimage.gaussian_filter(rho, blur)

    scale = upsample ** 3
    level = (_isolevel_for_volume(rho, target * scale)
             if (upsample > 1 or blur > 0) else threshold)

    # `descent` puts the normals on the outside of the solid. With the default
    # the mesh comes out inside-out: still watertight, still the right shape,
    # and with a *negative* enclosed volume -- which a slicer reads as a void,
    # so the part either fails to print or prints as its own negative.
    verts, faces, _, _ = measure.marching_cubes(
        rho, level=level, gradient_direction="descent")

    # Built and cleaned in *voxel* coordinates, scaled to metres afterwards.
    # `merge_vertices` welds by rounding to an absolute tolerance, so at metre
    # scale a 1.5 mm grid puts distinct vertices close enough together to be
    # collapsed into each other -- which makes non-manifold edges out of a
    # surface that was closed when it arrived. In voxel units neighbouring
    # vertices are order 1 apart and only genuine duplicates weld.
    #
    # No `vertex_normals=` and no processing on construction either: handing
    # trimesh per-vertex normals makes its merge treat coincident vertices with
    # different normals as distinct, so the merge silently does nothing.
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Marching cubes emits a closed surface with duplicated vertices along
    # every shared edge. Welding them is what makes it a solid rather than a
    # pile of triangles, and it has to come *before* degenerate faces are
    # dropped: a zero-area triangle between unwelded vertices is still holding
    # two patches together, and deleting it punctures the surface.
    mesh.merge_vertices()
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
    else:                                                # pragma: no cover
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)

    # Undo the upsample and the pad, scale to metres, centre on the origin.
    mesh.apply_scale(domain.element / upsample)
    mesh.apply_translation([
        -(margin + domain.nx / 2.0) * domain.element,
        -(margin + domain.ny / 2.0) * domain.element,
        -(margin + domain.nz / 2.0) * domain.element])

    if smooth and taubin:
        trimesh.smoothing.filter_taubin(mesh, iterations=taubin)
    # Belt and braces: a negative volume means the winding is still inverted.
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def drill(mesh, fasteners, clearance: float | None = None):
    """Subtract exact bolt holes from a meshed structure.

    The optimiser works on a voxel grid and a marching-cubes isosurface of it,
    so a hole it carves is only ever as round as the element is small: an M3
    clearance hole is 3.4 mm, which at a 2 mm element is under two cells
    across. Asking the grid for fastener tolerance is asking the wrong part of
    the pipeline. The grid's job is the load path; the hole is a boolean
    subtraction at full precision afterwards, and it lands exactly on the bolt
    circle whatever the mesh did.

    Holes run right through, so this works whichever face the boss is on.
    """
    import trimesh

    if not fasteners:
        return mesh
    lo, hi = mesh.bounds[0][2], mesh.bounds[1][2]
    height = (hi - lo) * 2.0 + 1e-3
    cutters = []
    for f in fasteners:
        r = f.hole if clearance is None else clearance
        c = trimesh.creation.cylinder(radius=r, height=height, sections=48)
        c.apply_translation([f.x, f.y, (lo + hi) / 2.0])
        cutters.append(c)
    # The holes are disjoint, so one concatenated cutter is a valid solid and
    # costs one boolean instead of twenty.
    out = mesh.difference(trimesh.util.concatenate(cutters))

    # The boolean leaves slivers where a cutter grazes a surface: on this frame
    # eight sixteen-face shells of 1e-6 mm^3 each, next to a body of 34,000.
    # They are not structure and they are not nothing -- a slicer lists them as
    # separate objects and will happily try to print them.
    parts = out.split(only_watertight=False)
    if len(parts) > 1:
        keep = [q for q in parts if abs(q.volume) > 1e-6 * abs(out.volume)]
        if len(keep) == 1:
            return keep[0]
        if keep:
            return trimesh.util.concatenate(keep)
    return out


def ned_to_z_up(mesh):
    """Rotate a mesh from the NED body frame into Z-up, for manufacturing.

    Camber models in NED, z down (GEOM-02). Slicers, CAM and every CAD package
    are Z-up, so a NED file opens upside down -- which is survivable for a
    symmetric part and exactly wrong for one with a motor face. The conversion
    is a half-turn about x, not a sign flip on z: a flip is a reflection, and a
    reflected airframe is its own mirror image, which prints as a frame whose
    props spin the wrong way round the body.

    Only manufacturing exports convert. The CFD case is built in NED, with the
    disc momentum source pointing +z (CFD-06), and must be handed NED.
    """
    import trimesh

    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        np.pi, [1.0, 0.0, 0.0]))
    return mesh


def structural_mass(result: Result, domain: Domain, density: float) -> float:
    """Mass of the generated structure, from the density field directly."""
    volume = float(result.density.sum()) * domain.element ** 3
    return volume * density
