"""Mass, centre of gravity and the inertia tensor of an assembly.

Exact, and computed from solids rather than point masses. The distinction is
not pedantry: a 4S LiPo modelled as a point at its centre understates the
vehicle's roll inertia by a large fraction, and roll inertia is what sets
control authority and agility. A design optimised against a wrong inertia
tensor is optimised for an aircraft that does not exist.

The composition rule is the parallel-axis theorem in its tensor form:

    I_total = SUM_i [ R_i I_i R_i^T + m_i (|d_i|^2 E - d_i d_i^T) ]

where d_i is the offset from the assembly CG to part i's centroid, R_i takes
part i's local axes into body axes, and E is the identity. The first term
rotates each part's own inertia into a common frame; the second accounts for
its displacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .spec.units import Vec3


@dataclass(frozen=True)
class MassProperties:
    """Mass, centre of gravity and inertia about that CG, in body axes."""

    mass: float
    cg: Vec3
    inertia: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "inertia", np.asarray(self.inertia, dtype=float).reshape(3, 3)
        )

    @property
    def principal_moments(self) -> np.ndarray:
        """Eigenvalues of the inertia tensor, ascending."""
        return np.linalg.eigvalsh(self.inertia)

    def inertia_about(self, point: Vec3) -> np.ndarray:
        """Shift the tensor to a different reference point."""
        d = np.array((point - self.cg).as_list())
        return self.inertia + self.mass * (float(d @ d) * np.eye(3) - np.outer(d, d))

    def is_physical(self, tol: float = 1e-12) -> bool:
        """Do the principal moments satisfy the triangle inequalities?

        For any real rigid body I1 + I2 >= I3 for every permutation. A tensor
        that violates this cannot be produced by a positive mass distribution,
        so a failure here means a sign error or a bad transform upstream --
        which is exactly the kind of mistake that otherwise stays invisible
        until a flight model behaves strangely.
        """
        i1, i2, i3 = sorted(self.principal_moments)
        return i1 >= -tol and (i1 + i2) >= i3 - tol

    def __repr__(self) -> str:
        ix, iy, iz = np.diag(self.inertia)
        return (f"MassProperties(mass={self.mass:.4f} kg, "
                f"cg=({self.cg.x:+.4f}, {self.cg.y:+.4f}, {self.cg.z:+.4f}) m, "
                f"diag(I)=({ix:.3e}, {iy:.3e}, {iz:.3e}) kg m^2)")


@dataclass(frozen=True)
class Part:
    """One contribution to an assembly: mass, where it is, how it is turned."""

    mass: float
    centroid: Vec3
    #: Inertia about the part's own centroid, expressed in the part's axes.
    inertia_local: np.ndarray
    #: Maps the part's local axes into body axes.
    rotation: np.ndarray

    def inertia_body_axes(self) -> np.ndarray:
        r = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        return r @ np.asarray(self.inertia_local, dtype=float).reshape(3, 3) @ r.T


def combine(parts: Sequence[Part] | Iterable[Part]) -> MassProperties:
    """Assemble parts into total mass, CG and inertia about that CG."""
    parts = list(parts)
    if not parts:
        return MassProperties(0.0, Vec3(), np.zeros((3, 3)))

    total = sum(p.mass for p in parts)
    if total <= 0.0:
        raise ValueError(
            "assembly has zero or negative total mass; every part needs a mass"
        )

    cg_arr = sum(
        (p.mass * np.array(p.centroid.as_list()) for p in parts),
        start=np.zeros(3),
    ) / total
    cg = Vec3(*cg_arr)

    inertia = np.zeros((3, 3))
    for p in parts:
        d = np.array(p.centroid.as_list()) - cg_arr
        inertia += p.inertia_body_axes()
        inertia += p.mass * (float(d @ d) * np.eye(3) - np.outer(d, d))

    # Symmetrise: the arithmetic is symmetric in exact terms, and floating
    # point drift of 1e-19 off-diagonal makes eigvalsh results look asymmetric
    # in a way that alarms readers of the output without meaning anything.
    inertia = 0.5 * (inertia + inertia.T)
    return MassProperties(total, cg, inertia)
