"""Orientation, as a rotation from body axes to a part's own axes.

Aerospace convention: intrinsic Z-Y-X Euler angles, applied yaw then pitch then
roll, which is what ArduPilot, PX4 and the flight-dynamics literature mean by
those words. Stored as a matrix because that is what the inertia transform and
the thrust-direction rotation both need, and because recovering angles from a
matrix is only needed for display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .units import Vec3


@dataclass(frozen=True)
class Rotation:
    """Rotation from the parent frame into this part's frame.

    ``matrix`` maps a vector expressed in the part's own axes into parent axes,
    so ``R @ v_local == v_parent``. That direction is the one used by both the
    inertia transform (``R I Rᵀ``) and thrust vectors, and getting it backwards
    is the classic silent error -- it only shows up as a sign flip in a moment
    once the whole stack is assembled.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float).reshape(3, 3)
        object.__setattr__(self, "matrix", m)
        if not np.allclose(m @ m.T, np.eye(3), atol=1e-9):
            raise ValueError("rotation matrix is not orthonormal")
        det = float(np.linalg.det(m))
        if not math.isclose(det, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"rotation matrix has determinant {det:.6f}; a reflection "
                f"(det = -1) is not a rotation and would mirror the part"
            )

    @classmethod
    def identity(cls) -> "Rotation":
        return cls(np.eye(3))

    @classmethod
    def from_euler(cls, roll: float = 0.0, pitch: float = 0.0,
                   yaw: float = 0.0) -> "Rotation":
        """Intrinsic Z-Y-X (yaw, then pitch, then roll), radians."""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        return cls(rz @ ry @ rx)

    @classmethod
    def from_axis_angle(cls, axis: Vec3, angle: float) -> "Rotation":
        """Rodrigues' formula. `axis` need not be normalised."""
        k = np.array(axis.unit().as_list())
        kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        return cls(np.eye(3) + math.sin(angle) * kx
                   + (1.0 - math.cos(angle)) * (kx @ kx))

    @classmethod
    def aligning(cls, source: Vec3, target: Vec3) -> "Rotation":
        """Shortest rotation taking `source` onto `target`.

        Used to point a thruster: give it the part's natural axis and the
        direction you want thrust in.
        """
        a = np.array(source.unit().as_list())
        b = np.array(target.unit().as_list())
        c = float(np.dot(a, b))
        if math.isclose(c, 1.0, abs_tol=1e-12):
            return cls.identity()
        if math.isclose(c, -1.0, abs_tol=1e-12):
            # Antiparallel: any perpendicular axis will do, so pick a stable one.
            perp = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            axis = np.cross(a, perp)
            return cls.from_axis_angle(Vec3(*axis), math.pi)
        axis = np.cross(a, b)
        return cls.from_axis_angle(Vec3(*axis), math.acos(max(-1.0, min(1.0, c))))

    def apply(self, v: Vec3) -> Vec3:
        """Take a vector from this part's axes into the parent's."""
        return Vec3(*(self.matrix @ np.array(v.as_list())))

    def inverse(self) -> "Rotation":
        return Rotation(self.matrix.T)

    def compose(self, inner: "Rotation") -> "Rotation":
        """self ∘ inner: apply `inner` first, then self."""
        return Rotation(self.matrix @ inner.matrix)

    def as_axis_angle(self) -> tuple[Vec3, float]:
        """Recover (axis, angle in radians). Axis is arbitrary when angle is 0.

        Preferred over Euler angles when handing the rotation to another
        library, because it has no convention to disagree about: there is no
        intrinsic-versus-extrinsic or ordering question to get wrong.
        """
        m = self.matrix
        cos_a = (float(np.trace(m)) - 1.0) / 2.0
        angle = math.acos(max(-1.0, min(1.0, cos_a)))

        if angle < 1e-12:
            return Vec3(0.0, 0.0, 1.0), 0.0
        if abs(angle - math.pi) < 1e-6:
            # At 180 degrees the skew part vanishes; recover the axis from the
            # symmetric part instead, taking the largest diagonal for accuracy.
            d = np.sqrt(np.maximum((np.diag(m) + 1.0) / 2.0, 0.0))
            k = int(np.argmax(d))
            axis = np.zeros(3)
            axis[k] = d[k]
            for j in range(3):
                if j != k:
                    axis[j] = (m[k, j] + m[j, k]) / (4.0 * d[k])
            return Vec3(*axis).unit(), math.pi

        axis = np.array([m[2, 1] - m[1, 2],
                         m[0, 2] - m[2, 0],
                         m[1, 0] - m[0, 1]]) / (2.0 * math.sin(angle))
        return Vec3(*axis).unit(), angle

    def as_euler(self) -> tuple[float, float, float]:
        """Recover (roll, pitch, yaw). Gimbal lock at |pitch| = 90 deg."""
        m = self.matrix
        pitch = math.asin(max(-1.0, min(1.0, -m[2, 0])))
        if math.isclose(abs(m[2, 0]), 1.0, abs_tol=1e-9):
            return math.atan2(-m[1, 2], m[1, 1]), pitch, 0.0
        return (math.atan2(m[2, 1], m[2, 2]), pitch, math.atan2(m[1, 0], m[0, 0]))
