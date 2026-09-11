"""Camber models in NED; MuJoCo's world is z-up. One matrix converts between.

GEOM-02 fixes the body frame as NED -- x forward, y right, z DOWN -- because
that is what ArduPilot, PX4 and the flight-dynamics literature use. MuJoCo, like
every renderer and ROS (REP-103), is z-up: x forward, y LEFT, z up (FLU).

The map between them is a half-turn about x: (x, y, z) -> (x, -y, -z). It is a
rotation (determinant +1), not a reflection, so it is safe to apply to
positions, forces, moments and inertia tensors alike -- and it is its own
inverse. Negating z alone would be a reflection, and a reflected quadcopter has
its props spinning the wrong way round.
"""

from __future__ import annotations

import numpy as np

NED_TO_FLU = np.diag([1.0, -1.0, -1.0])


def vec(v) -> np.ndarray:
    """A NED vector (or Vec3) in FLU axes."""
    a = np.asarray(v.as_list() if hasattr(v, "as_list") else v, dtype=float)
    return NED_TO_FLU @ a


def tensor(inertia: np.ndarray) -> np.ndarray:
    """A NED inertia tensor in FLU axes: R I R^T."""
    return NED_TO_FLU @ np.asarray(inertia, dtype=float) @ NED_TO_FLU.T


def rotation(matrix: np.ndarray) -> np.ndarray:
    """A part's orientation (local -> NED parent) re-expressed local -> FLU.

    Conjugating by R rather than left-multiplying keeps the part's own local
    axes meaning what they meant: a box's lx is still along its local x.
    """
    return NED_TO_FLU @ np.asarray(matrix, dtype=float) @ NED_TO_FLU


def wrench(allocation: np.ndarray) -> np.ndarray:
    """A 6 x N NED allocation matrix, rows (F, M), in FLU axes."""
    R6 = np.zeros((6, 6))
    R6[:3, :3] = NED_TO_FLU
    R6[3:, 3:] = NED_TO_FLU
    return R6 @ np.asarray(allocation, dtype=float)
