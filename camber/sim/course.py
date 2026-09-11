"""Race gates: where the vehicle has to go, and whether it went."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Gate:
    """A square gate, standard 1.5 m inner opening, facing along `normal`."""

    centre: np.ndarray
    normal: np.ndarray
    size: float = 1.5

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=float)
        n = np.asarray(self.normal, dtype=float)
        self.normal = n / np.linalg.norm(n)
        self._side = None

    def crossed(self, p: np.ndarray) -> bool:
        """True the first step the vehicle is past the gate plane, inside it."""
        d = float(np.dot(np.asarray(p) - self.centre, self.normal))
        inside = np.linalg.norm((np.asarray(p) - self.centre)
                                - d * self.normal) < self.size / 2
        return d > 0.0 and inside

    def mjcf(self, i: int) -> str:
        n = self.normal
        yaw = np.arctan2(n[1], n[0])
        h, t = self.size / 2 + 0.05, 0.05
        c = self.centre
        q = f"{np.cos(yaw / 2):.6f} 0 0 {np.sin(yaw / 2):.6f}"
        bars = [(0, h, 0, t, t, h), (0, -h, 0, t, t, h),
                (0, 0, h, t, h, t), (0, 0, -h, t, h, t)]
        out = [f'<body name="gate{i}" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" quat="{q}">']
        for k, (x, y, z, sx, sy, sz) in enumerate(bars):
            out.append(f'  <geom name="gate{i}_{k}" type="box" pos="{x} {y} {z}" '
                       f'size="{sx} {sy} {sz}" rgba="1 0.45 0.1 1" '
                       f'contype="0" conaffinity="0"/>')
        pole = c[2] - h
        out.append(f'  <geom name="gate{i}_pole" type="box" pos="0 0 {-(h + pole / 2):.4f}" '
                   f'size="0.03 0.03 {max(pole / 2, 0.01):.4f}" rgba="0.3 0.3 0.32 1" '
                   f'contype="0" conaffinity="0"/>')
        out.append("</body>")
        return "\n".join(out)


@dataclass
class Course:
    gates: list = field(default_factory=list)
    finish: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def mjcf(self) -> str:
        return "\n".join(g.mjcf(i) for i, g in enumerate(self.gates))

    @classmethod
    def oval(cls, length: float = 24.0, width: float = 12.0,
             height: float = 1.8) -> "Course":
        """Eight gates round an oval: two straights and two hairpins."""
        a, b = length / 2, width / 2
        gates = []
        for k in range(8):
            th = 2 * np.pi * k / 8
            p = np.array([a * np.cos(th), b * np.sin(th), height])
            tangent = np.array([-a * np.sin(th), b * np.cos(th), 0.0])
            gates.append(Gate(p, tangent))
        return cls(gates, finish=np.array([a, -2.0, 1.0]))
