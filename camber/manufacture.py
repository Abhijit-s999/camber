"""Can this actually be made, by the process the user chose?

MFG-01: a pluggable `Process`, 3-axis CNC first. The order matters. Three-axis
milling is the most geometrically restrictive of the common processes -- it
cannot cut an undercut at all, because every surface must be reachable straight
down from one of the setups -- so a design that satisfies it satisfies the
additive processes almost for free. Built the other way round, half the CNC
designs come out unmakeable.

These checks run *inside* the optimiser, not after it. An unconstrained search
will converge on a beautiful shape nobody can produce, and discovering that at
the end means discarding the answer rather than steering away from it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Violation:
    rule: str
    detail: str
    #: How far outside the rule, in metres or radians. Continuous, so the
    #: optimiser can walk back rather than seeing a wall.
    amount: float


@dataclass
class Feature:
    """A geometric feature to be checked, described in process-neutral terms."""

    name: str
    kind: str                   # "pocket" | "wall" | "hole" | "corner" | "overhang"
    size: float = 0.0           # characteristic width or diameter, m
    depth: float = 0.0          # m
    radius: float = 0.0         # internal corner radius, m
    #: Angle from the build/tool axis, radians. 0 is aligned with the axis.
    angle: float = 0.0
    undercut: bool = False


class Process(ABC):
    """One manufacturing route, and what it refuses to make."""

    name: str

    @abstractmethod
    def check(self, feature: Feature) -> list[Violation]:
        ...

    def check_all(self, features: list[Feature]) -> list[Violation]:
        out: list[Violation] = []
        for f in features:
            out.extend(self.check(f))
        return out

    def violation_total(self, features: list[Feature]) -> float:
        return sum(v.amount for v in self.check_all(features))

    def makeable(self, features: list[Feature]) -> bool:
        return not self.check_all(features)


@dataclass
class CNC3Axis(Process):
    """Three-axis milling: the tool only ever points one way.

    The restriction that shapes everything is that the cutter approaches along
    a single axis per setup, so nothing may be hidden beneath anything else.
    A part needing five setups is not impossible, it is expensive, and the
    count is worth reporting rather than hiding.
    """

    name: str = "cnc-3axis"
    tool_diameter: float = 0.003            # 3 mm end mill
    #: Depth beyond about 4x diameter and the tool chatters and deflects.
    max_depth_ratio: float = 4.0
    min_wall: float = 0.0012
    setups: int = 2

    def check(self, f: Feature) -> list[Violation]:
        out: list[Violation] = []
        tool_r = self.tool_diameter / 2.0

        if f.undercut:
            out.append(Violation(
                "undercut",
                f"{f.name}: a 3-axis cutter approaches along one axis, so a "
                f"surface hidden beneath another cannot be reached. Reorient "
                f"the feature, split the part, or choose an additive process.",
                f.size or 0.001))

        if f.kind == "corner" and 0.0 < f.radius < tool_r:
            out.append(Violation(
                "internal-corner",
                f"{f.name}: internal radius {f.radius*1000:.2f} mm is smaller "
                f"than the {tool_r*1000:.2f} mm tool radius. A round cutter "
                f"cannot produce a sharper internal corner -- this is a "
                f"geometric impossibility, not a tolerance.",
                tool_r - f.radius))

        if f.kind in ("pocket", "hole") and 0.0 < f.size < self.tool_diameter:
            out.append(Violation(
                "feature-smaller-than-tool",
                f"{f.name}: {f.size*1000:.2f} mm is narrower than the "
                f"{self.tool_diameter*1000:.2f} mm cutter.",
                self.tool_diameter - f.size))

        if f.kind == "pocket" and f.size > 0 and f.depth > 0:
            ratio = f.depth / f.size
            if ratio > self.max_depth_ratio:
                out.append(Violation(
                    "pocket-too-deep",
                    f"{f.name}: depth/width {ratio:.1f} exceeds "
                    f"{self.max_depth_ratio:.0f}; the tool deflects and chatters.",
                    (ratio - self.max_depth_ratio) * f.size))

        if f.kind == "wall" and 0.0 < f.size < self.min_wall:
            out.append(Violation(
                "thin-wall",
                f"{f.name}: {f.size*1000:.2f} mm wall will chatter or deflect "
                f"under cutting load.",
                self.min_wall - f.size))
        return out


@dataclass
class FDM(Process):
    """Fused deposition: layers, so overhangs and bridges are the limits.

    Strictly a relaxation of the 3-axis constraints in the way that matters --
    it will happily print an undercut -- but it adds its own: an overhang past
    about 45 degrees needs support, and printed parts are weak across layers,
    which the structural side already accounts for by using across-layer
    strength for printed materials.
    """

    name: str = "fdm"
    nozzle: float = 0.0004
    min_wall_lines: int = 3
    max_overhang: float = math.radians(50.0)
    max_bridge: float = 0.010

    @property
    def min_wall(self) -> float:
        return self.nozzle * self.min_wall_lines

    def check(self, f: Feature) -> list[Violation]:
        out: list[Violation] = []
        if f.kind == "wall" and 0.0 < f.size < self.min_wall:
            out.append(Violation(
                "thin-wall",
                f"{f.name}: {f.size*1000:.2f} mm is under {self.min_wall_lines} "
                f"nozzle widths ({self.min_wall*1000:.2f} mm).",
                self.min_wall - f.size))
        if f.kind == "overhang" and f.angle > self.max_overhang:
            out.append(Violation(
                "overhang",
                f"{f.name}: {math.degrees(f.angle):.0f} deg from vertical "
                f"exceeds {math.degrees(self.max_overhang):.0f} deg and needs "
                f"support.",
                f.angle - self.max_overhang))
        if f.kind == "hole" and 0.0 < f.size < 2 * self.nozzle:
            out.append(Violation(
                "hole-too-small",
                f"{f.name}: {f.size*1000:.2f} mm will close up.",
                2 * self.nozzle - f.size))
        return out


PROCESSES: dict[str, type[Process]] = {"cnc-3axis": CNC3Axis, "fdm": FDM}


def get_process(name: str) -> Process:
    if name not in PROCESSES:
        raise KeyError(f"unknown process {name!r}; have {sorted(PROCESSES)}")
    return PROCESSES[name]()
