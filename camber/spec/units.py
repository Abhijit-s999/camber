"""Vectors, and the unit convention everything else assumes.

**SI throughout, no exceptions.** Metres, kilograms, seconds, radians, newtons,
watts, volts, amperes. Component datasheets are quoted in inches, grams, mAh,
RPM and degrees; those are converted at the catalogue boundary and never appear
inside the model. A propeller is 0.1270 m in diameter here, not "5 inch".

The reason for the rule is that mixed units survive unit tests. Every value
passes an assertion about its own type and none about its dimension, so a
diameter in millimetres flows through the whole stack and emerges as a
plausible, wrong answer -- which is precisely the failure mode this project
cannot tolerate. Convert once, at the edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

#: Conversions permitted only at the catalogue/IO boundary.
INCH = 0.0254
GRAM = 1e-3
DEG = math.pi / 180.0


@dataclass(frozen=True)
class Vec3:
    """A point or direction in the body frame, in metres.

    Body frame convention, right-handed, matching the aerospace norm used by
    every flight controller this will ever export a mixer to:

    * ``+x`` forward (nose)
    * ``+y`` right (starboard)
    * ``+z`` down

    ``z`` down is the part that surprises people, and it is not negotiable: it
    is what ArduPilot, PX4 and every rotation-matrix convention in the flight
    dynamics literature assume. Thrust from a hovering rotor is therefore
    **negative** z.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Vec3":
        return Vec3(self.x / k, self.y / k, self.z / k)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    @property
    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def unit(self) -> "Vec3":
        n = self.norm
        if n == 0.0:
            raise ValueError("cannot normalise a zero vector")
        return self / n

    def distance_to(self, other: "Vec3") -> float:
        return (self - other).norm

    @classmethod
    def parse(cls, value) -> "Vec3":
        """Accept a Vec3, a 3-sequence, or a mapping with x/y/z."""
        if isinstance(value, Vec3):
            return value
        if isinstance(value, dict):
            return cls(float(value.get("x", 0.0)),
                       float(value.get("y", 0.0)),
                       float(value.get("z", 0.0)))
        seq = list(value)
        if len(seq) != 3:
            raise ValueError(f"expected 3 components, got {len(seq)}: {value!r}")
        return cls(*(float(v) for v in seq))

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]


ORIGIN = Vec3(0.0, 0.0, 0.0)
#: Unit gravity direction in the body frame at level attitude.
DOWN = Vec3(0.0, 0.0, 1.0)
UP = Vec3(0.0, 0.0, -1.0)
FORWARD = Vec3(1.0, 0.0, 0.0)
