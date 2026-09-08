"""Decide whether a layout is worth a CFD run at all.

The 40-case campaign spent 22 of 30 usable runs establishing that rotors held
well clear of the airframe do not interfere with it. Those runs cost roughly
45 minutes and taught almost nothing: they landed in the flat region where the
correct answer is "about zero", and their only effect on the fitted model was
to make its aggregate error look better than it is (CAL-02).

The question those runs answered -- *is any part of the airframe underneath a
rotor disc* -- is geometry, not fluid dynamics. Projecting each disc straight
down and measuring the overlapped fraction of body planform costs microseconds.

Validated against the 30 labelled CFD cases from that campaign: correlation
0.991 against measured download, and a threshold of 5% overlap skips 22 cases
with **zero** missed interference cases. The screen is deliberately biased
toward running rather than skipping -- a wasted CFD run costs a minute, while a
missed interference case silently poisons the calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..control import G
from ..spec.vehicle import Rotor, Vehicle

#: Overlap below which a layout is treated as interference-free without CFD.
#: Measured: at this threshold the screen skipped 22 of 30 cases and missed
#: none. Chosen well below the smallest overlap of any case that showed real
#: download, because the failure modes are not symmetric.
SKIP_THRESHOLD = 0.05


@dataclass
class Screen:
    """What the geometric screen thinks, before any flow is solved."""

    overlap: float          # fraction of body planform under a disc, 0-1
    disc_loading: float     # N/m^2
    score: float            # overlap * loading, the quantity that tracks loss
    worth_solving: bool
    reason: str = ""


def wake_overlap(vehicle: Vehicle, samples: int = 20000,
                 seed: int = 0) -> tuple[float, float]:
    """Fraction of body plan area lying under a rotor disc, and disc loading.

    Monte Carlo rather than exact polygon intersection: the bodies are axis
    aligned boxes and the discs are circles, so the exact area is computable,
    but the union of four overlapping circles against several rectangles is
    fiddly enough that the closed form would be the more likely place for a
    bug. Twenty thousand samples give about half a percent of noise, which is
    far finer than the threshold needs.
    """
    rotors = [t for t in vehicle.thrusters if isinstance(t, Rotor)]
    bodies = vehicle.all_bodies()
    if not rotors or not bodies:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    hx = max(abs(b.position.x) + b.shape.half_extents().x for b in bodies)
    hy = max(abs(b.position.y) + b.shape.half_extents().y for b in bodies)
    if hx <= 0 or hy <= 0:
        return 0.0, 0.0

    px = rng.uniform(-hx, hx, samples)
    py = rng.uniform(-hy, hy, samples)

    in_body = np.zeros(samples, dtype=bool)
    for b in bodies:
        h = b.shape.half_extents()
        in_body |= ((np.abs(px - b.position.x) <= h.x)
                    & (np.abs(py - b.position.y) <= h.y))
    if not in_body.any():
        return 0.0, 0.0

    under = np.zeros(samples, dtype=bool)
    for r in rotors:
        under |= ((px - r.position.x) ** 2 + (py - r.position.y) ** 2) <= r.radius ** 2

    overlap = float((in_body & under).sum() / in_body.sum())
    disc_area = sum(math.pi * r.radius ** 2 for r in rotors)
    loading = vehicle.mass_properties().mass * G / disc_area if disc_area else 0.0
    return overlap, loading


def screen(vehicle: Vehicle, threshold: float = SKIP_THRESHOLD,
           samples: int = 20000) -> Screen:
    """Is this layout worth spending a CFD run on?"""
    overlap, loading = wake_overlap(vehicle, samples=samples)
    worth = overlap >= threshold
    return Screen(
        overlap=overlap, disc_loading=loading, score=overlap * loading,
        worth_solving=worth,
        reason=("" if worth else
                f"only {overlap:.1%} of the airframe sits under a disc "
                f"(threshold {threshold:.0%}); no measurable interference is "
                f"expected, so the run would confirm what geometry already says"),
    )


def estimated_loss(vehicle: Vehicle, threshold: float = SKIP_THRESHOLD) -> float:
    """Download to assume for a screened-out layout.

    Zero, deliberately, and only ever used for layouts the screen rejected. The
    screen is not a model of interference and must not be used as one: it says
    where interference *cannot* be, which is a much weaker and much more
    reliable claim than saying how much there is.
    """
    s = screen(vehicle, threshold)
    if s.worth_solving:
        raise ValueError(
            "this layout needs a CFD run; the screen only supplies a value for "
            "layouts it has ruled out"
        )
    return 0.0
