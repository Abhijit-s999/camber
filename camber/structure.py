"""Will the airframe survive its own thrust, and a landing?

P5 generates geometry, and geometry that has not been sized is a shape rather
than a design. This module is what turns one into the other.

**The design load is not hover.** A frame sized for the thrust needed to hang
still is a frame that breaks. Three cases matter and the largest governs:

* *manoeuvre* -- full thrust on one arm, which is what a hard roll commands,
* *landing* -- an impact deceleration through the legs and into the frame,
* *crash* -- a survivability case the user may or may not want to buy.

Sizing on hover thrust and then wondering why arms snap is the classic
first-build mistake, and an optimiser will make it far more enthusiastically
than a person: minimising mass drives every section to its limit, so whichever
load case is omitted is the one that breaks.

Arms are treated as cantilever beams. That is the right model for a multirotor
arm -- built in at the hub, loaded at the tip by its rotor -- and it is exact
enough that its error is smaller than the uncertainty in the material data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .control import G


@dataclass(frozen=True)
class Material:
    """Enough to size a beam, and no more than is honestly known.

    Values are conservative published figures for the stock a hobbyist
    actually buys, not laboratory optima: pultruded tube rather than autoclaved
    prepreg, 6061-T6 rather than 7075. A frame sized on optimistic properties
    is a frame sized on someone else's material.
    """

    name: str
    density: float              # kg/m^3
    youngs_modulus: float       # Pa
    yield_strength: float       # Pa (or ultimate, for brittle materials)
    #: Brittle materials get no plastic warning, so they carry more margin.
    brittle: bool = False
    source: str = ""


#: Conservative stock properties. Sources are the point: an unsourced number
#: here would be exactly the invented constant AI-05 forbids.
MATERIALS = {
    "cf_tube": Material(
        "carbon fibre tube (pultruded)", 1550.0, 70e9, 600e6, brittle=True,
        source="pultruded unidirectional CFRP, typical published minimum"),
    "al6061": Material(
        "aluminium 6061-T6", 2700.0, 68.9e9, 276e6,
        source="ASM 6061-T6 yield"),
    "pla": Material(
        "PLA (FDM, layer-adhesion limited)", 1240.0, 3.5e9, 35e6, brittle=True,
        source="printed strength across layers, well below datasheet bulk"),
    "petg": Material(
        "PETG (FDM)", 1270.0, 2.1e9, 30e6, brittle=True,
        source="printed strength across layers"),
    "nylon_cf": Material(
        "nylon + chopped CF (FDM)", 1200.0, 6.0e9, 70e6, brittle=True,
        source="printed, across layers"),
}


@dataclass(frozen=True)
class Section:
    """A beam cross-section. Round tube or solid rectangle."""

    kind: str                   # "tube" | "rect"
    a: float                    # tube: outer diameter; rect: width
    b: float                    # tube: wall thickness; rect: height

    def __post_init__(self) -> None:
        if self.a <= 0 or self.b <= 0:
            raise ValueError("section dimensions must be positive")
        if self.kind == "tube" and self.b >= self.a / 2:
            raise ValueError(
                f"wall {self.b*1000:.1f} mm cannot equal or exceed the radius "
                f"of a {self.a*1000:.1f} mm tube")
        if self.kind not in ("tube", "rect"):
            raise ValueError(f"unknown section kind {self.kind!r}")

    @property
    def area(self) -> float:
        if self.kind == "tube":
            ro, ri = self.a / 2, self.a / 2 - self.b
            return math.pi * (ro ** 2 - ri ** 2)
        return self.a * self.b

    @property
    def second_moment(self) -> float:
        """Second moment of area about the bending axis, m^4."""
        if self.kind == "tube":
            ro, ri = self.a / 2, self.a / 2 - self.b
            return math.pi / 4.0 * (ro ** 4 - ri ** 4)
        return self.a * self.b ** 3 / 12.0

    @property
    def extreme_fibre(self) -> float:
        """Distance from the neutral axis to the most stressed fibre."""
        return self.a / 2 if self.kind == "tube" else self.b / 2


@dataclass
class LoadCase:
    name: str
    tip_force_n: float
    description: str = ""


def load_cases(hover_thrust_n: float, max_thrust_n: float, vehicle_mass_kg: float,
               n_arms: int, landing_g: float = 6.0) -> list[LoadCase]:
    """The loads an arm must actually survive.

    `landing_g` of 6 is a modest touchdown, not a crash: a 1 m/s arrival
    stopped in about 8 mm of leg flex. Racing frames are routinely built to
    far more.
    """
    return [
        LoadCase("hover", hover_thrust_n,
                 "steady state; almost never the governing case"),
        LoadCase("manoeuvre", max_thrust_n,
                 "one arm at full thrust, as a hard roll commands"),
        LoadCase("landing", vehicle_mass_kg * G * landing_g / max(n_arms, 1),
                 f"{landing_g:g}g touchdown shared across arms"),
    ]


@dataclass
class BeamResult:
    case: str
    stress_pa: float
    deflection_m: float
    margin: float               # allowable / actual; >1 passes
    passes: bool


def analyse_arm(length: float, section: Section, material: Material,
                load: LoadCase, safety_factor: float | None = None,
                max_tip_deflection_ratio: float = 0.02) -> BeamResult:
    """Cantilever with a tip load: stress, deflection, and the margin.

        sigma = M c / I,  M = F L        delta = F L^3 / (3 E I)

    Deflection is checked as well as stress, and not as a comfort criterion. A
    visibly flexing arm tilts its rotor away from vertical, which steals thrust
    and feeds the flight controller a moment it did not ask for; frames usually
    fail this limit long before they approach yield.
    """
    if length <= 0:
        raise ValueError("arm length must be positive")
    if safety_factor is None:
        # Brittle materials fail without warning, so they carry more margin.
        safety_factor = 3.0 if material.brittle else 2.0

    moment = load.tip_force_n * length
    stress = moment * section.extreme_fibre / section.second_moment
    deflection = (load.tip_force_n * length ** 3
                  / (3.0 * material.youngs_modulus * section.second_moment))

    allowable_stress = material.yield_strength / safety_factor
    allowable_deflection = max_tip_deflection_ratio * length

    stress_margin = allowable_stress / stress if stress > 0 else math.inf
    deflection_margin = (allowable_deflection / deflection
                         if deflection > 0 else math.inf)
    margin = min(stress_margin, deflection_margin)
    return BeamResult(load.name, stress, deflection, margin, margin >= 1.0)


def arm_mass(length: float, section: Section, material: Material) -> float:
    return length * section.area * material.density


def size_arm(length: float, material: Material, loads: list[LoadCase],
             kind: str = "tube", candidates: int = 60,
             safety_factor: float | None = None) -> tuple[Section, float] | None:
    """Lightest section from a stock ladder that survives every load case.

    Searched over a discrete ladder rather than solved in closed form, because
    tube is bought in sizes. A continuous optimum of 11.3 mm outer diameter is
    not purchasable, and rounding it afterwards either wastes mass or quietly
    invalidates the margin it was chosen for.
    """
    best: tuple[Section, float] | None = None
    if kind == "tube":
        outers = [d / 1000.0 for d in
                  (8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30)]
        walls = [t / 1000.0 for t in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)]
        pairs = [(o, w) for o in outers for w in walls if w < o / 2]
    else:
        widths = [w / 1000.0 for w in (8, 10, 12, 15, 20, 25, 30)]
        heights = [h / 1000.0 for h in (2, 3, 4, 5, 6, 8, 10, 12)]
        pairs = [(w, h) for w in widths for h in heights]

    for a, b in pairs[:candidates * 10]:
        try:
            section = Section(kind, a, b)
        except ValueError:
            continue
        if all(analyse_arm(length, section, material, load,
                           safety_factor).passes for load in loads):
            mass = arm_mass(length, section, material)
            if best is None or mass < best[1]:
                best = (section, mass)
    return best
