"""Turning a vehicle specification into something an optimiser can search.

The translation is the interesting part, and it is deliberately not automatic
in the places where being automatic would be wrong.

**What is free.** A body is a decision variable unless a constraint pins it.
`Fixed` removes it entirely; `Relative` removes the dependent one and lets the
pair move together. That is why constraints are declarative data (ARCH-02) --
the optimiser reads them to discover its own search space, which it cannot do
from imperative validation code.

**What is a hard constraint.** Controllability and hover feasibility prune
candidates outright rather than being penalised, because a design that cannot
be flown is not a worse design, it is not a design. Geometric constraints --
clearance, view cones, CG box -- become penalties, because they are continuous
and an optimiser can walk downhill out of a mild violation toward feasibility.
Making that split wrongly in either direction is expensive: penalise
controllability and the search wastes its budget among unflyable candidates;
harden the view cone and it cannot find its way back in.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..control import G, controllability, solve_hover
from ..spec.constraints import Fixed, Relative
from ..spec.units import Vec3
from ..spec.vehicle import Battery, Objective, Rotor, Vehicle

#: Objectives are always minimised. A maximisation is negated on the way in,
#: once, here -- so no downstream code has to remember the sign.
MINIMISE = True


@dataclass
class DesignVariable:
    """One scalar the optimiser may change, and how to apply it."""

    name: str
    lower: float
    upper: float
    apply: Callable[[Vehicle, float], None]
    #: Continuous unless stated; integers are rounded before application.
    integral: bool = False
    #: An index into a list with no order, such as a motor in the catalogue:
    #: index 7 is no nearer to 8 than to 30, so the search must not treat
    #: it as a number (OPT-08). Implies integral.
    categorical: bool = False

    def __post_init__(self) -> None:
        if self.categorical:
            self.integral = True

    def clip(self, value: float) -> float:
        v = min(max(value, self.lower), self.upper)
        return round(v) if self.integral else v


@dataclass
class Evaluation:
    """What one candidate is worth, and why it might be rejected."""

    objectives: np.ndarray
    #: Total constraint violation. Zero means feasible.
    violation: float
    feasible: bool
    reason: str = ""
    detail: dict = field(default_factory=dict)


def position_variables(vehicle: Vehicle, span: float = 0.06) -> list[DesignVariable]:
    """Free translation for every body no constraint has pinned.

    `span` is the half-range in metres each free body may move along each axis
    from where the spec placed it. Bounding the search near the author's
    starting layout is deliberate: the spec is a design intent, not an
    arbitrary initial guess, and letting a payload wander the whole envelope
    mostly finds infeasible layouts more slowly.
    """
    pinned: set[str] = set()
    for c in vehicle.constraints:
        if isinstance(c, Fixed):
            pinned.add(c.body)
        elif isinstance(c, Relative):
            pinned.add(c.a)  # `a` follows `b`; only `b` is free

    variables: list[DesignVariable] = []
    for body in vehicle.bodies:
        if body.name in pinned:
            continue
        origin = body.position
        for axis in ("x", "y", "z"):
            start = getattr(origin, axis)

            def make(name: str, ax: str):
                def apply(v: Vehicle, value: float) -> None:
                    b = v.body(name)
                    p = b.position
                    coords = {"x": p.x, "y": p.y, "z": p.z}
                    coords[ax] = value
                    b.placement = type(b.placement)(
                        Vec3(**coords), b.placement.rotation)
                return apply

            variables.append(DesignVariable(
                f"{body.name}.{axis}", start - span, start + span,
                make(body.name, axis)))
    return variables


def rotor_radius_variable(vehicle: Vehicle, lower: float, upper: float) -> DesignVariable:
    """One radius shared by every rotor. Mixed sizes are not a real design."""
    def apply(v: Vehicle, value: float) -> None:
        for t in v.thrusters:
            if isinstance(t, Rotor):
                t.radius = value
    return DesignVariable("rotor.radius", lower, upper, apply)


def arm_length_variable(vehicle: Vehicle, lower: float, upper: float) -> DesignVariable:
    """Scale the rotor arms in plan, preserving the layout's shape."""
    originals = [(t.name, t.position) for t in vehicle.thrusters]
    reference = max((math.hypot(p.x, p.y) for _, p in originals), default=1.0) or 1.0

    def apply(v: Vehicle, value: float) -> None:
        scale = value / reference
        for name, p in originals:
            t = v.thruster(name)
            t.placement = type(t.placement)(
                Vec3(p.x * scale, p.y * scale, p.z), t.placement.rotation)
    return DesignVariable("arm.length", lower, upper, apply)


@dataclass
class DesignProblem:
    """A searchable design space built from one vehicle specification."""

    template: Vehicle
    variables: list[DesignVariable]
    #: Callable returning endurance in seconds, or None if not wired up.
    endurance_model: Callable[[Vehicle], float] | None = None
    #: Minimum acceptable controllability margin; below this a design is
    #: rejected outright rather than penalised. See the module docstring.
    min_control_margin: float = 1e-3
    min_rank: int = 4
    #: Calibrated interference correction (camber/calibrate.py). When present
    #: the optimiser searches the *corrected* cheap model, which is the whole
    #: point of the fidelity ladder (ARCH-01): CFD never evaluates a candidate,
    #: it only teaches L1 where L1 is wrong.
    interference: object | None = None
    #: The catalogue a component search picks from (CAT-01). When present,
    #: rotor size, mass and thrust come from the chosen parts.
    components: object | None = None

    @property
    def n_var(self) -> int:
        return len(self.variables)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([v.lower for v in self.variables]),
                np.array([v.upper for v in self.variables]))

    @property
    def categorical(self) -> np.ndarray:
        return np.array([v.categorical for v in self.variables], dtype=bool)

    @property
    def objective_names(self) -> list[str]:
        obj = self.template.mission.objective
        if obj is Objective.MAX_ENDURANCE:
            return ["-endurance_s", "mass_kg"]
        if obj is Objective.MAX_SPEED:
            return ["-thrust_margin", "-endurance_s"]
        raise ValueError(f"no objective function for {obj.value!r}")

    def build(self, x: Sequence[float]) -> Vehicle:
        """A concrete vehicle for one point in the search space."""
        return self.resolve(x)[0]

    def resolve(self, x: Sequence[float]):
        """The vehicle for one point, and the hardware it chose, if any."""
        v = copy.deepcopy(self.template)
        for variable, value in zip(self.variables, x):
            variable.apply(v, variable.clip(float(value)))
        choice = None
        if self.components is not None:
            self.components.pin(v)
            choice = self.components.resolve(v)
        return v, choice

    def evaluate(self, x: Sequence[float]) -> Evaluation:
        big = np.array([1e6, 1e6], dtype=float)
        try:
            vehicle, choice = self.resolve(x)
        except Exception as exc:  # noqa: BLE001
            return Evaluation(big, 1e6, False, f"malformed candidate: {exc}")

        # --- hard gates: not flyable is not a design ----------------------
        if choice is not None and choice.stalled:
            return Evaluation(big, 1e3, False,
                              f"{choice.motor.label} cannot turn a {choice.prop.label} "
                              f"on {choice.cells}S at all")
        c = controllability(vehicle)
        if c.rank < self.min_rank:
            return Evaluation(big, 1e3 + (self.min_rank - c.rank), False,
                              f"rank {c.rank} < {self.min_rank}: "
                              f"cannot command {', '.join(c.uncontrollable_axes)}")
        if c.margin < self.min_control_margin:
            return Evaluation(big, 1e3, False,
                              f"control margin {c.margin:.3g} below the floor")

        hover = solve_hover(vehicle)
        if not hover.feasible:
            return Evaluation(big, 1e3, False, f"cannot hover: {hover.reason}")

        # --- soft constraints: continuous, so the search can climb out ----
        positions = vehicle.part_positions()
        violation = 0.0
        breaches: dict[str, float] = {}
        for constraint in vehicle.constraints:
            try:
                amount = constraint.violation(positions, vehicle)
            except Exception:  # noqa: BLE001
                amount = 0.0
            if amount > 1e-9:
                breaches[type(constraint).__name__] = amount
                violation += amount
        overlap = prop_overlap(vehicle)
        if overlap > 1e-9:
            breaches["prop overlap"] = overlap
            violation += overlap
        if choice is not None:
            for name, amount in choice.violations.items():
                breaches[name] = amount
                violation += amount

        mp = vehicle.mass_properties()

        # Interference makes the rotors work harder for the same lift: to hold
        # the aircraft up they must produce weight / (1 - loss) of *isolated*
        # thrust, because the airframe gives some of it back as download. That
        # extra thrust costs current, which costs endurance -- so the
        # correction reaches the objective through the physics rather than
        # being bolted on as a fudge factor.
        loss = 0.0
        if self.interference is not None:
            loss, _ = self.interference.correction(vehicle)
            loss = min(max(loss, 0.0), 0.5)
            violation += max(0.0, loss - 0.15)   # heavy overlap is a bad design

        objectives = self._objectives(vehicle, hover, mp, c, loss)

        # The mission's floor on flight time, which the spec has always
        # carried and the search never read. Scaled like the other limits:
        # 10% short costs 0.1.
        floor = vehicle.mission.min_endurance
        if floor:
            names = self.objective_names
            endurance = -objectives[names.index("-endurance_s")]
            short = 1.0 - endurance / floor
            if short > 1e-9:
                breaches["mission endurance"] = short
                violation += short

        return Evaluation(
            objectives=objectives,
            violation=violation,
            feasible=violation <= 1e-9,
            reason="" if violation <= 1e-9 else
                   "; ".join(f"{k} {v:.3g}" for k, v in breaches.items()),
            detail={"mass_kg": mp.mass, "control_margin": c.margin,
                    "interference_loss": loss,
                    "max_utilisation": float(np.nanmax(hover.utilisation))
                    if hover.utilisation is not None else float("nan"),
                    **({"components": choice.bill()} if choice is not None else {})},
        )

    def _objectives(self, vehicle, hover, mp, control,
                    interference_loss: float = 0.0) -> np.ndarray:
        obj = vehicle.mission.objective

        if obj is Objective.MAX_ENDURANCE:
            if self.endurance_model is not None:
                endurance = self.endurance_model(
                    vehicle, thrust_scale=1.0 / (1.0 - interference_loss))
            else:
                # Without the physics tier wired in, fall back to stored energy
                # per unit hover power estimated from momentum theory. Crude,
                # and explicitly a placeholder rather than a claim.
                endurance = self._momentum_endurance(vehicle, mp)
            return np.array([-endurance, mp.mass])

        if obj is Objective.MAX_SPEED:
            installed = 0.0
            for t in vehicle.thrusters:
                try:
                    installed += t.max_thrust()
                except ValueError:
                    installed = float("nan")
                    break
            margin = installed / (mp.mass * G) if installed == installed else 0.0

            # Thrust-to-weight against *endurance*, not against mass. Installed
            # thrust is fixed by the chosen motors, so minimising mass is
            # exactly maximising T:W -- the same axis twice, and the front
            # collapses to a single point. Pairing it with endurance is the
            # trade a racer actually makes: a bigger pack flies longer and
            # accelerates worse.
            if self.endurance_model is not None:
                endurance = self.endurance_model(
                    vehicle, thrust_scale=1.0 / (1.0 - interference_loss))
            else:
                endurance = self._momentum_endurance(vehicle, mp)
            return np.array([-margin, -endurance])

        raise ValueError(f"no objective function for {obj.value!r}")

    @staticmethod
    def _momentum_endurance(vehicle: Vehicle, mp) -> float:
        """Ideal-hover endurance from momentum theory and stored energy.

        Deliberately optimistic and known to be: it assumes a perfect rotor and
        ignores motor and ESC losses, so it overstates endurance about five
        times (OPT-03). It exists so the optimiser has a smooth, monotone
        surrogate before the blade-element tier is attached, not to be quoted.
        """
        energy = sum(b.energy_wh for b in vehicle.all_bodies()
                     if isinstance(b, Battery)) * 3600.0
        if energy <= 0:
            return 0.0
        area = sum(math.pi * t.radius ** 2 for t in vehicle.thrusters
                   if isinstance(t, Rotor))
        if area <= 0:
            return 0.0
        thrust = mp.mass * G
        induced = math.sqrt(thrust / (2.0 * 1.225 * area))
        power = thrust * induced
        return energy / power if power > 0 else 0.0


def prop_overlap(vehicle: Vehicle, gap: float = 0.004) -> float:
    """How far neighbouring discs at the same height cut into each other.

    Summed per pair, as a fraction of the radius, below a 4 mm gap (the same
    warning line as the Design step's clearance check). The search always
    varied arm length and prop size independently and never checked this: at
    the shortest arms it allowed, two 5-inch discs overlapped by 14 mm.
    """
    rotors = [t for t in vehicle.thrusters if isinstance(t, Rotor)]
    total = 0.0
    for i, a in enumerate(rotors):
        for b in rotors[i + 1:]:
            if abs(a.position.z - b.position.z) >= 0.01:
                continue    # stacked or coaxial: they may overlap in plan
            centre = math.hypot(a.position.x - b.position.x, a.position.y - b.position.y)
            short = gap - (centre - a.radius - b.radius)
            if short > 0:
                total += short / max(a.radius, b.radius)
    return total


# --------------------------------------------------------------------------
# Wiring the real physics tier in place of the momentum-theory placeholder.
# --------------------------------------------------------------------------

def battery_capacity_variable(vehicle: Vehicle, name: str,
                              lower_ah: float, upper_ah: float,
                              wh_per_kg: float = 180.0) -> DesignVariable:
    """Size the pack, with its mass following from its energy.

    This is what makes endurance-versus-mass a real trade-off rather than a
    degenerate axis. Without it the optimiser has no way to spend mass on
    energy, every candidate weighs the same, and the "front" collapses to a
    single point repeated across the population -- which is exactly what the
    first run produced.

    180 Wh/kg is a reasonable figure for a modern high-discharge LiPo including
    wiring and connector. It is a modelling constant and it is worth
    remembering it is one: cell chemistry moves this number.
    """
    def apply(v: Vehicle, value: float) -> None:
        pack = v.body(name)
        if not isinstance(pack, Battery):
            raise TypeError(f"{name!r} is not a battery")
        pack.capacity_ah = value
        pack.mass = value * pack.nominal_voltage / wh_per_kg

    return DesignVariable(f"{name}.capacity_ah", lower_ah, upper_ah, apply)


def physics_endurance_model(prop_key: str,
                            motor,
                            propdb_root: str | None = None,
                            blades: int = 2,
                            usable_fraction: float = 0.80):
    """An endurance model backed by the validated blade-element tier.

    One propeller map is built and reused for every candidate radius. That is
    legitimate because CT and CP are functions of blade *shape* -- c/R and
    twist against r/R -- and of Reynolds number, not of absolute size; thrust
    then scales as rho n^2 D^4. The Reynolds part is the approximation, and
    PHYS-04 records that it is not negligible for propellers this small, so
    comparing radii far apart carries a known error.
    """
    from ..catalogue import uiuc
    from ..physics.battery import Pack
    from ..physics.bemt import BladeDefinition
    from ..physics.propulsion import PropellerMap, thrust_for

    from ..paths import propdb
    db = uiuc.load(propdb_root or propdb())
    if prop_key not in db.geometry:
        raise KeyError(f"no geometry for {prop_key!r}")
    reference = db.geometry[prop_key]
    blade = BladeDefinition.from_uiuc(reference, blades).resample(20)
    base_map = PropellerMap.from_blade(blade, name=prop_key)

    def endurance(v: Vehicle, thrust_scale: float = 1.0) -> float:
        rotors = [t for t in v.thrusters if isinstance(t, Rotor)]
        packs = [b for b in v.all_bodies() if isinstance(b, Battery)]
        if not rotors or not packs:
            return 0.0
        pack_spec = packs[0]

        # Re-scale the map to this candidate's diameter.
        scaled = PropellerMap(
            diameter_m=2.0 * rotors[0].radius,
            advance_ratios=base_map.advance_ratios,
            ct=base_map.ct, cp=base_map.cp,
            rpm_sampled=base_map.rpm_sampled, name=base_map.name,
        )
        cells = max(1, round(pack_spec.nominal_voltage / 3.7))
        pack = Pack("pack", cells=cells, capacity_ah=pack_spec.capacity_ah,
                    resistance_ohm=0.018, max_c_rate=pack_spec.max_c_rate or 50.0)

        mp = v.mass_properties()
        # thrust_scale > 1 when interference is being accounted for: the
        # rotors must make more isolated thrust than the weight, because the
        # airframe takes some of it back.
        per_rotor = mp.mass * G / len(rotors) * thrust_scale
        try:
            op = thrust_for(motor, scaled, pack, per_rotor, v_inf=0.0)
        except ValueError:
            return 0.0  # cannot produce hover thrust at all
        # The pack's current, not the motors' (PHYS-10).
        total_current = op.supply_current(pack.terminal_voltage(0.0, 1.0)) * len(rotors)
        if total_current <= 0:
            return 0.0
        return pack.endurance_s(total_current, usable_fraction=usable_fraction)

    return endurance
