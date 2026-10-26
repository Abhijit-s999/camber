"""Choosing the motor, the propeller and the cell count (CAT-01).

Three genes: a motor from the shipped table, a propeller from the measured
archive, and the pack's cell count. Every rotor gets the same motor and
propeller, because a multirotor with mixed rotors is not a real design (the
same reason the old radius gene was shared).

**Resolved after the genes, not by them.** A rotor's thrust at full throttle
depends on the motor, the propeller and the pack voltage together, so it can
only be computed once all three are set. `resolve` does that, and writes the
results into the vehicle: rotor size, mass and measured-not-guessed static
thrust (AI-05). Everything downstream -- hover feasibility, controllability,
the saved spec, `generate`, the simulator -- then reads the chosen hardware
and nothing else.

**Limits are constraints, not filters** (CAT-03). Overcurrent, a pack outside
the motor's rated cells, a pack asked for more than its C rating and a
propeller past its tip-speed limit all add to the violation, scaled so that
10% over a limit costs 0.1. A filter would hide the reason a pairing was
rejected; a violation keeps it, and lets the search walk back from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..control import G
from ..physics.battery import CELL_FULL_V, CELL_NOMINAL_V, Pack
from ..physics.propulsion import solve_operating_point, thrust_for
from ..spec.vehicle import Battery, Rotor, Vehicle
from .problem import DesignVariable

#: Tip Mach number beyond which the measured coefficients are not trusted.
#: The tunnel ran the tips below Mach 0.3; past about 0.7 compressibility
#: drag rises steeply and nothing measured here knows about it (CAT-03).
TIP_MACH_LIMIT = 0.7
SPEED_OF_SOUND = 343.0

#: Whole-pack internal resistance per cell. The endurance model always used
#: 18 mOhm for a 4S pack; per cell it scales with the cell count chosen.
PACK_OHM_PER_CELL = 0.0045


@dataclass
class Choice:
    """One candidate's hardware and what it does at full throttle."""

    motor: object
    prop: object
    cells: int
    #: Per rotor, full throttle, nominal pack voltage.
    static_thrust_n: float = 0.0
    #: Per motor, full throttle, full pack: the worst case for the ratings.
    full_current_a: float = 0.0
    full_rpm: float = 0.0
    violations: dict[str, float] = field(default_factory=dict)
    #: Set when the motor cannot turn the propeller at all.
    stalled: bool = False

    def bill(self) -> dict:
        """The bill of materials row, with where each part's numbers came from."""
        m, p = self.motor, self.prop
        return {
            "motor": {"id": m.id, "label": m.label, "kv": m.kv, "mass_g": m.mass_kg * 1000,
                      "max_current_a": m.max_current_a, "url": m.url},
            "prop": {"key": p.key, "label": p.label, "diameter_in": p.diameter_in,
                     "pitch_in": p.pitch_in, "blades": p.blades,
                     "mass_g": p.mass_kg * 1000, "mass_source": p.mass_source,
                     "source": "UIUC Propeller Data Site (measured)"},
            "cells": self.cells,
            "static_thrust_n": self.static_thrust_n,
            "full_throttle_current_a": self.full_current_a,
            "full_throttle_rpm": self.full_rpm,
        }


class ComponentSpace:
    """The catalogue a search may pick from, and how a pick is applied."""

    def __init__(self, motors: Sequence, props: Sequence,
                 cells: tuple[int, int] = (1, 6), wh_per_kg: float = 180.0) -> None:
        if not motors or not props:
            raise ValueError("a component search needs at least one motor and one propeller")
        lo, hi = cells
        if not 1 <= lo <= hi:
            raise ValueError(f"cell range {lo}-{hi}")
        self.motors = list(motors)
        self.props = list(props)
        self.cells = (lo, hi)
        self.wh_per_kg = wh_per_kg
        self._motor = {m.id: m for m in self.motors}
        self._prop = {p.key: p for p in self.props}

    def motor(self, ident: str):
        return self._motor[ident]

    def prop(self, key: str):
        return self._prop[key]

    # -- genes -------------------------------------------------------------

    def variables(self, vehicle: Vehicle) -> list[DesignVariable]:
        out: list[DesignVariable] = []
        if len(self.motors) > 1:
            out.append(DesignVariable("motor", 0, len(self.motors) - 1,
                                      self._set_motor, categorical=True))
        if len(self.props) > 1:
            out.append(DesignVariable("prop", 0, len(self.props) - 1,
                                      self._set_prop, categorical=True))
        if vehicle.batteries and self.cells[1] > self.cells[0]:
            # Cells are ordinal: 5S really is between 4S and 6S.
            out.append(DesignVariable("battery.cells", *self.cells,
                                      self._set_cells, integral=True))
        return out

    def _set_motor(self, v: Vehicle, index: float) -> None:
        for t in _rotors(v):
            t.motor_sku = self.motors[int(index)].id

    def _set_prop(self, v: Vehicle, index: float) -> None:
        p = self.props[int(index)]
        for t in _rotors(v):
            t.prop_sku = p.key
            t.radius = p.diameter_m / 2.0
            t.pitch = p.pitch_m
            t.blades = p.blades

    def _set_cells(self, v: Vehicle, cells: float) -> None:
        pack = v.batteries[0]
        pack.nominal_voltage = int(cells) * CELL_NOMINAL_V
        # The same rule as the capacity gene, so the order they apply in
        # cannot change the answer: mass follows energy.
        pack.mass = pack.capacity_ah * pack.nominal_voltage / self.wh_per_kg

    def pin(self, v: Vehicle) -> None:
        """Apply a one-entry list or a fixed cell count, which has no gene."""
        if len(self.motors) == 1:
            self._set_motor(v, 0)
        if len(self.props) == 1:
            self._set_prop(v, 0)
        if v.batteries and self.cells[0] == self.cells[1]:
            self._set_cells(v, self.cells[0])

    # -- resolution --------------------------------------------------------

    def resolve(self, v: Vehicle) -> Choice | None:
        """Turn the chosen part numbers into thrust, mass and limits."""
        rotors = _rotors(v)
        if not rotors or not rotors[0].motor_sku or not rotors[0].prop_sku:
            return None
        motor = self._motor[rotors[0].motor_sku]
        prop = self._prop[rotors[0].prop_sku]
        packs = v.batteries
        cells = max(1, round(packs[0].nominal_voltage / CELL_NOMINAL_V)) if packs else self.cells[0]
        choice = Choice(motor, prop, cells)
        for t in rotors:
            t.mass = motor.mass_kg + prop.mass_kg

        m = motor.motor
        try:
            nominal = solve_operating_point(m, prop.measured, cells * CELL_NOMINAL_V)
            full = solve_operating_point(m, prop.measured, cells * CELL_FULL_V)
        except ValueError:
            choice.stalled = True
            for t in rotors:
                t.static_thrust_n = 1e-6
            return choice

        choice.static_thrust_n = nominal.thrust_n
        choice.full_current_a = full.current_a
        choice.full_rpm = full.rpm
        # Torque per unit thrust, from the measured coefficients: Q/T = CP D /
        # (2 pi CT). It sets yaw authority, which a default of 0.016 m guessed.
        q_per_t = nominal.torque_nm / nominal.thrust_n if nominal.thrust_n > 0 else None
        for t in rotors:
            t.static_thrust_n = nominal.thrust_n
            if q_per_t:
                t.torque_coefficient = q_per_t

        over = choice.violations
        def add(name: str, amount: float) -> None:
            if amount > 1e-9:
                over[name] = amount

        add("motor current", full.current_a / motor.max_current_a - 1.0)
        add("motor cells", max(motor.cells_min - cells, cells - motor.cells_max, 0) / 1.0)
        if packs and packs[0].max_c_rate > 0:
            rating = packs[0].capacity_ah * packs[0].max_c_rate
            add("pack current", full.current_a * len(rotors) / rating - 1.0)
        tip = full.rpm / 60.0 * 3.141592653589793 * prop.diameter_m / SPEED_OF_SOUND
        add("tip speed", tip / TIP_MACH_LIMIT - 1.0)
        if prop.max_rpm:
            add("prop rpm", full.rpm / prop.max_rpm - 1.0)
        return choice

    # -- endurance ---------------------------------------------------------

    def endurance_model(self, usable_fraction: float = 0.80):
        """Hover endurance on the chosen motor, propeller and pack.

        The same physics as `physics_endurance_model` -- motor and measured
        propeller matched by root-finding, the pack coulomb-counted with sag --
        but on the parts this candidate chose instead of fixed ones.
        """
        def endurance(v: Vehicle, thrust_scale: float = 1.0) -> float:
            rotors = _rotors(v)
            packs = [b for b in v.all_bodies() if isinstance(b, Battery)]
            if not rotors or not packs or not rotors[0].motor_sku:
                return 0.0
            motor = self._motor[rotors[0].motor_sku].motor
            prop = self._prop[rotors[0].prop_sku].measured
            spec = packs[0]
            cells = max(1, round(spec.nominal_voltage / CELL_NOMINAL_V))
            pack = Pack("pack", cells=cells, capacity_ah=spec.capacity_ah,
                        resistance_ohm=PACK_OHM_PER_CELL * cells,
                        max_c_rate=spec.max_c_rate or 50.0)
            per_rotor = v.mass_properties().mass * G / len(rotors) * thrust_scale
            try:
                op = thrust_for(motor, prop, pack, per_rotor, v_inf=0.0)
            except ValueError:
                return 0.0
            current = op.supply_current(pack.terminal_voltage(0.0, 1.0)) * len(rotors)
            if current <= 0:
                return 0.0
            return pack.endurance_s(current, usable_fraction=usable_fraction)
        return endurance


def screen(problem, keep: int, limit: int = 20000, seed: int = 0):
    """Every combination of the discrete genes, at the author's layout.

    The parts are a small discrete space -- tens of motors, tens of
    propellers, a handful of cell counts -- and the layout a continuous one.
    A genetic search over both settles on whichever motor it met early, so
    the answer depended on the seed: over six seeds at the app's budget,
    quad5's best endurance ran from 30.2 to 36.6 min and racer5's T:W from
    6.6 to 13.7. So the discrete part is enumerated first, with every
    continuous gene at the middle of its range, and the best `keep` start the
    search (CAT-05). Above `limit` combinations a random sample stands in.

    Returns candidate rows, best first by constrained rank.
    """
    import itertools

    import numpy as np

    from .nsga2 import _constrained_ranks
    from .pareto import crowding_distance

    lower, upper = problem.bounds
    discrete = [i for i, v in enumerate(problem.variables) if v.integral]
    if not discrete:
        return np.empty((0, problem.n_var))
    ranges = [range(int(lower[i]), int(upper[i]) + 1) for i in discrete]
    total = int(np.prod([len(r) for r in ranges]))
    middle = (lower + upper) / 2
    if total <= limit:
        combos = itertools.product(*ranges)
    else:
        rng = np.random.default_rng(seed)
        combos = (tuple(rng.integers(r.start, r.stop) for r in ranges) for _ in range(limit))
    rows = []
    for combo in combos:
        x = middle.copy()
        x[discrete] = combo
        rows.append(x)
    rows = np.array(rows)
    evals = [problem.evaluate(x) for x in rows]
    objectives = np.array([e.objectives for e in evals])
    picked: list[int] = []
    for front in _constrained_ranks(evals, objectives):
        if len(picked) + len(front) <= keep:
            picked += front
            continue
        # Spread out along the trade-off rather than crowd one end of it.
        d = crowding_distance(objectives[front]) if len(front) > 2 else np.ones(len(front))
        picked += [front[i] for i in np.argsort(-d)[:keep - len(picked)]]
        break
    return rows[picked]


def _rotors(v: Vehicle) -> list[Rotor]:
    return [t for t in v.thrusters if isinstance(t, Rotor)]
