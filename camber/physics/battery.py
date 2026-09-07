"""Lithium polymer pack: energy, sag under load, and when it actually stops.

Two failure modes matter and they are not the same. A pack can run *out* --
capacity exhausted -- or it can sag below the voltage the system needs while
plenty of charge remains. Sizing on energy alone finds designs that brown out
on the first aggressive manoeuvre, which is why internal resistance and C-rate
are first-class here rather than annotations.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Nominal and full-charge volts for one LiPo cell.
CELL_NOMINAL_V = 3.7
CELL_FULL_V = 4.2
#: Below this a LiPo cell is being damaged; treat it as empty.
CELL_CUTOFF_V = 3.3


@dataclass(frozen=True)
class Pack:
    name: str
    cells: int
    capacity_ah: float
    #: Whole-pack internal resistance. Roughly 3-6 mOhm per cell for a healthy
    #: high-discharge hobby pack; measured values belong here when available.
    resistance_ohm: float
    max_c_rate: float
    mass_kg: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        if self.cells < 1 or self.capacity_ah <= 0:
            raise ValueError(f"{self.name}: cells and capacity must be positive")

    @property
    def nominal_voltage(self) -> float:
        return self.cells * CELL_NOMINAL_V

    @property
    def full_voltage(self) -> float:
        return self.cells * CELL_FULL_V

    @property
    def cutoff_voltage(self) -> float:
        return self.cells * CELL_CUTOFF_V

    @property
    def energy_wh(self) -> float:
        return self.capacity_ah * self.nominal_voltage

    @property
    def max_current_a(self) -> float:
        return self.capacity_ah * self.max_c_rate

    def open_circuit_voltage(self, state_of_charge: float) -> float:
        """Resting cell voltage against remaining charge.

        A piecewise-linear fit to the usual LiPo curve: a steep drop over the
        first few per cent, a long flat plateau, then a cliff below about 15
        per cent. The flat middle is why terminal voltage says so little about
        remaining charge, and why endurance is tracked by coulomb counting
        here rather than inferred from volts.
        """
        s = min(max(state_of_charge, 0.0), 1.0)
        if s > 0.90:
            per_cell = 4.20 - (1.0 - s) * 2.0
        elif s > 0.15:
            per_cell = 4.00 - (0.90 - s) * 0.40
        else:
            per_cell = 3.70 - (0.15 - s) * 2.67
        return self.cells * per_cell

    def terminal_voltage(self, current: float, state_of_charge: float = 1.0) -> float:
        return self.open_circuit_voltage(state_of_charge) - current * self.resistance_ohm

    def endurance_s(self, current_a: float, usable_fraction: float = 0.80,
                    dt: float = 1.0, max_s: float = 7200.0) -> float:
        """Time to empty at a constant current draw.

        `usable_fraction` defaults to 0.8 because discharging a LiPo to zero
        destroys it; the practical reserve is what pilots actually fly to.

        Coulomb-counted rather than divided out, so the sag cutoff can end the
        flight early when internal resistance -- not capacity -- is the binding
        constraint. That is the case a simple Ah/A division silently misses.
        """
        if current_a <= 0:
            raise ValueError("current must be positive")
        usable_ah = self.capacity_ah * usable_fraction
        drawn_ah = 0.0
        t = 0.0
        while t < max_s:
            soc = 1.0 - drawn_ah / self.capacity_ah
            if soc <= 1.0 - usable_fraction:
                break
            if self.terminal_voltage(current_a, soc) <= self.cutoff_voltage:
                break
            drawn_ah += current_a * dt / 3600.0
            t += dt
        return t

    def sag_limited(self, current_a: float) -> bool:
        """True if the pack browns out before its capacity is used."""
        by_capacity = self.capacity_ah * 0.80 / current_a * 3600.0
        return self.endurance_s(current_a) < by_capacity * 0.98
