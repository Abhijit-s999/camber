"""Motor, propeller and pack solved together as one system.

The propeller is not consulted directly here. Each blade is first reduced to a
**map** -- CT and CP tabulated against advance ratio -- because the coupled
motor-propeller match is a root-find that evaluates propeller torque twenty or
thirty times, and BEMT costs milliseconds. Calling it inside the loop makes one
operating point cost 100 ms, and an optimiser needs thousands of them.

Tabulating once and interpolating costs microseconds and is not an
approximation in any meaningful sense: CT and CP are smooth in J, and the
sampling is finer than the wind-tunnel data the model is validated against.
This is the same move the fidelity ladder makes at a larger scale (ARCH-01) --
pay for physics once, then search a cheap surrogate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .battery import Pack
from .bemt import RHO_AIR, BladeDefinition, analyse
from .motor import Motor, match_speed


@dataclass
class PropellerMap:
    """CT and CP against advance ratio, for one propeller."""

    diameter_m: float
    advance_ratios: np.ndarray
    ct: np.ndarray
    cp: np.ndarray
    rpm_sampled: float
    name: str = ""

    @classmethod
    def from_blade(cls, blade: BladeDefinition, rpm: float = 5000.0,
                   j_max: float = 1.2, n: int = 30, polar=None,
                   name: str = "") -> "PropellerMap":
        diameter = 2.0 * blade.radius_m
        rev_s = rpm / 60.0
        js = np.linspace(0.0, j_max, n)
        ct, cp = np.zeros(n), np.zeros(n)
        for i, j in enumerate(js):
            if j < 1e-6:
                # The J = 0 column is filled by extrapolation below: the
                # induction factor is defined against the freestream, so the
                # formulation degenerates exactly at static.
                ct[i], cp[i] = np.nan, np.nan
                continue
            r = analyse(blade, j * rev_s * diameter, rpm, polar)
            ct[i] = r.ct if r.converged else np.nan
            cp[i] = r.cp if r.converged else np.nan

        ct, cp = cls._fill(js, ct), cls._fill(js, cp)
        return cls(diameter, js, ct, cp, rpm, name)

    @staticmethod
    def _fill(js: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Interpolate over failed points; extrapolate the static end linearly."""
        good = np.isfinite(values)
        if good.sum() < 2:
            raise ValueError("propeller map has too few converged points")
        out = np.interp(js, js[good], values[good])
        first = int(np.argmax(good))
        if first > 0:
            j0, j1 = js[first], js[first + 1]
            v0, v1 = values[first], values[first + 1]
            slope = (v1 - v0) / (j1 - j0)
            out[:first] = np.maximum(v0 + slope * (js[:first] - j0), 1e-4)
        return out

    def coefficients(self, advance_ratio: float) -> tuple[float, float]:
        j = float(np.clip(advance_ratio, self.advance_ratios[0],
                          self.advance_ratios[-1]))
        return (float(np.interp(j, self.advance_ratios, self.ct)),
                float(np.interp(j, self.advance_ratios, self.cp)))

    def thrust_torque(self, omega: float, v_inf: float,
                      rho: float = RHO_AIR) -> tuple[float, float]:
        """Thrust (N) and shaft torque (N m) at a shaft speed and airspeed."""
        if omega <= 1e-6:
            return 0.0, 0.0
        n = omega / (2.0 * math.pi)
        j = v_inf / (n * self.diameter_m)
        ct, cp = self.coefficients(j)
        thrust = ct * rho * n ** 2 * self.diameter_m ** 4
        power = cp * rho * n ** 3 * self.diameter_m ** 5
        return thrust, power / omega


@dataclass
class MeasuredPropeller:
    """A propeller as the wind tunnel measured it, not as a model predicts it.

    Static CT and CP against rpm, and where the archive has them, sweeps of
    advance ratio at fixed rpm. For a propeller someone can buy, this is better
    than any blade-element prediction of it: it is the thing itself, Reynolds
    effects included (PHYS-04 is why that matters at this size).

    Outside the measured rpm range the end values are held, not extrapolated.
    The archive's small propellers were measured to about 8000 rpm and a racing
    quad spins them at 25,000 or more; CT and CP still rise slowly with
    Reynolds number up there, so holding them under-reads thrust a little, and
    tip compressibility, which the tunnel never reached, works the other way
    (CAT-02).
    """

    diameter_m: float
    static_rpm: np.ndarray
    static_ct: np.ndarray
    static_cp: np.ndarray
    #: (rpm, J, CT, CP) per wind-tunnel sweep, each J ascending.
    runs: tuple = ()
    name: str = ""

    def coefficients(self, rpm: float, advance_ratio: float = 0.0) -> tuple[float, float]:
        r = float(np.clip(rpm, self.static_rpm[0], self.static_rpm[-1]))
        ct0 = float(np.interp(r, self.static_rpm, self.static_ct))
        cp0 = float(np.interp(r, self.static_rpm, self.static_cp))
        if advance_ratio <= 1e-9 or not self.runs:
            return ct0, cp0
        # The sweep nearest in rpm, anchored at J = 0 by the static point at
        # this rpm: the tunnel sweeps start at J of 0.1 or more.
        _, js, cts, cps = min(self.runs, key=lambda run: abs(run[0] - rpm))
        js = np.concatenate([[0.0], js])
        j = float(np.clip(advance_ratio, 0.0, js[-1]))
        return (float(np.interp(j, js, np.concatenate([[ct0], cts]))),
                float(np.interp(j, js, np.concatenate([[cp0], cps]))))

    def thrust_torque(self, omega: float, v_inf: float,
                      rho: float = RHO_AIR) -> tuple[float, float]:
        """Thrust (N) and shaft torque (N m) at a shaft speed and airspeed."""
        if omega <= 1e-6:
            return 0.0, 0.0
        n = omega / (2.0 * math.pi)
        ct, cp = self.coefficients(n * 60.0, v_inf / (n * self.diameter_m))
        thrust = ct * rho * n ** 2 * self.diameter_m ** 4
        power = cp * rho * n ** 3 * self.diameter_m ** 5
        return thrust, power / omega


@dataclass
class OperatingPoint:
    rpm: float
    thrust_n: float
    torque_nm: float
    shaft_power_w: float
    electrical_power_w: float
    current_a: float
    voltage_v: float
    motor_efficiency: float
    system_efficiency: float
    advance_ratio: float

    @property
    def thrust_per_watt(self) -> float:
        """Newtons per electrical watt. The figure of merit for endurance."""
        return self.thrust_n / self.electrical_power_w if self.electrical_power_w > 0 else 0.0

    def supply_current(self, pack_voltage: float) -> float:
        """What the pack delivers for this point, which is not the motor's current.

        Throttle is the ESC switching the full pack voltage on and off, so the
        motor sees `voltage_v` on average while the pack sees its own voltage.
        Power is what passes through: at half throttle the motor draws twice
        the pack's current. Counting motor current as pack current halved the
        hover endurance of every design (PHYS-10). The ESC is taken as
        lossless; its few per cent are one of the losses still missing.
        """
        return self.electrical_power_w / pack_voltage if pack_voltage > 0 else 0.0


def solve_operating_point(motor: Motor, prop: PropellerMap, voltage: float,
                          v_inf: float = 0.0, rho: float = RHO_AIR) -> OperatingPoint:
    """Where the motor and propeller settle at a given applied voltage."""
    omega = match_speed(motor, voltage,
                        lambda w: prop.thrust_torque(w, v_inf, rho)[1])
    thrust, torque = prop.thrust_torque(omega, v_inf, rho)
    state = motor.operating_point(omega, voltage)
    p_elec = state.power_in
    n = omega / (2.0 * math.pi)
    return OperatingPoint(
        rpm=omega * 60.0 / (2.0 * math.pi),
        thrust_n=thrust,
        torque_nm=torque,
        shaft_power_w=torque * omega,
        electrical_power_w=p_elec,
        current_a=state.current,
        voltage_v=voltage,
        motor_efficiency=state.efficiency,
        system_efficiency=(thrust * v_inf / p_elec) if (p_elec > 0 and v_inf > 0) else 0.0,
        advance_ratio=v_inf / (n * prop.diameter_m) if n > 0 else 0.0,
    )


def thrust_for(motor: Motor, prop: PropellerMap, pack: Pack, target_thrust_n: float,
               v_inf: float = 0.0, soc: float = 1.0,
               rho: float = RHO_AIR) -> OperatingPoint:
    """Find the throttle that produces a required thrust, and what it costs.

    This is the question hover actually asks: not "what does full throttle
    give" but "what does holding this aircraft up draw from the pack".
    """
    from scipy.optimize import brentq

    def excess(throttle: float) -> float:
        v = pack.terminal_voltage(0.0, soc) * throttle
        try:
            return solve_operating_point(motor, prop, v, v_inf, rho).thrust_n - target_thrust_n
        except ValueError:
            return -target_thrust_n

    lo, hi = 0.05, 1.0
    if excess(hi) < 0:
        raise ValueError(
            f"full throttle gives less than the required {target_thrust_n:.2f} N"
        )
    throttle = brentq(excess, lo, hi, xtol=1e-4)
    v = pack.terminal_voltage(0.0, soc) * throttle
    return solve_operating_point(motor, prop, v, v_inf, rho)
