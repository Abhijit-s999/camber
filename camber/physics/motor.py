"""Brushless DC motor, as the first-order model the datasheets support.

Three parameters, all of which manufacturers publish or which a thrust-stand
run recovers: the speed constant Kv, the winding resistance Rm, and the no-load
current I0. That is deliberately the whole model -- iron loss, saturation and
temperature dependence would need data nobody supplies for hobby motors, and
inventing coefficients for them would violate AI-06 while looking rigorous.

    Q      = Kt (I - I0)              torque, less what spinning itself costs
    V_term = I Rm + omega / Kv_rad    applied volts, less the back-EMF
    Kt     = 1 / Kv_rad               in SI these are the same number

The last identity is worth stating because Kv is quoted in RPM per volt and Kt
in newton-metres per amp, which look unrelated until both are in SI. Treating
them as independent inputs is a common way to build a motor that violates
conservation of energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

#: RPM per volt -> rad/s per volt.
RPM_V_TO_RAD_S_V = 2.0 * math.pi / 60.0


@dataclass(frozen=True)
class Motor:
    name: str
    kv_rpm_v: float
    resistance_ohm: float
    no_load_current_a: float
    mass_kg: float = 0.0
    max_current_a: float | None = None
    max_voltage_v: float | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if self.kv_rpm_v <= 0:
            raise ValueError(f"{self.name}: Kv must be positive")
        if self.resistance_ohm <= 0:
            raise ValueError(f"{self.name}: winding resistance must be positive")

    @property
    def kv_rad(self) -> float:
        """Speed constant in rad/s per volt."""
        return self.kv_rpm_v * RPM_V_TO_RAD_S_V

    @property
    def kt(self) -> float:
        """Torque constant, N m per amp. Numerically 1 / kv_rad in SI."""
        return 1.0 / self.kv_rad

    def current_for(self, omega: float, voltage: float) -> float:
        """Current drawn at a given shaft speed and applied voltage."""
        return (voltage - omega / self.kv_rad) / self.resistance_ohm

    def torque(self, omega: float, voltage: float) -> float:
        return self.kt * (self.current_for(omega, voltage) - self.no_load_current_a)

    def operating_point(self, omega: float, voltage: float) -> "MotorState":
        current = self.current_for(omega, voltage)
        torque = self.kt * (current - self.no_load_current_a)
        p_in = voltage * current
        p_out = torque * omega
        return MotorState(
            omega=omega, rpm=omega * 60.0 / (2.0 * math.pi), voltage=voltage,
            current=current, torque=torque, power_in=p_in, power_out=p_out,
            efficiency=(p_out / p_in) if p_in > 1e-9 else 0.0,
        )


@dataclass(frozen=True)
class MotorState:
    omega: float
    rpm: float
    voltage: float
    current: float
    torque: float
    power_in: float
    power_out: float
    efficiency: float


def match_speed(motor: Motor, voltage: float, load_torque, omega_max_guess: float | None = None) -> float:
    """Find the shaft speed where motor torque equals the load's.

    `load_torque` is a callable omega -> N m, which is where the propeller
    enters. Solved by bracketing rather than iteration because the two curves
    cross exactly once: motor torque falls linearly with speed while propeller
    torque rises roughly as its square, so the residual is monotonic and a
    bracketed solve cannot wander off.
    """
    omega_noload = voltage * motor.kv_rad
    hi = omega_max_guess or omega_noload

    def residual(omega: float) -> float:
        return motor.torque(omega, voltage) - load_torque(omega)

    lo = 1e-3
    if residual(lo) <= 0.0:
        raise ValueError(
            "motor cannot turn this load at all at this voltage: stall torque "
            "is below the propeller's torque at rest"
        )
    # At no-load speed the motor makes no torque, so the residual is negative
    # there unless the load is negligible; walk out a little if not.
    for _ in range(6):
        if residual(hi) < 0.0:
            break
        hi *= 1.5
    else:
        raise ValueError("no crossing found below 11x no-load speed")

    return brentq(residual, lo, hi, xtol=1e-6)
