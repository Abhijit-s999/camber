"""Lifting and control surfaces: what makes a plane a plane.

A multirotor's body aerodynamics are drag -- a force that only ever opposes
motion. A wing is different: it turns forward speed into a force across the
direction of flight, and that force is what holds the aircraft up and lets it
turn. A control surface is a hinged trailing part of a wing whose deflection
changes that force on command: an elevator to pitch, ailerons to roll, a
rudder to yaw.

The model here is a strip model, one panel per surface, the same idea flight
simulators have used for decades:

* The air a surface meets is the body's velocity plus the spin carried to
  where the surface sits, ``v + omega x r`` -- a tail well behind the centre of
  mass feels the pitch rate as a change in angle of attack, which is most of
  what damps a plane's pitch. Ignoring it makes an aircraft that porpoises.
* In the plane across the span, the angle of attack is the angle between the
  chord and the local flow. Lift acts across the flow, drag along it.
* Lift grows linearly with angle of attack until the wing stalls, then
  collapses. A model without stall lets a controller pull as hard as it likes,
  and no real wing does.
* A control surface adds camber: its deflection shifts the lift curve, so the
  same airflow makes more or less force.

Coefficients are per surface and have aircraft-typical defaults; none is a
constant buried in the maths. What this does *not* model: compressibility,
Reynolds effects, wing-body interference, and the downwash a wing sheds onto
the tail. Those matter to a real design and belong in the CFD tier, not in the
loop that flies the plane a thousand times a second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

RHO = 1.225


@dataclass
class Surface:
    """One lifting panel, with an optional hinged control fraction.

    Geometry is in body FLU axes (x forward, y left, z up):

    * ``centre`` -- the aerodynamic centre, roughly quarter-chord, in metres.
    * ``area`` -- planform area, m^2.
    * ``span_axis`` -- along the hinge line; the flow component along it does no
      aerodynamic work. Body +y (left) for a main wing or tailplane, +z (up)
      for a fin.
    * ``normal`` -- the way positive lift points at positive angle of attack.
      Body +z (up) for a wing, +y (left) for a fin.

    The chord direction is ``normal x span_axis``, pointing forward.
    """

    name: str
    centre: np.ndarray
    area: float
    span_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    chord: float = 0.2
    #: Lift-curve slope, per radian. Thin-aerofoil theory gives 2*pi; real
    #: finite wings sit lower, near 5, because the tips leak.
    cl_alpha: float = 5.0
    #: Angle of attack of zero lift, radians. A cambered wing lifts at zero
    #: incidence, so this is slightly negative; a flat fin is zero.
    alpha_0: float = 0.0
    #: Stall angle, radians. Beyond it lift falls instead of rising.
    stall: float = math.radians(15.0)
    #: Profile drag at zero lift, and the induced-drag factor k in CD = CD0 + k CL^2.
    cd0: float = 0.02
    k: float = 0.06
    #: Lift gained per radian of control deflection. Zero for a fixed surface.
    cl_delta: float = 0.0
    #: Which command channel drives this surface, and with what sign. Two
    #: ailerons share a channel with opposite signs; the two halves of an
    #: elevator share one with the same sign.
    channel: int | None = None
    channel_sign: float = 1.0

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=float)
        self.span_axis = _unit(self.span_axis)
        self.normal = _unit(self.normal)
        if self.area <= 0.0:
            raise ValueError(f"surface {self.name!r} needs a positive area")
        # Chord points forward, completing a right-handed set with normal and
        # span. For a wing (normal +z, span +y) this is +x, forward.
        self.chord_dir = _unit(np.cross(self.normal, self.span_axis))

    def coefficients(self, alpha: float, deflection: float) -> tuple[float, float]:
        """Lift and drag coefficients at this angle of attack and deflection.

        Past the stall angle the lift curve is folded back down to a flat-plate
        value, so a controller cannot buy unlimited force by pulling harder.
        """
        a = alpha - self.alpha_0 + self.cl_delta / self.cl_alpha * deflection
        if abs(a) <= self.stall:
            cl = self.cl_alpha * a
        else:
            # Beyond stall, blend to the flat-plate curve 2 sin a cos a, which
            # peaks near 45 degrees and is what a stalled surface really does.
            peak = self.cl_alpha * self.stall
            flat = math.copysign(2.0 * abs(math.sin(a)) * math.cos(a), a)
            drop = math.exp(-(abs(a) - self.stall) / math.radians(10.0))
            cl = math.copysign(peak, a) * drop + flat * (1.0 - drop)
        cd = self.cd0 + self.k * cl * cl
        return cl, cd

    def wrench(self, v_body: np.ndarray, omega: np.ndarray, deflection: float,
               rho: float = RHO) -> tuple[np.ndarray, np.ndarray]:
        """Force and moment about the origin, body FLU, at this flight state.

        ``v_body`` is the air-relative velocity of the body origin; ``omega``
        the body angular rate. The surface sees the sum carried to where it is.
        """
        v = np.asarray(v_body, float) + np.cross(np.asarray(omega, float), self.centre)
        # Work in the plane across the span: drop the spanwise component.
        v_plane = v - np.dot(v, self.span_axis) * self.span_axis
        speed = float(np.linalg.norm(v_plane))
        if speed < 1e-6:
            return np.zeros(3), np.zeros(3)
        along = v_plane / speed
        fwd = float(np.dot(v_plane, self.chord_dir))
        up = float(np.dot(v_plane, self.normal))
        alpha = math.atan2(-up, abs(fwd)) if fwd != 0.0 else math.copysign(math.pi / 2, -up)
        cl, cd = self.coefficients(alpha, deflection)
        q = 0.5 * rho * speed * speed * self.area
        # Lift is perpendicular to the flow, in the chord-normal plane, on the
        # side the normal points; drag is straight downstream.
        lift_dir = _unit(np.cross(self.span_axis, along))
        if np.dot(lift_dir, self.normal) < 0.0:
            lift_dir = -lift_dir
        force = q * (cl * lift_dir - cd * along)
        moment = np.cross(self.centre, force)
        return force, moment


@dataclass
class SurfaceSet:
    """A plane's whole set of surfaces, summed. Drop-in for the sim's aero.

    Holds the current control deflections (radians per channel), which the
    flight controller sets each step. ``wrench`` is the same shape the body
    aero returns, so a plane's lift and a multirotor's drag go through one path.
    """

    surfaces: list[Surface]
    body_drag: object | None = None
    max_deflection: float = math.radians(25.0)
    rho: float = RHO

    def __post_init__(self) -> None:
        self.n_channels = 1 + max((s.channel for s in self.surfaces
                                   if s.channel is not None), default=-1)
        self.deflections = np.zeros(self.n_channels)
        self.extrapolated = 0

    def set_controls(self, channels) -> None:
        """Commanded deflections in [-1, 1] per channel, scaled to radians."""
        c = np.clip(np.asarray(channels, float), -1.0, 1.0)
        n = min(len(c), self.n_channels)
        self.deflections[:n] = c[:n] * self.max_deflection

    def wrench(self, v_body: np.ndarray, omega: np.ndarray,
               ) -> tuple[np.ndarray, np.ndarray]:
        f_total = np.zeros(3)
        m_total = np.zeros(3)
        for s in self.surfaces:
            d = (self.deflections[s.channel] * s.channel_sign
                 if s.channel is not None else 0.0)
            f, m = s.wrench(v_body, omega, d, self.rho)
            f_total += f
            m_total += m
        if self.body_drag is not None:
            fb, mb = self.body_drag.force(v_body)
            f_total += fb
            m_total += mb
        return f_total, m_total

    # A plane's body still has parasitic drag; when no separate body-drag model
    # is given, `wrench` is all the aero there is. The old `force(v)` interface
    # is kept so a SurfaceSet can stand in for body aero at zero rate.
    def force(self, v_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.wrench(np.asarray(v_body, float), np.zeros(3))


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("direction vector has zero length")
    return v / n
