"""Flight control: what a user programs, and the cascade underneath it.

The layering is the usual one, from the outside in:

    position/velocity -> attitude -> body rates -> moments -> mixer -> motors

A user controller can plug in at any level. `Controller.update` returns motor
commands directly, so the bottom of the stack is always available; the classes
here are there so nobody has to write a quaternion attitude loop to find out
whether their race line works.

Gains are stated as *accelerations*, and multiplied by the vehicle's own mass
and inertia tensor before they become forces. That is what lets one set of
gains fly a 390 g racer and a 2 kg lifter: the designed vehicle is known
exactly, so there is no reason to make the user re-tune for it.
"""

from __future__ import annotations

import numpy as np

from .motors import MotorBank

G = 9.80665


class Mixer:
    """Desired collective thrust and body moments -> per-rotor thrust.

    Built by inverting the allocation matrix -- Camber's, the one the optimiser
    designed against -- so a vehicle with unusual rotor placement or tilted
    rotors mixes correctly without a hand-written table. Rows used: Fz, Mx, My,
    Mz. Fx and Fy are not commandable on a planar multirotor (rank 4 of 6); the
    vehicle tilts to translate, which is the attitude loop's job.

    When the request is out of reach it gives up yaw first, then collective, and
    holds roll and pitch longest -- losing roll authority is how a quad flips,
    losing a little yaw is how it drifts.
    """

    def __init__(self, allocation: np.ndarray, max_thrust: np.ndarray) -> None:
        self.B = np.asarray(allocation)[[2, 3, 4, 5], :]
        self.pinv = np.linalg.pinv(self.B)
        self.max_thrust = np.asarray(max_thrust, dtype=float)

    def thrusts(self, fz: float, moment: np.ndarray) -> np.ndarray:
        m = np.asarray(moment, dtype=float)
        base = self.pinv @ np.array([fz, 0.0, 0.0, 0.0])
        rp = self.pinv @ np.array([0.0, m[0], m[1], 0.0])
        yaw = self.pinv @ np.array([0.0, 0.0, 0.0, m[2]])
        lo, hi = np.zeros_like(self.max_thrust), self.max_thrust

        def fits(t):
            return (t >= lo - 1e-9).all() and (t <= hi + 1e-9).all()

        t = base + rp + yaw
        if fits(t):
            return t
        for k in np.linspace(1.0, 0.0, 11):             # shed yaw
            t = base + rp + k * yaw
            if fits(t):
                return t
        t = base + rp                                   # re-centre collective
        shift = 0.0
        if (t < lo).any():
            shift = float(np.max(lo - t))
        elif (t > hi).any():
            shift = -float(np.max(t - hi))
        unit = self.pinv @ np.array([1.0, 0.0, 0.0, 0.0])
        unit = unit / max(float(np.mean(unit)), 1e-12)
        t = t + shift * unit
        if not fits(t):                                  # last resort: scale rp
            for k in np.linspace(1.0, 0.0, 21):
                t = base + k * rp
                if fits(t):
                    break
        return np.clip(t, lo, hi)

    def commands(self, fz: float, moment: np.ndarray) -> np.ndarray:
        return MotorBank.command_for(self.thrusts(fz, moment), self.max_thrust)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


class Controller:
    """Subclass and override `update`. Return motor commands in [0, 1]."""

    def reset(self, sim) -> None:
        """Called once before flight with the Simulation, for its vehicle data."""

    def update(self, state, sim) -> np.ndarray:
        raise NotImplementedError


class Cascade(Controller):
    """Position -> attitude -> rate control, feeding the allocation mixer.

    Set `target` (position, world) and optionally `velocity` and `yaw`; or call
    `attitude()`/`rates()` directly from a subclass to fly acro.
    """

    def __init__(self, *, pos_gain=2.2, vel_gain=3.2, att_gain=9.0,
                 rate_gain=22.0, rate_int=6.0, max_tilt=np.radians(65),
                 max_rate=np.radians(720)) -> None:
        self.kp, self.kv = pos_gain, vel_gain
        self.katt, self.krate, self.kint = att_gain, rate_gain, rate_int
        self.max_tilt, self.max_rate = max_tilt, max_rate
        self.target = np.zeros(3)
        self.velocity = np.zeros(3)
        self.accel = np.zeros(3)
        self.yaw = 0.0

    def reset(self, sim) -> None:
        self.mixer = Mixer(sim.allocation, sim.max_thrust)
        self.mass, self.inertia = sim.mass, sim.inertia
        self.hover = sim.weight
        self._integral = np.zeros(3)
        self.target = sim.state().position.copy()

    # --- the three loops, usable on their own ---------------------------------
    def rates(self, state, rate_sp: np.ndarray, fz: float, dt: float) -> np.ndarray:
        err = np.asarray(rate_sp) - state.rates
        self._integral = np.clip(self._integral + err * dt, -0.5, 0.5)
        alpha = self.krate * err + self.kint * self._integral
        w = state.rates
        # Gyroscopic term, so the rate loop does not have to discover it.
        moment = self.inertia @ alpha + np.cross(w, self.inertia @ w)
        return self.mixer.commands(fz, moment)

    def attitude(self, state, q_sp: np.ndarray, fz: float, dt: float) -> np.ndarray:
        q_err = _quat_mul(_quat_conj(state.quaternion), q_sp)
        if q_err[0] < 0:
            q_err = -q_err                             # shortest way round
        rate_sp = np.clip(2.0 * self.katt * q_err[1:], -self.max_rate, self.max_rate)
        return self.rates(state, rate_sp, fz, dt)

    def update(self, state, sim) -> np.ndarray:
        a = (self.kp * (self.target - state.position)
             + self.kv * (self.velocity - state.velocity) + self.accel)
        f = self.mass * (a + np.array([0.0, 0.0, G]))
        # Tilt limit: keep enough of the thrust vector vertical to hold height.
        horiz, vert = np.linalg.norm(f[:2]), max(f[2], 0.2 * self.hover)
        if horiz > vert * np.tan(self.max_tilt):
            f[:2] *= vert * np.tan(self.max_tilt) / horiz
        f[2] = vert
        z_b = f / np.linalg.norm(f)
        x_c = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        q_sp = quat_from_matrix(np.column_stack([x_b, y_b, z_b]))
        fz = float(f @ state.rotation[:, 2])           # thrust along body z
        return self.attitude(state, q_sp, max(fz, 0.0), sim.dt)


class Hover(Cascade):
    """Climb to `height` above the start and hold there."""

    def __init__(self, height: float = 1.5, **gains) -> None:
        super().__init__(**gains)
        self.height = height

    def reset(self, sim) -> None:
        super().reset(sim)
        self.target = self.target + np.array([0.0, 0.0, self.height])


class GateRacer(Cascade):
    """Fly a course: through each gate, in order, as fast as `speed` allows.

    Aims through a point a little past the gate centre with a velocity
    feed-forward along the gate normal, so it crosses square rather than
    clipping the frame on the way to the next one.
    """

    def __init__(self, course, speed: float = 8.0, **gains) -> None:
        super().__init__(**gains)
        self.course, self.speed = course, speed

    def reset(self, sim) -> None:
        super().reset(sim)
        self.gate = 0
        self.passed = []

    def update(self, state, sim) -> np.ndarray:
        if self.gate < len(self.course.gates):
            g = self.course.gates[self.gate]
            if g.crossed(state.position):
                self.passed.append(sim.t)
                self.gate += 1
        if self.gate >= len(self.course.gates):
            self.target, self.velocity = self.course.finish, np.zeros(3)
            self.accel = np.zeros(3)
            return super().update(state, sim)
        g = self.course.gates[self.gate]
        to_gate = g.centre - state.position
        dist = float(np.linalg.norm(to_gate))
        heading = to_gate / max(dist, 1e-6)
        # Blend toward the gate normal as it gets close, to cross it square.
        w = np.clip(1.0 - dist / 4.0, 0.0, 1.0)
        direction = (1 - w) * heading + w * g.normal
        direction /= np.linalg.norm(direction)
        self.velocity = self.speed * direction
        self.target = g.centre + 0.5 * g.normal
        self.accel = np.zeros(3)
        self.yaw = float(np.arctan2(direction[1], direction[0]))
        self.kp_eff = self.kp
        return super().update(state, sim)
