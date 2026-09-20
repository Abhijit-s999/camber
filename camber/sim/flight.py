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

    def __init__(self, *, pos_gain=2.2, vel_gain=3.2, vel_int=1.6, att_gain=9.0,
                 rate_gain=22.0, rate_int=40.0, max_tilt=np.radians(65),
                 max_rate=np.radians(720), vertical=(2.0, 1.6, 4.0)) -> None:
        # Gains per axis, (x, y, z). Height gets its own, stiffer set -- as it
        # does on PX4 and Betaflight -- because the disturbance it has to beat
        # is the one that grows fastest: aerodynamic down-load rising with the
        # square of speed as the vehicle accelerates. `vertical` multiplies
        # (position, velocity, integral) gain on z.
        kz = np.asarray(vertical, dtype=float)
        self.kp = pos_gain * np.array([1.0, 1.0, kz[0]])
        self.kv = vel_gain * np.array([1.0, 1.0, kz[1]])
        self.ki = vel_int * np.array([1.0, 1.0, kz[2]])
        # rate_int was 6, a quarter of the proportional gain. Measured on the
        # racer flying flat out on its CFD table: at 6 the integrator needed
        # seven seconds to wind up enough to hold the aerodynamic pitching
        # moment, which builds in one and a half. Real flight controllers run
        # rate I comparable to P.
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
        self._vint = np.zeros(3)
        # The rate integrator is what holds off a *steady* moment -- an
        # aerodynamic pitching moment at speed, a CG off the thrust centre.
        # Its clamp was a constant 0.5, which on the racer caps the moment it
        # can hold at 0.005 N m; the CFD table's nose-down moment at 36 m/s is
        # 0.14 N m, so proportional action alone had to carry it, at the cost
        # of a 25-degree attitude error -- tilt ran from the 65-degree limit to
        # 82 and the vehicle flew into the ground. The clamp is now sized from
        # this vehicle's own control authority: 30% of the moment it can make
        # about each axis, whatever the airframe.
        B = np.asarray(sim.allocation)[3:, :]
        authority = 0.5 * np.abs(B) @ np.asarray(sim.max_thrust)        # N m per axis
        inertia = np.maximum(np.diag(self.inertia), 1e-9)
        self._int_clamp = 0.3 * authority / (inertia * max(self.kint, 1e-9))
        self.target = sim.state().position.copy()

    # --- the three loops, usable on their own ---------------------------------
    def rates(self, state, rate_sp: np.ndarray, fz: float, dt: float) -> np.ndarray:
        err = np.asarray(rate_sp) - state.rates
        self._integral = np.clip(self._integral + err * dt,
                                 -self._int_clamp, self._int_clamp)
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
        # Integral action on the velocity loop, which is what rejects a steady
        # disturbance. Without it the loop can only push back in proportion to
        # how far it has already been pushed: flat out at the tilt limit, the
        # airframe's aerodynamic down-load (7 N at 38 m/s on the racer) needed
        # about 7 m of altitude error to balance, and the vehicle flew into the
        # ground from 3 m. PX4 and every real flight controller integrate here
        # for the same reason.
        err = (self.kp / self.kv) * (self.target - state.position) \
            + (self.velocity - state.velocity)
        a = self.kv * err + self.ki * self._vint + self.accel
        f = self.mass * (a + np.array([0.0, 0.0, G]))
        # Tilt limit: keep enough of the thrust vector vertical to hold height.
        horiz, vert = np.linalg.norm(f[:2]), max(f[2], 0.2 * self.hover)
        limited = horiz > vert * np.tan(self.max_tilt)
        if limited:
            f[:2] *= vert * np.tan(self.max_tilt) / horiz
        f[2] = vert
        # Anti-windup: height always integrates; the horizontal axes only
        # while the tilt limit is not already the thing holding them back,
        # or the integral winds up against a limit and overshoots on release.
        step = err * sim.dt
        if limited:
            step[:2] = 0.0
        self._vint = np.clip(self._vint + step, -8.0, 8.0)
        z_b = f / np.linalg.norm(f)
        x_c = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        q_sp = quat_from_matrix(np.column_stack([x_b, y_b, z_b]))
        # Collective is the desired force along the body's current z -- but
        # never less than 40% of it. The plain projection goes to zero when the
        # attitude has to swing more than 90 degrees, as it does from 65 degrees
        # of bank one way to 65 the other after a hairpin, and at zero
        # collective the rotors cannot make a moment either (they cannot spin
        # backwards): the vehicle lost thrust and attitude authority together
        # for half a second and fell 1.2 m into the next gate's approach.
        fz = max(float(f @ state.rotation[:, 2]), 0.4 * float(np.linalg.norm(f)))
        return self.attitude(state, q_sp, fz, sim.dt)


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

    def __init__(self, course, speed: float = 8.0, grip: float = 0.7, **gains) -> None:
        super().__init__(**gains)
        self.course, self.speed = course, speed
        # Sideways acceleration it will commit to in a corner, as a fraction
        # of what the tilt limit allows. A multirotor turns by tilting, so the
        # most it can pull is g tan(max_tilt); planning to use all of it leaves
        # nothing for the height loop, which is what put it into the floor.
        self.a_lat = grip * G * np.tan(self.max_tilt)

    def corner_speed(self, i: int) -> float:
        """Fastest it can pass gate i: sqrt(a R), R the radius of the circle
        through the gates either side. Without this the pilot flew every gate
        at the set speed, and the oval's hairpins -- 3 m radius -- set the lap
        at 8 s whether the set speed was 9 m/s or 14, and crashed it at 18."""
        gs = self.course.gates
        n = len(gs)
        a, b, c = gs[(i - 1) % n].centre, gs[i].centre, gs[(i + 1) % n].centre
        ab, bc, ca = (np.linalg.norm(b - a), np.linalg.norm(c - b),
                      np.linalg.norm(a - c))
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        if area < 1e-9:
            return self.speed
        radius = ab * bc * ca / (4.0 * area)
        return float(min(self.speed, np.sqrt(self.a_lat * radius)))

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
        # Brake in time: the fastest speed from which the corner speed can
        # still be reached over the distance left, at the same grip.
        # Braking is planned at half the cornering grip and with the velocity
        # loop's lag taken off the distance: the loop needs about 1/kv seconds
        # to act on a new setpoint, and planning as if it were instant had the
        # vehicle arrive at hairpins at 25 m/s and miss them.
        v_gate = self.corner_speed(self.gate)
        now = float(np.linalg.norm(state.velocity))
        room = max(0.0, dist - now / float(self.kv[0]))
        v = min(self.speed, float(np.sqrt(v_gate ** 2 + self.a_lat * room)))
        self.velocity = v * direction
        # Correct only the cross-track error. Aiming the position loop at a
        # point past the gate turned the whole remaining distance into a
        # demand for more speed: it flew 13.2 m/s against a 9 m/s setpoint and
        # tilted to 78 degrees against a 65-degree limit. Along the line, speed
        # is the velocity setpoint's job; the position loop only keeps the
        # vehicle *on* the line.
        along = float(np.dot(state.position - g.centre, direction))
        self.target = g.centre + along * direction
        # The line through a gate is at the gate's height. Projected along a
        # 3D direction it dips whenever the vehicle is above the gate, and
        # with a stiff height loop the vehicle followed it into the floor.
        self.target[2] = g.centre[2]
        self.accel = np.zeros(3)
        self.yaw = float(np.arctan2(direction[1], direction[0]))
        return super().update(state, sim)


class FixedWing:
    """An autopilot for an aeroplane: it flies on wings, not on vectored thrust.

    A multirotor points its whole thrust where it wants to go. A plane cannot;
    it holds height with the elevator, turns by banking with the ailerons, and
    holds speed with the throttle, and each of those is a slower loop than a
    quad's because the force it commands only exists while the aircraft has
    airspeed. The loops here are the standard cascade:

      speed  -> throttle
      height -> a target pitch attitude -> elevator
      heading-> a target bank angle    -> ailerons, with rudder to coordinate

    Attitude is flown in Euler angles, which is honest for an aircraft that
    stays the right way up and would be the wrong choice for aerobatics.
    Returns a `Command`: one motor channel and three surface channels
    (elevator, aileron, rudder), matching the plane built in the examples.
    """

    def __init__(self, *, speed: float = 18.0, altitude: float = 50.0,
                 max_bank=np.radians(35.0), max_pitch=np.radians(15.0),
                 k_speed=0.5, k_alt=0.03, k_pitch=2.2, k_q=0.3,
                 k_hdg=1.2, k_roll=2.5, k_p=0.25, k_yaw=0.5) -> None:
        self.speed, self.altitude = speed, altitude
        self.heading = 0.0
        self.max_bank, self.max_pitch = max_bank, max_pitch
        self.k_speed, self.k_alt, self.k_pitch, self.k_q = k_speed, k_alt, k_pitch, k_q
        self.k_hdg, self.k_roll, self.k_p, self.k_yaw = k_hdg, k_roll, k_p, k_yaw

    def reset(self, sim) -> None:
        from .motors import MotorBank
        self.max_thrust = sim.max_thrust
        # Throttle that holds the target speed in level flight, as a starting
        # point the speed loop trims around: roughly enough thrust to beat the
        # drag at cruise. Half of full is a safe, stable guess.
        self._throttle = 0.5
        s = sim.state()
        self.heading = float(np.arctan2(s.velocity[1], s.velocity[0]))

    def update(self, state, sim):
        from .sim import Command
        roll, pitch, yaw = (float(x) for x in state.euler)   # nose-DOWN pitch +ve
        speed = float(np.linalg.norm(state.velocity))
        rates = state.rates

        # Speed -> throttle. Simple integrator around the trim guess.
        self._throttle += self.k_speed * (self.speed - speed) * sim.dt
        self._throttle = float(np.clip(self._throttle, 0.0, 1.0))
        motors = np.array([np.sqrt(self._throttle)])

        # Height -> pitch demand. Climb is nose-up, which is *negative* pitch
        # in this frame, so being below the target asks for negative pitch.
        pitch_cmd = np.clip(-self.k_alt * (self.altitude - state.position[2]),
                            -self.max_pitch, self.max_pitch)
        # Elevator: a positive deflection pushes the nose down (positive pitch),
        # so drive the pitch error directly, and damp the pitch rate.
        elevator = self.k_pitch * (pitch_cmd - pitch) - self.k_q * rates[1]

        # Heading -> bank demand. Banking right (positive roll) turns the nose
        # right, which lowers the heading, so turning *to* a higher heading
        # asks for negative roll. Turn the short way round.
        hdg_err = _wrap(self.heading - yaw)
        roll_cmd = np.clip(-self.k_hdg * hdg_err, -self.max_bank, self.max_bank)
        # A positive aileron rolls left-wing-up (negative roll), so a positive
        # roll error needs a negative aileron; damp the roll rate.
        aileron = -self.k_roll * (roll_cmd - roll) + self.k_p * rates[0]
        # Rudder coordinates the turn (into the bank) and damps yaw rate. A
        # positive rudder yaws the nose right, the way a right bank turns.
        rudder = self.k_yaw * roll + self.k_p * rates[2]

        surfaces = np.clip([elevator, aileron, rudder], -1.0, 1.0)
        return Command(motors, surfaces)


def _wrap(a: float) -> float:
    """An angle folded to (-pi, pi], so a turn always takes the short way."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)
