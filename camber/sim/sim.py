"""Step the designed vehicle through time in MuJoCo.

Forces are applied through `xfrc_applied` -- a world-frame wrench at the body's
centre of mass -- rather than through MuJoCo actuators. MuJoCo's actuators are
linear in their control, and a rotor is not: thrust goes as speed squared, and
speed lags the command. Computing the wrench here keeps that physics in one
readable place, and keeps it *Camber's* physics: the wrench is the allocation
matrix times the rotor thrusts, the same matrix the optimiser designed against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import frames
from .aero import BodyDrag
from .model import build_mjcf
from .motors import MotorBank


@dataclass
class Command:
    """What a controller hands the plant when a vehicle has more than rotors:
    motor commands in [0, 1], and control-surface deflections in [-1, 1]."""

    motors: np.ndarray
    surfaces: np.ndarray | None = None


def _split_command(command):
    """Accept either a bare motor array or a Command, so a multirotor pilot
    that returns an array still works unchanged."""
    if hasattr(command, "motors"):
        return command.motors, command.surfaces
    return command, None


@dataclass
class State:
    """What a flight controller gets to see. World axes are x forward, y left,
    z up (ROS REP-103, and MuJoCo's own); body axes are FLU."""

    t: float
    position: np.ndarray        # centre of mass, world, m
    velocity: np.ndarray        # centre of mass, world, m/s
    quaternion: np.ndarray      # w, x, y, z; body -> world
    rates: np.ndarray           # body angular velocity, rad/s (gyro)
    specific_force: np.ndarray  # accelerometer, body, m/s^2
    thrust: np.ndarray          # per rotor, N

    @property
    def rotation(self) -> np.ndarray:
        w, x, y, z = self.quaternion
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    @property
    def euler(self) -> np.ndarray:
        """Roll, pitch, yaw in radians (ZYX), FLU: positive pitch is nose DOWN
        about +y left -- REP-103's convention, stated because it is the
        opposite sign from NED's."""
        R = self.rotation
        return np.array([np.arctan2(R[2, 1], R[2, 2]),
                         np.arcsin(-np.clip(R[2, 0], -1, 1)),
                         np.arctan2(R[1, 0], R[0, 0])])


class Simulation:
    """A Camber vehicle in MuJoCo: `step(command) -> State`."""

    def __init__(self, vehicle, *, course=None, dt: float = 0.002,
                 motor_tau: float = 0.03, aero=None, wind=(0.0, 0.0, 0.0),
                 start=None, latency: float = 0.006, prop_mass=None) -> None:
        import mujoco

        from ..control import allocation_matrix

        self._mj = mujoco
        self.vehicle = vehicle
        self.course = course
        mp = vehicle.mass_properties()
        self.mass = mp.mass
        self.inertia = frames.tensor(mp.inertia)
        self.start = np.asarray(start if start is not None
                                else (0.0, 0.0, self._rest_height()), float)
        self.xml = build_mjcf(vehicle, course=course, timestep=dt,
                              start=self.start)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.body = self.model.body("airframe").id
        self.dt = float(self.model.opt.timestep)
        # The optimiser's allocation matrix, in the simulator's axes.
        self.allocation = frames.wrench(allocation_matrix(vehicle))
        self.max_thrust = np.array([t.max_thrust() for t in vehicle.thrusters])
        # What each rotor's spinning mass is, and how fast it turns at full
        # throttle: the airframe feels both when the speeds change and when it
        # is rotated. Propeller mass, where the spec does not carry one, from
        # the usual relation for injection-moulded props -- about 4 g on a
        # 5-inch. It is an estimate and is stated as one.
        radii = np.array([getattr(t, "radius", 0.0) or 0.05
                          for t in vehicle.thrusters])
        masses = (np.broadcast_to(np.asarray(prop_mass, float), radii.shape)
                  if prop_mass is not None else 0.0043 * (radii / 0.0635) ** 3)
        self.motors = MotorBank(
            self.max_thrust, motor_tau,
            inertia=MotorBank.disc_inertia(masses, radii),
            max_rate=MotorBank.rate_at_full(
                self.max_thrust, radii,
                [getattr(t, "blades", 2) for t in vehicle.thrusters]),
            spin=np.array([float(getattr(t, "spin", None).value)
                           if getattr(t, "spin", None) is not None else 1.0
                           for t in vehicle.thrusters]))
        # Thrust axes in body FLU, for the momentum terms.
        self.thrust_axes = np.array([frames.vec(t.direction())
                                     for t in vehicle.thrusters])
        # Everything between deciding and acting: the flight controller's own
        # loop, the link, and the ESC. Small, but it is the difference between
        # a rate loop that can be tuned hot and one that rings.
        self.latency = max(0.0, float(latency))
        self._queue: list = []
        self.aero = aero if aero is not None else BodyDrag.estimate(vehicle)
        self.wind = np.asarray(wind, dtype=float)
        self._gyro = self.model.sensor("gyro").adr[0]
        self._accel = self.model.sensor("accel").adr[0]
        self.reset()

    # ------------------------------------------------------------------ setup
    def _rest_height(self) -> float:
        """Body-origin height at which the lowest part just clears the floor."""
        low = 0.0
        for b in self.vehicle.all_bodies():
            z = (frames.vec(b.position)[2]
                 - abs(frames.NED_TO_FLU @ np.asarray(
                     b.shape.half_extents().as_list()))[2])
            low = min(low, z)
        return -low + 0.002

    def reset(self, position=None, quaternion=(1.0, 0.0, 0.0, 0.0),
              velocity=(0.0, 0.0, 0.0), spin: float = 0.0) -> "State":
        mj = self._mj
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.start if position is None else position
        self.data.qpos[3:7] = quaternion
        self.data.qvel[:3] = velocity
        self.motors.reset(spin)
        self._prime(spin)
        self.t = 0.0
        self._thrust = self.max_thrust * self.motors.speed ** 2
        mj.mj_forward(self.model, self.data)
        return self.state()

    # ------------------------------------------------------------------ step
    def step(self, command) -> "State":
        """Advance one timestep.

        ``command`` is the motor commands in [0, 1] -- an array -- or, for a
        vehicle with control surfaces, an object carrying ``.motors`` and
        ``.surfaces`` (deflections in [-1, 1] per channel). One call, whether
        the vehicle flies on rotors, on wings, or on both.
        """
        motors, surfaces = _split_command(command)
        motors = self._delayed(motors)
        thrust = self.motors.step(motors, self.dt)
        wrench = self.allocation @ thrust                 # body FLU, about CG
        R = self.data.xmat[self.body].reshape(3, 3)       # body -> world
        v_com = self._com_velocity()
        v_body = R.T @ (v_com - self.wind)
        if surfaces is not None and hasattr(self.aero, "set_controls"):
            self.aero.set_controls(surfaces)
        f_aero, m_aero = self._aero_wrench(v_body)
        m_rotor = self._rotor_moments(R)
        self.data.xfrc_applied[self.body, :3] = R @ (wrench[:3] + f_aero)
        self.data.xfrc_applied[self.body, 3:] = R @ (wrench[3:] + m_aero + m_rotor)
        self._mj.mj_step(self.model, self.data)
        self.t += self.dt
        self._thrust = thrust
        return self.state()

    def _aero_wrench(self, v_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Body aero force and moment. A wing set needs the body rate too --
        the tail's angle of attack depends on how fast the nose is pitching --
        so a model offering ``wrench(v, omega)`` is given both; a plain drag
        model keeps the ``force(v)`` interface."""
        if hasattr(self.aero, "wrench"):
            omega = np.asarray(self.data.qvel[3:6], dtype=float)
            return self.aero.wrench(v_body, omega)
        return self.aero.force(v_body)

    def _prime(self, spin) -> None:
        """Fill the delay line with what the rotors are already doing. A
        vehicle reset mid-hover has commands in flight; starting the pipeline
        empty fed it zeros for the first few milliseconds and it sank."""
        n = (max(1, int(round(self.latency / self.dt)))
             if self.latency > 0.0 else 0)
        held = np.broadcast_to(np.asarray(spin, float), self.max_thrust.shape)
        self._queue = [held.copy() for _ in range(n)]

    def _delayed(self, command) -> np.ndarray:
        """Hold each command for `latency` before it reaches the motors."""
        command = np.asarray(command, dtype=float)
        if self.latency <= 0.0:
            return command
        self._queue.append(command.copy())
        n = max(1, int(round(self.latency / self.dt)))
        if len(self._queue) <= n:
            return np.zeros_like(command)                 # nothing has arrived yet
        return self._queue.pop(0)

    def _rotor_moments(self, R: np.ndarray) -> np.ndarray:
        """What the spinning rotors do to the airframe, in body axes.

        Two terms, both absent from the allocation matrix, which knows only
        about steady thrust and the drag torque that comes with it:

        * -J dw/dt about each rotor axis, the reaction to spinning it up. It
          leads the aerodynamic yaw torque, because drag torque cannot change
          until the speed has.
        * -w_body x h, the gyroscopic couple, where h is the rotors' angular
          momentum. Pitch a spinning rotor and it answers in roll.
        """
        h = self.thrust_axes.T @ self.motors.angular_momentum        # body, N m s
        dh = self.thrust_axes.T @ self.motors.momentum_change        # body, N m
        w = np.asarray(self.data.qvel[3:6], dtype=float)             # body rates
        return -dh - np.cross(w, h)

    def _com_velocity(self) -> np.ndarray:
        out = np.zeros(6)
        self._mj.mj_objectVelocity(self.model, self.data,
                                   self._mj.mjtObj.mjOBJ_BODY, self.body, out, 0)
        return out[3:].copy()                             # [rot; lin], world

    def state(self) -> State:
        d = self.data
        return State(
            t=self.t,
            position=d.xipos[self.body].copy(),
            velocity=self._com_velocity(),
            quaternion=d.qpos[3:7].copy(),
            rates=d.sensordata[self._gyro:self._gyro + 3].copy(),
            specific_force=d.sensordata[self._accel:self._accel + 3].copy(),
            thrust=np.asarray(self._thrust).copy())

    @property
    def weight(self) -> float:
        return self.mass * 9.80665

    @property
    def crashed(self) -> bool:
        """Touching something while inverted, or while moving fast.

        Being upside down is not a crash: it is the whole of a flip. An earlier
        version counted it as one and declared every flip a crash at 90 degrees
        of roll. Contact is what makes it a crash -- and contact at walking pace,
        upright, is a landing.
        """
        touching = any(
            self.model.geom_bodyid[c.geom1] == self.body or
            self.model.geom_bodyid[c.geom2] == self.body
            for c in self.data.contact[:self.data.ncon])
        if not touching:
            return False
        s = self.state()
        return bool(s.rotation[2, 2] < 0.0 or np.linalg.norm(s.velocity) > 3.0)
