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
                 start=None) -> None:
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
        self.motors = MotorBank(self.max_thrust, motor_tau)
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
        self.t = 0.0
        self._thrust = self.max_thrust * self.motors.speed ** 2
        mj.mj_forward(self.model, self.data)
        return self.state()

    # ------------------------------------------------------------------ step
    def step(self, command) -> "State":
        """Advance one timestep with motor commands in [0, 1]."""
        thrust = self.motors.step(command, self.dt)
        wrench = self.allocation @ thrust                 # body FLU, about CG
        R = self.data.xmat[self.body].reshape(3, 3)       # body -> world
        v_com = self._com_velocity()
        f_aero, m_aero = self.aero.force(R.T @ (v_com - self.wind))
        self.data.xfrc_applied[self.body, :3] = R @ (wrench[:3] + f_aero)
        self.data.xfrc_applied[self.body, 3:] = R @ (wrench[3:] + m_aero)
        self._mj.mj_step(self.model, self.data)
        self.t += self.dt
        self._thrust = thrust
        return self.state()

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
