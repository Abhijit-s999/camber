"""A flight you can watch and fly while it happens.

The simulator normally runs as fast as it can and hands back a log. Here it
runs in a thread paced to the wall clock, keeping one current state that the
app reads at the browser's frame rate and writes sticks into. The physics
stays exactly the same -- same `Simulation`, same pilots, same aero table --
only the clock and the audience change.

Pacing: catch up to the wall clock in whole timesteps, up to a limit. If the
machine cannot keep up, time is *dropped* rather than accumulated, and the
snapshot reports the real-time factor honestly instead of running in slow
motion and pretending.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..sim import Simulation
from ..sim.aero import AeroTable, BodyDrag
from ..sim.course import Course
from ..sim.flight import GateRacer, Hover
from ..sim.manual import MODES, Manual, Sticks

MAX_CATCHUP = 0.1          # s of simulation per wake-up; beyond this, drop time
IDLE_STOP = 20.0           # s without anyone watching before it shuts itself down


class Flight:
    """One live flight. Thread-safe: the browser's thread only touches the
    public methods, which hold the lock."""

    def __init__(self, vehicle, *, pilot: str = "manual", mode: str = "assist",
                 course: bool = True, aero=None, speed: float = 14.0,
                 pilot_factory=None) -> None:
        self.course = Course.oval() if course else None
        self.sim = Simulation(vehicle, course=self.course, aero=aero)
        self.kind = pilot
        self.speed = speed
        self._factory = pilot_factory
        self.pilot = self._make(pilot, mode)
        self.lock = threading.RLock()
        self.thread = None
        self.running = False
        self.seen = time.monotonic()
        self.rate = 1.0
        self.headroom = 1.0
        self._done = self._wall = 0.0
        self.gates, self.lap, self.best = [], None, None
        self.crashed_at = None
        self.state = self.sim.reset()
        self.pilot.reset(self.sim)

    # --- setup ----------------------------------------------------------------
    def _make(self, kind: str, mode: str):
        if kind == "manual":
            return Manual(mode if mode in MODES else "assist")
        if kind == "hover":
            return Hover(height=1.5)
        if kind == "gates" and self.course is not None:
            return GateRacer(self.course, speed=self.speed)
        if self._factory is not None:
            return self._factory(kind, self.course, self.speed)
        raise ValueError(f"cannot fly {kind!r} without a course")

    @property
    def manual(self) -> Manual | None:
        return self.pilot if isinstance(self.pilot, Manual) else None

    # --- running --------------------------------------------------------------
    def start(self) -> "Flight":
        with self.lock:
            if self.running:
                return self
            self.running = True
            self.seen = time.monotonic()
            self._t0 = time.monotonic() - self.sim.t
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def stop(self) -> None:
        with self.lock:
            self.running = False
        t = self.thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _run(self) -> None:
        while self.running:
            now = time.monotonic()
            if now - self.seen > IDLE_STOP:      # nobody is watching any more
                self.running = False
                break
            behind = (now - self._t0) - self.sim.t
            if behind > MAX_CATCHUP:
                self._t0 += behind - MAX_CATCHUP         # drop it, do not bank it
                behind = MAX_CATCHUP
            steps = int(behind / self.sim.dt)
            if steps <= 0:
                time.sleep(0.001)
                continue
            began = time.monotonic()
            with self.lock:
                for _ in range(steps):
                    self.state = self.sim.step(self.pilot.update(self.state, self.sim))
                    self._watch()
            spent = time.monotonic() - began
            done = steps * self.sim.dt
            # Two different numbers, and reporting the wrong one flatters the
            # machine: `headroom` is how much faster than real time it *could*
            # run, `rate` is how much of real time it actually kept up with.
            # Paced properly, rate is 1.00 and headroom is whatever is spare.
            self.headroom = 0.9 * self.headroom + 0.1 * (done / max(spent, 1e-6))
            self._done += done
            self._wall += time.monotonic() - now
            if self._wall > 0.5:
                self.rate = self._done / self._wall
                self._done = self._wall = 0.0
            time.sleep(0.001)

    def _watch(self) -> None:
        """Gates and crashes, tracked for any pilot -- the racing pilot counts
        its own, but someone flying by hand deserves a lap time too."""
        if self.course is not None:
            i = len(self.gates) % len(self.course.gates)
            n = len(self.course.gates)
            if self.course.gates[i].crossed(self.state.position):
                self.gates.append(self.sim.t)
                if len(self.gates) % n == 0:
                    # A lap ends at every nth gate: from the same gate one lap
                    # back, or from the start for the first one.
                    prev = self.gates[-n - 1] if len(self.gates) > n else 0.0
                    self.lap = self.sim.t - prev
                    self.best = self.lap if self.best is None else min(self.best, self.lap)
        if self.sim.crashed and self.crashed_at is None:
            self.crashed_at = self.sim.t

    # --- the app's handles ------------------------------------------------------
    def touch(self) -> None:
        self.seen = time.monotonic()

    def set_sticks(self, roll, pitch, yaw, throttle) -> None:
        m = self.manual
        if m is not None:
            m.sticks = Sticks(roll, pitch, yaw, throttle).clipped()

    def set_mode(self, mode: str) -> None:
        m = self.manual
        if m is None:
            raise ValueError("only a hand-flown vehicle has modes")
        if mode not in MODES:
            raise ValueError(f"no such mode: {mode}")
        with self.lock:
            m.mode = mode
            m._apply_mode()
            m.yaw = float(self.state.euler[2])
            m._hold = self.state.position.copy()

    def arm(self, on: bool) -> None:
        m = self.manual
        if m is not None:
            m.armed = bool(on)

    def reset(self) -> None:
        with self.lock:
            self.state = self.sim.reset()
            self.sim.t = 0.0
            self.pilot.reset(self.sim)
            self.gates, self.lap, self.crashed_at = [], None, None
            self._t0 = time.monotonic()

    # --- what the app draws -------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            s, m = self.state, self.manual
            q = s.quaternion
            roll, pitch, yaw = (float(x) for x in s.euler)
            tilt = float(np.degrees(np.arccos(np.clip(s.rotation[2, 2], -1.0, 1.0))))
            gates = len(self.course.gates) if self.course else 0
            return {
                "running": self.running,
                "t": round(self.sim.t, 3),
                "rate": round(float(self.rate), 3),
                "headroom": round(float(self.headroom), 2),
                "pos": [round(float(x), 4) for x in s.position],
                "quat": [round(float(x), 5) for x in q],
                "vel": [round(float(x), 3) for x in s.velocity],
                "rates": [round(float(x), 3) for x in s.rates],
                "thrust": [round(float(x), 3) for x in s.thrust],
                "speed": round(float(np.linalg.norm(s.velocity)), 2),
                "ground_speed": round(float(np.linalg.norm(s.velocity[:2])), 2),
                "altitude": round(float(s.position[2]), 2),
                "climb": round(float(s.velocity[2]), 2),
                "tilt_deg": round(tilt, 1),
                "rpy_deg": [round(float(np.degrees(x)), 1) for x in (roll, pitch, yaw)],
                "throttle": round(float(np.sum(s.thrust) / max(np.sum(self.sim.max_thrust), 1e-9)), 3),
                "pilot": self.kind,
                "mode": m.mode if m else None,
                "armed": bool(m.armed) if m else True,
                "sticks": ([m.sticks.roll, m.sticks.pitch, m.sticks.yaw,
                            m.sticks.throttle] if m else None),
                "crashed": self.crashed_at is not None,
                "gates_passed": len(self.gates),
                "gates_total": gates,
                "lap": round(self.lap, 2) if self.lap else None,
                "best_lap": round(self.best, 2) if self.best else None,
            }

    def scene(self) -> dict:
        """Sent once when the flight starts: what to build the world from."""
        return {
            "gates": [{"centre": [float(x) for x in g.centre],
                       "normal": [float(x) for x in g.normal],
                       "size": float(g.size)}
                      for g in (self.course.gates if self.course else [])],
            "start": [float(x) for x in self.sim.start],
            "max_thrust": [float(x) for x in self.sim.max_thrust],
            "hover_thrust": float(self.sim.weight),
            "mass_kg": round(float(self.sim.mass), 4),
        }


def aero_for(project, want_cfd: bool):
    """The measured table if the project has one and it was asked for."""
    table = project.meta.get("aero_table")
    if want_cfd and table:
        return AeroTable.load(project.path(table))
    return BodyDrag.estimate(project.vehicle)
