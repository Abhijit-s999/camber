"""Programming the vehicle in Python: a figure-eight.

    camber sim examples/racer5.toml --pilot examples/pilots/figure_eight.py --viewer

A pilot is a class with an `update(state, sim)` method returning motor
commands. Subclassing `Cascade` means you only say *where to be* -- it turns
that into attitude, rates and motor commands through the vehicle's own
allocation mixer. `state` has position, velocity, quaternion, rates (gyro),
specific_force (accelerometer) and thrust, in REP-103 axes: x forward, y left,
z up.
"""

import numpy as np

from camber.sim.flight import Cascade


class Pilot(Cascade):
    size = (7.0, 3.5)       # half-length, half-width of the eight, m
    height = 2.0
    omega = 0.55            # rad/s: one eight every 11.4 s

    def update(self, state, sim):
        t, (a, b), w = sim.t, self.size, self.omega
        ramp = min(1.0, t / 3.0)                 # ease in from the hover
        # Lemniscate of Gerono: x = a sin(wt), y = b sin(2wt). Position,
        # velocity and acceleration are all given, so the cascade flies the
        # curve rather than chasing a point along it.
        self.target = np.array([a * np.sin(w * t), b * np.sin(2 * w * t),
                                self.height]) * [ramp, ramp, 1.0]
        self.velocity = ramp * np.array([a * w * np.cos(w * t),
                                         2 * b * w * np.cos(2 * w * t), 0.0])
        self.accel = ramp * np.array([-a * w * w * np.sin(w * t),
                                      -4 * b * w * w * np.sin(2 * w * t), 0.0])
        if np.linalg.norm(self.velocity[:2]) > 0.5:
            self.yaw = float(np.arctan2(self.velocity[1], self.velocity[0]))
        return super().update(state, sim)
