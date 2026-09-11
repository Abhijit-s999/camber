"""Programming at the bottom of the stack: a flip, flown on body rates.

    camber sim examples/racer5.toml --pilot examples/pilots/flip.py --viewer

`Cascade.rates(state, rate_setpoint, collective, dt)` is the rate loop on its
own -- the layer an acro pilot flies. Nothing here knows about positions: climb,
then command 900 deg/s of roll with the throttle cut so the vehicle is not
driven into the ground inverted, then hand back to position hold.
"""

import numpy as np

from camber.sim.flight import Cascade


class Pilot(Cascade):
    def reset(self, sim):
        super().reset(sim)
        self.phase, self.rolled = "climb", 0.0

    def update(self, state, sim):
        if self.phase == "climb":
            self.target = np.array([0.0, 0.0, 4.0])
            if state.position[2] > 3.7 and abs(state.velocity[2]) < 0.4:
                self.phase = "punch"
            return super().update(state, sim)
        if self.phase == "punch":                     # up, so there is room
            if state.velocity[2] > 3.0:
                self.phase = "flip"
            return self.rates(state, np.zeros(3), 1.6 * sim.weight, sim.dt)
        if self.phase == "flip":
            self.rolled += state.rates[0] * sim.dt
            if self.rolled >= 2 * np.pi - 0.35:       # let the loop finish it
                self.phase = "recover"
                self.target = state.position.copy()
                self.target[2] = max(self.target[2], 2.0)
            return self.rates(state, np.array([np.radians(900), 0.0, 0.0]),
                              0.25 * sim.weight, sim.dt)
        return super().update(state, sim)             # recover: hold position
