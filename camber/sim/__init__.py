"""Flying the designed vehicle in a physics engine (MuJoCo).

The simulator is built *from* the Camber vehicle, not alongside it: mass,
centre of gravity and the full inertia tensor come from `mass_properties()`,
and the wrench each rotor produces comes from the same allocation matrix the
optimiser designed against (`control.allocation_matrix`). A second, hand-written
model of the aircraft would be a second thing to get wrong -- and would let the
simulator fly a better aircraft than the one that gets built.

MuJoCo is an optional dependency (`pip install camber[sim]`); nothing in the
core imports this package (AI-07).
"""

from .frames import NED_TO_FLU
from .sim import Simulation, State

__all__ = ["NED_TO_FLU", "Simulation", "State"]
