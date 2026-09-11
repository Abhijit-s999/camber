"""Placing parts: symmetry, measurement, and exact rotor layouts.

Everything here works on the vehicle spec directly and needs only numpy, so the
layout editor can call it on every drag (AI-07). Positions are metres, NED
(GEOM-02); the app converts to millimetres and forward/right/up for display.
"""

from .measure import balance, clearances, distance, wheelbase
from .presets import PRESETS, rotor_layout
from .symmetry import (Pairing, find_pairs, mirror, symmetrise,
                       symmetry_report, yaw_balance)

__all__ = ["PRESETS", "Pairing", "balance", "clearances", "distance",
           "find_pairs", "mirror", "rotor_layout", "symmetrise",
           "symmetry_report", "wheelbase", "yaw_balance"]
