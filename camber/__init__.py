"""Camber -- closed-loop airframe optimiser.

Nothing here may import from `bench`. The bench is development tooling that
builds this package; it is not part of it, it is excluded from the wheel, and
tests/test_isolation.py fails the build if that boundary is ever crossed.
See docs/DECISIONS.md AI-04.
"""

__version__ = "0.0.1"
