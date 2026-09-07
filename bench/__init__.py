"""The bench: development-time model orchestration for building Camber.

This package is a tool, not a component. Nothing under `camber/` may import
from it, and it is excluded from the built wheel. See docs/DECISIONS.md AI-04.
"""

__all__ = ["config", "ratelimit", "cache", "callog", "client", "roles"]
