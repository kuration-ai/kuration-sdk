"""Public facade for Kuration's domain-agnostic curation core.

The historical top-level imports remain supported. New code that wants to make the
domain boundary explicit can import the same stable surface from :mod:`kuration.core`.
"""
from __future__ import annotations

from .. import *  # noqa: F403
from .. import __all__ as __all__
