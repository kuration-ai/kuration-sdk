"""Kuration's public package surface.

Implementation lives in explicit product domains:

- :mod:`kuration.core` contains the dataset-agnostic curation SDK.
- :mod:`kuration.gameplay` contains maintained gameplay capabilities.
- :mod:`kuration.selection` contains built-in dataset-selection recipes
  (balanced coverage, rare-event weighting, cluster-balanced diversity).

Core symbols remain available from :mod:`kuration` for a concise default API, while
domain-specific code is imported from its domain package.
"""
from __future__ import annotations

from .core import *  # noqa: F403
from .core import __all__ as __all__
