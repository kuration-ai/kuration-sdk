"""Compatibility import for the former pixel-shift plugin package.

The implementation now ships as :mod:`kuration.pixel_shift`. This module remains so
environments with the old plugin entry-point metadata can upgrade without discovery
failures.
"""

from kuration.pixel_shift import *  # noqa: F403
from kuration.pixel_shift import __all__ as __all__
from kuration.pixel_shift import register as register
