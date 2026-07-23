"""Compatibility import for the former action-state consistency plugin package.

The implementation now ships as :mod:`kuration.action_state_consistency`. This module
keeps previously installed plugin entry-point metadata functional during migration.
"""

from kuration.action_state_consistency import *  # noqa: F403
from kuration.action_state_consistency import __all__ as __all__
from kuration.action_state_consistency import register as register
