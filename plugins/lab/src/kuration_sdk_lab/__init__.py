"""kuration-sdk-lab — JupyterLab extension for the kuration SDK.

Keep this module import-light: JupyterLab and jupyter_server discover the extension through
the two ``_jupyter_*`` functions below, and everything heavy loads lazily inside them.
"""
from __future__ import annotations

__version__ = "0.1.0"

API_VERSION = 1


def _jupyter_labextension_paths():
    return [{"src": "labextension", "dest": "kuration-sdk-lab"}]


def _jupyter_server_extension_points():
    from .app import KurationLabApp

    return [{"module": "kuration_sdk_lab.app", "app": KurationLabApp}]
