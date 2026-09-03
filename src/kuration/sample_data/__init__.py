"""Bundled sample datasets for docs and quickstarts.

Ships as package data so the README hero example and quickstart guides can call
``input(...)`` directly against real data with no download and no in-memory
construction step required from the reader.
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent


def sample_sessions_path() -> str:
    """Glob (as a string, ready for ``input(...)``) matching the bundled six-session
    sensor sample -- one Parquet file per session (``t``, ``device``, ``signal``,
    ``fault`` columns, 48 ticks each) -- used by the README's Getting started example.
    """
    return str(_DIR / "sessions" / "*.parquet")
