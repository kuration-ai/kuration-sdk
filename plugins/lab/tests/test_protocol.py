"""The on-disk run protocol: event stamping, cursor semantics, atomic status."""
from __future__ import annotations

import json

from kuration_lab.jobs import protocol


def test_event_writer_stamps_monotonic_seq_and_run_id(tmp_path):
    writer = protocol.EventWriter(tmp_path, "run-1")
    writer.emit({"phase": "scoring", "event": "start", "total": 2})
    writer.emit({"phase": "scoring", "event": "advance", "current": 1})
    writer.lifecycle("complete")
    writer.close()

    events, cursor = protocol.read_events(tmp_path)
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert cursor == 3
    assert all(e["run_id"] == "run-1" for e in events)
    assert all("ts" in e for e in events)
    assert events[2] == {**events[2], "phase": "lifecycle", "event": "complete"}


def test_read_events_since_and_limit_never_skip(tmp_path):
    writer = protocol.EventWriter(tmp_path, "r")
    for index in range(10):
        writer.emit({"phase": "scoring", "event": "advance", "current": index})
    writer.close()

    events, cursor = protocol.read_events(tmp_path, since=0, limit=4)
    assert [e["seq"] for e in events] == [1, 2, 3, 4]
    assert cursor == 4                       # cursor stops at the truncated batch
    events, cursor = protocol.read_events(tmp_path, since=cursor, limit=100)
    assert [e["seq"] for e in events] == [5, 6, 7, 8, 9, 10]
    assert cursor == 10
    events, cursor = protocol.read_events(tmp_path, since=cursor)
    assert events == [] and cursor == 10


def test_read_events_tolerates_a_torn_tail_line(tmp_path):
    writer = protocol.EventWriter(tmp_path, "r")
    writer.emit({"phase": "scoring", "event": "start"})
    writer.close()
    with open(tmp_path / protocol.EVENTS_FILE, "a", encoding="utf-8") as handle:
        handle.write('{"seq": 2, "phase": "scor')  # torn mid-write

    events, cursor = protocol.read_events(tmp_path)
    assert [e["seq"] for e in events] == [1]
    assert cursor == 1


def test_status_roundtrip_and_atomic_replace(tmp_path):
    protocol.write_status(tmp_path, protocol.STATE_RUNNING, pid=123)
    status = protocol.read_status(tmp_path)
    assert status["state"] == protocol.STATE_RUNNING and status["pid"] == 123
    assert status["updated"]
    protocol.write_status(tmp_path, protocol.STATE_FAILED, exit={"ok": False})
    assert protocol.read_status(tmp_path)["state"] == protocol.STATE_FAILED
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert leftovers == []                   # temp file was replaced, not abandoned

    raw = json.loads((tmp_path / protocol.STATUS_FILE).read_text())
    assert raw["exit"] == {"ok": False}


def test_missing_files_read_as_defaults(tmp_path):
    assert protocol.read_status(tmp_path)["state"] == protocol.STATE_PENDING
    assert protocol.read_events(tmp_path) == ([], 0)
    assert protocol.read_json(tmp_path / "nope.json") is None
