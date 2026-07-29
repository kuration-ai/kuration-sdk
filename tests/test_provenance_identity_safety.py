"""Release/reuse identities must be collision-resistant and process-stable."""
from __future__ import annotations

import os
import subprocess
import sys

from kuration.core.checkpoints import artifact_checkpoint_summary
from kuration.core.provenance import _state_hash, component_identity


_REFERENCED_LIMIT = 1


def _uses_referenced_global(value):
    return value + _REFERENCED_LIMIT


def _same_line_lambdas():
    return (lambda value: value + 1), (lambda value: value + 1)


def test_same_line_lambdas_have_distinct_code_identity():
    left, right = _same_line_lambdas()

    assert component_identity(left) != component_identity(right)


def test_referenced_global_value_participates_in_callable_identity(monkeypatch):
    before = component_identity(_uses_referenced_global)
    monkeypatch.setattr(sys.modules[__name__], "_REFERENCED_LIMIT", 2)
    after = component_identity(_uses_referenced_global)

    assert before != after


def test_state_hash_is_independent_of_mapping_and_set_iteration_order():
    left = {"letters": {"gamma", "alpha", "beta"}, "nested": {"x": 1, "y": 2}}
    right = {"nested": {"y": 2, "x": 1}, "letters": {"beta", "gamma", "alpha"}}

    assert _state_hash(left) == _state_hash(right)


def test_state_hash_is_stable_across_python_hash_seeds():
    script = (
        "from kuration.core.provenance import _state_hash\n"
        "keys = {'alpha', 'beta', 'gamma', 'delta'}\n"
        "value = {key: {key, key.upper()} for key in keys}\n"
        "print(_state_hash(value))\n"
    )
    hashes = []
    for seed in ("1", "271828"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        hashes.append(subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=environment,
            text=True,
        ).strip())

    assert hashes[0] == hashes[1]


def test_artifact_checkpoint_summary_canonicalizes_partition_evidence():
    first = {
        "stage_id": "stage000",
        "identity": {"plan": "same"},
        "identity_fingerprint": "identity",
        "rows": 3,
        "commit": True,
        "partitions": [
            {
                "partition_id": "source:1",
                "identity": "identity",
                "rows": 1,
                "sha256": "b" * 64,
                "columns": ["z", "a"],
                "size_bytes": 2,
            },
            {
                "partition_id": "source:0",
                "identity": "identity",
                "rows": 2,
                "sha256": "a" * 64,
                "columns": ["a", "z"],
                "size_bytes": 3,
            },
        ],
    }
    second = {**first, "partitions": [
        {**entry, "columns": list(reversed(entry["columns"]))}
        for entry in reversed(first["partitions"])
    ]}

    assert artifact_checkpoint_summary(first) == artifact_checkpoint_summary(second)
