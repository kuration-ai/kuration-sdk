# 02: Composition model

## Decision

Three equivalent ways to compose Operations. All produce identical results.

### Direct apply

```python
result = op.apply(pipeline).options(seed=1).collect()
```

For the "one operation" case (CLAUDE.md principle 2: a client can pick just one).

### `run()` with instances

```python
from kuration_sdk_wrapper import BalancedSelector, RareEventCoverage, run

result = run(pipeline, [
    RareEventCoverage(of="collision"),
    BalancedSelector(feature="motion", of="turn_rate", bins=8, count=100),
]).options(seed=1).collect()
```

Ordered application. Same semantics as chaining `.apply()` calls, but reads left-to-right.

### `run()` with config dicts

```python
result = run(pipeline, [
    {"type": "RareEventCoverage", "of": "collision"},
    {"type": "BalancedSelector", "feature": "motion", "of": "turn_rate",
     "bins": 8, "count": 100},
]).options(seed=1).collect()
```

Same list, config-driven. `run()` resolves each dict via the operations registry (see `03_config_model.md`).

## `run()` signature

```python
def run(
    pipeline: kuration.Pipeline,
    operations: list[Operation | dict],
) -> kuration.Pipeline:
    ...
```

- Accepts a mixed list (instances AND dicts): a partial config can be augmented programmatically at the boundary.
- Applies operations in list order.
- Returns the final pipeline. Caller decides what to do with it (`.collect()`, `.output(...)`, `.describe()`).

## Interleaving with the SDK

The wrapper does not hide the SDK. Users can freely mix kit Operations with SDK methods on the same pipeline:

```python
pipeline = (input("data/*.csv", time="t")
            .window(8, stride=4))                            # SDK

pipeline = run(pipeline, [
    BalancedSelector(feature="motion", of="turn_rate"),      # kit
])

pipeline = pipeline.options(seed=1)                          # SDK
result = pipeline.collect()                                  # SDK
```

Never breaks: kit Operations return SDK Pipelines, so SDK methods keep working on them.

## Config-parity (CLAUDE.md principle 3)

The three composition styles above are guaranteed to produce identical results. This is a hard invariant enforced by a test:

```python
def test_config_parity(pipeline_fixture):
    op_instances = [BalancedSelector(feature="motion", of="turn_rate")]
    config_dicts = [{"type": "BalancedSelector", "feature": "motion", "of": "turn_rate"}]

    r_direct = op_instances[0].apply(pipeline_fixture).options(seed=1).collect()
    r_list = run(pipeline_fixture, op_instances).options(seed=1).collect()
    r_config = run(pipeline_fixture, config_dicts).options(seed=1).collect()

    assert len(r_direct.selected) == len(r_list.selected) == len(r_config.selected)
```

If they ever diverge, the wrapper is broken.

## Alternative considered and rejected

**A dedicated `Pipeline` class in kuration_sdk_wrapper that wraps `kuration.Pipeline`.** Rejected because it would force users to learn a new interface for the SDK methods they can already use directly (`.window`, `.measure`, `.filter`, etc.). The additive-not-replacing rule (CLAUDE.md principle 4) forbids hiding the SDK.

**A DAG-based composition model with named edges.** Rejected as premature: every v0.0.1 use case is a linear chain. Add DAG when a real use case demands it.

## GTM angle

The three composition styles are progressive disclosure made concrete. Tier 1 uses `.apply()`: one line, one import. Tier 2 uses `run(pipeline, config)` from a YAML file: the pattern a team adopts. Tier 3 mixes instances and dicts programmatically. Config-parity as a guaranteed invariant, verified by test, is what keeps this from becoming the declarative-versus-programmatic drift trap common libraries fall into once they add a config layer.

## Non-decisions (deferred)

- Parallel `run()` (execute independent branches concurrently): v0.0.2 if a real use case emerges.
- Rollback / dry-run mode: the SDK's `.describe()` already covers dry-run.
