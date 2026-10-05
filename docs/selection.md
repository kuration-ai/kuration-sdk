# Selection recipes

`kuration.selection` ships three general-purpose dataset-curation recipes — no gameplay or video
assumptions — plus a small config-driven runner for chaining them (or your own) from a dict, YAML,
or JSON config instead of hand-written Python. Importing the module registers all three as
`Pipeline` methods; see [Operators §5](operators.md#5-built-in-operator-catalog) for the catalog
entry and [the inventory](operator-inventory.md) for exact parameter tables.

Every recipe compiles to ordinary core primitives (`.measure()`, `.bin()`, `.filter()`,
`.sample()`) and exposes `.describe()` to show that exact chain — there is no hidden behavior a
user couldn't reproduce by hand.

## BalancedSelector — cover the whole range of a signal

Registered as the operator `kuration/balanced-selector`.

Most naturally-occurring signals are middle-heavy: measuring and sampling by natural distribution
under-represents the tails. `BalancedSelector` measures a per-window feature, quantile-bins it
across the observed range, and samples an equal count from every bin, so a training set covers the
whole spread instead of the mode.

```python
pipeline.balanced_selector(feature="activity", of="signal", kind="std", bins=8, count=100)
```

Compiles to `.measure(feature, kind, of=of).bin(feature, bins=bins).sample("equal", n=count)`.

## RareEventCoverage — don't let rare events get diluted

Registered as the operator `kuration/rare-event-coverage`.

A rare failure or contact event living inside a much larger stream of healthy telemetry gets
dropped by ordinary distribution matching. `RareEventCoverage` measures the per-window rate of a
boolean/0-1 event field, then weighted-samples so windows containing more of the event are favored
proportional to occupancy.

```python
pipeline.rare_event_coverage(of="fault", feature="event_rate", count=100, weight=1.0)
```

Compiles to `.measure(feature, "rate", of=of).sample("weighted", n=count, weights={feature: weight})`.

`.sample("weighted", ...)` has no bounded/streaming execution path yet — it materializes the
whole scored candidate pool in memory, not just the `count` selected (`explain()` flags this as
`unbounded-operator`). Fine whenever that candidate pool fits in memory; for a run that needs a
bounded path (`memory_budget=`/`streaming=True`), there isn't a weighted equivalent today —
`BalancedSelector` (bounded) is the nearest sibling recipe if rarity-weighting specifically isn't
required, or narrow the pool with an earlier `.filter()`.

## DiverseSubset — compress without losing coverage

Registered as the operator `kuration/diverse-subset`.

Given a set of already-measured features, `DiverseSubset` optionally drops near-duplicates by
cosine similarity, then samples equally from k-means clusters over the joint feature space.
Requires `kuration-sdk[methods]` (scikit-learn) at execution time, and expects `features` to
already be present on the pipeline via `.measure(...)`.

```python
pipeline.diverse_subset(features=("act", "temp_mean"), count=100, n_clusters=32, dedup_threshold=None)
```

Compiles to an optional `.filter("dedup", feature_space=features, threshold=dedup_threshold)`
followed by `.sample("cluster-balanced", n=count, feature_space=features, clusters=n_clusters)`.

## Direct dataclass use

Each recipe is also a plain, frozen dataclass with the same fields as its operator-method form, an
`.apply(pipeline) -> pipeline` method, and a `.describe() -> str` method:

```python
from kuration.selection import BalancedSelector

op = BalancedSelector(feature="activity", of="signal", bins=4, count=20)
result = op.apply(pipeline).options(seed=1).collect()
print(op.describe())
# BalancedSelector: .measure('activity', 'std', of='signal') -> .bin('activity', bins=4) -> .sample('equal', n=20)
```

Constructor validation is fail-fast (`ValueError` at construction, before the pipeline is
touched) and names the invalid field.

## Config-driven chaining

`kuration.selection.run`/`plan`/`describe` accept a list of Operation instances, `{"type": ...}`
config dicts, or a mix of both — all three call styles (operator method, direct dataclass, config
dict) produce identical results.

```python
from kuration.selection import run

result = run(pipeline, [
    {"type": "BalancedSelector", "feature": "activity", "of": "signal", "bins": 4, "count": 20},
]).options(seed=1).collect()
```

- `plan(operations)` resolves dicts into instances without applying them, for inspection or
  logging before running.
- `describe(operations)` returns a multi-line human summary naming each operation's compiled SDK
  chain.
- Unresolvable config entries raise a `ValueError` naming the offending index and, for close
  misspellings of a `type` or keyword, a "did you mean" hint.

Register your own `.apply(pipeline) -> pipeline` class for config-dict use with
`register_operation_type`:

```python
from dataclasses import dataclass
from kuration.selection import register_operation_type

@dataclass(frozen=True)
class OnlyActiveWindows:
    threshold: float = 0.1

    def apply(self, pipeline):
        return pipeline.measure("act", "std", of="signal").filter("act", ">", self.threshold)

register_operation_type("OnlyActiveWindows", OnlyActiveWindows)
run(pipeline, [{"type": "OnlyActiveWindows", "threshold": 0.15}])
```

This registry is independent of the SDK's operator `Registry` — it only affects how
dict-shaped entries resolve inside `run`/`plan`/`describe`, not what's callable as
`pipeline.<name>(...)`.
