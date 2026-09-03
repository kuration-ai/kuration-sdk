# What your pipeline compiles to, and what gets replayed from disk

You don't write this by hand. Every pipeline — `input(...).window(...).top_k(...)`, however you
built it — compiles to a **strategy**: a small JSON document describing *how* to turn scored
segments into a curated set (inline metric definitions, optional bin axes, a selection method and
its knobs). That compiled strategy is what actually lands in `config.json` alongside the written
shards, and it's what a later `kur-migrate`/replay/audit reads back — the resolved strategy *is*
the run's self-describing, auditable contract, not a summary of it. Replay also requires the same
immutable/content-addressed sources and compatible component implementations.

This page exists for three reasons: to explain what's actually inside `config.json` when you
need to audit or debug a run, to document the JSON keys (`"selection": {"method": ...}`,
`"filters": [...]`, `"scorers": [...]`) a job spec or the `kur-*` CLI tools read directly, and to
name the underlying method/filter/scorer/target-distribution semantics your pipeline's operators
(`.top_k(...)`, `.anomaly(...)`, `.action_residual(...)` — see [operators.md](operators.md) and
[operator-inventory.md](operator-inventory.md)) compile down to. You don't need this page to use
the pipeline; every operator produces one of these strategies for you.

```python
from kuration import load_strategy
strategy = load_strategy(spec_dict_or_path_or_name, schema)
```

`load_strategy` validates loudly at load: every `define` metric compiles against the schema;
every metric referenced by axes/weights/target must be defined, registered, or a scorer
output; unknown `selection` keys raise (a typo'd `"tag_covrage"` can't silently disable
coverage — the known method params are `clusters`, `dsir_bins`, `knn_k`, `rare_sample`,
`feature_normalize`); and a dict `target` must carry its metric key (`"metric"` for
target-distribution; `"metric"` or `"anchor_metric"` for dsir) so it can't KeyError after the
feature pass.

## Anatomy

```jsonc
{
  "name": "dynamic",
  "define": {                                  // inline DSL metrics (see dsl.md)
    "speed": {"source": "drone.pos", "map": "speed", "reduce": "mean"},
    "turn":  {"source": "drone.att", "map": "angle_delta", "reduce": "mean"}
  },
  "binning": {"axes": [{"metric": "speed", "bins": 10},
                       {"metric": "turn", "bins": 10}]},   // quantile bins per axis
  "selection": {
    "method": "top-k",                         // uniform-bins | weighted | top-k | target-distribution
    "weights": {"speed": 1.0, "turn": 0.5},    // weighted / top-k
    "normalize": "quantile",                   // quantile (default) | minmax | zscore
    "target": "flat",                          // target-distribution: "flat" | {"metric": name}
    "tag_coverage": {"min_tag_frac": 0.01},    // optional floor per categorical tag
    "group_cap": {"max_group_overshoot": 1.5}  // optional: no group > 1.5× its fair share
  }
}
```

## Selection methods

| method | picks | needs |
|---|---|---|
| `uniform-bins` | each occupied quantile bin uniformly (diversity) | `binning.axes` |
| `weighted` | sampled without replacement ∝ weighted score | `weights` |
| `top-k` | the highest weighted scores (deterministic) | `weights` |
| `target-distribution` | per-bin quota, flat or ∝ a metric | `binning.axes` |
| `filter` | exactly the segments matching DSL-metric predicates | `predicates` |

`uniform-bins` and `target-distribution` land on exactly `min(target_size, pool)` — bins spread
the budget as evenly (or as proportionally) as possible, with uneven bins spilling their
shortfall into fuller ones; they never overshoot when occupied bins outnumber the target.

The **`filter`** method selects clips by criteria rather than ranking — e.g. "every window
where mean speed ≥ 10 and turn-rate ≤ 5":

```jsonc
"selection": {"method": "filter", "predicates": [
  {"metric": "speed", "op": ">=", "value": 10},
  {"metric": "turn",  "op": "<=", "value": 5}
]}   // ops: > >= < <= == !=  (all predicates must hold)
```

Combined with the curated-dataset adapter ([adapters.md](adapters.md)), this is how you look
*within* existing datasets for clips that match a spec — and `write_dataset` then emits only
those (non-matching clips get `-inf` priority and are **never** admitted to the hours budget,
even when it has room to spare).

`selection.predicates` are not exclusive to `method: filter` — they are a **universal
pre-filter that composes with every selection method**. With `method: filter` the predicates
*are* the selection; with any other method (`uniform-bins`, `weighted`, `top-k`,
`target-distribution`, the feature-space methods, …) failing segments are dropped from the
pool before it runs — on the count path (`select`) they're removed outright; on the hours
path (`segment_priorities`) they're priced `-inf`, so `fill_to_hours` never admits them even
when the budget has room to spare:

```jsonc
{"selection": {"method": "top-k", "weights": {"speed": 1.0},
               "predicates": [{"metric": "hp", "op": "<=", "value": 50}]}}
// ranks by speed, but a segment with hp > 50 is never selected, on either path
```

The pipeline's `.filter(metric, op, value)` (see [pipeline.md](pipeline.md)) compiles
straight to a `selection.predicates` entry, under any `.sample(method, ...)`.

Scores combine each metric's values, normalized (`quantile` rank by default — robust to
outliers) and weighted.

### Feature-space methods (optional, `[methods]` extra)

Coverage- and representativeness-driven selection over the metric vectors (a `feature_space`
metric-name list, or the binning axes). Knobs go directly in `selection`.

| method | does | key knobs |
|---|---|---|
| `cluster-balanced` | k-means the joint space, round-robin across clusters (joint coverage) | `clusters` |
| `k-center` | farthest-point cluster order → spreads to the extremes | `clusters` |
| `facility-location` | submodular representativeness (best represents the whole) | `clusters` |
| `dsir` | importance resampling toward a target (flat / `{"metric"}` / `{"anchor_metric","anchor_quantile"}`) | `dsir_bins`, `target` |
| `rare` | kNN density-inverse (atypical footage first) | `knn_k` |
| `typical` | kNN density (prototypical footage first) | `knn_k` |

```jsonc
{"name": "coverage", "feature_space": ["speed", "turn", "altitude"],
 "selection": {"method": "k-center", "clusters": 512}}
```

`rare`/`typical` are contrast arms (difficulty-only selection underperforms coverage at high
pruning). All are CPU-only, seeded, and deterministic.

### Pre-pass filters (optional, `[methods]` extra)

Run *before* selection to prune the pool. Each filter's knobs are its own kwargs:

```jsonc
"filters": [
  {"type": "anomaly", "contamination": 0.02},              // Isolation-Forest outlier removal
  {"type": "dedup", "threshold": 0.98, "per_group": true}  // feature-space SemDeDup, scoped per group
]
```

A filter may carry its own `feature_space` (gate on a different vector space than the selector),
and `per_group: true` scopes it within each `Segment.group` (avoids global dedup skewing the mix).

## Binning

`bin_segments(segments, axes)` assigns each segment an N-tuple bin via **equal-frequency
(quantile)** edges per axis — so each occupied bin hosts the same share of segments and
"slightly active" gets the same budget as "very active" (equal-width bins would pile most
segments into one bucket). Tie-robust: a heavily tied feature (say 90% zeros) keeps the tied
mass in its own bin instead of absorbing everything above it. A NaN feature value on a bin
axis raises (it would silently poison the quantile edges); a constant axis warns — it adds no
selection signal. Vectorized; not parallelized.

## Scorers (learned / whole-pool metrics)

The DSL is per-segment and geometric. For **whole-pool, model-backed, action-aware** signals
(obs→action mutual information, datamodel/validation-loss influence, learned quality) declare a
`scorers` block. A scorer sees the entire pool at once, may load a model, and writes a value
into each segment's features under its output name — which you then reference like any metric:

```jsonc
"scorers": [
  {"name": "action_residual", "as": "informativeness", "obs": "obs", "action": "action", "k": 8}
],
"selection": {"method": "top-k", "weights": {"informativeness": 1.0}}
```

Bare names work too (`"scorers": ["my_scorer"]`, output = the name). This composes with
coverage — e.g. `weights: {"informativeness": 1.0}` *and* a `feature_space` k-center arm — so
quality and diversity are separate, weighted axes rather than an either/or. Register custom
scorers with `register_scorer(name, factory)`; `action_residual` (an action-conditioned
predictive-residual, needs `[methods]`) ships as a reference. Validated at load (scorer must be
registered). A scorer must return exactly one finite scalar per segment (`shape == (n,)`);
malformed or non-finite output is rejected before selection. Reproducibility holds *given the
same pinned model*.

## Tags (declarative, for tag coverage)

Tag coverage needs each segment's `tags` populated — derive them from data with a top-level
`tags` block, no callback needed:

```jsonc
"tags": {"from": ["weapon", "grenade.type"]}   // categorical fields and/or event streams
"tags": ["weapon", "grenade.type"]             // bare-list shorthand — normalized to the dict form
```

A segment's tags become the distinct categorical values observed in its window
(`"weapon:AK-47"`, `"grenade.type:Molotov"`, …). This feeds tag coverage in **both** paths: the
count-based `curate` and the hours-based `write_dataset`. A `tags_of` callback remains as an
escape hatch (and overrides the declarative tags).

## Post-passes

- **tag coverage** — after the base pick, add or swap segments within `target_size` until every
  feasible categorical `tag` reaches a floor count (preferring underrepresented bins), so a rare
  label isn't dropped without expanding the count budget. The floor is rounded up and coverage
  draws only from the predicate-passing pool, so it cannot restore a filtered segment. In
  `write_dataset` this is a budget **reserve**: each tag's highest-priority track is admitted
  first, bounded by `min_tag_frac` of the hour budget.
- **group cap** — drop excess from any `group` exceeding its fair share × `max_group_overshoot`,
  cutting its most over-represented bins first. (`group`/`tags` are set via `group_of` / `tags_of`.)

## The two execution paths a strategy runs through

Every strategy — however it was produced, an operator chain on a pipeline or a hand-written
JSON spec — runs through exactly one of these two paths. `pipeline.collect()`/`.output()` call
straight into them; there's no separate execution engine underneath the pipeline.

**Count-based** — `curate(...)` returns a `CurateResult(selected, all_segments, splits, ...)`:

```python
from kuration import curate, Segmenter
res = curate(schema=schema, adapter=adapter, sources=sources, strategy=strategy,
             segmenter=Segmenter(64, 32), target_size=5000, seed=0,
             group_of=lambda src: src.split("/")[0],   # optional grouping key
             tags_of=None, test_frac=0.1)               # optional source-level split
```

**Hour-budget-based** — `write_dataset(...)` fills a train/test hour budget and writes shards
(see [output.md](output.md)). This is the apples-to-apples way to compare strategies: the
emitted N hours is the strategy's *best* N hours, not a random draw.

Calling either directly (instead of through a pipeline) is the "classic" API: useful when you
already have a `Strategy` object — loaded from a saved `config.json`, built by a job spec, or
assembled by tooling outside this process — and want to run it without reconstructing the
pipeline that produced it.

## Determinism

Segments are sorted by `(source_id, start_idx)` before binning/selection, and all RNGs are
seeded — so results are identical regardless of the order sources were processed. Splits
(`assign_splits`) are leakage-free (a whole source goes to one split) and seeded.
