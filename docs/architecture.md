# Architecture

A design deep-dive: the layers, the contracts between them, and the decisions that make the
SDK dataset-agnostic. For *how to use* it, see [overview.md](overview.md); this is about *why
it's shaped this way*.

## The one idea

Curation is the same pipeline for every dataset — *segment a stream, score the windows, keep
the best, write them out*. What differs between datasets is only **vocabulary** (what fields
exist) and **format** (how to read raw bytes / write shards). So the SDK pushes all
domain knowledge to the edges:

- **Vocabulary → data.** A `SchemaSpec` declares fields and their semantics; the metric
  grammar validates against it. No field names are hardcoded in the engine.
- **Format → two code seams.** An `InputAdapter` (raw → canonical) and an `OutputEncoder`
  (derive a modality tensor). Everything between them is generic and never imports a domain.

Everything else in this document follows from that.

## Layers & data contracts

The pipeline is a chain of stages connected by four small, explicit data types. The contracts
are the architecture — each stage only knows the type before and after it.

```
                         SchemaSpec  (declared vocabulary; validates the grammar; in config.json)
                              │
 raw source ──[InputAdapter]──┼──▶  Tick stream          (canonical per-timestep records)
                              │
        ──[Segmenter]─────────┼──▶  Segment[]             (window metadata: source, idx range)
                              │
        ──[FeatureEngine]─────┼──▶  Segment.features      (built on a per-window WindowBuffer)
                              │        via compile_metric(spec, schema): fn(WindowBuffer)->float
        ──[SegmentScorer?]────┼──▶  Segment.features[…]   (optional whole-pool / learned / action-aware
                              │        scoring — model-backed; outputs land as features)
        ──[bin_segments]──────┼──▶  Segment.bin           (quantile coordinates)
                              │
        ──[Strategy]──────────┼──▶  selected Segment[]    (count)   ── curate()
                              │     or track priorities   (hours)   ── write_dataset()
        ──[tracks_from_ticks]─┼──▶  Track[]               (one entity's timestep sequence = output unit)
                              │
        ──[ShardWriter]───────┴──▶  shards + manifests + config.json
```

| contract | produced by | consumed by | what it is |
|---|---|---|---|
| **`Tick`** | adapter | buffer builder | `{time, entities:{id→fields}, events, globals}` — the canonical record |
| **`WindowBuffer`** | `build_window` | compiled metrics | per-field `(T,E,D)` arrays + `valid` mask + vocabs + events/globals |
| **`Segment`** | segmenter | binning, selection | window metadata + generic `features`/`bin`/`tags`/`group` |
| **`Track`** | `tracks_from_ticks` | shard writer | one entity's record sequence — the indivisible output unit |

The power of these contracts: an adapter only owes a `Tick` stream (no numpy, no metrics); a
metric only sees a `WindowBuffer` (no raw format); selection only sees `Segment.features` by
name (no schema); the writer only sees a `Track` + `OutputSpec` (no curation logic).

## Components, and why each exists

### `schema.py` — SchemaSpec
The seam that replaces hardcoded vocabulary with data. Fields carry a `kind`
(scalar/vector/categorical) and a `semantics` tag (`position`/`angular`/`bitmask`). The tags
are load-bearing: they let *universal* maps stay type-safe without naming fields — `speed`
accepts any `position` vector, `angle_delta` any `angular` field, `bit:` any `bitmask`. A
field's `max_rate` is the outlier clamp. The schema is frozen, round-trippable, and embedded
in `config.json`.

### `dsl/grammar.py` + `buffer.py` — the closed metric grammar
`compile_metric(spec, schema)` returns `fn(WindowBuffer)->float`. The grammar is **closed**
(every token enumerated) and validated at **compile time** against the schema — unknown
fields, a map that doesn't match a field's semantics, a vector fed to a scalar reduce all
raise at load, not mid-run. The reduce machinery (entropy, percentiles, jerk, convex-hull
area, …) is domain-agnostic numpy; only source/map/filter resolution consults the schema.

`WindowBuffer` is the shared, per-window extraction: every field as a NaN-padded `(T,E,D)`
array over the window's entity roster. It's built **once per segment** so a whole strategy's
metrics amortize a single extraction — the reason a profile of many metrics stays cheap.

### `adapters/` — the input seam
`InputAdapter.ticks(uri) -> Iterable[Tick]` is the entire contract. Because building the
`WindowBuffer` from canonical ticks is generic (`build_window`), an adapter does *only* parse
+ group-by-timestep — never anything numeric. The generic `TimeseriesAdapter` covers
"columns of numbers" (CSV/Parquet/rows) with zero per-dataset code; bespoke formats implement
the same tiny method. Adapters register by name so worker processes reconstruct them rather
than pickling a live object.

### `segment.py` — Segmenter & Segment
Overlapping fixed-width windows, in steps or seconds (`sample_hz`). `Segment` holds *generic*
state — `features`/`bin`/`tags`/`group` keyed by name — so adding a metric never touches the
dataclass. The iterator-shaped interface is deliberately streaming-ready; the implementation
is in-memory for now (a conscious "design for streaming, ship in-memory" choice).

### `features.py` — FeatureEngine
Compiles a strategy's `define` metrics against the schema, pulls registered metrics by name,
and runs all of them against one `WindowBuffer` per segment. The single amortization point.

### `binning.py` — quantile binning
Equal-**frequency** (not equal-width) bins: each occupied bin hosts the same share of
segments, so "slightly active" gets the same selection budget as "very active" (equal-width
would pile most windows into one bucket). Vectorized — one `searchsorted` per axis — which is
*why it isn't parallelized*: the compute is a global reduction and beats process-pool IPC.

### `strategy.py` + `select.py` — the selection engine
A `Strategy` is declarative JSON (define + bin axes + method + knobs), validated loudly at
load. The four base methods (uniform-bins / weighted / top-k / target-distribution) operate
purely on `Segment.features`/`bin` *by name* — which is exactly why this layer needed almost
no generalization from the CS2 original. Two shared post-passes patch the result: **tag
coverage** (a floor per categorical label) and a **group cap** (no group exceeds its fair
share). `assign_splits` partitions *sources* into train/test — leakage-free (a whole source →
one split), seeded, group-stratified.

### `output/` — the output layer
Mirrors the input side. A **Track** (one entity's timestep sequence) is the output unit —
the generalization of a per-POV episode. `OutputSpec` declares modalities and routes each
(`field:`/`segment:`/`encoder:`); the `OutputEncoder` is where domain logic (frame decode,
action binning) lives as a plugin; `ShardWriter` does the generic mechanics (chunk,
short-policy keep/pad/drop, container write, manifest entry). `fill.py` is the hours-as-budget
fill: rank tracks by their source's segment priority, fill train/test hour budgets with
source-level leakage-free splits and a group cap. `dataset.py::write_dataset` ties it all
together and writes `dataset_manifest.json` + `config.json`.

## Cross-cutting decisions

- **Closed grammar + semantics tags.** Generality without arbitrary code: the grammar is
  fixed, the *vocabulary* is data. Semantics tags are what keep universal maps safe across
  unknown fields.
- **Validate at compile time, fail loud.** A bad metric token raises at `load_strategy` /
  `compile_metric`, before any data is read. No silent fallbacks — a masked bug is worse than
  a loud one.
- **Amortized `WindowBuffer`.** One extraction per segment, shared by all metrics.
- **Determinism under parallelism.** Segments are sorted by `(source_id, start_idx)` before
  binning/selection; all RNGs are seeded. Results are identical regardless of the order
  sources were processed (and thus of worker count).
- **Hours-as-budget over tracks.** The output volume is *driven* by the strategy (best N
  hours), not seeded by a random draw — and the budget unit is the non-overlapping track, not
  the overlapping analysis window (you can't budget hours over overlapping windows without
  double-counting; this is why the fill lives in the output layer, where the unit exists).
- **Reproducible from the artifact.** Resolved schema + strategy + output spec all live in
  `config.json`; a rerun against the same sources reproduces the dataset.

## Extension points

| to support… | implement / register | not |
|---|---|---|
| a new raw format | an `InputAdapter` (`ticks` → `Tick`s) | editing the engine |
| a derived output tensor (frame, action) | an `OutputEncoder` (`timestep`) | core output code |
| a per-segment metric the grammar can't express | `@register_metric` over a `WindowBuffer` | a grammar change |
| a **whole-pool / learned / influence** metric | a `SegmentScorer` (`score(segments, ctx)`) | the closed grammar |
| a new on-disk container | a `containers.write_shard` branch (extra) | the writer |
| a new dataset's fields | a `SchemaSpec` (data) | any code |

### Two scoring seams, two shapes

The closed-grammar DSL and `@register_metric` are both **per-segment, single-pass, stateless,
model-free** — the right shape for geometric/statistical features. Learned estimators (obs→action
mutual information, datamodel/validation-loss influence, learned quality — the signals the
robotics-curation literature argues actually rank *trajectory quality*) are the opposite:
**whole-pool, often two-stage, model-backed**. Those get the `SegmentScorer` seam instead — a
distinct stage that sees the entire pool, may load a model/GPU/reference set, and writes a
per-segment value into `seg.features`, after which the *existing* selection machinery consumes
it like any other feature.

This keeps two objectives orthogonal and **composable** rather than competing: geometric
*coverage* (k-center / facility / DSIR — "is this region represented?") and learned *quality /
influence* ("is this trajectory good/informative?"). You weight a scorer's output alongside a
coverage priority; you don't choose one or the other. (`needs_buffers=True` asks the pipeline
for each segment's raw obs+action sequences, re-extracted once per source — the honest cost of
sequence-level scoring; feature-only scorers pay nothing.)

## Parallelism & scale

The feature pass (`scoring.py::score_sources`, shared by curate / experiment / write_dataset)
runs per-source over a `ProcessPoolExecutor` when `workers>1`. Each worker **rebuilds the
FeatureEngine in-process** — the compiled DSL metrics are closures and can't be pickled, so
they're recompiled from the JSON `define` + schema (both picklable); the adapter and segmenter
ship as-is. Determinism is preserved by sorting the pooled segments by `(source_id, start_idx)`
before binning/selection, so the result is independent of worker count and completion order.
(Default start method on Linux is `fork`, so the metric registry is inherited; under `spawn`,
custom `@register_metric` functions must be importable at module scope.)

A source that raises during scoring is **skipped-and-logged** (`on_error="skip"`: stderr line
+ collected in the result's `errors` + a summary), never silently dropped; `on_error="raise"`
restores fail-fast. `write_dataset(resume=True)` skips re-encoding any shard whose file already
exists (chunk boundaries are deterministic, so the manifest entry is still recorded), so an
interrupted run only does the missing work.

The hot path (binning) stays vectorized rather than parallelized — IPC would cost more than the
compute. Interfaces are iterator-shaped so a streaming segmenter can later consume a lazy tick
iterator without changing callers; today's implementation materializes per source (the next
scale step if corpora outgrow memory).

## Lineage: generalizing a domain-specific pipeline

The SDK was lifted from a working CS2→world-model curation pipeline (`asg2diamond`). What was
bound, and what became data:

| was hardcoded there | became here |
|---|---|
| player fields (`pos`/`ang`/`hp`/`btn`/weapon) | `SchemaSpec` fields + semantics tags |
| the `data.json` tick shape, `build_window_arrays` | `InputAdapter` → `Tick` → generic `build_window` |
| `speed` clamp `MAX_LEGAL_VZ`, yaw wrap, 16 Hz | field `max_rate`, `angular` semantics, `sample_hz` |
| button-bit / flag maps | `bitmask` fields with a declared `bits` map |
| `objects` (weapon/grenade tags), `map_name` cap | generic `tags` coverage, `group` cap |
| per-POV clip = output unit; mp4/HDF5 writer | `Track` + `OutputEncoder` + `ShardWriter` |

The selection engine (binning, strategy methods, splits) was *already* domain-agnostic there —
it keyed on features by name — so it ported almost verbatim. The work was at the two ends.

## Invariants (don't break these)

1. The core imports no domain. Domain enters only via adapter / encoder / registered metric.
2. Unknown grammar tokens raise at load. No silent fallbacks.
3. `config.json` fully determines a rerun (schema + strategy + output spec embedded).
4. Canonical segment sort precedes binning/selection (determinism).
5. Splits are source-level (leakage-free): a whole source goes to one split.
