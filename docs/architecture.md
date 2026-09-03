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
- **Format/model → explicit seams.** `InputAdapter` and `MediaProvider` produce canonical/lazy
  input; `BatchTransform` derives bounded scalar features; `OutputEncoder` derives application
  modalities. Orchestration between them remains generic.

Everything else in this document follows from that.

## Layers & data contracts

The pipeline is a chain of stages connected by four small, explicit data types. The contracts
are the architecture — each stage only knows the type before and after it.

```
                         SchemaSpec  (ordinary vocabulary + auxiliary MediaSpec metadata)
                              │
 raw source ──[InputAdapter]──┼──▶  Tick stream          (canonical per-timestep records)
                              │
        ──[Segmenter]─────────┼──▶  Segment[]             (window metadata: source, idx range)
                              │
        ──[FeatureEngine]─────┼──▶  Segment.features      (built on a per-window WindowBuffer)
                              │        via compile_metric(spec, schema): fn(WindowBuffer)->float
        ──[BatchTransform?]───┼──▶  Segment.features[…]   (bounded field/feature/media batches)
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
| **`Tick`** | adapter | buffer builder/transforms/output | `{time, entities, events, globals, media}`; media values are lazy references |
| **`WindowBuffer`** | `build_window` | compiled metrics | per-field `(T,E,D)` arrays + `valid` mask + vocabs + events/globals |
| **`Segment`** | segmenter | binning, selection | window metadata + generic `features`/`bin`/`tags`/`group` |
| **`Track`** | `tracks_from_ticks` | shard writer | one entity's record sequence — the indivisible output unit |

The power of these contracts: an adapter only owes a `Tick` stream; auxiliary frames stay lazy
and outside ordinary buffers; a
metric only sees a `WindowBuffer` (no raw format); selection only sees `Segment.features` by
name (no schema); the writer only sees a `Track` + `OutputSpec` (no curation logic).

## Components, and why each exists

### `schema.py` — SchemaSpec
The seam that replaces hardcoded vocabulary with data. Fields carry a `kind`
(scalar/vector/categorical) and a `semantics` tag (`position`/`angular`/`bitmask`). The tags
are load-bearing: they let *universal* maps stay type-safe without naming fields — `speed`
accepts any `position` vector, `angle_delta` any `angular` field, `bit:` any `bitmask`. A
field's `max_rate` is the outlier clamp. The schema is frozen, round-trippable, and embedded
in `config.json`. Its separate `media` namespace carries `MediaSpec` shape/layout/color/timing
metadata without exposing image tensors to the metric grammar.

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
also has a bounded deque streaming implementation; streaming remains opt-in.

### `media.py` + `transforms.py` — lazy media and bounded model execution
`FrameRef` is a pickle-safe locator, not decoded pixels. `MediaProvider` probes streams, yields
presentation-ordered references, and decodes only requested bounded batches. `VideoAdapter` and
`JoinedMediaAdapter` expose the same `media:<column>` contract for video-only and multimodal
sources. Registered `BatchTransform`s consume `field:`/`feature:`/`media:` inputs, validate media
semantics before decode, load once per runner, and return finite scalar features aligned to stable
segment IDs.

The tokens a transform declares at registration are *roles*, not fixed batch keys: a per-use
`inputs=` remapping substitutes one same-kind token per role (validated at pipeline-construction
time), and `ctx.inputs` / `ctx.input_token(role)` publish the resolution, so an implementation
never hardcodes a key that a remapped use would not have. `field:` batches are
`(B,T,E,D)` — numeric entity fields keep the entity/dim axes even for a one-entity scalar field —
and are NaN-padded to the widest extent in the batch, so every transform also receives
`ctx.valid_steps` (`valid_steps` in the batch mapping): the real timestep count per row, which
makes a ragged batch's padding observable instead of silently averaged. `lookbehind`/`lookahead`
may be one shared tick count or a `{declared input token: count}` mapping, so a transform that
only reads one input's boundary context does not pay to extract (or decode) the others'.

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
(`field:`/`segment:`/`encoder:`/`media:`); provider-backed media uses batch decode, while the
`OutputEncoder` handles application derivations and may implement `timestep` or `encode_batch`;
`ShardWriter` does the generic mechanics (chunk,
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
- **Auditable from the artifact.** Resolved schema + strategy + output spec all live in
  `config.json`. A replay also requires the same content-addressed or immutable source versions
  and compatible component implementations.

## Extension points

Every row below is a real, distinct registration contract — building a genuinely new selection
algorithm, filter, or scorer still means implementing one of these, not writing a `Pipeline`
combinator (`register_operator`'s own contract is a pipeline-DAG macro, one layer above these,
and forcing a new algorithm through it means "branching in selection core" exactly as the table
says not to). But every *built-in* Method/Filter/Scorer is additionally exposed as a named
`register_operator` wrapper (`src/kuration/core/builtin_operators.py`) purely for discoverability
— see [operators.md](operators.md)/[operator-inventory.md](operator-inventory.md) — so a user
never needs to learn "Method" vs. "Filter" vs. "Scorer" as vocabulary to find or call a built-in
by name; only someone *implementing* a new one needs the distinction below.

| to support… | implement / register | not |
|---|---|---|
| a new dataset's fields | a `SchemaSpec` (data) | any code |
| a new raw format | an `InputAdapter` (`ticks` → `Tick`s) | editing the engine |
| a new media decoder | a `MediaProvider` (`probe`/`iter_refs`/`decode_batch`) | eager frame fields |
| a bounded model/device feature | a registered `BatchTransform` | repurposing the metric DSL |
| a derived output tensor | an `OutputEncoder` (`timestep`/`encode_batch`) | core output code |
| a per-segment metric the grammar can't express | `@register_metric` over a `WindowBuffer` | a grammar change |
| a **whole-pool / learned / influence** metric | a `SegmentScorer` (`score(segments, ctx)`) | the closed grammar |
| a stateful stream or artifact audit | `register_analyzer`; expose `Pipeline.enrich` or a domain operator when its result should stay on the pipeline | pretending it is a segment metric |
| reusable pipeline composition | `register_operator` | a second pipeline implementation |
| a new pool-ranking algorithm | `register_method` | branching in selection core |
| a reusable pre-selection exclusion | `register_filter` | an undocumented predicate |
| a new on-disk container | `register_container` | the writer |
| a reusable output policy | `register_preset` | bypassing `OutputSpec` validation |

### Plugin discovery

Registry extensions can ship without a custom CLI wrapper. A job may list import-side-effect
modules under `plugins`, or an installed distribution may expose a zero-argument initializer:

```toml
[project.entry-points."kuration.plugins"]
my_plugin = "my_package.kuration_plugin:register"
```

Discovery is lazy and idempotent. Explicit modules and entry-point modules are propagated into
spawned feature workers only when required by that execution registry. Registered implementation
modules—not just entry-point initializer modules—are fingerprinted in run provenance, alongside
the descriptors for registered metrics and transforms. Jobs may set
`include_plugin_entry_points: false` to obtain a built-ins-plus-explicit-modules snapshot even if
ambient entry points were discovered earlier in the process. Explicit plugin modules fail fast.
An installed entry point that fails is rolled back transactionally, reported as a runtime warning
and an error record from `load_plugins()`, and does not prevent later healthy entry points from
loading. The diagnostic remains available from `plugin_failures()` after Python's one-shot warning
has been consumed; discovery CLIs print it to stderr, and run provenance contains a stable failure
marker directing operators to `kur-plugin-check` for the full error.

### Progress events

`score_sources`, `curate`, `run_experiment`, and `write_dataset` accept `progress=callback`.
Events always contain `phase` and `event`; phases include `provenance`, `scoring`, `transforms`, `scorers`,
`selection`, and `writing`, while events are `start`, `advance`, and `complete`. Advance events include
`current`, `total`, and where applicable `source` and `status`. Pipeline users pass the same
callback through `.options(progress=...)`.

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
runs per-source over a `ProcessPoolExecutor` when local `workers>1`, or ordered Ray tasks when
`executor="ray"`. Each local process **rebuilds the
FeatureEngine in-process** — the compiled DSL metrics are closures and can't be pickled, so
they're recompiled from the JSON `define` + schema (both picklable); the adapter and segmenter
ship as-is. Determinism is preserved by sorting the pooled segments by `(source_id, start_idx)`
before binning/selection, so the result is independent of worker count and completion order.
(The SDK explicitly uses `spawn`, avoiding unsafe forks from multithreaded runtimes and keeping
behavior stable across Python versions. Workers
rebuild the registry at import, so custom
metrics must be importable at module scope; one registered only in `__main__` raises "unknown
registered metric" in every worker, which surfaces as all sources skipped + the loud
empty-pool error. Rule of thumb: put `@register_metric` functions in an importable module.)

A source that raises during scoring, a transform reread, or scorer-buffer materialization is
**skipped-and-logged** (`on_error="skip"`: stderr line + collected in the result's `errors` +
a summary), never silently dropped. The source is the isolation unit, including any partially
computed transform rows; transform component state is rebuilt before another source continues.
`on_error="raise"` restores fail-fast. A whole-pool model failure that cannot be attributed to
one source still raises. Total failure is loud: if the pool ends up **empty** (every source
skipped, or every source shorter than the window), execution raises rather than letting an
empty-but-valid-looking dataset propagate. `write_dataset(resume=True)` reuses a shard only when
the file matches the byte size and SHA-256 in the previous committed manifest; missing, legacy,
or modified shards are deterministically rebuilt. Two guards make that sound:
local shards use same-directory temp files plus replace, and remote shards publish a completed
attempt object before moving it into place. The complete run and resolved output spec are
fingerprinted in `.kuration_output_spec.json`; the final commit pins that marker's checksum, so
a missing, tampered, or legacy-unpinned marker cannot authorize shard reuse. A separate
`dataset.commit.json` changes to `state=writing` before shard replacement and is published last
as `state=committed` with manifest/config/resume-metadata checksums; readers therefore reject
interrupted or mixed-generation metadata while resume can still checksum prior entries. A
destination-scoped writer lease covers planning through final commit; bounded checkpoints use a
run-fingerprint namespace rather than overwriting another run's cache. Resume and persisted
checkpoint reuse require an adapter fingerprint, local content hash, or immutable remote object
version; mutable logical IDs and metadata-only remote listings cannot authorize stale reuse.

One reader operation likewise stays on one generation: it captures the exact commit bytes,
checks them before and after each shard, and verifies shard bytes from one private snapshot.
Current manifests must agree with the commit-pinned output config's modality names, shapes, and
dtypes. Local current-format shard paths must remain ordinary non-symlink files beneath the
dataset root.

The hot path (binning) stays vectorized rather than parallelized — IPC would cost more than the
compute.

**Out-of-core (`streaming=True`).** A streaming-capable source larger than RAM is handled as a
lazy tick iterator on *both* passes: `Segmenter.stream` slides a `deque(maxlen=window)` (never
more than one window resident) feeding `FeatureEngine.score_window`, and the writer
(`output/stream.py`) flushes shards from the tick stream holding at most `shard_length` records
per active (entity, span). The budget fill needs only counts, so admission happens between the
two passes without holding data. The core contracts are unchanged — only the iteration drivers
differ — so streaming is opt-in and the in-memory path stays the default. NPZ and partitioned
Parquet dataset mode reject streaming because they cannot honor bounded-memory semantics. Two
further caveats: a
`SegmentScorer(needs_buffers=True)` can't hold the whole pool's sequences (use a per-segment
embedding feature instead), and a *total segment pool* exceeding RAM (distinct from a single
big source) would need a further spill/external-sort tier, not yet built.

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
3. `config.json` records the resolved run contract; replay additionally requires the same
   content-addressed or immutable sources and compatible component implementations.
4. Canonical segment sort precedes binning/selection (determinism).
5. Splits are source-level (leakage-free): a whole source goes to one split.
## Physical planning and bounded feature tables

`Pipeline.explain()` compiles resolved logical state into a serializable `PhysicalPlan` before
source rows are read. Window-local maps fuse into pipeline-parallel islands; global, multipass,
or explicitly non-fusible operators form barriers. Registered component descriptors may declare
validated `execution_traits` covering scope, shape, fusion, passes, state bounds, projected
columns, resources, determinism, and checkpoint policy.

With the optional `[scale]` extra, `SegmentBatch` stores stable segment identities and projected
features in bounded Arrow `RecordBatch` values. `StageCheckpoint` writes attempt-specific Arrow
IPC files, validates them, atomically publishes one partition commit, and publishes the stage
commit only after every expected checksum is valid. Restarts reuse only partitions whose logical
plan, physical plan, inputs, schema, components, and parameters match.

The first automatic runtime covers exact local top-k count output without whole-pool scorers,
and uniform-bin count output without whole-pool scorers, global filters, or coverage postpasses.
It fuses streaming segmentation, metrics, universal
predicates, and bounded transforms; exact normalization and stable top-k use a disk-backed index.
Other logical plans continue through the established reference runtime and are identified by
`Pipeline.explain()` warnings until their bounded reducers are implemented.

The hours writer also stores span candidates, overlap summaries, bounds, tags, and budget
ordering externally. Only admitted spans return to Python for source reread and shard emission.
Exact top-k, weighted, and uniform-bin hours plans automatically use checkpointed bounded features and
disk-backed priorities, including predicates, transforms, tag coverage, group caps, subdivision,
and resume. Plans containing whole-pool scorers or global prefilters retain the reference path.

The local and Ray execution contracts also expose ordered `iter_batches` scheduling. Producers
cannot advance beyond `max_in_flight`; the fused local feature island uses this to overlap the
next segmentation batch with one long-lived scoring/transform worker. Stage telemetry records
partition count, checkpoint hits/misses and bytes, candidate/output rows, batch size, and queue
depth in `config.json`. Exact feature-hash dedup uses disk-backed SHA-256 lookup plus full-payload
collision resolution; cosine/near dedup remains a distinct approximate algorithm.
