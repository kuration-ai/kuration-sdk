# DESIGN.md — the progressive-disclosure and pipeline-carrier thesis

## What this document is

Every layer of kuration was designed around one thesis, restated independently in at least
five places (`docs/pipeline.md`, `docs/customization.md`, `docs/overview.md`,
`docs/documentation-contract.md`, `README.md`) but never written down once, end to end, as a
single governing statement with its concrete mechanism, its deliberate limits, and its known
gaps. That's what this document is for — a durable reference for **design decisions**, read
before adding a seam, a pipeline method, or a new escalation rung, by a human or an LLM.

**It is not** another design deep-dive (`docs/architecture.md` already owns "why this layer
works this way"), not the build log (`PLAN.md` owns "what shipped, in what phase"), and not
usage documentation (`docs/usage.md`, `docs/pipeline.md` own "how to call this"). Where those
documents already say something well, this one cites and links rather than re-explains
(`docs/documentation-contract.md`: "Content has one canonical home"). What's new here is
the cross-cutting synthesis, including the pipeline-carrier direction adopted after the first
gameplay analyzers exposed a gap between pipeline configuration and stateful enrichment.

## The thesis

> A beginner should be able to drive the entire SDK through one `Pipeline` object,
> knowing almost nothing about how it works. An expert should be able to replace or extend any
> single stage without forking the engine or learning the rest. Both must be the *same*
> pipeline, not two implementations that drift.

For input-backed work, this has a second, equally important consequence:

> The `Pipeline` is the durable carrier of the work in progress. An operation that augments,
> qualifies, or interprets input data should return a new `Pipeline` and keep its result on that
> pipeline. A detached result is appropriate only when the caller explicitly asks to inspect,
> analyze, compare, publish, or otherwise terminate the pipeline.

This is stated at the API level in `docs/pipeline.md` (the pipeline "calls the existing
engine... the explicit APIs remain available"), turned into documentation *policy* in
`docs/documentation-contract.md` ("The pipeline API is the default teaching interface. The
classic API is the explicit path... Documentation should name the path being used"), and is the
implicit shape of `docs/customization.md`'s opening sentence: every seam — adapters, metrics,
scorers, batch transforms, encoders, selection methods, filters, containers, presets, operators,
analyzers — "resolve... through the same descriptor model" (`docs/customization.md`). One
mechanism, many depths, not many mechanisms.

## The decision ladder

For each concern, there is a beginner-facing default and an expert-facing escape hatch. This is
the table to consult when deciding where a new piece of behavior belongs.

| concern | beginner surface | expert escape hatch | why the ceiling sits there |
|---|---|---|---|
| Schema | `input(src)` infers it (`infer_schema`) | explicit `SchemaSpec` / `fields=` | inference is "deliberately conservative and loud … no silent guessing" (`infer.py`) — it refuses to guess rather than guess wrong |
| Source format | file-kind sniffing picks an adapter (`pipeline.md`) | `input(adapter=<name or InputAdapter>, sources=[...])` — "escape hatch — no inference" (`pipeline.md`) | the pipeline layer only builds shipped adapters; a bespoke raw format needs a real `InputAdapter` |
| Per-segment metric | `.measure(name="...")` against the closed DSL grammar | `@register_metric` — "the escape hatch in `registry.py`" (`dsl/grammar.py` docstring) | the grammar is closed and enumerated *on purpose* (loud validation); an escape hatch is one plain function over a `WindowBuffer`, sharing the same signature so the registry treats both interchangeably (`dsl/registry.py`) |
| Whole-pool / learned scoring | `.score(name="...")` | `SegmentScorer` + `register_scorer` | this shape is deliberately *not* bent through `@register_metric` — whole-pool/learned/action-aware estimators are "the wrong shape" for a per-segment, single-pass, stateless function (`scorers.py` docstring) |
| Bounded model/device compute | `.transform(name="...")` | `BatchTransform` + `register_batch_transform` | needs custom NumPy/model code over declared `field:`/`feature:`/`media:` roles — declarative strategy can't express arbitrary compute |
| Selection | `.bin()/.filter()/.sample(n=…/hours=…)` | raw `Strategy` dict / `load_strategy()` | strategy stays pure data so it round-trips through `config.json` and reproduces from the artifact alone (`strategies.md`) |
| Reusable curation decisions | generated pipeline method (`pipeline.pixel_shift(...)`; `.apply(name, **params)` is the compatibility spelling for serialized integrations) | `register_operator` | an operator is "a composition seam, not a native execution-stage seam" (`pipeline.md`) — see Limits below |
| Stateful / cross-segment checks | generated domain method that runs the named analyzer and returns a pipeline with namespaced metadata (for example `.action_state_consistency(...)`) | `.analyze(name, parameters=...)` for a detached result, or `get_analyzer()` + drive `add_batch`/`check_dataset` by hand | stateful work is still explicit and uses the analyzer lifecycle; the default domain spelling preserves the pipeline while `analyze` remains the intentional result boundary |
| Pipeline enrichment metadata | immutable, JSON-safe `pipeline.metadata`, populated by pipeline-returning methods | the registered analyzer/transform result plus a generic metadata-attachment primitive | enrichment must survive further pipeline calls without becoming a schema field, selection feature, or ad hoc side object |
| Output | `.output(path, preset="…")` (resolved by `output_preset`) | custom `OutputEncoder` / `containers.write_shard` branch | mirrors the input side: declarative `OutputSpec` + pluggable encoder + generic `ShardWriter` (`output.md`) |
| Orchestration | JSON job + `kur-curate`/`kur-init`/`kur-preview` | `curate(...)` / `write_dataset(...)` called directly in Python | "every supported job-spec execution key maps onto its kwargs, so graduating from a job file to Python is mechanical" (`jobs.md`) |

Read top to bottom, each row's beginner surface is strictly less powerful than the row's
escape hatch, and every escape hatch is either a *named, registered component* or explicit
data/direct calls into the *same engine* (a `SchemaSpec`, a `Strategy` dict,
`curate()`/`write_dataset()`) — never a fork of the engine.

## The one mechanism underneath every rung

`Registry` (`registry.py`, "Typed, discoverable component registry with parent/overlay
isolation") is the single generic index behind all eleven kinds of extension point
(`adapter, metric, scorer, encoder, method, filter, container, preset, operator, transform,
analyzer` — `_KINDS` in `registry.py`). Every row in the ladder above is a `register_*()` call into
this same structure, validated the same way (JSON-schema parameter descriptors, loud on
duplicate names), discoverable the same way (`kur-list`, `kur-describe`, `list_components`,
`describe_component`). An expert doesn't learn eleven different plugin systems — they learn one,
and apply it eleven times.

## The single-`Pipeline` guarantee

Escalation only stays safe because there is exactly one `Pipeline` class, never a per-domain
subclass. `kuration.gameplay` is the SDK's own proof of this, registered through the same
`kuration.plugins` entry point a third party would use
(`pyproject.toml` `[project.entry-points."kuration.plugins"]`):

1. `import kuration.gameplay` runs `register()` immediately at import
   (`src/kuration/gameplay/__init__.py`), which calls `pixel_shift.register()`,
   `action_state_consistency.register()`, `slam.register()`, and registers a gameplay
   stats enricher. The gameplay package owns the method implementations and registration:
   importing `kuration` alone must never define or eagerly register gameplay methods (normal
   plugin discovery may later load the same registrations).
2. `pixel_shift.register()` calls `register_batch_transform(...)` **and**
   `register_operator(...)` under the name `aintropy/pixel-shift`
   (`gameplay/pixel_shift/__init__.py`).
3. `Pipeline.__getattr__` (`pipeline.py`) resolves `pipeline.pixel_shift(...)` purely by
   looking up the registry at attribute-access time — no subclassing, no method defined on
   `Pipeline` anywhere in the gameplay package. `assert type(pipeline) is Pipeline` holds even
   after `.pixel_shift(...)` (`domains.md`).
4. A collision (a plugin operator whose generated method name shadows a real `Pipeline`
   attribute like `report`/`window`/`filter`) is rejected **at registration time**
   (`operators.md`, enforced by `operator_name_conflict()` in `pipeline.py`) — the
   ladder can't silently break itself.

So an expert who writes `register_batch_transform` + `register_operator` against the low-level
seams hands a beginner a plain pipeline method, with zero new concepts to learn and zero risk of
silently shadowing something that already existed.

## The pipeline is the carrier of input-backed state

The pipeline API originally treated `Pipeline` mainly as an immutable execution-plan builder.
That is necessary but not sufficient for domain workflows. A gameplay check can read the same
input, window, media join, options, and registry view as the rest of the chain; returning a loose
dict at that point makes the caller manually keep two related objects synchronized. The default
domain API should instead keep that interpretation on the pipeline that produced it.

There are three distinct boundaries:

| boundary | reads input? | returns | examples |
|---|---:|---|---|
| **configuration** | no | a new `Pipeline` | `.window()`, `.transform()`, `.filter()`, `.sample()` |
| **enrichment** | possibly | a new `Pipeline` carrying materialized metadata | gameplay `.action_state_consistency(...)`; future domain audits or classifications |
| **observation / terminal** | possibly | a result, visualization, or artifact | `.analyze()`, `.stats()`, `.report()`, `.collect()`, `.output()`, `.check()`, `.experiment()` |

An enrichment is allowed to execute eagerly when its result cannot exist without reading data,
but the method name and documentation must say that it reads input. It must not silently change
the adapter, schema, source list, segmenter, selection strategy, or output destination. It returns
an immutable clone whose metadata contains the new evidence. Downstream selection and output
calls preserve it. Any later call that changes the input/schema/media join/window/registry view on
which the evidence depended must fail loudly or mark the attachment stale; it may never present
old evidence as if it described the new plan. Calling `.analyze(...)` remains valid and
intentionally different: it is the generic, explicit request to leave the pipeline and receive
the analyzer result directly.

### `Pipeline.metadata` contract

The public field is `pipeline.metadata`, not a `SchemaSpec.fields` entry and not a segment
feature. That distinction is load-bearing:

- **Schema fields** describe values available at each input tick.
- **Segment features** are finite scalars that selection can filter, bin, weight, or score.
- **Pipeline metadata** describes the pipeline or a whole-source/cross-segment interpretation of
  its input. It is not automatically a selection signal.

`Pipeline.metadata` is a read-only property that returns a defensive deep copy of an internal,
JSON-safe mapping.
Enrichments write under a stable namespace, normally
`metadata["<domain>"]["<capability>"]`; sibling capabilities must not share an unstructured
top-level bag. Reusing an occupied key raises unless the caller explicitly asks to replace it.
Every attachment records enough identity to explain where it came from: stable component name
and descriptor/fingerprint, execution mode, normalized public parameters (or a digest/reference
when values are large or sensitive), result summary, warnings/errors, and the source/pipeline
identity against which it was computed.

The source/pipeline identity is also the staleness guard. The implementation allows
ordinary downstream decisions (`filter`, `bin`, `score`, `tag`, `split`, `group`, `sample`,
expectations, and output-only options) to preserve an attachment, while rejecting upstream
changes such as a new media join, window, schema/rate refinement, input-role remapping, or
registry switch after materialization. Broader dependency-aware invalidation can come later;
silently stale metadata cannot.

Large per-instance ledgers do not belong inline. Store a bounded summary plus an explicit
artifact/reference or keep the detailed analyzer instance on the expert path. Publication also
remains explicit: `.describe()` may expose the in-memory metadata, while `.output()` must apply
the same path/privacy and JSON-safety rules as existing provenance before placing a publishable
subset in an artifact. Attaching metadata must never leak local source paths merely because a
pipeline is later written.

### Domain methods, not domain subclasses

A maintained domain capability that enriches input should have a direct generated method on the
ordinary `Pipeline`. Its implementation and registration live only in the domain package and
are activated by the domain import or the ordinary plugin-discovery path; core provides only the
generic metadata and execution plumbing.

For action-state consistency, the intended surface is:

```python
import kuration.gameplay  # registers the method; `import kuration` alone does not
from kuration import input

pipeline = (
    input(rows, schema=schema)
    .join_video(videos)
    .window(64, stride=32)
    .action_state_consistency(
        baselines=baselines,
        controls=["W"],
    )
)

verdict = pipeline.metadata["gameplay"]["action_state_consistency"]["result"]
```

The method should reuse the registered `aintropy/action-state-consistency` analyzer and the same
bounded stream/dataset drivers as `.analyze(...)`; it must not implement a second measurement
loop. Registering the same stable component name as an operator is acceptable because registry
kinds are distinct. The operator expansion must end in a generic pipeline metadata-attachment
primitive so it still obeys the immutable operator contract and its call remains visible in
`_ops`/provenance.

The detached form remains available when the result itself is the goal:

```python
result = pipeline.analyze(
    "aintropy/action-state-consistency",
    {"anomaly_type": "action_state", "baselines": baselines},
    controls=["W"],
)
```

This is not two analysis implementations. It is one analyzer and one driver with two return
policies: attach-to-pipeline (domain default) or return-result (generic terminal).

## Cost discipline is part of the same promise

Progressive disclosure isn't just an API-surface story — it's a runtime-cost story too.
`import kuration` imports no domain code and no heavy dependency; `import kuration.gameplay`
registers pixel-shift, action-state-consistency, and SLAM but **still** does not import OpenCV,
PyTorch, FAISS, PyAV, Matplotlib, or Trimesh (`domains.md`,
`documentation-contract.md`, pinned by a subprocess test:
`tests/test_gameplay.py`). A beginner who never calls `.pixel_shift(...)` never pays for it. An
expert who does gets it lazily, on first real use.

The same discipline governs execution: the in-memory, single-process path is the default, and
every scale feature (`executor`, `workers`, `streaming`, `resume`, `on_error`) is an opt-in,
additive knob (`.options(...)` in `pipeline.md`'s operator table, or the same kwargs on
`curate()`/`write_dataset()`) — scaling up never churns the beginner's default path.

## Two escalation axes, not one

It's tempting to flatten "pipeline vs. classic" and "CLI/job vs. Python" into a single
beginner/expert line. They're actually orthogonal:

|  | **declarative (JSON job / preset)** | **imperative (Python)** |
|---|---|---|
| **pipeline ergonomics** | `kur-init` scaffold → `kur-curate job.json` | `input(...).window(...).measure(...).sample(...)` |
| **classic/explicit control** | hand-authored job JSON referencing named custom components | `SchemaSpec`, `load_strategy`, `curate()`/`write_dataset()`, arbitrary `group_of`/`tags_of` callables |

A JSON job can already reference a custom, expert-registered adapter or scorer *by name* — the
real boundary `docs/jobs.md` draws is data/names (jobs) vs. arbitrary code
(`write_dataset(...)` directly): "If you need arbitrary code, call `write_dataset(...)`
directly... graduating from a job file to Python is mechanical" (`jobs.md`). Don't
assume "the CLI is beginner, Python is expert" — plenty of expert work (registering a new
operator, writing a `BatchTransform`) happens once, in Python, and is then consumed by *both*
a Python pipeline chain and a JSON job, at the same rung.

## Deliberate limits — where disclosure stops on purpose

Progressive disclosure is bounded, and the bounds are load-bearing, not accidental gaps:

- **The core imports no domain code, ever** (`architecture.md` "Invariants", `PLAN.md`
  design principle 7). There is no "advanced mode" that lets domain logic into
  `src/kuration/core/`; the seam table is exhaustive, not a suggestion.
- **The closed grammar has no runtime toggle to loosen it.** Unknown DSL/strategy tokens raise
  at load, always — "no silent fallbacks" (`architecture.md` "Invariants"). The escape hatch is a new
  named metric, never a bypass flag.
- **Operators are a shallow seam, easy to mistake for the top of the ladder.** They read as the
  most chain-native extension mechanism, but `pipeline.md` and
  `customization.md` both stress a handler "cannot introduce a new terminal or bypass
  the validation and execution contracts of existing operators" — it can only *compose* already
  validated operations. The deeper seams (`BatchTransform`, `SegmentScorer`, a custom
  `InputAdapter`) are where genuinely new execution behavior belongs; `register_operator` is
  packaging, not new capability.
- **Analyzers are never implicit selection stages.** Cross-segment, stateful checks are not
  safely composable as stateless per-segment/whole-pool scores. A named domain enrichment may
  explicitly drive an analyzer and attach its result to a returned pipeline, but it must reuse
  the analyzer lifecycle, disclose the input read, and never make its verdict a selection
  feature without a separate explicit conversion.
- **`kur-init`'s scaffold is a starting point, explicitly not a recommendation** — "it picks
  the first scalar field as a binning axis because it cannot know which metric matters. Edit
  `strategy` before trusting a run" (`jobs.md`). The beginner path is honest about being
  a beginner path.

## Adherence audit (2026-07-28)

This audit covers the public core surface, maintained gameplay package, examples, tests, and
documentation. It intentionally excludes vendored SLAM model internals: those are an embedded
runtime implementation, not kuration API design.

### Core

| area | status | finding |
|---|---|---|
| `input()` and pipeline configuration (`pipeline.py`) | adheres | input, schema refinement, media joins, segmentation, features, selection, expectations, and execution options all preserve the single immutable pipeline |
| registered operators (`pipeline.py`, `registry.py`) | adheres | generated methods return a derived `Pipeline`, retain registry identity/provenance, and reject method-name collisions |
| transforms/scorers/metrics | adheres | they contribute segment features inside execution rather than inventing domain result containers |
| inspectors (`head`, `describe`, `preview`, `explain`) | intentional boundary | they explicitly ask to observe pipeline state or bounded input and therefore return inspection values |
| terminals (`collect`, `review`, `output`, `report`, `stats`, `check`, `experiment`) | intentional boundary | each method names the result/artifact it produces; detached output is the point of the call |
| analyzer drivers, `enrich`, and `analyze` | adheres | one driver owns stateful execution; `enrich` attaches the result to a pipeline while `analyze` intentionally returns it |
| classic functional APIs and CLI commands | adheres as escape hatches | `curate`, `write_dataset`, analyzer drivers, reports, and CLI outputs are deliberately explicit non-pipeline/result surfaces |

`Pipeline.metadata` is defensively copied, preserved by cloning, described and published with
the pipeline, and populated through strict-JSON `with_metadata`/`enrich` primitives.

### Gameplay

| capability | current primary surface | status | direction |
|---|---|---|---|
| Pixel Shift segment features | `.pixel_shift(...) -> Pipeline` | adheres | keep; it is the model for domain registration and lazy optional dependencies |
| Pixel Shift per-instance evidence | `PixelShiftReport` side object | partial | keep as an expert detailed-ledger API; add pipeline metadata only if a bounded summary/reference has a concrete downstream use |
| Action-state consistency | `.action_state_consistency(...) -> Pipeline` | adheres | the companion operator runs the existing analyzer driver and attaches its result under `metadata["gameplay"]["action_state_consistency"]`; `.analyze()` and `action_state_checker()` remain deeper paths |
| SLAM trajectory input | `input_slam(...) -> Pipeline`, `gameplay.slam_pipeline(...) -> Pipeline` | adheres | keep both as pipeline factories; the latter may internally produce/reuse a reconstruction artifact |
| SLAM motion and quality | `.slam_motion_diversity(...)` / `.slam_quality(...) -> Pipeline` | adheres | keep; both are import-scoped generated methods |
| SLAM reconstruction | `SLAMReconstructor` / `SLAMReconstruction` plus `slam_pipeline(...)` | mostly adheres | teach `slam_pipeline(...)` as the default because it returns a pipeline; retain reconstruction objects as expert source-creation/artifact APIs |
| SLAM selection and point-cloud curation | `curate_slam_gameplay(...)`, `curate_point_cloud(...)` | intentional terminal | these calls explicitly create selection/point-cloud outputs; do not disguise them as input enrichments |
| SLAM preflight | `preflight_slam(...) -> SLAMPreflightResult` | intentional inspector | it exists to report readiness before execution |
| Video comparison | `compare_videos(...) -> VideoComparison` | intentional inspector | comparison/render/save is the requested output, not input enrichment |
| Gameplay stats | `pipeline.stats() -> StatsReport` with a registered domain enricher | intentional terminal | stats already uses the core terminal and attaches domain sections to that report; it should not mutate pipeline metadata |

The public grouping in `gameplay/__init__.py` is useful, but grouping free functions is not a
substitute for pipeline methods when input is already represented by a pipeline. The rule is not
"every function must return `Pipeline`." It is: every maintained **input-enrichment**
capability gets a pipeline-preserving default; inspectors, analyzers requested for their result,
artifact producers, direct model runtimes, and low-level helpers remain honest about their
different boundary.

### Documentation and examples

- `docs/pipeline.md`, `docs/action-state-consistency.md`, `docs/api.md`, and
  `examples/07_action_state_consistency.py` teach `.action_state_consistency(...)` first and
  retain `.analyze(...)` as the detached-result alternative.
- `docs/pixel-shift.md` correctly teaches `.pixel_shift(...)` first; `PixelShiftReport` remains
  the detailed expert ledger path.
- `docs/slam.md`, `README.md`, and `examples/06_gameplay_operator_methods.py` already teach
  pipeline-returning gameplay methods. The example should add action-state once it can run
  without obscuring the model-free SLAM demonstration, or a focused replacement should cover
  the new method.
- The test suite protects import laziness, one `Pipeline` type, dynamic gameplay registration,
  analyzer driving, operator collision behavior, metadata immutability/conflicts,
  attachment provenance/privacy, staleness guards, and the direct action-state method.

## Implemented migration

The migration was implemented in small contracts while preserving `.analyze(...)`
compatibility:

1. **Pipeline metadata state.** Initialize an empty mapping in `Pipeline.__init__`, expose it
   through a read-only property returning a defensive deep copy, deep-copy it in `_clone`,
   include it in `describe`, and define collision/replace behavior. Validate JSON safety at
   attachment time.
2. **Core enrichment primitive.** It accepts a namespace/key, a materialized JSON-safe
   value, and component/provenance context; returns a clone and records the attachment in
   `_ops`. It is domain-neutral, and `.output()` publishes a path-safe JSON representation in
   `config.json`.
3. **Factored analyzer return policy.** Reuse `run_stream_analyzer` /
   `drive_dataset_analyzer` once, then either return the result (`analyze`) or attach it
   (enrichment). Do not rerun an analyzer merely to support both surfaces.
4. **Action-state companion operator.** Under the existing
   `aintropy/action-state-consistency` name, add an operator registration in the gameplay
   module. Its generated `.action_state_consistency(...)` method validates the same parameters
   and options, drives the same analyzer, and attaches the result to the gameplay namespace.
   Importing core alone must not reveal the method.
5. **Repeated-call semantics.** Default to a loud conflict for the same metadata key.
   `metadata_key=` labels multiple checks and `replace=True` is the only overwrite path.
6. **Validity across later calls.** Bind each materialized attachment to the input/window/
   registry identity it analyzed. Preserve it through downstream selection/output configuration;
   reject upstream mutations instead of silently presenting stale evidence.
7. **User surfaces.** Change the action-state guide, pipeline/API references, gameplay
   examples, and docstrings so the pipeline-returning method is primary and `.analyze(...)`,
   checker instances, and manual loops are progressively deeper alternatives.
8. **Contract tests.** Cover method appearance only after `import kuration.gameplay`,
   unchanged `type(pipeline) is Pipeline`, original-pipeline immutability, metadata preservation
   through downstream operators, upstream-change rejection, namespace collision failure,
   analyzer-result parity, single-pass execution, JSON safety, descriptor/provenance capture,
   path redaction, artifact publication, optional-dependency laziness, registry overlays, and
   stream/dataset modes.
9. **Maintained-domain review.** For each new gameplay capability, classify it as
   configuration, enrichment, inspector, or terminal during review. A free-standing result
   object in the primary input path requires an explicit reason in its canonical documentation.

## Known tensions (surfaced, not smoothed over)

- **`PLAN.md` predates the pipeline layer.** Its "non-negotiable" design principles
  and phase log never mention `input()` or pipeline operators at all — they
  were written for the classic-API-only era. If you're looking for *why* the pipeline layer was
  added on top, that rationale lives in `docs/pipeline.md` and
  `docs/documentation-contract.md`, not in the historical build log. Treat `PLAN.md` as
  authoritative for the classic engine's invariants, and this document plus `pipeline.md` as
  authoritative for the two-tier API story layered on top later.
- **Job-spec vs. Python is not a strict subset of pipeline vs. classic** — see the two-axes table
  above. Don't reason "CLI = simple, Python = powerful"; reason "declarative-by-name" vs.
  "arbitrary code," independently of pipeline vs. classic.
- **Every extension you add should be checked against the ladder, not just shipped.** Before
  registering a new component kind or adding a `Pipeline` method by hand, ask: does an existing
  row in the decision ladder already cover this shape? If a new capability doesn't fit any of
  the eleven registry kinds, that's a signal to extend the registry's kind set deliberately
  (rare), not to special-case the `Pipeline` class.

## Using this document

When adding a new capability, ask, in order:
1. Which row of the decision ladder does this belong to? (If none fit, that's worth a
   conversation before writing code — see the tension above.)
2. Does the beginner-facing default stay a strict subset of the expert escape hatch — could a
   pipeline method for this be *generated* from a registry entry, the way operators are, rather
   than hand-written?
3. Does it preserve "one `Pipeline` type" and the collision guard, so an expert's registration
   can't silently break a beginner's existing call?
4. Does it stay import-lazy and loud-on-missing for any optional dependency it needs, so a
   beginner who never touches it never pays for it?
5. Does it fail loudly at load/validation time rather than silently, consistent with the closed
   grammar's "no silent fallbacks" invariant?
6. If it augments input-backed work, does its default surface return a pipeline and attach
   namespaced, provenance-bound metadata — or is it honestly an inspector/terminal whose result
   is the point of the call?

If the answer to all six is yes, it belongs in the SDK exactly the way every existing seam
does. Update the canonical home for that layer (`docs/customization.md`, `docs/operators.md`,
etc. — see `docs/documentation-contract.md`'s ownership table) rather than duplicating an
explanation here. This document owns the stable thesis, cross-cutting contracts, and dated
adherence audit; implementation detail still belongs in the architecture and canonical feature
guides.
