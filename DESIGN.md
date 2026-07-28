# DESIGN.md — the progressive-disclosure thesis

## What this document is

Every layer of kuration was designed around one thesis, restated independently in at least
five places (`docs/pipeline.md`, `docs/customization.md`, `docs/overview.md`,
`docs/documentation-contract.md`, `README.md`) but never written down once, end to end, as a
single governing statement with its concrete mechanism, its deliberate limits, and its known
gaps. That's what this document is for — a durable reference for **design decisions**, read
before adding a seam, a fluent method, or a new escalation rung, by a human or an LLM.

**It is not** another design deep-dive (`docs/architecture.md` already owns "why this layer
works this way"), not the build log (`PLAN.md` owns "what shipped, in what phase"), and not
usage documentation (`docs/usage.md`, `docs/pipeline.md` own "how to call this"). Where those
documents already say something well, this one cites and links rather than re-explains
(`docs/documentation-contract.md`: "Content has one canonical home"). What's new here is
the cross-cutting synthesis and the tensions that surface only once you read all of them
together.

## The thesis

> A beginner should be able to drive the entire SDK through one fluent `Pipeline` object,
> knowing almost nothing about how it works. An expert should be able to replace or extend any
> single stage without forking the engine or learning the rest. Both must be the *same*
> pipeline, not two implementations that drift.

This is stated at the API level in `docs/pipeline.md` (the fluent API "calls the existing
engine... the explicit APIs remain available"), turned into documentation *policy* in
`docs/documentation-contract.md` ("The fluent API is the default teaching interface. The
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
| Source format | file-kind sniffing picks an adapter (`pipeline.md`) | `input(adapter=<name or InputAdapter>, sources=[...])` — "escape hatch — no inference" (`pipeline.md`) | the fluent layer only builds shipped adapters; a bespoke raw format needs a real `InputAdapter` |
| Per-segment metric | `.measure(name="...")` against the closed DSL grammar | `@register_metric` — "the escape hatch in `registry.py`" (`dsl/grammar.py` docstring) | the grammar is closed and enumerated *on purpose* (loud validation); an escape hatch is one plain function over a `WindowBuffer`, sharing the same signature so the registry treats both interchangeably (`dsl/registry.py`) |
| Whole-pool / learned scoring | `.score(name="...")` | `SegmentScorer` + `register_scorer` | this shape is deliberately *not* bent through `@register_metric` — whole-pool/learned/action-aware estimators are "the wrong shape" for a per-segment, single-pass, stateless function (`scorers.py` docstring) |
| Bounded model/device compute | `.transform(name="...")` | `BatchTransform` + `register_batch_transform` | needs custom NumPy/model code over declared `field:`/`feature:`/`media:` roles — declarative strategy can't express arbitrary compute |
| Selection | `.bin()/.filter()/.sample(n=…/hours=…)` | raw `Strategy` dict / `load_strategy()` | strategy stays pure data so it round-trips through `config.json` and reproduces from the artifact alone (`strategies.md`) |
| Reusable curation decisions | generated fluent method (`pipeline.pixel_shift(...)`; `.apply(name, **params)` is the compatibility spelling for serialized integrations) | `register_operator` | an operator is "a composition seam, not a native execution-stage seam" (`pipeline.md`) — see Limits below |
| Stateful / cross-segment checks | `.analyze(name, parameters=...)` | `get_analyzer()` + drive `add_batch`/`check_dataset` by hand | the SDK never implicitly schedules an analyzer as a pipeline stage — it's user-owned state, not a stateless transform (`analyzers.py` docstring) |
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
   stats enricher.
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
seams hands a beginner a plain fluent method, with zero new concepts to learn and zero risk of
silently shadowing something that already existed.

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

It's tempting to flatten "fluent vs. classic" and "CLI/job vs. Python" into a single
beginner/expert line. They're actually orthogonal:

|  | **declarative (JSON job / preset)** | **imperative (Python)** |
|---|---|---|
| **fluent ergonomics** | `kur-init` scaffold → `kur-curate job.json` | `input(...).window(...).measure(...).sample(...)` |
| **classic/explicit control** | hand-authored job JSON referencing named custom components | `SchemaSpec`, `load_strategy`, `curate()`/`write_dataset()`, arbitrary `group_of`/`tags_of` callables |

A JSON job can already reference a custom, expert-registered adapter or scorer *by name* — the
real boundary `docs/jobs.md` draws is data/names (jobs) vs. arbitrary code
(`write_dataset(...)` directly): "If you need arbitrary code, call `write_dataset(...)`
directly... graduating from a job file to Python is mechanical" (`jobs.md`). Don't
assume "the CLI is beginner, Python is expert" — plenty of expert work (registering a new
operator, writing a `BatchTransform`) happens once, in Python, and is then consumed by *both*
a fluent Python chain and a JSON job, at the same rung.

## Deliberate limits — where disclosure stops on purpose

Progressive disclosure is bounded, and the bounds are load-bearing, not accidental gaps:

- **The core imports no domain code, ever** (`architecture.md` "Invariants", `PLAN.md`
  design principle 7). There is no "advanced mode" that lets domain logic into
  `src/kuration/core/`; the seam table is exhaustive, not a suggestion.
- **The closed grammar has no runtime toggle to loosen it.** Unknown DSL/strategy tokens raise
  at load, always — "no silent fallbacks" (`architecture.md` "Invariants"). The escape hatch is a new
  named metric, never a bypass flag.
- **Operators are a shallow seam, easy to mistake for the top of the ladder.** They read as the
  most "fluent-feeling" extension mechanism, but `pipeline.md` and
  `customization.md` both stress a handler "cannot introduce a new terminal or bypass
  the validation and execution contracts of existing operators" — it can only *compose* already
  validated operations. The deeper seams (`BatchTransform`, `SegmentScorer`, a custom
  `InputAdapter`) are where genuinely new execution behavior belongs; `register_operator` is
  packaging, not new capability.
- **Analyzers are never implicit.** Unlike every other registered kind, an analyzer is not
  scheduled automatically by any pipeline stage — "the caller owns the instance and its update
  lifecycle" (`analyzers.py` docstring). This is intentional: cross-segment, stateful checks are
  not safely composable the way stateless per-segment/whole-pool scoring is.
- **`kur-init`'s scaffold is a starting point, explicitly not a recommendation** — "it picks
  the first scalar field as a binning axis because it cannot know which metric matters. Edit
  `strategy` before trusting a run" (`jobs.md`). The beginner path is honest about being
  a beginner path.

## Known tensions (surfaced, not smoothed over)

- **`PLAN.md` predates the fluent pipeline.** Its "non-negotiable" design principles
  and phase log never mention `input()` or fluent operators at all — they
  were written for the classic-API-only era. If you're looking for *why* the fluent layer was
  added on top, that rationale lives in `docs/pipeline.md` and
  `docs/documentation-contract.md`, not in the historical build log. Treat `PLAN.md` as
  authoritative for the classic engine's invariants, and this document plus `pipeline.md` as
  authoritative for the two-tier API story layered on top later.
- **Job-spec vs. Python is not a strict subset of fluent vs. classic** — see the two-axes table
  above. Don't reason "CLI = simple, Python = powerful"; reason "declarative-by-name" vs.
  "arbitrary code," independently of fluent vs. classic.
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
   fluent method for this be *generated* from a registry entry, the way operators are, rather
   than hand-written?
3. Does it preserve "one `Pipeline` type" and the collision guard, so an expert's registration
   can't silently break a beginner's existing call?
4. Does it stay import-lazy and loud-on-missing for any optional dependency it needs, so a
   beginner who never touches it never pays for it?
5. Does it fail loudly at load/validation time rather than silently, consistent with the closed
   grammar's "no silent fallbacks" invariant?

If the answer to all five is yes, it belongs in the SDK exactly the way every existing seam
does. Update the canonical home for that layer (`docs/customization.md`, `docs/operators.md`,
etc. — see `docs/documentation-contract.md`'s ownership table) rather than duplicating an
explanation here; this document should stay the short, stable statement of the thesis and its
ladder, not grow into a second architecture reference.
