# Customization and discovery

Kuration resolves adapters, metrics, scorers, batch transforms, encoders, selection methods,
pre-pass filters, containers, output presets, compositional operators, and stateful analyzers
through the same descriptor model.
Names may be simple built-in names or namespaced plugin names such as `acme/quality-v2`.
Duplicate registration is loud.

```python
from kuration import Registry, default_registry, register_method

register_method("acme/quality-v2", priority_fn,
                description="Rank by our quality model",
                parameters={"type": "object", "properties": {"temperature": {"type": "number"}},
                            "additionalProperties": False},
                capabilities={"whole_pool": True})

local = Registry(parent=default_registry)
local.register("method", "lab/experiment", another_priority_fn)
pipeline = pipeline.using(local)
```

A method returns one finite priority per segment. A filter accepts
`(segments, feature_names, seed=, workers=, **parameters)` and returns retained segments. A
container supplies an extension plus writer and reader callables through `register_container`.
Plugins loaded by a JSON job register the same names before preflight, so unknown names and bad
parameters fail before source rows are read.

A transform factory is registered with `register_batch_transform()` (or `register_transform`),
declaring scalar outputs, `field:`/`feature:`/`media:` inputs, semantic media requirements, and
resource hints. Boundary-sensitive transforms may additionally declare `lookbehind` and
`lookahead` tick counts, either as one integer shared by every input or as a
`{declared input token: count}` mapping when only some inputs need context. The typed
`TransformContext` exposes per-row arrays through
`ctx.before("field:signal")` and `ctx.after("field:signal")`;
`ownership="rightmost-start"` also supplies immutable `ctx.owned_lengths`, preventing overlapping
windows from claiming the same event start. The rc2 batch keys remain available during the
release-candidate compatibility window.

Declared input tokens are *roles*. A per-use `inputs=` remapping supplies one same-kind token per
role (arity and kind are validated when the pipeline is built), so read every input through
`ctx.input_token("media:frame")` — or `ctx.inputs`, the role→token mapping in declaration order —
rather than hardcoding the registration token, which a remapped use would not have. `field:`
inputs arrive as `(B,T,E,D)` arrays NaN-padded to the widest extent in the batch; `ctx.valid_steps`
(also `batch["valid_steps"]`) gives each row's real timestep count, so a ragged batch's padding is
never averaged into a measurement by accident. A pipeline invokes the transform with
`.transform(...)`; outputs behave like any other scalar selection feature. Use `get_transform()`
for direct lookup. Accelerator libraries must be imported inside the factory or execution method
so discovery and `import kuration` remain dependency-light.

Discover components without materializing a dataset using `list_components()` and
`describe_component()` (see [Command-line interface](#command-line-interface) for the CLI
equivalents). Descriptors include a parameter JSON Schema, capability flags, dependency/install
hints, package/version metadata, documentation/stability fields, module identity, and a code
fingerprint. Resolved extension descriptors and parameters are included in run provenance.

## Stateful analyzers

Use `register_analyzer()` when a capability must accumulate evidence across multiple segments
before it can return a result and therefore cannot honestly run as an independent batch
transform. The registered callable is a dependency-light factory; `get_analyzer()` validates
its keyword parameters, constructs a user-owned instance, and requires that instance to expose
`summary()`. Its other update methods are capability-specific and documented by the plugin.

```python
from kuration import get_analyzer, register_analyzer

register_analyzer("acme/clip-check", build_clip_checker,
                  description="Accumulate evidence across every segment in a clip",
                  parameters=PARAMETERS)
checker = get_analyzer("acme/clip-check", **config)
for segment in segments:
    checker.add_segment(segment)
result = checker.summary()
```

Analyzers are discoverable like any other component (see
[Command-line interface](#command-line-interface)), but are not implicit selection stages. Keeping lifecycle ownership explicit prevents worker-local state
from being mistaken for a dataset-wide result. Two optional hooks make an analyzer drivable, and
declaring one is what `Pipeline.enrich()` and `Pipeline.analyze()` dispatch on:

| hook | driven by | evidence it sees |
|---|---|---|
| `check_dataset(dataset, **options)` → mapping with a boolean `passed` | `run_dataset_analyzer()`, `Pipeline.enrich()` / `Pipeline.analyze()` on an artifact-backed pipeline, `kur-check DATASET --analyzer NAME [--analyzer-config PATH] [--analyzer-options PATH]` | written shards and their manifest |
| `add_batch(batch, ctx)` → `None` | `run_stream_analyzer()`, `Pipeline.enrich()` / `Pipeline.analyze()` on any `input(...)` pipeline | segmented raw ticks with lazy media, in one bounded pass |

`add_batch` receives exactly what a batch transform's `transform_batch` receives, because it is
fed by the same bounded machinery: a batch keyed by the analyzer's declared input tokens plus
`segment_ids` and `valid_steps`, and a `TransformContext` whose `input_token(role)`,
`before(token)`, and `after(token)` resolve declared roles and boundary context. It accumulates
instead of returning scalars, so it must return `None`. Declare the inputs at registration —
`field:`/`media:` tokens only, since a single raw pass has no feature stage to produce a
`feature:` input:

```python
register_analyzer("acme/clip-check", build_clip_checker,
                  description="Accumulate evidence across every segment in a clip",
                  parameters=PARAMETERS,
                  inputs=("media:frame", "field:action"),
                  media_requirements={"kind": "video-frame", "layout": "HWC"},
                  lookbehind={"field:action": 1}, lookahead=3)

pipeline = pipeline.enrich(
    "acme/clip-check",
    config,
    namespace="acme",
    key="clip_check",
    controls=["W"],
)
result = pipeline.metadata["acme"]["clip_check"]["result"]
```

Declarations are validated at registration (and by `kur-plugin-check`), so a malformed one fails
at import rather than inside a long run. `**options` passed to `enrich()` or `analyze()` reach
the hook being driven; `inputs=`/`batch_size=` remap and size the stream path. `enrich()` returns
a new pipeline carrying the result, while `analyze()` is the explicit detached-result form.
A `stream_result()` method may return a mapping with a boolean `passed` (mirroring
`check_dataset`) that is merged into the driver's result — the place to state a verdict and, if
the analyzer records positions, which coordinate space they live in.

Segment ids on the stream path are the real `(source_id, start_idx, end_idx)` of the source
window, so an analyzer's own cross-window de-duplication works in true source coordinates. The
artifact path cannot offer that (a manifest records no source start offset); an analyzer that
measures positions should say so rather than imply source frames.

The SDK includes `kuration/split-leakage` for auditing existing or imported split assignments.
It accepts repeated observations (for example, many segments from one clip), detects a source
assigned to multiple splits, and optionally detects groups spanning split boundaries:

```python
from kuration import get_analyzer

audit = get_analyzer("kuration/split-leakage")
audit.add_assignments(
    {"clip-1": "train", "clip-2": "train", "clip-3": "test"},
    source_groups={"clip-1": "session-a", "clip-2": "session-a", "clip-3": "session-b"},
)
report = audit.summary()
assert report["passed"]
```

For row- or segment-oriented inputs, call
`audit.add_assignment(source_id, split, group=group)` once per observation. Repeating the same
assignment is not leakage. Set `check_groups=False` only when the dataset has no meaningful
group boundary; source-level leakage is always checked. The report is deterministic and
JSON-safe, with `source_leaks`, `group_leaks`, and `source_group_conflicts` details. Source,
split, and supplied group identifiers must be non-empty strings; convert an application's
numeric or compound group keys to stable strings before adding them. See the
[API reference](api.md#component-registries-and-discovery) for the complete report schema and
validation behavior.

The dependency-light validator supports the descriptor subset used by SDK contracts:
`type`, `enum`, `const`, `allOf`, `anyOf`, `oneOf`, numeric bounds, string length and `pattern`,
array length/uniqueness/`items`, and nested object `properties`, `required`, and
`additionalProperties`. Validation is recursive and reports the failing parameter path.

## Built-ins and external plugins

First-party standard capabilities such as pixel shift, action-state consistency, and SLAM ship in
the SDK wheel under `kuration.gameplay`, and general-purpose curation recipes such as
`BalancedSelector`, `RareEventCoverage`, and `DiverseSubset` ship under `kuration.selection`
(see [Selection recipes](selection.md)); both domains register when imported or discovered and
keep heavy frameworks in optional extras.
For independently versioned or externally distributed behavior, publish your own package that owns
its dependencies, tests, documentation, and `kuration.plugins` entry point (see the
[operator guide](operators.md#6-choose-built-in-or-external-packaging) for the expected layout).

Validate an already-installed distribution's loading, descriptor package/version/install
metadata, closed parameter schemas, namespacing, and transform declarations with
`check_plugins()` (see [Command-line interface](#command-line-interface) for the CLI form),
without retaining its registry mutations.

## Compositional operators

Use an operator when you want to name and reuse a sequence of ordinary operations. The
handler must return a pipeline derived from the one it receives and add at least one ordinary
operator. It may compose `.join_video()` and `.transform()`, but cannot invent an unvalidated
stage or terminal. The named call and its expansion are both retained in provenance.

The complete [operator guide](operators.md) owns registration, transform-backed computation,
contract testing, discovery, and plugin-packaging examples, including the full source of the
runnable companion, `examples/05_custom_operators.py`.

Artifacts and manifests carry `artifact_format_version`; jobs carry `job_schema_version`.
Unversioned documents are legacy `0.0`. The current artifact format is `1.2` and the current job
schema is `1.0`; documents newer than the corresponding supported version are rejected, including
future minor versions. Use `migrate()` or `kur-migrate` for deterministic upgrades where a
migration is available.

## Command-line interface

Discover components without materializing a dataset:

```bash
kur-list methods --json
kur-describe method acme/quality-v2 --json
```

For an already installed distribution, `kur-plugin-check <entry-point-name>` validates loading,
descriptor package/version/install metadata, closed parameter schemas, namespacing, and transform
declarations without retaining its registry mutations.
