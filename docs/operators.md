# Building operators

An operator is a named, validated recipe that expands into ordinary fluent pipeline operations.
It is the right abstraction for a curation decision your team wants to review once and reuse
across datasets or projects.

<p align="center">
  <img src="assets/operator-composition.svg" alt="An operator combines a validated parameter contract with ordinary pipeline operations and optional bounded transforms" width="96%">
</p>

Operators do not introduce a hidden execution system. Their expansion is validated using the
same schema and strategy rules as handwritten pipeline code, and both the named call and expanded
operations are recorded in artifact provenance.

## Programming model

There is one `Pipeline` type. An operator registration adds a fluent method to that type; it does
not create a domain-specific pipeline or require string dispatch:

```python
from kuration import Pipeline, input
import acme_kuration  # imports a module that calls register_operator(...)

base = input("sessions/*.parquet", time="t").window(64, stride=32)
pipeline = base.signal_diversity(field="motor_current", bins=6)

assert type(base) is Pipeline
assert type(pipeline) is Pipeline
```

The method is resolved from the registered component name:

| registered name | normal Python spelling | always-available qualified spelling |
|---|---|---|
| `acme/signal-diversity` | `.signal_diversity(...)` | `.acme__signal_diversity(...)` |
| `aintropy/pixel-shift` | `.pixel_shift(...)` | `.aintropy__pixel_shift(...)` |

Hyphens and dots become underscores in the readable spelling. If two namespaces expose the same
local name, the short spelling fails with the qualified alternatives instead of choosing one (an
un-namespaced operator whose name *is* the short spelling keeps it, and the namespaced ones stay
callable under their qualified aliases). The qualified form is also useful in shared libraries
that want to make component ownership explicit.

A local name that is already a `Pipeline` attribute — `report`, `window`, `filter`, `output`,
`schema`, … — is **rejected at registration** (and by `kur-plugin-check`): Python resolves the
real attribute before the generated method, so such an operator would be silently shadowed.
Rename it (`acme/report-balance`).

Import order is flexible for the process-wide registry: importing a registration module makes
its methods available even on an existing default-registry pipeline. A pipeline created with an
explicit registry remains isolated and resolves operators only through that registry snapshot.
`Pipeline.apply(name, **params)` remains a compatibility path for integrations that receive a
component name as data; ordinary Python code should use the generated method.

## 1. Package an ordinary pipeline recipe

This operator measures variability and adds it as a diversity axis:

```python
from kuration import register_operator

SIGNAL_PARAMETERS = {
    "type": "object",
    "properties": {
        "field": {"type": "string"},
        "bins": {"type": "integer", "minimum": 2},
    },
    "required": ["field"],
    "additionalProperties": False,
}

@register_operator(
    "acme/signal-diversity",
    description="Balance selected windows by signal variability",
    parameters=SIGNAL_PARAMETERS,
    version="1",
)
def signal_diversity(pipeline, *, field, bins=8):
    return (pipeline
            .measure("signal_variability", "std", of=field)
            .bin("signal_variability", bins=bins))
```

Use it like a built-in operation:

```python
pipeline = (
    input("sessions/*.parquet", time="t")
    .window(64, stride=32)
    .signal_diversity(field="motor_current", bins=6)
    .sample("equal", n=10_000)
)
```

Unknown parameters, missing `field`, invalid bin counts, unknown fields, and feature-name
collisions all fail when the pipeline is built—not halfway through a long run.

## 2. Wrap bounded window computation

The closed metric DSL should remain the first choice for ordinary reductions. Use a batch
transform when computation needs custom NumPy/model code, multiple fields, or decoded media.

```python
import numpy as np

from kuration import register_batch_transform, register_operator

@register_batch_transform(
    "acme/window-range",
    outputs=("window_range",),
    inputs=("field:signal",),
    resources={"preferred_batch_size": 64},
)
class WindowRange:
    def transform_batch(self, batch, ctx):
        values = np.asarray(batch["field:signal"], dtype=np.float64)
        axes = tuple(range(1, values.ndim))
        return {"window_range": np.ptp(values, axis=axes)}

@register_operator(
    "acme/range-diversity",
    description="Compute a custom window range and balance by it",
    parameters=SIGNAL_PARAMETERS,
    version="1",
)
def range_diversity(pipeline, *, field, bins=8):
    return (pipeline
            .transform("acme/window-range", inputs=[f"field:{field}"])
            .bin("window_range", bins=bins))
```

The transform receives bounded batches. It must return one finite scalar per segment for every
declared output. Imports for heavyweight frameworks belong inside the implementation path, so
component discovery and ordinary core imports remain dependency-light.

For edge-sensitive events, registration accepts `lookbehind=<ticks>`, `lookahead=<ticks>`, and
`ownership="rightmost-start"`. Read the typed context with `ctx.before("field:signal")` and
`ctx.after("field:signal")`; source-edge entries are `None`. Immutable `ctx.owned_lengths` gives
the prefix of each logical window whose event starts belong to that segment. The corresponding
rc2 batch keys remain available during the release-candidate compatibility window. Either count
may instead be a `{declared input token: count}` mapping (`lookbehind={"field:signal": 1}`) when
only one input needs boundary context — the runner then extracts, and for media decodes, nothing
for the others.

A caller may remap the declared inputs per use (`.transform(name, inputs=["media:shot",
"field:signal"])`), supplying one same-kind token per declared role. Address the batch through
`ctx.input_token("media:frame")` (or the `ctx.inputs` role→token mapping) so a remapped use works;
a hardcoded registration token would pass construction and fail mid-run. `field:` batches are
`(B,T,E,D)` and NaN-padded to the batch's widest extent, so use `ctx.valid_steps` (also
`batch["valid_steps"]`) to bound each row to its real timesteps.

For video, declare `inputs=("media:frame",)` and a semantic requirement such as
`media_requirements={"kind": "video-frame", "layout": "HWC", "color_space": "RGB"}`.
Kuration validates the media contract before decoding frames.

## 3. Test the contract

Test the operator at three levels:

```python
def test_signal_operator_expands_and_selects():
    base = input([{"signal": float(i)} for i in range(16)]).window(4)
    pipeline = base.signal_diversity(field="signal", bins=3)

    description = pipeline.describe()
    assert "signal_variability" in description["strategy"]["define"]
    assert description["strategy"]["binning"]["axes"] == [
        {"metric": "signal_variability", "bins": 3}
    ]

    result = pipeline.sample("equal", n=2).collect()
    assert len(result.selected) == 2
```

Also cover:

- Parameter-schema failures and unknown fields.
- Exact output names, shape, dtype, and non-finite rejection for transforms.
- Discovery metadata through `kur-list` and `kur-describe`.
- Clean installed-package discovery—not only imports from a checkout.
- Provenance in a written `config.json`.
- Optional dependency absence with an actionable installation message.

## 4. Discover and inspect

```bash
kur-list operators
kur-describe operator acme/signal-diversity --json
kur-describe transform acme/window-range --json
```

Python callers use `list_components("operator")` and
`describe_component("operator", "acme/signal-diversity")`. Descriptors include parameter
schemas, dependency hints, capabilities, package and module identity, stability metadata, and a
code fingerprint.

Registration also creates the fluent import flow. Import a custom registration module, or let
Kuration discover an installed `kuration.plugins` entry point, then call the operator by its
normalized local name:

```python
import acme_kuration

pipeline = pipeline.signal_diversity(field="signal")
```

`acme/signal-diversity` maps to `signal_diversity`; the always-available qualified spelling is
`acme__signal_diversity`. Ambiguous short names fail with the qualified alternatives instead of
silently choosing one.

## 5. Built-in operator catalog

Importing `kuration.gameplay` registers every maintained gameplay operator on the ordinary core
pipeline. Importing the domain remains dependency-light; optional frameworks load only when an
operator's execution path needs them.

| registered operator | fluent method | expands to | execution dependency |
|---|---|---|---|
| `aintropy/pixel-shift` | `.pixel_shift(controls=None, boundary="exclude")` | the bounded transform of the same name | OpenCV via `[pixel-shift]` |
| `aintropy/slam-motion-diversity` | `.slam_motion_diversity(axis="slam_path_length", bins=8)` | `aintropy/slam-motion` plus a bin axis | NumPy for a precomputed trajectory |
| `aintropy/slam-quality` | `.slam_quality(min_score=None, **quality_options)` | `aintropy/slam-quality` plus an optional predicate | NumPy for a precomputed trajectory |

```python
import kuration.gameplay  # registers the methods
from kuration import input

pipeline = (
    input("telemetry/*.parquet", time="timestamp")
    .join_video(lambda source: source.replace(".parquet", ".mp4"), tolerance=0.020)
    .window(32, stride=16)
    .pixel_shift(controls=["W", "S", "A", "D"])
    .bin("w_shift_mean", bins=8)
    .sample("equal", n=2_000)
)
```

Pixel Shift additionally needs DIAMOND-compatible action rows and aligned video frames. The SLAM
operators expect the trajectory schema produced by `gameplay.slam_pipeline(...)` or
`gameplay.input_slam(...)`. See [Pixel Shift](pixel-shift.md) and [SLAM](slam.md) for those input
contracts.

Action-state consistency is intentionally an `analyzer`, not a pipeline operator: it accumulates
evidence across multiple segments in a user-owned object. Construct it with
`gameplay.action_state_checker(...)`; see
[Action-state consistency](action-state-consistency.md).

The ordinary core operations—`.window()`, `.measure()`, `.transform()`, `.filter()`, `.bin()`,
`.score()`, `.tag()`, `.sample()`, and the rest—are listed with exact signatures in the
[Pipeline operator reference](pipeline.md#operator-reference). At runtime,
`kur-list operators` and `list_components("operator")` are the authoritative catalogs for the
currently installed and imported environment.

## 6. Choose built-in or external packaging

First-party gameplay operators accepted into the SDK live under `src/kuration/gameplay/`, register
when that domain is imported or discovered, and declare optional framework requirements through
SDK extras. For an independently versioned or external plugin, copy
[`plugins/_template`](https://github.com/aintropy-ai/kuration-sdk/tree/main/plugins/_template)
to `plugins/<plugin-name>`. The external package uses this layout:

```text
acme-kuration/
├── pyproject.toml
├── src/acme_kuration/__init__.py
├── src/acme_kuration/operators.py
└── tests/test_operators.py
```

Expose a registration module through the `kuration.plugins` entry-point group:

```toml
[project.entry-points."kuration.plugins"]
acme = "acme_kuration.operators"
```

Registration happens when kuration loads installed plugins. Use namespaced names (`acme/...`) to
avoid collisions. Keep dataset-specific assumptions visible in the descriptor, README, parameter
schema, and tests. The [customization guide](customization.md) covers registry overlays and the
other extension types.

Before opening a contribution, reproduce the isolated CI contract from the repository root:

```bash
python scripts/check_plugin.py plugins/<plugin-name>
```

For installed packages, `kur-plugin-check acme --json` performs the descriptor and discovery
portion transactionally.

## Complete runnable examples

[`examples/05_custom_operators.py`](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/05_custom_operators.py) registers both styles,
discovers their descriptors, runs them over small in-memory sessions, and verifies their selected
features. It requires only the NumPy core:

```bash
python examples/05_custom_operators.py
```

[`examples/06_gameplay_operator_methods.py`](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/06_gameplay_operator_methods.py)
uses a synthetic precomputed camera trajectory to demonstrate the maintained gameplay methods,
the namespace-qualified spelling, and the fact that every call returns the same core `Pipeline`:

```bash
python examples/06_gameplay_operator_methods.py
```
