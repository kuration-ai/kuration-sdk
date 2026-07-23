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
rc2 batch keys remain available during the release-candidate compatibility window.

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

## 5. Choose built-in or external packaging

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

## Complete runnable example

[`examples/05_custom_operators.py`](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/05_custom_operators.py) registers both styles,
discovers their descriptors, runs them over small in-memory sessions, and verifies their selected
features. It requires only the NumPy core:

```bash
python examples/05_custom_operators.py
```
