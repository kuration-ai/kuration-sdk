# Customization and discovery

Kuration resolves adapters, metrics, scorers, encoders, selection methods, pre-pass filters,
containers, output presets, and compositional operators through the same descriptor model.
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

Discover components without materializing a dataset:

```bash
kur-list methods --json
kur-describe method acme/quality-v2 --json
```

Python equivalents are `list_components()` and `describe_component()`. Descriptors include a
parameter JSON Schema, capability flags, dependency/install hints, package/version metadata,
documentation/stability fields, module identity, and a code fingerprint. Resolved extension
descriptors and parameters are included in run provenance.

## Compositional operators

Use an operator when you want to name and reuse a sequence of ordinary fluent operations:

```python
from kuration import register_operator

@register_operator(
    "acme/balance-signal",
    description="Measure and balance signal intensity",
    parameters={
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "bins": {"type": "integer", "minimum": 1},
        },
        "required": ["field"],
        "additionalProperties": False,
    },
    version="1",
)
def balance_signal(pipeline, *, field, bins=8):
    return (pipeline
            .measure("signal_level", "mean", of=field)
            .bin("signal_level", bins=bins))

pipeline = pipeline.apply("acme/balance-signal", field="signal", bins=6)
```

The handler must return a pipeline derived from the one it receives and must add at least one
ordinary operator. It cannot add a new execution stage. The named call and its expansion are
both retained in pipeline provenance. Discover registrations with `kur-list operators` and
inspect their parameter contract with `kur-describe operator acme/balance-signal`.

Artifacts and manifests carry `artifact_format_version`; jobs carry `job_schema_version`.
Unversioned documents are legacy `0.0`, current documents are `1.0`, and unsupported future
major versions are rejected. Use `migrate()` or `kur-migrate` for deterministic upgrades.
