# Customization and discovery

Kuration resolves adapters, metrics, scorers, batch transforms, encoders, selection methods,
pre-pass filters, containers, output presets, and compositional operators through the same
descriptor model.
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
resource hints. Boundary-sensitive transforms may additionally declare integer `lookbehind` and
`lookahead` tick counts. Their batches receive `lookbehind:<input>`/`lookahead:<input>` lists;
`ownership="rightmost-start"` also supplies `owned_lengths`, preventing overlapping windows from
claiming the same event start. A pipeline invokes the transform with `.transform(...)`; outputs behave like any
other scalar selection feature. Use `get_transform()` for direct lookup. Accelerator libraries
must be imported inside the factory or execution method so discovery and `import kuration` remain
dependency-light.

Discover components without materializing a dataset:

```bash
kur-list methods --json
kur-describe method acme/quality-v2 --json
```

Python equivalents are `list_components()` and `describe_component()`. Descriptors include a
parameter JSON Schema, capability flags, dependency/install hints, package/version metadata,
documentation/stability fields, module identity, and a code fingerprint. Resolved extension
descriptors and parameters are included in run provenance.

The dependency-light validator supports the descriptor subset used by SDK contracts:
`type`, `enum`, `const`, `allOf`, `anyOf`, `oneOf`, numeric bounds, string length and `pattern`,
array length/uniqueness/`items`, and nested object `properties`, `required`, and
`additionalProperties`. Validation is recursive and reports the failing parameter path.

## Repository plugins

New maintained extensions belong under the repository's
[`plugins/` workspace](https://github.com/aintropy-ai/kuration-sdk/tree/main/plugins). Each child is an
independent Python distribution with its own dependencies, tests, documentation, and
`kuration.plugins` entry point; plugin code is intentionally excluded from the core SDK wheel.
Copy `plugins/_template`, use a namespaced component name, and let the plugin CI job exercise the
same installation and wheel checks used for every sibling.

## Compositional operators

Use an operator when you want to name and reuse a sequence of ordinary fluent operations. The
handler must return a pipeline derived from the one it receives and add at least one ordinary
operator. It may compose `.join_video()` and `.transform()`, but cannot invent an unvalidated
stage or terminal. The named call and its expansion are both retained in provenance.

The complete [operator guide](operators.md) owns registration, transform-backed computation,
contract testing, discovery, and plugin-packaging examples. The runnable companion is
[`examples/05_custom_operators.py`](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/05_custom_operators.py).

Artifacts and manifests carry `artifact_format_version`; jobs carry `job_schema_version`.
Unversioned documents are legacy `0.0`, current documents are `1.0`, and unsupported future
major versions are rejected. Use `migrate()` or `kur-migrate` for deterministic upgrades.
