# Contributing to kuration-sdk

Thanks for helping improve kuration-sdk. Contributions may include bug fixes, adapters,
selection methods, documentation, tests, and runnable examples.

For user and integrator questions, see [SUPPORT.md](https://aintropy-solutions.github.io/kuration-sdk/support/). Project ownership, decisions,
and release responsibilities are described in [GOVERNANCE.md](https://aintropy-solutions.github.io/kuration-sdk/governance/).

## Development setup

kuration-sdk requires Python 3.11 or newer. The supported-version CI matrix explicitly covers
Python 3.11 through 3.14. From a repository checkout, create and activate a virtual environment,
then install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install every optional runtime integration when working across adapters, containers, learned
methods, or framework support:

```bash
python -m pip install -e ".[all,dev]"
```

Optional dependencies must remain optional. The base install depends only on NumPy; imports of
heavy integrations should be lazy and should raise a clear installation hint when unavailable.

## Design rules

- Keep the execution framework domain-neutral. First-party domain behavior may ship as a
  namespaced built-in adapter, analyzer, operator, or transform when it has an owner, a closed
  contract, focused tests, documentation, and lazy optional dependencies. External or
  independently versioned behavior belongs behind the same extension seams as a plugin.
- Validate configuration early and loudly. Unknown fields, operators, methods, and parameters
  must not silently fall back.
- Preserve deterministic results across worker counts for a fixed seed.
- Keep written datasets reproducible: changes affecting execution must be represented in
  `config.json` and, when applicable, versioned or migrated.
- Preserve pipeline immutability: every operator returns a new `Pipeline`.
- Prefer `.bin()` and `.sample()` in new pipeline code. `.organize()` and `.curate()` are
  compatibility aliases and must not be removed without a documented migration.

See [docs/architecture.md](https://aintropy-solutions.github.io/kuration-sdk/architecture/) for the data contracts and
[docs/customization.md](https://aintropy-solutions.github.io/kuration-sdk/customization/) for supported extension points.

## Making a change

Add focused tests with every behavior change. A regression test should fail before the fix and
exercise the public interface where practical. Tests that should produce data must use loud
error handling and assert that their candidate and output pools are non-empty.

Keep public interfaces synchronized across:

- generic implementation and exports in `src/kuration/core/`;
- maintained domain implementation in `src/kuration/gameplay/`;
- exact signatures in [docs/api.md](https://aintropy-solutions.github.io/kuration-sdk/api/);
- pipeline behavior in [docs/pipeline.md](https://aintropy-solutions.github.io/kuration-sdk/pipeline/);
- runnable examples when the recommended workflow changes;
- artifact or job schema versions when serialized contracts change.

New formats and optional integrations should include tests that skip cleanly when their extra is
not installed. A missing optional dependency at runtime should identify the extra needed to use
the feature.

## Choosing an extension seam

Use the narrowest supported seam for the behavior:

| need | extension |
|---|---|
| Reuse a named sequence of pipeline calls | compositional operator |
| Read a new source format | input adapter |
| Compute bounded model/device features over batches | batch transform |
| Compute one scalar from one segment window | registered metric |
| Score segments using a whole pool, model, or reference set | segment scorer |
| Accumulate state across a stream or audit a written artifact | analyzer |
| Derive an output tensor or encoded value | output encoder |
| Change how segments receive selection priority | selection method |
| Remove segments before selection | pre-pass filter |
| Store and read shards in another format | container |
| Package a reusable `OutputSpec` policy | output preset |

Do not add dataset-specific parsing, model imports, or storage branches to the generic pipeline
when one of these seams fits. See [docs/customization.md](https://aintropy-solutions.github.io/kuration-sdk/customization/) for descriptors,
isolated registries, discovery, and versioning.

### Compositional operators

An operator names a validated sequence of existing pipeline calls. It does not create a new
execution stage:

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
    return (
        pipeline
        .measure("signal_level", "mean", of=field)
        .bin("signal_level", bins=bins)
    )


pipeline = pipeline.balance_signal(field="signal", bins=6)
```

The handler must return a pipeline derived from its argument and add at least one ordinary
operator. Add tests for parameter validation, immutable expansion, resolved strategy, and the
operator/expansion provenance stored in `config.json`.

### Input adapters

An adapter converts one source into ordered canonical `Tick` values:

```python
from kuration import Tick, register_adapter


class AcmeLogAdapter:
    supports_streaming = True

    def __init__(self, schema, *, reader):
        self.schema = schema
        self.reader = reader

    def check_source(self, source_uri):
        self.reader.probe(source_uri)  # metadata-only validation when possible

    def iter_ticks(self, source_uri):
        for row in self.reader.rows(source_uri):
            yield Tick(
                time=row.timestamp,
                entities={row.agent_id: {"signal": float(row.signal)}},
            )

    def ticks(self, source_uri):
        return list(self.iter_ticks(source_uri))


register_adapter(
    "acme/log",
    AcmeLogAdapter,
    description="Read ACME telemetry logs",
    capabilities={"streaming": True},
    version="1",
)
```

Validate mapped fields, lengths, shapes, ordering, and source metadata loudly. Keep imports for
optional parsers inside the adapter and provide an `extra` plus `install_hint` descriptor. Test
both `ticks` and `iter_ticks` parity when streaming is supported, metadata-only `check_source`,
multiple entities, malformed input, and `workers>1` pickling/plugin reconstruction.

### Metrics and scorers

Use a metric for stateless, per-window scalar computation:

```python
import numpy as np
from kuration import register_metric


@register_metric("acme/peak-signal", description="Maximum signal in a segment", version="1")
def peak_signal(window):
    return float(np.nanmax(window.numeric_field("signal")))
```

Use a scorer when computation needs the full segment pool, model state, or raw window buffers:

```python
import numpy as np
from kuration import register_scorer


class QualityScorer:
    needs_buffers = True

    def __init__(self, *, scale=1.0):
        self.scale = float(scale)

    def score(self, segments, ctx):
        values = [np.nanmean(buf.numeric_field("signal")) for buf in ctx["buffers"]]
        return self.scale * np.asarray(values, dtype=np.float64)


register_scorer("acme/quality", QualityScorer, version="1")
```

Metrics and scorers must return finite scalars. Registered metrics used with multiprocessing
must live in an importable module, not only in `__main__`. A scorer returns exactly one value per
segment; declare `needs_buffers = False` when it uses only existing segment features so the SDK
does not reread source windows.

### Batch transforms and media

Use a registered batch transform for bounded model/device work over ordinary fields, existing
features, or lazy media. Declare scalar outputs and semantic media requirements at registration:

```python
from kuration import register_batch_transform


@register_batch_transform(
    "acme/frame-quality",
    outputs=("frame_quality",),
    inputs=("media:frame",),
    media_requirements={"kind": "video-frame", "layout": "HWC", "color_space": "RGB"},
    resources={"num_gpus": 1, "preferred_batch_size": 16},
)
class FrameQuality:
    def __init__(self, *, device="cpu"):
        self.model = load_model(device)  # optional framework import/load belongs here

    def transform_batch(self, batch, ctx):
        return {"frame_quality": self.model(batch["media:frame"])}
```

Outputs must be one finite scalar per input segment. Test bounded batch sizes, deterministic
segment alignment, media semantic mismatches, missing optional dependencies, and CPU parity when
the implementation supports CPU. Never silently change model, device, dtype, or precision.

### Output encoders

An encoder derives declared `encoder:` modalities from one output record, or implements
`encode_batch(records, ctx)` for bounded batched encoding:

```python
import numpy as np
from kuration import register_encoder


class ActionEncoder:
    def timestep(self, record, ctx):
        return {"action": np.asarray(record["action"], dtype=np.float32)}


register_encoder("acme/action", ActionEncoder, version="1")
```

The corresponding modality uses `"source": "encoder:action"`. Test exact dtype and shape,
missing values, padding, round trips through the intended container, and reconstruction by name
for declarative jobs. Encoders must not mutate input records.

### Methods and filters

A custom method returns one finite priority per segment, with higher values selected first:

```python
import numpy as np
from kuration import register_method


def newest_first(segments, strategy, *, seed=0, workers=1):
    return np.asarray([segment.start_idx for segment in segments], dtype=np.float64)


register_method("acme/newest", newest_first, capabilities={"whole_pool": True}, version="1")
```

A filter returns retained `Segment` objects in deterministic order:

```python
import math
from kuration import register_filter


def finite_features(segments, feature_names, *, seed=0, workers=1):
    return [
        segment for segment in segments
        if all(math.isfinite(segment.features[name]) for name in feature_names)
    ]


register_filter("acme/finite", finite_features, version="1")
```

Declare a parameter JSON Schema for any method/filter knobs. Test empty and tiny pools,
determinism across worker counts, output shape/order, non-finite failures, and interaction with
predicates, tag coverage, group caps, and hours-based filling when relevant.

### Containers and presets

Register a container with a filename extension and inverse writer/reader callables:

```python
import numpy as np
from kuration import register_container


def write_acme(path, arrays, n_steps):
    with open(path, "wb") as stream:
        np.savez(stream, _n_steps=np.asarray(n_steps), **arrays)


def read_acme(path):
    with np.load(path) as stored:
        arrays = {name: stored[name] for name in stored.files if name != "_n_steps"}
        return arrays, int(stored["_n_steps"])


register_container(
    "acme/tensor",
    extension=".acme",
    writer=write_acme,
    reader=read_acme,
    version="1",
)
```

Container writers receive the final path, modality arrays, and real unpadded length. Readers
must return `(arrays, n_steps)`. Test scalar and higher-rank tensors, short/padded shards, remote
path behavior if advertised, resume, manifest loading, and dtype/shape round trips.

An output preset is a factory `(schema, **parameters) -> OutputSpec` registered with
`register_preset`. It should assemble ordinary declared modalities rather than bypass output
validation. Include a closed parameter schema and test strict categorical behavior.

### Packaging external plugins

Package extensions that should remain independently versioned or distributed outside
`kuration-sdk` as their own installable Python package. Registration code may be loaded explicitly
through a job's `plugins` list or automatically by the `kuration.plugins` entry-point group:

```toml
[project.entry-points."kuration.plugins"]
acme = "acme_kuration:register"
```

The target may be a module that registers on import or a zero-argument registration function.
Keep plugin imports lightweight and lazy-load optional frameworks inside factories or execution
methods. Supply descriptor metadata such as `description`, `parameters`, `capabilities`, `extra`,
`install_hint`, `package`, `version`, `documentation`, and `stability`. Verify discovery with:

```bash
kur-list operators --json
kur-describe operator acme/balance-signal --json
```

Plugin tests should start in a clean process so registration/import behavior matches spawned
workers and installed entry points. They must also cover an isolated `Registry`; use
`"include_plugin_entry_points": false` in declarative-job tests that must not depend on whatever
other plugins happen to be installed.

## Checks

Run the same core checks as CI:

```bash
python -m pytest tests/ -q
python -m compileall -q src/kuration examples
python -m ruff check src tests examples
python -m mypy
python -m build
```

CI additionally installs the built wheel into a clean environment and verifies that `kuration`,
every installed `kur-*` console command, `schemas/job.schema.json`, and `py.typed` work without
the repository on `sys.path`.

For changes spanning the pipeline or output path, also run the relevant self-contained examples:

```bash
python examples/quickstart.py --out quickstart-out
python examples/01_timeseries_to_shards.py
python examples/02_robotics_action_aware.py
python examples/03_compare_strategies.py
python examples/04_operator_pipeline.py
python examples/05_custom_operators.py
```

The examples synthesize their own data. Do not commit their generated shards or manifests.

## Documentation and style

Use Python 3.11-compatible syntax and keep lines within 100 characters. Follow the existing
module style, use type hints where they clarify public contracts, and write actionable validation
errors. Documentation has one canonical home; summarize and link instead of duplicating a full
explanation across pages.

Any new public object exported through `kuration.__all__`, or any new public `Pipeline` method,
must be documented in `docs/api.md`. User-facing behavior should include a minimal example or a
link to a runnable one.

## Pull requests

Keep pull requests focused and describe:

- the problem and intended behavior;
- the compatibility or artifact-format impact;
- the extension kind affected (adapter, analyzer, container, encoder, filter, method, metric,
  operator, preset, scorer, or transform);
- tests and examples run;
- optional dependencies or compute implications;
- documentation updated.

Before requesting review, confirm that unrelated generated files are absent and that all relevant
CI checks pass. Breaking changes before 1.0 must still be deliberate, documented, and accompanied
by migration guidance when they affect public APIs, jobs, or stored artifacts.

## Releases

Contributors should not publish package artifacts. Maintainers release through the trusted PyPI
workflow described in [docs/releasing.md](https://aintropy-solutions.github.io/kuration-sdk/releasing/). The project license and author
metadata must be resolved before the first public release.
