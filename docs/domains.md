# Domain modules

Kuration separates its dataset-agnostic engine from maintained domain capabilities while
shipping both in one distribution:

```text
src/kuration/
├── __init__.py             # concise package-level exports only
├── py.typed
├── core/                   # generic SDK implementation
│   ├── adapters/
│   ├── cli/
│   ├── dsl/
│   ├── output/
│   └── ...
└── gameplay/               # maintained gameplay implementation
    ├── action_state_consistency/
    ├── pixel_shift/
    └── slam/
        └── _vendor/        # private SLAM runtime dependencies
```

No implementation modules or domain compatibility packages live beside `core/` and
`gameplay/`. Third-party code used only by SLAM is private to that domain.

The `_vendor/` subtree is an attributed upstream source snapshot, not Kuration-owned
implementation. CI validates its packaging, imports, configuration resolution, and integration
through the public SLAM boundary, while first-party line coverage measures the surrounding
Kuration adapters and orchestration. The subtree is also excluded from mypy; changes to the
vendored snapshot require the focused SLAM regression suite and preserved license notices.

| module | responsibility | required runtime |
|---|---|---|
| `kuration.core` | schemas, adapters, segmentation, features, selection, output, registries | NumPy |
| `kuration.gameplay.pixel_shift` | action-aligned optical-flow measurements | `[pixel-shift]` |
| `kuration.gameplay.action_state_consistency` | cross-segment baseline analysis | `[action-state]` when measuring frames |
| `kuration.gameplay.slam` | reconstruction, trajectory quality/diversity, point clouds | `[slam]`, `[slam-viz]` |

The concise package-level core API remains supported through exports from `kuration.core`:

```python
from kuration import input
from kuration.core import Pipeline, SchemaSpec
```

Gameplay starts from its own facade and returns a `GameplayPipeline`, which preserves its
domain methods across every immutable core operation:

```python
import kuration.gameplay as gameplay

pipeline = (
    gameplay.slam_pipeline("slam-export/manifest.json")
    .window(32, stride=16)
    .slam_motion()
    .slam_quality(min_score=0.6)
)
```

The methods call the same registered descriptors used by custom modules, plugins, JSON jobs,
discovery, and provenance. The generic core therefore contains no gameplay component names.
Every registered operator gets the same direct fluent spelling, including operators imported
after the pipeline was created:

```python
pipeline.slam_quality(min_score=0.6)
```

For a custom name such as `acme/signal-diversity`, use
`pipeline.signal_diversity(...)`. The namespace-qualified
`pipeline.acme__signal_diversity(...)` spelling disambiguates collisions.

Importing `kuration` does not import gameplay. Importing `kuration.gameplay` registers the
maintained gameplay components but still does not import OpenCV, PyTorch, FAISS, PyAV, or Trimesh.
Those frameworks load only when their execution paths are reached.

## Installation

```bash
pip install kuration-sdk
pip install "kuration-sdk[gameplay]"
```

Granular extras remain available:

```bash
pip install "kuration-sdk[pixel-shift]"
pip install "kuration-sdk[action-state]"
pip install "kuration-sdk[slam]"
```

The `standard-operators` extra remains a compatibility alias for `gameplay`.
