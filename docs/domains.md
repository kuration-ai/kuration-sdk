# Domain modules

Kuration separates its dataset-agnostic engine from maintained domain capabilities while
shipping both in one distribution:

| module | responsibility | required runtime |
|---|---|---|
| `kuration.core` | schemas, adapters, segmentation, features, selection, output, registries | NumPy |
| `kuration.gameplay.pixel_shift` | action-aligned optical-flow measurements | `[pixel-shift]` |
| `kuration.gameplay.action_state_consistency` | cross-segment baseline analysis | `[action-state]` when measuring frames |
| `kuration.gameplay.slam` | reconstruction, trajectory quality/diversity, point clouds | `[slam]`, `[slam-viz]` |

The historical top-level core API remains supported:

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

The methods call the same registered descriptors used by `.apply(...)`, JSON jobs, discovery,
and provenance. The generic core therefore contains no gameplay component names, while dynamic
callers can continue using:

```python
pipeline.apply("aintropy/slam-quality", min_score=0.6)
```

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
