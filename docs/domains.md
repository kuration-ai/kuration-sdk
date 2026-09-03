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
├── gameplay/               # maintained gameplay implementation
│   ├── action_state_consistency/
│   ├── collision/
│   ├── pixel_shift/
│   └── slam/
│       └── _vendor/        # private SLAM runtime dependencies
├── selection/              # general-purpose curation recipes
│   ├── operations.py       # BalancedSelector, RareEventCoverage, DiverseSubset
│   ├── operators.py        # their operator wrappers
│   └── plan.py             # config-driven run/plan/describe runner
└── sample_data/            # bundled sample datasets for docs and quickstarts
```

No implementation modules live beside `core/`, `gameplay/`, `selection/`, and `sample_data/`.
Third-party code used only by SLAM is private to that domain.

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
| `kuration.gameplay.collision` | CS2 wall-collision detection/classification against map geometry | `[collision]` |
| `kuration.gameplay.slam` | reconstruction, trajectory quality/diversity, point clouds | `[slam]`, `[slam-viz]` |
| `kuration.gameplay` stats/video review | automatic action, media, and SLAM summaries; synchronized comparisons | `[stats]` for plots, `[video]` for probing |
| `kuration.selection` | balanced coverage, rare-event weighting, cluster-balanced diversity; see [Selection recipes](selection.md) | NumPy; `[methods]` for `DiverseSubset` |

The concise package-level core API remains supported through exports from `kuration.core`:

```python
from kuration import input
from kuration.core import Pipeline, SchemaSpec
```

Gameplay is an operator domain over the same `Pipeline`. Importing it registers its maintained
operators; manifest helpers and every immutable operation return the core pipeline type:

```python
import kuration.gameplay as gameplay
from kuration import Pipeline

pipeline = (
    gameplay.slam_reconstruct("slam-export/manifest.json")
    .window(32, stride=16)
    .slam_motion_diversity()
    .slam_quality(min_score=0.6)
)

assert type(pipeline) is Pipeline
report = pipeline.stats()
report.plot()
report.examples(metric="slam_quality_score", compare="selected-vs-rejected")
```

The lowercase `gameplay.slam_reconstruct(...)` is a convenience function, not a class. It performs
or reuses reconstruction and returns the ordinary `Pipeline`.
`gameplay.SLAMReconstructor` coordinates the global, artifact-producing
video-to-reconstruction workflow, then hands its manifest to `Pipeline` for bounded curation.

The methods call the same registered descriptors used by custom modules, plugins, JSON jobs,
discovery, and provenance. The generic core therefore contains no gameplay component names.
Every registered operator gets the same direct method spelling, including operators imported
after the pipeline was created:

```python
pipeline = pipeline.slam_quality(min_score=0.6)
pipeline = pipeline.action_state_consistency(baselines=baselines, controls=["W"])
```

Input enrichments keep their result on the derived pipeline. For action-state consistency (a
dataset-mode-only audit -- `pipeline` above must be artifact-backed) it is available at
`pipeline.metadata["gameplay"]["action_state_consistency"]["result"]`; `.analyze(...)` remains
the generic detached-result alternative. For a live pipeline, `.pixel_shift(...).score(...)` is
the equivalent per-window path -- see [action-state-consistency.md](action-state-consistency.md).

For a custom name such as `acme/signal-diversity`, use
`pipeline.signal_diversity(...)`. The namespace-qualified
`pipeline.acme__signal_diversity(...)` spelling disambiguates collisions.

Importing `kuration` does not import gameplay. Importing `kuration.gameplay` registers the
maintained gameplay components and a gameplay section for the ordinary `StatsReport`, but still
does not import OpenCV, PyTorch, FAISS, PyAV, Matplotlib, or Trimesh. Those libraries load only
when their execution or visualization paths are reached.

Video review follows the same domain boundary:

```python
review = gameplay.compare_videos("capture.mp4", "selected.mp4")
review                         # synchronized rich display in Jupyter
review.plot(frames=4)          # matched contact sheet
review.save("review.html")     # standalone synchronized players
```

For a live video-backed pipeline, start with `pipeline.stats().examples(...)`: gameplay adds
bounded frame strips, action summaries, and named SLAM diagnostics directly to the ordinary
evidence result. `compare_videos()` remains the explicit whole-video tool.

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
