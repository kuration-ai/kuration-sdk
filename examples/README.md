# Examples

Self-contained, runnable demos — each **synthesizes its own tiny data**, so they run anywhere
with no downloads. The quick start writes to the directory passed with `--out`; the other demos
use a temporary directory and print its path.

```bash
pip install -e '.[all]'                           # install every optional capability
python examples/quickstart.py --out quickstart-out
python examples/01_timeseries_to_shards.py
python examples/02_robotics_action_aware.py
python examples/03_compare_strategies.py
python examples/04_operator_pipeline.py
python examples/05_custom_operators.py
python examples/06_gameplay_operator_methods.py
```

| script | shows |
|---|---|
| **quickstart** | the tested first run: infer a schema, measure and bin windows, sample a diverse subset, write NPZ shards, and read one back. |
| **01_timeseries_to_shards** | the 5-minute start: declare a schema, map CSV columns to fields, build a fluent pipeline, sample generic device logs for diversity + mode coverage, and write train/test shards. |
| **02_robotics_action_aware** | robotics credibility: angular joints + positional end-effector, a learned **action-aware scorer** (`action_residual`) **composed** with geometric coverage + phase coverage — quality and diversity as separate weighted axes. |
| **03_compare_strategies** | the experiment harness: score once, select per arm (baseline / fast / high-amplitude), report each arm's Jensen-Shannon divergence from baseline per metric. |
| **04_operator_pipeline** | the fluent primary path: infer rows, compose measures/filters/bins, preflight, write count- and hours-based datasets, and inspect provenance. |
| **05_custom_operators** | reusable extension path: register, discover, and run compositional and transform-backed operators through short and namespace-qualified fluent methods. |
| **06_gameplay_operator_methods** | maintained domain path: import gameplay, call SLAM trajectory operators directly on the core `Pipeline`, and use the qualified spelling without a model or GPU. |
| **notebooks/gameplay_stats** | JupyterLab lifecycle: automatic notebook progress, uncurated/curated/emitted stats, gameplay plots, and optional synchronized before/after video review. |
| **notebooks/slam_kurious** | output-free SLAM reconstruction, trajectory curation, and notebook point-cloud exploration. |
| **public_data/teapearce_csgo** | a bounded public-data workflow: adapt TeaPearce's frame-keyed HDF5, balance action diversity, preserve rare combat events, and optionally emit raw frame tensors. |
| **public_data/open_p2p** | parse Open P2P protobuf annotations, balance cross-game actions, retain system corrections and instruction spans, and join selected indices back to video. |

For the full recipe catalog (cloud I/O, streaming, filters, composition, custom adapters/
encoders/metrics/scorers, the CLIs), see [`../docs/usage.md`](../docs/usage.md).

For dataset provenance, licensing notes, bounded download instructions, and offline structural
demos, see [`public_data/open_p2p.md`](public_data/open_p2p.md) and
[`public_data/teapearce_csgo.md`](public_data/teapearce_csgo.md).
