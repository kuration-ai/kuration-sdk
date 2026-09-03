# Examples

Self-contained, runnable demos. The numbered demos **synthesize their own tiny data**, so they
run anywhere with no downloads. The quick start writes to the directory passed with `--out`;
the other demos use a temporary directory and print its path.

The `public_data/` walkthroughs work with **real published datasets**. Each takes a local path,
fetches the data itself with `--download`, or runs offline against synthetic stand-ins with
`--demo`. Downloading is never part of CI.

```bash
python examples/public_data/ett.py --download --out ett-out
python examples/public_data/person_activity.py --download --out activity-out
```

```bash
pip install "kuration-sdk[all]"                    # install every optional capability
python examples/quickstart.py --out quickstart-out
python examples/01_timeseries_to_shards.py
python examples/02_robotics_action_aware.py
python examples/03_compare_strategies.py
python examples/04_operator_pipeline.py
python examples/05_custom_operators.py
python examples/06_gameplay_operator_methods.py
python examples/07_action_state_consistency.py
```

| script | shows |
|---|---|
| **quickstart** | the tested first run: infer a schema, measure and bin windows, sample a diverse subset, write NPZ shards, and read one back. |
| **01_timeseries_to_shards** | the 5-minute start: point `input()` at a folder of CSVs and let it infer everything — diversity bins, coverage of a rare operating mode, an hours budget, and train/test shards. |
| **02_robotics_action_aware** | robotics credibility: angular joints + positional end-effector, a learned **action-aware scorer** (`action_residual`) **composed** with geometric coverage + phase coverage — quality and diversity as separate weighted axes. |
| **03_compare_strategies** | the experiment harness: score once, select per arm (baseline / fast / high-amplitude), report each arm's Jensen-Shannon divergence from baseline per metric. |
| **04_operator_pipeline** | the pipeline path in depth: one chain run by **count** and by **hours**, `.describe()` as the resolved-but-unrun plan, and the operator chain recorded in `config.json`. |
| **05_custom_operators** | reusable extension path: register, discover, and run compositional and transform-backed operators through short and namespace-qualified methods. |
| **06_gameplay_operator_methods** | maintained domain path: import gameplay, call SLAM trajectory operators directly on the core `Pipeline`, and use the qualified spelling without a model or GPU. |
| **07_action_state_consistency** | pipeline-preserving stateful analysis: join lazy frames, run `.action_state_consistency(...)`, and inspect passing/failing verdict metadata without PyAV. |
| **notebooks/gameplay_stats** | JupyterLab lifecycle: automatic notebook progress, uncurated/curated/emitted stats, gameplay plots, and optional synchronized before/after video review. |
| **notebooks/slam_kurious** | output-free SLAM reconstruction, trajectory curation, and notebook point-cloud exploration. |
| **public_data/ett** | real data, pipeline API only: infer a schema from an hourly transformer CSV, bin day-length windows on load and temperature swing, and select a spread — no schema, adapter, or strategy dict. |
| **public_data/person_activity** | the same plain path at full shape: 25 real wearable recordings, four tags as the entity axis, windows in seconds, a coverage floor that keeps rare activities, and source-level splits stratified by person. |
| **public_data/teapearce_csgo** | a bounded public-data workflow: adapt TeaPearce's frame-keyed HDF5, balance action diversity, preserve rare combat events, and optionally emit raw frame tensors. |
| **public_data/open_p2p** | parse Open P2P protobuf annotations, balance cross-game actions, retain system corrections and instruction spans, and join selected indices back to video. |

For the full recipe catalog (cloud I/O, streaming, filters, composition, custom adapters/
encoders/metrics/scorers, the CLIs), see [`../docs/usage.md`](../docs/usage.md).

For dataset provenance, licensing notes, bounded download instructions, offline structural
demos, and the equivalent command-line sessions, see
[`public_data/ett.md`](public_data/ett.md),
[`public_data/person_activity.md`](public_data/person_activity.md),
[`public_data/open_p2p.md`](public_data/open_p2p.md), and
[`public_data/teapearce_csgo.md`](public_data/teapearce_csgo.md).
