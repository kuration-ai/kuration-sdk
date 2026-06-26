# Examples

Self-contained, runnable demos — each **synthesizes its own tiny data**, so they run anywhere
with no downloads and leave nothing behind (output goes to a tempdir, printed at the end).

```bash
pip install -e '.[methods]'                       # 02 needs scikit-learn; 01 & 03 need only numpy
python examples/01_timeseries_to_shards.py
python examples/02_robotics_action_aware.py
python examples/03_compare_strategies.py
```

| script | shows |
|---|---|
| **01_timeseries_to_shards** | the 5-minute start: declare a schema, map CSV columns to fields (no code), curate generic device logs for diversity + mode-coverage, write train/test shards. |
| **02_robotics_action_aware** | robotics credibility: angular joints + positional end-effector, a learned **action-aware scorer** (`action_residual`) **composed** with geometric coverage + phase coverage — quality and diversity as separate weighted axes. |
| **03_compare_strategies** | the experiment harness: score once, select per arm (baseline / fast / high-amplitude), report each arm's Jensen-Shannon divergence from baseline per metric. |

For the full recipe catalog (cloud I/O, streaming, filters, composition, custom adapters/
encoders/metrics/scorers, the CLIs), see [`../docs/usage.md`](../docs/usage.md).
