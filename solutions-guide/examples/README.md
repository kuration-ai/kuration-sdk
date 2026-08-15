# Gallery

Three self-contained examples. Each folder holds a runnable script and its own
README with expected output.

- [**Balanced sensor shards**](./balanced-sensor-shards/) — cover the whole
  range of a signal's variability instead of the natural distribution peak.
- [**Rare-event coverage**](./rare-event-coverage/) — sample so rare fault
  events don't get dropped by ordinary distribution matching.
- [**Diverse subset**](./diverse-subset/) — cluster in feature space to
  compress a redundant dataset while preserving coverage.

Also here:

- [`chained.py`](./chained.py) — the three call styles (direct `.apply()`,
  `run()` with instances, `run()` with a config dict) all producing identical
  results. Not a gallery entry — a config-parity reference for developers
  extending Kuration AI.

## Run any of them

```
pip install kuration-kuration-ai
python examples/<folder>/<script>.py
```

Each script is a complete, self-contained ~30 lines of Python that constructs
a small synthetic pipeline and applies one Operation. No downloads, no
credentials, no network required.
