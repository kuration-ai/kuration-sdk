# kuration-ai

Curated training-dataset selection for Python: the sampling half of world-model curation. Three composable operations. `BalancedSelector` for motion-diversity coverage, `RareEventCoverage` for oversampling collisions and other rare events, and `DiverseSubset` for multi-axis diversity sampling. Plus a small pattern for writing your own.

## Install

```bash
pip install kuration-ai
```

Runs on Python 3.11 or later. Depends on `kuration-sdk[methods,parquet]`.

## First curated result

```python
from kuration_ai import input, RareEventCoverage

sessions = {"clip-1": [{"t": float(i), "collision": 1 if i % 30 == 0 else 0} for i in range(200)]}
pipeline = input(sessions, time="t", hz=10.0).window(8, stride=4)
result = (RareEventCoverage(of="collision", count=15, weight=2.0)
          .apply(pipeline).options(seed=1).collect())
print(f"selected {len(result.selected)} windows")
```

Prints `selected 15 windows`.

## Full documentation

Everything else lives at the repo root: notebooks, the full use-case gallery, design docs, and the composition/config/extension guides.

**→ https://github.com/aintropy-solutions/kuration-sdk**

## License

Apache 2.0. See [`LICENSE`](../LICENSE) at the repo root.
