# Diverse subset

Compress a redundant dataset by clustering in feature space and sampling
equally from each cluster. An optional dedup pre-pass filters near-duplicate
windows before clustering. Useful when your corpus has heavy redundancy (same
device replaying the same conditions) and you want a smaller subset that
covers the actual feature diversity.

**Composes:** `.filter("dedup", ...) -> .sample("cluster-balanced", ...)` .

**Extras needed:** `[methods]` (scikit-learn). `pip install kuration-kuration-ai` includes it.

## Run

```
python examples/diverse-subset/diverse_subset.py
```

## Expected output

```
selected: 15 windows
features on first window: ['signal_mean', 'signal_std', 'temp_std']

compiled chain:
  DiverseSubset: .filter('dedup', feature_space=['signal_std', 'signal_mean', 'temp_std'], threshold=0.98) -> .sample('cluster-balanced', n=15, feature_space=['signal_std', 'signal_mean', 'temp_std'], clusters=5)
```

## What to try next

- Drop `dedup_threshold=None` to skip the dedup pre-pass entirely.
- Increase `n_clusters` (5 → 32 → 128) for larger datasets — the pipeline builder default is 2048.
- Change the `features` list to cluster on different measured columns.
- Measure additional features before applying the operation to see how the diversity axis changes.

## Related

- [Kit gallery](../README.md)
- [`design/01_operation_abstraction.md`](../../design/01_operation_abstraction.md) — the Operation contract
- [`design/03_config_model.md`](../../design/03_config_model.md) — driving this from a YAML config
