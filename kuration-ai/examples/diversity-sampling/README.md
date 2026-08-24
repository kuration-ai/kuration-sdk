# Diversity sampling

**Cluster in feature space so redundant windows don't crowd out the diverse ones.**

Most activity-based metrics move together. A chaotic firefight is high on every axis at once; an idle stretch is low on all of them. Pick a single axis to sample on and you inherit whatever the other axes happen to correlate with. Cluster the windows in a multi-axis feature space and sample equally from each cluster, and the redundant part of the corpus stops crowding out the parts that actually differ from it.

The Intro World-Model-Data-Curation blog names this "diversity sampling", one of the things "collision detection, kinematics, action-state checks, and diversity sampling all get easier on a dataset that previously had no notion of space." `DiverseSubset` is the operator for the sampling half: cluster over any set of pre-measured features and take an equal share from every cluster. Optional dedup pre-pass filters near-duplicates before clustering.

**Composes:** `.filter("dedup", ...) -> .sample("cluster-balanced", ...)` .

**Extras needed:** `[methods]` (scikit-learn). `pip install kuration-ai` includes it.

## Run

```
python examples/diversity-sampling/diversity_sampling.py
```

## Expected output

```
selected: 15 windows
features on first window: ['climb_std', 'motion_mean', 'motion_std']

compiled chain:
  DiverseSubset: .filter('dedup', feature_space=['motion_std', 'motion_mean', 'climb_std'], threshold=0.98) -> .sample('cluster-balanced', n=15, feature_space=['motion_std', 'motion_mean', 'climb_std'], clusters=5)
```

## What to try next

- Drop `dedup_threshold=None` to skip the dedup pre-pass entirely.
- Increase `n_clusters` (5 → 32 → 128) for larger datasets. The pipeline builder default is 2048.
- Change the `features` list to cluster on different measured columns. The SDK's SLAM motion axes (`slam_path_length`, `slam_turn_degrees`, `slam_max_turn_degrees`, `slam_motion_efficiency`, `slam_spatial_extent`) make a natural feature space when combined.
- Measure additional features before applying the operation to see how the diversity axis changes.

## Related

- [Kit gallery](../README.md)
- [`design/01_operation_abstraction.md`](../../design/01_operation_abstraction.md): the Operation contract
- [`design/03_config_model.md`](../../design/03_config_model.md): driving this from a YAML config
