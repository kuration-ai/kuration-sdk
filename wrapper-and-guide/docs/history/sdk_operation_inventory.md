# Kuration SDK Registered-Component Inventory

Source of truth: registration calls found under `~/work/aintropy/kuration-sdk/src/kuration/`. Extras are the SDK's `[project.optional-dependencies]` names from `pyproject.toml`. "Runtime imports" lists top-level heavy imports that would fail at *execution* if the extra isn't installed (all guarded by lazy `import`s inside functions: descriptors still register cleanly).

Generated: 2026-08-11 as Phase 0.1 deliverable for issue #4.

## Operators

| Name | Registered in | One-sentence purpose | Required params | Optional params (default) | Descriptor `extra` / `install_hint` | Runtime imports needed |
|---|---|---|---|---|---|---|
| `aintropy/action-state-consistency` | `src/kuration/gameplay/action_state_consistency/__init__.py` (`attach_action_state_consistency`) | Check action-state consistency and attach the verdict to pipeline metadata. | (`baselines` OR `controls`) validated by handler | `anomaly_type: str = "action_state"`, `calibration: object`, `boundary: "exclude"|"error"|"assume-inactive" = "exclude"`, `mode: "stream"|"dataset"`, `inputs: str[]`, `batch_size: int`, `frames: str`, `actions: str`, `color_space: str`, `verify: bool`, `metadata_key: str = "action_state_consistency"`, `replace: bool = False` | `extra="action-state"`, `install_hint='pip install "kuration-sdk[action-state]"'` | opencv-python (`cv2`, lazy in `diamond_action_decode._prep_gray`) |
| `aintropy/collision` | `src/kuration/gameplay/collision/__init__.py` (`collision_mechanics`) | Compose the CS2 collision-detection transform into a fluent pipeline. | `map_for_source: object`, `map_registry: object`, `dist_path: str` | `disk_cache_dir: str = ".kuration_collision_cache"`, `min_score: number = 2.0`, `proximity: number = 40.0`, `min_approach: number = 250.0`, `max_residual: number = 20.0`, `slide_angle: number = 75.0`, `max_slide_speed: number = 400.0`, `min_release_accel: number = 100.0`, `max_release_vn: number = 100.0` | `extra="collision"`, `install_hint='pip install "kuration-sdk[collision]"'` | trimesh + rtree (lazy in `mesh_cache.load_mesh_cached`) |
| `aintropy/collision-directional` | `src/kuration/gameplay/collision/__init__.py` (`attach_directional_collision`) | Run directional collision detection over the tick stream and attach the verdict to pipeline metadata. | `map_registry: object`, `ramp_models_path: str` | `map_for_source: object`, `default_map: str`, `fallback_ramp: [number,number]`, `disk_cache_dir: str`, `proximity: number`, `min_missing_speed: number`, `concentration_ratio: number (0..1)`, `batch_size: int`, `inputs: str[]`, `metadata_key: str = "directional_collision"`, `replace: bool = False` | `extra="collision"`, `install_hint='pip install "kuration-sdk[collision]"'` | trimesh + rtree (lazy) |
| `aintropy/pixel-shift` | `src/kuration/gameplay/pixel_shift/__init__.py` (`pixel_shift`) | Measure pixel shift for every isolated key/mouse press. | (none) | `controls: str[]` (default: all 8 controls W/S/A/D/MOUSE_X_POS/…/MOUSE_Y_NEG), `boundary: "exclude"|"error"|"assume-inactive" = "exclude"` | `extra="pixel-shift"`, `install_hint='pip install "kuration-sdk[pixel-shift]"'` | opencv-python (`cv2`, lazy) |
| `aintropy/slam-motion-diversity` | `src/kuration/gameplay/slam/__init__.py` (`slam_motion_diversity`) | Add SLAM motion features and quantile-bin a selected diversity axis. | (none) | `axis: enum(slam_path_length|slam_displacement|slam_motion_efficiency|slam_mean_step|slam_max_step|slam_turn_degrees|slam_mean_turn_degrees|slam_max_turn_degrees|slam_spatial_extent) = "slam_path_length"`, `bins: int ≥ 2 = 8` | `extra="slam"`, `install_hint='pip install "kuration-sdk[slam]"'` | numpy only for this operator; SLAM extra pulls torch/torchvision/faiss-cpu/av/opencv/etc. needed by sibling reconstruction |
| `aintropy/slam-quality` | `src/kuration/gameplay/slam/__init__.py` (`slam_quality`) | Add SLAM reconstruction-quality signals and optionally filter windows. | (none) | `min_tracking_confidence: number [0..1] = 0.5`, `translation_ratio: number > 0 = 10.0`, `translation_mad_multiplier: number > 0 = 8.0`, `max_translation_step: number > 0 = None`, `max_turn_degrees: number (0..180] = 90.0`, `min_score: number [0..1] = None` | `extra="slam"`, `install_hint='pip install "kuration-sdk[slam]"'` | numpy only for this operator |

## Methods

Base methods and feature-space methods are registered without a JSON-Schema descriptor for their own kwargs (the strategy validates `method_params` at load), so param columns show the callable's kwargs rather than a descriptor schema.

| Name | Registered in | Purpose | Params (defaults) | `extra` | Runtime imports |
|---|---|---|---|---|---|
| `uniform-bins` | `src/kuration/core/strategy.py` line 559 (registers `_raw_priorities`) | Quantile-bin along declared axes and round-robin the bins for coverage-first selection. | strategy-level `axes`, no method-specific kwargs | none | numpy only |
| `weighted` | `src/kuration/core/strategy.py` line 559 | Rank by a weighted linear combination of normalized features. | strategy-level `weights` / `normalize` | none | numpy only |
| `top-k` | `src/kuration/core/strategy.py` line 559 | Rank by a single feature (or weighted score) and take the highest priorities. | strategy-level `weights` | none | numpy only |
| `target-distribution` | `src/kuration/core/strategy.py` line 559 | Match the pool to an explicit target distribution over a scalar feature. | strategy-level `target` | none | numpy only |
| `filter` | `src/kuration/core/strategy.py` line 559 | Keep every segment passing the predicate list: no ranking. | strategy-level `predicates` | none | numpy only |
| `cluster-balanced` | `src/kuration/core/methods.py` line 407 (`cluster_balanced_priorities`) | Joint k-means-cluster round-robin coverage over the feature space (DINOv2-style balanced subsampling). | `n_clusters=2048`, `normalize="zscore"`, `seed=0`, `workers=1` | `extra="methods"` | scikit-learn (`KMeans` / `MiniBatchKMeans`, lazy) |
| `k-center` | `src/kuration/core/methods.py` line 407 (`kcenter_priorities`) | Farthest-point traversal of the cluster centroids ordered by coverage. | `n_clusters=2048`, `normalize="zscore"`, `seed=0`, `workers=1` | `extra="methods"` | scikit-learn |
| `facility-location` | `src/kuration/core/methods.py` line 407 (`facility_location_priorities`) | Submodular facility-location cluster ordering (greedy coverage gain). | `n_clusters=2048`, `normalize="zscore"`, `seed=0`, `workers=1` | `extra="methods"` | scikit-learn |
| `dsir` | `src/kuration/core/methods.py` line 407 (`dsir_priorities`) | Multivariate importance resampling with Gumbel-top-k over a quantized joint feature space. | `n_bins=8`, `target=None|{metric}|{anchor_metric,anchor_quantile}`, `normalize="quantile"`, `seed=0`, `workers=1` | `extra="methods"` | numpy only |
| `rare` | `src/kuration/core/methods.py` line 407 (`rare_priorities`) | Density-inverse kNN atypicality (sparse = admitted first). | `k=10`, `normalize="zscore"`, `seed=0`, `workers=1`, `sample=20000` | `extra="methods"` | scikit-learn (`NearestNeighbors`) |
| `typical` | `src/kuration/core/methods.py` line 407 (`typical_priorities`) | Prototypicality: the negative of `rare` (densest first). | same as `rare` | `extra="methods"` | scikit-learn |

## Adapters

| Name | Registered in | Purpose | Required init params | Optional params (default) | `extra` | Runtime imports |
|---|---|---|---|---|---|---|
| `aintropy/collision-asg` | `src/kuration/gameplay/collision/__init__.py` (`AsgCollisionAdapter`) | Read ASG `data.json` CS2 match replays as one Tick stream per (match, player). | (none: descriptor empty) | n/a | `extra="collision"` | stdlib `json` + numpy |
| `aintropy/collision-teapearce` | `src/kuration/gameplay/collision/__init__.py` (`TeaPearceDust2Adapter`) | Read TeaPearce/CounterStrike_Deathmatch dust2 `.npy` metadata files. | (none in descriptor) | n/a | `extra="collision"` | numpy |
| `aintropy/slam-trajectory` | `src/kuration/gameplay/slam/__init__.py` (`SLAMTrajectoryAdapter`) | Read per-clip camera trajectories from a Kuration SLAM manifest. | `manifest: str` | `schema=None` | `extra="slam"` | numpy (adapter only) |
| `curated` | `src/kuration/core/adapters/curated.py` (`CuratedDatasetAdapter`) | Wrap one or more Kuration dataset directories as a re-curatable source. | `dataset_dir: str \| str[]` | `schema: SchemaSpec = None`, `registry = None` | delegates to container extras | h5py / pyarrow lazily |
| `hdf5` | `src/kuration/core/adapters/hdf5.py` (`HDF5Adapter`) | Map datasets in one HDF5 file to a canonical single-entity tick stream. | `schema: SchemaSpec`, `fields: dict` | `globals: dict = None` | runtime hint → `[array]` | h5py (lazy) |
| `huggingface` | `src/kuration/core/adapters/huggingface.py` (`HuggingFaceAdapter`) | Read a Hugging Face dataset/split as a canonical Tick stream. | `schema`, `fields`, `dataset: str` | `split="train"`, `config=None`, `time_column=None`, `entity_column=None`, `globals=None`, `source_column=None`, `load_kwargs=None` | runtime hint → `[hf]` | `datasets` (lazy) |
| `jsonl` | `src/kuration/core/adapters/jsonl.py` (`JsonlAdapter`) | Read a JSON-Lines file as a Tick stream. | `schema`, `fields` | `time_column=None`, `entity_column=None`, `globals=None` | none | stdlib `json` |
| `npz` | `src/kuration/core/adapters/npz.py` (`NpzAdapter`) | Map arrays in one NumPy NPZ archive to a canonical single-entity tick stream. | `schema`, `fields` | `globals=None` | none | numpy (`np.load`) |
| `parquet-dataset` | `src/kuration/core/adapters/parquet_dataset.py` (`ParquetDatasetAdapter`) | Partitioned-Parquet directory/glob as a set of curation sources. | `schema`, `fields`, `path: str` | `source_per="part"\|"dataset"`, `time_column=None`, `entity_column=None`, `globals=None` | runtime hint → `[parquet]` | pyarrow.parquet (lazy) |
| `timeseries` | `src/kuration/core/adapters/timeseries.py` (`TimeseriesAdapter`) | Turn a CSV/Parquet table (or in-memory list of row dicts) into a canonical Tick stream via a column map. | `schema`, `fields` | `time_column=None`, `entity_column=None`, `globals=None` | none for CSV; parquet needs `[parquet]` | csv (stdlib) or pyarrow (lazy) |
| `zarr` | `src/kuration/core/adapters/zarr_store.py` (`ZarrAdapter`) | Map arrays in one local or remote Zarr store to a canonical tick stream. | `schema`, `fields` | `globals=None` | runtime hint → `[zarr]` (+ `[cloud]` remote) | `zarr` (lazy); `fsspec` for remote |

## Analyzers

| Name | Registered in | Purpose | Required params | Optional params | `extra` | Runtime imports |
|---|---|---|---|---|---|---|
| `aintropy/action-state-consistency` | `src/kuration/gameplay/action_state_consistency/__init__.py` (`action_state_consistency`) | Classify clips from cross-segment pixel-shift evidence against user-supplied action baselines. | `anomaly_type: "action_state"` | `baselines: object|null`, `calibration: {min_clips, min_samples, sign_consensus, tolerance}` | `extra="action-state"` | opencv-python (lazy) |
| `aintropy/collision-directional` | `src/kuration/gameplay/collision/__init__.py` (`directional_collision_analyzer`) | Flag per-tick candidate wall collisions using a fitted acceleration-ramp model + map geometry. | `map_registry: object`, `ramp_models_path: str` | `map_for_source`, `default_map`, `fallback_ramp: [tau,cap]=(3.0,215.0)`, `disk_cache_dir=".kuration_collision_cache"`, `proximity=40.0`, `min_missing_speed=40.0`, `concentration_ratio=0.6` | `extra="collision"` | trimesh + rtree (lazy) |
| `kuration/split-leakage` | `src/kuration/core/analyzers.py` (`_split_leakage_analyzer`) | Audit source and (optionally) group assignments across dataset splits for train/test leakage. | (none) | `check_groups: bool = True` | none | stdlib only |

## Containers

Registered via a loop in `src/kuration/core/output/containers.py` with `capabilities={"remote_paths": True, "streaming": True}`, no `extra` on descriptor (runtime error still guides).

| Name | Extension | On-disk purpose | Deps needed at runtime |
|---|---|---|---|
| `hdf5` | `.h5` | Compressed HDF5 shard (`n_steps` attr in-band). | h5py (`[array]`) |
| `json` | `.json` | Dependency-free JSON dump of arrays + `_n_steps`. | stdlib |
| `npz` | `.npz` | Dependency-free NumPy `.npz` archive of arrays + `_n_steps`. | numpy |
| `parquet` | `.parquet` | Flat Parquet shard (scalar/tensor columns) with `_n_steps` in schema metadata. | pyarrow (`[parquet]`) |

## Filters

| Name | Registered in | Purpose | Params (defaults) | `extra` | Runtime imports |
|---|---|---|---|---|---|
| `anomaly` | `src/kuration/core/methods.py` line 411 (`anomaly_filter`) | Drop multivariate outliers with an Isolation Forest. | `contamination=0.02`, `normalize="zscore"`, `seed=0`, `workers=1` | `extra="methods"` | scikit-learn (`IsolationForest`, lazy) |
| `dedup` | `src/kuration/core/methods.py` line 411 (`dedup_filter`) | Feature-space SemDeDup: within each cluster keep one representative per near-duplicate group. | `threshold=0.98`, `n_clusters=2048`, `normalize="zscore"`, `seed=0`, `workers=1` | `extra="methods"` | scikit-learn |

## Presets

| Name | Registered in | Purpose | Params (defaults) | `extra` | Runtime imports |
|---|---|---|---|---|---|
| `numpy-timeseries` | `src/kuration/core/output/presets.py` (`_numpy`) | NumPy NPZ time-series shards. | `container: str = "npz"`, `shard_length: int ≥ 1 = 64`, `short_policy: enum{keep,pad,drop} = "keep"`, `categorical: enum{error,omit,integer-vocab,encoder}`, `categorical_encoders: object`, `categorical_vocabs: object`, `presence_mask: bool` | none | numpy |
| `pytorch-sequences` | `src/kuration/core/output/presets.py` (`_pytorch`) | Padded sequence shards for PyTorch (default `short_policy="pad"`). | as above with `short_policy = "pad"` | none | numpy (Torch only in consumer `ShardDataset`) |
| `tabular` | `src/kuration/core/output/presets.py` (`_tabular`) | Flat Parquet time-series shards (`container="parquet"`, `shard_length=1024`, `short_policy="keep"`). | parquet defaults | `extra="parquet"` | pyarrow |

## Scorers

| Name | Registered in | Purpose | Required params | Optional params | `extra` | Runtime imports |
|---|---|---|---|---|---|---|
| `action_residual` | `src/kuration/core/scorers.py` (`ActionResidualScorer`) | Action-conditioned k-nearest-neighbor predictive residual (whole-pool, buffer-using demonstrator scorer). | `obs: str`, `action: str` | `k: int ≥ 1 = 8` | none on descriptor (runtime → `[methods]`) | scikit-learn (`KNeighborsRegressor`, lazy) |

## Transforms

| Name | Registered in | Purpose | Params + outputs | `extra` | Runtime imports |
|---|---|---|---|---|---|
| `aintropy/collision` | `src/kuration/gameplay/collision/__init__.py` (`CollisionMechanics`) | Detect and classify CS2 wall collisions from per-tick position/angle telemetry raycast against map geometry. | Params: same nine as sibling operator. Outputs: `collision_count, head_on_count, angled_count, w_slide_count, wa_slide_count, slide_end_count, slide_constrained_count, max_anomaly_score, min_dist_to_wall_hu` | `extra="collision"` | trimesh + rtree (lazy) |
| `aintropy/pixel-shift` | `src/kuration/gameplay/pixel_shift/__init__.py` (`PixelShift`) | Per-segment pixel-shift statistics for isolated key/mouse presses. | Params: `controls: str[]` (default: all 8), `boundary: enum = "exclude"`. Outputs: `{control_lower}_{shift_min|shift_max|shift_mean|shift_measured|n_instances|requested}` for each control | `extra="pixel-shift"` | opencv-python (lazy) |
| `aintropy/slam-motion` | `src/kuration/gameplay/slam/__init__.py` (`SLAMMotion`) | Measure translation and camera-turn motion in SLAM trajectory windows. | Params: `EMPTY_PARAMETERS`. Outputs: `slam_path_length, slam_displacement, slam_motion_efficiency, slam_mean_step, slam_max_step, slam_turn_degrees, slam_mean_turn_degrees, slam_max_turn_degrees, slam_spatial_extent`. Inputs: `field:{CAMERA_POSITION}`, `field:{CAMERA_Z_AXIS}` | `extra="slam"` | numpy only |
| `aintropy/slam-quality` | `src/kuration/gameplay/slam/__init__.py` (`SLAMQuality`) | Flag tracking gaps, pose jumps, and unstable trajectory scale over aligned camera trajectories. | Params: `min_tracking_confidence=0.5`, `translation_ratio=10.0`, `translation_mad_multiplier=8.0`, `max_translation_step=None`, `max_turn_degrees=90.0`. Outputs: `slam_tracking_confidence_mean, slam_tracking_gap_fraction, slam_translation_jump_fraction, slam_rotation_jump_fraction, slam_step_scale_drift, slam_quality_score` | `extra="slam"` | numpy only |

## Uncertain / could-not-confirm findings

- **Adapter descriptors have no JSON-Schema `parameters` on the register call** (`register_adapter("hdf5", HDF5Adapter)` etc. pass just the class). "Required / optional" reported above comes from the class `__init__` signature; `describe_component("adapter", name)` will return the empty `{"type":"object","properties":{}, "additionalProperties":False}` default, not the class kwargs.
- **`aintropy/collision-asg` and `aintropy/collision-teapearce` adapter descriptors** are explicitly registered with an empty schema: the descriptor accepts no kwargs. Init also takes no configuration (fully driven by schema constant + source URI).
- **Base methods (`uniform-bins`, `weighted`, `target-distribution`, `top-k`, `filter`)** are registered with only `capabilities={"streaming": True}`: no `description`, `parameters`, or `extra`. "Purpose" cells are from module docstring, not descriptor.
- **Feature-space methods and filters** are registered with `capabilities={"whole_pool": True}` and `extra="methods"`: no `description` or `parameters` on descriptor.
- **`pytorch-sequences` preset**: preset is `[parquet]`-free (writes NPZ by default), no explicit `extra=` on register call. Overrides to `container="parquet"` still hit pyarrow at runtime.
- **`json` and `npz` containers**: registered without `extra`; truly no non-stdlib deps.
- **SLAM transforms**: own `transform_batch` code is numpy-only, but `[slam]` extra pulls torch/torchvision/faiss/av/opencv because sibling `SLAMReconstructor` needs them for upstream reconstruction that produces the manifests these transforms consume.
- **Base adapters** (`hdf5`, `huggingface`, `jsonl`, `npz`, `parquet-dataset`, `timeseries`, `zarr`, `curated`): no `install_hint` on descriptor. Errors surface only at first use.

## Extras → deps mapping (from `pyproject.toml`)

| Extra | Packages | Components that reference it |
|---|---|---|
| `methods` | `scikit-learn>=1.5` | methods `cluster-balanced` / `k-center` / `facility-location` / `dsir` / `rare` / `typical`, filters `anomaly` / `dedup`, scorer `action_residual` |
| `parquet` | `pyarrow>=23.0.1` | adapter `parquet-dataset`, adapter `timeseries` (parquet input), container `parquet`, preset `tabular` |
| `scale` | `pyarrow>=23.0.1` | (Arrow-IPC feature tables: no registered component references this by name) |
| `array` | `h5py>=3.10` | adapter `hdf5`, container `hdf5` |
| `cloud` | `fsspec>=2024.2` | remote source URIs used by `zarr` / any adapter reading `gs://` / `s3://` / `az://` |
| `s3` | `fsspec>=2024.2`, `s3fs>=2024.2` | `s3://` I/O |
| `gcs` | `fsspec>=2024.2`, `gcsfs>=2024.2` | `gs://` I/O |
| `azure` | `fsspec>=2024.2`, `adlfs>=2024.2` | `az://` I/O |
| `torch` | `torch>=2.13` | `kuration.core.torch.ShardDataset` consumer |
| `hf` | `datasets>=2.18` | adapter `huggingface` |
| `zarr` | `zarr>=2.17` | adapter `zarr` |
| `video` | `av>=12` | optional CPU MP4 media provider |
| `ray` | `ray>=2.56.1` | optional distributed execution backend |
| `stats` | `matplotlib>=3.8` | `StatsReport.plot()` |
| `p2p` | `protobuf>=5.29.6` | Open P2P example only |
| `cli` | `jsonschema>=4.21` | `kur-validate --json-schema` |
| `pixel-shift` | `opencv-python>=4.9` | operator + transform `aintropy/pixel-shift` |
| `action-state` | `opencv-python>=4.9` | operator + analyzer `aintropy/action-state-consistency` |
| `collision` | `trimesh>=4`, `rtree>=1.0` | operator/transform `aintropy/collision`; operator/analyzer `aintropy/collision-directional`; adapters `aintropy/collision-asg`, `aintropy/collision-teapearce` |
| `slam` | `av>=12`, `torch>=2.13`, `torchvision>=0.28`, `huggingface_hub>=0.20`, `omegaconf>=2.3`, `addict>=2.4`, `einops>=0.7`, `opencv-python>=4.9`, `imageio>=2.31`, `tqdm>=4.66.3`, `pillow>=12.3`, `safetensors>=0.4`, `faiss-cpu>=1.8` | transforms `aintropy/slam-motion`, `aintropy/slam-quality`; operators `aintropy/slam-motion-diversity`, `aintropy/slam-quality`; adapter `aintropy/slam-trajectory` |
| `slam-viz` | `trimesh>=4` | SLAM notebook viewer |
| `gameplay` | union of `[slam]` + `[collision]` (+ friends) | superset covering every built-in gameplay component |
| `standard-operators` | alias of `[gameplay]` | (same as `gameplay`) |
| `all` | union of every extra above | everything |
| `dev` / `docs` | dev tooling | not user-facing |
