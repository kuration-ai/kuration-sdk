# Registered-component inventory

A point-in-time listing of every component registered in `src/kuration/`. Extras are the SDK's
`[project.optional-dependencies]` names from `pyproject.toml`. "Runtime imports" lists top-level
heavy imports that would fail at *execution* if the extra isn't installed (all guarded by lazy
`import`s inside functions: descriptors still register cleanly).

Every pipeline-step component — what used to be documented as separate "Methods", "Filters",
"Scorers" — is now a single **Operator**, callable as `pipeline.<name>(...)` exactly like any
other named operator (see [operators.md](operators.md) for the taxonomy and
[strategies.md](strategies.md) for how the generic `.curate()`/`.filter()`/`.score()` escape
hatches these operators wrap actually work underneath). `register_method`/`register_filter`/
`register_scorer` remain real, distinct registration APIs one level below `register_operator`
(building a genuinely new selection algorithm or pre-pass filter still means implementing one of
those contracts, not literally writing a `Pipeline` combinator) — but as vocabulary a user
browsing capabilities needs to learn, there is only "operator."

This is a snapshot, not a generated artifact — nothing currently regenerates it
automatically when `src/kuration/` changes, so treat it as accurate as of its last
edit rather than always-current.

## Operators

Category (Selection / Annotation / Aggregation) follows the taxonomy in
[operators.md](operators.md#5-built-in-operator-catalog); no built-in operator is Aggregation
today (a confirmed gap, not an oversight). "Backed by" names the registry kind actually doing the
work underneath the operator, for anyone extending the SDK rather than just using it.

| Name | Category | Backed by | Registered in | One-sentence purpose | Required params | Optional params (default) | Descriptor `extra` / `install_hint` | Runtime imports needed |
|---|---|---|---|---|---|---|---|---|
| `aintropy/action-state-consistency` | Annotation | analyzer (dataset mode) | `src/kuration/gameplay/action_state_consistency/__init__.py` (`attach_action_state_consistency`) | Dataset-mode only: audit an already-written dataset's shards for action-state consistency and attach the per-clip verdict to pipeline metadata. For a live pipeline use `action-residual`-style scorer access instead (`.score("aintropy/action-state-consistency", ...)`; no operator wrapper yet). | (`baselines` OR `controls`) validated by handler | `anomaly_type: str = "action_state"`, `calibration: object`, `boundary: "exclude"|"error"|"assume-inactive" = "exclude"`, `frames: str`, `actions: str`, `color_space: str`, `verify: bool`, `metadata_key: str = "action_state_consistency"`, `replace: bool = False` | `extra="action-state"`, `install_hint='pip install "kuration-sdk[action-state]"'` | opencv-python (`cv2`, lazy in `diamond_action_decode._prep_gray`) |
| `aintropy/collision` | Annotation | transform | `src/kuration/gameplay/collision/__init__.py` (`collision_mechanics`) | Compose the CS2 collision-detection transform into a pipeline. | `map_for_source: object`, `map_registry: object`, `dist_path: str` | `disk_cache_dir: str = ".kuration_collision_cache"`, `min_score: number = 2.0`, `proximity: number = 40.0`, `min_approach: number = 250.0`, `max_residual: number = 20.0`, `slide_angle: number = 75.0`, `max_slide_speed: number = 400.0`, `min_release_accel: number = 100.0`, `max_release_vn: number = 100.0` | `extra="collision"`, `install_hint='pip install "kuration-sdk[collision]"'` | trimesh + rtree (lazy in `mesh_cache.load_mesh_cached`) |
| `aintropy/collision-directional` | Annotation | analyzer | `src/kuration/gameplay/collision/__init__.py` (`attach_directional_collision`) | Run directional collision detection over the tick stream and attach the verdict to pipeline metadata. | `map_registry: object`, `ramp_models_path: str` | `map_for_source: object`, `default_map: str`, `fallback_ramp: [number,number]`, `disk_cache_dir: str`, `proximity: number`, `min_missing_speed: number`, `concentration_ratio: number (0..1)`, `batch_size: int`, `inputs: str[]`, `metadata_key: str = "directional_collision"`, `replace: bool = False` | `extra="collision"`, `install_hint='pip install "kuration-sdk[collision]"'` | trimesh + rtree (lazy) |
| `aintropy/pixel-shift` | Annotation | transform | `src/kuration/gameplay/pixel_shift/__init__.py` (`pixel_shift`) | Measure pixel shift for every isolated key/mouse press. | (none) | `controls: str[]` (default: all 8 controls W/S/A/D/MOUSE_X_POS/…/MOUSE_Y_NEG), `boundary: "exclude"|"error"|"assume-inactive" = "exclude"` | `extra="pixel-shift"`, `install_hint='pip install "kuration-sdk[pixel-shift]"'` | opencv-python (`cv2`, lazy) |
| `aintropy/slam-motion-diversity` | Annotation | transform | `src/kuration/gameplay/slam/__init__.py` (`slam_motion_diversity`) | Add SLAM motion features and quantile-bin a selected diversity axis. | (none) | `axis: enum(slam_path_length|slam_displacement|slam_motion_efficiency|slam_mean_step|slam_max_step|slam_turn_degrees|slam_mean_turn_degrees|slam_max_turn_degrees|slam_spatial_extent) = "slam_path_length"`, `bins: int ≥ 2 = 8` | `extra="slam"`, `install_hint='pip install "kuration-sdk[slam]"'` | numpy only for this operator; SLAM extra pulls torch/torchvision/faiss-cpu/av/opencv/etc. needed by sibling reconstruction |
| `aintropy/slam-quality` | Annotation | transform | `src/kuration/gameplay/slam/__init__.py` (`slam_quality`) | Add SLAM reconstruction-quality signals and optionally filter windows. | (none) | `min_tracking_confidence: number [0..1] = 0.5`, `translation_ratio: number > 0 = 10.0`, `translation_mad_multiplier: number > 0 = 8.0`, `max_translation_step: number > 0 = None`, `max_turn_degrees: number (0..180] = 90.0`, `min_score: number [0..1] = None` | `extra="slam"`, `install_hint='pip install "kuration-sdk[slam]"'` | numpy only for this operator |
| `kuration/balanced-selector` | Selection | recipe (`.measure`/`.bin`/`.sample`) | `src/kuration/selection/operators.py` (`balanced_selector`) | Sample equally across quantile bins of a measured feature. | `feature: str`, `of: str` | `kind: str = "std"`, `bins: int ≥ 2 = 8`, `count: int ≥ 1 = 100` | none | numpy only |
| `kuration/rare-event-coverage` | Selection | recipe | `src/kuration/selection/operators.py` (`rare_event_coverage`) | Weighted-sample toward windows containing a rare event. | `of: str` | `feature: str = "event_rate"`, `count: int ≥ 1 = 100`, `weight: number > 0 = 1.0` | none | numpy only |
| `kuration/diverse-subset` | Selection | recipe | `src/kuration/selection/operators.py` (`diverse_subset`) | Cluster-balanced selection over pre-measured features. | `features: str[] (non-empty)` | `count: int ≥ 1 = 100`, `n_clusters: int ≥ 2 = 32`, `dedup_threshold: number [-1..1] = None` | `extra="methods"`, `install_hint='pip install "kuration-sdk[methods]"'` | scikit-learn via `.sample("cluster-balanced", ...)` |
| `top-k` | Selection | method (base) | `src/kuration/core/builtin_operators.py` (`top_k`) | Rank by a weighted score and take the highest priorities. | `weights: object` | `normalize`, `n`/`hours`/`test_hours`, `group_cap` | none | numpy only |
| `weighted` | Selection | method (base) | `src/kuration/core/builtin_operators.py` (`weighted`) | Sample without replacement, proportional to a weighted score (Gumbel-top-k). | `weights: object` | `normalize`, `n`/`hours`/`test_hours`, `group_cap` | none | numpy only |
| `uniform-bins` | Selection | method (base) | `src/kuration/core/builtin_operators.py` (`uniform_bins`) | Round-robin declared `.bin(...)` axes for coverage-first selection. Requires `.bin(...)` axes already declared. | (none) | `n`/`hours`/`test_hours`, `group_cap` | none | numpy only |
| `target-distribution` | Selection | method (base) | `src/kuration/core/builtin_operators.py` (`target_distribution`) | Match the pool to an explicit target distribution over a scalar feature. Requires `.bin(...)` axes already declared. | (none) | `target: "flat"\|{metric}\|{anchor_metric,anchor_quantile}`, `n`/`hours`/`test_hours`, `group_cap` | none | numpy only |
| `cluster-balanced` | Selection | method (module) | `src/kuration/core/builtin_operators.py` (`cluster_balanced`) | Joint k-means-cluster round-robin coverage over the feature space (DINOv2-style balanced subsampling). | `feature_space: str[]` | `clusters: int = 2048`, `feature_normalize: str = "zscore"`, `n`/`hours`/`test_hours`, `group_cap` | `extra="methods"` | scikit-learn (`KMeans` / `MiniBatchKMeans`, lazy) |
| `k-center` | Selection | method (module) | `src/kuration/core/builtin_operators.py` (`k_center`) | Farthest-point traversal of the cluster centroids ordered by coverage. | `feature_space: str[]` | same as `cluster-balanced` | `extra="methods"` | scikit-learn |
| `facility-location` | Selection | method (module) | `src/kuration/core/builtin_operators.py` (`facility_location`) | Submodular facility-location cluster ordering (greedy coverage gain). | `feature_space: str[]` | same as `cluster-balanced` | `extra="methods"` | scikit-learn |
| `dsir` | Selection | method (module) | `src/kuration/core/builtin_operators.py` (`dsir`) | Multivariate importance resampling with Gumbel-top-k over a quantized joint feature space. | `feature_space: str[]` | `dsir_bins: int = 8`, `target: "flat"\|{metric}\|{anchor_metric,anchor_quantile}`, `feature_normalize: str = "quantile"`, `n`/`hours`/`test_hours`, `group_cap` | `extra="methods"` | numpy only |
| `rare` | Selection | method (module) | `src/kuration/core/builtin_operators.py` (`rare`) | Density-inverse kNN atypicality (sparse = admitted first). | `feature_space: str[]` | `knn_k: int = 10`, `rare_sample: int = 20000`, `feature_normalize: str = "zscore"`, `n`/`hours`/`test_hours`, `group_cap` | `extra="methods"` | scikit-learn (`NearestNeighbors`) |
| `typical` | Selection | method (module) | `src/kuration/core/builtin_operators.py` (`typical`) | Prototypicality: the negative of `rare` (densest first). | `feature_space: str[]` | same as `rare` | `extra="methods"` | scikit-learn |
| `anomaly` | Annotation | filter | `src/kuration/core/builtin_operators.py` (`anomaly`) | Drop multivariate outliers with an Isolation Forest, before selection runs. | (none) | `contamination: number = 0.02`, `normalize: str = "zscore"` | `extra="methods"` | scikit-learn (`IsolationForest`, lazy) |
| `dedup` | Annotation | filter | `src/kuration/core/builtin_operators.py` (`dedup`) | Feature-space SemDeDup: within each cluster keep one representative per near-duplicate group, before selection runs. | `feature_space: str[]` | `threshold: number = 0.98`, `n_clusters: int = 2048`, `normalize: str = "zscore"` | `extra="methods"` | scikit-learn |
| `action-residual` | Annotation | scorer | `src/kuration/core/builtin_operators.py` (`action_residual`) | Action-conditioned k-nearest-neighbor predictive residual (whole-pool, buffer-using). | `obs: str`, `action: str` | `k: int = 8`, `out: str = "action_residual"` | none on descriptor (runtime → `[methods]`) | scikit-learn (`KNeighborsRegressor`, lazy) |

`top-k`/`weighted`/`uniform-bins`/`target-distribution` ("method (base)" above) and
`cluster-balanced`/`k-center`/`facility-location`/`dsir`/`rare`/`typical` ("method (module)")
translate a handful of their operator kwargs before forwarding to the underlying method
(`clusters`→internal `n_clusters`, `feature_normalize`→internal `normalize`, `dsir_bins`→internal
`n_bins`, `knn_k`→internal `k`, `rare_sample`→internal `sample`) — confirmed empirically against
`strategy.py`'s `_resolve`, not merely read off the internal function signatures, since the two
differ. `anomaly`/`dedup`/`action-residual` keep their parameter names verbatim; there is no
translation layer for filters or scorers.

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

## Dataset audits without an operator yet

Everything else stateful is reachable as a named operator (see the Operators table's "Backed
by" column); `kuration/split-leakage` is the one exception — a `check_dataset`-only analyzer
with no operator wrapper, reachable via `.analyze("kuration/split-leakage", ...)`/`.enrich(...)`.

| Name | Registered in | Purpose | Required params | Optional params | `extra` | Runtime imports |
|---|---|---|---|---|---|---|
| `kuration/split-leakage` | `src/kuration/core/analyzers.py` (`_split_leakage_analyzer`) | Audit source and (optionally) group assignments across dataset splits for train/test leakage. | (none) | `check_groups: bool = True` | none | stdlib only |

## Containers

Registered via a loop in `src/kuration/core/output/containers.py` with `capabilities={"remote_paths": True, "streaming": True}`, no `extra` on descriptor (runtime error still guides).

| Name | Extension | On-disk purpose | Deps needed at runtime |
|---|---|---|---|
| `hdf5` | `.h5` | Compressed HDF5 shard (`n_steps` attr in-band). | h5py (`[array]`) |
| `json` | `.json` | Dependency-free JSON dump of arrays + `_n_steps`. | stdlib |
| `npz` | `.npz` | Dependency-free NumPy `.npz` archive of arrays + `_n_steps`. | numpy |
| `parquet` | `.parquet` | Flat Parquet shard (scalar/tensor columns) with `_n_steps` in schema metadata. | pyarrow (`[parquet]`) |

## Presets

| Name | Registered in | Purpose | Params (defaults) | `extra` | Runtime imports |
|---|---|---|---|---|---|
| `numpy-timeseries` | `src/kuration/core/output/presets.py` (`_numpy`) | NumPy NPZ time-series shards. | `container: str = "npz"`, `shard_length: int ≥ 1 = 64`, `short_policy: enum{keep,pad,drop} = "keep"`, `categorical: enum{error,omit,integer-vocab,encoder}`, `categorical_encoders: object`, `categorical_vocabs: object`, `presence_mask: bool` | none | numpy |
| `pytorch-sequences` | `src/kuration/core/output/presets.py` (`_pytorch`) | Padded sequence shards for PyTorch (default `short_policy="pad"`). | as above with `short_policy = "pad"` | none | numpy (Torch only in consumer `ShardDataset`) |
| `tabular` | `src/kuration/core/output/presets.py` (`_tabular`) | Flat Parquet time-series shards (`container="parquet"`, `shard_length=1024`, `short_policy="keep"`). | parquet defaults | `extra="parquet"` | pyarrow |

## Uncertain / could-not-confirm findings

- **Adapter descriptors have no JSON-Schema `parameters` on the register call** (`register_adapter("hdf5", HDF5Adapter)` etc. pass just the class). "Required / optional" reported above comes from the class `__init__` signature; `describe_component("adapter", name)` will return the empty `{"type":"object","properties":{}, "additionalProperties":False}` default, not the class kwargs.
- **`aintropy/collision-asg` and `aintropy/collision-teapearce` adapter descriptors** are explicitly registered with an empty schema: the descriptor accepts no kwargs. Init also takes no configuration (fully driven by schema constant + source URI).
- **The underlying `"method"`/`"filter"`/`"scorer"` registrations still exist and still carry
  their own (often sparser) descriptors** — base methods register with only
  `capabilities={"streaming": True}` and no `description`/`parameters`/`extra`; feature-space
  methods and filters register with `capabilities={"whole_pool": True}` and `extra="methods"`
  but no `description`/`parameters` either. The Operators table above documents the operator
  wrapper's descriptor (`src/kuration/core/builtin_operators.py`), which does carry a
  real description and JSON-Schema parameters — that is now the canonical, discoverable form;
  the underlying kind's sparser descriptor is an implementation detail.
- **`pytorch-sequences` preset**: preset is `[parquet]`-free (writes NPZ by default), no explicit `extra=` on register call. Overrides to `container="parquet"` still hit pyarrow at runtime.
- **`json` and `npz` containers**: registered without `extra`; truly no non-stdlib deps.
- **SLAM transforms**: own `transform_batch` code is numpy-only, but `[slam]` extra pulls torch/torchvision/faiss/av/opencv because sibling `SLAMReconstructor` needs them for upstream reconstruction that produces the manifests these transforms consume.
- **Base adapters** (`hdf5`, `huggingface`, `jsonl`, `npz`, `parquet-dataset`, `timeseries`, `zarr`, `curated`): no `install_hint` on descriptor. Errors surface only at first use.

## Extras → deps mapping (from `pyproject.toml`)

| Extra | Packages | Components that reference it |
|---|---|---|
| `methods` | `scikit-learn>=1.5` | operators `cluster-balanced` / `k-center` / `facility-location` / `dsir` / `rare` / `typical` / `anomaly` / `dedup` / `action-residual` |
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
| `action-state` | `opencv-python>=4.9` | operator + analyzer (dataset mode) + scorer (live pipeline) `aintropy/action-state-consistency` |
| `collision` | `trimesh>=4`, `rtree>=1.0` | operator/transform `aintropy/collision`; operator/analyzer `aintropy/collision-directional`; adapters `aintropy/collision-asg`, `aintropy/collision-teapearce` |
| `slam` | `av>=12`, `torch>=2.13`, `torchvision>=0.28`, `huggingface_hub>=0.20`, `omegaconf>=2.3`, `addict>=2.4`, `einops>=0.7`, `opencv-python>=4.9`, `imageio>=2.31`, `tqdm>=4.66.3`, `pillow>=12.3`, `safetensors>=0.4`, `faiss-cpu>=1.8` | transforms `aintropy/slam-motion`, `aintropy/slam-quality`; operators `aintropy/slam-motion-diversity`, `aintropy/slam-quality`; adapter `aintropy/slam-trajectory` |
| `slam-viz` | `trimesh>=4` | SLAM notebook viewer |
| `gameplay` | union of `[slam]` + `[collision]` (+ friends) | superset covering every built-in gameplay component |
| `standard-operators` | alias of `[gameplay]` | (same as `gameplay`) |
| `all` | union of every extra above | everything |
| `dev` / `docs` | dev tooling | not user-facing |
