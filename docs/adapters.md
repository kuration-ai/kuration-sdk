# Input adapters — raw source → canonical ticks

An `InputAdapter` is the ordinary-record format seam; `MediaProvider` is the auxiliary-media
format seam. The adapter contract is deliberately tiny: turn a source into an ordered stream
of canonical `Tick`s. Everything
downstream — buffer building, segmentation, metrics, selection, output — is generic and never
knows which dataset it's looking at.

```python
class InputAdapter(Protocol):
    schema: SchemaSpec
    def ticks(self, source_uri: str) -> Iterable[Tick]: ...
```

Adapters may also participate in `kur-import` auto-detection:

```python
class CaptureAdapter:
    @staticmethod
    def detect_source(uri):
        return 100 if uri.endswith(".capture") else 0
```

`detect_source()` returns false/zero for no match, or a positive confidence. The uniquely
highest-confidence registered adapter wins; ties fail and ask for `--adapter NAME`. Tensor
adapters can additionally provide a static
`import_arrays(uri) -> dict[str, numpy.ndarray]` hook to preserve raw modality ranks and dtypes.
Adapters without that hook run through their normal `ticks()` contract and require either safe
schema inference or an explicit schema/config.

## The canonical `Tick`

```python
Tick(time, entities, events={}, globals={}, media={})
#   entities: {entity_id -> {field_name -> value}}   value: number | (components...) | str
#   events:   {stream -> [ {field -> value}, ... ]}
#   globals:  {name -> scalar}
#   media:    {column -> lazy media reference}; excluded from WindowBuffer/metric DSL
```

Building the `(T,E,D)` `WindowBuffer` from a window of ticks (`build_window(schema, ticks)`)
is generic SDK code — so an adapter only does parse + group-by-timestep, nothing numeric.

Reading one back (e.g. while debugging a custom adapter, or inside code that walks ticks
directly instead of through a `WindowBuffer`) is plain dict access:

```python
value = tick.entities[entity_id][field_name]   # a single entity's field at this timestep
one_entity = tick.entities[entity_id]           # that entity's whole {field_name: value} dict
global_value = tick.globals.get("battery_pct")  # entity-independent scalar, may be absent
```

A missing field for an entity present at this tick is simply absent from its dict (`.get(...)`
or a `KeyError` on indexing); `build_window` treats that the same as the entity being absent
from the tick entirely.

## Video and synchronized media

`input("clips/*.mp4")` recognizes MP4/M4V/MOV sources and constructs a `VideoAdapter` using
the optional `[video]` PyAV provider. One video is one source; each presentation timestamp yields
a tick with an empty ordinary entity and lazy `media["frame"]: FrameRef`. Probing establishes a
`MediaSpec` without decoding the full recording, and ordinary metrics never decode or copy frames.

```python
from kuration import input

video = input("clips/*.mp4").window(64)
joined = input("telemetry/*.parquet", time="timestamp").join_video(
    lambda source: source.replace(".parquet", ".mp4"),
    on="timestamp", tolerance=0.020,
)
```

`.join_video()` also accepts an explicit `{source_id: video_uri}` mapping. Frame-index joins are
exact; timestamp joins select the nearest presentation timestamp within the explicit tolerance.
The default mismatch policy is `error`; `drop` must be requested explicitly. Implement another
decoder by satisfying `MediaProvider.probe`, `iter_refs`, and `decode_batch`; providers declare
capabilities and return `(B,H,W,C)` arrays in bounded batches.

## The generic timeseries adapter (no per-dataset code)

A CSV / Parquet table (or a list of row dicts) where rows sharing a `time_column` value form
one timestep and an optional `entity_column` splits the timestep into entities:

```python
from kuration import TimeseriesAdapter

adapter = TimeseriesAdapter(
    schema,
    fields={"pos": ["x", "y", "z"], "battery": "batt"},   # vector -> [cols]; scalar/categorical -> col
    time_column="t",          # omit -> each row is its own timestep
    entity_column="drone_id", # omit -> one entity per timestep
    globals={"alive": "n_alive"},   # optional per-tick scalar columns
)

ticks = adapter.ticks("flight.csv")     # or adapter.ticks("flight.parquet")  ([parquet] extra)
ticks = adapter.from_rows(list_of_dicts)
```

Rows must be ordered by time (timesteps are runs of equal `time_column` values) — **enforced**:
a time value reappearing after its timestep closed (e.g. rows sorted entity-major) raises, as
does a duplicate `(time, entity)` row (the later row would otherwise silently overwrite the
earlier one). Sort by time first. Numeric fields are coerced to float; categorical to str;
vectors assembled from their listed columns.

The adapter also implements the optional **`check_source(uri)`** preflight hook:
`kur-validate --check-sources` calls it per source to sniff the CSV header (or Parquet file
schema) and verify every mapped column — time, entity, field, and global columns — exists,
without reading any data rows. A column typo therefore fails at preflight instead of
surfacing as every source skipped at run time.

## Array files: HDF5 and NPZ

One file = one source, one entity; each mapped key holds `(T, *shape)` (scalar → float,
vector → tuple, categorical → str). `HDF5Adapter` needs the `[array]` extra (h5py); `NpzAdapter`
is dependency-free (numpy `.npz`).

```python
from kuration import HDF5Adapter, NpzAdapter
HDF5Adapter(schema, fields={"pos": "xyz_dataset", "battery": "batt_dataset"}).ticks("flight.h5")
NpzAdapter(schema, fields={"pos": "xyz", "battery": "batt"}).ticks("flight.npz")
```

## JSON Lines

`JsonlAdapter` is the timeseries adapter over a `.jsonl` file — one JSON object per line is a
row, with the same `fields` / `time_column` / `entity_column` mapping (stdlib only).

```python
from kuration import JsonlAdapter
JsonlAdapter(schema, fields={"pos": ["x", "y"], "act": "act"}, time_column="t").ticks("log.jsonl")
```

## Partitioned Parquet (a dataset of parts)

`ParquetDatasetAdapter` generalizes the single-file Parquet read to a directory or glob of
parts (e.g. Hive-style `date=*/part-*.parquet`), same column mapping. `source_per` decides the
unit: `"part"` (default) makes **each file its own source** — the natural choice when each file
is one trajectory/episode, and it keeps the per-source train/test split leakage-free; `"dataset"`
treats **all parts as one source** (rows concatenated, sorted by `time_column` if set). In
`"dataset"` mode, parts that overlap in time need an `entity_column` distinguishing their rows —
same-time rows for the same entity raise rather than silently last-write-win. Reuses the
`[parquet]` extra — no new dependency. Remote globs (`gs://…/**/*.parquet`) use the matching
provider extra (`[gcs]`, `[s3]`, or `[azure]`), or `[cloud]` when the environment supplies its own
fsspec backend.

```python
from kuration import ParquetDatasetAdapter
ad = ParquetDatasetAdapter(schema, path="gs://bucket/logs", source_per="part",
                           fields={"pos": ["x", "y"], "act": "act"}, time_column="t")
ad.sources()            # -> every part file (or [path] for source_per="dataset")
```

## HuggingFace datasets

`HuggingFaceAdapter` loads a HF dataset/split (hub or local) as ticks, same mapping. One split is
one source by default; set `source_column` to split it into **one source per distinct value**
(e.g. per `episode_id`). Needs the `[hf]` extra (`datasets`).
Single-source mode supports `streaming=True`. Partition discovery through `source_column` requires
a whole-split index and therefore rejects bounded streaming; use `streaming=False` or expose each
episode as an independently readable source.

```python
from kuration import HuggingFaceAdapter
ad = HuggingFaceAdapter(schema, dataset="org/name", split="train",
                        fields={"pos": ["x", "y"], "act": "act"},
                        time_column="t", source_column="episode_id")
```

(Vector fields map one column per dim, as elsewhere — `pos: ["x", "y"]`, not a single
list-valued column.)

## Zarr array store

`ZarrAdapter` mirrors HDF5/NPZ for chunked Zarr stores: one store = one source, one entity, each
mapped array `(T, *shape)`. Needs the `[zarr]` extra; remote stores (`gs://`, `s3://`, …) open via
fsspec (the matching `[gcs]`, `[s3]`, or `[azure]` extra; `[cloud]` is fsspec-only).

```python
from kuration import ZarrAdapter
ZarrAdapter(schema, fields={"pos": "px", "act": "a"}).ticks("s3://bucket/run.zarr")
```

## Curated datasets as a source (composition & introspection)

`CuratedDatasetAdapter` reads a dataset written by `write_dataset` back in — closing the loop
so curated sets are composable. Each shard becomes a source (segmentation looks *within* a
clip) and each output modality becomes a schema field, so the DSL runs over the modalities (a
schema is derived from the dataset's `config.json`, or pass your own).

```python
from kuration import CuratedDatasetAdapter, assign_splits

adapter = CuratedDatasetAdapter(["datasetA/", "datasetB/"])   # one or many dataset dirs
sources = adapter.sources()                                    # every shard, across datasets

# random 25/75 split across the two datasets, stratified by which dataset each shard came from
split = assign_splits({s: adapter.dataset_of[s] for s in sources}, test_frac=0.25, seed=0)

# or: find clips matching DSL criteria within them (see strategies.md `filter` method)
```

Higher-rank tensor modalities (e.g. images) aren't DSL-addressable and are skipped from the
derived schema; pass an explicit `schema=` to override. All four containers
(npz/json/hdf5/parquet) re-ingest. A dataset directory that has been **moved or copied** still
loads (shard paths are rebased onto the dir you pass when the recorded ones no longer exist).
With multiple datasets, each must carry the modalities the schema requires — a missing
modality raises at construction instead of silently yielding NaN features; pass an explicit
narrower `schema=` to mix heterogeneous datasets.

## Writing your own adapter

For a bespoke format (a ROS bag, a proprietary log, a nested JSON), implement `ticks`:

```python
class MyAdapter:
    def __init__(self, schema):
        self.schema = schema
    def ticks(self, source_uri):
        for raw_step in read_my_format(source_uri):
            yield Tick(time=raw_step.t,
                       entities={raw_step.id: {"pos": raw_step.xyz, "battery": raw_step.batt}})
```

Register it for reconstruction in worker processes (the parallel feature pass rebuilds
adapters by name rather than pickling them):

```python
from kuration import register_adapter, get_adapter
register_adapter("mine", MyAdapter)
adapter = get_adapter("mine", schema=schema)
```

Optionally implement **`check_source(uri)`** — a cheap metadata-only preflight (header,
dataset keys, array shapes; never data rows) that raises if the source can't serve the
adapter's mapping. `kur-validate --check-sources` calls it when present, so config mistakes
fail before a run instead of as skipped sources during one. Every built-in adapter implements
it: CSV/JSONL sniff the header/first row; Parquet reads the footer schema; HDF5/NPZ/Zarr check
mapped keys, step-count agreement, and per-step widths against the schema from store metadata;
HuggingFace checks the builder's declared features (no download); curated checks manifest
membership + shard existence. If your adapter's sources aren't filesystem paths (like the HF
dataset names), also set `sources_are_paths = False` so `--check-sources` skips the existence
stat and relies on your `check_source`.

## Segmentation

A `Segmenter` slices the tick stream into overlapping windows:

```python
from kuration import Segmenter
Segmenter(64, 32)                                   # 64-step windows, 32-step stride
Segmenter(4, 2, unit="seconds", sample_hz=20)       # 4s windows, 2s stride (needs sample_hz)
segments = Segmenter(64, 32).segments("flight1", ticks)
```

Each `Segment` carries `source_id`, the tick-index range, and generic `features` / `bin` /
`tags` / `group` filled by later stages.
