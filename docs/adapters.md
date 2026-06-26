# Input adapters — raw source → canonical ticks

An `InputAdapter` is the *only* place a raw format's knowledge enters the SDK. Its contract is
deliberately tiny: turn a source into an ordered stream of canonical `Tick`s. Everything
downstream — buffer building, segmentation, metrics, selection, output — is generic and never
knows which dataset it's looking at.

```python
class InputAdapter(Protocol):
    schema: SchemaSpec
    def ticks(self, source_uri: str) -> Iterable[Tick]: ...
```

## The canonical `Tick`

```python
Tick(time, entities, events={}, globals={})
#   entities: {entity_id -> {field_name -> value}}   value: number | (components...) | str
#   events:   {stream -> [ {field -> value}, ... ]}
#   globals:  {name -> scalar}
```

Building the `(T,E,D)` `WindowBuffer` from a window of ticks (`build_window(schema, ticks)`)
is generic SDK code — so an adapter only does parse + group-by-timestep, nothing numeric.

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

Rows must be ordered by time (timesteps are runs of equal `time_column` values). Numeric
fields are coerced to float; categorical to str; vectors assembled from their listed columns.

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
derived schema; pass an explicit `schema=` to override. Containers: npz / json.

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
