"""Read side: kuration.core.loaders (numpy) + kuration.core.torch.ShardDataset (the [torch] extra)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from kuration import (
    OutputSpec,
    SchemaSpec,
    Segmenter,
    Tick,
    iter_shards,
    load_strategy,
    read_shard_arrays,
    shard_entries,
    write_dataset,
)

SCHEMA = SchemaSpec.from_dict({
    "name": "d", "entity_axis": "e", "sample_hz": 1.0,
    "fields": {"pos": {"kind": "vector", "dims": ["x", "y"]}, "act": {"kind": "scalar"}},
})


def test_loader_does_not_hide_future_config_version(tmp_path):
    from kuration.core.versioning import ARTIFACT_FORMAT_VERSION

    (tmp_path / "dataset_manifest.json").write_text(json.dumps({
        "artifact_format_version": ARTIFACT_FORMAT_VERSION, "shards": []}))
    (tmp_path / "config.json").write_text(json.dumps({
        "artifact_format_version": "999.0", "output": {"container": "json"}}))
    with pytest.raises(ValueError, match="unsupported future artifact"):
        shard_entries(str(tmp_path))


def test_loader_does_not_hide_malformed_config(tmp_path):
    from kuration.core.versioning import ARTIFACT_FORMAT_VERSION

    (tmp_path / "dataset_manifest.json").write_text(json.dumps({
        "artifact_format_version": ARTIFACT_FORMAT_VERSION, "shards": []}))
    (tmp_path / "config.json").write_text("{")
    with pytest.raises(ValueError, match="not valid JSON"):
        shard_entries(str(tmp_path))


class MemAdapter:
    def __init__(self, schema, data):
        self.schema, self.data = schema, data

    def ticks(self, src):
        return self.data[src]


def _write(tmp_path, *, container="npz", short_policy="keep", shard_length=4, train_hours=1.0):
    # one entity, 12-step tracks; act increases with L so top-k admits the high-act sources
    data = {f"s{L}": [Tick(time=i, entities={0: {"pos": (float(i), float(L)), "act": float(L)}})
                      for i in range(12)] for L in range(1, 5)}
    strat = load_strategy({"name": "hi", "define": {"am": {"source": "e.act", "reduce": "mean"}},
                           "selection": {"method": "top-k", "weights": {"am": 1.0}}}, SCHEMA)
    spec = OutputSpec.from_dict({"shard_length": shard_length, "container": container,
                                 "short_policy": short_policy,
                                 "modalities": {"obs": {"dtype": "float32", "shape": [2],
                                                        "source": "field:pos"},
                                                "act": {"dtype": "float32", "source": "field:act"}}})
    out = str(tmp_path)
    write_dataset(out, schema=SCHEMA, adapter=MemAdapter(SCHEMA, data), sources=list(data),
                  strategy=strat, segmenter=Segmenter(4, 4), spec=spec,
                  train_hours=train_hours, test_hours=0, seed=0)
    return out


@pytest.mark.parametrize("container", ["npz", "json", "parquet"])
def test_roundtrip_shapes_and_values(tmp_path, container):
    out = _write(tmp_path, container=container)
    shards = list(iter_shards(out))
    assert shards, "no shards read back"
    for arr in shards:
        assert set(arr) == {"obs", "act", "_present"}
        assert arr["_present"].dtype == np.bool_ and arr["_present"].all()
        assert arr["obs"].ndim == 2 and arr["obs"].shape[1] == 2   # (T, 2) — reshape restored
        assert arr["act"].ndim == 1                                # scalar modality -> (T,)
        # obs = field:pos = (step, L): x-column steps consecutively, y-column is constant L
        x = arr["obs"][:, 0]
        assert np.allclose(np.diff(x), 1) if x.size > 1 else True
        assert np.allclose(arr["obs"][:, 1], arr["obs"][0, 1])     # constant source id L
        assert np.allclose(arr["act"], arr["obs"][0, 1])           # act == L too


def test_split_filter_and_meta(tmp_path):
    out = _write(tmp_path)
    entries, container = shard_entries(out, split="train")
    assert entries and all(e["split"] == "train" for e in entries)
    assert shard_entries(out, split="test")[0] == []          # test_hours=0 -> no test shards
    arr, entry = next(iter_shards(out, split="train", with_meta=True))
    assert "source_id" in entry and entry["split"] == "train"


def test_modality_subset(tmp_path):
    out = _write(tmp_path)
    arr = next(iter_shards(out, modalities=["act"]))
    assert set(arr) == {"act"}


def test_trim_drops_padding(tmp_path):
    # 12-step track, shard_length 5, pad -> last shard has 2 real rows padded to 5
    out = _write(tmp_path, short_policy="pad", shard_length=5)
    entries, container = shard_entries(out)
    padded = [e for e in entries if e["padded"]]
    assert padded, "expected a padded tail shard"
    e = padded[0]
    trimmed = read_shard_arrays(e, container, trim=True)["obs"]
    full = read_shard_arrays(e, container, trim=False)["obs"]
    assert trimmed.shape[0] == e["n_steps"] < full.shape[0] == 5


def test_resolve_by_dir_jsonpath_and_dict(tmp_path):
    out = _write(tmp_path)
    import json
    man = json.loads((tmp_path / "dataset_manifest.json").read_text())
    by_dir = list(iter_shards(out))
    by_json = list(iter_shards(f"{out}/dataset_manifest.json"))
    with pytest.raises(ValueError, match="pass the dataset directory"):
        list(iter_shards(man, container="npz"))
    assert len(by_dir) == len(by_json) > 0


def test_read_shard_reshapes_flattened_parquet_tensor(tmp_path):
    # The critical Parquet path: a rank-2 modality is stored flat (L, 4) and must come back
    # (L, 2, 2). Exercises read_shard_arrays' reshape directly (write_dataset's modalities are
    # only rank<=1, so this guards the encoder/image case).
    pytest.importorskip("pyarrow")
    from kuration.core.output.containers import write_shard
    L = 6
    img = np.arange(L * 2 * 2, dtype="float32").reshape(L, 2, 2)
    path = write_shard(str(tmp_path / "s"), "parquet", {"img": img}, n_steps=L)
    entry = {"path": path, "shape": {"img": [2, 2]}, "n_steps": L}
    back = read_shard_arrays(entry, "parquet", verify=False)["img"]
    assert back.shape == (L, 2, 2) and np.allclose(back, img)


def test_torch_shard_dataset(tmp_path):
    torch = pytest.importorskip("torch")
    from kuration.core.torch import ShardDataset
    out = _write(tmp_path, short_policy="pad", shard_length=4)
    ds = ShardDataset(out, split="train", include_meta=True)
    assert len(ds) > 0
    item = ds[0]
    assert isinstance(item["obs"], torch.Tensor) and item["obs"].shape[1] == 2
    assert item["_meta"]["split"] == "train"


def test_torch_shard_dataset_uses_isolated_custom_container_registry(tmp_path):
    torch = pytest.importorskip("torch")
    from kuration.core.checkpoints import canonical_hash
    from kuration.core.io import write_text
    from kuration.core.manifest import commit_dataset, portable_shard_entries
    from kuration.core.output.containers import Container
    from kuration.core.output.shardwriter import write_track
    from kuration.core.output.track import Track
    from kuration.core.registry import Registry, default_registry
    from kuration.core.torch import ShardDataset
    from kuration.core.versioning import ARTIFACT_FORMAT_VERSION

    def writer(path, arrays, n_steps):
        with open(path, "wb") as stream:
            np.savez(stream, _n_steps=n_steps, **arrays)

    def reader(path):
        with np.load(path, allow_pickle=False) as archive:
            return (
                {name: np.asarray(archive[name]) for name in archive.files
                 if name != "_n_steps"},
                int(archive["_n_steps"]),
            )

    registry = Registry(default_registry)
    registry.register_container(
        "test/isolated-torch", Container(".torchtest", writer, reader))
    spec = OutputSpec.from_dict({
        "container": "test/isolated-torch",
        "shard_length": 2,
        "modalities": {
            "act": {"dtype": "float32", "shape": [], "source": "field:act"},
        },
    }, registry=registry)
    entries = write_track(
        str(tmp_path),
        Track("source", 0, [{"act": 1.0}, {"act": 2.0}]),
        spec,
        schema=SCHEMA,
        split="train",
        registry=registry,
    )
    manifest = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "commit_protocol": "sidecar-v1",
        "count": len(entries),
        "shards": portable_shard_entries(str(tmp_path), entries),
    }
    config = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "output": spec.to_dict(),
        "results": {"shards": len(entries)},
        "provenance": {"fingerprint": canonical_hash({"test": "torch-registry"})},
    }
    write_text(tmp_path / "dataset_manifest.json", json.dumps(manifest))
    write_text(tmp_path / "config.json", json.dumps(config))
    commit_dataset(str(tmp_path))

    item = ShardDataset(str(tmp_path), registry=registry)[0]

    assert isinstance(item["act"], torch.Tensor)
    assert item["act"].tolist() == [1.0, 2.0]
