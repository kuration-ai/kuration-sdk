"""PyTorch `Dataset` over a curated kuration dataset — the read-side integration (`[torch]`).

A thin, duck-typed wrapper over `kuration.loaders`: one shard = one item, each modality a
tensor. All the work (resolve manifest, pick container, read + reshape + trim) is the numpy
core in `loaders`; this only turns the arrays into tensors, so the framework-specific surface
stays tiny. Use it directly with `torch.utils.data.DataLoader`.

    from kuration.torch import ShardDataset
    ds = ShardDataset("out_dir", split="train")
    item = ds[0]                       # {modality -> torch.Tensor (T, *shape)}

Note: with `short_policy="keep"`, shards have variable T — supply a custom `collate_fn` (or
write with `short_policy="pad"`/a fixed `shard_length`) before batching.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from .loaders import read_shard_arrays, shard_entries


class ShardDataset:
    """A map-style dataset (`__len__` + `__getitem__`) — works with `DataLoader` without
    subclassing `torch.utils.data.Dataset`. `transform(item)` runs per item if given."""

    def __init__(self, dataset, *, split: Optional[str] = None,
                 modalities: Optional[Sequence[str]] = None, trim: bool = True,
                 container: Optional[str] = None,
                 transform: Optional[Callable[[dict], dict]] = None,
                 include_meta: bool = False):
        try:
            import torch
        except ImportError as e:
            raise ImportError("kuration.torch needs PyTorch: "
                              "pip install 'kuration-sdk[torch]'") from e
        self._torch = torch
        self.entries, self.container = shard_entries(dataset, split=split, container=container)
        self.modalities = modalities
        self.trim = trim
        self.transform = transform
        self.include_meta = include_meta

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        entry = self.entries[i]
        arrays = read_shard_arrays(entry, self.container, modalities=self.modalities, trim=self.trim)
        item: dict[str, Any] = {
            k: self._torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}
        if self.include_meta:
            item["_meta"] = {"source_id": entry.get("source_id"), "entity_id": entry.get("entity_id"),
                             "split": entry.get("split"), "n_steps": entry.get("n_steps"),
                             "padded": entry.get("padded")}
        if self.transform is not None:
            item = self.transform(item)
        return item
