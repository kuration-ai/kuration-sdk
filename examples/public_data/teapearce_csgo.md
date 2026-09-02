# Worked example — TeaPearce Counter-Strike Deathmatch

This example curates human Counter-Strike play for behavioral-cloning or world-model training.
It favors coverage across keyboard and mouse activity while reserving part of the selected set
for windows containing rare kill and death events.

The source is the public
[`TeaPearce/CounterStrike_Deathmatch`](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch)
dataset and its
[`Counter-Strike_Behavioural_Cloning`](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning)
code. The dataset card describes approximately 748 GB of HDF5 archives. Each recording contains
1,000 ordered frames, and the official configuration runs at 16 Hz with 150×280×3 frames.

## See the workflow without downloading gameplay

Install the HDF5 extra and run the structural demo:

```bash
pip install "kuration-sdk[array]"
mkdir -p adapters
curl -O https://raw.githubusercontent.com/aintropy-solutions/kuration-sdk/main/examples/public_data/teapearce_csgo.py
curl -o adapters/teapearce.py https://raw.githubusercontent.com/aintropy-solutions/kuration-sdk/main/examples/public_data/adapters/teapearce.py
curl -o adapters/__init__.py https://raw.githubusercontent.com/aintropy-solutions/kuration-sdk/main/examples/public_data/adapters/__init__.py
python teapearce_csgo.py --demo --out teapearce-out
```

The demo generates four tiny HDF5 files with the upstream key layout and synthetic flat-color
images. It contains no copied gameplay or upstream data. Add `--include-frames` to exercise the
image encoder seam as well as action output.

## Run on an upstream recording

The Hugging Face repository stores recordings inside multi-gigabyte archives; it does not offer
an official tiny sample. The smallest practical bounded route is to download one expert archive,
extract one HDF5 member, and run only that recording. For example, the `othermaps` archive is
listed as approximately 3.85 GB:

```bash
python -m pip install -U huggingface_hub
hf download TeaPearce/CounterStrike_Deathmatch dataset_dm_expert_othermaps.tar \
  --repo-type dataset --local-dir data/teapearce
tar -tf data/teapearce/dataset_dm_expert_othermaps.tar
tar -xf data/teapearce/dataset_dm_expert_othermaps.tar <one-member-name-from-the-list>
python teapearce_csgo.py <extracted-recording.hdf5> \
  --out teapearce-out --n 12
```

Downloading the archive is not part of CI. Check current sizes and access terms on the dataset
page before downloading. The card labels the dataset MIT; gameplay imagery depicts third-party
game content, so confirm that storage, redistribution, and use fit your project.

## Raw-to-canonical mapping

The upstream file uses per-frame keys rather than one time-major array per field:

| upstream key | canonical representation |
|---|---|
| `frame_<i>_y` | original 51-value `action` vector plus decoded keyboard, click, and mouse fields |
| `frame_<i>_helperarr[0]` | scalar `kill` and categorical `combat.kind:kill` event |
| `frame_<i>_helperarr[1]` | scalar `death` and categorical `combat.kind:death` event |
| `frame_<i>_x` | optional output-only `frame` tensor through `FrameEncoder` |
| `frame_<i>_xaux` | intentionally omitted; the upstream project marks this auxiliary input unused by its final agent |

The 51 action values follow the official layout: 11 keyboard bits, left/right click bits, 23
horizontal mouse bins, and 15 vertical mouse bins. Images are not metric fields: copying a large
frame tensor into every feature window would waste memory. The adapter carries raw stored channel
values only when `--include-frames` is requested; it performs no color conversion or
normalization, and an output encoder writes them after selection.

## The curation decision

[`teapearce_csgo.py`](teapearce_csgo.py) builds this chain over the custom adapter:

```python
pipeline = (
    input(adapter=adapter, sources=sources, schema=schema)
    .window(96, stride=48)
    .measure("action_density", "mean", of="keys_down")
    .measure("mouse_activity", "mean", of="mouse_motion")
    .measure("shoot_rate", "rate", of="left_click")
    .bin("action_density", bins=4)
    .bin("mouse_activity", bins=4)
    .tag("combat.kind", min_frac=0.08)
    .sample("equal", n=12)
    .options(mode="strict", seed=21)
)
```

The 96-frame window matches the upstream recurrent sequence length. Equal-bin selection avoids
letting common low-motion play dominate, while tag coverage asks for kill and death windows when
those events exist in the candidate pool. The output preserves the original action target,
decoded mouse values, kill/death labels, presence mask, and optionally raw frame tensors.

## Scaling beyond one recording

Pass several extracted files as separate sources. The script adds a source-level test split when
more than one recording is present, so overlapping windows from one recording cannot leak across
train and test. Shell globs work when expanded by the shell:

```bash
python teapearce_csgo.py data/teapearce/extracted/*.hdf5 \
  --out teapearce-curated --n 5000 --include-frames
```

Start without `--include-frames` to iterate cheaply on the selection strategy. Image output is
the dominant storage cost. For large runs, extract archives incrementally, use multiple source
files, and keep the emitted `config.json` with the shards for provenance.

The example adapter lives in [`adapters/teapearce.py`](adapters/teapearce.py). It is deliberately
outside `src/kuration/`: dataset-specific decoding belongs at the SDK's adapter and encoder seams,
not in the dataset-agnostic core.
