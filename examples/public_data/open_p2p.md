# Worked example — Open Pixel2Play

This example curates Open P2P recordings across games, action patterns, correction sources, and
text-instruction availability. It reads `annotation.proto` directly and deliberately leaves
video decoding out of the feature pass.

The sources are Elefant AI's
[`p2p-toy-examples`](https://huggingface.co/datasets/elefantai/p2p-toy-examples) and
[`p2p-full-data`](https://huggingface.co/datasets/elefantai/p2p-full-data), with the
[`open-p2p`](https://github.com/elefant-ai/open-p2p) training repository as the executable format
reference. The full dataset card describes more than 8,300 hours across over 40 games at 20 FPS;
the official toy dataset is the bounded starting point.

## See the workflow without downloading gameplay

```bash
pip install "kuration-sdk[p2p]"
mkdir -p adapters
curl -O https://raw.githubusercontent.com/aintropy-solutions/kuration-sdk/main/examples/public_data/open_p2p.py
curl -o adapters/open_p2p.py https://raw.githubusercontent.com/aintropy-solutions/kuration-sdk/main/examples/public_data/adapters/open_p2p.py
curl -o adapters/__init__.py https://raw.githubusercontent.com/aintropy-solutions/kuration-sdk/main/examples/public_data/adapters/__init__.py
python open_p2p.py --demo --out open-p2p-out
```

The demo creates four small, wire-compatible `annotation.proto` files containing synthetic
actions, environments, correction frames, and instructions. It contains no video or upstream
data.

## Run on the official toy dataset

Use the upstream download script, which downloads and prepares the toy recording directories:

```bash
git clone https://github.com/elefant-ai/open-p2p.git data/open-p2p-repo
cd data/open-p2p-repo
uv run python scripts/download_data.py --toy --output ../open-p2p-toy
cd ../..
python open_p2p.py data/open-p2p-toy/dataset/*/annotation.proto \
  --out open-p2p-curated --n 100
```

Check the current toy-dataset size and license before downloading. Both dataset cards use a
Modified MIT license with an explicit third-party game-content disclaimer; users remain
responsible for ensuring their use of recorded game media is permitted.

## Raw-to-canonical mapping

| upstream protobuf field | canonical representation |
|---|---|
| `metadata.env` and `env_subtype` | categorical `environment` tag |
| `frame_annotations[].user_action` | keyboard, mouse-button, delta, and scroll fields |
| `frame_annotations[].system_action` | same fields with categorical `action_source=system` |
| `frame_text_annotation[].instruction` | duration-expanded instruction state and output text |
| frame-list position | `frame_index`, retained for joining a selected shard to `192x192.mp4` |

The adapter follows the official loader's precedence: a known system action wins when a frame
also contains a known user action. Unknown frames remain present with `known_action=0`. Text is
expanded for `duration × 20 FPS`, matching the training loader, so selection targets the full
instruction-conditioned span rather than only the first annotated frame.

The example uses a minimal wire-compatible protobuf descriptor. Unknown upstream fields are
ignored by the Protocol Buffers runtime, so the example does not pull in PyTorch, TorchCodec,
Rust extensions, or the rest of Open P2P's training stack.

## The curation decision

```python
pipeline = (
    input(adapter=adapter, sources=sources, schema=schema)
    .window(40, stride=20)
    .measure("key_density", "mean", of="keys_down")
    .measure("mouse_activity", "mean", of="mouse_motion")
    .measure("instruction_rate", "rate", of="instruction_present")
    .bin("key_density", bins=4)
    .bin("mouse_activity", bins=4)
    .tag("environment", "action_source", "instruction_state", min_frac=0.05)
    .sample("equal", n=100)
    .options(mode="strict", seed=33)
)
```

Forty frames represent two seconds. Equal-bin selection balances keyboard and mouse intensity;
tag coverage reserves capacity for available environments, user/system actions, and instruction
states. Output shards preserve mouse deltas, known-action masks, key/button strings, instruction
text, and frame indices. The sibling video stays in the source dataset and can be decoded only
for selected spans.

## Scale to full-data batches

The official downloader supports inclusive batch ranges:

```bash
cd data/open-p2p-repo
uv run python scripts/download_data.py --start 1 --end 5 --output ../open-p2p-batches
cd ../..
python open_p2p.py data/open-p2p-batches/dataset/*/annotation.proto \
  --out open-p2p-curated --n 5000
```

Each recording is a separate source, and the example adds a source-level test split when several
recordings are passed. This prevents overlapping windows from one gameplay recording appearing
in both train and test. Iterate on annotations first; decode or copy video only after selection.

The adapter is [`adapters/open_p2p.py`](adapters/open_p2p.py). Like the TeaPearce adapter, it
lives outside the SDK core because protobuf and game-action conventions are dataset-specific.
