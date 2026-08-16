# action-state-consistency

A per-clip verdict on whether pressing a key actually did what the training data claims it did. Real optical flow, real pipeline, no download needed.

![](https://img.shields.io/badge/51--dim_action-DIAMOND_encoding-0D9488?style=for-the-badge)
![](https://img.shields.io/badge/8_frames-1_isolated_W_press-0D9488?style=for-the-badge)
![](https://img.shields.io/badge/synthetic_provider-real_optical_flow-0D9488?style=for-the-badge)

![](https://img.shields.io/badge/verdicts-consistent_inconsistent_inconclusive-0D9488)
![](https://img.shields.io/badge/per--clip-baseline_compared-0D9488)

[![Colab](https://img.shields.io/badge/Colab-open-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aintropy-solutions/beta-solutions/blob/main/kuration-wrapper-and-guide/notebooks/action-state-consistency/action-state-consistency.ipynb) [![Binder](https://img.shields.io/badge/Binder-launch-579ACA?logo=jupyter&logoColor=white)](https://mybinder.org/v2/gh/aintropy-solutions/beta-solutions/HEAD?labpath=kuration-wrapper-and-guide%2Fnotebooks%2Faction-state-consistency%2Faction-state-consistency.ipynb)

> **Launcher status.** The URLs above point at `aintropy-solutions/beta-solutions`. Colab and Binder can only clone public repositories, so the badges activate once this repo goes public. In the meantime, clone and run the notebook locally with `pip install kuration-sdk[gameplay]`.

<details><summary><b>Contents</b></summary>

- [Setup](#setup)
- [The idea in one paragraph](#the-idea-in-one-paragraph)
- [A minimal synthetic dataset](#a-minimal-synthetic-dataset)
- [The pipeline](#the-pipeline)
- [Flipping the baseline](#flipping-the-baseline)
- [Compose downstream](#compose-downstream)
- [Extending to real footage](#extending-to-real-footage)
- [Try it locally](#try-it-locally)
- [Related](#related)

</details>

## Setup

One install cell registers `pixel_shift` and `action_state_consistency` on the pipeline. The `gameplay` extra pulls in the optical-flow backend the analyzer uses.

```python
%pip install --quiet 'kuration-sdk[gameplay]'
```

See cell 2.

## The idea in one paragraph

An action-conditioned world model learns whatever rule the training footage shows it. Press W, camera moves forward. Except sometimes it does not, because a wall was in the way, because the engine buffered the input, because the scope changed the movement math. The `action_state_consistency` operator gives every clip a verdict: `consistent`, `inconsistent`, or `inconclusive`. It works by isolating single, uninterrupted key presses, projecting optical flow onto the axis that press is supposed to move the camera on, and comparing against a baseline built from the dataset's own typical behavior.

## A minimal synthetic dataset

Frames come from an in-memory `FakeVideoProvider` that yields shifting-noise textures optical flow can actually track. Actions follow the real 51-dim DIAMOND encoding (`ACTION_WIDTH = N_KEYS(11) + 2 + N_MOUSE_X(23) + N_MOUSE_Y(15) = 51`). One 8-frame clip. One isolated W press at frame 3. Everything else at rest.

```python
N_FRAMES = 8
PRESS_AT = 3
actions = [action_vector() for _ in range(N_FRAMES)]
actions[PRESS_AT] = action_vector(w_pressed=True)
```

See cell 5.

## The pipeline

Three lines. Input the rows against a schema, join a video source, slice into windows. Then a single `action_state_consistency` call with the baselines and controls to check. A `baseline` says: for this key press, expect pixel shift in this direction and roughly this magnitude. In real use, you compute baselines from the dataset itself (10th to 90th percentile per control cell). Here the notebook hand-writes a plausible one for W.

```python
baselines = {"W": {"1": {"expected_sign": 1, "min": 0.0, "max": 1e9}}}

pipeline = (
    input({"clip1": rows}, schema=SCHEMA, fields={"action": ACTION_COLUMNS}, time="t")
    .join_video({"clip1": "fake.mp4"}, provider=provider, on="frame_index")
    .window(4, stride=2)
)

checked = pipeline.action_state_consistency(baselines=baselines, controls=["W"])
result = checked.metadata["gameplay"]["action_state_consistency"]["result"]
```

See cell 7.

## Flipping the baseline

If the baseline expects W to shift pixels in the *negative* direction (backward-looking convention, or a mislabeled control), the same footage now fails. Measured motion is the opposite of what the baseline says it should be. Same clip, different verdict.

```python
flipped_baselines = {"W": {"1": {"expected_sign": -1, "min": 0.0, "max": 1e9}}}
```

See cell 9.

## Compose downstream

The verdict lives on the pipeline. The pipeline is still just a pipeline. So a `.filter()` on the flagged windows, or a `.sample()` that weights against them, chains directly on top. `.describe()` prints the exact operator chain.

See cell 11.

## Extending to real footage

The SDK ships a full worked example against the public TeaPearce Counter-Strike Deathmatch dataset at [`examples/public_data/teapearce_csgo.md`](https://github.com/aintropy-ai/kuration-sdk/blob/main/examples/public_data/teapearce_csgo.md). The upstream Hugging Face repo stores recordings inside multi-gigabyte archives (no official tiny sample). The smallest practical route is to download one expert archive, extract one HDF5 member, and run only that recording:

```bash
pip install -U huggingface_hub
hf download TeaPearce/CounterStrike_Deathmatch dataset_dm_expert_othermaps.tar \
  --repo-type dataset --local-dir data/teapearce
tar -xf data/teapearce/dataset_dm_expert_othermaps.tar <one-member>
python examples/public_data/teapearce_csgo.py <extracted-recording.hdf5> --out out/ --n 12
```

See cell 12.

## Try it locally

```bash
git clone https://github.com/aintropy-solutions/beta-solutions.git
cd beta-solutions/kuration-wrapper-and-guide/notebooks/action-state-consistency
uv pip install 'kuration-sdk[gameplay]' jupyter
uv run jupyter lab action-state-consistency.ipynb
```

## Related

- Paired long-form post: [aintropy.ai/blog/action-state-consistency](https://aintropy.ai/blog/action-state-consistency)
- Back to the wrapper index: [`../../README.md`](../../README.md)
