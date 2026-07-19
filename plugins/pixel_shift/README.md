# pixel_shift

A kuration-sdk operator + batch transform that measures pixel shift for every isolated
key/mouse press found in a dataset recorded in the DIAMOND/TeaPearce 51-dim action-vector
convention (11 keys + 2 clicks + 23-bin mouse_x one-hot + 15-bin mouse_y one-hot). It makes
no judgment about whether the measured shift is "consistent" with anything — it only
reports what pixel motion actually happened for a given control. That's deliberate: a
separate, genuinely dataset-agnostic outlier-finding operator is meant to be layered on
top of this one's output (config-driven controls + expected direction per control, not
hardcoded), kept out of this contribution.

## What it produces

1. **Per-segment scalar features**, so it composes inside an ordinary pipeline
   (`.bin()`/`.filter()`/`.sample()`): for each of 8 controls (W/S/A/D and the four signed
   mouse-axis controls), 4 numbers — `{control}_shift_min`, `{control}_shift_max`,
   `{control}_shift_mean`, `{control}_n_instances` — collapsing across whatever
   press-lengths (1/2/3 frames, for W/S/A/D) or delta magnitudes (for the mouse controls)
   were found for that control in that segment. `min`/`max` are the true worst-case/
   best-case signed shift among however many instances were found in that segment, not an
   average that could hide one outlier instance among several typical ones.

2. **A console report** (`print_report()`, prints to stdout, writes nothing to disk) — for
   every (control, press_len/delta) combination found across the whole dataset: instance
   count, an ASCII histogram of the shift values, and which clips (by source + frame
   range) each instance came from.

## Install

This operator measures optical flow with `cv2`, which is *not* part of kuration-sdk's core
dependencies (the core stays free of heavy, domain-specific libraries -- see "Why this lives
in `plugins/`" below). Install it with:

```bash
pip install "kuration-sdk[pixel_shift]"
```

`cv2` is imported lazily, only inside the functions that actually touch a frame -- so
installing kuration-sdk *without* this extra still works fine for everything else,
including `kur-list`/`kur-describe` (which import every registered plugin module to read
its descriptors, this one included). Calling into this operator's flow-measuring code
without the extra installed raises a clear `ImportError` telling you to install it, rather
than either crashing plugin discovery for unrelated users or failing with a bare
`ModuleNotFoundError`.

## Usage

```python
import plugins.pixel_shift.pixel_shift_operator as pixel_shift

pipeline = (
    input(telemetry, fields={"action": action_columns}, time="t")
    .join_video({"my_source": "clip.mp4"}, on="frame_index")
    .window(32, stride=32)
    .apply("aintropy/pixel-shift", controls=["W", "S", "A", "D"])
)
result = pipeline.collect(n=1000)
pixel_shift.print_report()
```

Registered under `aintropy/pixel-shift` for both the batch transform (`kur-describe
transforms aintropy/pixel-shift`) and the compositional operator (`kur-describe operators
aintropy/pixel-shift`) -- the Python module itself stays `pixel_shift` (import-identifier
convention), only the registration string is hyphenated.

## Method — the specifics results depend on

The measurement is deliberately narrow: one Farneback optical-flow computation between
exactly two frames per isolated-press instance, projected onto one axis. Every parameter
below was inherited unchanged from the internal batch script this measurement methodology
was originally validated against (`batch_job_sensitivity_w.py`, not part of this SDK), so
none of these are arbitrary — but they are also not universal constants, and results
depend on all five.

**1. Frame crop (`BOTTOM_CROP_FRAC = 0.18`)** — the bottom 18% of every frame is dropped
before computing optical flow (`_prep_gray`): `gray[: int(h * 0.82), :]`. This removes the
weapon viewmodel / ammo HUD band, which sits in the bottom portion of the frame in CS2/
CS:GO and does not move with camera rotation the way the world does — including it would
inject a constant near-zero-motion region into every flow computation regardless of the
actual camera/player motion. This value is empirical, marked "provisional" in its source,
and assumes a HUD layout similar to CS2's; a dataset with a different HUD position (or no
HUD baked into frame) would need a different crop fraction, not necessarily 0.18.

**2. Farneback parameters** (`FARNEBACK_KWARGS = dict(pyr_scale=0.5, levels=3, winsize=15,
iterations=3, poly_n=5, poly_sigma=1.2, flags=0)`, passed straight to
`cv2.calcOpticalFlowFarneback`) — `winsize=15` in particular is a pixel count, not a
fraction of the frame, so it is implicitly tuned to the resolution these values were
validated at (DIAMOND's 150×280 low/full-res frame size). At a much higher resolution
(e.g. raw 1920×1080 capture), a 15-pixel window covers a proportionally tiny fraction of
the frame and would likely need re-tuning; the code itself is resolution-agnostic (no
hardcoded frame size), the calibration behind the numbers is not.

**3. Frame resolution** — only validated end-to-end at DIAMOND's 150×280 frame size (real
ASG capture, see the PR's test plan). The functions place no hardcoded assumption on `(H,
W)` — `_radial_unit_vectors` and the crop both derive from the actual array shape — but
"runs without erroring at another resolution" and "produces calibration-comparable
numbers at another resolution" are different claims; only the former is currently true by
construction, the latter is untested.

**4. Coordinate / sign convention**:
- Frames must be **BGR** (`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` assumes BGR channel
  order, matching how ASG/DIAMOND HDF5 frames are stored). Feeding RGB frames would not
  error, but `cv2`'s BGR-weighted luminance conversion would silently swap the red/blue
  channel weighting, systematically biasing the resulting grayscale image and therefore
  the flow.
- Axis projection and sign, matching DIAMOND's own convention
  (`asg2diamond/action_map.py`, cross-validated by that project's `asg2d-verify-alignment`
  tool): W/S project onto **radial** flow (outward from frame-center = positive = W;
  inward = negative = S) since forward/backward motion produces expansion/contraction
  around the center, uncontaminated by lateral motion or camera rotation under this
  projection. A/D and the mouse-yaw controls project onto **horizontal** flow directly (no
  center-relative projection — strafing/turning has no in-frame focus of expansion);
  mouse-pitch controls project onto **vertical** flow directly. Positive horizontal flow
  = world sliding right = consistent with `A` (moving left) or a positive `mouse_x` (per
  DIAMOND's convention, positive mouse_x = looking right, so content sweeps left against a
  rightward turn — i.e. `mouse_x_pos`'s expected sign is negative on this axis, confirmed
  empirically, not assumed).

**5. Only the release-transition frame pair is measured, not the whole press+aftermath
sequence** — for a `press_len=N` isolated press, the ONE frame pair measured is index
`N-1` to `N` (the last held frame to the first aftermath frame), not every frame within
the sequence `find_isolated_press_sequences` returns. This is deliberate, not an
oversight: the reference calibration data this operator's sibling consistency-checking
layer is built against was itself measured at exactly this transition (see
`batch_job_sensitivity_w.py`'s "position 0 = release transition" convention) — pixel
shift measurably decays over the frames following a key release, so measuring any other
point in the sequence would not be comparable to that calibration at all.

## Why this lives in `plugins/`, not `src/kuration/` or `examples/`

This is intentionally domain-specific — it only works on datasets using the DIAMOND
action-vector layout, and (unlike `examples/02_robotics_action_aware.py`) it has no
synthetic self-contained demo mode; it's meant to run against real recorded gameplay.
`CONTRIBUTING.md`'s own design rule keeps domain knowledge out of the shared core, and the
existing `examples/` are all self-generated-data demonstrations, not real-dataset tooling.
`plugins/` is registered through the `kuration.plugins` entry-point mechanism described in
`CONTRIBUTING.md` (see `pyproject.toml`'s `[project.entry-points."kuration.plugins"]`), so
`kur-list`/`kur-describe` discover it automatically on install, the same as any other
installed plugin -- confirmed against a real, clean, non-editable install (see the test
plan in the PR description).
