# Kuration Pixel Shift

A kuration-sdk operator + batch transform that measures pixel shift for every isolated
key/mouse press found in a dataset recorded in the DIAMOND/TeaPearce 51-dim action-vector
convention (11 keys + 2 clicks + 23-bin mouse_x one-hot + 15-bin mouse_y one-hot). It makes
no judgment about whether the measured shift is "consistent" with anything — it only
reports what pixel motion actually happened for a given control. That's deliberate: a
separate [action-state consistency analyzer](action-state-consistency.md) layers
config-driven controls, expected directions, and baselines over these measurements. Keeping
that stateful judgment separate makes this transform reusable on its own.

## What it produces

1. **Per-segment scalar features**, so it composes inside an ordinary pipeline
   (`.bin()`/`.filter()`/`.sample()`): for each of 8 controls (W/S/A/D and the four signed
   mouse-axis controls), 6 numbers — `{control}_shift_min`, `{control}_shift_max`,
   `{control}_shift_mean`, `{control}_shift_measured`, `{control}_n_instances`, and
   `{control}_requested` — collapsing
   across whatever press-lengths (1/2/3 frames, for W/S/A/D) or delta magnitudes (for the
   mouse controls) were found for that control in that segment. `min`/`max` are the true
   worst-case/best-case signed shift among however many instances were found in that
   segment, not an
   average that could hide one outlier instance among several typical ones.
   `{control}_requested` is `1` for controls the call measured and `0` for controls omitted
   from `controls`; together with `n_instances`, it distinguishes “not requested” from
   “requested, but no qualifying event was found.” Shift statistics are placeholders when
   `n_instances == 0` and must not be interpreted as measured zero motion. The core transform
   contract admits only finite scalars, so there is no NaN to mark them with:
   `{control}_shift_measured` is the explicit 0/1 flag to select on, e.g.

   ```python
   pipeline.pixel_shift(controls=["W"]).filter("w_shift_measured", ">=", 1).bin("w_shift_mean", bins=8)
   ```

   which keeps only segments where a W shift was actually measured (equivalent to
   `w_n_instances >= 1`).

2. **A per-instance report** (`PixelShiftReport`) — `.print_report()` prints counts,
   histograms, and clip/frame provenance to stdout, while `.to_records()` returns that
   ledger as flat dictionaries for storage or combination with action-state records.
   `PixelShiftReport` is a plain object you construct and feed yourself, deliberately
   independent of `PixelShift.transform_batch` -- see "Why the report is a separate
   object" below.

## Install

This operator measures optical flow with `cv2`, which is not part of Kuration's core
dependencies. The implementation ships in the SDK wheel; install its runtime extra with:

```bash
pip install "kuration-sdk[pixel-shift]"
```

Use `[gameplay]` to install this operator together with action-state and SLAM.
`cv2` is imported lazily, only inside functions that touch a frame, so importing Kuration
and discovering the built-in operator remain safe without OpenCV.

For repository development:

```bash
python -m pip install -e ".[dev,pixel-shift]"
python -m pytest tests/test_pixel_shift.py -q
```

## Usage

**As a pipeline transform** — gets you the 6 aggregated scalars per control, per segment,
composable with `.bin()`/`.filter()`/`.sample()` like any other kuration feature:

```python
import kuration.gameplay as gameplay
from kuration import input

pipeline = (
    input(telemetry, schema=schema, fields={"action": action_columns}, time="t")
    .join_video({"my_source": "clip.mp4"}, on="frame_index")
    .window(32, stride=32)
    .pixel_shift(controls=["W", "S", "A", "D"], boundary="exclude")
    .bin("w_shift_mean", bins=8)
    .sample("equal", n=1000)
)
result = pipeline.collect()
```

Registered under `aintropy/pixel-shift` for both the batch transform and the compositional
operator kinds. The method delegates to that registered operator, so serialized jobs and
provenance retain the stable component name. The lower-level Python API lives at
`kuration.gameplay.pixel_shift`.

**For the fuller per-instance report** — `PixelShiftReport` is a plain object you build
yourself, by calling `add_segment()` directly over your own frames/actions, outside pipeline
execution (see "Why the report is a separate object" below for why it isn't just wired into
the transform above):

```python
from kuration.gameplay import pixel_shift

report = pixel_shift.PixelShiftReport()
for frames, actions, segment_id in your_own_segments:
    # segment_id is (source_id, start_idx, end_idx) -- add_segment uses start_idx to give
    # each instance it finds its own precise (source, frame) location, not one label
    # shared by every instance found in this segment (see "Why the report is a separate
    # object" -- this is also why print_report()'s clip listing shows real, distinct
    # per-instance locations instead of collapsing them).
    report.add_segment(frames, actions, segment_id, controls=["W", "S", "A", "D"])
report.print_report()
records = report.to_records()
```

Every record contains `source_id`, `control`, `cell_key`, `frame`, `shift`,
`consistent`, `verdict`, `n_instances`, `frame_space`, and `frame_space_note`.
Pixel-shift measures rather than judges, so `consistent` and `verdict` are `None`;
`frame_space` is `"source"`. `n_instances` is the total recorded for that `source_id`.
This is the same schema returned by action-state's `AnomalyChecker.to_records()`, so the
two lists can be concatenated directly.

### Why the report is a separate object

This is the detailed expert ledger path; the default input-backed surface remains the
pipeline-preserving `.pixel_shift(...)` method.

An earlier version accumulated per-instance detail into a module-level dict as a side effect
of `transform_batch`. That state was invisible to the SDK contract and made results depend on
where a physical plan happened to run transforms. Today the ordinary count path runs transforms
after the parallel feature pass, while bounded plans fuse transforms into their local feature
stage; neither placement promises a process-global result channel.

So `transform_batch` now does exactly one thing: compute the 6 aggregated scalars per
control, through the one channel the SDK actually supports for returning data (declared
per-segment outputs) -- correct regardless of worker count. `PixelShiftReport` gets the
fuller per-instance detail by being called directly, in your own code, outside pipeline
execution entirely -- there's no ambiguity about worker aggregation to get wrong, because
pipeline execution never touches it. It's also a plain object rather than a module-level
singleton, so there's no `reset_ledger()`-style footgun between separate reports in the
same process -- just construct a new one.

## Input and segment-boundary contract

The operator fails early on malformed data instead of allowing `argmax`, OpenCV, or an
all-zero result to hide the cause. Actions must be a finite numeric `(T, 51)` array: the
11 key and 2 click fields are binary, and both mouse ranges are strict one-hot rows.
Frames must be `(T, H, W, 3)`, have the same `T`, use `uint8` or `float32`, contain only
finite values, and be at least 4×4 pixels. Control names, duplicates, empty control lists,
press lengths, axes, color spaces, segment IDs, and lookaround shapes are validated too.
Python callers may use any non-empty sequence such as a list or tuple for `controls`.

An arbitrary window boundary can cut through a press. The registered pipeline transform declares
one lookbehind row **on the action input only** (`lookbehind={"field:action": 1}` — the
measurement never reads a lookbehind frame, so none is decoded) and three lookahead action/frame
rows, so the SDK supplies enough bounded
context to classify every supported one-, two-, or three-frame press. Overlapping windows use the
SDK's `rightmost-start` ownership rule: an event belongs to the latest window whose start is at or
before the event, and per-segment counts do not duplicate it. Source edges still have shorter
context. The default `boundary="exclude"` conservatively rejects a press whose true start or
release lies outside the source; `boundary="error"` makes missing source-edge context loud, and
`boundary="assume-inactive"` supplies the explicit compatibility behavior.

Direct callers of `find_isolated_press_sequences`, `release_transition_displacement`,
`segment_scalars`, and `PixelShiftReport.add_segment` can also provide bounded context:
`lookbehind_actions` disambiguates the left edge, while matching `lookahead_actions` and
`lookahead_frames` make a right-edge release measurable. Only the final lookbehind row and
at most `press_len` lookahead rows are consumed. Direct calls over overlapping windows do not
automatically share the pipeline's ownership metadata, so `PixelShiftReport` tracks each recorded
instance's global `(source_id, control, cell_key, frame)` and skips exact duplicates. Direct
per-window `segment_scalars` calls should pass `owned_length` when they need the same
rightmost-start ownership rule.

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
construction, the latter is untested. Radial geometry uses float32 and a two-resolution LRU
cache: that bounds process-global memory to roughly two sets of three image-sized grids, at the
cost of recomputation if one process repeatedly alternates among more than two resolutions.

**4. Coordinate / sign convention**:
- `_prep_gray(img, color_space="BGR")` defaults to **BGR**, matching how ASG/DIAMOND HDF5
  frames are stored, and picks the matching `cv2` conversion flag (`COLOR_BGR2GRAY` vs
  `COLOR_RGB2GRAY`) rather than reordering the frame array -- getting this flag wrong
  silently swaps the red/blue channel weighting in the grayscale conversion, systematically
  biasing the resulting flow, without erroring. This matters concretely, not just
  hypothetically: kuration-sdk's `.join_video()` always tags its media column
  `color_space: "RGB"` (`src/kuration/core/media.py` — its first-party PyAV provider decodes
  `rgb24`), the *opposite* of this function's default. The registered `aintropy/pixel-shift`
  transform handles this for you -- `PixelShift.transform_batch` reads the real declared
  `color_space` off the media column the run actually resolved for its `media:frame` role
  (`ctx.input_token("media:frame")` against `ctx["schema"].media`, populated by the SDK runtime
  itself, so a per-use `inputs=("media:frames2", "field:controls")` remapping works too)
  and threads it down through `segment_scalars` and the measurement helpers to
  `_prep_gray`, so calling this operator through a
  normal kuration `Pipeline` is correct regardless of source, with no extra array copy.
  This threading only happens at that SDK-integration boundary; if you call
  `diamond_action_decode`'s functions directly on raw arrays (bypassing the SDK transform),
  the default stays BGR and you're responsible for passing `color_space="RGB"` yourself if
  that's what you actually have.
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

## Domain placement

This is intentionally domain-specific — it only works on datasets using the DIAMOND
action-vector layout, and (unlike `examples/02_robotics_action_aware.py`) it has no
synthetic self-contained demo mode; it's meant to run against real recorded gameplay.

The implementation lives under `src/kuration/gameplay/pixel_shift` and is included in the
`kuration-sdk` wheel. Importing `kuration.gameplay` registers it for `Registry.snapshot()`
and programmatic discovery; the generic core remains free of gameplay component names.
OpenCV remains isolated in the `pixel-shift` extra.

## Command-line interface

Once `kuration.gameplay` is imported, both registered components are discoverable from the
command line: `kur-describe transform aintropy/pixel-shift` and `kur-describe operator
aintropy/pixel-shift` show each one's parameter contract; `kur-list operators` and
`kur-list transforms` list them alongside every other registered component.
