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
transform aintropy/pixel-shift`) and the compositional operator (`kur-describe operator
aintropy/pixel-shift`) -- the Python module itself stays `pixel_shift` (import-identifier
convention), only the registration string is hyphenated.

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
