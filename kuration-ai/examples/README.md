# Gallery

Three self-contained examples. Each folder holds a runnable script and its own
README with expected output.

- [**Motion-diversity coverage**](./motion-coverage/): cover a motion axis
  (turning, speed, climb) evenly instead of matching the natural distribution,
  so the tail isn't crowded out by the boring middle.
- [**Rare-event coverage**](./rare-event-coverage/): oversample rare events
  (like collisions) that a world model needs but that ordinary play barely
  contains.
- [**Diversity sampling**](./diversity-sampling/): cluster in feature space and
  sample equally from each cluster, so redundant windows don't crowd out the
  diverse ones.

Also here:

- [`chained.py`](./chained.py): the three call styles (direct `.apply()`,
  `run()` with instances, `run()` with a config dict) all producing identical
  results. Not a gallery entry: a config-parity reference for developers
  extending Kuration AI.

## Run any of them

```
pip install kuration-ai
python examples/<folder>/<script>.py
```

Each script is a complete, self-contained ~30 lines of Python that constructs
a small synthetic pipeline and applies one Operation. No downloads, no
credentials, no network required.
