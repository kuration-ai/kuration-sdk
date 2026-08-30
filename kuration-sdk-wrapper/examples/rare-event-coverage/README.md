# Rare-event coverage

**Oversample the moments a model needs to see, but good players avoid.**

A world model only learns the physics that show up consistently in its training data. And the interactions it most needs to learn (a moving object hitting a solid one, a player-object contact) are the ones good players avoid. That's why DIAMOND's CS:GO world model lets players double- and triple-jump against walls: successive jumps don't appear often enough in real play for the model to learn they shouldn't be possible.

The fix isn't more footage. It's finding the events already in the footage you have and oversampling them past their natural base rate. `RareEventCoverage` is the oversampling half: measure the rare event's per-window rate, then weight the sampler so those windows show up far more often than they do in the wild.

**Composes:** `.measure("event_rate", "rate", of="collision") -> .sample("weighted", n=15, weights={"event_rate": 2.0})` .

**Extras needed:** none. Core install works.

## Run

```
python examples/rare-event-coverage/rare_event_coverage.py
```

## Expected output

```
selected: 15 windows
features on first window: ['event_rate']

compiled chain:
  RareEventCoverage: .measure('event_rate', 'rate', of='collision') -> .sample('weighted', n=15, weights={'event_rate': 2.0})
```

## What to try next

- Increase `weight` (2.0 → 5.0) to pull the sampler further into the tail. Coverage has to be on the axis you care about, and stronger weights buy more of it at the cost of other diversity.
- Change `of="collision"` to another 0/1 event column: `glitch` from the SDK quickstart, a violation-rate column derived from `action_state_consistency`, or your own flag column.
- Rename the measured feature via `feature="collision_rate"` if the default `event_rate` isn't specific enough.

## How this fits the wider curation loop

That's the oversampling side. Upstream, the SDK does the flagging: `collision` for player-object and player-wall contact, `action_state_consistency` for actions whose screen motion breaks the dataset's baseline. Flag first, then oversample.

## Related

- [Kit gallery](../README.md)
- [`design/01_operation_abstraction.md`](../../design/01_operation_abstraction.md): the Operation contract
- [`design/03_config_model.md`](../../design/03_config_model.md): driving this from a YAML config
