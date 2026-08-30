# 03: Configuration model

## Decision

Three tiers of configuration, one per user sophistication level. All three describe the same underlying pipeline.

## Tier 1: Defaults

Every operation has sensible defaults. A tier-1 user writes:

```python
from kuration_sdk_wrapper import BalancedSelector
result = BalancedSelector(feature="motion", of="turn_rate").apply(pipeline).collect()
```

Only the two truly-required parameters (`feature`, `of`) are named. Bins, count, kind: all defaulted (`bins=8`, `count=100`, `kind="std"`).

## Tier 2: Declarative (config file)

Same pipeline, expressed as YAML:

```yaml
# curate.yaml
operations:
  - type: RareEventCoverage
    of: collision
    count: 200
  - type: BalancedSelector
    feature: motion
    of: turn_rate
    bins: 8
    count: 100
```

Loaded and executed with:

```python
import yaml
from kuration_sdk_wrapper import run
with open("curate.yaml") as f:
    config = yaml.safe_load(f)
result = run(pipeline, config["operations"]).options(seed=1).collect()
```

No YAML dependency at import time: users load their own config however they like (YAML, JSON, TOML, dict literal). `run()` takes a Python list of dicts.

## Tier 3: Programmatic

Full Python. Custom operations, dynamic parameters, whatever:

```python
from dataclasses import dataclass
from kuration_sdk_wrapper import Operation, run

@dataclass(frozen=True)
class ThresholdedBalancer:
    threshold: float
    def apply(self, pipeline):
        return (pipeline
                .measure("motion", "std", of="turn_rate")
                .filter("motion", ">", self.threshold)
                .bin("motion", bins=8)
                .sample("equal", n=self.dynamic_count()))
    def dynamic_count(self):
        # user's own logic
        return 100

result = run(pipeline, [ThresholdedBalancer(threshold=0.15)]).collect()
```

Tier-3 users never touch the config-dict path unless they want to.

## Precedence (defaults → declarative → programmatic)

**Instance overrides dict.** Overriding happens at construction, not at merge time: a user passing both `BalancedSelector(bins=16)` AND `{"type": "BalancedSelector", "bins": 8}` in the same `run()` call is a bug, not a merge. `run()` errors with:

```
config item 3 conflicts with an instance for the same slot: pass one or the other, not both.
```

Explicit is better than implicit.

## Config resolution errors (actionable, per CLAUDE.md principle 4)

Bad config dicts fail with a message that names both the problem and the fix:

- Missing `type` key:
  ```
  config item 2 missing 'type' key. expected one of: BalancedSelector, RareEventCoverage, DiverseSubset
  ```
- Unknown `type`:
  ```
  config item 2 type='balancedselector' not found. did you mean 'BalancedSelector'? (case-sensitive)
  ```
- Unknown parameter:
  ```
  BalancedSelector got unexpected keyword 'bin': did you mean 'bins'?
  ```

Error messages carry more detail than exceptions in the SDK do: this is a wrapper's job (issue #16).

## Alternative considered and rejected

**JSON Schema for validation, mirroring the SDK's operator descriptors.** Rejected because kit Operations are Python classes with typed dataclass fields: the class *is* the schema. Duplicating it as a JSON schema means two sources of truth and inevitable drift.

**Config merging across layers (e.g. defaults from `~/.kuration-sdk-wrapper.yaml`).** Rejected as premature: no v0.0.1 use case needs machine-wide defaults, and adding them opens hairy questions about precedence. Add if real demand shows up.

## GTM angle

The config file is the reproducibility artifact. A team commits `curate.yaml` to their repo; a teammate a year later runs it against the same SDK version and gets identical shards. This is the concrete behavior behind the reproducibility differentiator: not an abstract claim, a specific file you can point at in a demo. The three tiers are also visible in the config surface: Tier 1 uses no config, Tier 2 uses YAML, Tier 3 wraps in Python.

## Non-decisions (deferred)

- CLI (`ksw run curate.yaml`): could ship as a `[project.scripts]` entry point in v0.0.2, trivial.
- Environment-variable overrides: `${VAR}` substitution in config values. Trivial to add later; adds a moving part today.
