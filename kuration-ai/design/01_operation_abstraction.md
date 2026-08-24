# 01: Operation abstraction and return contract

## Decision

An **Operation** is any object that satisfies the following protocol:

```python
class Operation(Protocol):
    def apply(self, pipeline: kuration.Pipeline) -> kuration.Pipeline: ...
```

That's the entire contract. Structural, not inheritance-based: a user writes a class with `.apply(...)`, they don't import an ABC.

## Return contract

`.apply()` always returns a Kuration Pipeline. It never returns `None`, never returns anything else. A "no-op" operation still returns its input pipeline unchanged: the caller can then chain freely.

## Immutability

Operations never mutate their input pipeline. Kuration pipelines are immutable, so `.apply()` returns a *new* pipeline built from calling SDK methods on the input. Operation instances themselves are `@dataclass(frozen=True)`: parameters are set at construction and never change.

## Error contract

- **Bad parameters** at construction time → `ValueError("<field>: <what's wrong>: <fix>")`.
- **Missing extras** at `.apply()` time → `RuntimeError("this operation needs an optional dependency: install with pip install kuration-ai")`.
- **Semantic mismatch** (e.g. wrong feature name for the pipeline schema) → let the SDK raise, then the wrapper's error surface (issue #16) wraps it with a "did you mean X?" hint.

No silent failures. No default-to-empty-result. If something is wrong, it fails loudly with an actionable message.

## Real call sites

**Tier 1: construct with defaults:**
```python
op = BalancedSelector(feature="activity", of="signal")
```

**Tier 3: user's own operation:**
```python
from dataclasses import dataclass
from kuration_ai import Operation  # Protocol

@dataclass(frozen=True)
class OnlyActiveWindows:
    threshold: float = 0.1

    def apply(self, pipeline):
        return (pipeline
                .measure("motion", "std", of="turn_rate")
                .filter("motion", ">", self.threshold))
```

No import inheritance, no register call required for the class to work standalone. Registration is only needed for the config-dict style (see `03_config_model.md`).

## Alternative considered and rejected

**`class Operation(ABC)` with `.apply` as abstract method.** Rejected because it forces third-party operations to inherit from a specific class, which conflicts with users who already have their own class hierarchy (e.g. a dataclass sitting in their pipeline package). Protocol-based duck typing is strictly cheaper and equally safe with modern type checkers.

## GTM angle

The `.apply(pipeline) -> pipeline` protocol is what makes the "writing your own is a class with one method" line in the README true. It also underpins the no-lock-in tenet: a user leaving the kit doesn't unwrap anything, because every Operation only calls SDK methods that were already available. The extension surface is one class with one method: small enough to fit in the README, small enough to inspect before a prospect commits.

## Non-decisions (deferred)

- Async operations (`.apply_async`): nothing in the SDK is async today, and none of the v0.0.1 use cases need it.
- Composition operators (`op1 | op2`): cute, but adds a special-case surface. `run()` already covers this cleanly.
