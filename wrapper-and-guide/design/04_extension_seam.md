# 04: Extension seam for user-registered operations

## Decision

Anything with `.apply(pipeline) -> pipeline` is a valid Operation. Users:

1. Write a class (dataclass recommended, not required).
2. Use it directly by importing and instantiating: no registration needed.
3. Register it with `kit.register("Name", Class)` only if they want to use it via the config-dict style.

Registration is optional and additive. Direct use is always available.

## Worked example (real user code, executable)

A robotics engineer wants "only windows where the end effector is actually moving":

```python
from dataclasses import dataclass
from kuration_ai import run, BalancedSelector

@dataclass(frozen=True)
class ActiveEndEffectorOnly:
    min_speed: float = 0.05

    def apply(self, pipeline):
        # Uses a distinct feature name so downstream operations that measure
        # their own columns don't collide: SDK measure names must be unique
        # per pipeline.
        return (pipeline
                .measure("_ee_speed_gate", "mean", of="end_effector_velocity")
                .filter("_ee_speed_gate", ">", self.min_speed))

# Use directly
result = run(pipeline, [
    ActiveEndEffectorOnly(min_speed=0.1),
    BalancedSelector(feature="ee_activity", of="end_effector_velocity", kind="std", bins=8, count=100),
]).collect()
```

Six lines of user code for a domain-specific operation. Ready to ship.

## Optional registration for config-dict use

If the same user wants to drive `ActiveEndEffectorOnly` from a YAML file, they register it once at module load:

```python
from kuration_ai import register
register("ActiveEndEffectorOnly", ActiveEndEffectorOnly)
```

Now this YAML works:

```yaml
operations:
  - type: ActiveEndEffectorOnly
    min_speed: 0.1
  - type: BalancedSelector
    feature: activity
    of: ee_speed
    bins: 8
    count: 100
```

`register()` is a plain function, not a decorator. A class can be re-used across projects and registered under different names in each.

## Interaction with the SDK's `@register_operator`

Kit Operations and SDK operators serve different needs:

| SDK `@register_operator` | Kit Operation |
|---|---|
| Adds a fluent method `pipeline.<name>(...)` on every SDK Pipeline. Discoverable via `kur list operators`. | Adds a class-shaped, standalone unit. Discoverable via `dir(kuration_ai)` or the kit gallery. |
| Register-time JSON Schema validates params. | Dataclass field types validate at construction. |
| Namespace-qualified spelling `.example__op(...)` for collisions. | Python class naming rules. |
| Best when: users think fluent chain. | Best when: users think "which operation do I want". |

Both are additive. A user can wrap an SDK operator inside a kit Operation:

```python
@dataclass(frozen=True)
class PixelShiftAll:
    def apply(self, pipeline):
        return pipeline.pixel_shift()  # SDK-registered fluent method
```

That is the intended interop: kit Operations are a curated presentation layer over whatever the SDK exposes.

## Registration signature

```python
def register(name: str, operation_class: type) -> None:
    """Register a user Operation class under a name for config-dict use.

    Raises:
        ValueError: if `name` is already registered with a different class.
    """
```

Idempotent for the same (name, class) pair: safe to re-run on module reload. Loud on collision: a user redefining `BalancedSelector` gets an error, not silent shadowing.

## Alternative considered and rejected

**Entry-point-based auto-discovery (like the SDK's `kuration.plugins`).** Rejected for v0.0.1 because it forces users to publish a wheel to get discoverability. Most kit users write the operation inside their own project's code: one `register()` call in their `__init__.py` is enough. Add entry-point support in v0.0.2 if third parties want to ship kit-compatible packages.

**Requiring inheritance from a base class.** Same reasoning as `01_operation_abstraction.md`: Protocol beats ABC for user extensibility.

## GTM angle

This is the no-lock-in tenet made real. A user who has invested in a custom Operation is not captured by the kit: the class is theirs, uses only SDK methods, and can be dropped into raw SDK code without modification. The user story is direct: delete the `kuration_ai` imports and your Operations still work. That claim is only credible because the extension seam requires no kit-specific inheritance or registration to function.

The extension surface is also the answer to a common pitch objection: "what if I need a curation step you don't have?" The answer is a class with one method plus, if you want config-dict use, one `register()` call.

## Non-decisions (deferred)

- `unregister()`: nobody needs this in v0.0.1. Adds surface area for zero real benefit.
- Namespaced registry (`user/OpName`): the SDK uses this because it has 45+ registered components. Kit will ship with three. Revisit at 20+.
