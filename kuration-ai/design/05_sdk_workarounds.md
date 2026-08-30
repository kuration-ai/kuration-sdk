# 05: Workarounds where the SDK shape fights the wrapper

## Decision

Every awkward SDK behavior is handled inside the wrapper. The SDK is not modified. Where a workaround adds real cost, that cost is documented so we can decide later whether to file an upstream SDK issue.

## Workaround table

| SDK behavior | Wrapper workaround | Cost of the workaround |
|---|---|---|
| `kur list` requires a `kind` argument with no top-level hint | `kai list` prints per-kind counts and an example command. `kai list <kind>` still works. | One extra CLI subcommand to maintain. |
| Extras are not visible in `kur list operators`: a user can see `aintropy/slam-quality` and then hit a runtime install-hint error only at execution | Each kit Operation carries a class-level `EXTRAS = [...]` attribute; `kai list` annotates which extras are needed for each Operation to actually run. | Adds an attribute on every Operation. Manual sync with SDK extras. |
| Operators registered under `namespace/name` (e.g. `aintropy/pixel-shift`), methods bare (e.g. `uniform-bins`): inconsistent naming across kinds | Kit Operations always use `PascalCase` Python class names. The namespace convention is invisible to a kit user. | None: a naming policy. |
| Several base methods (`uniform-bins`, `weighted`, `top-k`, `target-distribution`, `filter`) register with empty JSON-Schema descriptors: `kur describe` cannot introspect them | Kit re-declares parameters as dataclass fields on each Operation. The class is the schema. | Duplication of parameter knowledge: but the kit uses fewer SDK methods than the SDK exposes, so the surface is bounded. |
| SDK README defaults to `python -m pip install -e .`: a dev-install pattern shown as the primary install | Kit README leads with `pip install kuration-ai` from PyPI (once launched) or the source-install fallback. Editable install lives under a separate contributor heading. | None: README organization only. |
| SDK README does not tell you to create a venv first | Kit README leads with venv creation. Every install command assumes an activated venv. | None. |
| `pip install '.[gameplay]'` pulls 34 packages including torch (~150 MB) | Kit does not ship a `gameplay` profile in v0.0.1. Only `timeseries` (small, numpy + scikit-learn + pyarrow). Heavier profiles land in v0.0.2. | Delays gameplay coverage. Users who want SLAM installs must reach for SDK extras directly. |
| SDK's `.describe()` output uses jargon (`execution: reference; backend=unrecorded`) with no glossary | Kit's `Operation.__repr__` and `run(...).describe()` add friendlier field names for the fields kit exposes. SDK output is passed through when kit does not have a rewrite. | Requires kit to know which SDK fields deserve renaming. |

## Alternative considered and rejected

**File issues on `aintropy-solutions/kuration-sdk` for the SDK to fix.** Rejected for v0.0.1 scope. The SDK's shape is settled (CLAUDE.md principle 4). We can file the issues later as "wrapper implications": but blocking on SDK fixes turns a 2-day ship into a multi-week ship. Every friction in the table above is solvable inside the wrapper today.

## Note for future SDK collaboration

[`future-sdk-issues.md`](../docs/internal/future-sdk-issues.md) tracks these friction items with a per-row verdict (escalate / permanent-wrapper / maybe) and a suggested SDK issue title for each escalation candidate. Early candidates:

- `kur list` with no kind → SDK could show a summary. Wrapper's workaround becomes unnecessary if this lands.
- Operator ↔ method naming convergence → longer-term SDK design conversation.
- Extras-visible-in-registry → could be an SDK descriptor field.

## GTM angle

Every entry in the workaround table is a reason the kit exists. If the SDK smoothed these itself, the kit's value proposition weakens. But the SDK's shape is a settled thing: the kit is the right layer to smooth it, and the workaround list is a concrete answer to prospects asking "what does the wrapper do that the SDK doesn't?"

The pitch answer to "why not just use kuration-sdk directly?": the SDK is a 45-component fluent API with 26 install extras. The kit is a three-operation gallery with one install profile. Same underlying engine, different amount of decision-making required to get to first shards.
