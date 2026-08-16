# 03: Planned issues (drafts, network-blocked)

All 19 issues drafted and ready to create. Network to `api.github.com` was unreachable when the batch create script ran; every `gh issue create` failed with `dial tcp: connect: network is unreachable`. The script at `/private/tmp/claude-501/-Users-sahitidharmavaram/6088ed70-327a-49f9-937f-53961b47aa26/scratchpad/create_issues.py` re-runs the whole batch idempotently when network returns.

## Labels to create (11)

| Name | Color (hex) | Description |
|---|---|---|
| `epic` | `8B5CF6` | North-star product framing |
| `setup` | `6E7681` | Foundational setup work |
| `phase-0` | `3B82F6` | Ground yourself: SDK inventory + reference study |
| `phase-1` | `3B82F6` | Pick the use cases |
| `phase-2` | `3B82F6` | Design the wrapper API and config model |
| `phase-3` | `3B82F6` | Repo and packaging plan |
| `phase-5` | `3B82F6` | Build |
| `design-decision` | `10B981` | One design decision, one issue |
| `packaging` | `F97316` | README, gallery, naming, errors, output |
| `deferred` | `4B5563` | Explicitly deferred out of v0.0.1 |
| `blocked` | `EF4444` | Blocked by another issue or external dependency |

## Milestone to create

`v0.0.1: 2-day ship`: first public-external-dev-ready ship. All non-deferred issues under this milestone.

## Issues

Numbering assumes creation order after existing #1 (scope-check). Numbers 2 through 20 will be assigned in order.

### #2: Epic: kuration-kit v0.0.1: onboarding kit for external devs
Labels: `epic`. Milestone: yes.

```
## What this is
The canonical "why" for kuration-kit. Anyone: a fresh Claude session, a new teammate,
Kunal eight days from now: should be able to open this issue alone and know what we're
building, why, and what it isn't.

## Product framing
kuration-kit is an onboarding kit and pre-built operation catalog on top of
kuration-sdk. Target user: external developers at other companies who want to build
reproducible curated training datasets without spending their first hour figuring out
which of the 26 SDK extras to install, which operators need which extras, or how to
compose the fluent API for a common task.

The wrapper does not reshape the SDK's API. It packages, curates, and smooths the
onboarding surface.

## Priority-ordered principles
Canonical version lives in CLAUDE.md.

1. Progressive disclosure
2. Operation as unit
3. Config-parity
4. SDK is settled
5. Copy-pasteable examples
6. Cost-aware install
7. Verifiable in one command

## Non-goals
- Rewriting or forking kuration-sdk
- A plugin marketplace or discovery service
- Non-Python language wrappers
- A hosted service or web UI
- Domain-specific verticals beyond curated install profiles

## Milestone
v0.0.1: 2-day ship.
```

### #3: Setup: write CLAUDE.md
Labels: `setup`. Milestone: yes. **Note: already written on disk during session; issue exists to track completion.**

```
Write CLAUDE.md at repo root. Matches the epic's product framing + priority-ordered
principles + non-goals sections. Settled context every future session reads first.

Acceptance:
- File at repo root as CLAUDE.md
- Product framing section
- Priority-ordered principles (numbered)
- Non-goals section
- Committed on main
```

### #4: Phase 0: inventory every Kuration SDK operation from source
Labels: `phase-0`. Milestone: yes.

Scope: exhaustive inventory of every operator/metric/scorer/method/filter/transform/analyzer/preset/container/adapter/encoder registered in the SDK. Table in `scratchpad/sdk_operation_inventory.md`. Depends on #3.

### #5: Phase 0: study IBM Watson-NLP repo for structural patterns
Labels: `phase-0`. Milestone: yes.

Scope: fetch `github.com/ibm-self-serve-assets/Watson-NLP`. Read README + 2-3 asset folders. Note mechanics: landing-to-code path, per-asset self-sufficiency, top-level promises. 5-8 bullets in `scratchpad/watson_nlp_takeaways.md`. Copy nothing. Depends on #3.

### #6: Phase 1: propose 4-6 candidate use cases, rank, recommend cuts
Labels: `phase-1`. Milestone: yes.

Scope: 4-6 candidate use cases in the user's words, not ours. Per candidate: problem statement, SDK operations chained, rough LOC of user-facing code, runnability in this repo, what it uniquely proves. Rank. Recommend cuts. Blocks all Phase 2 and Phase 5. Depends on #4.

### #7: Phase 2: design decision: operation abstraction and return contract
Labels: `phase-2`, `design-decision`. Milestone: yes.

Deliverable at `design/01_operation_abstraction.md`. Signature, apply contract, immutability, error contract, one alternative rejected. Depends on #6.

### #8: Phase 2: design decision: composition model
Labels: `phase-2`, `design-decision`. Milestone: yes.

Deliverable at `design/02_composition_model.md`. Chain style, `run()` signature, interaction with mid-pipeline SDK methods, one alternative rejected. Depends on #6, #7.

### #9: Phase 2: design decision: configuration model
Labels: `phase-2`, `design-decision`. Milestone: yes.

Deliverable at `design/03_config_model.md`. Config file shape (YAML/JSON), equivalent programmatic call, override hierarchy, one alternative rejected. Depends on #6, #7.

### #10: Phase 2: design decision: extension seam
Labels: `phase-2`, `design-decision`. Milestone: yes.

Deliverable at `design/04_extension_seam.md`. Minimum user-op contract, registration path, one worked example, interaction with SDK's `@register_operator`. Depends on #6, #7.

### #11: Phase 2: design decision: workarounds where SDK shape fights the wrapper
Labels: `phase-2`, `design-decision`. Milestone: yes.

Deliverable at `design/05_sdk_workarounds.md`. Table: SDK behavior → wrapper workaround → cost. Note upstream-worthy tensions as future SDK issues (not now). Depends on #7, #8, #9, #10.

### #12: Phase 3: repo layout + naming
Labels: `phase-3`, `packaging`. Milestone: yes.

Deliverable at `design/06_repo_layout.md`. Directory tree. Package/module/entry-point naming. One naming alternative per key decision. Depends on Phase 2.

### #13: Phase 3: README structure: section order, hero block, use case gallery
Labels: `phase-3`, `packaging`. Milestone: yes.

Deliverable at `design/07_readme_structure.md`. Section order + reasoning, above-the-fold ~400 words, verbatim hero code block, gallery organization. Depends on #6, #12.

### #14: Phase 5: build the chosen use cases (placeholder)
Labels: `phase-5`. Milestone: yes.

Populated after #6 lands. Each use case becomes a sub-issue: problem statement, code + README entry + runnable example, end-to-end acceptance criterion. Depends on #6, Phase 2.

### #15: Phase 5: install profiles wired to SDK extras
Labels: `phase-5`, `packaging`. Milestone: yes.

Wire kuration-kit's install profiles in pyproject.toml to correct SDK extras. Each profile in pyproject pulls the right SDK extras. `pip install kuration-kit[<profile>]` works. Mapping in README. Depends on #12.

### #16: Phase 5: error messages: bad config produces actionable output
Labels: `phase-5`, `packaging`. Milestone: yes.

Every error message names what went wrong and the fix. Enumerated surfaces: config resolution, unknown operation, missing param, extras not installed. At least one test per surface. Depends on Phase 2.

### #17: Phase 5: output formatting: result printouts, .describe()-style transparency
Labels: `phase-5`, `packaging`. Milestone: yes.

Standardized `__repr__` per operation. `run(...)` prints (or exposes) resolved-but-unrun plan. Result objects have printable summary. Depends on Phase 2.

### #18: Deferred: `kit init <profile>` scaffolder CLI
Labels: `deferred`. Milestone: no.

Bootstraps a project directory with the right extras + starter script. Revisit after PyPI launch. Triggers: multiple external-dev asks, or PyPI launch complete.

### #19: Deferred: full test suite (pytest + property tests)
Labels: `deferred`. Milestone: no.

v0.0.1 uses example scripts as smoke tests. Formal suite is a follow-up. Trigger: v0.0.2 or first external PR.

### #20: Deferred: PyPI publication (blocked on kuration-sdk PyPI launch)
Labels: `deferred`, `blocked`. Milestone: no.

Externally blocked. Trigger: kuration-sdk 0.1 lands on PyPI as `kuration-sdk`.

## Dependency graph (for cross-linking after creation)

```
#3 (CLAUDE.md)
├── blocks #4, #5
#4 (SDK inventory)
├── blocks #6
#6 (use cases)
├── blocks #7, #8, #9, #10, #14
#7 (op abstraction)
├── blocks #8, #9, #10, #11
#8, #9, #10
├── blocks #11
Phase 2 (#7-#11)
├── blocks #12, #16, #17
#12 (layout)
├── blocks #13, #15
```
