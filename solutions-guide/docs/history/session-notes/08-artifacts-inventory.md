# 08 — Artifacts inventory

Every file, URL, and artifact created during this session and where it lives.

## On disk — repo working tree (this repo)

| Path | What | Committed? | Pushed? |
|---|---|---|---|
| `README.md` | Product-facing README with 4 use patterns | No | No |
| `CLAUDE.md` | Settled product context | No | No |
| `DECISIONS.md` | Audit trail of unescalated calls | No | No |
| `pyproject.toml` | kuration-kit meta-package + install profiles | No | No |
| `src/kuration_kit/__init__.py` | Public API exports | No | No |
| `src/kuration_kit/operation.py` | Operation Protocol + `run()` helper | No | No |
| `src/kuration_kit/operations/__init__.py` | Catalog | No | No |
| `src/kuration_kit/operations/balance.py` | `BalancedSelector` | No | No |
| `src/kuration_kit/operations/fault_filter.py` | `FaultThresholdFilter` | No | No |
| `examples/one_operation.py` | Standalone use example, verified running | No | No |
| `examples/chained.py` | 3 call styles, verified identical | No | No |
| `session-notes/README.md` | Index of session notes | No | No |
| `session-notes/01-state.md` | State snapshot + when-network-returns checklist | No | No |
| `session-notes/02-kickoff-prompt.md` | The 5-phase operating plan verbatim | No | No |
| `session-notes/03-planned-issues.md` | 19 issue drafts + labels + milestone | No | No |
| `session-notes/04-dx-walkthrough-log.md` | The walkthrough log posted as issue #1 comment | No | No |
| `session-notes/05-issue-1-scope-check.md` | Body of issue #1 | No | No |
| `session-notes/06-pain-points.md` | Internal DX pain-points inventory | No | No |
| `session-notes/07-conversation-summary.md` | Session narrative | No | No |
| `session-notes/08-artifacts-inventory.md` | This file | No | No |

## On disk — scratchpad (session-scoped, ephemeral)

| Path | What |
|---|---|
| `/private/tmp/claude-501/-Users-sahitidharmavaram/6088ed70-327a-49f9-937f-53961b47aa26/scratchpad/create_issues.py` | Idempotent script that creates labels + milestone + 19 issues on `kuration-ai/kuration-kit`. Re-runnable when network returns. |
| `.../scratchpad/dx_painpoints.md` | Original DX pain-points draft (mirrored to `session-notes/06-pain-points.md`) |
| `.../scratchpad/issue1_walkthrough_comment.md` | Original walkthrough comment draft (mirrored to `session-notes/04-dx-walkthrough-log.md`) |
| `.../scratchpad/task_overview_draft.md` | Original scope-check body draft (mirrored to `session-notes/05-issue-1-scope-check.md`) |

## On GitHub

| URL | What |
|---|---|
| https://github.com/kuration-ai/kuration-kit | Private repo created 2026-08-08 |
| https://github.com/kuration-ai/kuration-kit/issues/1 | Scope-check issue (open, tagged Kunal + Nirmit + Eric) |
| https://github.com/kuration-ai/kuration-kit/issues/1#issuecomment-5228328576 | Full walkthrough log comment (tagged Eric + Kunal) |

## Not yet on GitHub — pending network

- 11 labels: `epic`, `setup`, `phase-0`, `phase-1`, `phase-2`, `phase-3`, `phase-5`, `design-decision`, `packaging`, `deferred`, `blocked`
- 1 milestone: `v0.0.1 — 2-day ship`
- 19 issues (details in `03-planned-issues.md`)
- Any commit of the working tree above

## Referenced but not created

| Repo | URL | Referenced from |
|---|---|---|
| kuration-sdk (upstream) | https://github.com/aintropy-ai/kuration-sdk | Phase 0 issue #4 depends on |
| kuration-sdk (local clone) | `~/work/aintropy/kuration-sdk` | Where I read source for the walkthrough |
| IBM Watson-NLP | https://github.com/ibm-self-serve-assets/Watson-NLP | Phase 0 issue #5 depends on |

## Task list (in-session tool)

| # | Status | Description |
|---|---|---|
| 1 | completed | Fresh-user DX test of Kuration SDK |
| 2 | in progress | MVP wrapper scaffold — iterate based on shape check |
| 3 | completed | Draft bootstrapping GH issue with to-do list for the wrapper |
| 4 | completed | Grow issue #1 walkthrough log and post as comment when done |
