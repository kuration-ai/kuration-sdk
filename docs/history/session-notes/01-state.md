# 01 — State snapshot

Timestamp: end of session 2026-08-11, network to `api.github.com` unreachable.

## What exists on disk (this repo)

```
kuration-kit/
├── .gitignore                         # from gh repo create
├── LICENSE                            # Apache-2.0, from gh repo create
├── README.md                          # rewritten with real use patterns
├── CLAUDE.md                          # settled product context
├── DECISIONS.md                       # calls made without escalating
├── pyproject.toml                     # kuration-kit meta-package
├── src/
│   └── kuration_kit/
│       ├── __init__.py
│       ├── operation.py               # Operation protocol + run() helper
│       └── operations/
│           ├── __init__.py            # catalog
│           ├── balance.py             # BalancedSelector
│           └── fault_filter.py        # FaultThresholdFilter
├── examples/
│   ├── one_operation.py               # standalone use, verified running
│   └── chained.py                     # 3 call styles, verified identical
└── session-notes/                     # this dir
```

Both examples verified running against a kuration-sdk source install in the shared venv at `~/work/aintropy/kuration-sdk/.venv/`. Nothing pushed to remote.

## What exists on GitHub

- Repo `kuration-ai/kuration-kit` (private, created earlier this session)
- Issue #1: "scope check — kuration-sdk onboarding wrapper — does my understanding match?" (open, tagged Kunal + Nirmit + Eric)
- Issue #1 has one comment: the full walkthrough log (tagged Eric + Kunal)

Nothing else is on GitHub yet.

## What is NOT on GitHub (network-blocked, ready to push when network returns)

- 11 labels (`epic`, `setup`, `phase-0`, `phase-1`, `phase-2`, `phase-3`, `phase-5`, `design-decision`, `packaging`, `deferred`, `blocked`)
- 1 milestone (`v0.0.1 — 2-day ship`)
- 19 issues (drafts in `03-planned-issues.md`, script at `/private/tmp/claude-501/-Users-sahitidharmavaram/6088ed70-327a-49f9-937f-53961b47aa26/scratchpad/create_issues.py`)
- All working-tree files above (never committed, never pushed)

## The blocker

`gh` calls fail with `dial tcp: connect: network is unreachable`. `ping github.com` gets `No route to host`. `curl api.github.com` gets `Could not resolve host`. The Mac has no network route to GitHub right now. This is not a script bug or an auth problem — everything is drafted correctly and ready to execute the moment connectivity returns.

## When network returns — checklist in order

1. `ping -c 1 github.com` to confirm connectivity actually restored.
2. Re-run the issue creation script:
   ```bash
   python /private/tmp/claude-501/-Users-sahitidharmavaram/6088ed70-327a-49f9-937f-53961b47aa26/scratchpad/create_issues.py
   ```
   Creates 11 labels, 1 milestone, 19 issues. Prints each issue URL.
3. Verify with `gh issue list --repo kuration-ai/kuration-kit --limit 30`.
4. Commit the working tree locally (Sahiti drives the git actions per her rule):
   ```bash
   cd ~/work/aintropy/kuration-kit
   git status
   # she runs git add + commit + push herself
   ```
5. Reference each on-disk file in the corresponding issue as needed (e.g. link CLAUDE.md from #3, DECISIONS.md from the epic).
6. Start Phase 0 — work issue #4 (SDK inventory) and #5 (Watson-NLP study) in parallel.

## Non-checklist reminders

- Sahiti drives all git actions (`git add`, `commit`, `push`, `branch`) per her `feedback_no_git_actions.md` rule. Do not run these on her behalf without asking.
- Nothing gets posted / commented / emailed without explicit permission per `feedback_no_unauthorized_messages.md`.
- Kunal reviews Sahiti's PRs — self-review with Claude before merge per `feedback_self_review_before_merge.md`.
