# DECISIONS.md: calls made without escalating

Per the standing rule in the kickoff prompt: any call I make on a reversible decision
without pausing to escalate lands here, so it can be audited without interrupting.
Newest first.

## 2026-08-12 (post-ship rename)

### Repo renamed to `kuration-ai/python-sdk`
Discovered during a final QA pass that the repo was renamed from `kuration-ai/kit` to `kuration-ai/python-sdk` (canonical name confirmed via `gh repo view kuration-ai/python-sdk`). Presumably done by Sahiti or Kunal after issue #28 (the naming ping) settled: the new name matches the openai/openai-python and anthropics/anthropic-sdk-python convention that earlier research called out as standard for company Python SDKs.

Updated in one atomic pass:
- README badge URL, issue-tracker link
- CLAUDE.md ship-target line
- CONTRIBUTING.md git-clone URL and `cd` step
- Local git remote (`git remote set-url origin https://github.com/kuration-ai/python-sdk.git`)

Old `kuration-ai/kit` URLs still work via GitHub's automatic redirect. Historical audit-trail prose in earlier DECISIONS.md entries mentioning `kuration-ai/kit` preserved verbatim as historical fact.

Also caught during the same QA pass:
- README's "51-test smoke suite" was outdated (actual count is 52 after adding the `kai` bare-invocation test at `f0d5ffe`). Fixed in README + CLAUDE.md.
- README's phrase "the full API is documented under the SDK" (in the Fluent pipeline methods section) was a broken promise: the underlying `kuration-sdk` docs are private. Dropped the misleading clause.

## 2026-08-11 (post-ship polish)

### Moved internal docs to `docs/internal/`
Sahiti asked for the repo's visual appeal to be cleaned up. Top-level file list had six markdown files; three were customer-facing (`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`) and three were internal audit / escalation docs (`DECISIONS.md`, `future-sdk-issues.md`, plus `CLAUDE.md` which stays at root by Claude Code convention).

Moved via `git mv`:
- `DECISIONS.md` → `docs/internal/DECISIONS.md`
- `future-sdk-issues.md` → `docs/internal/future-sdk-issues.md`

Cross-references updated in `CLAUDE.md`, `CONTRIBUTING.md`, and `design/05_sdk_workarounds.md`. Historical prose (including entries inside this file that mention the old paths) preserved verbatim as audit-trail fact.

### Added `KURATION_SDK_TOKEN` CI secret + green badge
CI workflow was already in place but needed a token to actually run tests (since `kuration-sdk` is a private repo pre-PyPI). Piped Sahiti's existing `gh` CLI token into the repo secret without echoing (`gh auth token | gh secret set KURATION_SDK_TOKEN --repo kuration-ai/kit`). Verified first run at commit `7b875a6` came back green in 51s. Added the tests-passing badge next to Python + License + Status badges at the top of the README.

Security note: the reused `gh` token has more scopes than strictly needed (`gist`, `project`, `read:org`, `repo`, `workflow`). A dedicated PAT scoped to just `repo` would be tighter. Sahiti can rotate at any time via `Settings → Secrets and variables → Actions`.

## 2026-08-11 (v0.0.1 ship completion)

### Renamed repo `kuration-ai/kuration-ai` → `kuration-ai/kit`
Sahiti's direction: doubled "kuration-ai/kuration-ai" URL is awkward from a marketing
perspective: she wants "kuration-ai" to appear only once in the URL customers see.

Considered:
- `kuration-ai/python` (openai/openai-python style): rejected because Kuration AI is
  Python-only per CLAUDE.md non-goals, so a `-python` suffix would misrepresent the
  product as one language among many.
- `kuration-ai/library`: generic, not distinctive.
- `kuration-ai/kit`: short, honest for what the library actually is (an onboarding
  kit + operations catalog). Chosen.

Rename executed via `gh repo rename kit --repo kuration-ai/kuration-ai --yes`.
Local git remote updated. Internal doc URLs (README issue-tracker link, CLAUDE.md
ship-target line, release notes) updated to the new path. Old URLs auto-redirect on
GitHub. The `pip install kuration-ai` package name is unaffected: repo name and
PyPI name are separate namespaces.

### Renamed product to Kuration AI
Sahiti's direction: "call it Kuration AI. not kuration-kit. This is what a client is
going to see."

Full atomic rename:
- Product name in all customer-facing docs: `kuration-kit` → **Kuration AI**
- Repo: `kuration-ai/kuration-kit` → `kuration-ai/kuration-ai` (via `gh repo rename`)
- Python package: `kuration-kit` → `kuration-ai` (pyproject `name`)
- Python module: `kuration_kit` → `kuration_ai` (via `git mv src/`)
- CLI command: `kit` → `kai` (via `[project.scripts]`)
- Install command: `pip install kuration-kit[timeseries]` → `pip install kuration-ai`
  (sklearn + pyarrow are now regular deps instead of extras: no customer-side
  profile-picking)
- README rewritten to Watson-NLP shape: Contents outline first, use cases before
  code, dropped internal Reference section (CLAUDE.md / DECISIONS.md not linked from
  customer view)

Historical prose in earlier DECISIONS.md entries and inside `docs/history/session-notes/`
still references `kuration-kit`. Preserved verbatim as historical fact: retroactively
editing them would corrupt the audit trail.

Design docs (`design/01-05`) retain internal shorthand ("kit", "wrapper") in prose
that describes the architecture. Only user-visible strings: code imports, install
commands, CLI examples: were updated to the new names. Reviewers reading the design
docs get a technically accurate view; customers reading the README get the product
view.

### Archived `aintropy-ai/kuration-kit`
Old repo archived (read-only) via `gh api /repos/aintropy-ai/kuration-kit -X PATCH -f archived=true` after all files, 25 issues, walkthrough comment, 11 custom labels, and the `v0.0.1: 2-day ship` milestone verified intact on `kuration-ai/kuration-kit`. Safety check confirmed the new repo is NOT archived. URLs still resolve on the old repo; issues/comments preserved as read-only. Delete not chosen: archive is safer and reversible.

### Wrote `future-sdk-issues.md`
Closed the internal reference from `design/05_sdk_workarounds.md`. All 8 workaround-table rows carry a verdict (escalate / permanent-wrapper / maybe) and a suggested SDK issue title for the escalation candidates. Doc is the escalation queue, not the trigger: filing happens post-ship at Sahiti's discretion.

### Tagged `v0.0.1` + published GitHub release
First public release marker. Release notes cover install, three-op gallery, three call styles, extension seam, `kit` CLI, non-goals, and the 51-test verification headline.

## 2026-08-11 (evening: cleanup)

### Moved session-notes/ and scratchpad/ under docs/history/
The top-level `ls` now reads as a package layout, not a work journal. Motivation: the
professional-perception bar from CLAUDE.md GTM tenet 2 (the README is the sales page).

- `session-notes/` (9 files) → `docs/history/session-notes/`
- `scratchpad/sdk_operation_inventory.md` → `docs/history/sdk_operation_inventory.md`
- Empty `scratchpad/` directory removed at repo root

Historical prose in earlier DECISIONS.md entries and inside the moved session-notes
files still references the old paths. Those references are preserved verbatim as
historical fact: retroactively editing them would corrupt the audit trail.

Also wrote `docs/history/watson_nlp_takeaways.md`: the Phase 0.2 deliverable that was
referenced in issue #5 but never actually saved to disk during the original session.

### One-off git hygiene lesson
On the gallery-structure commit (`a8b05d4`), `git add -A` swept up the parallel-Claude
terminal's `tests/` work-in-progress from the shared working tree, and my commit
message did not name those files. Tests pass and are useful, but the commit accuracy
suffered. From here on: explicit `git add <path>` per commit, never `-A`.

## 2026-08-11

### Filled the two bracketed slots in the kickoff prompt
- **[PATH OR REPO URL TO THE SDK]** → `~/work/aintropy/kuration-sdk` (local clone) and
  `https://github.com/aintropy-ai/kuration-sdk` (upstream).
- **[OWNER/REPO]** → `kuration-ai/kuration-kit`.

Both unambiguous from context: asking would have been overhead.

### Existing MVP scaffold treated as prior thinking, not authoritative
An earlier MVP scaffold exists in this repo's working tree (`src/kuration_kit/`,
`examples/`, `pyproject.toml`): it was built before the kickoff prompt landed. It
demonstrates the Operation + `run()` shape but does not go through the phase-gated
process the plan requires.

Treated as: reference implementation, not final. Survives if Phase 2 converges on the
same shape. Replaced if Phase 1 use case selection points elsewhere. Not defended.

### Bundled Phase 0 into two issues, not one
Phase 0 has two distinct work streams: SDK inventory (code reading) and Watson-NLP
reference study (public-repo reading). Different acceptance criteria, different skills,
so filed as separate issues (#4 and #5) rather than one issue with two checkboxes.

### All five Phase 2 design decisions kept as separate issues
Even though five design issues is heavy for a 2-day ship, the kickoff prompt is
explicit: "one issue per design decision from Phase 2." Followed the letter. Willing to
merge #7 + #8 if requested.

### Phase 5 build-use-cases as one placeholder issue (#14)
Instead of pre-filing empty use-case issues before Phase 1 picks the set, filed one
placeholder that will split into per-use-case sub-issues after Gate 1.

### CLAUDE.md drafted at the same time as issue #3 was created
The prompt requires CLAUDE.md before Phase 0 starts. Since "go ahead" authorized
execution, drafted the content into `CLAUDE.md` at repo root immediately alongside
issue creation. Edit freely: issue #3 acceptance criteria only require the file to
exist with the three sections; content is starter material.

### Nothing pushed to remote
Working tree only. `git status` will show untracked / modified files; you drive the
commit + push cadence.

### Built `session-notes/` directory to survive the network outage
Network to `api.github.com` went unreachable during batch issue creation. Everything
that would have gone into issues + comments is now on disk under `session-notes/`, so
this state is recoverable across a full session reset. `session-notes/01-state.md` is
the entry point; the rest of the files back it up. When network returns, re-run
`/private/tmp/.../scratchpad/create_issues.py` to actually create the issues.

### Migrated to kuration-ai/kuration-kit (2026-08-11 evening)
Sahiti wanted Kunal to have visibility. He's a member of `kuration-ai` org (not
`aintropy-ai`). Repo transfer failed: she's a member of aintropy-ai, not an owner.
Migration path taken instead:
1. Created `kuration-ai/kuration-kit` (private).
2. Re-filed 20 issues (scope check as #1 + 19 planned as #2–#20) via
   `scratchpad/migrate_to_kuration_ai.py`.
3. Re-posted walkthrough log as a comment on new #1.
4. Recreated 11 labels + `v0.0.1: 2-day ship` milestone.
5. `sed`-replaced `aintropy-ai/kuration-kit` → `kuration-ai/kuration-kit` in every
   `.md` file in the repo.
6. Repointed local origin to new repo, committed everything, pushed to main.

`aintropy-ai/kuration-kit` left in place per Sahiti's instruction until the new repo
is verified 100%. See task #20 for cleanup.

### Added GTM tenets to CLAUDE.md
Sahiti's redirect on 2026-08-11: "think of it from a marketing GTM aspect the whole
time." Added six GTM tenets alongside the seven engineering principles, plus a "who
this is for" section (three personas). Every design doc from Phase 2 now has an
explicit `## GTM angle` section tying the technical decision to which tenets it
serves.

Separately: she flagged that my draft hook ("One install, three commands, real
training shards. Bring your own operation when you need to.") read AI-generated. Saved
a durable memory (`feedback-no-ai-marketing-voice`) to keep future copy specific,
restrained, and free of triple-parallelism / hype-verb / "bring-your-own-X" patterns.

### Phase 2 complete: 5 design docs at `design/01-05_*.md`
- `01_operation_abstraction.md`: `Operation` Protocol, `.apply()` contract, error rules
- `02_composition_model.md`: three composition styles + config-parity invariant
- `03_config_model.md`: tier 1 defaults / tier 2 YAML / tier 3 programmatic
- `04_extension_seam.md`: user-registered Operations, optional `register()`, SDK interop
- `05_sdk_workarounds.md`: 8 SDK frictions × wrapper workarounds × cost
Each doc has an explicit GTM angle section. Ready for Gate 2 review.
