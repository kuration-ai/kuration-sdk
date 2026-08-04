## Problem

`kuration-sdk`'s self-hosted ACA runner (`github-runner-kuration-sdk`) uses Azure Container Apps'
built-in KEDA `github-runner` scale rule to decide how many runner replicas to spin up. Every
poll (every 30s), it has to call GitHub's `/actions/runs/{run_id}/jobs` endpoint for each
currently-queued run, to see which jobs are waiting and match our runner labels.

`ci.yml` is a single workflow file that fans out to ~31 jobs on every push/PR via matrix
strategies (`test`: 4 Python versions, `optional-integrations`: 10 extras x 2 dependency sets,
plus ~7 singleton jobs). Because it's one workflow file, GitHub creates exactly **one run** per
push, and that run's job list is what the scaler has to enumerate in a single API call.

GitHub's response time for that endpoint scales with job count, and with ~31 jobs it's
consistently slow enough (~4s observed) to exceed the scaler's hard-coded HTTP client timeout
(not configurable — confirmed against KEDA's own scaler docs). When that one call times out, the
*entire* poll's scaling decision falls back to zero, so **no jobs scale up at all** until that
run leaves the queue.

We already applied `enableEtags=true` to the scale rule (makes steady-state polling cheaper via
conditional requests), but that doesn't help the first, uncached call for a new run — it can
still time out.

## Fix: split `ci.yml` into multiple workflow files

Splitting the single workflow into several smaller ones means each push creates multiple
independent runs, each with a much smaller job list — faster for GitHub to respond to, safely
under the scaler's timeout. Total test coverage and job count stay identical; only the grouping
into runs changes.

### Proposed split

All files keep the same `on: push` / `on: pull_request` triggers and the same
`concurrency: group: ci-${{ github.workflow }}-...` block as today (adjusted per file so each
workflow cancels its own in-progress runs independently).

| New file | Jobs moved in | Job count |
|---|---|---|
| `ci-versions.yml` | `test` (Python 3.11-3.14 matrix) | 4 |
| `ci-integrations.yml` | `optional-integrations` (10 extras x current) | 10 |
| `ci-integrations-minimum.yml` | `optional-integrations` (10 extras x minimum) | 10 |
| `ci-core.yml` | `minimum-dependencies`, `core-only`, `docs`, `dependency-audit`, `plugins` | 5 |
| `ci-build.yml` | `build` | 1 |
| `ci.yml` (kept, trimmed) | `notebook-ui` (`ubuntu-latest`), `platform-smoke` (`macos-latest`/`windows-latest`) | 3 (not self-hosted, not part of the bottleneck) |

This splits the current 20-job `optional-integrations` matrix in half by `dependency_set`
(current vs. minimum) since 20 self-hosted jobs in one run is still large enough to risk the same
timeout. If `ci-integrations.yml`/`ci-integrations-minimum.yml` (10 jobs each) still time out in
practice, split further by extras (e.g. two files of 5).

### The catch: `needs:` doesn't work across workflow files

Today, `build` needs `[test, minimum-dependencies, core-only, platform-smoke,
optional-integrations, docs, dependency-audit, plugins]`, and `required-ci` needs everything
including `build`. `needs:` only works *within* a single workflow file, so this in-workflow
gating breaks once the jobs live in separate files.

**Replacement: GitHub branch protection required status checks.** Instead of one aggregator job
(`required-ci`) that GitHub Actions evaluates internally, configure the repo's branch protection
rule (Settings -> Branches -> `main` -> required status checks) to directly require each
individual job's check name (e.g. `Python 3.11`, `Python 3.12`, ..., `Optional integration (zarr,
current)`, `Documentation site`, `Build and wheel smoke test`, etc.) as a required check on PRs
into `main`. GitHub blocks merge until every listed check reports success, regardless of which
workflow file produced it. This is a standard, well-supported pattern for split/matrix CI and
requires no `required-ci`-style job at all.

Two things to carry over explicitly:
- `build` still needs its own dependencies. Since those dependencies (`test`,
  `minimum-dependencies`, etc.) are now in different files, `build` can no longer use `needs:` to
  wait for them within the workflow graph. Simplest option: drop the `needs:` on `build` and rely
  on branch protection to keep it from merging if `build` succeeds but an upstream job failed —
  `build` doesn't functionally depend on their *output* (no artifacts consumed from them today),
  it only exists as a gate. If that's not acceptable, an alternative is a `workflow_run`-triggered
  gate workflow that polls the other workflows' conclusions via the API, but that's meaningfully
  more complex and probably not worth it here.
- Any existing GitHub branch protection rule that currently lists `Required CI` as the sole
  required check needs to be updated to list the new individual per-file check names instead,
  otherwise merges will be blocked forever (no more `Required CI` check will ever run).

### Rollout steps

1. Split `ci.yml` into the files above (copy job blocks verbatim, adjust `concurrency.group` per
   file, drop `needs:` from `build`, delete `required-ci`).
2. Push to a branch, open a draft PR, confirm each new workflow file produces its own run with
   the expected job count, and check Log Analytics
   (`ContainerAppSystemLogs_CL` in workspace `kurious-cicd-logs`, filter
   `JobName_s == 'github-runner-kuration-sdk'`) to confirm `KEDAScalerFailed`/timeout events stop
   appearing and executions actually start (`az containerapp job execution list -g
   kurious-rg-dev -n github-runner-kuration-sdk`).
3. Update branch protection on `main`: remove `Required CI` from required checks, add the
   individual job check names from the new files instead.
4. Merge, watch a couple of real pushes to confirm steady-state behavior.

### Not in scope / already done

- `enableEtags=true` is already applied to the scale rule (`kurious-rg-dev` /
  `github-runner-kuration-sdk`) — no action needed there.
- No change needed to the ACA job, KEDA scaler config, or runner image itself — this is purely a
  `ci.yml` reorganization plus a branch-protection settings change.
