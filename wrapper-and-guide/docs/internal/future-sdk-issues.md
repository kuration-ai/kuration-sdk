# future-sdk-issues.md: SDK friction we may escalate upstream

Referenced from [`design/05_sdk_workarounds.md`](./design/05_sdk_workarounds.md). Each row of the workaround table in that doc gets an entry here recording whether we want to file it as an issue on [`aintropy-ai/kuration-sdk`](https://github.com/aintropy-ai/kuration-sdk) after our v0.0.1 ships, or whether the wrapper's workaround is the intended permanent home.

**Verdict per entry:**

- **escalate**: worth an SDK issue post-v0.0.1
- **permanent-wrapper**: this workaround lives in the wrapper forever
- **maybe**: depends on user feedback or measurement

## 1. `kur list` requires a `kind` argument

Workaround: `kai list` shows per-kind counts + a suggested follow-up command.

**Verdict:** escalate.

Reasoning: a loud CLI-first affordance would help every SDK user, not just kit users. A summary mode with no args (or an interactive prompt) would land naturally in the SDK CLI.

**Suggested SDK issue title:** `kur list` with no argument: show a summary of counts per kind instead of a usage error

## 2. Extras not visible in `kur list operators`

Workaround: each kit Operation carries `EXTRAS: ClassVar[tuple[str, ...]]`, and `kai list` prints `needs kuration-ai[X]` annotations.

**Verdict:** escalate.

Reasoning: a `requires_extras: list[str]` field on the SDK component descriptor would let the SDK's own `kur list` annotate every registered component, not just kit's. Currently a user can list `aintropy/slam-quality` and then hit a runtime install-hint error only at execution: the friction shows up at the wrong point.

**Suggested SDK issue title:** Add `requires_extras` field to component descriptors so `kur list <kind>` can annotate which optional deps each entry needs at runtime

## 3. Operator (namespaced) vs method (bare) naming inconsistency

Workaround: kit Operations always use PascalCase Python class names, hiding the namespace convention from the user surface.

**Verdict:** maybe.

Reasoning: this is a real SDK design decision, not a bug. Whether `namespace/name` (operators) and bare-name (methods) should converge is worth raising as a design discussion post-v0.0.1, not filing as a bug.

**Suggested SDK discussion title:** Operator/method naming: is bare-name-for-methods vs `namespace/name`-for-operators intentional divergence, or a candidate for convergence?

## 4. Empty JSON-Schema descriptors on base methods

The five base methods (`uniform-bins`, `weighted`, `top-k`, `target-distribution`, `filter`) register with `capabilities` only: no `parameters`. Their kwargs are strategy-level, not method-level.

Workaround: kit re-declares parameters as dataclass fields on each Operation.

**Verdict:** permanent-wrapper.

Reasoning: the SDK's approach is correct for its architecture: base method kwargs live on the strategy. Trying to force them into the descriptor would either duplicate information or split it awkwardly. Wrapper's class-is-schema approach is the right layer for kit-friendly documentation, not the SDK's problem to solve.

## 5. SDK README defaults to editable install

Workaround: Kuration AI README leads with `pip install kuration-ai`; editable install lives under a separate contributor heading.

**Verdict:** escalate: but only after `kuration-sdk` hits PyPI.

Reasoning: pre-PyPI, editable install is the only real path. Once the SDK is on PyPI, the SDK's own README should mirror the same pattern (PyPI install for users, editable install for contributors) to reduce onboarding friction. Filing before PyPI launch would be premature.

**Suggested SDK issue title:** Post-PyPI-launch: README leads with `pip install kuration-sdk`, moves editable install under a Contributor section

## 6. SDK README omits venv step

Workaround: Kuration AI README leads with `python -m venv .venv && source .venv/bin/activate` before any pip command.

**Verdict:** escalate.

Reasoning: trivial one-line fix, high signal to fresh users. On modern macOS with Homebrew Python or PEP 668-managed distros, `pip install` without a venv silently fails or pollutes system Python. Single easiest SDK improvement to land.

**Suggested SDK issue title:** README install section: prepend venv creation step so PEP 668-managed environments accept the install cleanly

## 7. `.[gameplay]` install is ~150MB

Workaround: kit does not ship a `gameplay` profile in v0.0.1. Deferred to v0.0.2.

**Verdict:** permanent-wrapper.

Reasoning: the install size is a function of the underlying deps (torch alone is ~110MB, plus faiss + av + opencv + huggingface_hub). Not something the SDK can meaningfully shrink without dropping features. Wrapper's job is to hide the cost until the user opts in: nothing for the SDK to fix.

## 8. `.describe()` uses jargon like `execution: reference; backend=unrecorded`

Workaround: kit's `Operation.describe()` renames fields for the wrapper's narrow surface. `run(...).describe()` (via `kuration_ai.describe`) returns friendlier labels.

**Verdict:** maybe.

Reasoning: an SDK-side glossary or friendlier default labels would help every user, but this is easier for the wrapper to solve for its narrow surface (three ops) than for the SDK to solve for its 45-component surface. Worth revisiting once we have real user feedback on which specific fields cause confusion.

**Suggested SDK issue title (contingent on user feedback):** `.describe()` output: add friendlier field labels or a glossary for the terms `execution`, `backend`, `reference`, and `unrecorded`

## When to file

Do NOT file any of these on `aintropy-ai/kuration-sdk` while v0.0.1 is in flight. Post-ship, this doc becomes the escalation queue: Sahiti drives which ones actually get filed based on team priority and Eric's SDK roadmap. This is the queue, not the trigger.
