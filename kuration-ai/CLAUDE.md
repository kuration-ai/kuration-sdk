# CLAUDE.md: kuration-ai product framing

Anyone opening a session on this repo should read this file first. It is the settled
context: what we're building, in what priority order, and what we deliberately are not.
The full "why" lives on the epic issue; this is the short, in-repo version.

## Product framing

`kuration-ai` is a companion package to the open-source **Kuration SDK** (`kuration-sdk`),
published separately as its own pip-installable package. It adds three built-in curation
operations plus an extension pattern for writing your own on top of the SDK's
general-purpose pipeline engine.

Earlier drafts of this doc described the underlying engine as "internal" and "kept
invisible to end users." That premise no longer holds: `kuration-sdk` is the public
product going through open-source release, not a hidden implementation detail behind
this package. `kuration-ai` is a convenience layer on top of a SDK end users can and do
use directly — see "No lock-in" below, which already assumed this and is unaffected.

**Target user:** an external developer at another company who wants to build a
reproducible curated training dataset without spending their first hour figuring out
sampling strategy, dedup, or class-imbalance handling from scratch.

**What it does:** ships a small catalog of standalone, configurable operations: each
one a self-contained composition of pipeline primitives for a recognizable job.
Operations compose. The same shape works as instances, as a list, or as a config dict.

**What it is not:** a rewrite of the underlying engine. A plugin marketplace. A hosted
service. A domain-specific vertical. The engine's shape is settled from `kuration-ai`'s
side. `kuration-ai` packages, curates, and smooths the developer experience on top of it.

## Priority-ordered engineering principles

Read top to bottom. Higher-numbered rules yield to lower-numbered ones when they collide.

1. **Progressive disclosure.** One line for tier 1. Three lines for tier 2. Full
   customization at tier 3. A beginner should never see an expert concept in a tier-1 path.
2. **Operation as the unit.** Every capability `kuration-ai` exposes is a standalone,
   configurable Operation. Operations compose. A user can pick one, chain several, or
   drive them from a config.
3. **Config-parity.** The same operations are expressible three ways: as instances,
   as a list of instances, or as a list of config dicts. All three produce identical
   results. If they diverge, that's a bug.
4. **The underlying engine is settled from this side.** Never modify `kuration-sdk` from
   `kuration-ai`. Every workaround for an awkward SDK shape lives inside the wrapper. If tension is real
   enough to warrant an upstream change, it goes in `docs/internal/future-sdk-issues.md` for later.
5. **Copy-pasteable examples.** Every README snippet has been executed against the real
   library before it lands. No aspirational code. No pseudo-code.
6. **One install decision.** `pip install kuration-ai` gets you everything the gallery
   needs. No profile-picking on the customer side. Sub-packages ship when they're
   genuinely optional (large deps, distinct use cases).
7. **Verifiable in one command.** Every use case in the gallery has a single shell
   command that proves it works end-to-end.

## GTM tenets

These sit alongside the engineering principles above. Every product decision: API
shape, error message, README section, gallery entry: needs to survive a "how does this
help us sell" test, not just a "is this technically clean" one.

1. **First shards in 15 minutes.** A cold-start external developer gets to their first
   curated dataset within 15 minutes of landing on the repo. That budget covers install,
   understanding one gallery entry, running it, and seeing verified output. Any gallery
   entry that violates this bar gets cut or fixed, not accommodated.

2. **The README is the sales page.** The above-the-fold Contents outline maps to
   concrete sections. Prose comes before code. Every gallery entry links to a
   self-contained runnable folder. There is no burying the value under an architecture
   overview or a badge parade.

3. **Reproducibility is the differentiator.** `kuration-ai` guarantees deterministic,
   provenance-tracked curation. The customer story: commit your config file, get
   identical shards from the same package versions a year later. No competitor in the
   data-curation space (Snorkel, Ray Data, dbt, LlamaIndex data loaders) offers this as
   a first-class feature. Every doc surface should make it easy to notice.

4. **No lock-in.** A user who outgrows `kuration-ai` can drop into the raw underlying
   Kuration SDK without rewriting anything. Every Operation is transparent about which
   pipeline primitives it calls; `.describe()` shows the actual chain. `kuration-ai` is
   additive over the SDK, never replacing it or hiding it. This is what neutralizes the
   "wrapper trap" objection.

5. **Complexity is opt-in.** Tier 1 is one line and works. Tier 2 is a YAML config.
   Tier 3 is a custom Operation. Same product, three sophistication surfaces. A tier-1
   user must never see a tier-3 concept in their code path.

6. **Every claim is executed.** Every code block, install command, gallery entry, and
   one-liner in the README has been run against the shipped package. Nothing aspirational.
   The bar: paste it into a fresh terminal and it works.

## Who this is for

`kuration-ai` exists for three overlapping personas:

- The ML engineer at a Series B/C startup building a data pipeline from scratch. Wants
  something between a raw pipeline library and a full ML platform.
- The MLOps engineer at an enterprise standing up reproducible dataset curation. Cares
  about audit trails and identical rebuilds across environments.
- The researcher who wants to reproduce someone else's curated dataset without a
  support call. Cares about the config-file-as-artifact story.

## Non-goals for v0.0.1

Deferred to a later cut. Do not re-derive these when they resurface.

- Rewriting or forking the underlying pipeline engine
- A plugin marketplace or discovery service
- Non-Python language wrappers
- A hosted service, web UI, or dashboard
- Domain-specific verticals beyond the shipped operation gallery
- A full pytest + property-based test suite (v0.0.1 uses runnable examples plus a
  52-test smoke suite; property tests follow)
- `kai init <profile>` scaffolder CLI (revisit after PyPI launch)
- PyPI publication (blocked on `kuration-sdk` PyPI launch)

## Ship target

v0.0.1 shipped 2026-08-11 at `kuration-ai/python-sdk` (repo renamed from `kuration-ai/kit`
on 2026-08-12 to match the openai/openai-python convention for company Python SDKs).
All non-deferred issues from the `v0.0.1: 2-day ship` milestone are on `main`.
Cut scope, not polish.

That standalone repo no longer exists: `kuration-ai` was subsequently merged into
`aintropy-solutions/kuration-sdk` as the `kuration-ai/` companion package (see
`MIGRATION_NOTE.md`). Links and clone instructions elsewhere in this package's docs
should point at the monorepo, not `kuration-ai/python-sdk`.
