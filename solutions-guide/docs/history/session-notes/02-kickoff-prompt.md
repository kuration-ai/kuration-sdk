# 02 — Kickoff prompt (verbatim)

Sahiti pasted this on 2026-08-11 mid-session as the formal operating plan. Treated as settled — followed exactly. Brackets filled in as recorded in `DECISIONS.md`.

---

# Kickoff prompt — paste into Claude Code

> Run this from the repo root, with `CLAUDE.md` already in place and `gh` authenticated.
> Fill the two bracketed slots before pasting.

---

Read `CLAUDE.md` first. It contains the product framing, the priority order of our
principles, and our non-goals. Treat it as settled and work within it.

Act as the **staff engineer owning this** — not as an assistant waiting for
instructions. That means you form opinions, you make calls on reversible decisions,
you tell me when you think I'm wrong, and you escalate only what genuinely needs me.

We ship in 1–2 days. Work in the phases below and **stop at each gate** so I can
redirect cheaply. Do not run ahead into implementation.

---

### Phase 0 — Ground yourself (no output yet beyond a summary)

1. Read the Kuration SDK source at **`~/work/aintropy/kuration-sdk`** (upstream: `https://github.com/aintropy-ai/kuration-sdk`). Inventory
   **every operation it exposes**: name, what it actually does, its inputs and outputs,
   its parameters and which ones matter, its dependencies, and how expensive it is to
   run. I have not given you this list on purpose — derive it from the code, since the
   code is the truth and my memory of it isn't.
2. Fetch `https://github.com/ibm-self-serve-assets/Watson-NLP` and read the top-level
   README plus two or three asset folders. Identify the *mechanics*: how a reader is
   moved from landing to running code, how each asset folder is made self-sufficient,
   what promises are made up front. Copy nothing.

**Gate 0 — show me:** the operation inventory as a table, plus 5–8 bullets on what the
reference repo does structurally that's worth stealing (and anything it does badly that
we should avoid). Flag any operation whose purpose you couldn't confidently determine
from the code.

---

### Phase 1 — Pick the use cases

From the operation inventory, propose **4–6 candidate use cases**. A use case is a
recognizable job someone already has, not a demo of a feature. For each one give me:

- the one-line problem statement, in the user's words, not ours
- which operations it composes, and in what order
- roughly how many lines of user-facing code it takes (if it's a lot, that's a signal
  our API is wrong — say so)
- whether it's genuinely runnable in this repo, and what data or credentials it needs
- what it *proves* about the SDK that the other use cases don't

Then rank them and recommend which make the cut for a 1–2 day ship. Bias toward ones
that are both immediately recognizable and cheap to build. Say plainly which you'd cut.

**Gate 1 — I pick the final set before you design around them.**

---

### Phase 2 — Design the wrapper API and config model

Design against the chosen use cases, not in the abstract. Deliver:

- The **operation abstraction**: the uniform interface every operation conforms to, and
  the return contract.
- The **composition model**: how a user picks operations, orders them, and runs them.
  Show the actual call sites.
- The **configuration model**: the defaults → declarative → programmatic hierarchy from
  `CLAUDE.md`, made concrete. Show a real config file and the equivalent code.
- The **extension seam**: the contract for a user registering their own operation, plus
  one worked example. Keep this minimal — we are not designing their operation.
- **Tier 1 / 2 / 3 snippets side by side**, so we can see whether progressive disclosure
  actually holds or whether Tier 1 leaks Tier 3's concepts.
- Any place the existing SDK's shape fights this design, and your recommended
  workaround **inside the wrapper** (do not change the SDK).

Write it as a short design doc — call sites and code shapes, not prose about
architecture. Where you had a real choice, name the alternative and why you rejected it.

**Gate 2 — design review with me.**

---

### Phase 3 — Repo and packaging plan

Propose the full repo layout and the README structure: exact section order, what goes
above the fold, the hero code block verbatim, and how the use case gallery is organized
so each entry stands alone. Include the naming — package name, module names, the primary
entry-point function — since names are most of what "intuitive" means in practice.

**Gate 3 — quick sign-off, then straight into issues.**

---

### Phase 4 — Write everything down as GitHub issues

Create issues in **`kuration-ai/kuration-kit`** using `gh`. **Show me the full list of titles and bodies
before creating any of them.** Then:

- One **`epic`** issue that states the product framing, the priority-ordered principles,
  and the non-goals. This is the canonical "why," so anyone (including you in a fresh
  session) can be brought up to speed from it alone.
- One issue per **design decision** from Phase 2, each recording the decision, the
  alternatives considered, and the reasoning.
- One issue per **use case**, with acceptance criteria including "runs end-to-end and
  produces the documented output."
- Issues for the **packaging work**: README, gallery structure, naming, error messages,
  output formatting. These are features. Do not bundle them into a "docs" catch-all.
- Anything deliberately **deferred**, labelled as such, so we stop rediscovering it.

Use labels and a single milestone for the 1–2 day ship. Cross-link dependencies.
Write bodies for someone with zero conversational context.

---

### Phase 5 — Build

Work the milestone in dependency order. Build the use cases and the packaging in the
same pass — a use case is not done until its README entry exists and its code has been
executed successfully.

Rules while building:
- **Run every example.** Never commit a snippet you haven't executed.
- If implementation reveals the design was wrong, stop, tell me, and update the relevant
  issue. Don't silently diverge from an issue.
- If we're going to miss the window, tell me early and propose what to cut. Cut scope,
  not polish.

---

### Standing instructions

- Ask before creating or closing issues, force-pushing, or touching the SDK repo.
- Keep a running `DECISIONS.md` for calls you made without me, so I can audit rather
  than be interrupted.
- When you're unsure whether something is convenient enough, assume it isn't, and say so.
