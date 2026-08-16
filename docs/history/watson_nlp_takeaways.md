# IBM Watson-NLP — structural mechanics worth stealing

Phase 0.2 deliverable for issue #5. Source: https://github.com/ibm-self-serve-assets/Watson-NLP (fetched 2026-08-11).

Copy nothing — reference by mechanic only.

## Worth stealing (structural)

1. **Above-fold hook says what the repo *enables*, not what tech it uses.** "Quickly embed Watson NLP in your own applications" comes before any framework mention. First move: outcome, not stack.

2. **"Outline" section maps directly to folders.** No prose narrative before the folder index. Reader can navigate to the asset that matches their job in one click, without reading past sales copy.

3. **Nested self-contained asset folders.** Each use case (Emotion Classification, Sentiment Analysis, ...) is one folder containing tutorials, notebooks, and datasets together. Reader can open one folder and have everything they need.

4. **Cross-links to authoritative external docs.** IBM TechZone, Developer tutorials, and API references are linked rather than restated in-repo. Reduces docs drift.

5. **Topics/tags for cross-repo discoverability.** GitHub topic system used for 7 tags. Cross-repo search finds it without SEO copy in the README.

## Worth avoiding

6. **No root-level install / prerequisites section.** They defer install to individual sub-READMEs. A fresh dev landing at the top has to click twice before they know what setup they need. Real onboarding friction.

7. **No inline code examples in the top-level README.** Every code block is behind a notebook link. Reader can't skim the "what does this do" answer without opening Jupyter first.

8. **Flat link structure gets dense with 20+ items.** No visual hierarchy indicating "start here." A fresh dev sees a wall of hyperlinks and has no way to pick their entry point.

9. **No badges / TOC / "get started in 5 minutes" callout.** Professional-perception bar not met by GitHub conventions the ecosystem expects (build status, coverage, quickstart).

10. **Team/legal disclaimers embedded high in the README.** Governance and creator credits appear before the value proposition. Adds friction to skimming.

## How this informs the kuration-kit README (design/07)

Take: outcome-first above-fold hook, outline-to-folders index, self-contained asset folders (Watson's #1-#3).

Avoid: install-deferred-to-sub-READMEs, no-code-in-README, unstyled link wall, missing badges (Watson's #6-#10).

Concrete implications for our design/07:
- README opens with a runnable code block (fixes #7)
- Explicit `## Install` section above the gallery (fixes #6)
- Gallery links to `examples/<slug>/` folders each with their own README (steals #3)
- Badges: version / license / python-version at top (fixes #9)
- No governance/team text above the fold (fixes #10)
