# 05 — Issue #1 body (scope check)

Filed on 2026-08-08 as issue #1 of `kuration-ai/kuration-kit`. This is the scope-check post that opened the whole engagement. The walkthrough log in `04-dx-walkthrough-log.md` was later posted as a comment on this issue.

Live URL: https://github.com/kuration-ai/kuration-kit/issues/1

Title: `scope check: kuration-sdk onboarding wrapper — does my understanding match?`

---

hey @kunal-savvy @nirmit-desai-aintropy @esongaintropy, quick sanity check before i actually start building.

my understanding of the task: package the kuration-sdk as a wrapper so external devs from other companies have a smooth experience getting started with it. the sdk itself is already really modular and the extension surface is clean, so this isnt about reshaping the api. its the onboarding + dev-ux layer on top. specifically, making it easy to install just what you actually need (theres 26 extras in pyproject rn, thats a lot to reason through cold), easy to pick up and use the existing operations, and easy to add/edit/extend your own operators without needing to ping the sdk author every time.

step 0, which im doing rn: testing the sdk end to end as if i were a fresh external dev landing on the repo. install, quickstart, custom operator, plugin path. logging every friction as i go and filing anything big as a bug on kuration-sdk. small ones so far, readme doesnt tell you to create a venv before install, the extras list in the readme is a subset of whats actually in pyproject (7 in readme vs 26 in pyproject), and `kur list` requires a `kind` arg without hinting what kinds exist upfront. nothing dealbreaking though, the sdk works out of the box and the quickstart runs clean end to end.

after step 0 im drafting a spec for what the wrapper actually is (leaning toward a small starter package + template repo + curated install profiles like `kit[robotics]` / `kit[gaming]` / `kit[timeseries]` so devs dont have to reason through all 26 extras) and opening an issue with a task list.

before that i wanted to check the framing above lines up with what you had in mind. lmk if im missing scope or misreading the intent anywhere.
