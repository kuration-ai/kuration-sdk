# 04 — DX walkthrough log

The full fresh-external-dev walkthrough of the Kuration SDK, run by Sahiti on her machine on 2026-08-08. Also posted verbatim as a comment on issue #1 of this repo.

Steps: install, quickstart, discovery, custom operator, extras install.

---

hey @esongaintropy @kunal-savvy, quick update. got the sdk installed on my mac and walked through it as if i were a fresh external dev landing on the repo today. ran through install, the quickstart, component discovery via `kur list`, the custom operator example, and installing an extra. everything worked end to end, the quickstart runs clean, and the custom operator example wires up a pipeline that actually curates. i havent authored my own pipeline against my own data yet, thats the next thing im gonna do. below is the full walkthrough log with a couple of small onboarding frictions i noticed as a fresh dev, nothing broken, mostly small things a wrapper could smooth over so a first-time dev doesnt have to reason through them cold.

---

## step 1 — install

ran the venv + `pip install -e .` + `kuration.__version__` + `kur --help` sequence. install finished in about 5 seconds, only dep pulled was numpy 2.5.1, version came back `0.1.0rc5.dev0`, and `kur --help` listed 20 subcommands (including `slam` shown as "provided by an installed plugin", so the plugin discovery hook does surface in help output).

frictions i noticed: the readme jumps straight to `pip install -e .` with zero mention of creating a venv first, and on a modern mac with homebrew python youd hit PEP 668's `externally-managed-environment` error and not know what to do. pip also nags with a "new release of pip is available" notice right after the success message, which as a fresh dev is a "wait should i upgrade first?" moment. after the install nothing tells you which extras came in (none, just core) or which are still available, so you dont know what surface area you actually have. and `kur --help` marks `slam` as "provided by an installed plugin" without naming the plugin, so a fresh dev might wonder if they accidentally installed something.

## step 2 — quickstart

ran `python examples/quickstart.py --out /tmp/kur-sahiti-out` followed by `kur-inspect /tmp/kur-sahiti-out --verify`. both ran essentially instantly (subsecond). the quickstart built the whole synthetic dataset (288 rows across 6 sessions, 42 candidate windows, 12 selected, 12 shards in train/test), and `kur-inspect` came back with `verification: PASS; 12 shards decoded`. no errors, no warnings.

the preflight block at the top of the quickstart is actually really good. it tells you the source adapter, schema decision, window setup, selection method, execution mode, and shard/window estimates before anything runs. that upfront transparency is the kind of thing a lot of ml-data tools skip and then leave the user guessing.

small things i noticed as a fresh dev: `execution: reference; backend=unrecorded` uses two terms i wouldnt understand without googling. and the `_present()/bool` modality in the inspect output shows up without any explanation of what it is or why its added. neither is a real blocker, just polish.

## step 3 — component discovery

ran `kur list`, `kur list operators`, `kur describe operator uniform-bins`, and `kur list methods` back to back. this was the first spot where things actually felt rough.

`kur list` alone errored with `error: the following arguments are required: kind`. the error dump lists the 11 kinds ({adapters,analyzers,containers,encoders,filters,methods,metrics,operators,presets,scorers,transforms}) but doesnt suggest a starting point. my instinct was that `kur list` should just show everything, or at least tell me "run `kur list operators` to see operators". neither happened, so as a fresh dev youre stuck picking one of 11 kinds cold with no idea which one you actually want.

`kur list operators` worked and returned 6 namespaced ops (`aintropy/pixel-shift`, `aintropy/slam-quality`, etc). `kur list methods` also worked and returned 11 methods but with bare names like `uniform-bins`, `k-center`, `dsir` — no namespace. thats an inconsistency i wasnt expecting. the design docs describe operators and methods as being registered through the same registry mechanism, but their naming convention is different at the surface. small thing but confusing.

`kur describe operator uniform-bins` errored with `unknown operator 'uniform-bins'; registered: [aintropy/action-state-consistency, ...]`. the error is technically correct (uniform-bins isnt an operator, its a method) but doesnt say "did you mean `kur describe method uniform-bins`?". thats the kind of hint that saves a fresh dev 5 minutes of confused re-typing, since `uniform-bins` clearly exists somewhere in the registry.

also worth flagging: `kur list methods` gives just names, no descriptions. `kur list operators` gives names + descriptions. uneven output polish across the two.

## step 4 — build your own operator

ran `python examples/05_custom_operators.py`, then cat'd the example to actually read it.

script ran instantly and showed both patterns working: a compositional operator (`signal_diversity` — chains an existing measure + bin under one reusable name) and a transform-backed operator (`range_diversity` — computes a genuinely new feature in bounded batches, then bins it). the print at the end confirms the registry knows about the new operator, and that both `.range_diversity(...)` and `.example__range_diversity(...)` spellings resolve.

honestly, the "add your own" surface is really clean. its ~10 lines of code per custom op, the header docstring in the example clearly explains both patterns before the code shows them, parameters are validated via JSON Schema so `kur describe` can print the contract without importing anything, and you get discoverability for free once you register. a fresh dev could copy this example, swap names, and have a custom operator working without touching any other docs.

small things a fresh dev might puzzle over: the double-underscore in `.example__range_diversity(...)` for the namespace-qualified spelling is a bit unusual (why `__` and not `.`?), and the JSON Schema `PARAMETERS` dict feels heavy for someone writing their very first custom op — a "just get me going" no-schema version could be a nice on-ramp. neither is a blocker, this surface is solid.

## step 5 — install an extra

ran `python -m pip install -e '.[gameplay]'` and then `kur list operators` right after, to see what the extra added.

the install pulled in 34 packages, including some heavy ones: torch 2.13 (111MB), opencv-python 5.0 (48MB), pyav 18 (18MB), plus huggingface_hub, faiss-cpu, torchvision, safetensors, trimesh, sympy, and more. took several minutes end to end (torch alone was about 3 min on my connection). pip also threw a bunch of "Cache entry deserialization failed, entry ignored" warnings along the way — not the sdks fault but noisy to read past.

now heres the thing i wasnt expecting: after all that install work, `kur list operators` returned the exact same 6 operators as before the extra was installed. same list, same order. so as a fresh dev i was left going "wait did the install actually do anything? did anything new get added?".

after digging in, the pattern makes sense: extras dont add new operators to the registry, they add the underlying implementation dependencies (torch, opencv, pyav, faiss, etc) that already-listed operators need to actually run. `aintropy/slam-quality` and `aintropy/pixel-shift` were in `kur list operators` from step 3, before any extra was installed — but they wouldnt have actually EXECUTED without the `[gameplay]` extra installed.

thats a real dev-ux gap. as a fresh dev, if i want to work with slam data, theres nothing in the registry output telling me i need `[gameplay]` (or is it `[video]`? `[torch]`? `[slam]`?) to actually run the slam operators. `kur list operators` should probably annotate each entry with something like "needs `[gameplay]` extra" or fail with a clean install-hint error if you try to use one without its deps. this is exactly the kind of confusion install profiles in a wrapper would solve — one command like `pip install kuration-kit[slam]` to pull "everything needed for slam work" without a dev having to reason through 26 extras and guess which combo they need.

---

tldr the sdks core is really solid, all the frictions cluster in the pre-flight surface (which extras to install, which ops need what, where to start looking) and thats exactly the layer i think an onboarding wrapper on top can smooth over. gonna go actually use the sdk for real now, then spec the wrapper.
