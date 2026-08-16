# 06: Fresh-user DX pain points (internal working log)

Source material for the walkthrough log in `04-dx-walkthrough-log.md`. Originally at `scratchpad/dx_painpoints.md`: copied here for durability.

Persona: fresh external developer at another company landing on the public `aintropy-ai/kuration-sdk` repo, no inside knowledge.

Legend:
- P0 = blocks a fresh user from finishing setup or first successful call
- P1 = costs meaningful time or trust
- P2 = polish, would raise the bar

## Findings

### P1: README install skips venv step
README "Install" section jumps straight to `python -m pip install -e .` with no mention of creating a virtual environment. On modern Python installs (Homebrew, PEP 668 externally-managed) this fails with a cryptic error. On conda/system Python it silently pollutes the base env. Both are a bad first impression on a public launch.
- Fix: prepend `python -m venv .venv && source .venv/bin/activate` (or the platform-appropriate variant) with a one-sentence rationale.
- Bonus: the "Try it in five minutes" block re-does the clone + install. Combining venv step there too.

### P1: Extras list in README is a subset of what pyproject actually ships
README documents ~7 extras (`methods,parquet`, `video`, `stats`, `gameplay`, `ray`, `s3`). `pyproject.toml` actually declares **26**: `methods, parquet, scale, array, cloud, s3, gcs, azure, torch, hf, zarr, video, ray, stats, p2p, cli, pixel-shift, action-state, collision, slam, slam-viz, gameplay, standard-operators, all, dev, docs`. A fresh dev either misses relevant extras or has to grep pyproject to find them.
- Fix: either link out to a canonical extras table in the docs, or generate the extras table from pyproject at doc-build time.
- This is the strongest single signal that the wrapper should ship curated install *profiles* rather than force devs to reason across all 26.

### P1: `kur list` requires a `kind` arg but doesn't hint what kinds exist
`kur list` errors with "the following arguments are required: kind" and then shows the choices `{adapters,analyzers,containers,encoders,filters,methods,metrics,operators,presets,scorers,transforms}`. A fresh dev's first move is `kur list` (matching every CLI convention where bare list = list everything).
- Fix: bare `kur list` prints a small header per kind with counts (`operators: 6, methods: 11, adapters: 12, ...`), then hints "use `kur list <kind>` to expand."

### P2: `kur describe` name format not surfaced
`kur describe operator uniform-bins` errors with the full registered list. Registered names use `namespace/name` format (e.g. `aintropy/pixel-shift`) but core methods are bare (`uniform-bins`). Mixed convention confuses the discovery flow.
- Fix: `kur describe <kind> <name>` shows the naming rule in its error, or `kur list <kind>` includes the full canonical name for copy-paste.

### P2: Version isn't a visible sanity-check anchor
After install, `python -c "import kuration; print(kuration.__version__)"` returns `0.1.0rc5.dev0`. README doesn't say "you should see roughly Xyz." Post-PyPI launch this becomes important: a dev with the wrong pinned version has no anchor.
- Fix: README verify step includes the expected version pattern.

### P2: Editable install advice ships in the "Install" section but a public dev shouldn't need `-e`
Post-PyPI launch, `pip install kuration-sdk` is the right command. `pip install -e .` is a developer/contributor path. Mixing them under one heading confuses public devs into cloning + editable-installing when they don't need to.
- Fix: split "Install" into "Install (PyPI)" and "Install from source (contributors)".

## Non-findings: surfaces that actually work great

- **Quickstart end-to-end.** `python examples/quickstart.py --out /tmp/out` runs clean, produces a full artifact with manifests + commit file + shards, `kur-inspect ... --verify` PASSes. Zero errors, zero warnings. Real numbers: 6 sessions → 288 rows → 42 candidate windows → 12 selected → 12 shards.
- **Custom operator surface.** `@register_operator` + `@register_batch_transform` in `examples/05_custom_operators.py` works exactly as advertised. Both `.range_diversity(...)` and `.example__range_diversity(...)` spellings resolve. Registry describes them. The "build your own operation in 5 lines" claim in DESIGN.md is true.
- **`kur` CLI coverage.** Full lifecycle covered: `infer`, `init`, `preview`, `validate`, `explain`, `curate`, `experiment`, `report`, `inspect`, `stats`, `check`, `diff`, `export`, `import`, `list`. Shell-completion available. This is genuinely one of the best-designed data-tool CLIs I have poked at.
- **Extras install stacking.** `pip install -e '.[methods,parquet]'` on top of a plain install pulls in `scikit-learn, scipy, pyarrow, narwhals, joblib, threadpoolctl` with no version conflicts. Clean.
- **Plugin template.** `plugins/_template` layout is minimal and self-explanatory: `pyproject.toml`, `README.md`, `src/<pkg>/`, `tests/`. Entry-point key documented. `scripts/check_plugin.py` for isolated validation.

## Wrapper implications

The pain points cluster in **one narrow slice**: the *first 15 minutes*: install choice friction, discovery friction, README-vs-reality gap. Nothing structural is broken.

That means the "wrapper" is smaller and cleaner than first thought:

1. **A `kuration-kit` meta-package** with curated install profiles: `kit[robotics]`, `kit[gaming]`, `kit[timeseries]`, `kit[video-ml]`. Each maps to a hand-picked bundle of SDK extras so a dev never types `[methods,parquet,torch,hf,video]` themselves.
2. **A starter template repo** cloneable via `git clone` (or `uv init --template` if we go that far): pyproject, one working `quickstart.py`, one custom-operator example, one plugin skeleton, README that walks all three.
3. **A `kit init <profile>` CLI** (optional, only if 1+2 aren't enough) that scaffolds a project dir + installs the right extras interactively.

Priority: 1 + 2 first. 3 is a nice-to-have if the install-choice friction persists after 1.
