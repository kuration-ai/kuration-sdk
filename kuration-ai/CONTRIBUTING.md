# Contributing

## Development install

```bash
# kuration-ai depends on kuration-sdk, which is a private repo until PyPI launch.
# kuration-ai lives inside that same repo as a companion package, so one clone
# gets you both:
git clone git@github.com:aintropy-solutions/kuration-sdk.git
cd kuration-sdk
python -m venv .venv
source .venv/bin/activate

pip install -e '.[methods,parquet]'

# Then install kuration-ai editable + dev extras:
cd kuration-ai
pip install -e .[dev]
```

Access to `aintropy-solutions/kuration-sdk` is required. If you do not have it, please contact know@aintropy.ai.

## Run the tests

```bash
pytest tests/ -q
```

Expected: `52 passed`.

## Continuous integration

CI runs on push to `main` and on pull requests. It requires a `KURATION_SDK_TOKEN` repository secret: a personal access token with read access to `aintropy-solutions/kuration-sdk`. Without the secret, the workflow passes green with a notice explaining how to enable it.

## Adding a new operation

1. Write a new class in `src/kuration_ai/operations/`. Follow the shape of `BalancedSelector`:
   - `@dataclass(frozen=True)`
   - `EXTRAS: ClassVar[tuple[str, ...]]` for any optional runtime deps
   - `__post_init__` validation naming the field, the received value, and the valid range
   - `apply(pipeline) -> pipeline` composing SDK primitives
   - `describe() -> str` returning a one-line human summary of the compiled chain
2. Add the class to `src/kuration_ai/operations/__init__.py` and re-export from `src/kuration_ai/__init__.py`.
3. Add a happy-path test in `tests/test_operations.py` with a fixed `seed=1` and an exact selected-count assertion.
4. Add validation tests in `tests/test_validation.py` covering every branch of `__post_init__`.
5. Optional: add a self-contained example folder under `examples/<slug>/` with a runnable script and a README carrying the exact expected output. If you add it, include the folder in the top-level README gallery.
6. Update `CHANGELOG.md` under an `Unreleased` heading.

## Style

- Frozen dataclasses everywhere. Immutability is a design invariant, not a suggestion.
- Error messages should name the field, the received value, and the fix (e.g. `"bins must be >= 2 (got 1). Bin coverage needs at least two buckets."`).
- Every code block in the README must have been executed against the current package. `tests/test_readme_examples.py` verifies this for the hero snippet and the extension seam example.
- No em-dashes as beat separators in customer-facing prose (README, gallery folder READMEs). Use commas, colons, or two sentences instead.

## Design docs

Every non-trivial design decision has a doc under `design/`. Read those before proposing significant changes:

- `design/01_operation_abstraction.md`: the `Operation` Protocol and return contract
- `design/02_composition_model.md`: three call styles and config-parity
- `design/03_config_model.md`: defaults / declarative / programmatic tiers
- `design/04_extension_seam.md`: `register()` contract and user-op guarantees
- `design/05_sdk_workarounds.md`: SDK frictions and the wrapper's response, with escalation candidates in [`docs/internal/future-sdk-issues.md`](./docs/internal/future-sdk-issues.md)

New design decisions land as new files in `design/` before the code change.

## Audit trail

`docs/internal/DECISIONS.md` records every reversible call made without escalating. New entries go at the top, historical prose stays verbatim.
