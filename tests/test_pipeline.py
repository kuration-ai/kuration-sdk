"""Pipeline facade: input() -> operators -> terminals (src/kuration/core/pipeline.py).

Data is synthesized in-memory with a controlled distribution: source s{i} has hp ~ 25*i
(+ a small j%5 ripple), so filter/bin assertions are checkable. Every test that should
produce output uses on_error="raise" and asserts a non-empty pool (the repo's documented
vacuous-success trap).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from kuration import (
    Registry,
    SchemaSpec,
    Segment,
    bin_segments,
    default_registry,
    iter_shards,
    load_strategy,
    segment_priorities,
    select,
)
from kuration.core.pipeline import Pipeline, input as kur_input

N_SOURCES, N_STEPS = 4, 64


def _rows(n_sources=N_SOURCES, n=N_STEPS):
    """{source_id: rows}; hp mean of source s{i} is 25*i + 2 -> s0,s1 pass hp<=50."""
    return {
        f"s{i}": [{"t": float(j), "hp": 25.0 * i + float(j % 5),
                   "action": float((i + j) % 3),
                   "team": "red" if i % 2 == 0 else "blue"}
                  for j in range(n)]
        for i in range(n_sources)
    }


def _pipe(**kw):
    kw.setdefault("time", "t")
    return kur_input(_rows(), **kw)


# ======================================================================================
# input() + inspectors
# ======================================================================================
def test_input_infers_schema_and_sources():
    p = _pipe()
    assert p.sources == [f"s{i}" for i in range(N_SOURCES)]
    assert "hp" in p.schema.fields and p.schema.fields["hp"].kind == "scalar"
    assert p.schema.fields["team"].kind == "categorical"
    assert "t" not in p.schema.fields          # the time column is not a field
    assert p.schema.sample_hz == pytest.approx(1.0)


def test_input_list_of_rows_single_source():
    p = kur_input(_rows(1)["s0"], time="t").window(16).measure("hp") \
        .organize("hp", bins=2).curate(n=2).options(on_error="raise")
    res = p.collect()
    assert res.all_segments and res.selected


def test_head_and_describe():
    p = _pipe()
    ticks = p.head(3)
    assert len(ticks) == 3 and hasattr(ticks[0], "entities")
    d = p.describe()
    assert set(d) == {"schema", "strategy", "output", "segmenter", "options", "sources",
                      "expectations", "metadata"}
    assert d["output"] is None
    assert d["segmenter"] == {"window": 64, "stride": 64, "unit": "steps"}
    assert d["sources"] == [f"s{i}" for i in range(N_SOURCES)]
    assert d["strategy"]["define"] == {}
    assert d["metadata"] == {}


# ======================================================================================
# immutability
# ======================================================================================
def test_immutability_branching():
    base = _pipe().window(16)
    a = base.measure("hp").organize("hp", bins=2)
    b = base.measure("action")
    da, db, dbase = a.describe(), b.describe(), base.describe()
    assert set(da["strategy"]["define"]) == {"hp"}
    assert set(db["strategy"]["define"]) == {"action"}
    assert dbase["strategy"]["define"] == {}            # base unchanged
    assert "binning" in da["strategy"] and "binning" not in db["strategy"]
    # further chaining on a does not mutate a
    a2 = a.filter("hp", "<=", 50)
    assert "predicates" not in a.describe()["strategy"].get("selection", {})
    assert a2.describe()["strategy"]["selection"]["predicates"] == [
        {"metric": "hp", "op": "<=", "value": 50.0}]


def test_pipeline_metadata_is_namespaced_immutable_and_preserved_by_chaining():
    base = _pipe()
    annotated = base.with_metadata("project", "review", {"owner": "team-a", "ready": True})

    assert base.metadata == {}
    assert annotated.metadata == {
        "project": {"review": {"owner": "team-a", "ready": True}}
    }
    external = annotated.metadata
    external["project"]["review"]["owner"] = "mutated"
    assert annotated.metadata["project"]["review"]["owner"] == "team-a"
    assert annotated.measure("hp").metadata == annotated.metadata


def test_pipeline_metadata_rejects_implicit_replacement_and_non_json_values():
    annotated = _pipe().with_metadata("project", "review", {"version": 1})
    with pytest.raises(ValueError, match="already exists"):
        annotated.with_metadata("project", "review", {"version": 2})
    replaced = annotated.with_metadata(
        "project", "review", {"version": 2}, replace=True
    )
    assert replaced.metadata["project"]["review"] == {"version": 2}
    with pytest.raises(ValueError, match="strict JSON"):
        _pipe().with_metadata("project", "bad", {"value": float("nan")})
    with pytest.raises(ValueError, match="metadata namespace"):
        _pipe().with_metadata("not/a/namespace", "bad", {})


# ======================================================================================
# measure — all four forms + loud errors
# ======================================================================================
def test_measure_bare_name_defaults_to_mean():
    p = _pipe().measure("hp")
    ax = p.schema.entity_axis
    assert p.describe()["strategy"]["define"]["hp"] == {
        "source": f"{ax}.hp", "reduce": "mean"}


def test_measure_reduce_string_with_of():
    p = _pipe().measure("hp_p95", "p95", of="hp")
    ax = p.schema.entity_axis
    assert p.describe()["strategy"]["define"]["hp_p95"] == {
        "source": f"{ax}.hp", "reduce": "p95"}


def test_measure_full_dsl_dict():
    base = _pipe()
    ax = base.schema.entity_axis
    spec = {"source": f"{ax}.hp", "reduce": "max"}
    p = base.measure("hp_max", spec)
    assert p.describe()["strategy"]["define"]["hp_max"] == spec


def test_measure_callable_registers_and_is_referenceable():
    p = _pipe().measure("wlen", lambda w: float(w.T))
    assert "wlen" not in p.describe()["strategy"]["define"]   # callables are not DSL specs
    p2 = p.filter("wlen", ">=", 1)                            # known metric -> no raise
    assert p2._predicates == [{"metric": "wlen", "op": ">=", "value": 1.0}]


def test_measure_unknown_field_loud():
    with pytest.raises(ValueError) as e:
        _pipe().measure("nope")
    assert "nope" in str(e.value) and "fields" in str(e.value)


def test_measure_bad_dsl_dict_loud():
    with pytest.raises(ValueError):
        _pipe().measure("x", {"source": "entity.hp", "reduce": "not_a_reduce"})


def test_measure_duplicate_loud():
    p = _pipe().measure("hp")
    with pytest.raises(ValueError) as e:
        p.measure("hp")
    assert "already declared" in str(e.value)


# ======================================================================================
# filter — op aliases + loud errors
# ======================================================================================
@pytest.mark.parametrize("alias,canon", [
    ("le", "<="), ("LTE", "<="), ("LessOrEqual", "<="), ("leq", "<="), ("<=", "<="),
    ("ge", ">="), ("gte", ">="), ("GreaterOrEqual", ">="), ("geq", ">="), (">=", ">="),
    ("lt", "<"), ("less", "<"), ("<", "<"),
    ("gt", ">"), ("greater", ">"), (">", ">"),
    ("eq", "=="), ("equal", "=="), ("=", "=="), ("==", "=="),
    ("ne", "!="), ("neq", "!="), ("NotEqual", "!="), ("!=", "!="),
])
def test_filter_op_aliases(alias, canon):
    p = _pipe().measure("hp").filter("hp", alias, "50")
    assert p._predicates == [{"metric": "hp", "op": canon, "value": 50.0}]


def test_filter_unknown_op_loud():
    with pytest.raises(ValueError) as e:
        _pipe().measure("hp").filter("hp", "~", 50)
    assert "unknown filter op" in str(e.value) and "aliases" in str(e.value)


def test_filter_unknown_metric_loud():
    with pytest.raises(ValueError) as e:
        _pipe().measure("hp").filter("nope", "<=", 50)
    assert "nope" in str(e.value) and "'hp'" in str(e.value)


@pytest.mark.parametrize("value", ["not-a-number", float("nan"), float("inf"), True])
def test_filter_invalid_value_is_eagerly_loud(value):
    with pytest.raises(ValueError, match="finite number"):
        _pipe().measure("hp").filter("hp", "<=", value)


def test_filter_prepass_entry():
    p = _pipe().measure("hp").filter("dedup", feature_space=["hp"])
    assert p._prepass == [{"type": "dedup", "feature_space": ["hp"]}]
    assert p.describe()["strategy"]["filters"] == [{"type": "dedup",
                                                    "feature_space": ["hp"]}]


def test_filter_prepass_metric_references_are_eagerly_validated():
    with pytest.raises(ValueError, match="undeclared metrics"):
        _pipe().filter("dedup", feature_space=["missing"])


# ======================================================================================
# universal predicates — end-to-end pin + direct unit checks
# ======================================================================================
def test_predicates_prefilter_pool_end_to_end():
    p = (_pipe().window(16).measure("hp").filter("hp", "<=", 50)
         .organize("hp", bins=4).curate(n=100).options(on_error="raise"))
    res = p.collect()
    assert res.all_segments and res.selected
    assert any(s.features["hp"] > 50 for s in res.all_segments)   # pool has failing segs
    assert all(s.features["hp"] <= 50 for s in res.selected)       # ...selection doesn't
    kept_sources = {s.source_id for s in res.selected}
    assert kept_sources == {"s0", "s1"}


_UNIT_SCHEMA = SchemaSpec.from_dict({
    "name": "u", "entity_axis": "e", "fields": {"m": {"kind": "scalar"}}})
_UNIT_STRATEGY = load_strategy({
    "name": "pred", "define": {"m": {"source": "e.m", "reduce": "mean"}},
    "binning": {"axes": [{"metric": "m", "bins": 2}]},
    "selection": {"method": "uniform-bins",
                  "predicates": [{"metric": "m", "op": "<=", "value": 4}]}}, _UNIT_SCHEMA)


def _unit_segments():
    return [Segment(source_id="s", start_idx=4 * i, end_idx=4 * i + 4,
                    time_range=(i, i + 1), features={"m": float(i)}) for i in range(10)]


def test_predicates_drop_segments_in_select():
    segs = _unit_segments()
    bin_segments(segs, _UNIT_STRATEGY.axes)
    sel = select(segs, _UNIT_STRATEGY, target_size=100, seed=0)
    assert sel and len(sel) == 5
    assert all(s.features["m"] <= 4 for s in sel)


def test_predicates_price_neg_inf_in_priorities():
    segs = _unit_segments()
    bin_segments(segs, _UNIT_STRATEGY.axes)
    prio = segment_priorities(segs, _UNIT_STRATEGY, seed=0)
    for i, s in enumerate(segs):
        if s.features["m"] > 4:
            assert prio[i] == float("-inf")
        else:
            assert np.isfinite(prio[i])


# ======================================================================================
# organize
# ======================================================================================
def test_organize_kwarg_and_positional_forms_agree():
    base = _pipe().measure("hp")
    a = base.organize("hp", bins=3)
    b = base.organize("hp", "bin", 3)
    c = base.organize("hp", 3)
    axes = a.describe()["strategy"]["binning"]["axes"]
    assert axes == [{"metric": "hp", "bins": 3}]
    assert b.describe()["strategy"]["binning"]["axes"] == axes
    assert c.describe()["strategy"]["binning"]["axes"] == axes


def test_bin_and_sample_are_preferred_compatible_names():
    old = _pipe().measure("hp").organize("hp", bins=2).curate(n=3)
    new = _pipe().measure("hp").bin("hp", bins=2).sample(n=3)
    assert old.describe()["strategy"] == new.describe()["strategy"]
    assert [op["op"] for op in new._ops[-2:]] == ["bin", "sample"]


def test_bin_warns_on_bins_equal_one():
    """F20: bins=1 was silently accepted even though it makes equal-per-bin selection
    meaningless -- every segment lands in the single bin, same as the already-warned
    constant-axis case."""
    import warnings
    with pytest.warns(UserWarning, match="every segment lands in one bin"):
        p = _pipe().measure("hp").bin("hp", bins=1)
    assert p._axes == [("hp", 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _pipe().measure("hp").bin("hp", bins=2)  # bins=2 raises no warning


def test_organize_unknown_metric_loud():
    with pytest.raises(ValueError) as e:
        _pipe().organize("nope", bins=2)
    assert "nope" in str(e.value)


def test_organize_duplicate_axis_loud():
    p = _pipe().measure("hp").organize("hp", bins=2)
    with pytest.raises(ValueError) as e:
        p.organize("hp", bins=3)
    assert "duplicate" in str(e.value)


# ======================================================================================
# curate
# ======================================================================================
def test_curate_n_hours_mutual_exclusion():
    p = _pipe().measure("hp").organize("hp", bins=2)
    with pytest.raises(ValueError, match="exactly one"):
        p.curate(n=5, hours=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        p.curate()


def test_curate_no_method_resolvable_loud():
    with pytest.raises(ValueError) as e:
        _pipe().measure("hp").curate(n=5)
    msg = str(e.value)
    assert "bin" in msg and "weights" in msg and "method" in msg


@pytest.mark.parametrize("alias,canon", [
    ("equal", "uniform-bins"), ("uniform", "uniform-bins"),
    ("top", "top-k"), ("topk", "top-k"), ("top-k", "top-k"),
    ("target", "target-distribution"),
    ("cluster", "cluster-balanced"), ("facility", "facility-location"),
    ("weighted", "weighted"), ("filter", "filter"),
])
def test_curate_method_aliases(alias, canon):
    p = _pipe().measure("hp")
    if canon == "filter":
        p = p.filter("hp", ">=", 0)
    p = p.organize("hp", bins=2).curate(alias, n=5, weights={"hp": 1.0})
    assert p.describe()["strategy"]["selection"]["method"] == canon


def test_curate_unknown_method_loud():
    with pytest.raises(ValueError) as e:
        _pipe().measure("hp").organize("hp", bins=2).curate("bogus", n=5)
    assert "bogus" in str(e.value)


def test_curate_defaults_uniform_bins_with_axes_and_topk_with_weights():
    with_axes = _pipe().measure("hp").organize("hp", bins=2).curate(n=5)
    assert with_axes.describe()["strategy"]["selection"]["method"] == "uniform-bins"
    with_weights = _pipe().measure("hp").curate(n=5, weights={"hp": 1.0})
    assert with_weights.describe()["strategy"]["selection"]["method"] == "top-k"


def test_curate_references_and_method_params_are_eagerly_validated():
    base = _pipe().measure("hp")
    with pytest.raises(ValueError, match="undeclared metrics"):
        base.curate("top-k", n=1, weights={"missing": 1})
    with pytest.raises(ValueError, match="unknown selection key"):
        base.curate("top-k", n=1, weights={"hp": 1}, typo=2)
    with pytest.raises(ValueError, match="normalize"):
        base.curate("top-k", n=1, weights={"hp": 1}, normalize="bogus")


def test_tag_fraction_is_eagerly_validated():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _pipe().tag("team", min_frac=1.5)


def test_overlay_registry_scorer_is_accepted_and_executed():
    class ConstantScorer:
        needs_buffers = False

        def score(self, segments, ctx):
            return np.ones(len(segments))

    registry = Registry(default_registry)
    registry.register_scorer("test/constant", lambda: ConstantScorer())
    p = (_pipe().using(registry).window(16).score("test/constant", out="quality")
         .curate("top-k", n=2, weights={"quality": 1}).options(on_error="raise"))
    result = p.collect()
    assert result.selected and all(s.features["quality"] == 1 for s in result.selected)


# ======================================================================================
# collect() end-to-end
# ======================================================================================
def test_collect_end_to_end():
    p = (_pipe().window(16).measure("hp").measure("wlen", lambda w: float(w.T))
         .filter("wlen", ">=", 1).organize("hp", bins=4).curate(n=6)
         .options(on_error="raise"))
    res = p.collect()
    assert res.all_segments and len(res.all_segments) == N_SOURCES * (N_STEPS // 16)
    assert len(res.selected) == 6
    assert res.errors == []
    s = res.selected[0]
    assert "hp" in s.features and s.features["wlen"] == 16.0


def test_collect_n_overrides_and_requires_a_count():
    p = _pipe().window(16).measure("hp").organize("hp", bins=2).options(on_error="raise")
    assert len(p.collect(n=3).selected) == 3      # n passed at collect() time
    with pytest.raises(ValueError, match="needs a count"):
        p.collect()


def test_collect_n_none_returns_every_scored_and_filtered_segment():
    # No bin axes, weights, or named method configured at all -- n=None needs none of them.
    p = _pipe().window(16).measure("hp").options(on_error="raise")
    res = p.collect(n=None)
    assert res.selected == res.all_segments
    assert len(res.selected) == N_SOURCES * (N_STEPS // 16)
    # The universal predicate pre-filter still applies under n=None.
    filtered = p.filter("hp", ">=", 0).collect(n=None)
    assert len(filtered.selected) <= len(filtered.all_segments)
    # An explicit n=None overrides a pipeline-level n set via .sample()/.curate().
    sampled = p.organize("hp", bins=2).curate(n=3)
    overridden = sampled.collect(n=None)
    assert overridden.selected == overridden.all_segments
    assert len(overridden.selected) != 3
    # .stats() works the same way it does for a real selection.
    res.stats()


def test_split_on_count_path():
    p = (_pipe().window(16).measure("hp").organize("hp", bins=2).split(test=0.5)
         .curate(n=8).options(on_error="raise"))
    res = p.collect()
    assert res.selected
    assert set(res.splits.values()) == {"train", "test"}
    assert sorted(res.splits) == [f"s{i}" for i in range(N_SOURCES)]


# ======================================================================================
# output() — count path
# ======================================================================================
def test_output_count_path_end_to_end(tmp_path):
    out_dir = tmp_path / "ds"
    p = (_pipe().window(16).measure("hp").organize("hp", bins=4).curate(n=4)
         .options(on_error="raise"))
    out = p.output(str(out_dir))
    assert out["selected"] == 4 and out["shards"]
    assert out["errors"] == []

    man = json.loads((out_dir / "dataset_manifest.json").read_text())
    assert man["count"] == len(man["shards"]) > 0

    cfg = json.loads((out_dir / "config.json").read_text())
    for key in ("schema", "strategy", "output", "pipeline"):
        assert key in cfg, f"config.json missing {key!r}"
    # provenance: the operator chain, in order
    ops = [o["op"] for o in cfg["pipeline"]]
    assert ops == ["input", "window", "measure", "organize", "curate", "options"]
    # categorical field skipped from the default modalities; numeric fields present
    assert "team" not in cfg["output"]["modalities"]
    assert {"hp", "action"} <= set(cfg["output"]["modalities"])
    assert cfg["output"]["container"] == "npz"

    shards = list(iter_shards(str(out_dir), split="train"))
    assert len(shards) == man["count"]
    for arrays in shards:
        assert arrays["hp"].shape == (16,)          # window-sized sub-track, scalar modality
        assert arrays["hp"].dtype == np.float32
        assert arrays["action"].shape == (16,)


def test_output_accepts_a_trailing_slash(tmp_path):
    out_dir = str(tmp_path / "ds") + "/"
    p = (_pipe().window(16).measure("hp").organize("hp", bins=4).curate(n=4)
         .options(on_error="raise"))
    out = p.output(out_dir)
    assert out["selected"] == 4 and out["shards"]
    assert all("//" not in shard["path"] for shard in out["shards"])

    man = json.loads((tmp_path / "ds" / "dataset_manifest.json").read_text())
    assert man["count"] == len(man["shards"]) > 0


def test_output_publishes_pipeline_metadata_with_private_paths_redacted(tmp_path):
    private_path = "/private/customer/session-42.bin"
    out_dir = tmp_path / "metadata"
    pipeline = (
        _pipe()
        .with_metadata("project", "source_note", {"original": private_path})
        .window(16)
        .measure("hp")
        .organize("hp", bins=2)
        .curate(n=2)
        .options(on_error="raise")
    )
    pipeline.output(str(out_dir))

    cfg = json.loads((out_dir / "config.json").read_text())
    published = cfg["metadata"]["project"]["source_note"]["original"]
    assert "customer" not in published
    assert published.startswith("session-42.bin-")


def test_output_include_exclude(tmp_path):
    p = (_pipe().window(16).measure("hp").organize("hp", bins=2).curate(n=2)
         .options(on_error="raise"))
    out = p.output(str(tmp_path / "only_hp"), include=["hp"])
    cfg = json.loads((tmp_path / "only_hp" / "config.json").read_text())
    assert set(cfg["output"]["modalities"]) == {"hp"}
    assert out["shards"]
    with pytest.raises(ValueError, match="include"):
        p.output(str(tmp_path / "bad"), include=["nope"])
    with pytest.raises(ValueError, match="exclude"):
        p.output(str(tmp_path / "bad2"), exclude=["nope"])


def test_output_preset_filters_modalities_instead_of_ignoring_them(tmp_path):
    p = (_pipe().window(16).measure("hp").organize("hp", bins=2).curate(n=2)
         .options(on_error="raise"))
    p.output(str(tmp_path / "preset"), preset="numpy-timeseries",
             categorical="omit", include=["hp"])
    cfg = json.loads((tmp_path / "preset" / "config.json").read_text())
    assert set(cfg["output"]["modalities"]) == {"hp"}
    assert set(cfg["output_preset"]["resolved"]["modalities"]) == {"hp"}


def test_output_rejects_conflicting_spec_and_preset_arguments(tmp_path):
    p = _pipe().measure("hp").curate("top-k", n=1, weights={"hp": 1})
    spec = {"shard_length": 8, "modalities": {
        "hp": {"dtype": "float32", "shape": [], "source": "field:hp"}}}
    with pytest.raises(ValueError, match="direct escape hatch"):
        p.output(tmp_path / "spec", spec=spec, container="hdf5")
    with pytest.raises(ValueError, match="cannot be combined"):
        p.output(tmp_path / "preset", preset="numpy-timeseries",
                 categorical="omit", modalities=spec["modalities"])


def test_output_resume_on_count_path_loud(tmp_path):
    p = (_pipe().window(16).measure("hp").organize("hp", bins=2).curate(n=2)
         .options(resume=True))
    with pytest.raises(ValueError, match="hours path"):
        p.output(str(tmp_path / "x"))


# ======================================================================================
# output() — hours path
# ======================================================================================
def test_output_hours_path_end_to_end(tmp_path):
    out_dir = tmp_path / "dsh"
    # hz infers 1.0 from the numeric "t" column; 0.02 h = 72 s -> two 64 s tracks admitted
    p = (_pipe().window(16).measure("hp").organize("hp", bins=4).curate(hours=0.02)
         .options(on_error="raise"))
    out = p.output(str(out_dir))
    assert out["results"]["shards"] > 0
    assert out["errors"] == []

    cfg = json.loads((out_dir / "config.json").read_text())
    for key in ("schema", "strategy", "output", "pipeline"):
        assert key in cfg, f"config.json missing {key!r}"
    assert [o["op"] for o in cfg["pipeline"]][0] == "input"

    shards = list(iter_shards(str(out_dir), split="train"))
    assert shards
    assert shards[0]["hp"].shape == (64,)           # whole-track shards, shard_length=64


def test_config_split_counts_are_emitting_sources_not_assigned_sources(tmp_path):
    """`n_train_sources`/`n_test_sources` in config.json count sources that actually emitted
    shards, not sources that were assigned a split — pinned on both write paths so the meaning
    can't drift back. (F4: these were previously named `n_train`/`n_test`, which read as a
    window count sitting next to `selected: N`; renamed for clarity, same values.)"""
    def emitters(out_dir):
        man = json.loads((out_dir / "dataset_manifest.json").read_text())
        by_split = {}
        for entry in man["shards"]:
            by_split.setdefault(entry["split"], set()).add(entry["source_id"])
        return by_split

    # ---- count path (_write_selected): one selected segment -> exactly one emitting source
    count_dir = tmp_path / "ds_count_splits"
    p = (_pipe().window(16).measure("hp").organize("hp", bins=4).split(test=0.5)
         .curate(n=1).options(on_error="raise"))
    assigned = p.collect().splits
    assert len(assigned) == N_SOURCES                       # every source was assigned a split
    assert len({s for s, sp in assigned.items() if sp == "train"}) >= 2
    out = p.output(str(count_dir))
    results = out["results"]
    written = emitters(count_dir)
    assert sum(len(sources) for sources in written.values()) == 1
    assert results["n_train_sources"] == len(written.get("train", ()))
    assert results["n_test_sources"] == len(written.get("test", ()))
    assert results["n_train_sources"] + results["n_test_sources"] == 1  # not the 4 assigned
    assert json.loads((count_dir / "config.json").read_text())["results"] == results

    # ---- hours path (write_dataset): the budget admits fewer sources than it assigns
    hours_dir = tmp_path / "ds_hours_splits"
    hp = (_pipe().window(16).measure("hp").organize("hp", bins=4).split(test=0.5)
          .curate(hours=0.02).options(on_error="raise"))
    hout = hp.output(str(hours_dir))
    hresults, hwritten = hout["results"], emitters(hours_dir)
    assert len(hout["split"]) == N_SOURCES
    assert hresults["n_train_sources"] == len(hwritten.get("train", ()))
    assert hresults["n_test_sources"] == len(hwritten.get("test", ()))
    for split in ("train", "test"):
        assigned_count = len({s for s, sp in hout["split"].items() if sp == split})
        assert hresults[f"n_{split}_sources"] < assigned_count  # assigned, emitted nothing


def test_output_hours_path_split(tmp_path):
    out_dir = tmp_path / "dsh_split"
    p = (_pipe().window(16).measure("hp").organize("hp", bins=4).split(test=0.5)
         .curate(hours=0.04).options(on_error="raise"))
    out = p.output(str(out_dir))
    assert out["results"]["shards"] > 0
    assert set(out["split"].values()) == {"train", "test"}
    man = json.loads((out_dir / "dataset_manifest.json").read_text())
    assert {e["split"] for e in man["shards"]} == {"train", "test"}


# ======================================================================================
# execution guards
# ======================================================================================
def test_workers_gt1_on_in_memory_loud():
    p = (_pipe().window(16).measure("hp").organize("hp", bins=2).curate(n=4)
         .options(workers=2))
    with pytest.raises(ValueError, match="workers>1"):
        p.collect()
    with pytest.raises(ValueError, match="workers>1"):
        p.report()


def test_options_unknown_key_loud():
    with pytest.raises(ValueError, match="unknown option"):
        _pipe().options(worker=2)


# ======================================================================================
# report / experiment
# ======================================================================================
def test_report_smoke():
    rep = _pipe().window(16).measure("hp").options(on_error="raise").report()
    assert rep["n_segments"] == N_SOURCES * (N_STEPS // 16)
    assert rep["n_sources"] == N_SOURCES
    assert "hp" in rep["metrics"]
    assert rep["metrics"]["hp"]["max"] > rep["metrics"]["hp"]["min"]


def test_experiment_smoke_pipeline_and_dict_arms():
    base = _pipe().window(16).measure("hp").options(on_error="raise")
    ax = base.schema.entity_axis
    hi = base.curate("top-k", n=6, weights={"hp": 1.0})       # Pipeline arm
    lo = {"name": "lo",                                        # raw strategy dict arm
          "define": {"hp": {"source": f"{ax}.hp", "reduce": "mean"}},
          "selection": {"method": "top-k", "weights": {"hp": -1.0}}}
    out = base.experiment({"hi": hi, "lo": lo}, n=6, baseline="hi")
    assert set(out["arms"]) == {"hi", "lo"}
    assert out["arms"]["hi"]["n_selected"] == 6
    assert out["arms"]["hi"]["metrics"]["hp"]["mean"] > \
        out["arms"]["lo"]["metrics"]["hp"]["mean"]
    assert "js_vs_baseline" in out["arms"]["lo"]["metrics"]["hp"]
    assert "js_vs_baseline" not in out["arms"]["hi"]["metrics"]["hp"]
