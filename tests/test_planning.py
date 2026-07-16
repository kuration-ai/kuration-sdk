"""Logical traits and deterministic physical-plan compilation."""
import json

import pytest

from kuration import (OperatorTraits, Registry, compile_physical_plan, default_registry,
                       input)


def _pipeline():
    return (input([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}])
            .window(2)
            .measure("level", "mean", of="x")
            .filter("level", ">=", 1.0)
            .bin("level", bins=2)
            .sample("top-k", n=1, weights={"level": 1.0}))


def test_operator_traits_validate_closed_values_and_columns():
    traits = OperatorTraits.from_dict({
        "scope": "window", "shape": "map", "fusible": True,
        "columns": {"required": ["field:x"], "produced": ["feature:place"]},
        "resources": {"num_cpus": 2, "preferred_batch_size": 16},
    })
    assert traits.required_columns == ("field:x",)
    assert traits.resources.cpu == 2
    with pytest.raises(ValueError, match="execution trait scope"):
        OperatorTraits.from_dict({"scope": "universe", "shape": "map", "fusible": True})
    with pytest.raises(ValueError, match="unknown execution trait"):
        OperatorTraits.from_dict({"scope": "row", "shape": "map", "fusible": True,
                                  "surprise": True})


def test_component_descriptor_accepts_and_normalizes_execution_traits():
    registry = Registry(default_registry)
    registry.register_transform(
        "acme/place", lambda: None,
        execution_traits={"scope": "window", "shape": "map", "fusible": True,
                          "state_bound": "bounded-by-parameter"})
    raw = registry.descriptor("transform", "acme/place").to_dict()
    assert raw["execution_traits"]["scope"] == "window"
    assert raw["execution_traits"]["resources"]["cpu"] == 1.0


def test_physical_plan_fuses_maps_and_isolates_global_barriers():
    plan = compile_physical_plan(_pipeline())
    names = [[operator.name for operator in stage.operators] for stage in plan.stages]
    assert names[0] == ["read", "segment", "metrics", "predicates"]
    assert names[1:] == [["organize"], ["selection:top-k"], ["output"]]
    assert plan.stages[0].mode == "pipeline_parallel"
    assert plan.stages[1].mode == "data_parallel"
    assert plan.stages[1].passes == 2
    assert "global scope" in plan.stages[1].barrier_reason


def test_explain_embeds_json_safe_physical_plan_deterministically():
    pipeline = _pipeline()
    first = pipeline.explain()
    second = pipeline.explain()
    assert first.physical_plan == second.physical_plan
    assert first.stages[0]["streaming"] is True
    assert first.stages[1]["materializes_total_pool"] is False
    assert first.stages[1]["streaming"] is False  # bounded multipass barrier, not row streaming
    assert json.loads(json.dumps(first.to_dict()))["physical_plan"]["version"] == 1
    rendered = first.render(detailed=True)
    assert "selection: top-k (count=1)" in rendered
    assert "optional extras: none (NumPy core)" in rendered
    assert "total-pool materialization=none" in rendered
    assert "mode=pipeline_parallel" in rendered
    assert "barrier:" in rendered


def test_explain_warnings_include_actionable_fixes():
    plan = input([{"label": "a"}, {"label": "b"}]).explain()
    rendered = plan.render()
    assert "warning [selection-unresolved]" in rendered
    assert "warning [categorical-output-policy]" in rendered
    assert "fix: Add .sample" in rendered
    assert "categorical='omit'" in rendered


def test_physical_plan_memory_budget_validates_without_reading_data():
    with pytest.raises(ValueError, match="memory_budget"):
        compile_physical_plan(_pipeline(), memory_budget=0)
    plan = compile_physical_plan(_pipeline(), memory_budget=1024, spill_path="/tmp/spill")
    assert plan.memory_budget == 1024 and plan.spill_path == "/tmp/spill"


def test_pipeline_planner_controls_are_validated_and_explained():
    pipeline = _pipeline().options(memory_budget=4096, batch_size=32,
                                   spill_dir="/tmp/custom-spill", approximation="exact")
    plan = pipeline.explain().physical_plan
    assert plan["memory_budget"] == 4096
    assert plan["spill_path"] == "/tmp/custom-spill"
    assert pipeline.describe()["options"]["batch_size"] == 32
    with pytest.raises(ValueError, match="batch_size"):
        pipeline.options(batch_size=0)
    with pytest.raises(ValueError, match="approximation"):
        pipeline.options(approximation="maybe")


def test_selection_stage_declares_exact_and_approximate_contracts():
    stage = next(stage for stage in compile_physical_plan(_pipeline()).stages
                 if stage.name == "selection:top-k")
    assert stage.operators[0].parameters["execution_contract"] == {
        "exact": "bounded-multipass", "approximate": "unsupported"}
