import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.ga_only import run_ga_only
from experiments.random_search import run_random_search
from experiments.common import result_from_evaluator


def fake_evaluator(design):
    return {
        "constraint_valid": True,
        "constraint_compliance": 0.9,
        "computed_downforce": 1000.0 + design["root_chord"],
        "computed_drag": 100.0,
        "computed_efficiency": 10.0,
        "aerodynamic_score": 80.0,
        "safety_factor": 3.0,
        "buckling_safety_factor": 2.0,
        "natural_frequency": 30.0,
    }


def records(path):
    return [json.loads(line) for line in (path / "evaluations.jsonl").read_text().splitlines()]


def test_random_search_enforces_budget_and_replays_without_duplicates(tmp_path):
    calls = []

    def counting(design):
        calls.append(design)
        return fake_evaluator(design)

    output = tmp_path / "random"
    run_random_search(7, 5, output, evaluator=counting)
    first = records(output)
    assert len(calls) == 5
    assert [item["evaluation"] for item in first] == list(range(5))
    assert all(item["design_id"] and item["duration_seconds"] >= 0 for item in first)
    assert all(item["context"]["proposal"] == "random_search" for item in first)
    run_random_search(7, 5, output, evaluator=counting)
    assert len(calls) == 5
    assert records(output) == first


def test_random_and_ga_share_base_design_but_ga_is_budgeted(tmp_path):
    random_output = tmp_path / "random"
    ga_output = tmp_path / "ga"
    run_random_search(11, 6, random_output, evaluator=fake_evaluator)
    run_ga_only(11, 6, ga_output, population_size=3, evaluator=fake_evaluator)
    random_records = records(random_output)
    ga_records = records(ga_output)
    assert random_records[0]["design"] == ga_records[0]["design"]
    assert len(random_records) == len(ga_records) == 6
    ga_metadata = json.loads((ga_output / "metadata.json").read_text())
    assert ga_metadata["external_solver_invoked"] is False
    assert ga_metadata["neural_guidance_enabled"] is False
    assert ga_metadata["neural_training_enabled"] is False
    assert ga_records[0]["context"]["proposal"] == "initial_population"
    assert ga_records[-1]["context"]["proposal"] == "crossover_mutation"


def test_ga_is_deterministic_across_independent_runs(tmp_path):
    first_output = tmp_path / "ga_first"
    second_output = tmp_path / "ga_second"
    run_ga_only(19, 8, first_output, population_size=3, evaluator=fake_evaluator)
    run_ga_only(19, 8, second_output, population_size=3, evaluator=fake_evaluator)

    def stable_fields(record):
        return {
            "evaluation": record["evaluation"],
            "design_id": record["design_id"],
            "design": record["design"],
            "context": record["context"],
            "result": record["result"],
        }

    assert [stable_fields(item) for item in records(first_output)] == [
        stable_fields(item) for item in records(second_output)
    ]


def test_failed_evaluator_is_recorded_fail_closed(tmp_path):
    def failing(_):
        raise RuntimeError("synthetic failure")

    output = tmp_path / "failed"
    run_random_search(3, 1, output, evaluator=failing)
    result = records(output)[0]["result"]
    assert result["valid"] is False
    assert result["objective"] is None
    assert "synthetic failure" in result["failure_reason"]


def test_critical_structural_failure_is_infeasible():
    raw = fake_evaluator({"root_chord": 280})
    raw["buckling_safety_factor"] = 1.49
    result = result_from_evaluator(raw)
    assert result["valid"] is False
    assert result["objective"] is None
