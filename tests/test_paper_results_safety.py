import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphadesign.cfd_analysis import STLWingAnalyzer
from alphadesign.experiment_runner import ExperimentRecorder, paper_experiment_specs
from alphadesign.genetic_algo_components.fitness_evaluation import FitnessEval
from alphadesign.main_pipeline import AlphaDesignPipeline


def test_stl_unit_inference_and_geometry_bounds():
    assert STLWingAnalyzer.infer_stl_unit_scale(
        [[0, 0, 0], [1800, 300, 500]], "mm"
    ) == pytest.approx(1e-3)
    assert STLWingAnalyzer.infer_stl_unit_scale(
        [[0, 0, 0], [1.8, 0.3, 0.5]]
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Unsupported STL length unit"):
        STLWingAnalyzer.infer_stl_unit_scale([[0, 0, 0], [1, 1, 1]], "inch")


def test_generated_mm_stl_sidecar_produces_si_scale_and_finite_forces(tmp_path):
    elements = []
    for element_index in range(3):
        mesh = trimesh.creation.box(
            extents=[300.0, 1600.0, 30.0],
            transform=trimesh.transformations.translation_matrix(
                [150.0, 800.0, element_index * 80.0]
            ),
        )
        elements.append(mesh)
    mesh = trimesh.util.concatenate(elements)
    stl_path = tmp_path / "generated_wing.stl"
    mesh.export(stl_path)

    sidecar = {
        "metadata": {"units": {"length": "mm"}},
        "geometry": {
            "main_element": {
                "root_chord_mm": 300.0,
                "reference_area_m2": 0.48,
            },
            "flaps": [
                {"root_chord_mm": 240.0, "reference_area_m2": 0.336,
                 "geometric_angle_deg": 0.0, "vertical_offset_mm": 80.0,
                 "camber_ratio": 0.06},
                {"root_chord_mm": 180.0, "reference_area_m2": 0.252,
                 "geometric_angle_deg": 0.0, "vertical_offset_mm": 160.0,
                 "camber_ratio": 0.05},
            ],
            "total_elements": 3,
            "total_reference_area_m2": 1.068,
        },
        "airfoil_properties": {
            "main_element": {"camber_ratio": 0.08, "max_thickness_ratio": 0.10},
            "flaps": [
                {"camber_ratio": 0.06, "thickness_ratio": 0.10},
                {"camber_ratio": 0.05, "thickness_ratio": 0.10},
            ],
        },
        "multi_element_interactions": {
            "slot_gaps_mm": [12.0, 10.0],
            "slot_gap_to_chord_ratios": [0.05, 0.055],
            "overlap_ratios": [0.10, 0.10],
        },
    }
    sidecar_path = tmp_path / "generated_wing_cfd_params.json"
    sidecar_path.write_text(json.dumps(sidecar))

    analyzer = STLWingAnalyzer(str(stl_path), str(sidecar_path))
    result = analyzer.multi_element_analysis(55.56, 75, 0)

    assert analyzer.stl_unit_scale == pytest.approx(1e-3)
    assert analyzer.wingspan == pytest.approx(1.6, abs=1e-3)
    assert all(0.02 < chord < 0.5 for chord in analyzer.chord_lengths)
    assert result["signed_lift_N"] < 0
    assert 0 < result["total_downforce"] < 1e5
    assert 0 < result["total_drag"] < 1e5
    assert np.isfinite(result["efficiency_ratio"])
    assert result["efficiency_ratio"] > 0


def test_failed_cfd_result_has_no_physical_defaults(tmp_path):
    result = FitnessEval(cfd_results_dir=tmp_path).get_default_cfd_score(
        "unit test failure"
    )
    assert result["cfd_valid"] is False
    assert result["failure_reason"] == "unit test failure"
    assert result["cfd_downforce"] is None
    assert result["cfd_drag"] is None
    assert result["cfd_efficiency"] is None


def test_equal_budget_metadata_and_recording(tmp_path):
    specs = paper_experiment_specs(seed=7, evaluation_budget=4, population_size=2)
    assert {spec.evaluation_budget for spec in specs} == {4}
    assert [spec.name for spec in specs] == [
        "random_search", "ga_only", "neural_guided_ga_frozen_policy"
    ]

    recorder = ExperimentRecorder(tmp_path, specs[1], config={"test": True})
    recorder.record_evaluation({"valid": True, "fitness": 1.5})
    recorder.close()

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["seed"] == 7
    assert metadata["evaluation_budget"] == 4
    assert metadata["execution_mode"] == "metadata_only"
    assert metadata["results_validated"] is False
    assert metadata["status"] == "completed"
    lines = (tmp_path / "evaluations.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["evaluation"] == 0


def test_recorder_rejects_budget_overrun(tmp_path):
    spec = paper_experiment_specs(seed=1, evaluation_budget=1)[0]
    recorder = ExperimentRecorder(tmp_path, spec)
    recorder.record_evaluation({"valid": False})
    with pytest.raises(ValueError, match="budget exceeded"):
        recorder.record_evaluation({"valid": False})


def test_selection_reuses_generation_fitness_scores():
    from unittest.mock import Mock

    pipeline = AlphaDesignPipeline.__new__(AlphaDesignPipeline)
    pipeline.current_population = [{"root_chord": 280}, {"root_chord": 285}]
    pipeline.fitness_eval = Mock()
    pipeline.fitness_eval.evaluate_pop.side_effect = AssertionError(
        "selection must not reevaluate the current population"
    )
    pipeline.crossover_ops = Mock()
    pipeline.crossover_ops.f1_aero_crossover.return_value = (
        {"root_chord": 281}, {"root_chord": 286}
    )
    pipeline.mutation_ops = Mock()
    pipeline.mutation_ops.f1_wing_mutation.side_effect = lambda individual: individual

    class Progress:
        def update(self, count):
            pass

    scores = [
        {"total_fitness": 2.0, "valid": True},
        {"total_fitness": 1.0, "valid": True},
    ]
    population = pipeline.generate_next_population_with_progress(Progress(), scores)

    assert len(population) == 2
    pipeline.fitness_eval.evaluate_pop.assert_not_called()


def test_policy_guidance_is_disabled_without_explicit_ablation():
    pipeline = AlphaDesignPipeline.__new__(AlphaDesignPipeline)
    pipeline.neural_network = object()
    pipeline.neural_guidance_enabled = False

    class Progress:
        def __init__(self):
            self.updated = 0

        def update(self, count):
            self.updated += count

    progress = Progress()
    population = [{"root_chord": 280}]
    assert pipeline.apply_neural_guidance_with_progress(population, progress) is population
    assert progress.updated == 1


def test_enabled_guidance_uses_frozen_policy_output():
    import torch

    class FrozenPolicy(torch.nn.Module):
        param_count = 2

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, params):
            return torch.full_like(params, 0.25), torch.zeros((params.shape[0], 1))

    class Tweaker:
        def __init__(self):
            self.policy = None

        def apply_neural_tweaks(self, params, policy_output, exploration=True):
            self.policy = policy_output.detach().clone()
            return params

    pipeline = AlphaDesignPipeline.__new__(AlphaDesignPipeline)
    pipeline.neural_network = FrozenPolicy()
    pipeline.neural_guidance_enabled = True
    pipeline.param_tweaker = Tweaker()

    class Progress:
        def __init__(self):
            self.updated = 0

        def update(self, count):
            self.updated += count

        def set_postfix(self, **kwargs):
            pass

    progress = Progress()
    population = [{"root_chord": 280, "tip_chord": 250}]
    guided = pipeline.apply_neural_guidance_with_progress(population, progress)

    assert len(guided) == 1
    assert pipeline.param_tweaker.policy is not None
    assert torch.allclose(pipeline.param_tweaker.policy, torch.tensor([[0.25, 0.25]]))
    assert progress.updated == 1
