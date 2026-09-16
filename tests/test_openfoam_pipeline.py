import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.common import default_design
from experiments.openfoam import select_cases
from experiments.openfoam import generate_case
from experiments.openfoam import case_files as cf


def _fake_result(objective, valid=True, downforce=1200.0, drag=250.0, compliance=0.75):
    return {
        "valid": valid,
        "objective": objective if valid else None,
        "objective_name": "computed_efficiency",
        "constraints": {
            "constraint_valid": valid,
            "constraint_compliance": compliance,
            "safety_factor": 2.0,
            "buckling_safety_factor": 2.0,
            "natural_frequency": 20.0,
        },
        "aerodynamics": {
            "computed_downforce": downforce,
            "computed_drag": drag,
            "computed_efficiency": objective if valid else None,
            "aerodynamic_score": 80.0,
        },
        "failure_reason": None if valid else "infeasible",
    }


def _make_records(base_design):
    records = []
    evaluation = 0
    for strategy in ["random_search", "ga_only"]:
        for seed in range(2):
            for i in range(20):
                design = dict(base_design)
                design["root_chord"] = 250.0 + i * 3.0 + (10.0 if strategy == "ga_only" else 0.0)
                design["total_span"] = 1600.0 + i * 5.0
                objective = 2.0 + 0.1 * i + (1.0 if strategy == "ga_only" else 0.0)
                record = {
                    "evaluation": evaluation,
                    "design_id": f"{strategy}-{seed}-{i}",
                    "design": design,
                    "duration_seconds": 0.001,
                    "context": {},
                    "result": _fake_result(objective, valid=(i % 4 != 0)),
                }
                records.append(dict(record, strategy=strategy, seed=seed))
                evaluation += 1
    return records


def test_select_cases_covers_required_roles():
    base_design = default_design()
    records = _make_records(base_design)

    cases = select_cases.select(records, min_count=12, max_count=20)
    roles = {role for case in cases for role in case["role"].split(", ")}

    assert 12 <= len(cases) <= 20
    assert "baseline" in roles
    assert "best_random_search" in roles
    assert "best_ga_only" in roles
    assert "near_constraint_boundary" in roles
    assert "geometrically_diverse" in roles
    # every selected design must actually be feasible except the baseline, which
    # is included regardless of feasibility as a fixed reference point.
    for case in cases:
        if case["role"] != "baseline":
            assert case["surrogate"]["valid"] is True


def test_generate_case_produces_expected_openfoam_case(tmp_path):
    design = default_design()
    case = {
        "case_id": "unit_test_case",
        "role": "unit_test",
        "strategy": "unit_test",
        "seed": 0,
        "evaluation": 0,
        "design_id": "unit-test-design",
        "design": design,
        "surrogate": _fake_result(4.5),
    }

    generate_case.write_case(case, tmp_path, mesh_level="coarse", cpu_cores=2)
    case_dir = tmp_path / "unit_test_case"

    assert (case_dir / "constant" / "triSurface" / "wing.stl").stat().st_size > 0
    for name in ["blockMeshDict", "snappyHexMeshDict", "decomposeParDict", "controlDict", "fvSchemes", "fvSolution"]:
        assert (case_dir / "system" / name).read_text().strip()
    for name in ["U", "p", "k", "omega", "nut"]:
        assert (case_dir / "0.orig" / name).read_text().strip()

    metadata = json.loads((case_dir / "case_metadata.json").read_text())
    assert metadata["mesh_level"] == "coarse"
    assert metadata["operating_point"]["freestream_velocity_ms"] == cf.U_INF
    # the geometry's lowest point must sit exactly at the surrogate's ground clearance
    assert abs(metadata["geometry_bbox_m"]["zmin"] - cf.GROUND_CLEARANCE_M) < 1e-6
    # the domain floor must be the ground plane
    assert metadata["domain_m"]["zmin"] == 0.0

    block_mesh_text = (case_dir / "system" / "blockMeshDict").read_text()
    for patch in ["inlet", "outlet", "ground", "top", "sides"]:
        assert patch in block_mesh_text

    control_dict_text = (case_dir / "system" / "controlDict").read_text()
    assert "forces" in control_dict_text
    assert "forceCoeffs" in control_dict_text
