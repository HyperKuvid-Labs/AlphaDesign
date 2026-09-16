"""Small, deterministic building blocks for the phase-one benchmark.

The benchmark deliberately calls only ``FitnessEval.evaluate_formula_constratins``.
That is the package's empirical surrogate path; it does not generate an STL or
invoke an external OpenFOAM process.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from alphadesign.cli import load_base_parameters
from alphadesign.genetic_algo_components.fitness_evaluation import FitnessEval


OBJECTIVE_NAME = "computed_efficiency"
FEASIBILITY_NAME = (
    "constraint_valid and constraint_compliance >= 0.60 and safety_factor >= 1.50 "
    "and buckling_safety_factor >= 1.50 and natural_frequency >= 15 Hz "
    "and computed_downforce >= 500 N"
)
SCHEMA_VERSION = 1

# These are the same bounded fields used by the existing population
# initializer.  Sampling is implemented here because that initializer mutates
# nested lists through shallow copies.
VARIATION_FACTORS = {
    "total_span": 0.10,
    "root_chord": 0.15,
    "tip_chord": 0.15,
    "sweep_angle": 0.25,
    "dihedral_angle": 0.20,
    "max_thickness_ratio": 0.20,
    "camber_ratio": 0.25,
    "camber_position": 0.10,
    "endplate_height": 0.15,
    "endplate_max_width": 0.20,
    "endplate_min_width": 0.20,
    "y250_step_height": 0.25,
    "y250_transition_length": 0.20,
    "central_slot_width": 0.20,
    "flap_cambers": 0.20,
    "flap_slot_gaps": 0.20,
    "flap_vertical_offsets": 0.15,
    "flap_horizontal_offsets": 0.15,
}

BOUNDS = {
    "total_span": (1600.0, 1800.0),
    "root_chord": (250.0, 330.0),
    "tip_chord": (200.0, 300.0),
    "sweep_angle": (2.0, 8.0),
    "dihedral_angle": (1.0, 6.0),
    "max_thickness_ratio": (0.04, 0.20),
    "camber_ratio": (0.06, 0.15),
    "camber_position": (0.35, 0.50),
    "endplate_height": (250.0, 325.0),
    "endplate_max_width": (100.0, 150.0),
    "endplate_min_width": (35.0, 70.0),
    "y250_step_height": (15.0, 25.0),
    "y250_transition_length": (80.0, 120.0),
    "central_slot_width": (0.0, 30.0),
    "flap_cambers": (0.08, 0.16),
    "flap_slot_gaps": (8.0, 18.0),
    "flap_vertical_offsets": (20.0, 125.0),
    "flap_horizontal_offsets": (25.0, 145.0),
}


def seed_everything(seed: int) -> random.Random:
    """Seed package RNGs and return the local proposal RNG."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    return random.Random(seed)


def default_design() -> dict[str, Any]:
    """Return an independent copy of the package's standard base design."""
    return copy.deepcopy(asdict(load_base_parameters(None)))


def _bounded(value: float, name: str) -> float:
    lower, upper = BOUNDS[name]
    return max(lower, min(upper, value))


def sample_variant(base: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
    """Sample one independent variant using the shared initial distribution."""
    design = copy.deepcopy(dict(base))
    for name, factor in VARIATION_FACTORS.items():
        value = design[name]
        if isinstance(value, list):
            design[name] = [
                _bounded(float(item) * (1.0 + rng.uniform(-factor, factor)), name)
                for item in value
            ]
        else:
            design[name] = _bounded(float(value) * (1.0 + rng.uniform(-factor, factor)), name)
    design["chord_taper_ratio"] = design["tip_chord"] / design["root_chord"]
    return design


def crossover(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Simple uniform crossover with deep copies for all mutable values."""
    child = copy.deepcopy(dict(first))
    for name in VARIATION_FACTORS:
        if rng.random() < 0.5:
            child[name] = copy.deepcopy(second[name])
    child["chord_taper_ratio"] = child["tip_chord"] / child["root_chord"]
    return child


def mutate(design: Mapping[str, Any], rng: random.Random, strength: float = 0.10) -> dict[str, Any]:
    """Mutate a deep-copied design while respecting the shared bounds."""
    child = copy.deepcopy(dict(design))
    for name, factor in VARIATION_FACTORS.items():
        if rng.random() >= 0.5:
            continue
        sigma = factor * strength / 0.10
        value = child[name]
        if isinstance(value, list):
            child[name] = [
                _bounded(float(item) * (1.0 + rng.gauss(0.0, sigma)), name)
                for item in value
            ]
        else:
            child[name] = _bounded(float(value) * (1.0 + rng.gauss(0.0, sigma)), name)
    child["chord_taper_ratio"] = child["tip_chord"] / child["root_chord"]
    return child


def normalize(value: Any) -> Any:
    """Convert NumPy/scalar values to stable JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if hasattr(value, "item"):
        return normalize(value.item())
    if hasattr(value, "tolist"):
        return normalize(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def design_key(design: Mapping[str, Any]) -> str:
    payload = json.dumps(normalize(design), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_surrogate_evaluator(output_dir: Path) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Create the formula evaluator with per-run result isolation."""
    evaluator = FitnessEval(cfd_results_dir=str(output_dir / "cfd_results"))

    def evaluate(design: Mapping[str, Any]) -> dict[str, Any]:
        # Formula constraints never call ``evaluate_cfd_perf``.  Keeping this
        # wrapper explicit prevents a future driver edit from silently using
        # the package's STL/CFD branch.
        return evaluator.evaluate_formula_constratins(copy.deepcopy(dict(design)))

    return evaluate


def result_from_evaluator(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one fail-closed feasibility/objective policy to raw output."""
    normalized = normalize(dict(raw))
    constraint_valid = bool(normalized.get("constraint_valid", False))
    compliance = normalized.get("constraint_compliance")
    objective = normalized.get(OBJECTIVE_NAME)
    safety_factor = normalized.get("safety_factor")
    buckling_safety_factor = normalized.get("buckling_safety_factor")
    natural_frequency = normalized.get("natural_frequency")
    numeric = {
        "computed_downforce": normalized.get("computed_downforce"),
        "computed_drag": normalized.get("computed_drag"),
        "computed_efficiency": objective,
        "aerodynamic_score": normalized.get("aerodynamic_score"),
    }
    failure = normalized.get("failure_reason")
    valid = (
        constraint_valid
        and isinstance(compliance, (int, float))
        and math.isfinite(compliance)
        and compliance >= 0.60
        and isinstance(safety_factor, (int, float))
        and math.isfinite(safety_factor)
        and safety_factor >= 1.50
        and isinstance(buckling_safety_factor, (int, float))
        and math.isfinite(buckling_safety_factor)
        and buckling_safety_factor >= 1.50
        and isinstance(natural_frequency, (int, float))
        and math.isfinite(natural_frequency)
        and natural_frequency >= 15.0
        and isinstance(numeric["computed_downforce"], (int, float))
        and numeric["computed_downforce"] >= 500.0
        and isinstance(objective, (int, float))
        and math.isfinite(objective)
        and all(item is not None and isinstance(item, (int, float)) and math.isfinite(item)
                for item in numeric.values())
    )
    if not valid and not failure:
        failure = "infeasible or non-finite surrogate result"
    return {
        "valid": valid,
        "objective": float(objective) if valid else None,
        "objective_name": OBJECTIVE_NAME,
        "constraints": {
            "constraint_valid": constraint_valid,
            "constraint_compliance": compliance,
            "safety_factor": safety_factor,
            "buckling_safety_factor": buckling_safety_factor,
            "natural_frequency": natural_frequency,
        },
        "aerodynamics": numeric,
        "failure_reason": failure,
        "raw": normalized,
    }


class RunRecorder:
    """Append-only JSONL records with deterministic restart/replay."""

    def __init__(self, output_dir: os.PathLike[str] | str, metadata: Mapping[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.output_dir / "metadata.json"
        self.records_path = self.output_dir / "evaluations.jsonl"
        self.metadata = normalize(dict(metadata))
        if self.metadata_path.exists():
            existing = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            expected = dict(self.metadata)
            expected.pop("created_at_utc", None)
            actual = dict(existing)
            actual.pop("created_at_utc", None)
            if actual != expected:
                raise ValueError("existing metadata does not match this run")
            # Metadata is immutable.  Keep the original creation timestamp on
            # restart rather than rewriting the file.
            self.metadata = existing
        else:
            with self.metadata_path.open("x", encoding="utf-8") as handle:
                json.dump(self.metadata, handle, indent=2, sort_keys=True)
                handle.write("\n")
        self.records = self._read_records()

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        records = []
        for expected, line in enumerate(self.records_path.read_text(encoding="utf-8").splitlines()):
            record = json.loads(line)
            if record.get("evaluation") != expected:
                raise ValueError("evaluation records must have contiguous IDs")
            records.append(record)
        return records

    def evaluate_or_replay(
        self,
        evaluation: int,
        design: Mapping[str, Any],
        evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_design = normalize(dict(design))
        identifier = design_key(design)
        if evaluation < len(self.records):
            record = self.records[evaluation]
            if record["design"] != normalized_design:
                raise ValueError("deterministic replay proposed a different design")
            if record["design_id"] != identifier:
                raise ValueError("stored design_id does not match its design")
            return record["result"]
        if evaluation != len(self.records):
            raise ValueError("evaluation records are not contiguous")
        started = time.perf_counter()
        try:
            result = result_from_evaluator(evaluator(design))
        except Exception as exc:  # fail closed, while preserving the reason
            result = {
                "valid": False,
                "objective": None,
                "objective_name": OBJECTIVE_NAME,
                "constraints": {},
                "aerodynamics": {},
                "failure_reason": f"evaluator exception: {exc}",
                "raw": {},
            }
        duration_seconds = time.perf_counter() - started
        record = {
            "evaluation": evaluation,
            "design_id": identifier,
            "design": normalized_design,
            "duration_seconds": duration_seconds,
            "context": normalize(dict(context or {})),
            "result": normalize(result),
        }
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        return record["result"]

    def finish(self, status: str = "completed") -> None:
        valid = [record["result"] for record in self.records if record["result"].get("valid")]
        summary = {
            "status": status,
            "evaluations_recorded": len(self.records),
            "valid_evaluations": len(valid),
            "best_objective": max((item["objective"] for item in valid), default=None),
        }
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def metadata(
    strategy: str,
    seed: int,
    budget: int,
    population_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    if budget <= 0 or population_size <= 0:
        raise ValueError("budget and population_size must be positive")
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        dirty = bool(subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {
        "schema_version": SCHEMA_VERSION,
        "strategy": strategy,
        "seed": seed,
        "evaluation_budget": budget,
        "population_size": population_size,
        "objective": OBJECTIVE_NAME,
        "feasibility": FEASIBILITY_NAME,
        "evaluator": "FitnessEval.evaluate_formula_constratins",
        "execution_mode": "empirical_surrogate_only",
        "neural_guidance_enabled": False,
        "neural_training_enabled": False,
        "external_solver_invoked": False,
        "physically_validated": False,
        "search_space": {
            "bounds": BOUNDS,
            "variation_factors": VARIATION_FACTORS,
        },
        "output_dir": str(output_dir),
        "git_commit": commit,
        "git_dirty": dirty,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
    }


__all__ = [
    "RunRecorder", "crossover", "default_design", "design_key", "make_surrogate_evaluator",
    "metadata", "mutate", "sample_variant", "seed_everything",
]
