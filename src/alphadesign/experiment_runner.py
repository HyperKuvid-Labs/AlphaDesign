"""Reproducible experiment metadata and equal-budget paper-run scaffolding.

This module deliberately does not run optimization or fabricate results.  It
defines the records that an experiment driver must write before and during a
run, so random search, GA-only, and frozen-policy ablations can be compared
under the same evaluation budget.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PAPER_STRATEGIES = (
    "random_search",
    "ga_only",
    "neural_guided_ga_frozen_policy",
)


@dataclass(frozen=True)
class ExperimentSpec:
    """One reproducible optimization condition."""

    name: str
    seed: int
    evaluation_budget: int
    population_size: int = 20

    def __post_init__(self) -> None:
        if self.name not in PAPER_STRATEGIES:
            raise ValueError(f"Unknown experiment strategy: {self.name}")
        if self.evaluation_budget <= 0:
            raise ValueError("evaluation_budget must be positive")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")


def paper_experiment_specs(seed: int, evaluation_budget: int,
                           population_size: int = 20) -> tuple[ExperimentSpec, ...]:
    """Return matched conditions with identical evaluation budgets."""
    return tuple(
        ExperimentSpec(name, seed, evaluation_budget, population_size)
        for name in PAPER_STRATEGIES
    )


def seed_everything(seed: int) -> None:
    """Seed available random number generators for a run."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_commit(repository_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_run_metadata(spec: ExperimentSpec, config: Optional[Dict[str, Any]] = None,
                       repository_root: Optional[os.PathLike[str] | str] = None) -> Dict[str, Any]:
    """Build provenance metadata without claiming that results are valid."""
    root = Path(repository_root or Path(__file__).resolve().parents[2])
    return {
        "schema_version": 1,
        "status": "planned",
        "execution_mode": "metadata_only",
        "results_validated": False,
        "legacy_artifacts_excluded": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": asdict(spec),
        "config": config or {},
        "seed": spec.seed,
        "evaluation_budget": spec.evaluation_budget,
        "git_commit": _git_commit(root),
        "python": sys.version,
        "platform": platform.platform(),
    }


class ExperimentRecorder:
    """Write immutable run metadata and one JSON record per evaluation."""

    def __init__(self, output_dir: os.PathLike[str] | str, spec: ExperimentSpec,
                 config: Optional[Dict[str, Any]] = None,
                 repository_root: Optional[os.PathLike[str] | str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.metadata_path = self.output_dir / "metadata.json"
        self.evaluations_path = self.output_dir / "evaluations.jsonl"
        self.metadata = build_run_metadata(spec, config, repository_root)
        self._evaluation_count = 0
        self._write_metadata()

    def _write_metadata(self) -> None:
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def record_evaluation(self, result: Dict[str, Any]) -> None:
        if self._evaluation_count >= self.spec.evaluation_budget:
            raise ValueError("evaluation budget exceeded")
        record = {
            "evaluation": self._evaluation_count,
            "strategy": self.spec.name,
            "seed": self.spec.seed,
            "valid": bool(result.get("valid", False)),
            "result": result,
        }
        with self.evaluations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._evaluation_count += 1

    def close(self, status: str = "completed") -> None:
        if status not in {"completed", "failed", "aborted"}:
            raise ValueError(f"Unsupported run status: {status}")
        self.metadata.update({
            "status": status,
            "evaluations_recorded": self._evaluation_count,
        })
        self._write_metadata()


__all__ = [
    "ExperimentSpec",
    "ExperimentRecorder",
    "PAPER_STRATEGIES",
    "build_run_metadata",
    "paper_experiment_specs",
    "seed_everything",
]
