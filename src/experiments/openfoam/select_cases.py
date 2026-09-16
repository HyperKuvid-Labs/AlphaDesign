"""Select representative designs for OpenFOAM validation (todo.md section 5).

Reads the completed phase-two full-protocol records and picks roughly 12 to 20
feasible designs covering: the baseline geometry, the best design from each
strategy, median and poor-but-feasible designs per strategy, geometrically
diverse designs, and designs near a constraint boundary. Writes a manifest
that downstream OpenFOAM case generation reads; it does not run any solver.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import VARIATION_FACTORS, design_key, normalize  # noqa: E402

STRATEGIES = ["random_search", "ga_only"]
CONSTRAINT_THRESHOLDS = {
    "constraint_compliance": 0.60,
    "safety_factor": 1.50,
    "buckling_safety_factor": 1.50,
    "natural_frequency": 15.0,
    "computed_downforce": 500.0,
}


def load_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for strategy in STRATEGIES:
        for seed_dir in sorted((root / strategy).glob("seed_*"), key=lambda p: int(p.name.split("_")[1])):
            seed = int(seed_dir.name.split("_")[1])
            for line in (seed_dir / "evaluations.jsonl").read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                record["strategy"] = strategy
                record["seed"] = seed
                records.append(record)
    return records


def is_feasible(record: Mapping[str, Any]) -> bool:
    return bool(record["result"].get("valid"))


def objective_of(record: Mapping[str, Any]) -> float:
    return record["result"]["objective"]


def normalized_margin(record: Mapping[str, Any]) -> float:
    constraints = dict(record["result"]["constraints"])
    constraints["computed_downforce"] = record["result"]["aerodynamics"]["computed_downforce"]
    margins = []
    for name, threshold in CONSTRAINT_THRESHOLDS.items():
        value = constraints.get(name)
        if not isinstance(value, (int, float)):
            return float("inf")
        margins.append((value - threshold) / threshold)
    return min(margins)


def param_vector(design: Mapping[str, Any]) -> list[float]:
    vector = []
    for name in VARIATION_FACTORS:
        value = design[name]
        if isinstance(value, list):
            vector.extend(float(v) for v in value)
        else:
            vector.append(float(value))
    return vector


def normalize_vector(vector: list[float], lo: list[float], hi: list[float]) -> list[float]:
    return [
        (v - l) / (h - l) if h > l else 0.0
        for v, l, h in zip(vector, lo, hi)
    ]


def farthest_point_selection(
    candidates: list[dict[str, Any]],
    already_selected: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if not candidates or count <= 0:
        return []
    vectors = [param_vector(c["design"]) for c in candidates]
    dims = len(vectors[0])
    lo = [min(v[i] for v in vectors) for i in range(dims)]
    hi = [max(v[i] for v in vectors) for i in range(dims)]
    normed = [normalize_vector(v, lo, hi) for v in vectors]

    chosen_vectors = [normalize_vector(param_vector(c["design"]), lo, hi) for c in already_selected]
    chosen_indices: list[int] = []

    def min_dist_to_chosen(vec: list[float]) -> float:
        if not chosen_vectors:
            return float("inf")
        return min(
            sum((a - b) ** 2 for a, b in zip(vec, other)) ** 0.5
            for other in chosen_vectors
        )

    remaining = list(range(len(candidates)))
    for _ in range(min(count, len(remaining))):
        best_idx = max(remaining, key=lambda i: min_dist_to_chosen(normed[i]))
        chosen_indices.append(best_idx)
        chosen_vectors.append(normed[best_idx])
        remaining.remove(best_idx)

    return [candidates[i] for i in chosen_indices]


def build_case(record: Mapping[str, Any], role: str) -> dict[str, Any]:
    return {
        "role": role,
        "strategy": record["strategy"],
        "seed": record["seed"],
        "evaluation": record["evaluation"],
        "design_id": record["design_id"],
        "design": record["design"],
        "surrogate": record["result"],
    }


def select(records: list[dict[str, Any]], min_count: int, max_count: int) -> list[dict[str, Any]]:
    feasible = [r for r in records if is_feasible(r)]
    if not feasible:
        raise ValueError("no feasible records found; cannot select representative cases")

    selected: dict[str, dict[str, Any]] = {}

    def add(record: Mapping[str, Any], role: str) -> None:
        key = record["design_id"]
        if key in selected:
            if role not in selected[key]["role"].split(", "):
                selected[key]["role"] = selected[key]["role"] + ", " + role
        else:
            selected[key] = build_case(record, role)

    # 1. Baseline geometry: evaluation 0 is the same base design in every run.
    baseline = records[0]
    add(baseline, "baseline")

    # 2. Best design from each strategy.
    for strategy in STRATEGIES:
        pool = [r for r in feasible if r["strategy"] == strategy]
        if pool:
            best = max(pool, key=objective_of)
            add(best, f"best_{strategy}")

    # 3. Median-performing and poor-but-feasible per strategy.
    for strategy in STRATEGIES:
        pool = sorted((r for r in feasible if r["strategy"] == strategy), key=objective_of)
        if pool:
            median = pool[len(pool) // 2]
            add(median, f"median_{strategy}")
            poor = pool[0]
            add(poor, f"poor_but_feasible_{strategy}")

    # 4. Designs near a constraint boundary (smallest normalized margin).
    by_margin = sorted(feasible, key=normalized_margin)
    for record in by_margin[:4]:
        add(record, "near_constraint_boundary")

    # 5. Geometrically diverse designs, filling remaining budget.
    already = list(selected.values())
    remaining_budget = max(0, min_count - len(selected))
    diverse_target = max(remaining_budget, 4)
    diverse_pool = [r for r in feasible if r["design_id"] not in selected]
    diverse = farthest_point_selection(diverse_pool, already, diverse_target)
    for record in diverse:
        add(record, "geometrically_diverse")
        if len(selected) >= max_count:
            break

    cases = list(selected.values())
    if len(cases) > max_count:
        priority = {
            "baseline": 0, "best": 1, "median": 2, "poor_but_feasible": 2,
            "near_constraint_boundary": 3, "geometrically_diverse": 4,
        }

        def rank(case):
            roles = case["role"].split(", ")
            return min(priority.get(role.split("_")[0] if role.startswith("best") or role.startswith("median")
                                     or role.startswith("poor") else role, 5) for role in roles)

        cases = sorted(cases, key=rank)[:max_count]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/phase_two"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/openfoam_candidates/manifest.json"))
    parser.add_argument("--min-count", type=int, default=12)
    parser.add_argument("--max-count", type=int, default=20)
    args = parser.parse_args()

    records = load_records(args.input)
    cases = select(records, args.min_count, args.max_count)

    for index, case in enumerate(cases):
        case["case_id"] = f"case_{index:02d}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(normalize({"cases": cases}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"selected {len(cases)} cases -> {args.output}")
    for case in cases:
        print(f"  {case['case_id']}: {case['role']} (objective={case['surrogate']['objective']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
