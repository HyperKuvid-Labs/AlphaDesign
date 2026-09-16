"""Compare surrogate predictions against completed OpenFOAM runs.

Reads every ``openfoam_result.json`` produced by parse_forces.py under a
cases directory and reports rank correlation, force error, systematic bias,
and agreement among the top-ranked designs, per the section 5 requirement.
Only cases with a completed OpenFOAM run are included; this does not run
anything itself.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_results(cases_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(cases_dir.glob("*/openfoam_result.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def spearman(x: list[float], y: list[float]) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).correlation)
    except ImportError:
        def rank(values: list[float]) -> list[float]:
            order = sorted(range(len(values)), key=lambda i: values[i])
            ranks = [0.0] * len(values)
            for position, index in enumerate(order):
                ranks[index] = position + 1
            return ranks
        rx, ry = rank(x), rank(y)
        return statistics.correlation(rx, ry) if hasattr(statistics, "correlation") else float("nan")


def relative_error(openfoam: float, surrogate: float) -> float | None:
    if surrogate == 0:
        return None
    return (openfoam - surrogate) / abs(surrogate)


def compare(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for r in results:
        surrogate_aero = r["surrogate_prediction"]["aerodynamics"]
        of = r["openfoam"]
        rows.append({
            "case_id": r["case_id"],
            "role": r["role"],
            "surrogate_downforce": surrogate_aero["computed_downforce"],
            "surrogate_drag": surrogate_aero["computed_drag"],
            "surrogate_efficiency": surrogate_aero["computed_efficiency"],
            "openfoam_downforce": of["downforce_N"],
            "openfoam_drag": of["drag_N"],
            "openfoam_efficiency": of["computed_efficiency"],
        })

    def metric_block(surrogate_key: str, openfoam_key: str) -> dict[str, Any]:
        pairs = [(r[surrogate_key], r[openfoam_key]) for r in rows
                 if r[surrogate_key] is not None and r[openfoam_key] is not None]
        if len(pairs) < 2:
            return {"n": len(pairs)}
        surrogate_vals = [p[0] for p in pairs]
        openfoam_vals = [p[1] for p in pairs]
        rel_errors = [relative_error(o, s) for o, s in zip(openfoam_vals, surrogate_vals)]
        rel_errors = [e for e in rel_errors if e is not None]
        return {
            "n": len(pairs),
            "spearman_rank_correlation": spearman(surrogate_vals, openfoam_vals),
            "mean_absolute_relative_error": statistics.mean(abs(e) for e in rel_errors) if rel_errors else None,
            "systematic_bias_mean_signed_relative_error": statistics.mean(rel_errors) if rel_errors else None,
        }

    top_n = min(3, len(rows))
    by_surrogate = sorted(rows, key=lambda r: r["surrogate_efficiency"], reverse=True)[:top_n]
    by_openfoam = sorted(rows, key=lambda r: r["openfoam_efficiency"], reverse=True)[:top_n]
    surrogate_top_ids = {r["case_id"] for r in by_surrogate}
    openfoam_top_ids = {r["case_id"] for r in by_openfoam}
    overlap = surrogate_top_ids & openfoam_top_ids

    return {
        "n_cases_compared": len(rows),
        "downforce": metric_block("surrogate_downforce", "openfoam_downforce"),
        "drag": metric_block("surrogate_drag", "openfoam_drag"),
        "efficiency": metric_block("surrogate_efficiency", "openfoam_efficiency"),
        "top_ranked_agreement": {
            "top_n": top_n,
            "surrogate_top_cases": sorted(surrogate_top_ids),
            "openfoam_top_cases": sorted(openfoam_top_ids),
            "overlap_count": len(overlap),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("artifacts/openfoam_candidates/cases"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/openfoam_candidates/comparison.json"))
    args = parser.parse_args()

    results = load_results(args.cases)
    if not results:
        print(f"no completed openfoam_result.json files found under {args.cases}")
        return 1

    report = compare(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"compared {report['n_cases_compared']} cases -> {args.output}")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
