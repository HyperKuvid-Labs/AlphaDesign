"""Parse a run OpenFOAM case's forces output into a JSON record.

Reads ``postProcessing/forces/*/force.dat`` (the plain ``forces`` function
object, not ``forceCoeffs``, so the drag/downforce values do not depend on
guessing which coefficient-file column layout a given OpenFOAM version uses)
and pulls the total force vector from the last written time step. Also
best-effort scrapes checkMesh's log for a few mesh-quality numbers to record
alongside the result, per the section 5 requirement to record mesh quality.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def find_force_dat(case_dir: Path) -> Path:
    candidates = sorted((case_dir / "postProcessing" / "forces").glob("*/force.dat"))
    if not candidates:
        raise FileNotFoundError(f"no force.dat found under {case_dir}/postProcessing/forces")
    return candidates[-1]


def parse_last_force_row(force_dat: Path) -> tuple[float, float, float, float]:
    last_data_line = None
    for line in force_dat.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        last_data_line = stripped
    if last_data_line is None:
        raise ValueError(f"{force_dat} has no data rows")
    numbers = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", last_data_line)]
    if len(numbers) < 4:
        raise ValueError(f"could not parse a time + force vector from: {last_data_line!r}")
    time, fx, fy, fz = numbers[0], numbers[1], numbers[2], numbers[3]
    return time, fx, fy, fz


def parse_check_mesh(case_dir: Path) -> dict[str, Any]:
    log_path = case_dir / "log.checkMesh"
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    quality: dict[str, Any] = {}
    cells_match = re.search(r"cells:\s*(\d+)", text)
    if cells_match:
        quality["cells"] = int(cells_match.group(1))
    non_ortho_match = re.search(r"Max non-orthogonality\s*=?\s*([\d.]+)", text)
    if non_ortho_match:
        quality["max_non_orthogonality"] = float(non_ortho_match.group(1))
    skewness_match = re.search(r"Max skewness\s*=?\s*([\d.]+)", text)
    if skewness_match:
        quality["max_skewness"] = float(skewness_match.group(1))
    quality["mesh_ok"] = "Mesh OK" in text
    return quality


def build_result(case_dir: Path) -> dict[str, Any]:
    metadata = json.loads((case_dir / "case_metadata.json").read_text(encoding="utf-8"))
    force_dat = find_force_dat(case_dir)
    time, fx, fy, fz = parse_last_force_row(force_dat)

    drag = fx
    downforce = -fz
    efficiency = downforce / drag if drag > 0 else None

    return {
        "case_id": metadata["case_id"],
        "role": metadata["role"],
        "provenance": metadata["provenance"],
        "openfoam": {
            "converged_at_time": time,
            "drag_N": drag,
            "downforce_N": downforce,
            "side_force_N": fy,
            "computed_efficiency": efficiency,
            "force_dat_path": str(force_dat.relative_to(case_dir)),
            "mesh_level": metadata["mesh_level"],
            "solver": metadata["solver"],
            "turbulence_model": metadata["turbulence_model"],
            "mesh_quality": parse_check_mesh(case_dir),
        },
        "surrogate_prediction": metadata["surrogate_prediction"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = build_result(args.case_dir)
    output = args.output or (args.case_dir / "openfoam_result.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
