"""Generate one OpenFOAM validation case from a manifest entry.

Instantiates the existing wing geometry generator with the selected design,
converts the resulting STL from millimetres to metres, positions it at the
surrogate's 75 mm ground clearance, sizes an external-aerodynamics domain
around it, and writes a complete OpenFOAM case (blockMesh + snappyHexMesh +
simpleFoam, kOmegaSST) at the same 200 km/h operating point the surrogate
uses. Does not invoke OpenFOAM; see run_case.sh for that.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from stl import mesh as stl_mesh_module

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.openfoam import case_files as cf  # noqa: E402
from alphadesign.wing_generator import UltraRealisticF1FrontWingGenerator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
WING_OUTPUT_DIR = REPO_ROOT / "f1_wing_output"


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@contextlib.contextmanager
def _cwd(path: Path):
    """Temporarily run with CWD=path, regardless of the ambient CWD.

    ``UltraRealisticF1FrontWingGenerator.generate_complete_wing`` hardcodes a
    relative ``f1_wing_output/`` path, so this call must not depend on
    whatever the caller's working directory happens to be.
    """
    try:
        previous: Path | None = Path.cwd()
    except FileNotFoundError:
        # A prior process (e.g. another test) chdir'd into a directory that no
        # longer exists; there is nothing valid to restore afterward.
        previous = None
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous if previous is not None else path)


def export_geometry(design: dict[str, Any], case_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate the STL for ``design`` and return (vertices_mm, cfd_params)."""
    generator = UltraRealisticF1FrontWingGenerator(**design)
    filename = f"{case_id}.stl"
    with _cwd(REPO_ROOT):
        result = generator.generate_complete_wing(filename)
    if result is None:
        raise RuntimeError(f"wing generation failed for {case_id}")

    stl_path = WING_OUTPUT_DIR / filename
    params_path = WING_OUTPUT_DIR / f"{case_id}_cfd_params.json"
    wing = stl_mesh_module.Mesh.from_file(str(stl_path))
    vertices_mm = wing.vectors.reshape(-1, 3).copy()
    cfd_params = json.loads(params_path.read_text(encoding="utf-8"))

    stl_path.unlink(missing_ok=True)
    params_path.unlink(missing_ok=True)
    return vertices_mm, cfd_params


def transform_to_meters(vertices_mm: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale mm -> m and shift so the geometry's lowest point sits at ground clearance."""
    vertices_m = vertices_mm / 1000.0
    z_shift = cf.GROUND_CLEARANCE_M - float(vertices_m[:, 2].min())
    vertices_m[:, 2] += z_shift
    return vertices_m, z_shift


def write_transformed_stl(vertices_m: np.ndarray, out_path: Path) -> None:
    faces = vertices_m.reshape(-1, 3, 3)
    out_mesh = stl_mesh_module.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh_module.Mesh.dtype))
    out_mesh.vectors[:] = faces
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_mesh.save(str(out_path))


def compute_domain(bbox: dict[str, float]) -> dict[str, float]:
    depth = bbox["xmax"] - bbox["xmin"]
    return {
        "xmin": bbox["xmin"] - 5.0 * depth,
        "xmax": bbox["xmax"] + 12.0 * depth,
        "ymin": bbox["ymin"] - 3.0 * depth,
        "ymax": bbox["ymax"] + 3.0 * depth,
        "zmin": 0.0,
        "zmax": bbox["zmax"] + 8.0 * depth,
    }


def location_in_mesh(domain: dict[str, float]) -> tuple[float, float, float]:
    x = domain["xmax"] - 0.1 * (domain["xmax"] - domain["xmin"])
    y = domain["ymax"] - 0.1 * (domain["ymax"] - domain["ymin"])
    z = 0.05
    return (x, y, z)


def write_case(
    case: dict[str, Any],
    output_dir: Path,
    mesh_level: str,
    cpu_cores: int,
) -> None:
    design = case["design"]
    case_id = case["case_id"]
    level = cf.MESH_LEVELS[mesh_level]

    vertices_mm, cfd_params = export_geometry(design, case_id)
    vertices_m, z_shift = transform_to_meters(vertices_mm)

    bbox = {
        "xmin": float(vertices_m[:, 0].min()), "xmax": float(vertices_m[:, 0].max()),
        "ymin": float(vertices_m[:, 1].min()), "ymax": float(vertices_m[:, 1].max()),
        "zmin": float(vertices_m[:, 2].min()), "zmax": float(vertices_m[:, 2].max()),
    }
    domain = compute_domain(bbox)
    loc = location_in_mesh(domain)

    ref = cfd_params["cfd_recommended_settings"]
    cofr_mm = ref["reference_point_mm"]
    cofr_m = (
        cofr_mm[0] / 1000.0,
        cofr_mm[1] / 1000.0,
        cofr_mm[2] / 1000.0 + z_shift,
    )
    reference = {
        "cofr_x": cofr_m[0], "cofr_y": cofr_m[1], "cofr_z": cofr_m[2],
        "l_ref": ref["reference_length_m"],
        "a_ref": ref["reference_area_m2"],
    }
    k_value, omega_value = cf.turbulence_initial_conditions(ref["reference_length_m"])

    case_dir = output_dir / case_id
    if case_dir.exists():
        shutil.rmtree(case_dir)
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant" / "triSurface").mkdir(parents=True)
    (case_dir / "0.orig").mkdir(parents=True)

    write_transformed_stl(vertices_m, case_dir / "constant" / "triSurface" / "wing.stl")

    (case_dir / "system" / "blockMeshDict").write_text(cf.block_mesh_dict(domain, level["cells_per_metre"]))
    (case_dir / "system" / "snappyHexMeshDict").write_text(
        cf.snappy_hex_mesh_dict(loc, level["surface_refinement"], level["layers"])
    )
    (case_dir / "system" / "decomposeParDict").write_text(cf.decompose_par_dict(cpu_cores))
    (case_dir / "system" / "controlDict").write_text(cf.control_dict(level["end_time"], reference))
    (case_dir / "system" / "fvSchemes").write_text(cf.fv_schemes())
    (case_dir / "system" / "fvSolution").write_text(cf.fv_solution())
    (case_dir / "constant" / "transportProperties").write_text(cf.transport_properties())
    (case_dir / "constant" / "turbulenceProperties").write_text(cf.turbulence_properties())
    (case_dir / "0.orig" / "U").write_text(cf.field_u())
    (case_dir / "0.orig" / "p").write_text(cf.field_p())
    (case_dir / "0.orig" / "k").write_text(cf.field_k(k_value))
    (case_dir / "0.orig" / "omega").write_text(cf.field_omega(omega_value))
    (case_dir / "0.orig" / "nut").write_text(cf.field_nut())

    metadata = {
        "case_id": case_id,
        "role": case["role"],
        "provenance": {
            "strategy": case["strategy"], "seed": case["seed"],
            "evaluation": case["evaluation"], "design_id": case["design_id"],
        },
        "surrogate_prediction": case["surrogate"],
        "design": design,
        "mesh_level": mesh_level,
        "solver": "simpleFoam",
        "turbulence_model": "kOmegaSST",
        "operating_point": {
            "freestream_velocity_ms": cf.U_INF,
            "air_density_kgm3": cf.RHO,
            "kinematic_viscosity_m2s": cf.NU,
            "ground_clearance_m": cf.GROUND_CLEARANCE_M,
            "assumed_turbulence_intensity": cf.TURBULENCE_INTENSITY,
        },
        "domain_m": domain,
        "geometry_bbox_m": bbox,
        "reference": reference,
        "mesh_settings": level,
        "cpu_cores": cpu_cores,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "generator_cfd_params": cfd_params,
    }
    (case_dir / "case_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/openfoam_candidates/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/openfoam_candidates/cases"))
    parser.add_argument("--mesh-level", choices=sorted(cf.MESH_LEVELS), default="coarse")
    parser.add_argument("--cpu-cores", type=int, default=4)
    parser.add_argument("--case-id", type=str, default=None, help="Generate only this case (default: all)")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if args.case_id:
        cases = [c for c in cases if c["case_id"] == args.case_id]
        if not cases:
            raise SystemExit(f"no such case: {args.case_id}")

    for case in cases:
        print(f"generating {case['case_id']} ({case['role']}) at mesh level {args.mesh_level}...")
        write_case(copy.deepcopy(case), args.output, args.mesh_level, args.cpu_cores)
        print(f"  -> {args.output / case['case_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
