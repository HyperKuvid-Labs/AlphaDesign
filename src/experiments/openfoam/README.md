# OpenFOAM validation pipeline (section 5)

Scripts to select representative designs from the phase-two full-protocol
run and build OpenFOAM cases for them, at the same operating point the
empirical surrogate uses (200 km/h, 75 mm ground clearance, 0 deg wing
angle, air at 1.225 kg/m^3, kinematic viscosity 1.5e-5 m^2/s -- see
`fitness_evaluation.py`'s call into `multi_element_analysis`). Nothing here
invokes OpenFOAM itself: case generation is CPU-cheap and has been run and
tested locally; the solve has not, since OpenFOAM is not installed on this
laptop. It needs a remote sandbox with OpenFOAM (targets `opencfd/openfoam-
default:2412`).

## Pipeline

1. `select_cases.py` -- reads `artifacts/phase_two/`, picks 12 to 20
   feasible designs (baseline geometry, best per strategy, median and
   poor-but-feasible per strategy, designs near a constraint boundary, and
   geometrically diverse designs), writes `manifest.json`.
2. `generate_case.py` -- for each selected design: generates the STL with
   the existing `UltraRealisticF1FrontWingGenerator`, converts it from
   millimetres to metres, positions it at 75 mm ground clearance, sizes an
   external-aero domain around it (blockMesh background hex + snappyHexMesh
   off the STL), and writes a complete `simpleFoam`/`kOmegaSST` case
   (coarse/medium/fine mesh levels, matching the section 5 mesh-convergence
   requirement).
3. `run_case.sh <case_dir>` -- meant to run on the sandbox: blockMesh,
   decomposePar, parallel snappyHexMesh, checkMesh, parallel simpleFoam,
   reconstruct.
4. `parse_forces.py <case_dir>` -- after a run completes, parses
   `postProcessing/forces/*/force.dat` into `openfoam_result.json`
   (downforce, drag, computed_efficiency, alongside the surrogate's own
   prediction for that design and a best-effort checkMesh summary).
5. `compare_surrogate.py` -- once several cases have `openfoam_result.json`,
   reports rank correlation, mean absolute relative error, systematic bias,
   and top-ranked-design agreement between the surrogate and OpenFOAM.

## Commands

```sh
python src/experiments/openfoam/select_cases.py
python src/experiments/openfoam/generate_case.py --mesh-level coarse --cpu-cores 4
# on the sandbox, per case:
bash src/experiments/openfoam/run_case.sh artifacts/openfoam_candidates/cases/case_00
python src/experiments/openfoam/parse_forces.py artifacts/openfoam_candidates/cases/case_00
# once several cases have run:
python src/experiments/openfoam/compare_surrogate.py
```

## Known assumptions, unverified until a real run

- Turbulence intensity at the inlet is assumed to be 5%; the surrogate does
  not specify one.
- The domain is sized from each design's own bounding box (5 chords
  upstream, 12 downstream, 3 to the sides, 8 above), not hand-tuned per
  case.
- `parse_forces.py` extracts the first three numbers after the timestamp on
  the last `force.dat` row as the total force vector. This is robust to
  both the plain-column and parenthesised-tuple `force.dat` layouts used by
  different OpenFOAM versions, but has not been checked against a real
  `force.dat` file yet.
- Mesh quality parsing from `log.checkMesh` is best-effort regex and may
  need adjusting once real checkMesh output is available.
