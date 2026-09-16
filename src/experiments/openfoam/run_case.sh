#!/usr/bin/env bash
# Run one generated OpenFOAM case: blockMesh, parallel snappyHexMesh, simpleFoam,
# reconstruct. Meant to run inside an OpenFOAM-equipped sandbox (see
# generate_case.py for how the case was built), not on the laptop.
#
# Usage: run_case.sh <case_dir>
set -euo pipefail

CASE_DIR="$1"
cd "$CASE_DIR"

if ! command -v blockMesh >/dev/null 2>&1; then
  # OpenFOAM's own etc/bashrc uses `local` var tricks that raise
  # "pop_var_context: head of shell_variables not a function context" when
  # sourced from a backgrounded/non-interactive shell (as run_case.sh may
  # be, when driven by a detached remote job). Set the handful of variables
  # it exports directly instead of sourcing it.
  for candidate in /opt/openfoam*/etc/bashrc /usr/lib/openfoam/openfoam*/etc/bashrc; do
    if [ -f "$candidate" ]; then
      WM_PROJECT_DIR="$(dirname "$(dirname "$candidate")")"
      PLATFORM_DIR=$(find "$WM_PROJECT_DIR/platforms" -maxdepth 1 -type d -name 'linux64*' | head -1)
      export WM_PROJECT_DIR
      export FOAM_APPBIN="$PLATFORM_DIR/bin"
      export FOAM_LIBBIN="$PLATFORM_DIR/lib"
      export PATH="/usr/lib/x86_64-linux-gnu/openmpi/bin:$FOAM_APPBIN:$WM_PROJECT_DIR/bin:$WM_PROJECT_DIR/wmake:$PATH"
      export LD_LIBRARY_PATH="$FOAM_LIBBIN:$FOAM_LIBBIN/sys-openmpi:$FOAM_LIBBIN/dummy:/usr/lib/x86_64-linux-gnu/openmpi/lib:${LD_LIBRARY_PATH:-}"
      break
    fi
  done
fi

command -v blockMesh >/dev/null 2>&1 || {
  echo "OpenFOAM environment not found; source its etc/bashrc before running this script" >&2
  exit 1
}

NPROCS=$(foamDictionary -entry numberOfSubdomains -value system/decomposeParDict)

# Sandboxes commonly run as root; OpenMPI refuses that by default.
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

echo "== blockMesh =="
blockMesh > log.blockMesh 2>&1

echo "== decomposePar -copyZero =="
decomposePar -copyZero > log.decomposePar 2>&1

echo "== snappyHexMesh (parallel, $NPROCS ranks) =="
mpirun -np "$NPROCS" snappyHexMesh -overwrite -parallel > log.snappyHexMesh 2>&1

echo "== checkMesh (parallel) =="
mpirun -np "$NPROCS" checkMesh -parallel > log.checkMesh 2>&1 || {
  echo "checkMesh reported issues; see log.checkMesh" >&2
}

echo "== simpleFoam (parallel) =="
mpirun -np "$NPROCS" simpleFoam -parallel > log.simpleFoam 2>&1

echo "== reconstructParMesh -constant =="
reconstructParMesh -constant > log.reconstructParMesh 2>&1

echo "== reconstructPar -latestTime =="
reconstructPar -latestTime > log.reconstructPar 2>&1

echo "done: $CASE_DIR"
