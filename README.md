# AlphaDesign

AlphaDesign is a research prototype for constrained genetic optimization of
Formula 1 front-wing geometries. Its aerodynamic evaluator is an empirical,
physics-informed surrogate with SI-unit normalization. It is not a validated
Navier–Stokes CFD solver, and the repository makes no reinforcement-learning
performance claim.

## Repository layout

- `src/alphadesign/`: installable Python package and the supported optimizer
- `tests/`: package and safety regression tests
- `configs/`: runtime and test configuration JSON
- `docs/`: architecture, integration notes, flowcharts, roadmap, and regulation references
- `data/`: source coordinates and reference input data
- `examples/legacy/`: one-off scripts retained for provenance
- `artifacts/legacy/`: generated meshes, logs, checkpoints, and model weights
- `paper/legacy/`: historical paper files and figures

Legacy artifacts are not paper evidence. Regenerate results with a recorded
configuration, seed, evaluation budget, and independent validation before
using them in a paper.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

The package can also be used directly from a checkout with `PYTHONPATH=src`.

```bash
python -m alphadesign --help
alphadesign --dry-run --config configs/config.json
pytest
```

The experiment recorder currently writes reproducibility metadata only. It
does not execute random-search, GA-only, or neural-guided-GA comparisons and
does not fabricate their results. Any neural guidance is disabled by default
because the current pipeline does not collect action/reward trajectories.

## Validation limits

Tests cover package imports, geometry units and sanity checks, failed-evaluation
handling, fitness conventions, policy-gating behavior, and experiment
metadata. They do not establish high-fidelity CFD agreement, physical wind
tunnel validity, or paper-level benchmark results.

See [the architecture notes](docs/architecture.md), [the integration guide](docs/integration/CFD_JSON_INTEGRATION.md), and [the roadmap](docs/roadmap.md) for current scope and open work.
