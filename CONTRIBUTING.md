# Contributing

AlphaDesign is a Python research prototype for constrained genetic
optimization using an empirical aerodynamic surrogate. The supported source
layout is:

- `src/alphadesign/`: installable package
- `tests/`: regression and component tests
- `configs/`: runtime and test configuration
- `docs/`: architecture and integration notes
- `examples/legacy/`: historical one-off scripts

The surrogate is not a validated high-fidelity CFD solver. Keep descriptions
of results proportional to the tests and measurements that support them.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Run the supported checks from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m alphadesign --help
pytest
git diff --check
```

The package can be imported directly from a checkout with `PYTHONPATH=src`.
Use `configs/config.json` as the starting runtime configuration. Keep
generated outputs outside the source tree or under the ignored artifact
locations described in the root README.

## Code and documentation changes

Keep changes focused and preserve existing numerical behavior unless the
change is explicitly about that behavior. Add a focused regression test for
new safety or API behavior. Use package imports (`alphadesign...`) in tests
and avoid importing modules through the historical `RL/` paths.

Documentation should identify surrogate predictions, planned metadata, and
legacy artifacts clearly. Do not present unexecuted experiment records or
legacy generated outputs as paper evidence.
