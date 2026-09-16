# phase-one experiments

These drivers compare random search and a genetic algorithm under the same
number of calls to the formula-based empirical surrogate. A design must pass
the evaluator, reach 0.60 compliance, produce at least 500 N of predicted
downforce, and meet the existing critical safety, buckling, and
natural-frequency thresholds. The primary objective is to maximize
`computed_efficiency`. This objective and feasibility rule are phase-one pilot
choices and must be frozen or reconsidered before the final paper protocol.

The runs do not generate STL files, call OpenFOAM, or claim physical
validation. Only the per-run `cfd_results` directory is reserved (the formula
path should leave it empty); `cfd_temp_files` and `f1_wing_output` are not
created or reused by these drivers. Each output directory contains immutable
`metadata.json` and append-only `evaluations.jsonl`, plus a derived
`summary.json`. A rerun with the same arguments resumes by replaying existing
records and does not duplicate evaluator calls. Each evaluation records a
deterministic `design_id`, measured `duration_seconds`, and strategy context.

From the repository root, run each condition separately:

```sh
python src/experiments/random_search.py \
  --seed 0 --budget 100 \
  --output artifacts/phase_one/random_search/seed_0
```

```sh
python src/experiments/ga_only.py \
  --seed 0 --budget 100 --population-size 20 \
  --output artifacts/phase_one/ga_only/seed_0
```

For the planned three matched seeds, repeat those commands with `--seed 1` and
`--seed 2` and distinct output directories. For a smoke run, use
`--budget 4 --population-size 2`.
