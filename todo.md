# Paper Results TODO

The immediate goal is to establish whether the constraint-aware genetic
algorithm finds better feasible designs than random search under the same
surrogate-evaluation budget. OpenFOAM validation comes after this comparison is
reproducible.

## 1. Freeze the paper claim

- [ ] Use random search versus GA-only as the primary comparison.
- [ ] Treat frozen neural guidance as an optional ablation, not the main result.
- [ ] Do not make reinforcement-learning claims without recorded actions,
      rewards, and transitions.
- [ ] Define one primary objective and one feasibility rule before running the
      final experiments.

> **Phase-one pilot result:** Across three matched seeds and 100 surrogate
> evaluations per strategy, GA-only achieved a higher best predicted efficiency
> in all three seeds. Mean best efficiency increased from 4.971 for random
> search to 5.429 for GA-only (+9.2%), while the feasible-design rate increased
> from 42% to 63%. The gain came with lower predicted downforce, so this supports
> an efficiency claim rather than a higher-downforce claim. These runs used the
> empirical surrogate and are a pilot rather than the final paper protocol
> (§4). Raw metadata, JSONL records, and summaries are stored under
> [`artifacts/phase_one/`](artifacts/phase_one/).
>
> ![Phase-one pilot: GA-only vs. random search](paper/figures/phase_one_results.png)

## 2. Build the executable benchmark

- [ ] Implement random-search and GA-only drivers using the same evaluator,
      parameter bounds, and initial sampling distribution.
- [ ] Enforce the budget at the evaluator-call level rather than estimating it
      from generations multiplied by population size.
- [ ] Disable neural-network training and guidance for the GA-only condition.
- [ ] Record each design vector, constraint result, aerodynamic result, failure
      reason, seed, runtime, strategy, and evaluation number.
- [ ] Write immutable JSONL evaluation records and complete run metadata.
- [ ] Add restart support without duplicating completed evaluations.
- [ ] Add tests for exact budget enforcement, deterministic replay, failed
      evaluations, and duplicate prevention.

## 3. Run an integrity pilot

Run two strategies, three matched seeds, and 100 evaluations per strategy and
seed. This pilot uses the empirical surrogate and must not invoke OpenFOAM.

- [ ] Confirm exactly 100 evaluator calls per run.
- [ ] Confirm identical seeds reproduce identical results.
- [ ] Confirm there are no NaN or infinite values.
- [ ] Confirm failed designs remain marked as failed.
- [ ] Confirm every record contains its configuration and Git commit.
- [ ] Confirm every run finishes without manual intervention.
- [ ] Inspect runtime and between-seed variance before choosing the final budget.

## 4. Freeze and run the full protocol

- [x] Select 10 to 20 matched seeds based on pilot variance.
- [x] Select a fixed budget, initially considering 500 to 1,000 evaluations per
      strategy and seed.
- [x] Freeze the configuration before inspecting the final outcomes.
- [x] Report best feasible objective at the final budget as the primary metric.
- [x] Also report anytime performance, feasible-design rate, evaluations to a
      fixed threshold, invalid-design rate, failure rate, and wall-clock time.
- [x] Compare paired seed-level results using confidence intervals and an effect
      size. Do not treat evaluations within one run as independent samples.

> **Full-protocol run:** 20 matched seeds, 1,000 evaluations per strategy per
> seed, empirical surrogate only.
>
> Best feasible objective at the final budget: mean best efficiency was 5.19
> for random search versus 6.31 for GA-only. Feasible-design rate was 41.0%
> for random search versus 83.7% for GA-only; the rest were infeasible
> designs, not evaluator failures (0 execution failures across all 40,000
> evaluations for either strategy).
>
> Anytime performance (mean best-so-far across the 20 seeds, at evaluations
> 50 / 100 / 250 / 500 / 750 / 1,000): random search reached 4.72 / 4.86 /
> 5.01 / 5.07 / 5.14 / 5.19; GA-only reached 5.00 / 5.61 / 6.14 / 6.29 / 6.30 /
> 6.31. GA-only's average result after only 100 evaluations already exceeds
> random search's average result at the full 1,000-evaluation budget.
>
> Evaluations to reach a fixed threshold (5.19, random search's own final
> mean): GA-only reached it in all 20 seeds, after a mean of 56 evaluations
> (median 59). Random search reached it in only 6 of 20 seeds within the full
> 1,000-evaluation budget, taking a mean of 689 evaluations (median 722) when
> it did.
>
> Wall-clock time: the surrogate evaluator itself is cheap for both
> strategies, about 0.20 seconds of evaluator time per 1,000-evaluation run
> (about 0.20 ms per evaluation) for random search and GA-only alike.
>
> Paired seed-level comparison (20 matched seeds, best objective at the final
> budget): GA-only beat random search on 20 of 20 seeds. Mean paired gap
> +1.118 (sd 0.121), 95% CI [1.06, 1.17] (bootstrap and paired t-test agree),
> Cohen's dz = 9.22, paired t(19) = 41.2 (p < 0.0001), Wilcoxon signed-rank
> W = 0 (p < 0.0001).
>
> Raw metadata, JSONL records, and summaries are stored under
> [`artifacts/phase_two/`](artifacts/phase_two/).
>
> ![Full-protocol run: GA-only vs. random search](paper/figures/phase_two_results.png)

## 5. Validate physical credibility

Do not run OpenFOAM during every optimization evaluation. Select approximately
12 to 20 representative cases after the surrogate experiments:

- [ ] Baseline geometry.
- [ ] Best designs from both strategies.
- [ ] Median-performing and poor-but-feasible designs.
- [ ] Geometrically diverse designs.
- [ ] Designs near constraint boundaries.

Run these cases on remote CPU infrastructure rather than the laptop.

- [ ] Record the OpenFOAM version, solver, turbulence model, domain, boundary
      conditions, mesh quality, residuals, force convergence, runtime, and
      hardware.
- [ ] Run coarse, medium, and fine mesh checks for at least three representative
      cases.
- [ ] Keep calibration and held-out validation cases separate.
- [ ] Compare the surrogate and OpenFOAM using rank correlation, force error,
      systematic bias, and agreement among the top-ranked designs.
- [ ] If agreement is weak, limit the paper claim to optimization under the
      empirical surrogate.

## 6. Generate paper artifacts reproducibly

- [ ] Generate all tables and plots directly from immutable raw records.
- [ ] Plot anytime performance with confidence bands.
- [ ] Plot final-result distributions across seeds.
- [ ] Plot feasible-design rates.
- [ ] Plot surrogate-versus-OpenFOAM agreement and error.
- [ ] Produce statistical and validation summary tables.
- [ ] Export representative STL renders with traceable design identifiers.
- [ ] Archive the exact configuration, source commit, raw records, and analysis
      script used for every reported result.

## Completion gate

The Results section is ready to write only when the full runs are reproducible,
the statistical comparisons operate on independent seeds, and every physical
claim is bounded by the available external validation.
