"""Equal-budget genetic-algorithm-only driver (neural guidance disabled)."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

if __package__ in {None, ""}:  # support ``python src/experiments/ga_only.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (  # noqa: E402
    RunRecorder, crossover, default_design, design_key, make_surrogate_evaluator,
    metadata, mutate, sample_variant, seed_everything,
)


def _ranked(population):
    def objective(item):
        value = item[1].get("objective")
        return value if isinstance(value, (int, float)) else float("-inf")

    return sorted(
        population,
        key=lambda item: (bool(item[1].get("valid")), objective(item)),
        reverse=True,
    )


def run_ga_only(
    seed: int,
    budget: int,
    output_dir: str | Path,
    population_size: int = 20,
    evaluator=None,
) -> RunRecorder:
    output = Path(output_dir)
    rng = seed_everything(seed)
    recorder = RunRecorder(output, metadata("ga_only", seed, budget, population_size, output))
    evaluate = evaluator or make_surrogate_evaluator(output)
    base = default_design()
    population = []
    seen = set()

    for evaluation in range(budget):
        if evaluation < population_size:
            design = base if evaluation == 0 else sample_variant(base, rng)
        else:
            ranked = _ranked(population)
            parent_a = ranked[rng.randrange(min(len(ranked), max(2, population_size)))]
            parent_b = ranked[rng.randrange(min(len(ranked), max(2, population_size)))]
            design = mutate(crossover(parent_a[0], parent_b[0], rng), rng)
            # Avoid replaying a completed design when the bounded space makes
            # crossover/mutation collide.  This is only a proposal safeguard;
            # every accepted proposal still consumes exactly one call.
            for _ in range(100):
                if design_key(design) not in seen:
                    break
                design = mutate(sample_variant(base, rng), rng)

        result = recorder.evaluate_or_replay(
            evaluation,
            design,
            evaluate,
            context={
                "generation": evaluation // population_size,
                "proposal": (
                    "initial_population"
                    if evaluation < population_size
                    else "crossover_mutation"
                ),
            },
        )
        seen.add(design_key(design))
        population.append((copy.deepcopy(design), result))
        population = _ranked(population)[:population_size]

    recorder.finish()
    return recorder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_ga_only(args.seed, args.budget, args.output, args.population_size)
    print(f"completed ga_only: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
