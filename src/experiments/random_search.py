"""Equal-budget random-search driver."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:  # support ``python src/experiments/random_search.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (  # noqa: E402
    RunRecorder, default_design, make_surrogate_evaluator, metadata,
    sample_variant, seed_everything,
)


def run_random_search(
    seed: int,
    budget: int,
    output_dir: str | Path,
    evaluator=None,
) -> RunRecorder:
    output = Path(output_dir)
    rng = seed_everything(seed)
    recorder = RunRecorder(output, metadata("random_search", seed, budget, 1, output))
    evaluate = evaluator or make_surrogate_evaluator(output)
    base = default_design()
    for evaluation in range(budget):
        # Both phase-one strategies evaluate this exact base design first.
        design = base if evaluation == 0 else sample_variant(base, rng)
        recorder.evaluate_or_replay(
            evaluation, design, evaluate,
            context={"generation": 0, "proposal": "random_search"},
        )
    recorder.finish()
    return recorder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_random_search(args.seed, args.budget, args.output)
    print(f"completed random_search: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
