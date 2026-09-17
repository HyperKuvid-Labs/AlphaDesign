"""Regression tests for the data-fit ground-effect formula in formula_constraints.py.

This is the formula that actually drives every optimization result under
`evaluate_formula_constratins` (random_search.py / ga_only.py / the whole
phase-one and phase-two protocol). It is a separate implementation from
cfd_analysis.py::calculate_ground_effect (used only by the real-mesh CFD
evaluator, evaluate_cfd_perf) and needed its own, independent fix; see
docs/validation/README.md for the full story and tests/test_ground_effect_formula.py
for the parallel tests on the other file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.common import default_design
from alphadesign.formula_constraints import F1FrontWingParams, F1FrontWingAnalyzer


def _ground_effect_factor(root_chord_mm):
    design = dict(default_design())
    design["root_chord"] = root_chord_mm
    params = F1FrontWingParams(**design)
    analyzer = F1FrontWingAnalyzer(params)
    return analyzer.compute_ground_effect_physics()


def test_ground_effect_factor_is_smooth_and_bounded_over_full_hc_range():
    # Sweep root_chord far beyond the optimizer's actual bounds (250-330mm)
    # to exercise the full h/c range, including the old branch points at
    # h/c = 0.1 and 0.3 (ground_clearance_ref is fixed at 75mm).
    chords_mm = [50 + i * 5 for i in range(150)]  # h/c from ~1.5 down to ~0.032
    factors = []
    for chord in chords_mm:
        result = _ground_effect_factor(chord)
        factors.append(result["ground_effect_factor"])
        assert result["ground_effect_factor"] >= 1.0

    steps = [abs(b - a) for a, b in zip(factors, factors[1:])]
    assert max(steps) < 0.15, "no branch-point discontinuities across the swept range"


def test_ground_effect_peaks_near_measured_value_not_at_zero():
    # Sweep chord finely to find where ground_effect_factor peaks, at fixed
    # ground_clearance_ref = 0.075 m (75 mm).
    chords_mm = [10 + i * 0.5 for i in range(2000)]
    results = [(c, _ground_effect_factor(c)) for c in chords_mm]
    peak_chord, peak_result = max(results, key=lambda item: item[1]["ground_effect_factor"])
    peak_hc = peak_result["height_to_chord_ratio"]
    # Zerihan (2001): peak at h/c = 0.082, ground_effect_factor there ~ 2.49.
    assert abs(peak_hc - 0.082) < 0.02
    assert abs(peak_result["ground_effect_factor"] - 2.49) < 0.15


def test_search_space_chords_give_monotonic_sensible_values():
    # The optimizer only ever varies root_chord in [250, 330] mm, all of
    # which sit on the falling side of the peak (h/c in [0.227, 0.30]).
    # Smaller chord (larger h/c, farther from peak) must give a smaller
    # ground-effect factor than larger chord (smaller h/c, closer to peak).
    low = _ground_effect_factor(250)["ground_effect_factor"]
    mid = _ground_effect_factor(290)["ground_effect_factor"]
    high = _ground_effect_factor(330)["ground_effect_factor"]
    assert low < mid < high
