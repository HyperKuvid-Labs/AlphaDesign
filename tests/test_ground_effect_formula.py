"""Regression tests for the data-fit ground-effect formula.

The formula is fit to real digitized/quoted CL data from J. Zerihan's PhD
thesis (Univ. of Southampton, 2001); see cfd_analysis.py::calculate_ground_effect
for the full citation and docs/validation/README.md for the fit quality.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alphadesign.cfd_analysis import STLWingAnalyzer


def _analyzer():
    analyzer = STLWingAnalyzer.__new__(STLWingAnalyzer)
    analyzer.chord_lengths = [0.28, 0.22, 0.18, 0.14]
    return analyzer


def _factor_at_hc(analyzer, hc, element_idx):
    chord = analyzer.chord_lengths[element_idx]
    clearance_mm = hc * chord * 1000
    return analyzer.calculate_ground_effect(clearance_mm, element_idx)


def test_main_element_peak_matches_measured_data():
    analyzer = _analyzer()
    hs = [0.001 + 0.001 * i for i in range(1500)]
    values = [_factor_at_hc(analyzer, h, 0) for h in hs]
    peak_hc = hs[values.index(max(values))]
    # Zerihan (2001) measured the peak at h/c = 0.082, CL = 1.72 = 2.49x freestream.
    assert abs(peak_hc - 0.082) < 0.02
    assert abs(max(values) - 2.49) < 0.1


def test_ground_effect_is_bounded_and_continuous():
    analyzer = _analyzer()
    # Fine grid at two resolutions: a genuine discontinuity keeps the same
    # step size regardless of resolution, while a smooth function's step
    # size shrinks proportionally as the grid is refined.
    coarse_hs = [0.001 + 0.002 * i for i in range(1000)]
    fine_hs = [0.001 + 0.0002 * i for i in range(10000)]
    for element_idx in range(4):
        coarse_values = [_factor_at_hc(analyzer, h, element_idx) for h in coarse_hs]
        fine_values = [_factor_at_hc(analyzer, h, element_idx) for h in fine_hs]
        assert all(v >= 1.0 for v in coarse_values), "ground effect must never reduce lift below freestream"
        assert all(v <= 2.5 for v in coarse_values), "capped maximum must hold"
        coarse_max_step = max(abs(b - a) for a, b in zip(coarse_values, coarse_values[1:]))
        fine_max_step = max(abs(b - a) for a, b in zip(fine_values, fine_values[1:]))
        assert fine_max_step < coarse_max_step / 5, "no branch-point discontinuities"


def test_falls_off_approaching_the_ground_and_far_from_it():
    analyzer = _analyzer()
    peak = _factor_at_hc(analyzer, 0.082, 0)
    very_close = _factor_at_hc(analyzer, 0.01, 0)
    far = _factor_at_hc(analyzer, 2.0, 0)
    assert very_close < peak, "must fall below the peak approaching the ground, not keep rising"
    assert far < peak
    assert far < 1.2, "far from the ground the multiplier should approach 1.0 (no ground effect)"


def test_flap_elements_are_weaker_than_main_but_never_below_freestream():
    analyzer = _analyzer()
    main_peak = _factor_at_hc(analyzer, 0.082, 0)
    for element_idx in (1, 2, 3):
        flap_peak = _factor_at_hc(analyzer, 0.082, element_idx)
        assert 1.0 <= flap_peak < main_peak
    flap1 = _factor_at_hc(analyzer, 0.082, 1)
    flap2 = _factor_at_hc(analyzer, 0.082, 2)
    assert flap2 < flap1, "attenuation should increase further from the main element"
