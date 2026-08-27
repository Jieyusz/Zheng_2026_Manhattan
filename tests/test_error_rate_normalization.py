"""
Tests for the per-step corridor/tile error *rate* (calculate_seq_error_rate).

The rate is the per-step version of calculate_seq_error, using the non-decreasing
(``>= 0``) rule of localize_distance_seq — so pooled over a whole sequence it must
equal the error_propagation corridor error rate (``sum(counts) / sum(opps)``). These
tests pin that parity and the [0, 1] / chance-~0.5 properties, and confirm the legacy
count path is unchanged.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.analysis import (calculate_seq_error, calculate_seq_error_rate,
                                      localize_distance_seq)


def test_monotone_toward_goal_is_zero_rate():
    # strictly decreasing distance-to-goal => every step progresses => rate 0
    seq = [5, 4, 3, 2, 1, 0]
    assert calculate_seq_error_rate(seq) == 0.0


def test_monotone_away_from_goal_is_unit_rate():
    # strictly increasing distance => every step is non-progress => rate 1
    seq = [0, 1, 2, 3, 4, 5]
    assert calculate_seq_error_rate(seq) == 1.0


def test_matches_diff_ge_zero_over_steps():
    seq = [3, 2, 2, 3, 1, 1, 0]  # includes a flat (>=0) and an away step
    steps = np.diff(seq)
    expected = np.count_nonzero(steps >= 0) / (len(seq) - 1)
    assert calculate_seq_error_rate(seq) == pytest.approx(expected)
    assert 0.0 <= calculate_seq_error_rate(seq) <= 1.0


@pytest.mark.parametrize("seq", [[], [4]])
def test_too_short_is_nan(seq):
    assert np.isnan(calculate_seq_error_rate(seq))


def test_parity_with_localize_distance_seq_pooled():
    # Pooled over positions, localize_distance_seq's sum(counts)/sum(opps) must equal
    # the scalar rate whenever every distance sits within [0, n_pos) (all steps scored).
    rng = np.random.default_rng(0)
    for _ in range(200):
        seq = rng.integers(0, 9, size=int(rng.integers(2, 20))).tolist()
        n_pos = max(seq) + 1
        counts, opps = localize_distance_seq(seq, n_pos)
        pooled = counts.sum() / opps.sum()
        assert calculate_seq_error_rate(seq) == pytest.approx(pooled)


def test_count_path_uses_strict_rule_and_is_unchanged():
    # The legacy count uses strict > 0 (flat steps are NOT errors); the rate uses >= 0.
    # A sequence with a flat step distinguishes them.
    seq = [2, 2, 1, 0]                      # one flat step (2->2), then progress
    assert calculate_seq_error(seq) == 0    # strict >0: no away-step
    # rate counts the flat step as non-progress: 1 of 3 steps
    assert calculate_seq_error_rate(seq) == pytest.approx(1 / 3)
