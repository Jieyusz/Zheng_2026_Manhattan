"""
Tests for the reversal-based forward-bias estimator and the first-journey curve.

Covers ``random_walk.reversal_decisions`` / ``forward_bias_mle`` / ``reversal_forward_bias``
and ``analysis.first_journey_forward_bias_curve`` (fig:oa_supp I, fig:ac_oa_supp I), none of
which had any coverage before.

The curve's two defining properties are pinned here, because both are easy to regress:

  * **one smoothing stage** -- each plotted point is a single MLE over the decisions inside
    its own window, so it must equal an independent fit over that window. Any post-hoc
    smoothing (a moving average over neighbouring points, as an earlier version applied)
    breaks this equality.
  * **windows truncated at the ends of the journey** -- so the curve spans a true 0 to 1.
    Placing windows by their centre, as an earlier version did, left the first and last
    ``win / 2`` of every journey unmeasured.

Session objects are stubbed (see ``_FakeSession``) so the tests need no data files, matching
tests/test_random_walk.py and tests/test_mask_golden.py.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.analysis import first_journey_corridor_seq, first_journey_forward_bias_curve
from manhattan_maze.random_walk import (
    forward_bias_mle,
    reversal_decisions,
    reversal_forward_bias,
)


# --------------------------------------------------------------------------------------
# stubs: the estimator only needs a corridor sequence and the mask's corridor degrees
# --------------------------------------------------------------------------------------
def path_adjacency(n_nodes):
    """Adjacency of a path graph 0-1-...-(n-1): interior degree 2, both ends degree 1."""
    adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
    index = np.arange(n_nodes - 1)
    adjacency[index, index + 1] = 1
    adjacency[index + 1, index] = 1
    return adjacency


def path_degrees(n_nodes):
    adjacency = path_adjacency(n_nodes)
    active = np.where(adjacency.sum(0) > 0)[0]
    return {int(i): int((adjacency[i, active] > 0).sum()) for i in active}


class _FakeMask:
    def __init__(self, n_nodes):
        self.corridors_adj_mat = path_adjacency(n_nodes)


class _FakeBout:
    def __init__(self, corridors):
        self._corridors = list(corridors)

    def get_corridors(self):
        return self._corridors

    def is_sortie(self):
        return False

    def is_outbound(self):
        return False


class _FakeSession:
    """
    One session whose entire first journey is a single traverse bout.

    Implements just the Session surface that ``first_journey_corridor_seq`` uses:
    ``get_traverse_indices`` to locate traverse 0, ``slice_by_traverse_idx`` to take every
    bout up to and including it, and ``concat_corridors_df`` for the corridor column. With a
    single bout that *is* traverse 0, the slice ``bouts[0:1]`` is the whole stub, so
    ``slice_by_traverse_idx`` returns ``self``.
    """

    def __init__(self, corridor_seq, n_nodes):
        self.mask = _FakeMask(n_nodes)
        self._bouts = [_FakeBout(corridor_seq)]

    def __getitem__(self, index):
        return self._bouts[index]

    def get_traverse_indices(self):
        return [[0]]

    def slice_by_traverse_idx(self, start_traverse_idx=None, end_traverse_idx=0):
        return self

    def concat_corridors_df(self):
        return pd.DataFrame(
            {"corridor": np.concatenate([b.get_corridors() for b in self._bouts])})


def biased_walk(n_steps, beta, n_nodes, rng):
    """
    Random walk on a path graph with forward bias ``beta``.

    At an interior node (degree 2) the walker reverses with probability
    ``p_rev(beta, 2) = 1 - beta``; at either end the reversal is forced (and such nodes are
    excluded from scoring, so they do not bias the estimate).
    """
    walk = [n_nodes // 2, n_nodes // 2 + 1]
    while len(walk) < n_steps:
        current, previous = walk[-1], walk[-2]
        if current == 0:
            walk.append(1)
        elif current == n_nodes - 1:
            walk.append(n_nodes - 2)
        else:
            forward = current + (current - previous)
            walk.append(previous if rng.random() > beta else forward)
    return np.array(walk)


# --------------------------------------------------------------------------------------
# reversal_decisions / forward_bias_mle
# --------------------------------------------------------------------------------------
def test_reversal_decisions_scores_only_interior_nodes_of_degree_two():
    degrees = path_degrees(5)  # 0 and 4 are dead ends
    # 1 <-> 0 bounces off the degree-1 end, so node 0 must not be scored
    positions, is_reversal, scored_degrees = reversal_decisions([1, 0, 1, 2, 3], degrees)
    assert positions.tolist() == [2, 3]          # node 0 (dead end) and the two ends dropped
    assert is_reversal.tolist() == [False, False]
    assert scored_degrees.tolist() == [2.0, 2.0]


def test_path_graph_reduces_to_one_minus_reversal_rate():
    """With every scored degree equal to 2, beta_hat = 1 - R/N exactly (random_walk.py:385)."""
    degrees = path_degrees(10)
    #                  t: 1  2  3  4  5  6
    node_seq = [3, 4, 3, 4, 5, 6, 5, 6]
    _, is_reversal, scored_degrees = reversal_decisions(node_seq, degrees)
    n_reversals, n_scored = int(is_reversal.sum()), is_reversal.size
    assert forward_bias_mle(is_reversal, scored_degrees) == pytest.approx(
        1 - n_reversals / n_scored)


def test_boundary_and_empty_returns():
    degrees = path_degrees(10)
    never = reversal_decisions([2, 3, 4, 5, 6], degrees)
    always = reversal_decisions([3, 4, 3, 4, 3], degrees)
    assert forward_bias_mle(*never[1:]) == 1.0
    assert forward_bias_mle(*always[1:]) == 0.0
    assert np.isnan(forward_bias_mle(np.array([], dtype=bool), np.array([])))
    assert np.isnan(reversal_forward_bias([4, 5], degrees))  # no interior decision


def test_wrapper_agrees_with_the_split_api():
    degrees = path_degrees(20)
    rng = np.random.default_rng(0)
    node_seq = biased_walk(200, 0.65, 20, rng)
    assert reversal_forward_bias(node_seq, degrees) == pytest.approx(
        forward_bias_mle(*reversal_decisions(node_seq, degrees)[1:]))


def test_mle_recovers_a_known_bias_on_a_path_graph():
    degrees = path_degrees(40)
    rng = np.random.default_rng(7)
    node_seq = biased_walk(20000, 0.7, 40, rng)
    assert reversal_forward_bias(node_seq, degrees) == pytest.approx(0.7, abs=0.02)


# --------------------------------------------------------------------------------------
# first_journey_forward_bias_curve
# --------------------------------------------------------------------------------------
def test_grid_is_closed_and_shape_is_stable():
    rng = np.random.default_rng(1)
    sessions = [_FakeSession(biased_walk(400, 0.7, 40, rng), 40) for _ in range(4)]
    curve = first_journey_forward_bias_curve(sessions, n_points=18)
    assert curve.shape == (3, 18)
    np.testing.assert_allclose(curve[0], np.linspace(0, 1, 18))
    assert curve[0][0] == 0.0 and curve[0][-1] == 1.0


def test_same_mode_spans_the_whole_journey():
    """Regression: window *centres* could never reach the ends, leaving both extremes NaN."""
    rng = np.random.default_rng(2)
    sessions = [_FakeSession(biased_walk(400, 0.7, 40, rng), 40) for _ in range(4)]
    _, mean, se = first_journey_forward_bias_curve(sessions, mode="same")
    assert not np.isnan(mean).any(), "every point on a closed 0-1 grid should be measured"
    assert not np.isnan(se).any()


def test_valid_mode_keeps_only_fully_supported_points():
    """
    Default ``mode="valid"`` mirrors ``moving_average(..., mode="valid")`` as used for the
    smoothed solid lines of fig:ac_mem_gen A: a position is reported only if its whole
    window fits inside the journey. At the defaults that is x in [0.10, 0.90].
    """
    rng = np.random.default_rng(6)
    sessions = [_FakeSession(biased_walk(400, 0.7, 40, rng), 40) for _ in range(4)]
    grid, mean, se = first_journey_forward_bias_curve(sessions)
    reported = ~np.isnan(mean)
    expected = (grid >= 0.10) & (grid <= 0.90)
    np.testing.assert_array_equal(reported, expected)
    assert grid[reported][0] == pytest.approx(2 / 17)   # two points dropped per side
    assert grid[reported][-1] == pytest.approx(15 / 17)
    np.testing.assert_array_equal(np.isnan(se), ~expected)
    # where both modes report a point, they must agree exactly -- "valid" only masks
    _, same_mean, _ = first_journey_forward_bias_curve(sessions, mode="same")
    np.testing.assert_allclose(mean[reported], same_mean[reported])


def test_invalid_mode_rejected():
    sessions = [_FakeSession(biased_walk(200, 0.6, 40, np.random.default_rng(0)), 40)]
    with pytest.raises(ValueError, match="mode must be"):
        first_journey_forward_bias_curve(sessions, mode="full")


def test_every_point_equals_an_independent_fit_over_its_own_window():
    """
    Pins BOTH defining properties at once.

    Recomputing a point from scratch -- pool the decisions within win/2 of it, fit once --
    must reproduce it exactly. This fails if any post-hoc smoothing is reintroduced (the
    value would mix in neighbouring points) and it fails if the windows are not truncated at
    the journey's ends (the extreme points would draw on different decisions).
    """
    rng = np.random.default_rng(3)
    walks = [biased_walk(300, 0.6, 40, rng) for _ in range(3)]
    sessions = [_FakeSession(walk, 40) for walk in walks]
    win, n_points = 0.20, 18
    # mode="same" so every point is exercised, including the truncated end windows
    grid, mean, _ = first_journey_forward_bias_curve(sessions, win=win, n_points=n_points,
                                                     mode="same")
    degrees = path_degrees(40)

    for col, centre in enumerate(grid):
        expected = []
        for walk in walks:
            corridor_seq = first_journey_corridor_seq(_FakeSession(walk, 40))
            positions, is_reversal, scored_degrees = reversal_decisions(corridor_seq, degrees)
            fractions = positions / (corridor_seq.size - 1)
            selected = np.abs(fractions - centre) <= win / 2
            if selected.sum() >= 2:
                expected.append(forward_bias_mle(is_reversal[selected],
                                                 scored_degrees[selected]))
        assert mean[col] == pytest.approx(np.mean(expected)), f"point {col} (x={centre:.3f})"


def test_truncated_ends_see_only_local_behaviour():
    """
    A journey that reverses only at its very start must give beta_hat = 0 at x = 0 and
    beta_hat = 1 at x = 1 -- i.e. each end window reads that end of the journey, and nothing
    is blended in from elsewhere.
    """
    corridor_seq = np.concatenate([np.tile([30, 31], 6), np.arange(32, 60)])
    sessions = [_FakeSession(corridor_seq, 60) for _ in range(2)]
    _, mean, _ = first_journey_forward_bias_curve(sessions, mode="same")
    assert mean[0] == 0.0, "the first window must contain only the early reversals"
    assert mean[-1] == 1.0, "the last window must contain only the late forward steps"


def test_min_animals_and_min_decisions_gates():
    rng = np.random.default_rng(4)
    sessions = [_FakeSession(biased_walk(400, 0.7, 40, rng), 40) for _ in range(4)]
    # min_animals above the cohort size masks the whole curve
    _, mean, _ = first_journey_forward_bias_curve(sessions, min_animals=99)
    assert np.isnan(mean).all()
    # an unreachable decision count does the same, via the per-animal gate
    _, mean, _ = first_journey_forward_bias_curve(sessions, min_decisions=10 ** 6)
    assert np.isnan(mean).all()


def test_short_journeys_are_skipped_without_error():
    """min_length drops animals outright; an all-skipped cohort returns an all-NaN curve."""
    sessions = [_FakeSession([3, 4, 5], 10) for _ in range(3)]
    grid, mean, se = first_journey_forward_bias_curve(sessions, min_length=6)
    np.testing.assert_allclose(grid, np.linspace(0, 1, 18))
    assert np.isnan(mean).all() and np.isnan(se).all()


def test_curve_recovers_a_flat_known_bias():
    """A stationary walker gives a flat curve at its true beta, within sampling error."""
    rng = np.random.default_rng(5)
    sessions = [_FakeSession(biased_walk(3000, 0.65, 40, rng), 40) for _ in range(8)]
    _, mean, _ = first_journey_forward_bias_curve(sessions)
    assert np.nanmean(mean) == pytest.approx(0.65, abs=0.03)
    # interior points are better supported than the truncated ends, so check them tightly
    np.testing.assert_allclose(mean[3:-3], 0.65, atol=0.08)
