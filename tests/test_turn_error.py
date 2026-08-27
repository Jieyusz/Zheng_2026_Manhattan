"""
Turn-error regression tests (Step 4 — trajectory.py).

Turn error = fraction of hole-turns made in the wrong allocentric direction
(Bout.count_error + utils.get_hole_correctness).  This is a directly-published
quantity underlying every learning curve, so these values pin current behavior.

Notes
-----
- The recommended metric is ``include="first"`` (the default): one first-decision trial
  per shortest-path hole, approach-conditioned, with an exact chance level of 0.5. The
  synthetic paths here approach each hole on the correct corridor, so approach-conditioned
  and raw scoring agree; the paths that exercise wrong-corridor exclusion live in
  ``test_turn_error_approach.py``.
- ``include="approach"`` / ``include="all"`` are deprecated (inaccurate — they pool every
  crossing, so reversals inflate the denominator) and emit a ``DeprecationWarning``; the
  calls below assert both the preserved value and the warning via ``pytest.warns``.
- Mask D has no turn-error measure and must raise (Bout.count_error guard).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.mask import Mask
from manhattan_maze.trajectory import Bout

HOLES_A = np.array([[3, 5], [3, 8], [4, 8], [4, 9], [6, 9], [6, 2], [1, 2], [1, 1], [5, 1]])
HOME = (0, 5, 0)
OUT = (5, 9, 1)
SIZE = 11
FPS = 30


@pytest.fixture(scope="module")
def mask_a():
    return Mask(HOLES_A, SIZE, "A", HOME, OUT)


def make_bout(path, mask):
    df = pd.DataFrame(
        {
            "in_frame": [i * 10 for i in range(len(path))],
            "out_frame": [i * 10 + 9 for i in range(len(path))],
            "discrete_loc": [tuple(p) for p in path],
        }
    )
    return Bout(df, mask=mask, idx=0, session=None, trajectory=None, FPS=FPS)


class TestTurnErrorRate:
    def test_correct_turn_rate_zero(self, mask_a):
        # Approach hole (3,5) along row 5 and turn North (correct outbound turn).
        bout = make_bout([(0, 5), (1, 5), (2, 5), (3, 5), (3, 6), (3, 7)], mask_a)
        assert bout.get_allocentric_turns() == [((3, 5), "N")]
        assert bout.count_error(unit="turn", error_type="rate") == 0.0  # approach default
        with pytest.warns(DeprecationWarning):
            assert bout.count_error(unit="turn", include="all", error_type="rate") == 0.0

    def test_one_wrong_turn_rate_one(self, mask_a):
        # Approach hole (3,5) along row 5 and turn South (wrong; correct is North).
        bout = make_bout([(0, 5), (1, 5), (2, 5), (3, 5), (3, 4), (3, 3)], mask_a)
        assert bout.get_allocentric_turns() == [((3, 5), "S")]
        assert bout.count_error(unit="turn", error_type="rate") == 1.0  # approach default
        with pytest.warns(DeprecationWarning):
            assert bout.count_error(unit="turn", include="all", error_type="rate") == 1.0
        with pytest.warns(DeprecationWarning):
            assert bout.count_error(unit="turn", include="all", error_type="count") == 1

    def test_mixed_turns_rate_half(self, mask_a):
        # Hole (3,5): North (correct); hole (3,8): West (wrong, correct is East).
        bout = make_bout([(0, 5), (1, 5), (2, 5), (3, 5), (3, 6), (3, 7), (3, 8), (2, 8), (1, 8)], mask_a)
        assert bout.get_allocentric_turns() == [((3, 5), "N"), ((3, 8), "W")]
        assert bout.count_error(unit="turn", error_type="rate") == 0.5  # approach default

    def test_no_holes_visited_rate_nan(self, mask_a):
        # Straight run along home corridor visiting no hole -> no scored turns.
        bout = make_bout([(0, 5), (1, 5), (2, 5)], mask_a)
        assert bout.get_allocentric_turns() == []
        rate = bout.count_error(unit="turn", error_type="rate")  # approach default
        assert np.isnan(rate)


class TestTurnErrorGuards:
    def test_unvisited_holes_give_nan(self, mask_a):
        # include="first": vector spans all 9 holes, NaN for unvisited.
        bout = make_bout([(0, 5), (1, 5), (2, 5), (3, 5), (3, 6), (3, 7)], mask_a)
        vec = bout.get_hole_correctness_vec(include="first")
        assert len(vec) == len(HOLES_A)
        assert vec[0] == 1.0  # hole (3,5) visited, correct
        assert np.isnan(vec[1:]).all()  # other holes unvisited
        # include="first" is the recommended metric, not deprecated -> must NOT warn. The one
        # visited hole (3,5) was taken correctly, so the first-decision rate is 0.0 (the 8
        # unvisited holes are NaN in the vec and contribute no trials); the all-unvisited
        # case (rate NaN) is covered by test_no_holes_visited_rate_nan.
        assert bout.count_error(unit="turn", include="first", error_type="rate") == 0.0

    def test_mask_d_raises(self):
        mask_d = Mask(HOLES_A, SIZE, "D", HOME, OUT)
        bout = make_bout([(0, 5), (1, 5), (2, 5)], mask_d)
        with pytest.raises(ValueError, match="Mask D"):
            bout.count_error(unit="turn")

    def test_unknown_unit_raises(self, mask_a):
        bout = make_bout([(0, 5), (1, 5), (2, 5)], mask_a)
        with pytest.raises(ValueError, match="Unknown unit"):
            bout.count_error(unit="tile")
