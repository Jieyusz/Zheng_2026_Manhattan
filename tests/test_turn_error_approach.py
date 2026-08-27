"""
Approach-conditioned turn-error tests.

The default turn-error metric scores a hole crossing only when the mouse enters
on the shortest-path corridor at that hole (``Bout.count_error(include="approach")``).
A crossing entered on the perpendicular corridor cannot reach the correct
direction, so it is a forced error; conditioning excludes it, giving an exact
0.5 chance level. These tests pin that behavior and the ground-truth approach map.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze import utils
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


class TestCorrectApproachMap:
    def test_matches_correct_turns_and_records_approach(self, mask_a):
        cmap = mask_a.correct_approach_map(homebound=False)
        correct_turns = mask_a.get_correct_turns(homebound=False)
        # Same keys, and the exit direction matches get_correct_turns exactly.
        assert set(cmap) == set(correct_turns)
        assert {h: exit_dir for h, (_appr, exit_dir) in cmap.items()} == correct_turns
        # Shortest path enters (3,5) heading East (from home along row 5) and exits North.
        assert cmap[(3, 5)] == ("E", "N")


class TestAllocentricTurnsWithApproach:
    def test_records_approach_and_exit(self, mask_a):
        # Enter (3,5) along row 5 (approach E), exit North.
        bout = make_bout([(0, 5), (1, 5), (2, 5), (3, 5), (3, 6), (3, 7)], mask_a)
        assert bout.get_allocentric_turns_with_approach() == [((3, 5), "E", "N")]

    def test_tolerance_not_supported(self):
        with pytest.raises(NotImplementedError):
            utils.allocentric_turns_with_approach([(0, 0), (1, 0), (1, 1)], {(1, 0)}, tolerance=1)


class TestApproachConditioning:
    def test_wrong_corridor_crossing_excluded(self, mask_a):
        # Visit (3,5) twice:
        #   1) correct approach E -> exit N (correct),
        #   2) after backtracking up the vertical corridor, approach S -> exit E.
        # The 2nd crossing enters on the wrong (vertical) corridor: the correct
        # direction N is unreachable, so it is a forced error.
        bout = make_bout([(0, 5), (1, 5), (2, 5), (3, 5), (3, 6), (3, 5), (4, 5), (5, 5)], mask_a)
        triples = bout.get_allocentric_turns_with_approach()
        assert triples == [((3, 5), "E", "N"), ((3, 5), "S", "E")]

        # Approach-conditioned (default): only the correct-corridor crossing is
        # scored -> 0 errors.
        assert bout.count_error(unit="turn", error_type="rate") == 0.0
        # Raw "all": both crossings scored, the forced-error one counts -> 0.5.
        with pytest.warns(DeprecationWarning):
            assert bout.count_error(unit="turn", include="all", error_type="rate") == 0.5

    def test_conditioned_not_greater_than_raw(self, mask_a):
        # Removing forced errors can only lower (or hold) the rate.
        bout = make_bout([(0, 5), (1, 5), (2, 5), (3, 5), (3, 6), (3, 5), (4, 5), (5, 5)], mask_a)
        approach = bout.count_error(unit="turn", error_type="rate")
        with pytest.warns(DeprecationWarning):
            raw = bout.count_error(unit="turn", include="all", error_type="rate")
        assert approach <= raw
