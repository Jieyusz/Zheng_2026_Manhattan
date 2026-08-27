"""
Bout construction regression tests (Step 4 — trajectory.py).

Pins the conversion bout_df → tiles_df → corridors_df and the bout-type /
duration logic for a known outbound traverse on Mask A.  These golden values
were verified against the production code; any change that shifts them signals a
breaking change to floor-splitting, corridor encoding, or duration semantics.

Scientific invariants exercised:
- Hole tiles are split into two floor rows at the midpoint frame, so
  len(tiles_df) == len(bout_df) + n_holes_visited.
- tile = x + y*11 + z*121 (utils.xyz_to_ti).
- Duration = sum(min(time_in_cell, sleep_threshold)) in seconds, NOT wall-clock.
- Traverse = bout crossing from one port to the other.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.mask import Mask
from manhattan_maze.trajectory import Bout

# Mask A hole coordinates (from data/masks/holes_A.npy)
HOLES_A = np.array([[3, 5], [3, 8], [4, 8], [4, 9], [6, 9], [6, 2], [1, 2], [1, 1], [5, 1]])
HOME = (0, 5, 0)
OUT = (5, 9, 1)
SIZE = 11
FPS = 30

# Outbound traverse path home (0,5) -> out (5,9), passing holes (3,5) and (3,9).
OUTBOUND_PATH = [(0, 5), (1, 5), (2, 5), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 9), (5, 9)]


@pytest.fixture(scope="module")
def mask_a():
    return Mask(HOLES_A, SIZE, "A", HOME, OUT)


def make_bout(path, mask):
    """Build a Bout from a (col,row) path with in_frame=i*10, out_frame=i*10+9."""
    df = pd.DataFrame(
        {
            "in_frame": [i * 10 for i in range(len(path))],
            "out_frame": [i * 10 + 9 for i in range(len(path))],
            "discrete_loc": [tuple(p) for p in path],
        }
    )
    return Bout(df, mask=mask, idx=0, session=None, trajectory=None, FPS=FPS)


@pytest.fixture(scope="module")
def outbound_bout(mask_a):
    return make_bout(OUTBOUND_PATH, mask_a)


class TestTilesDf:
    def test_tiles_df_row_count(self, outbound_bout):
        # 10 bout rows + 2 hole splits (holes (3,5) and (3,9)) = 12 tile rows
        assert len(outbound_bout.tiles_df) == 12

    def test_tiles_sequence(self, outbound_bout):
        assert outbound_bout.tiles_df["tile"].tolist() == [
            55, 56, 57, 58, 179, 190, 201, 212, 223, 102, 103, 104
        ]


class TestCorridorsDf:
    def test_corridor_sequence(self, outbound_bout):
        # row 5 (corridor 5) -> column 3 vertical (corridor 14) -> row 9 (corridor 9)
        assert outbound_bout.corridors_df["corridor"].tolist() == [5, 14, 9]


class TestBoutType:
    def test_is_outbound_and_traverse(self, outbound_bout):
        assert outbound_bout.is_outbound() is True
        assert outbound_bout.is_traverse() is True
        assert outbound_bout.bout_type == "H-O"

    def test_homebound_reversed_is_traverse(self, mask_a):
        homebound = make_bout(OUTBOUND_PATH[::-1], mask_a)
        assert homebound.is_homebound() is True
        assert homebound.is_traverse() is True
        assert homebound.bout_type == "O-H"


class TestDuration:
    def test_duration_with_sleep_threshold(self, outbound_bout):
        # sum of min(time_in_cell, 5 s); splits halve hole-cell durations
        assert outbound_bout.get_duration_s(sleep_threshold=5) == pytest.approx(2.933, abs=1e-3)

    def test_duration_wall_clock(self, outbound_bout):
        # (last out_frame - first in_frame) / FPS = (99 - 0) / 30
        assert outbound_bout.get_duration_s(sleep_threshold=None) == pytest.approx(3.300, abs=1e-3)
