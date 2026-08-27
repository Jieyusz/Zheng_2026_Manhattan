"""
Regression tests for coordinate-encoding functions in utils.py.

All tile and corridor index values are derived from the 11×11×2 maze:
  tile  = x + y*11 + z*121
  corr  = y        if z == 0 (horizontal)
  corr  = x + 11   if z == 1 (vertical)
Home port: (x=0, y=5, z=0) → tile 55, corridor 5
Out port:  (x=5, y=9, z=1) → tile 225, corridor 16
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.utils import xyz_to_ti, ti_to_xyz, xyz_to_ci


class TestXyzToTi:
    def test_home_tile(self):
        assert xyz_to_ti([0, 5, 0]) == 55

    def test_out_tile(self):
        assert xyz_to_ti([5, 9, 1]) == 225

    def test_hole_on_top_floor(self):
        # (3, 5, 1): tile = 3 + 5*11 + 1*121 = 3 + 55 + 121 = 179
        assert xyz_to_ti([3, 5, 1]) == 179

    def test_max_tile(self):
        # (10, 10, 1): tile = 10 + 10*11 + 1*121 = 10 + 110 + 121 = 241
        assert xyz_to_ti([10, 10, 1]) == 241

    def test_origin_tile(self):
        assert xyz_to_ti([0, 0, 0]) == 0


class TestTiToXyz:
    def test_home_tile(self):
        xyz = ti_to_xyz(55)
        assert list(xyz) == [0, 5, 0]

    def test_out_tile(self):
        xyz = ti_to_xyz(225)
        assert list(xyz) == [5, 9, 1]

    def test_max_tile(self):
        xyz = ti_to_xyz(241)
        assert list(xyz) == [10, 10, 1]


class TestRoundTrip:
    @pytest.mark.parametrize("xyz", [
        [0, 5, 0],
        [5, 9, 1],
        [3, 5, 1],
        [10, 10, 1],
        [0, 0, 0],
        [7, 3, 0],
    ])
    def test_round_trip(self, xyz):
        ti = xyz_to_ti(xyz)
        recovered = ti_to_xyz(int(ti))
        assert list(recovered) == xyz


class TestXyzToCi:
    def test_home_corridor(self):
        # (0, 5, 0): z=0, corridor = y = 5
        ci = xyz_to_ci([0, 5, 0])
        assert ci[0] == 5

    def test_out_corridor(self):
        # (5, 9, 1): z=1, corridor = x + 11 = 16
        ci = xyz_to_ci([5, 9, 1])
        assert ci[0] == 16

    def test_vertical_corridor(self):
        # (3, 5, 1): z=1, corridor = 3 + 11 = 14
        ci = xyz_to_ci([3, 5, 1])
        assert ci[0] == 14

    def test_horizontal_corridor_row0(self):
        # (4, 0, 0): z=0, corridor = y = 0
        ci = xyz_to_ci([4, 0, 0])
        assert ci[0] == 0

    def test_batch_input(self):
        batch = np.array([[0, 5, 0], [5, 9, 1]])
        ci = xyz_to_ci(batch)
        assert list(ci) == [5, 16]
