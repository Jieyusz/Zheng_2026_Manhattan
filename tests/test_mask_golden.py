"""
Golden-value regression tests for Mask and MaskDSpecial (Step 3 — mask.py).

All expected values were computed from the actual holes_A.npy file and the production
utils functions (floyd_warshall, find_shortest_path, get_allocentric_turns).  Any change
that shifts these values signals a breaking change to maze geometry.

R15 / C10: Mask.n_guaranteed_transitions_for_adjusted_jaccard == 0
           MaskDSpecial.n_guaranteed_transitions_for_adjusted_jaccard == 3
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.mask import Mask, MaskDSpecial

# Mask A hole coordinates (from data/masks/holes_A.npy)
HOLES_A = np.array([[3, 5], [3, 8], [4, 8], [4, 9], [6, 9], [6, 2], [1, 2], [1, 1], [5, 1]])

HOME = (0, 5, 0)
OUT = (5, 9, 1)
SIZE = 11


@pytest.fixture(scope="module")
def mask_a():
    return Mask(HOLES_A, SIZE, "A", HOME, OUT)


@pytest.fixture(scope="module")
def mask_d_special():
    return MaskDSpecial(HOLES_A, SIZE, "A_as_D", HOME, OUT)


class TestMaskClassAttributes:
    def test_mask_n_guaranteed_transitions_is_zero(self):
        """R15: Mask default Jaccard correction is 0 (standard Jaccard)."""
        assert Mask.n_guaranteed_transitions_for_adjusted_jaccard == 0

    def test_mask_d_special_n_guaranteed_transitions_is_three(self):
        """R15: MaskDSpecial Jaccard correction is 3 (biclique bottleneck)."""
        assert MaskDSpecial.n_guaranteed_transitions_for_adjusted_jaccard == 3

    def test_mask_instance_inherits_class_attribute(self, mask_a):
        assert mask_a.n_guaranteed_transitions_for_adjusted_jaccard == 0

    def test_mask_d_special_instance_inherits_class_attribute(self, mask_d_special):
        assert mask_d_special.n_guaranteed_transitions_for_adjusted_jaccard == 3

    def test_mask_d_special_is_subclass_of_mask(self):
        assert issubclass(MaskDSpecial, Mask)

    def test_new_bypasses_init_still_has_attribute(self):
        """Class attribute accessible via __new__ (no __init__ needed)."""
        m = Mask.__new__(Mask)
        assert m.n_guaranteed_transitions_for_adjusted_jaccard == 0
        md = MaskDSpecial.__new__(MaskDSpecial)
        assert md.n_guaranteed_transitions_for_adjusted_jaccard == 3


class TestMaskPortsAndTiles:
    def test_home_tile(self, mask_a):
        """Home port (0,5,0) → tile index 0 + 5*11 + 0*121 = 55."""
        assert mask_a.home_tile == 55

    def test_out_tile(self, mask_a):
        """Out port (5,9,1) → tile index 5 + 9*11 + 1*121 = 225."""
        assert mask_a.out_tile == 225

    def test_home_corridor(self, mask_a):
        """Home (z=0, y=5) → corridor index 5."""
        assert mask_a.home_corridor == 5

    def test_out_corridor(self, mask_a):
        """Out (z=1, x=5) → corridor index 5 + 11 = 16."""
        assert mask_a.out_corridor == 16


class TestMaskShortestPath:
    def test_tiles_shortest_path_length(self, mask_a):
        assert len(mask_a.tiles_shortest_path) == 45

    def test_tiles_shortest_path_start(self, mask_a):
        assert mask_a.tiles_shortest_path[0] == 55

    def test_tiles_shortest_path_end(self, mask_a):
        assert mask_a.tiles_shortest_path[-1] == 225

    def test_corridors_shortest_path(self, mask_a):
        """Corridors on the shortest path (home→out) for Mask A."""
        expected = [5, 14, 8, 15, 9, 17, 2, 12, 1, 16]
        assert [int(c) for c in mask_a.corridors_shortest_path] == expected


class TestMaskCorrectTurns:
    def test_outbound_correct_turns(self, mask_a):
        """Outbound (home→out) correct turns for Mask A holes."""
        expected = {
            (3, 5): "N",
            (3, 8): "E",
            (4, 8): "N",
            (4, 9): "E",
            (6, 9): "S",
            (6, 2): "W",
            (1, 2): "S",
            (1, 1): "E",
            (5, 1): "N",
        }
        assert mask_a.get_correct_turns(homebound=False) == expected

    def test_homebound_correct_turns(self, mask_a):
        """Homebound (out→home) correct turns for Mask A holes."""
        expected = {
            (5, 1): "W",
            (1, 1): "N",
            (1, 2): "E",
            (6, 2): "N",
            (6, 9): "W",
            (4, 9): "S",
            (4, 8): "W",
            (3, 8): "S",
            (3, 5): "W",
        }
        assert mask_a.get_correct_turns(homebound=True) == expected


class TestMaskHelpers:
    def test_is_hole_true(self, mask_a):
        assert mask_a.is_hole(3, 5) is True

    def test_is_hole_false(self, mask_a):
        assert mask_a.is_hole(0, 0) is False

    def test_is_hole_tuple_input(self, mask_a):
        assert mask_a.is_hole((3, 5)) is True

    def test_get_holes_returns_list_of_tuples(self, mask_a):
        holes = mask_a.get_holes()
        assert isinstance(holes, list)
        assert all(isinstance(h, tuple) for h in holes)
        assert (3, 5) in holes

    def test_get_holes_count(self, mask_a):
        assert len(mask_a.get_holes()) == 9


class TestRemoveOutskirts:
    def test_size_reduced_by_two(self, mask_a):
        reduced = mask_a.remove_outskirts()
        assert reduced.size == SIZE - 2

    def test_name_has_reduced_suffix(self, mask_a):
        reduced = mask_a.remove_outskirts()
        assert reduced.name == "A_reduced"

    def test_home_port_shifted(self, mask_a):
        reduced = mask_a.remove_outskirts()
        # home (0, 5, 0) → (0, 4, 0) after shifting y by -1
        assert reduced.home_coordinates == (0, 4, 0)

    def test_out_port_shifted(self, mask_a):
        reduced = mask_a.remove_outskirts()
        # out (5, 9, 1) → (8, 8, 1): new_size-1=8, y-1=8
        assert reduced.out_coordinates == (8, 8, 1)

    def test_returns_mask_instance(self, mask_a):
        reduced = mask_a.remove_outskirts()
        assert isinstance(reduced, Mask)


class TestMaskRepr:
    def test_repr_contains_newlines(self, mask_a):
        r = repr(mask_a)
        assert "\n" in r

    def test_str_contains_size_and_holes(self, mask_a):
        s = str(mask_a)
        assert "size=11" in s
        assert "holes=" in s
