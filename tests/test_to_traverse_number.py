"""
Regression tests for utils.to_traverse_number (C8).

traverse_idx is 0-based (internal array index); traverse_number is the 1-based
manuscript/display number used for figure axis labels (docs/data_contracts.md).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.utils import to_traverse_number


def test_scalar_is_one_based():
    assert to_traverse_number(0) == 1
    assert to_traverse_number(19) == 20


def test_array_is_elementwise_plus_one():
    out = to_traverse_number(np.arange(5))
    assert np.array_equal(out, np.array([1, 2, 3, 4, 5]))


def test_inverse_of_zero_based_indexing():
    # traverse_number - 1 recovers the 0-based index used for array lookups.
    for idx in range(10):
        assert to_traverse_number(idx) - 1 == idx
