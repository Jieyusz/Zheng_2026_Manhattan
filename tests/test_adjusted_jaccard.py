"""
Regression tests for transition_vec_similarity (adjusted Jaccard) in utils.py.

R15 (C10): The Jaccard correction is mask-specific and must be passed explicitly.
- Mask D: n_guaranteed_transitions = 3 (three topologically mandatory transitions)
- All other masks: n_guaranteed_transitions = 0 (standard Jaccard)

Calling transition_vec_similarity without n_guaranteed_transitions raises TypeError
(no default is provided — the caller is responsible for supplying the correct value).
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.utils import transition_vec_similarity


def _make_mat(n_bits=6, size=22):
    """Return a 22×22 boolean matrix with n_bits True entries on the diagonal."""
    mat = np.zeros((size, size), dtype=bool)
    for i in range(min(n_bits, size)):
        mat[i, i] = True
    return mat


class TestTransitionVecSimilarity:
    def test_identical_matrices_mask_d(self):
        """Two identical matrices with 6 True bits → adjusted Jaccard = 1.0 with n=3."""
        mat = _make_mat(n_bits=6)
        result = transition_vec_similarity(mat, mat, n_guaranteed_transitions=3)
        assert result == pytest.approx(1.0)

    def test_identical_matrices_standard_jaccard(self):
        """Standard Jaccard (n=0) on identical matrices equals 1.0."""
        mat = _make_mat(n_bits=6)
        result = transition_vec_similarity(mat, mat, n_guaranteed_transitions=0)
        assert result == pytest.approx(1.0)

    def test_disjoint_matrices_standard_jaccard(self):
        """Standard Jaccard for disjoint matrices equals 0.0."""
        mat1 = np.zeros((22, 22), dtype=bool)
        mat2 = np.zeros((22, 22), dtype=bool)
        mat1[0, 0] = True
        mat2[1, 1] = True
        result = transition_vec_similarity(mat1, mat2, n_guaranteed_transitions=0)
        assert result == pytest.approx(0.0)

    def test_partial_overlap_standard_jaccard(self):
        """Standard Jaccard: intersection 1, union 3 → 1/3."""
        mat1 = np.zeros((22, 22), dtype=bool)
        mat2 = np.zeros((22, 22), dtype=bool)
        mat1[0, 0] = True
        mat1[1, 1] = True
        mat2[1, 1] = True
        mat2[2, 2] = True
        result = transition_vec_similarity(mat1, mat2, n_guaranteed_transitions=0)
        assert result == pytest.approx(1 / 3)

    def test_correction_reduces_value(self):
        """Applying n=3 correction lowers the similarity compared to n=0."""
        mat = _make_mat(n_bits=6)
        sim_standard = transition_vec_similarity(mat, mat, n_guaranteed_transitions=0)
        sim_adjusted = transition_vec_similarity(mat, mat, n_guaranteed_transitions=3)
        # Both are 1.0 for identical matrices, so test a non-identical pair
        mat2 = _make_mat(n_bits=4)
        r0 = transition_vec_similarity(mat, mat2, n_guaranteed_transitions=0)
        r3 = transition_vec_similarity(mat, mat2, n_guaranteed_transitions=3)
        # With correction: intersection decreases more than union → lower similarity
        assert r3 <= r0

    def test_zero_union_returns_zero(self):
        """If adjusted union is 0, return 0 (not division error)."""
        mat = np.zeros((22, 22), dtype=bool)
        result = transition_vec_similarity(mat, mat, n_guaranteed_transitions=0)
        assert result == 0.0

    def test_negative_union_raises(self):
        """n_guaranteed_transitions larger than the actual union raises ValueError."""
        mat = _make_mat(n_bits=2)
        with pytest.raises(ValueError):
            transition_vec_similarity(mat, mat, n_guaranteed_transitions=10)

    def test_no_default_raises_type_error(self):
        """
        R15: n_guaranteed_transitions has no default — callers must pass it explicitly.
        Omitting it raises TypeError.
        """
        mat = _make_mat(n_bits=6)
        with pytest.raises(TypeError):
            transition_vec_similarity(mat, mat)


class TestMaskAttributes:
    """
    Verify that Mask and MaskDSpecial expose n_guaranteed_transitions_for_adjusted_jaccard.
    """

    def test_mask_default_correction_is_zero(self):
        from manhattan_maze.mask import Mask
        mask = Mask.__new__(Mask)
        assert mask.n_guaranteed_transitions_for_adjusted_jaccard == 0

    def test_mask_d_correction_is_three(self):
        from manhattan_maze.mask import MaskDSpecial
        mask_d = MaskDSpecial.__new__(MaskDSpecial)
        assert mask_d.n_guaranteed_transitions_for_adjusted_jaccard == 3
