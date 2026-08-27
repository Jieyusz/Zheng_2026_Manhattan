"""
Golden-value regression tests for the first-order Markov walker (random_walk.py).

Pins the manuscript ``sec:walker`` reference values (Home -> Out, corridor graph):

    graph          tau(1/2)   tau(1)    E(1/2)    E(1)
    P10 (Mask A)      81         9         36        0
    Mask D           166.75    92.587    80.875    43.793

Also checks: beta=1/2 reduces to the memoryless walk (zero_order_average_steps); the
bipartite corridor-graph identity E(beta) = (tau(beta) - L)/2; effective-bias inversion
round-trips; and input validation. Hole coordinates are hardcoded (from data/masks) so the
tests need no data files, matching tests/test_mask_golden.py.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.mask import Mask
from manhattan_maze.graph import zero_order_average_steps
from manhattan_maze.random_walk import (
    completion_time,
    expected_corridor_errors,
    effective_forward_bias,
    walker_metrics,
)

# Hole coordinates (from data/masks/holes_A.npy and holes_D.npy)
HOLES_A = np.array([[3, 5], [3, 8], [4, 8], [4, 9], [6, 9], [6, 2], [1, 2], [1, 1], [5, 1]])
HOLES_D = np.array([
    [1, 1], [8, 1], [1, 2], [3, 2], [5, 2], [7, 2], [9, 2], [2, 3], [4, 3], [6, 3],
    [8, 3], [1, 4], [3, 4], [7, 4], [9, 4], [2, 5], [4, 5], [6, 5], [8, 5], [1, 6],
    [3, 6], [7, 6], [9, 6], [2, 7], [4, 7], [6, 7], [8, 7], [1, 8], [3, 8], [7, 8],
    [9, 8], [2, 9], [4, 9], [6, 9], [8, 9],
])

HOME = (0, 5, 0)
OUT = (5, 9, 1)
SIZE = 11
HOME_CORRIDOR = 5
OUT_CORRIDOR = 16


@pytest.fixture(scope="module")
def mask_a():
    return Mask(HOLES_A, SIZE, "A", HOME, OUT)


@pytest.fixture(scope="module")
def mask_d():
    return Mask(HOLES_D, SIZE, "D", HOME, OUT)


class TestCompletionTimeGolden:
    def test_p10_memoryless(self, mask_a):
        """P10 memoryless completion time = 81 = 9^2 (symmetric walk across nine holes)."""
        r = walker_metrics(mask_a, beta=0.5)["completion_time"]
        assert np.isclose(r, 81.0, atol=1e-6)

    def test_p10_no_reversal(self, mask_a):
        """P10 never-reversing walker follows the 9-hole shortest path deterministically."""
        r = walker_metrics(mask_a, beta=1.0)["completion_time"]
        assert np.isclose(r, 9.0, atol=1e-6)

    def test_maskd_memoryless(self, mask_d):
        """Mask D memoryless completion time (cyclic bicliques trap the walker)."""
        r = walker_metrics(mask_d, beta=0.5)["completion_time"]
        assert np.isclose(r, 166.75, atol=1e-3)

    def test_maskd_no_reversal(self, mask_d):
        """Mask D non-reversing walker still needs ~92.6 steps (cyclic trap)."""
        r = walker_metrics(mask_d, beta=1.0)["completion_time"]
        assert np.isclose(r, 92.587, atol=1e-3)


class TestCorridorErrorsGolden:
    def test_p10_memoryless(self, mask_a):
        """P10 memoryless walker makes 36 corridor errors."""
        r = walker_metrics(mask_a, beta=0.5)["expected_errors"]
        assert np.isclose(r, 36.0, atol=1e-6)

    def test_p10_no_reversal(self, mask_a):
        """P10 never-reversing walker makes 0 corridor errors (acyclic path)."""
        r = walker_metrics(mask_a, beta=1.0)["expected_errors"]
        assert np.isclose(r, 0.0, atol=1e-9)

    def test_maskd_memoryless(self, mask_d):
        """Mask D memoryless walker makes 80.875 corridor errors."""
        r = walker_metrics(mask_d, beta=0.5)["expected_errors"]
        assert np.isclose(r, 80.875, atol=1e-3)

    def test_maskd_no_reversal(self, mask_d):
        """Mask D non-reversing walker still makes ~43.8 corridor errors."""
        r = walker_metrics(mask_d, beta=1.0)["expected_errors"]
        assert np.isclose(r, 43.793, atol=1e-3)


class TestMemorylessEquivalence:
    @pytest.mark.parametrize("mask_name", ["a", "d"])
    def test_beta_half_equals_zero_order(self, mask_name, mask_a, mask_d):
        """beta=1/2 first-order completion time equals the memoryless (zero-order) hitting time."""
        mask = mask_a if mask_name == "a" else mask_d
        adj = np.asarray(mask.corridors_adj_mat)
        active = np.where(adj.sum(axis=0) > 0)[0]
        sub = adj[np.ix_(active, active)]
        start_idx = int(np.where(active == HOME_CORRIDOR)[0][0])
        goal_idx = int(np.where(active == OUT_CORRIDOR)[0][0])
        zero_order = zero_order_average_steps(sub)[goal_idx, start_idx]
        first_order = completion_time(adj, HOME_CORRIDOR, OUT_CORRIDOR, beta=0.5)
        assert np.isclose(first_order, zero_order, atol=1e-6)


class TestBipartiteIdentity:
    @pytest.mark.parametrize("beta", [0.5, 0.75, 1.0])
    @pytest.mark.parametrize("mask_name", ["a", "d"])
    def test_error_equals_steps_minus_L_over_two(self, beta, mask_name, mask_a, mask_d):
        """Corridor graph is bipartite, so E(beta) = (tau(beta) - L)/2 exactly."""
        mask = mask_a if mask_name == "a" else mask_d
        length = int(mask.corridors_shortest_distance[OUT_CORRIDOR, HOME_CORRIDOR])
        m = walker_metrics(mask, beta=beta)
        assert np.isclose(m["expected_errors"], (m["completion_time"] - length) / 2, atol=1e-6)


class TestEffectiveForwardBias:
    def test_roundtrip(self, mask_d):
        """Inverting E(beta) recovers the beta that generated it."""
        adj = mask_d.corridors_adj_mat
        dist = np.asarray(mask_d.corridors_shortest_distance)[OUT_CORRIDOR, :]
        target = expected_corridor_errors(adj, dist, HOME_CORRIDOR, OUT_CORRIDOR, beta=0.8)
        beta_hat = effective_forward_bias(target, adj, dist, HOME_CORRIDOR, OUT_CORRIDOR)
        assert np.isclose(beta_hat, 0.8, atol=1e-6)

    def test_unbracketed_raises(self, mask_d):
        """An error below the optimal-walker floor is not bracketed and must raise."""
        adj = mask_d.corridors_adj_mat
        dist = np.asarray(mask_d.corridors_shortest_distance)[OUT_CORRIDOR, :]
        with pytest.raises(ValueError):
            effective_forward_bias(0.0, adj, dist, HOME_CORRIDOR, OUT_CORRIDOR)


class TestValidation:
    def test_bad_unit(self, mask_a):
        with pytest.raises(ValueError):
            walker_metrics(mask_a, unit="hallway")

    @pytest.mark.parametrize("beta", [0.0, -0.1, 1.5])
    def test_bad_beta(self, mask_a, beta):
        with pytest.raises(ValueError):
            completion_time(mask_a.corridors_adj_mat, HOME_CORRIDOR, OUT_CORRIDOR, beta=beta)

    def test_isolated_node_raises(self, mask_a):
        """A corridor not used by the mask is isolated and cannot start a walk."""
        isolated = int(np.where(np.asarray(mask_a.corridors_adj_mat).sum(axis=0) == 0)[0][0])
        with pytest.raises(ValueError):
            completion_time(mask_a.corridors_adj_mat, isolated, OUT_CORRIDOR, beta=0.5)
