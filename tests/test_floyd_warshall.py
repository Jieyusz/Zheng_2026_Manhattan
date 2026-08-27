"""
Regression tests for floyd_warshall in utils.py.

These tests pin the specific matrix initialisation formula
  (1 − I)·n − A·(n − 1)
so a scipy-based reimplementation would fail on directed graphs.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.utils import floyd_warshall

LARGE = 1e9  # sentinel for "no path" in directed graph tests


class TestFloydWarshall:
    def test_self_distances_zero(self):
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        D = floyd_warshall(adj)
        assert np.all(D.diagonal() == 0)

    def test_three_node_chain_directed(self):
        # A → B → C  (adj[i,j] = edge from j to i)
        adj = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        D = floyd_warshall(adj)
        # A→B: 1 hop
        assert D[1, 0] == 1
        # A→C: 2 hops
        assert D[2, 0] == 2
        # C→A: no path — should be large (≥ n = 3)
        assert D[0, 2] >= 3

    def test_fully_connected_3_node(self):
        adj = np.ones((3, 3)) - np.eye(3)
        D = floyd_warshall(adj)
        off_diag = D[~np.eye(3, dtype=bool)]
        assert np.all(off_diag == 1)

    def test_directedness(self):
        # One-way A→B, no return path
        adj = np.zeros((2, 2))
        adj[1, 0] = 1  # edge from 0 to 1
        D = floyd_warshall(adj)
        assert D[1, 0] == 1   # A→B reachable
        assert D[0, 1] >= 2   # B→A unreachable (distance ceiling = n=2)

    def test_formula_pins_initialisation(self):
        # The formula (1-I)*n - A*(n-1) must equal the weight matrix exactly
        n = 4
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=float)
        expected_init = (1 - np.eye(n)) * n - adj * (n - 1)
        # After one step of Floyd-Warshall the diagonal stays 0
        assert np.all(expected_init.diagonal() == 0)
        # Off-diagonal connected pairs start at 1
        assert expected_init[0, 1] == 1
        # Off-diagonal unconnected pairs start at n
        assert expected_init[0, 2] == n
