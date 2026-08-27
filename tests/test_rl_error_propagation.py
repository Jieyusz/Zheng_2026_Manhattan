"""
Regression tests for the model-free-RL error-propagation prediction.

Two things are pinned:

1. ``rl_model.analytic_rl_staircase`` — the closed-form backward staircase actually drawn
   in the RL columns of ``plot_error_propagation_supp.py`` (via ``gen_error_propagation.py``).
   Golden values: chance on traverse 1 for every position; the low-error frontier steps out
   one position per traverse; ``dead_end_last`` pins the far (start-corridor) position to 0.

2. The trained per-animal agents REPRODUCE that staircase (the claim the figure makes, and
   what the validation scripts gen_rl_simulation.py / gen_rl_turn_simulation.py assert on real
   data). Here it is pinned data-free: the tests build the real Mask A from the committed
   ``holes_A.npy`` geometry and feed each agent a "perfect traverse" repeated N times (the
   mouse takes the correct exit at every shortest-path decision, reaching the goal). Because
   the readout happens BEFORE each traverse's reward and TD backs the reward up exactly one
   position per traverse, the per-position error must collapse onto the analytic staircase.

No raw trajectory data is required (only the committed mask files), so these run in CI.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.mask import Mask
from manhattan_maze.rl_model import (analytic_rl_staircase, agent_error_matrix,
                                     agent_turn_error_matrix)

# Mask A geometry (identical to test_mask_golden.py; from data/masks/holes_A.npy).
HOLES_A = np.array([[3, 5], [3, 8], [4, 8], [4, 9], [6, 9], [6, 2], [1, 2], [1, 1], [5, 1]])
HOME = (0, 5, 0)
OUT = (5, 9, 1)
SIZE = 11


@pytest.fixture(scope="module")
def mask_a():
    return Mask(HOLES_A, SIZE, "A", HOME, OUT)


class TestAnalyticStaircase:
    """Golden values for the closed-form prediction."""

    def test_traverse_1_is_all_chance(self):
        """Column 0 (traverse 1) is chance everywhere: no reward experienced yet."""
        stair = analytic_rl_staircase(5, 6)
        assert np.all(stair[:, 0] == 0.5)

    def test_frontier_steps_one_position_per_traverse(self):
        """Position i is chance until traverse i, then 0 (i >= t -> chance, else 0)."""
        stair = analytic_rl_staircase(4, 4)
        expected = np.array([
            [0.5, 0.0, 0.0, 0.0],   # nearest position solved from traverse 2 (col 1)
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5, 0.5],   # farthest position still chance through col 3
        ])
        assert np.array_equal(stair, expected)

    def test_dead_end_last_pins_far_position_to_zero(self):
        """dead_end_last=True holds the far (start-corridor) row at 0 for all traverses."""
        stair = analytic_rl_staircase(4, 4, dead_end_last=True)
        assert np.all(stair[-1] == 0.0)
        # the other rows are unchanged from the plain staircase
        assert np.array_equal(stair[:-1], analytic_rl_staircase(4, 4)[:-1])

    def test_custom_chance_level(self):
        stair = analytic_rl_staircase(3, 3, chance=0.25)
        assert stair[0, 0] == 0.25 and stair[-1, -1] == 0.25

    def test_shape(self):
        assert analytic_rl_staircase(9, 25).shape == (9, 25)


class TestTurnSimReproducesStaircase:
    """A trained turn agent fed perfect traverses reproduces the analytic staircase exactly."""

    def test_perfect_traverses_give_exact_staircase(self, mask_a):
        holes = mask_a.get_holes()
        cmap = mask_a.correct_approach_map(homebound=False)  # outbound, goal = OUT
        # One "perfect traverse" bout: correct exit at every shortest-path hole, ending OUT.
        perfect = [(hole, exit_dir) for hole, (_approach, exit_dir) in cmap.items()]
        n_traverses = 12
        journeys = [[(perfect, "OUT")] for _ in range(n_traverses)]

        result = agent_turn_error_matrix(journeys, cmap, holes, "OUT")
        expected = analytic_rl_staircase(len(holes), n_traverses)  # turns: no dead end
        assert np.array_equal(result, expected)

    def test_alpha_independent(self, mask_a):
        """The staircase is exact regardless of learning-rate magnitude (deterministic ordering)."""
        holes = mask_a.get_holes()
        cmap = mask_a.correct_approach_map(homebound=False)
        perfect = [(hole, exit_dir) for hole, (_a, exit_dir) in cmap.items()]
        journeys = [[(perfect, "OUT")] for _ in range(len(holes) + 2)]
        r_low = agent_turn_error_matrix(journeys, cmap, holes, "OUT", alpha=0.1)
        r_high = agent_turn_error_matrix(journeys, cmap, holes, "OUT", alpha=0.9)
        assert np.array_equal(r_low, r_high)


class TestCorridorSimReproducesStaircase:
    """A trained corridor agent fed perfect traverses reproduces the staircase frontier.

    The corridor readout is a stochastic hybrid walk, so unlearned positions sit above 0 at a
    random-walk rate rather than exactly at chance; but a LEARNED position is solved greedily
    on every walk and scores exactly 0. So the *zero-pattern* (which positions are solved at
    which traverse) is deterministic and must equal the analytic staircase's zeros.
    """

    def test_solved_frontier_matches_staircase(self, mask_a):
        path = mask_a.corridors_shortest_path        # home_corridor ... out_corridor
        start, goal = mask_a.home_corridor, mask_a.out_corridor
        adj = mask_a.corridors_adj_mat
        distances = mask_a.corridors_shortest_distance[:, goal]
        n_pos = len(path)
        n_traverses = n_pos + 2
        journeys = [[list(path)] for _ in range(n_traverses)]   # one perfect-traverse bout each

        result = agent_error_matrix(journeys, start, goal, adj, distances, n_pos,
                                    rng=np.random.default_rng(0), n_walks=200)

        # Rows 1..n_pos-1 are the plotted positions (reward row 0 dropped); the far row is the
        # forced dead-end start corridor -> dead_end_last=True.
        expected = analytic_rl_staircase(n_pos - 1, n_traverses, dead_end_last=True)
        solved_sim = np.isclose(result[1:], 0.0)     # NaN (never departed) -> not solved
        solved_expected = (expected == 0.0)
        assert np.array_equal(solved_sim, solved_expected)

    def test_nothing_solved_before_first_reward(self, mask_a):
        """Column 0 (readout before any reward): only the forced dead end is at 0."""
        path = mask_a.corridors_shortest_path
        start, goal = mask_a.home_corridor, mask_a.out_corridor
        distances = mask_a.corridors_shortest_distance[:, goal]
        n_pos = len(path)
        journeys = [[list(path)] for _ in range(3)]
        result = agent_error_matrix(journeys, start, goal, mask_a.corridors_adj_mat,
                                    distances, n_pos, rng=np.random.default_rng(0), n_walks=200)
        interior = result[1:-1, 0]                   # exclude reward row and dead-end row
        assert np.all(interior > 0.0)                # no genuine decision is solved yet
