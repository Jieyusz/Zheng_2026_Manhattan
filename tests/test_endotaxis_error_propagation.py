"""
Regression tests for the endotaxis (map-based) error-propagation prediction.

Two things are pinned, mirroring ``test_rl_error_propagation.py``:

1. ``endotaxis.analytic_endotaxis_step`` — the closed-form synchronized step actually drawn in
   the endotaxis columns of ``plot_error_propagation_supp.py`` (computed inline there). Golden
   values: chance on traverse 1 for every position (same no-signal anchor as the RL staircase);
   0 for every position from traverse 2 (all at once, not a back-to-front frontier);
   ``dead_end_last`` pins the far (start-corridor) position to 0.

2. The endotaxis SIMULATION reproduces that step (the claim the figure makes). Pinned data-free:
   the test builds the real Mask A from the committed ``holes_A.npy`` geometry, runs the
   from-scratch endotaxis pipeline (``random_walk_complete`` -> ``Learn_Mouse_tr`` ->
   ``endo_gradient_walk``), and asserts that ONE complete exploratory walk wires the full map so
   the reward-tagged gradient navigates error-free (0 corridor error) and turns correctly at
   every on-path hole -- i.e. every position is solved from traverse 2, which is exactly the
   analytic step.

No raw trajectory data is required (only the committed mask files), so these run in CI.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manhattan_maze.mask import Mask
from manhattan_maze.analysis import calculate_seq_error
from manhattan_maze.rl_model import analytic_rl_staircase
from manhattan_maze.endotaxis import (analytic_endotaxis_step, random_walk_complete,
                                      endo_gradient_walk, Learn_Mouse_tr, map_lin)

# Mask A geometry (identical to test_rl_error_propagation.py; from data/masks/holes_A.npy).
HOLES_A = np.array([[3, 5], [3, 8], [4, 8], [4, 9], [6, 9], [6, 2], [1, 2], [1, 1], [5, 1]])
HOME = (0, 5, 0)
OUT = (5, 9, 1)
SIZE = 11
# Endotaxis learning parameters (scripts/config.ENDOTAXIS_LEARNING_PARAMETERS): gain, threshold,
# goal-learning rate, synaptic decay.
GA, TH, AL, DE = (0.21, 0.2, 0.2, 0)


@pytest.fixture(scope="module")
def mask_a():
    return Mask(HOLES_A, SIZE, "A", HOME, OUT)


class TestAnalyticStep:
    """Golden values for the closed-form endotaxis prediction."""

    def test_traverse_1_is_all_chance(self):
        """Column 0 (traverse 1) is chance everywhere: no goal signal yet -> random walk."""
        step = analytic_endotaxis_step(5, 6)
        assert np.all(step[:, 0] == 0.5)

    def test_all_positions_solved_from_traverse_2(self):
        """Every position drops to 0 at traverse 2 simultaneously (a synchronized step)."""
        step = analytic_endotaxis_step(4, 4)
        expected = np.array([[0.5, 0.0, 0.0, 0.0]] * 4)
        assert np.array_equal(step, expected)

    def test_dead_end_last_pins_far_position_to_zero(self):
        """dead_end_last=True holds the far (start-corridor) row at 0 for all traverses."""
        step = analytic_endotaxis_step(4, 4, dead_end_last=True)
        assert np.all(step[-1] == 0.0)
        assert np.array_equal(step[:-1], analytic_endotaxis_step(4, 4)[:-1])

    def test_custom_chance_level(self):
        step = analytic_endotaxis_step(3, 3, chance=0.25)
        assert step[0, 0] == 0.25 and step[0, 1] == 0.0

    def test_shape(self):
        assert analytic_endotaxis_step(9, 25).shape == (9, 25)

    def test_contrasts_with_rl_staircase(self):
        """Same traverse-1 anchor as RL, opposite shape: endotaxis solves ALL positions at
        traverse 2, whereas the RL staircase leaves far positions at chance."""
        endo = analytic_endotaxis_step(5, 6)
        stair = analytic_rl_staircase(5, 6)
        assert np.array_equal(endo[:, 0], stair[:, 0])   # identical no-signal anchor
        assert np.all(endo[:, 1] == 0.0)                 # endotaxis: everything solved at col 1
        assert np.any(stair[:, 1] > 0.0)                 # RL: far positions still at chance


class TestSimulationReproducesStep:
    """One complete random walk wires the full map -> error-free gradient navigation, so every
    position is solved from traverse 2 (the analytic step's traverse-2+ prediction)."""

    @pytest.mark.parametrize("direction", ["H-O", "O-H"])
    def test_one_walk_gives_errorfree_navigation(self, mask_a, direction):
        adj = mask_a.corridors_adj_mat.astype(float)
        if direction == "H-O":
            start, goal = mask_a.home_corridor, mask_a.out_corridor
        else:
            start, goal = mask_a.out_corridor, mask_a.home_corridor
        dist = mask_a.corridors_shortest_distance[:, goal]

        # from-scratch: one exploratory walk covering every edge, then tag the goal + learn.
        path = random_walk_complete(adj, start, seed=0)
        _, Ms, Gs = Learn_Mouse_tr(adj, path, GA, TH, AL, DE, bi=True, goal=goal)

        # one complete walk recovers the full undirected map
        assert np.array_equal((Ms[-1] > 0).astype(int), (adj > 0).astype(int))

        signal = Gs[-1][0] @ map_lin(Ms[-1], GA)
        walk = endo_gradient_walk(adj, signal, start, goal)

        # corridor row, traverse-2 prediction: reaches goal with 0 distance-increasing steps
        assert walk[-1] == goal
        assert calculate_seq_error(dist[walk]) == 0

        # turn row, traverse-2 prediction: at every on-path decision the gradient points
        # reward-ward (a strictly-decreasing distance step), i.e. every turn is correct
        neighbors = [np.nonzero(a == 1)[0] for a in adj.T]
        for c in mask_a.corridors_shortest_path:
            if c == goal:
                continue
            choice = int(neighbors[c][np.argmax(signal[neighbors[c]])])
            assert dist[choice] < dist[c]
