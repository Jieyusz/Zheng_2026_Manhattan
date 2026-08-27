"""First-order Markov walker: expected completion time and corridor errors.

Implements the metrics of the manuscript methods subsection "Expected completion steps
by a first-order Markov model" (``sec:walker``): the mean completion time
:math:`\\bar{\\tau}(\\beta)` and the expected number of corridor errors
:math:`\\mathcal{E}(\\beta)` for a walker with a tunable *forward bias* ``beta``.

The walker keeps a one-step memory (its state is a directed edge ``j -> i``): from a node
it continues forward with weight ``beta`` and reverses with weight ``1 - beta``.
``beta = 1/2`` recovers the memoryless (uniform) random walk; ``beta = 1`` never reverses.

The absorbing first-passage solve and the edge-state transition matrix are reused from
:mod:`manhattan_maze.graph`; this module adds the forward-bias convenience layer, the
expected corridor-error computation, and the effective-forward-bias inversion.

See ``docs/notation_guide.md`` for the model's symbols and units.
"""
import numpy as np
from scipy.optimize import brentq

from manhattan_maze.graph import (
    first_order_average_steps,
    first_order_transition_matrix,
)

__all__ = [
    "completion_time",
    "expected_corridor_errors",
    "effective_forward_bias",
    "reversal_decisions",
    "forward_bias_mle",
    "reversal_forward_bias",
    "walker_metrics",
]


def _active_subgraph(adj):
    """
    Restrict an adjacency matrix to its non-isolated nodes.

    The maze adjacency matrices are sized for the full 11x11 assembly (22 corridors,
    242 tiles), but any single mask uses only a subset (e.g. P10 uses 10 of 22 corridors,
    Mask D uses 18). Isolated nodes (all-zero rows/columns) make the transition matrix
    ``S = A / A.sum(axis=0)`` divide by zero, so they must be removed before the
    first-passage solve.

    Parameters
    ----------
    adj : np.ndarray, shape (n, n)
        Symmetric adjacency matrix; ``adj[i, j] = 1`` when nodes i and j are connected.
        [corridor index 0-21] or [tile index 0-241] depending on the graph.

    Returns
    -------
    sub_adj : np.ndarray, shape (m, m)
        Adjacency matrix over the ``m`` non-isolated nodes only.
    active : np.ndarray, shape (m,)
        Original node indices retained, ascending. ``active[k]`` is the full-graph index
        of row/column ``k`` of ``sub_adj``; use it to map full-graph indices in and out.

    Notes
    -----
    A node is "active" when its column sum is nonzero. The maze graphs are symmetric, so
    column-active and row-active coincide.
    """
    adj = np.asarray(adj)
    active = np.where(adj.sum(axis=0) > 0)[0]
    sub_adj = adj[np.ix_(active, active)]
    return sub_adj, active


def _subgraph_index(active, node_id):
    """
    Map a full-graph node index into the active-subgraph index.

    Parameters
    ----------
    active : np.ndarray, shape (m,)
        Retained full-graph node indices from :func:`_active_subgraph`.
    node_id : int
        Full-graph node index (e.g. home corridor 5, out corridor 16). [node index]

    Returns
    -------
    int
        Position of ``node_id`` within ``active`` (its subgraph index).

    Raises
    ------
    ValueError
        If ``node_id`` is isolated in this mask (absent from ``active``), so no walk to
        or from it is defined.
    """
    hits = np.where(active == node_id)[0]
    if hits.size == 0:
        raise ValueError(
            f"node {node_id} is isolated in this mask graph (not in the connected subgraph)"
        )
    return int(hits[0])


def _check_beta(beta):
    """
    Validate the forward bias.

    Parameters
    ----------
    beta : float
        Forward bias; must lie in ``(0, 1]``. Dimensionless. ``beta = 1/2`` is the
        memoryless walk; ``beta = 1`` never reverses. ``beta = 0`` is disallowed because
        a walker that always reverses never reaches the goal.

    Raises
    ------
    ValueError
        If ``beta`` is not in ``(0, 1]``.
    """
    if not 0 < beta <= 1:
        raise ValueError(f"beta must be in (0, 1], got {beta}")


def completion_time(adj, start, goal, beta=0.5):
    """
    Expected number of steps for a first-order Markov walker to reach the goal.

    Computes the mean first-passage (hitting) time :math:`\\bar{\\tau}(\\beta)` of the
    forward-biased walker from ``start`` to ``goal``, treating the goal as an absorbing
    boundary. The absorbing-chain solve is delegated to
    :func:`manhattan_maze.graph.first_order_average_steps`; isolated nodes are dropped
    first so the transition matrix is well posed.

    Parameters
    ----------
    adj : np.ndarray, shape (n, n)
        Symmetric maze adjacency matrix (corridor or tile graph).
        [corridor index 0-21] or [tile index 0-241].
    start : int
        Start node index in the full graph (e.g. home corridor 5). [node index]
    goal : int
        Goal node index in the full graph (e.g. out corridor 16). [node index]
    beta : float, optional
        Forward bias in ``(0, 1]``, dimensionless. Default 0.5 (memoryless walk).

    Returns
    -------
    float
        Expected number of graph transitions from ``start`` to ``goal``. Units: corridor
        transitions (corridor graph) or tile transitions (tile graph).

    Raises
    ------
    ValueError
        If ``beta`` is outside ``(0, 1]``, or ``start``/``goal`` is isolated in the mask.

    Notes
    -----
    This is :math:`\\bar{\\tau}(\\beta)` in ``sec:walker``. At ``beta = 1/2`` it equals the
    memoryless hitting time (``zero_order_average_steps``); at ``beta = 1`` the walker
    follows the shortest path on an acyclic graph. The convention (averaging the
    first-passage time over the directed edges entering ``start``) is inherited from
    :func:`first_order_average_steps`, which returns ``steps[goal, start]``.
    """
    _check_beta(beta)
    sub_adj, active = _active_subgraph(adj)
    start_idx = _subgraph_index(active, start)
    goal_idx = _subgraph_index(active, goal)
    steps = first_order_average_steps(sub_adj, probability=beta)
    return float(steps[goal_idx, start_idx])


def expected_corridor_errors(adj, dist_to_goal, start, goal, beta=0.5):
    """
    Expected number of graph-distance-increasing steps (errors) before reaching the goal.

    A corridor error is a transition into a node farther from the goal in graph distance
    (manuscript Section ``sec:metric``). For the first-order walker this is computed from
    the edge-state chain: the immediate expected error out of each transient state is the
    probability its next step increases the goal distance, and these are propagated to
    absorption by the transient fundamental matrix :math:`(I - Q)^{-1}`. The result is
    :math:`\\mathcal{E}(\\beta)`, averaged over the directed edges entering ``start``
    (the same start convention as :func:`completion_time`).

    Parameters
    ----------
    adj : np.ndarray, shape (n, n)
        Symmetric maze adjacency matrix (corridor or tile graph).
        [corridor index 0-21] or [tile index 0-241].
    dist_to_goal : np.ndarray, shape (n,)
        Graph distance (in holes/steps) from every full-graph node to ``goal``, e.g.
        ``mask.corridors_shortest_distance[goal, :]``. [step count]
    start : int
        Start node index in the full graph. [node index]
    goal : int
        Goal node index in the full graph. [node index]
    beta : float, optional
        Forward bias in ``(0, 1]``, dimensionless. Default 0.5 (memoryless walk).

    Returns
    -------
    float
        Expected number of distance-increasing steps from ``start`` to ``goal``. Units:
        corridor errors (corridor graph) or tile errors (tile graph), per traverse.

    Raises
    ------
    ValueError
        If ``beta`` is outside ``(0, 1]``, or ``start``/``goal`` is isolated in the mask.

    Notes
    -----
    This is :math:`\\mathcal{E}(\\beta)` in ``sec:walker``, monotone decreasing in ``beta``.
    Because the maze corridor graph is bipartite, every step changes the
    goal distance by exactly one, so it satisfies the identity
    :math:`\\mathcal{E}(\\beta) = [\\bar{\\tau}(\\beta) - L]/2` with ``L`` the shortest-path
    length in holes; that identity is used as a regression check, not for this computation.
    The probability of a step depends on the incoming edge (the walker's memory), but
    whether the step is an error depends only on the current and next node.
    """
    _check_beta(beta)
    sub_adj, active = _active_subgraph(adj)
    start_idx = _subgraph_index(active, start)
    goal_idx = _subgraph_index(active, goal)
    dist = np.asarray(dist_to_goal)[active]  # subgraph node -> distance to goal

    # Edge-state chain: state ``state_of[i, j]`` = "at node i, arrived from node j".
    trans_matrix, state_of, precursors, successors, edge = first_order_transition_matrix(
        sub_adj, probability=beta
    )
    n_states = trans_matrix.shape[0]
    transition = trans_matrix.T  # row-stochastic: transition[a, b] = P(state a -> state b)

    goal_states = {int(state_of[goal_idx, p]) for p in precursors[goal_idx]}  # absorbing
    transient = [state for state in range(n_states) if state not in goal_states]
    pos = {state: row for row, state in enumerate(transient)}

    # Immediate expected error out of each transient state: probability the next step
    # moves to a node farther from the goal.
    immediate_error = np.zeros(len(transient))
    for state in transient:
        current = edge[state][0]
        for neighbor in successors[current]:
            if dist[neighbor] > dist[current]:
                immediate_error[pos[state]] += transition[state, int(state_of[neighbor, current])]

    q_sub = transition[np.ix_(transient, transient)]
    errors_from_state = np.linalg.solve(np.eye(len(transient)) - q_sub, immediate_error)

    start_states = [int(state_of[start_idx, p]) for p in precursors[start_idx]]
    per_start = [0.0 if state in goal_states else errors_from_state[pos[state]] for state in start_states]
    return float(np.mean(per_start))


def effective_forward_bias(observed_error, adj, dist_to_goal, start, goal, bounds=(0.5, 1.0)):
    """
    Invert the corridor-error curve to estimate a walker's effective forward bias.

    Solves :math:`\\mathcal{E}(\\hat{\\beta}) = \\text{observed\\_error}` for
    :math:`\\hat{\\beta}` by root-finding on the monotone-decreasing
    :func:`expected_corridor_errors` (manuscript inversion in ``sec:walker``). An animal
    with :math:`\\hat{\\beta} \\approx 1/2` traverses indistinguishably from the memoryless
    walker; :math:`\\hat{\\beta} \\to 1` indicates near-optimal, backtracking-free navigation.

    Parameters
    ----------
    observed_error : float
        Observed mean corridor errors per traverse to match. [error count per traverse]
    adj : np.ndarray, shape (n, n)
        Symmetric maze adjacency matrix (corridor or tile graph).
    dist_to_goal : np.ndarray, shape (n,)
        Graph distance from every full-graph node to ``goal``. [step count]
    start, goal : int
        Start and goal node indices in the full graph. [node index]
    bounds : tuple of float, optional
        Search interval ``(beta_low, beta_high)`` for the bias, default ``(0.5, 1.0)``
        (chance to optimal). Both endpoints must lie in ``(0, 1]``.

    Returns
    -------
    float
        Effective forward bias :math:`\\hat{\\beta}` in ``bounds``.

    Raises
    ------
    ValueError
        If ``observed_error`` is not bracketed by ``expected_corridor_errors`` on
        ``bounds`` (e.g. an animal below the optimal-walker error floor).

    Notes
    -----
    :math:`\\mathcal{E}` decreases monotonically in ``beta``, so a solution in ``bounds``
    exists and is unique whenever ``observed_error`` lies between the endpoint values
    ``E(beta_high) <= observed_error <= E(beta_low)``.
    """
    beta_low, beta_high = bounds
    _check_beta(beta_low)
    _check_beta(beta_high)

    def residual(beta):
        return expected_corridor_errors(adj, dist_to_goal, start, goal, beta=beta) - observed_error

    res_low, res_high = residual(beta_low), residual(beta_high)
    if res_low == 0:
        return float(beta_low)
    if res_high == 0:
        return float(beta_high)
    if np.sign(res_low) == np.sign(res_high):
        raise ValueError(
            f"observed_error={observed_error} is not bracketed by E(beta) on {bounds}: "
            f"E({beta_low})={res_low + observed_error:.4f}, "
            f"E({beta_high})={res_high + observed_error:.4f}"
        )
    return float(brentq(residual, beta_low, beta_high))


def walker_metrics(mask, beta=0.5, unit="corridor"):
    """
    Completion time and expected errors for one mask, from Home to Out.

    Convenience wrapper that reads the graph, distances, and ports from a ``Mask`` object
    and returns both first-order-walker metrics for the outbound (Home to Out) traverse.

    Parameters
    ----------
    mask : Mask
        Mask-like object exposing, for ``unit="corridor"``: ``corridors_adj_mat``,
        ``corridors_shortest_distance``, ``home_corridor``, ``out_corridor``; and for
        ``unit="tile"``: ``tiles_adj_mat``, ``tiles_shortest_distances``, ``home_tile``,
        ``out_tile``. (Duck-typed to avoid an import cycle with ``mask.py``.)
    beta : float, optional
        Forward bias in ``(0, 1]``, dimensionless. Default 0.5 (memoryless walk).
    unit : {"corridor", "tile"}, optional
        Graph on which to compute the walk. Default ``"corridor"``.

    Returns
    -------
    dict
        ``{"completion_time": float, "expected_errors": float}`` — the expected number of
        steps :math:`\\bar{\\tau}(\\beta)` and distance-increasing steps
        :math:`\\mathcal{E}(\\beta)`, in units of the chosen graph (corridor or tile).

    Raises
    ------
    ValueError
        If ``unit`` is not ``"corridor"`` or ``"tile"``, if ``beta`` is outside ``(0, 1]``,
        or if the ports are isolated in the mask graph.

    Notes
    -----
    Reference values (Home to Out): P10 (Mask A) — ``beta=1/2``: 81 steps / 36 errors,
    ``beta=1``: 9 steps / 0 errors; Mask D — ``beta=1/2``: 166.75 / 80.875,
    ``beta=1``: 92.587 / 43.793.
    """
    if unit == "corridor":
        adj = mask.corridors_adj_mat
        dist_mat = np.asarray(mask.corridors_shortest_distance)
        start, goal = mask.home_corridor, mask.out_corridor
    elif unit == "tile":
        adj = mask.tiles_adj_mat
        dist_mat = np.asarray(mask.tiles_shortest_distances)
        start, goal = mask.home_tile, mask.out_tile
    else:
        raise ValueError(f"unit must be 'corridor' or 'tile', got {unit!r}")

    dist_to_goal = dist_mat[goal, :]  # floyd_warshall: row `goal` = distance to `goal`
    return {
        "completion_time": completion_time(adj, start, goal, beta=beta),
        "expected_errors": expected_corridor_errors(adj, dist_to_goal, start, goal, beta=beta),
    }


def reversal_forward_bias(node_seq, degrees):
    """
    Estimate forward bias from observed reversals along a visited-node sequence.

    A second, independent estimator of the same :math:`\\hat{\\beta}` that
    :func:`effective_forward_bias` recovers by inverting the error curve. This one counts
    *reversals* instead: at an interior node of degree :math:`d`, the one-step-memory
    walker returns to the node it came from with probability

    .. math:: p_\\text{rev}(\\beta, d) = \\frac{1-\\beta}{(1-\\beta) + (d-1)\\beta},

    so with :math:`\\varphi = (1-\\beta)/\\beta` the maximum-likelihood estimate solves
    :math:`R/\\varphi = \\sum_t 1/(\\varphi + d_t - 1)` over the scored decisions, and
    :math:`\\hat{\\beta} = 1/(1+\\varphi)` (manuscript ``eq:betahat``). The degree
    correction matters because a reversal is more surprising at a high-degree junction; on
    a path graph where every interior node has :math:`d = 2` this reduces to
    :math:`\\hat{\\beta} = 1 - R/N`, one minus the reversal rate.

    Parameters
    ----------
    node_seq : array_like of int
        Sequence of visited node indices (e.g. a run-length-collapsed corridor sequence).
        Consecutive repeats must already be removed, since a "reversal" is defined as
        ``node_seq[t+1] == node_seq[t-1]``.
    degrees : dict of {int: int}
        Degree of each node in the maze graph. Nodes with degree < 2 (dead ends, where a
        reversal is forced rather than chosen) are excluded from scoring.

    Returns
    -------
    float
        Forward bias :math:`\\hat{\\beta}` in [0, 1], or NaN if the sequence contains no
        scorable interior decision. Returns exactly 1.0 when no reversal occurred and 0.0
        when every decision was a reversal (both are the likelihood's boundary solutions,
        where the root-find would not bracket).

    Notes
    -----
    ``beta = 0.5`` is the memoryless walker, so it is the null this readout is compared
    against; values above it indicate directional persistence.

    A thin convenience wrapper over :func:`reversal_decisions` followed by
    :func:`forward_bias_mle`. Use those two directly when decisions must be *selected*
    before fitting (e.g. by position along a journey), since slicing ``node_seq`` per window
    loses the decision at each slice boundary.
    """
    return forward_bias_mle(*reversal_decisions(node_seq, degrees)[1:])


def reversal_decisions(node_seq, degrees):
    """
    Split a visited-node sequence into its individually scorable reversal decisions.

    Returned as parallel arrays so decisions can be selected by *position* -- e.g. by where
    they fall along a journey -- and only then pooled into a single
    :func:`forward_bias_mle` fit. Selecting decisions is preferable to re-slicing
    ``node_seq`` once per window, because a slice silently discards the decision at each of
    its two ends (scoring needs both a predecessor and a successor), which matters most for
    the short, truncated windows at the ends of a sequence.

    Parameters
    ----------
    node_seq : array_like of int
        Sequence of visited node indices with consecutive repeats already removed, as
        produced by :func:`~manhattan_maze.analysis.first_journey_corridor_seq`.
    degrees : dict of {int: int}
        Degree of each node in the maze graph.

    Returns
    -------
    positions : np.ndarray of int
        Index into ``node_seq`` of each scored decision, in order. Always interior
        (``1 <= t <= len(node_seq) - 2``) and restricted to nodes of degree >= 2, since a
        reversal at a dead end is forced rather than chosen.
    is_reversal : np.ndarray of bool
        Whether each scored decision returned to the previously occupied node,
        ``node_seq[t+1] == node_seq[t-1]``.
    scored_degrees : np.ndarray of float
        Degree of the node at each scored decision.
    """
    node_seq = np.asarray(node_seq, dtype=int)
    scored = [t for t in range(1, len(node_seq) - 1)
              if degrees.get(int(node_seq[t]), 0) >= 2]
    return (np.array(scored, dtype=int),
            np.array([node_seq[t + 1] == node_seq[t - 1] for t in scored], dtype=bool),
            np.array([degrees[int(node_seq[t])] for t in scored], dtype=float))


def forward_bias_mle(is_reversal, scored_degrees):
    """
    Maximum-likelihood forward bias :math:`\\hat{\\beta}` over a set of scored decisions.

    The likelihood of :func:`reversal_forward_bias`, taking the decisions directly (from
    :func:`reversal_decisions`) rather than a node sequence: with
    :math:`\\varphi = (1-\\beta)/\\beta`, solves
    :math:`R/\\varphi = \\sum_t 1/(\\varphi + g_t - 1)` and returns
    :math:`\\hat{\\beta} = 1/(1+\\varphi)` (manuscript ``eq:betahat``).

    Parameters
    ----------
    is_reversal : array_like of bool
        Whether each scored decision was a reversal.
    scored_degrees : array_like of float
        Degree of the node at each scored decision. Must align with ``is_reversal``.

    Returns
    -------
    float
        :math:`\\hat{\\beta}` in [0, 1]; NaN if no decision was supplied. Returns exactly
        1.0 when no reversal occurred and 0.0 when every decision was a reversal (both are
        the likelihood's boundary solutions, where the root-find would not bracket).

    Notes
    -----
    The estimate is *self-normalising*: the sum runs over exactly the decisions supplied, so
    fitting fewer decisions widens the sampling distribution without shifting it. A window
    truncated at the end of a sequence therefore needs no divisor correction, unlike the
    fixed-divisor mean that :func:`~manhattan_maze.utils.moving_average` has to renormalise
    at its edges. On a graph whose scored nodes all have degree 2 (e.g. the P10 corridor
    path of Mask A) this reduces to the exactly unbiased :math:`\\hat{\\beta} = 1 - R/N` at
    any :math:`N`, and the boundary returns are the true MLE rather than a clipped value.
    """
    is_reversal = np.asarray(is_reversal, dtype=bool)
    scored_degrees = np.asarray(scored_degrees, dtype=float)
    n_scored = is_reversal.size
    if n_scored == 0:
        return np.nan

    n_reversals = int(np.count_nonzero(is_reversal))
    if n_reversals == 0:         # boundary: never reversed
        return 1.0
    if n_reversals == n_scored:  # boundary: always reversed
        return 0.0

    phi = brentq(lambda p: n_reversals / p - np.sum(1.0 / (p + scored_degrees - 1.0)),
                 1e-9, 1e9)
    return 1.0 / (1.0 + phi)
