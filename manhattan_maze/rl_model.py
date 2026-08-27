"""
Purely model-free reinforcement learning on Mask A, trained on the animals' own
trajectories — for BOTH the corridor (distance-to-reward) and turn (per-hole decision)
metrics of the error-propagation supplementary figure.

The scientific contrast is the same for both metrics: a model-free value learner given the
mice's *own* experience resolves errors by BACKWARD PROPAGATION from the reward (positions
nearest the reward become correct first; the low-error frontier steps outward journey by
journey), whereas the empirical mice improve across the whole path roughly in parallel. Both
agents converge to the same closed form, :func:`analytic_rl_staircase` — the curve actually
plotted in the figure; the trained agents here are its validation on real trajectories.

Common envelope (both agents)
-----------------------------
Two independent per-direction agents (outbound ``H-O``, homebound ``O-H``); each processes
one animal's journeys (leading sorties + terminating traverse) in chronological order by ONE
cumulative agent (never reset within a session), reward ``+1`` experienced on reaching the
direction's goal. Readout for traverse ``t`` uses only value learned on traverses ``1..t-1``
(readout BEFORE that traverse's reward), matching how the animals are scored, so traverse 1
is exactly chance. The cohort functions average the per-animal ``(n_positions, n_journeys)``
matrices into a population-mean ``(n_positions, width)`` matrix (NaN-pad to a common width,
``np.nanmean`` per position), mirroring :mod:`manhattan_maze.analysis`.

Corridor agent (:func:`agent_error_matrix`, :func:`cohort_rl_error_rate`)
------------------------------------------------------------------------
State = corridor; action = step to an adjacent corridor. TRAIN on the journey's entire
corridor sequence with TD(0); only *valid adjacent-corridor* steps update the table (bout-seam
repeats are skipped). READ OUT by SIMULATING many hybrid-policy walks (GREEDY where Q is
informative, RANDOM adjacent step otherwise) and SCORING each with the non-decreasing distance
rule of :func:`manhattan_maze.analysis.localize_distance_seq`, pooled into a per-corridor rate.

Turn agent (:func:`agent_turn_error_matrix`, :func:`cohort_turn_error_rate`)
--------------------------------------------------------------------------
State = maze hole; action = allocentric heading ``N``/``S``/``E``/``W``; ``Q[hole, dir]``.
Transitions come straight from the animal's data (``Bout.get_hole_decisions``: every crossing,
turn or pass). TRAIN on every decision (even wrong-approach crossings). READ OUT the FIRST-turn
error by marching the shortest-path holes (``Mask.correct_approach_map``): the approach axis
fixes the choice to the two perpendicular headings (chance 0.5), greedy when ``Q`` is
informative else chance — deterministic, so no Monte-Carlo rollouts.

This file is a path-free library. The Mask-A geometry, trajectory extraction, figure-data keys
and CLIs live in ``scripts/gen_rl_simulation.py`` (corridors) and
``scripts/gen_rl_turn_simulation.py`` (turns); the plotted staircase is written by
``scripts/gen_error_propagation.py``.
"""
import warnings

import numpy as np

from manhattan_maze.analysis import localize_distance_seq, calculate_seq_error, calculate_seq_error_rate
from manhattan_maze.utils import extract_array, turn_axis

__all__ = [
    "analytic_rl_staircase",
    # corridor agent
    "neighbors_from_adjacency",
    "train_on_journey",
    "simulate_walk",
    "hybrid_error_readout",
    "agent_error_matrix",
    "cohort_rl_error_rate",
    # self-play corridor agent (Mask-D bottleneck comparison)
    "random_walk_to_goal",
    "bottleneck_choice_prob",
    "greedy_corridor_error",
    "selfplay_curve",
    "cohort_selfplay",
    # turn agent
    "DIRECTIONS",
    "train_on_journey_decisions",
    "first_turn_error_vector",
    "agent_turn_error_matrix",
    "cohort_turn_error_rate",
]

# Turn-agent constants (allocentric heading table Q[hole, dir]).
DIRECTIONS = ("N", "S", "E", "W")
_DIR_IDX = {d: i for i, d in enumerate(DIRECTIONS)}
_PORTS = ("OUT", "HOME")
_PTP_TOL = 1e-12   # Q spread above which a hole counts as "learned" (informative)


def analytic_rl_staircase(n_positions, n_traverses, chance=0.5, dead_end_last=False):
    """
    Closed-form model-free-RL error prediction: the backward staircase.

    This is the exact, deterministic prediction that the trained per-animal simulations
    (:func:`cohort_rl_error_rate` for corridors, :func:`cohort_turn_error_rate` for turns)
    converge to, and the curve actually plotted in the RL column of the
    error-propagation figure. Value exists only where reward has been experienced and TD
    propagates it back exactly one position per rewarded traverse, so position ``i`` sits at
    ``chance`` until its onset traverse ``i`` and is ``0`` thereafter (chance on traverse 1
    for every position). On the Mask-A linear path each decision is binary (one correct of
    two reachable), so the pre-learning error is ``0.5`` for both corridor steps and turns.
    Direction-independent.

    Parameters
    ----------
    n_positions : int
        Number of path positions (rows); row ``i`` = the position at distance ``i+1`` from
        the reward when the reward row has been dropped, ordered close -> far.
    n_traverses : int
        Number of traverses/journeys (columns); column ``t`` is the ``t``-th traverse
        (0-based here; traverse 1 = column 0 = pre-reward = chance everywhere).
    chance : float, default 0.5
        Pre-learning error level (binary decision on the linear path).
    dead_end_last : bool, default False
        If True, hold the far position (index ``n_positions-1``) at ``0`` for all traverses.
        The corridor row needs this: its far end is the start corridor, a forced one-way
        dead end (single neighbour into the maze), so every departure necessarily reduces
        distance-to-reward and is scored correct from traverse 1 -- it is never a real
        decision. Both the animals and the trained sim show that position flat at 0. The
        turn row has no such position (all 9 holes are genuine two-way turns).

    Returns
    -------
    np.ndarray, shape (n_positions, n_traverses)
        Error rate per (position, traverse): ``chance`` where ``traverse <= position``
        (0-based), else ``0``; the far position pinned to ``0`` when ``dead_end_last``.

    Notes
    -----
    Row ``i`` at column ``t`` is ``chance`` iff ``i >= t`` (position not yet reached by the
    reward frontier), else ``0``. Column 0 (traverse 1) is ``chance`` for every position:
    no reward has been experienced, and unrewarded exploration teaches a model-free agent
    nothing. See ``docs/rl_error_propagation.md`` Section 5 for the derivation.
    """
    stair = np.array([[0.0 if i < t else chance for t in range(n_traverses)]
                      for i in range(n_positions)])
    if dead_end_last and n_positions:
        stair[-1] = 0.0
    return stair


def neighbors_from_adjacency(adj_mat):
    """
    Adjacency lists for every node of a corridor adjacency matrix.

    Parameters
    ----------
    adj_mat : np.ndarray, shape (n, n)
        Corridor adjacency matrix (``1`` = edge). May be the full 22-corridor maze matrix;
        isolated off-path corridors simply get empty neighbour lists.

    Returns
    -------
    list of np.ndarray
        ``neighbors[i]`` is the array of corridor indices adjacent to corridor ``i``.
    """
    return [np.where(adj_mat[i] > 0)[0] for i in range(adj_mat.shape[0])]


def train_on_journey(q_table, journey, goal, neighbors, adj_mat, gamma, alpha):
    """
    Cumulative model-free TD(0) update from one journey's entire corridor sequence.

    Every bout of the journey (sorties + terminating traverse) is read separately; only
    transitions that are true graph edges (``adj_mat[c, c2] == 1``) update ``q_table`` — so
    the corridor repeats / discontinuities at bout seams (the mouse exiting the maze) are
    skipped. Reward ``+1`` is experienced on entering ``goal``.

    Parameters
    ----------
    q_table : np.ndarray, shape (n, n)
        Action-value table ``Q[corridor, next_corridor]``. Modified in place.
    journey : sequence of array-like
        The journey as a list of per-bout corridor-index sequences.
    goal : int
        Reward corridor index.
    neighbors : list of np.ndarray
        Output of :func:`neighbors_from_adjacency`.
    adj_mat : np.ndarray
        Corridor adjacency matrix (validity check).
    gamma, alpha : float
        Discount and learning rate.
    """
    for seq in journey:
        seq = np.asarray(seq, dtype=int)
        for k in range(len(seq) - 1):
            ca, cb = int(seq[k]), int(seq[k + 1])
            if adj_mat[ca, cb] != 1:                    # repeat / seam / non-edge -> not a transition
                continue
            reward = 1.0 if cb == goal else 0.0
            target = reward if cb == goal else reward + gamma * q_table[cb, neighbors[cb]].max()
            q_table[ca, cb] += alpha * (target - q_table[ca, cb])


def simulate_walk(q_table, start, goal, neighbors, distances, rng, max_steps):
    """
    One hybrid greedy/random walk from ``start`` to ``goal``; return its distance sequence.

    At each corridor the agent acts GREEDILY when its Q over the available next-corridors is
    informative (``ptp > 0``), otherwise it takes a RANDOM adjacent step (random-walker
    fallback for not-yet-learned corridors). No learning happens here.

    Returns
    -------
    np.ndarray
        Distance-to-reward at each visited corridor (``distances`` indexed by the trajectory).
    """
    corridor = start
    traj = [corridor]
    for _ in range(max_steps):
        if corridor == goal:
            break
        acts = neighbors[corridor]
        qv = q_table[corridor, acts]
        if np.ptp(qv) > 1e-12:                          # informative -> greedy (ties random)
            best = acts[qv == qv.max()]
            corridor = best[rng.integers(len(best))]
        else:                                           # uninformative -> random adjacent step
            corridor = acts[rng.integers(len(acts))]
        traj.append(corridor)
    return distances[np.asarray(traj, dtype=int)]


def hybrid_error_readout(q_table, start, goal, neighbors, distances, n_pos, rng, n_walks):
    """
    Pooled per-corridor error rate over ``n_walks`` hybrid-policy simulated walks.

    Each walk is scored with :func:`localize_distance_seq`; errors and opportunities are
    pooled across walks so the per-departing-distance rate is smooth (not a single 0/1
    sample). Row ``d`` = distance-to-reward ``d``.

    Returns
    -------
    np.ndarray, shape (n_pos,)
        Error rate per departing distance; NaN where a distance was never departed from.
    """
    counts = np.zeros(n_pos)
    opps = np.zeros(n_pos)
    for _ in range(n_walks):
        dist_seq = simulate_walk(q_table, start, goal, neighbors, distances, rng, max_steps=n_pos * 60)
        c, o = localize_distance_seq(dist_seq, n_pos)
        counts += c
        opps += o
    return np.divide(counts, opps, out=np.full(n_pos, np.nan), where=opps > 0)


def agent_error_matrix(journeys, start, goal, adj_mat, distances, n_pos, rng,
                       n_walks=300, gamma=0.9, alpha=0.5):
    """
    One cumulative agent over one animal's journeys (one direction): ``(n_pos, n_journeys)``.

    For each journey in order, read out the hybrid-policy per-corridor error rate FIRST,
    then train on that journey (cumulative). So column ``t`` reflects only what was
    learned from the reward on journeys BEFORE ``t`` -- column 0 is the untrained
    random-walker rate, since no reward has been experienced yet. This "readout before
    reward" ordering matches how the animals are scored (a traverse's steps are made
    before its reward is collected); reading out AFTER each journey's reward instead
    would credit the agent with the current traverse's reward.

    Parameters
    ----------
    journeys : list of (list of array-like)
        The animal's journeys for this direction, each a list of per-bout corridor sequences.
    start, goal : int
        Start and reward corridor indices for this direction.
    adj_mat : np.ndarray
        Corridor adjacency matrix.
    distances : np.ndarray, shape (n,)
        Distance from each corridor to ``goal`` (``mask.corridors_shortest_distance[:, goal]``).
    n_pos : int
        Number of distance bins (rows); row ``d`` = distance-to-reward ``d``.
    rng : np.random.Generator
        RNG for this agent's readout rollouts.
    n_walks, gamma, alpha : see module docstring.

    Returns
    -------
    np.ndarray, shape (n_pos, len(journeys))
    """
    neighbors = neighbors_from_adjacency(adj_mat)
    q_table = np.zeros((adj_mat.shape[0], adj_mat.shape[0]), dtype=float)
    mat = np.full((n_pos, len(journeys)), np.nan)
    for t, journey in enumerate(journeys):
        mat[:, t] = hybrid_error_readout(q_table, start, goal, neighbors, distances,
                                         n_pos, rng, n_walks)  # readout before reward
        train_on_journey(q_table, journey, goal, neighbors, adj_mat, gamma, alpha)
    return mat


def cohort_rl_error_rate(animal_journeys, start, goal, adj_mat, distances, n_pos,
                         width=25, seed=0, **agent_kwargs):
    """
    Population-mean RL error-rate matrix over animals, mirroring
    :func:`manhattan_maze.analysis.cohort_position_error_rate`.

    Each animal yields a per-journey ``(n_pos, n_journeys)`` matrix; per position the animal
    rows are NaN-padded to ``width`` columns (head-aligned, via
    :func:`manhattan_maze.utils.extract_array`) and averaged with ``np.nanmean``.

    Parameters
    ----------
    animal_journeys : list of (list of (list of array-like))
        Per-animal journey lists for one direction (see :func:`agent_error_matrix`). Animals
        with no journeys in this direction are skipped.
    start, goal, adj_mat, distances, n_pos
        See :func:`agent_error_matrix`.
    width : int, default 25
        Common column width (traverses/journeys) — match the empirical corridor key.
    seed : int, default 0
        Base seed; each animal's agent gets an independent child RNG via
        ``np.random.SeedSequence(seed).spawn`` (reproducible).
    **agent_kwargs
        Forwarded to :func:`agent_error_matrix` (``n_walks``, ``gamma``, ``alpha``).

    Returns
    -------
    (np.ndarray, int)
        ``(cohort_matrix (n_pos, width), n_animals)``.
    """
    animals = [aj for aj in animal_journeys if aj]
    child_seeds = np.random.SeedSequence(seed).spawn(len(animals))
    per_animal = [
        agent_error_matrix(aj, start, goal, adj_mat, distances, n_pos,
                           rng=np.random.default_rng(ss), **agent_kwargs)
        for aj, ss in zip(animals, child_seeds)
    ]
    cohort = np.full((n_pos, width), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN positions
        for p in range(n_pos):
            stacked = extract_array([m[p, :] for m in per_animal], size=width)  # (n_animals, width)
            cohort[p, :] = np.nanmean(stacked, axis=0)
    return cohort, len(per_animal)


# ---------------------------------------------------------------------------------------
# Self-play corridor agent (Mask-D bottleneck comparison, fig:algo row 1).
#
# Off-policy self-play: the EXPERIENCE is a memoryless random walk to the goal (the animal's
# pre-reward sorties carry no reward signal, so they teach a model-free learner nothing — the
# random-walk traverse is the fair, information-equivalent stand-in), only the reward is real.
# READ OUT is greedy corridor error + P(gateway->bottleneck) BEFORE each reward, so traverse 1 is
# the beta=0.5 null. Unlike the linear-track staircase there is no closed form for the ramp: TD's
# max is nonlinear and the biclique splits reward propagation across parallel arms, so the speed is
# a Monte-Carlo statistic — hence this simulation (driven by gen_maskd_model_comparison.py, drawn
# in fig:algo). The Endotaxis counterpart IS analytic (one-pass map) and is computed in the plot.
# ---------------------------------------------------------------------------------------
def random_walk_to_goal(start, goal, neighbors, rng, max_steps):
    """Memoryless random walk from ``start`` to ``goal`` (the self-play experience).

    Each step is a uniform random adjacent corridor until the goal is reached or ``max_steps``.
    Returns the corridor-index trajectory (``np.ndarray``)."""
    corridor, traj = int(start), [int(start)]
    for _ in range(max_steps):
        if corridor == goal:
            break
        acts = neighbors[corridor]
        corridor = int(acts[rng.integers(len(acts))])
        traj.append(corridor)
    return np.asarray(traj, dtype=int)


def bottleneck_choice_prob(q_table, gateway, neighbors, bottleneck):
    """P(agent steps ``gateway``->``bottleneck``) under the greedy/random-fallback policy on the
    frozen Q-table: greedy (ties uniform) where Q over the gateway's neighbours is informative,
    else uniform over neighbours (chance ``1/deg``)."""
    acts = neighbors[gateway]
    qv = q_table[gateway, acts]
    if np.ptp(qv) > _PTP_TOL:
        best = acts[qv == qv.max()]
        return float(np.mean(best == bottleneck))
    return float(np.mean(acts == bottleneck))


def greedy_corridor_error(q_table, start, goal, neighbors, distances, rng, n_walks, max_steps,
                          error_type="count"):
    """Mean corridor error of the greedy/random-fallback policy over ``n_walks`` :func:`simulate_walk`
    rollouts on the frozen Q-table. ``error_type="count"`` uses the scalar
    :func:`manhattan_maze.analysis.calculate_seq_error` (unbounded count); ``"rate"`` uses
    :func:`manhattan_maze.analysis.calculate_seq_error_rate` (per-step non-progress fraction, [0,1],
    matching the mouse ``"corridor error rate"`` readout)."""
    readout = calculate_seq_error_rate if error_type == "rate" else calculate_seq_error
    return float(np.mean([
        readout(simulate_walk(q_table, start, goal, neighbors, distances, rng, max_steps))
        for _ in range(n_walks)]))


def selfplay_curve(start, goal, gateway, bottleneck, adj_mat, distances, rng,
                   n_traverses, n_walks, gamma, alpha, max_steps=5000, error_type="count"):
    """One self-play agent: per traverse, read out greedy corridor error + P(gateway->bottleneck)
    BEFORE the reward, then experience one memoryless random walk to the goal and TD(0)-update.

    ``error_type`` ("count"/"rate") selects the corridor-error readout units. Returns ``(err, bn)``
    length-``n_traverses`` vectors. Readout-before-reward makes traverse 0 (column 0) the beta=0.5
    null for both metrics."""
    neighbors = neighbors_from_adjacency(adj_mat)
    q = np.zeros((adj_mat.shape[0], adj_mat.shape[0]), dtype=float)
    err = np.empty(n_traverses)
    bn = np.empty(n_traverses)
    for t in range(n_traverses):
        err[t] = greedy_corridor_error(q, start, goal, neighbors, distances, rng, n_walks, max_steps,
                                       error_type=error_type)
        bn[t] = bottleneck_choice_prob(q, gateway, neighbors, bottleneck)
        experience = [random_walk_to_goal(start, goal, neighbors, rng, max_steps)]
        train_on_journey(q, experience, goal, neighbors, adj_mat, gamma, alpha)
    return err, bn


def cohort_selfplay(start, goal, gateway, bottleneck, adj_mat, distances, n_traverses, n_walks,
                    n_seeds=20, seed=0, gamma=0.9, alpha=0.5, max_steps=5000, error_type="count"):
    """Stack ``n_seeds`` independent self-play agents into ``(err, bn)``, each shape
    ``(n_seeds, n_traverses)``. ``error_type`` ("count"/"rate") selects the corridor-error units.
    The spread across seeds is the model's Monte-Carlo band (the ramp has no closed form; see the
    section comment). Reproducible via ``np.random.SeedSequence``."""
    child_seeds = np.random.SeedSequence(seed).spawn(n_seeds)
    curves = [selfplay_curve(start, goal, gateway, bottleneck, adj_mat, distances,
                             np.random.default_rng(ss), n_traverses, n_walks, gamma, alpha, max_steps,
                             error_type=error_type)
              for ss in child_seeds]
    err = np.array([c[0] for c in curves])
    bn = np.array([c[1] for c in curves])
    return err, bn


# ---------------------------------------------------------------------------------------
# Turn agent: model-free RL over maze holes (Q[hole, allocentric direction]).
# ---------------------------------------------------------------------------------------
def _hole(xy):
    """Normalize an ``(x, y)`` pair (possibly ``np.int64``) to a plain ``int`` tuple."""
    return (int(xy[0]), int(xy[1]))


def train_on_journey_decisions(q_table, hole_idx, journey, goal_port, gamma, alpha):
    """
    Cumulative model-free TD(0) update from one journey's entire decision sequence.

    Every bout's ordered ``(hole, dir)`` decisions update ``q_table`` in place. The
    action's outcome is the next thing the mouse crossed: the next decision's hole
    within the bout, or — for the bout's final decision — that bout's ending port.
    An action onto ``goal_port`` yields reward ``+1`` (terminal); onto the other port
    a terminal ``0``; onto another hole it bootstraps from that hole's best action.

    Parameters
    ----------
    q_table : np.ndarray, shape (n_holes, 4)
        Action-value table ``Q[hole_idx, dir_idx]``. Modified in place.
    hole_idx : dict
        ``{hole: row index}`` into ``q_table``.
    journey : sequence of (decisions, end_port)
        One entry per bout: ``decisions`` is the ordered ``[(hole, dir), ...]`` list
        (``Bout.get_hole_decisions``); ``end_port`` is ``"OUT"`` / ``"HOME"`` — the
        port the bout terminates at (``bout_type[-1]``).
    goal_port : {"OUT", "HOME"}
        The reward port for this direction's agent.
    gamma, alpha : float
        Discount and learning rate.
    """
    for decisions, end_port in journey:
        for k, (hole, direction) in enumerate(decisions):
            hole = _hole(hole)
            if hole not in hole_idx or direction not in _DIR_IDX:
                continue
            nxt = _hole(decisions[k + 1][0]) if k + 1 < len(decisions) else end_port
            if nxt in _PORTS:                        # terminal: reward, no bootstrap
                target = 1.0 if nxt == goal_port else 0.0
            else:                                    # onto another hole: bootstrap its best action
                target = gamma * q_table[hole_idx[nxt]].max()
            i, j = hole_idx[hole], _DIR_IDX[direction]
            q_table[i, j] += alpha * (target - q_table[i, j])


def first_turn_error_vector(q_table, hole_idx, cmap):
    """
    Exact per-hole first-turn error along the shortest path, ordered close->far.

    Marches the shortest-path holes of ``cmap`` (``Mask.correct_approach_map`` order,
    start->goal). At each hole the approach axis restricts the choice to the two
    perpendicular headings, exactly one being ``correct_exit`` (chance 0.5). If the
    two candidate Q-values are informative (``ptp>0``) the agent is greedy and scores
    0 (correct) or 1 (wrong); otherwise the hole is unlearned and scores 0.5.

    Parameters
    ----------
    q_table : np.ndarray, shape (n_holes, 4)
    hole_idx : dict
        ``{hole: row index}``.
    cmap : dict
        ``{hole: (approach, exit)}`` from ``Mask.correct_approach_map``.

    Returns
    -------
    np.ndarray, shape (n_holes,)
        First-turn error per hole, index 0 = closest to reward (``cmap`` order
        reversed, matching :func:`analysis.hole_error_rate_by_direction`).
    """
    errs = []
    for hole, (approach, exit_dir) in cmap.items():
        hole = _hole(hole)
        approach_axis = turn_axis(approach)
        # readout only makes sense at a genuine turn (exit perpendicular to approach)
        assert turn_axis(exit_dir) != approach_axis, (
            f"hole {hole}: exit {exit_dir} not perpendicular to approach {approach}"
        )
        candidates = ["N", "S"] if approach_axis == "H" else ["E", "W"]
        qv = np.array([q_table[hole_idx[hole], _DIR_IDX[c]] for c in candidates])
        if np.ptp(qv) > _PTP_TOL:                    # informative -> greedy
            chosen = candidates[int(np.argmax(qv))]
            errs.append(0.0 if chosen == exit_dir else 1.0)
        else:                                        # unlearned -> chance
            errs.append(0.5)
    return np.array(errs[::-1])


def agent_turn_error_matrix(journeys, cmap, holes, goal_port, gamma=0.9, alpha=0.5):
    """
    One cumulative agent over one animal's journeys (one direction): ``(n_holes, n_journeys)``.

    For each journey in order, read out the first-turn error vector FIRST, then train on
    that journey (cumulative). So column ``t`` reflects only what was learned from the
    reward on journeys BEFORE ``t`` -- column 0 is at chance (0.5) everywhere, since no
    reward has been experienced yet (unrewarded sorties teach a model-free agent
    nothing). This "readout before reward" ordering matches how the animals are scored
    (a traverse's turns are made before its reward is collected) and yields the exact
    backward staircase; reading out AFTER each journey's reward instead would credit the
    agent with the current traverse's reward and start column 0 below chance. Row order
    is close->far from reward.

    Parameters
    ----------
    journeys : list of (list of (decisions, end_port))
        The animal's journeys for this direction (per-bout decision lists + end port).
    cmap : dict
        ``{hole: (approach, exit)}`` for this direction.
    holes : sequence of (x, y)
        Maze holes (defines the ``Q`` rows / ``hole_idx``).
    goal_port : {"OUT", "HOME"}
    gamma, alpha : float

    Returns
    -------
    np.ndarray, shape (n_holes, len(journeys))
    """
    holes = [_hole(h) for h in holes]
    hole_idx = {h: i for i, h in enumerate(holes)}
    q_table = np.zeros((len(holes), len(DIRECTIONS)), dtype=float)
    mat = np.full((len(holes), len(journeys)), np.nan)
    for t, journey in enumerate(journeys):
        mat[:, t] = first_turn_error_vector(q_table, hole_idx, cmap)  # readout before reward
        train_on_journey_decisions(q_table, hole_idx, journey, goal_port, gamma, alpha)
    return mat


def cohort_turn_error_rate(animal_journeys, cmap, holes, goal_port,
                           width=25, gamma=0.9, alpha=0.5):
    """
    Population-mean turn-error matrix over animals, mirroring
    :func:`manhattan_maze.analysis.hole_error_rate_by_direction`.

    Each animal yields a per-journey ``(n_holes, n_journeys)`` matrix; per hole the
    animal rows are NaN-padded to ``width`` columns (head-aligned, via
    :func:`manhattan_maze.utils.extract_array`) and averaged with ``np.nanmean``.
    The readout is deterministic, so the result is fully reproducible (no RNG).

    Parameters
    ----------
    animal_journeys : list of (list of (list of (decisions, end_port)))
        Per-animal journey lists for one direction. Animals with no journeys in this
        direction are skipped.
    cmap, holes, goal_port, gamma, alpha
        See :func:`agent_turn_error_matrix`.
    width : int, default 25
        Common column width (journeys) — match the empirical hole key.

    Returns
    -------
    (np.ndarray, int)
        ``(cohort_matrix (n_holes, width), n_animals)``.
    """
    animals = [aj for aj in animal_journeys if aj]
    per_animal = [
        agent_turn_error_matrix(aj, cmap, holes, goal_port, gamma=gamma, alpha=alpha)
        for aj in animals
    ]
    n_holes = len(holes)
    cohort = np.full((n_holes, width), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN holes
        for h in range(n_holes):
            stacked = extract_array([m[h, :] for m in per_animal], size=width)
            cohort[h, :] = np.nanmean(stacked, axis=0)
    return cohort, len(per_animal)


if __name__ == "__main__":
    warnings.warn(
        "rl_model.py is a path-free library (corridor + turn model-free RL agents). Run the "
        "Mask-A cohort simulations and write their figure-data via scripts/gen_rl_simulation.py "
        "(corridors) and scripts/gen_rl_turn_simulation.py (turns), which own the geometry, "
        "trajectory extraction and figure-data keys.",
        stacklevel=2,
    )
