"""
Trajectory data objects for the Manhattan Maze pipeline.

This module holds the three nested data containers that represent one animal's
recorded behaviour, independent of how that behaviour is loaded or quality-
controlled (loading/QC lives in :mod:`manhattan_maze.data_loader`).

Terminology (C9 — used consistently across docstrings and column names)
----------------------------------------------------------------------
- **session** : one full recording under a single fixed mask configuration.
  Represented by :class:`Session`.
- **bout** : a single continuous run between two stops at a port (the atomic
  unit of trajectory data). Represented by :class:`Bout`.
- **traverse** : a completed port-to-port path (home→out ``"H-O"`` or
  out→home ``"O-H"``). A bout that starts and ends at the *same* port is a
  *sortie* (``"H-H"`` / ``"O-O"``), not a traverse.
- **journey** : the group of sorties followed by the traverse that ends them,
  i.e. one slice produced by :meth:`Session.slice_to_journeys`.
- **trajectory** : the ordered list of sessions for one experiment/nickname.
  Represented by :class:`Trajectory`.

Indexing conventions
--------------------
- ``bout_idx`` / ``traverse_idx`` are 0-based internal array indices.
- ``traverse_number`` / ``bout_number`` are 1-based display/manuscript numbers.
- The symbol ``b`` is reserved for *bout* in the manuscript and is not used as a
  generic loop variable here.
"""

import warnings
from collections import Counter
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manhattan_maze import utils, plot_utils


# Why each deprecated turn-error scoring is inaccurate (keyed by `include` mode).
_DEPRECATED_INCLUDE_REASONS = {
    "approach": (
        "pools every crossing, so the denominator is endogenous: reversals re-cross the "
        "same decision holes and inflate both numerator and denominator, counting correlated "
        "re-decisions of one hole as independent trials"
    ),
    "all": (
        "counts forced errors from wrong-corridor approaches (crossings where the correct "
        "direction is unreachable), which inflates the rate"
    ),
}


def _warn_deprecated_include(mode):
    """Warn that a deprecated turn-error scoring mode is inaccurate."""
    warnings.warn(
        f"include={mode!r} turn error rate is deprecated: it {_DEPRECATED_INCLUDE_REASONS[mode]}. "
        f"Use include='first' (the default): one first-decision trial per decision hole, "
        f"approach-conditioned, with an exact chance level of 0.5.",
        DeprecationWarning,
        stacklevel=3,
    )


class Trajectory:
    """
    Ordered collection of :class:`Session` objects for one experiment.

    A trajectory is one animal recording (one nickname). It owns the masks used
    across its sessions and the per-session frame segments. Index it to retrieve
    a session: ``trajectory[session_idx]`` returns the ``session_idx``-th
    :class:`Session`.

    Notes
    -----
    A trajectory can be built either from raw processed data (pass
    ``processed_trajectory``, ``mask_order`` and ``masks``) or from an existing
    list of :class:`Session` objects (pass ``sessions``), e.g. after filtering.
    """

    def __init__(self, processed_trajectory=None, mask_order=None, masks=None, name=None, frame_segments=None,
                 sessions=None, FPS=None, rwd_df=None):
        if sessions is None: # if sessions is None, we assume that processed_trajectory, mask_order and masks are provided
            assert (processed_trajectory is not None) and (mask_order is not None) and (masks is not None), \
                "If sessions is None, processed_trajectory, mask_order and masks must be provided."
            self.name = name # name of the trajectory, usually the experiment name/nickname
            self.masks = masks # dictionary of masks, load from dataloader
            self.mask_order = mask_order # list of mask objects in the order they appear in the trajectory
            self.frame_segments = frame_segments # list of tuples with (first_frame, last_frame) for each session
            self.rwd_df = rwd_df
            mask_list = [masks[mask_name] for mask_name in mask_order] # list of mask objects
            self.sessions = [Session(session, mask=mask, idx=session_index, trajectory=self,
                                     first_frame=frame_segment[0], last_frame=frame_segment[1], FPS=FPS, name=name)
                             for session_index, (session, mask, frame_segment)
                             in enumerate(zip(processed_trajectory, mask_list, frame_segments))] # sessions in one experiment saved in a list
            self.FPS = FPS # frames per second, used for bout duration calculations

        else:
            self.sessions = sessions # list of Session objects,
            self.name = name # name of the trajectory, usually the experiment name/nickname
            self.frame_segments = [(session.first_frame, session.last_frame) for session in sessions]
            self.mask_order = [session.mask.name for session in sessions]
            self.masks = {session.mask.name: session.mask for session in sessions}
            self.FPS = FPS
            self.rwd_df = rwd_df

    def filter(self, criteria):
        """
        Return a new Trajectory keeping only bouts that satisfy ``criteria``.

        Parameters
        ----------
        criteria : str, callable, or list
            A single criterion or list of criteria applied to each :class:`Bout`.
            A callable must return True/False for a bout. A string must be one of
            the predefined bout types: ``"outbound"``, ``"homebound"``,
            ``"traverse"``, ``"sortie"``, ``"H-O"``, ``"O-H"``, ``"H-H"``,
            ``"O-O"``.

        Returns
        -------
        Trajectory
            New trajectory whose sessions contain only the matching bouts.
        """
        filtered_sessions = [session.filter(criteria) for session in self.sessions]
        return Trajectory(sessions=filtered_sessions, name=self.name)

    def count_bouts(self):
        """
        Count the total number of bouts across all sessions in the trajectory.

        Returns
        -------
        int
            Sum of bout counts over every session.
        """
        return sum([len(session) for session in self.sessions])

    def __getitem__(self, idx):
        """Return the ``idx``-th :class:`Session` (0-based)."""
        return self.sessions[idx]

    def __len__(self):
        return len(self.sessions)

    def __str__(self):
        return f"Traj({self.mask_order})"

    def __repr__(self):
        return str(self)

# noinspection PyTypeChecker
class Session:
    """
    One recording session under a single mask configuration.

    A session owns an ordered list of :class:`Bout` objects (``self.bouts``), the
    :class:`~manhattan_maze.mask.Mask` used during the session, and the session's
    frame bounds. Index it to retrieve a bout: ``session[bout_idx]`` returns the
    ``bout_idx``-th :class:`Bout` (0-based).

    Notes
    -----
    - A session uses exactly one mask for its entire duration.
    - ``self.bouts`` holds the segmented bouts (renamed from the former
      ``self.self``, R10).
    - ``self.reward_interval_seconds`` holds inter-reward intervals in **seconds**
      (C7/R3). The deprecated alias ``self.rwd_int_array`` returns the same array.
    """
    def __init__(self, bouts_list, mask=None, idx=None, trajectory=None, first_frame=None, last_frame=None, FPS=None, name=None):
        self.mask = mask # Mask used in this session
        self.idx = idx # index of the session in the trajectory
        self.trajectory = trajectory # Trajectory object that this session belongs to
        self.first_frame = first_frame # first frame of the session, used for bout duration calculations
        self.last_frame = last_frame # last frame of the session, used for bout duration calculations
        self.FPS = FPS # frames per second, used for bout duration calculations
        self.name = name # name of the trajectory, usually the experiment name/nickname
        self.bouts = [Bout(bout=bout, mask=mask, idx=bout_idx, session=self, trajectory=self.trajectory, FPS=self.FPS)
                      for bout_idx, bout in enumerate(bouts_list) if len(bout)>1] # segmented bouts, must be provided as a list of bout-format content
        self.reward_interval_seconds = self.extract_rwd_intervals_array(trajectory.rwd_df, end_frame_window=None) # units: seconds
        self._n_rewards = len(self.reward_interval_seconds) # number of rewards based on LED data.

    @property
    def rwd_int_array(self):
        """
        Deprecated alias for :attr:`reward_interval_seconds` (units: **seconds**).

        Retained for one release (R3/C7) so existing scripts keep working; new
        code, exports, and plot labels must use ``reward_interval_seconds``.
        """
        return self.reward_interval_seconds

    def __getitem__(self, idx):
        """Return the ``idx``-th :class:`Bout` (0-based)."""
        return self.bouts[idx]

    def filter(self, criteria):
        """
        Return a new Session keeping only bouts that satisfy ``criteria``.

        Parameters
        ----------
        criteria : str, callable, or list
            Single criterion or list applied to each :class:`Bout`. See
            :meth:`Bout.satisfy` for the predefined string criteria.

        Returns
        -------
        Session
            New session with the same metadata containing only matching bouts.
        """
        filtered_bouts = [bout for bout in self.bouts if bout.satisfy(criteria)]
        return Session(filtered_bouts, self.mask, self.idx, self.trajectory, self.first_frame, self.last_frame, self.FPS, self.name)

    def get_tiles_per_corridor(self):
        """
        Mean number of tiles traversed per corridor over the whole session.

        Returns
        -------
        float
            ``n_tiles / n_corridors`` across all bouts. Values above ~1 indicate
            back-and-forth scanning within corridors.
        """
        n_corridors = len(self.concat_corridors_df())
        n_tiles = len(self.concat_tiles_df())
        return n_tiles/n_corridors

    def slice(self, start=None, end=None):
        """
        Return a new Session containing bouts ``[start:end]`` (by bout index).

        Parameters
        ----------
        start : int or None
            First bout index to include [bout_idx, 0-based]. Defaults to 0.
        end : int or None
            First bout index to exclude. Defaults to the number of bouts.

        Returns
        -------
        Session
            Sub-session spanning the selected bouts, with ``first_frame`` /
            ``last_frame`` adjusted to the sliced bouts. Bout indices are not
            re-numbered; call :meth:`reset_index` if continuous indices are needed.
        """
        if start is None:
            start = 0
        if end is None:
            end = len(self.bouts)
        first_frame = self.bouts[start].get_frames()[0, 0] if start < len(self.bouts) else self.first_frame
        last_frame = self.bouts[end - 1].get_frames()[-1, -1] if end <= len(self.bouts) else self.last_frame
        return Session(self.bouts[start:end], self.mask, self.idx, self.trajectory, first_frame, last_frame, self.FPS, self.name)

    def slice_by_traverse_idx(self, start_traverse_idx=None, end_traverse_idx=0):
        """
        Slice the session by traverse index, including the leading sorties.

        Parameters
        ----------
        start_traverse_idx : int or None
            First traverse to include [traverse_idx, 0-based]. ``None`` starts at
            the very first bout.
        end_traverse_idx : int or None, default 0
            Last traverse to include [traverse_idx, 0-based]. ``None`` runs to the
            last bout.

        Returns
        -------
        Session
            Sub-session covering the requested traverse range (the traverse bout
            at ``start_traverse_idx`` is excluded; the one at ``end_traverse_idx``
            is included). Returns the whole session if there are too few traverses.

        Notes
        -----
        ``traverse_idx`` is 0-based: 0 is the first completed traverse (typically
        an outbound H→O traverse).
        """
        ho_indices, oh_indices = self.get_traverse_indices() # get the indices of the homebound and outbound traverses separately
        # sort the two list
        traverse_indices = sorted(ho_indices+oh_indices)

        if start_traverse_idx is None and end_traverse_idx is None:
            print("At least one of start_traverse_idx or end_traverse_idx should be provided; returning the whole session.")
            return self

        # find the maximum value of the two indices
        max_traverse = max(start_traverse_idx if start_traverse_idx is not None else float("-inf"),
             end_traverse_idx if end_traverse_idx is not None else float("-inf"))
        if len(traverse_indices) < max_traverse:
            print(f"Not enough traverses to slice by traverse; returning the whole session.")
            return self

        # None means starting from the very first bout or until the very last bout
        start_idx = None if start_traverse_idx is None else traverse_indices[start_traverse_idx] + 1  # always exclude the first bout (a traverse)
        end_idx = None if end_traverse_idx is None else traverse_indices[end_traverse_idx] + 1  # always include the last bout (a traverse)
        return self.slice(start=start_idx, end=end_idx)

    def slice_to_journeys(self,):
        """
        Split the session into journeys, one per traverse boundary.

        Returns
        -------
        list of Session
            One sub-session per traverse boundary: each slice runs from just after
            the previous traverse up to and including the next traverse (a
            *journey* = leading sorties + their terminating traverse).

        Notes
        -----
        Not every returned slice contains a traverse. The trailing slice (after
        the session's last traverse) — and any slice where the animal produced
        only sorties without completing a traverse — contains sorties only. Callers
        that require a completed traverse per slice must filter accordingly (e.g.
        several analyses drop the last slice, ``slices[:-1]``).
        """
        ho_indices, oh_indices = self.get_traverse_indices()
        # sort the two list
        traverse_indices = [None]+np.arange(len(ho_indices)+len(oh_indices)).tolist()+[None]
        slices = []
        for i in range(len(traverse_indices) - 1):
            start_idx = traverse_indices[i]
            end_idx = traverse_indices[i + 1]
            slices.append(self.slice_by_traverse_idx(start_idx, end_idx))

        return slices

    def concat_tiles_df(self):
        """
        Concatenate the ``tiles_df`` of every bout in the session.

        Returns
        -------
        pandas.DataFrame
            Stacked tile rows (schema as in :attr:`Bout.tiles_df`). Returns the
            empty ``pd.DataFrame`` class sentinel if the session has no bouts.
        """
        return pd.concat([bout.tiles_df for bout in self.bouts]) if self.bouts else pd.DataFrame

    def concat_corridors_df(self):
        """
        Concatenate the ``corridors_df`` of every bout, tagged by bout index.

        Returns
        -------
        pandas.DataFrame
            Stacked corridor rows with an added ``bout_idx`` column [bout_idx,
            0-based]. Empty DataFrame if the session has no bouts.
        """
        df_list = []
        if not self.bouts:
            return pd.DataFrame()
        else:
            for bout in self.bouts:
                corridors_df = bout.corridors_df
                # add column for bout idx (for easier lookup)
                corridors_df['bout_idx'] = bout.idx
                df_list.append(corridors_df)
            return pd.concat(df_list)

    def concat_allocentric_turn_seq(self, **kwargs):
        """
        Concatenate the allocentric turn sequence over all bouts in the session.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`Bout.get_allocentric_turns` (e.g. ``tolerance``).

        Returns
        -------
        list of tuple
            Ordered ``(hole, direction)`` pairs, where direction is one of
            ``'N'``, ``'S'``, ``'E'``, ``'W'`` (allocentric).
        """
        turn_seq = []
        for bout in self.bouts:
            turn_seq.extend(bout.get_allocentric_turns(**kwargs))
        return turn_seq

    def concat_egocentric_turn_seq(self, **kwargs):
        """
        Concatenate the egocentric turn sequence over all bouts in the session.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`Bout.get_egocentric_turns`.

        Returns
        -------
        list of tuple
            Ordered ``(hole, turn)`` pairs, where turn is ``'L'``, ``'R'``,
            ``'B'`` or ``None`` (egocentric).
        """
        turn_seq = []
        for bout in self.bouts:
            turn_seq.extend(bout.get_egocentric_turns(**kwargs))
        return turn_seq

    def get_traverse_indices(self):
        """
        Return the bout indices of outbound and homebound traverses.

        Returns
        -------
        ho_indices : list of int
            Bout indices of H→O traverses [bout_idx, 0-based].
        oh_indices : list of int
            Bout indices of O→H traverses [bout_idx, 0-based].
        """
        ho_indices = [b.idx for b in self.filter("H-O")]
        oh_indices = [b.idx for b in self.filter("O-H")]
        return ho_indices, oh_indices

    def reset_index(self):
        """
        Renumber bouts in this (possibly sliced) session to be 0..n-1.

        Returns
        -------
        Session
            ``self``, with each bout's ``idx`` reset to its position.
        """
        for i, bout in enumerate(self.bouts):
            bout.idx = i
        return self

    def get_traverse_similarity_matrix(self, similarity_function=utils.transition_vec_similarity,
                                       n_guaranteed_transitions=None):
        """
        Compute H→O vs O→H traverse-pair similarity matrix.

        Parameters
        ----------
        similarity_function : callable
            Signature ``f(mat1, mat2, n_guaranteed_transitions) → float``.
            Default is :func:`utils.transition_vec_similarity`.
        n_guaranteed_transitions : int or None
            Correction subtracted from both intersection and union (R15).
            If None, reads ``self.mask.n_guaranteed_transitions_for_adjusted_jaccard``
            (set to 3 on MaskDSpecial, 0 on all other masks).

        Returns
        -------
        np.ndarray, shape (n_HO, n_OH) or None
            Similarity matrix; None if there are no H→O traverses.
        """
        if n_guaranteed_transitions is None:
            n_guaranteed_transitions = getattr(
                self.mask, "n_guaranteed_transitions_for_adjusted_jaccard", 0
            )
        ho_indices, oh_indices = self.get_traverse_indices()
        ho_bouts = [self.bouts[i] for i in ho_indices]
        oh_bouts = [self.bouts[i] for i in oh_indices]
        if len(ho_bouts) == 0:
            return None

        sim_matrix = np.full((len(ho_bouts), len(oh_bouts)), np.nan)
        for i, ho_bout in enumerate(ho_bouts):
            for j, oh_bout in enumerate(oh_bouts):
                ho_mat = ho_bout.get_corridor_transition_matrix(normalize=False)
                oh_mat = oh_bout.get_corridor_transition_matrix(normalize=False)
                # transpose OH matrix so direction matches H→O for retrace similarity
                sim_matrix[i, j] = similarity_function(ho_mat, oh_mat.T,
                                                        n_guaranteed_transitions)
        return sim_matrix

    def get_three_traverse_similarity_matrix(self, similarity_function=utils.transition_vec_similarity,
                                             n_guaranteed_transitions=None):
        """
        Compute H→O self-sim, O→H self-sim, and cross-direction similarity matrices.

        Parameters
        ----------
        similarity_function : callable
            Same as in :meth:`get_traverse_similarity_matrix`.
        n_guaranteed_transitions : int or None
            Same as in :meth:`get_traverse_similarity_matrix`.

        Returns
        -------
        tuple of (np.ndarray or None, np.ndarray or None, np.ndarray or None)
            ``(j_oo, j_hh, j_oh_prime)`` — the manuscript similarity matrices
            $J_{O,O}$ (outbound self), $J_{H,H}$ (homebound self), and
            $J_{O,H'}$ (outbound vs. reversed-homebound cross).
        """
        if n_guaranteed_transitions is None:
            n_guaranteed_transitions = getattr(
                self.mask, "n_guaranteed_transitions_for_adjusted_jaccard", 0
            )
        ho_indices, oh_indices = self.get_traverse_indices()
        ho_bouts = [self.bouts[i] for i in ho_indices]
        oh_bouts = [self.bouts[i] for i in oh_indices]
        j_oo = utils.self_similarity_matrix(
            ho_bouts, similarity_function=similarity_function,
            n_guaranteed_transitions=n_guaranteed_transitions,
        )
        j_hh = utils.self_similarity_matrix(
            oh_bouts, similarity_function=similarity_function,
            n_guaranteed_transitions=n_guaranteed_transitions,
        )
        j_oh_prime = self.get_traverse_similarity_matrix(
            similarity_function=similarity_function,
            n_guaranteed_transitions=n_guaranteed_transitions,
        )
        return j_oo, j_hh, j_oh_prime

    def get_time_in_corridors(self, corridor_order=None):
        """
        Time spent in each corridor during the session.

        Parameters
        ----------
        corridor_order : list of int or None
            Corridor indices to report, in order [corridor index 0-21]. Defaults
            to the mask's outbound shortest-path corridor order.

        Returns
        -------
        np.ndarray, shape (len(corridor_order),)
            Total time in each corridor in **seconds** (NaN never assigned; zero
            for unvisited corridors after summation).
        """
        corridors_df = self.concat_corridors_df()
        if corridor_order is None: # use outbound shortest path as default
            corridor_order = self.mask.corridors_shortest_path
        # find the time spent in each corridor
        time_in_cor = np.full(len(corridor_order), np.nan)
        for corridor_idx, c in enumerate(corridor_order):
            sub_df = corridors_df[corridors_df.corridor == c]
            time_in_cor[corridor_idx] = np.sum(sub_df.out_frame - sub_df.in_frame) / self.FPS
        return time_in_cor

    def get_transition_matrix(self, unit="corridor", normalize=True):
        """
        Sum per-bout transition matrices over the session.

        Parameters
        ----------
        unit : {"corridor", "tile"}, default "corridor"
            Whether transitions are counted between corridors or tiles.
        normalize : bool, default True
            If True, column-normalise to transition probabilities.

        Returns
        -------
        np.ndarray
            ``(22, 22)`` for corridors or ``(2·size², 2·size²)`` for tiles.
            ``mat[j, i]`` is the (normalised) count of i→j transitions.
        """
        if not self.bouts:
            return np.zeros_like(self.mask.corridors_adj_mat) if unit == "corridor" else np.zeros_like(self.mask.tiles_adj_mat)

        # first get counts
        trans_mats = [b.get_transition_matrix(unit=unit, normalize=False) for b in self.bouts]
        trans_mat = np.sum(trans_mats, axis=0)  # sum the transition matrices of all bouts in the session
        if normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                trans_mat = trans_mat / np.sum(trans_mat, axis=0, keepdims=True)
        return trans_mat

    # Exact-match per-bout statistics: unit -> fn(bout, kwargs). Families whose
    # behaviour depends on substrings of `unit` (errors, transition matrices) are
    # handled separately in get_bout_stats below.
    _BOUT_STAT_FNS = {
        "duration":             lambda b, kw: b.get_duration_s(**kw),
        "tile distance":        lambda b, kw: len(b.tiles_df),
        "corridor distance":    lambda b, kw: len(b.corridors_df),
        "farthest tile":        lambda b, kw: b.get_farthest_tile(**kw)[1],
        "farthest corridor":    lambda b, kw: b.get_farthest_corridor(**kw)[1],
        "speed":                lambda b, kw: b.get_speed(**kw),
        "unique corridors":     lambda b, kw: len(b.corridors_df.corridor.unique()),
        "corridor transitions": lambda b, kw: b.get_corridor_transitions(),
        "egocentric turns":     lambda b, kw: b.get_egocentric_turns(**kw),
        "allocentric turns":    lambda b, kw: b.get_allocentric_turns(**kw),
    }

    def get_bout_stats(self, unit="duration", **kwargs):
        """
        Return a per-bout statistic for every bout in the session.

        Parameters
        ----------
        unit : str, default "duration"
            Which statistic to compute per bout. Recognised values include:
            ``"duration"`` [seconds], ``"tile distance"`` / ``"corridor distance"``
            [counts], ``"farthest tile"`` / ``"farthest corridor"``,
            ``"tile error"`` / ``"corridor error"`` [counts], ``"speed"``
            [tiles/s], ``"error rate by hole"``, turn-error variants containing
            ``"error"`` [rate or count], transition-matrix variants,
            ``"unique corridors"``, ``"corridor transitions"``,
            ``"egocentric turns"``, ``"allocentric turns"``.
            ``"turn error rate"`` defaults to the canonical first-decision-per-hole
            approach-conditioned rate (chance 0.5); including ``"all"``/``"approach"``
            in the unit string selects the deprecated pooled scoring. ``"error rate
            by hole"`` is the approach-conditioned all-visits per-hole map.
        **kwargs
            Forwarded to the per-bout method selected by ``unit``.

        Returns
        -------
        list
            One entry per bout, in bout order. Element type depends on ``unit``.

        Raises
        ------
        ValueError
            If ``unit`` is not recognised.
        """
        # Fast path: exact-match units resolve to a single per-bout function.
        stat_fn = self._BOUT_STAT_FNS.get(unit)
        if stat_fn is not None:
            return [stat_fn(b, kwargs) for b in self]

        # Families whose behaviour is derived from substrings of `unit`. Order is
        # preserved from the original chain: the tile/corridor-error alias is
        # matched before the generic "error" branch.
        # Graph errors (tiles or corridors): count by default, per-step rate when the unit
        # ends in "rate". Matched before the generic "error" branch so the rate variants do
        # not fall through to the turn/hole count_error path.
        if unit in ("tile error", "corridor error", "tile error rate", "corridor error rate"):
            kwargs["unit"] = "tile" if "tile" in unit else "corridor"
            kwargs["error_type"] = "rate" if "rate" in unit else "count"
            return [b.get_graph_error(**kwargs) for b in self]
        if unit == "error rate by hole": # error rate per hole; returns (n_holes, n_bouts) array
            # approach-conditioned per-hole map (accurate; see Bout.count_error)
            bout_data = [1 - b.get_hole_correctness_vec(include="all", condition="approach") for b in self]
            return np.array(bout_data).T
        if "error" in unit and "hole" not in unit: # error rate/count from decisions at holes
            kwargs["unit"] = "turn" if "turn" in unit else "null" # more features can be added here
            # Default is the canonical first-decision-per-hole approach-conditioned rate;
            # "all"/"approach" in the unit string select the deprecated pooled modes (each
            # emits a DeprecationWarning).
            kwargs["include"] = "all" if "all" in unit else "approach" if "approach" in unit else "first"
            kwargs["error_type"] = "rate" if "rate" in unit else "count"
            return [b.count_error(**kwargs) for b in self]
        if "transition matrix" in unit: # transition matrix per bout (tiles or corridors)
            kwargs["unit"] = "corridor" if "corridor" in unit else "tile"
            return [b.get_transition_matrix(**kwargs) for b in self]
        raise ValueError(f"Unrecognized unit: {unit!r}")

    def get_reward_intervals(self, reference_frame=0):
        """
        Inter-reward intervals measured by in-maze bout timing.

        Rewards are delivered at traverse completions, so intervals are computed
        from the cumulative in-maze time at each traverse bout.

        Parameters
        ----------
        reference_frame : float, default 0
            Cumulative-time origin prepended before the first traverse [seconds].

        Returns
        -------
        np.ndarray or list
            Inter-traverse intervals in **seconds** (distinct from
            :attr:`reward_interval_seconds`, which is derived from LED frames).
            Empty list if the session contains no traverses.
        """
        times_in_maze = np.cumsum(self.get_bout_stats(unit="duration")) # get the cumulative time in the maze
        # get traverse index
        ho_indices, oh_indices = self.get_traverse_indices()
        # add two list and sort them
        reward_indices = sorted(ho_indices + oh_indices)
        if not reward_indices and times_in_maze.size !=0:
            return []
            # return times_in_maze[-1] # float (to be differentiated from reward
        else:
            reward_intervals = np.diff(times_in_maze[reward_indices], prepend=reference_frame, axis=0)  # get the time intervals
            return reward_intervals

    def get_sortie_counts(self, reference_index=0):
        """
        Number of sorties between consecutive rewards (traverses).

        Parameters
        ----------
        reference_index : int, default 0
            Index used to seed the first interval so the first reward's leading
            sorties are counted from bout -1.

        Returns
        -------
        np.ndarray
            Count of sorties preceding each traverse, in traverse order.
        """
        ho_indices, oh_indices = self.get_traverse_indices()
        # add two list and sort them
        reward_indices = sorted(ho_indices + oh_indices)
        sortie_intervals = np.diff(reward_indices, prepend=reference_index-1, axis=0)-1  # get the intervals and avoid repeated counts
        return sortie_intervals

    def get_slice_stats(self, unit="duration", **kwargs):
        """
        Compute a per-journey statistic by slicing the session at traverses.

        Parameters
        ----------
        unit : str, default "duration"
            Statistic to compute. ``"reward intervals"``, ``"time to first
            reward"`` and ``"sortie counts"`` are computed without traverse
            slicing; all other units aggregate over journeys (slices between
            traverses). See the body for the full set of recognised units.
        **kwargs
            Forwarded to the underlying computation.

        Returns
        -------
        list or scalar
            One value per journey (or a scalar for ``"time to first reward"``).
            Units follow ``unit`` (seconds for duration/intervals, counts for
            distances/errors).

        Raises
        ------
        ValueError
            If ``unit`` is not recognised.
        """
        # for intervals and counts slicing by traverse is not needed
        if unit == "reward intervals": # get the time intervals between rewards in the session
            slice_data = self.get_reward_intervals(**kwargs)
            return slice_data
        if unit == "time to first reward": # get the time to the first reward in the session
            slice_data = self.get_reward_intervals(**kwargs)
            return slice_data[0] if len(slice_data) > 0 else np.nan # return the first interval if exists, otherwise return nan
        elif unit == "sortie counts": # get the number of sorties between rewards in the session
            slice_data = self.get_sortie_counts(**kwargs)
            return slice_data
        else: # need to sli by traverse
            session_slices = self.slice_to_journeys()

        if unit == "duration" or unit == "corridor distance" or unit == "tile distance" or unit=="corridor error" or unit=="tile error": # sum that for each sli
            slice_data = [np.sum(sli.get_bout_stats(unit)) for sli in session_slices]  # sum that for each sli
        elif unit == "corridor overlap ratio": # get the overlap ratio of the traverse over its preceding sorties
            slice_data = []
            for sl in session_slices:
                if len(sl) <=1: # if there is only one bout, no overlap can be calculated
                    slice_data.append(0)
                else: # if there are multiple self, calculate the overlap ratio
                    pre_corridors = sl.slice(0, -1).concat_corridors_df()["corridor"].tolist() # get the corridors of the previous self
                    post_corridors = sl.slice(-1, None).concat_corridors_df()["corridor"].tolist() # get the corridors of the last bout, which is traverse
                    slice_data.append(utils.check_overlap_percentage(pre_corridors, post_corridors)) # get overlap
        elif unit == "farthest tile" or unit == "farthest corridor": # only look at the displacement relatively to each bout start
            slice_data = [np.max(sli.get_bout_stats(unit)) for sli in session_slices]
        elif unit == "unique corridors": # get the number of unique corridors in each sli (concatenated)
            slice_data = [len(sli.concat_corridors_df().corridor.unique()) if len(sli)> 0 else 0 for sli in session_slices]
        elif unit == "corridor ratios": # get the ratio of the corridor distance to the number of unique corridors in each sli
            distance = [np.sum(sli.get_bout_stats("corridor distance")) for sli in session_slices]
            unique = [len(sli.concat_corridors_df().corridor.unique()) if len(sli) > 0 else 0 for sli in session_slices]
            slice_data = [d / u if u > 0 else np.nan for d, u in zip(distance, unique)] # avoid division by zero
        elif "transition matrix" in unit: # get the transition matrix for each sli
            unit = "corridor" if "corridor" in unit else "tile" # set the unit based on the unit
            slice_data = [sli.get_transition_matrix(unit=unit, **kwargs) for sli in session_slices]
        elif unit =="bottleneck choice":
            slice_data = self.get_bottleneck_choice_ratio(**kwargs)
        else:
            raise ValueError("Unrecognized unit")

        return slice_data

    def get_bottleneck_choice_ratio(self, bottleneck_node=1, outbound_bottleneck_neighbor=19,
                              homebound_bottleneck_neighbor=12, rewarded=True):
        """
        Per-reward probability of choosing the bottleneck corridor (Mask D only).

        Parameters
        ----------
        bottleneck_node : int, default 1
            Corridor index of the bottleneck $c_b$ [corridor index 0-21].
        outbound_bottleneck_neighbor : int, default 19
            Outbound-side neighbor of the bottleneck (a member of $\\mathcal{N}^+(c_b)$);
            the corridor from which the outbound bottleneck choice is scored.
        homebound_bottleneck_neighbor : int, default 12
            Homebound-side neighbor of the bottleneck (a member of $\\mathcal{N}^+(c_b)$);
            the corridor from which the homebound bottleneck choice is scored.
        rewarded : bool, default True
            If True, return NaN when the animal obtained no rewards.

        Returns
        -------
        np.ndarray
            Bottleneck-choice ratio per journey (before the final unrewarded
            slice). NaN entries mark empty journeys.

        Raises
        ------
        AssertionError
            If the session mask is not Mask D.
        """
        assert self.mask.name == "D", "Bottleneck choice only applies to mask D for now"
        # check number of rewards in session
        session_slices = self.slice_to_journeys()
        if len(session_slices) <= 1 and rewarded:
            # if there is only one slice it means the mouse didn't get any reward
            return np.array([np.nan])
        choice_ratio_list = []
        for sli in session_slices[:-1]: # last one is after the last reward
            if len(sli) == 0:
                choice_ratio_list.append(np.nan)
            else:
                first_bout = sli[0]
                if first_bout.satisfy("outbound"):
                    bottleneck_neighbor = outbound_bottleneck_neighbor
                else:
                    bottleneck_neighbor = homebound_bottleneck_neighbor
                trans_mat = sli.get_transition_matrix(unit="corridor")
                choice_ratio_list.append(trans_mat[bottleneck_node, bottleneck_neighbor])

        return np.array(choice_ratio_list)


    def get_binned_hist(self, bw=3, tm=180, unit="tile"):
        """
        Histogram of step times (in-maze time) for speed estimation.

        Parameters
        ----------
        bw : float, default 3
            Bin width in **seconds**.
        tm : float, default 180
            Maximum time to bin in **seconds**.
        unit : {"tile", "corridor"}, default "tile"
            Which step sequence to histogram.

        Returns
        -------
        ed : np.ndarray
            Bin edges [seconds].
        sp : np.ndarray
            Step counts per bin.

        Raises
        ------
        ValueError
            If ``unit`` is not "tile" or "corridor".

        See Also
        --------
        manhattan_maze.plot_data.get_step_times_data : builds the step-time point process.
        manhattan_maze.plot_behavior.binned_step_counts : does the binning.
        """
        # in_maze_end_s is left at inf: this method never clamped tm to the in-maze end
        # (plot_speed did), and delegating must not quietly add that clamp.
        return plot_utils.binned_step_counts(
            utils.get_step_times_data(self, unit=unit)["step_time_s"].to_numpy(dtype=float),
            session_span_s=(self.last_frame - self.first_frame) / self.FPS,
            in_maze_end_s=np.inf, bw=bw, tm=tm)

    def plot_speed(self, ax, color, bw=3, tm=None, unit="tile", plot_hist=True, **kwargs):
        # histogram that point process to compute speed = d(steps)/d(time)
        """
        Plot session speed (steps per second) over in-maze time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on.
        color : color spec
            Line/bar color.
        bw : float, default 3
            Bin width in **seconds**.
        tm : float or None, default None
            Maximum time in **seconds**; defaults to the session end time.
        unit : {"tile", "corridor"}, default "tile"
            Step unit used for speed.
        plot_hist : bool, default True
            Plot as a histogram (True) or a line (False).
        **kwargs
            Forwarded to the matplotlib plotting call.

        Returns
        -------
        tuple or list
            Whatever ``ax.hist`` or ``ax.plot`` returned.

        See Also
        --------
        manhattan_maze.plot_data.get_step_times_data : the data half of this method.
        manhattan_maze.plot_behavior.plot_speed_hist : the render half.
        """
        return plot_utils.plot_speed_hist(
            ax, utils.get_step_times_data(self, unit=unit), color,
            session_span_s=(self.last_frame - self.first_frame) / self.FPS,
            in_maze_end_s=np.cumsum(self.get_bout_stats(unit="duration"))[-1],
            bw=bw, tm=tm, unit=unit, plot_hist=plot_hist, **kwargs)

    def plot_transition_correctness(self, ax, start_node=19, end_node=1, color="tab:blue", label=None,
                                    reward_color="tab:red", window_size=10, random_val=20,
                                    plot_rewards=True, mode="same", **line_kwargs):
        """
        Plot the moving average of the correctness of the transitions from start_node to end_node over the course of the session.

        .. note::
           **Unused (R9).** No caller anywhere in the repo (no ``gen_*``, ``plot_*``,
           test, or notebook). Unlike the five published panel methods it was therefore
           *not* split into a ``plot_data`` extractor + ``plot_utils`` renderer under R8,
           so it still needs a live ``Session``. Delete it or give it a consumer.

        :param ax:
        :param start_node:
        :param end_node:
        :param color:
        :param reward_color:
        :param window_size:
        :param random_val:
        :param plot_rewards:
        :param mode:
        :param label: graph label
        :param line_kwargs:
        :return:
        """
        if label is None:
            label = f"{start_node} to {end_node}"
        traverse_indices = [b.idx for b in self.filter("traverse")] # this might be empty
        corridor_transitions = self.get_bout_stats("corridor transitions")
        selected_transitions = [[counter for counter in sub_list if counter[0] == start_node] for sub_list in
                                corridor_transitions] # find all transitions starting from start_node
        rewarded_transitions = [[0] * (len(sub_list) - 1) + [1] if k in traverse_indices else [0] * len(sub_list) for
                                k, sub_list in enumerate(selected_transitions)] # mark which transitions are rewarded
            # (the last transition in a traverse)
        # plot a moving average of the chance of selecting the (19, 1)
        # now flatten all transitions:
        selected_transitions = [i for sub_list in selected_transitions for i in sub_list]
        rewarded_transitions = [i for sub_list in rewarded_transitions for i in sub_list]
        correct_transitions = [1 if counter[1] == end_node else 0 for counter in selected_transitions]
        # plot moving average
        if len(correct_transitions) == 0:
            print("No transitions found")
            return None, None,

        av = utils.moving_average(correct_transitions, window_size=window_size, mode=mode)*100
        xs = np.arange(len(av))+window_size//2 # center the x values
        ax.plot(xs, av, color=color, label=label, **line_kwargs)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("average choice %")

        if plot_rewards: # if plot rewards along with the number of transitions
            # plot n rewards in the background
            reward_counts = np.cumsum(rewarded_transitions)
            ax.plot(np.arange(len(reward_counts)), reward_counts, color=reward_color, **line_kwargs, label="Rewards")

        # add title:
        ax.set_xlabel("n(transitions)")
        ax.set_title(
            f"{self.name}: {start_node} to {end_node}",
            fontsize=plot_utils.TICK_SIZE)
        if random_val is not None:
            ax.axhline(y=random_val, color="black", linestyle="--", linewidth=0.5, zorder=20)
            ax.text(x=xs[-1], y=random_val, s="random", ha="right", va="bottom", fontsize=plot_utils.TICK_SIZE, color="black", zorder=20)
        ax.set_xlim(left=xs[0], right=xs[-1])
        return ax

    def plot_tile_distance_over_time(self, ax=None, figsize=None, reward_color="black", linewidth=1,
                                     reference_frame=None, plot_bout_types=True, bout_type_color_dict=None):
        """
        Plot distance-to-home over session time, one grey trace per bout.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, default None
            Axes to draw on; a new figure of ``figsize`` is created if None.
        figsize : tuple or None, default None
            Size of the created figure when ``ax`` is None.
        reward_color : color spec or None, default "black"
            Colour of the reward markers at traverse ends; falsy disables them.
        linewidth : float, default 1
            Trace line width.
        reference_frame : int or None, default None
            Time origin.  None uses this session's own ``first_frame``; pass the parent
            session's first frame to place a sliced segment on the full session's clock.
        plot_bout_types : bool, default True
            Shade each bout by type and annotate traverse numbers.
        bout_type_color_dict : dict or None, default None
            Bout-type to colour mapping; None uses
            :data:`~manhattan_maze.plot_utils.bout_type_color_dict`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.

        See Also
        --------
        manhattan_maze.plot_data.get_tile_distance_data : the data half of this method.
        manhattan_maze.plot_behavior.plot_tile_distance : the render half.
        """
        return plot_utils.plot_tile_distance(
            ax, utils.get_tile_distance_data(self), reference_frame=reference_frame,
            figsize=figsize, reward_color=reward_color, linewidth=linewidth,
            plot_bout_types=plot_bout_types, bout_type_colors=bout_type_color_dict)


    def get_average_step_matrix(self, maskd_special_params=None):
        # plot session:
        mask = self.mask
        corridor_order_indices = maskd_special_params.plot_corridor_order if mask.name == "D" else mask.corridors_shortest_path
        corridors = self.concat_corridors_df().corridor
        average_steps = utils.get_average_step_matrix(corridors, corridor_order_indices)
        return average_steps

    def plot_corridor_average_steps(self, ax, maskd_special_params=None, **kwargs):
        # Unused (R9): reachable only from plot_behavior.plot_markov_comparisons_average_steps,
        # which is exported in __all__ but never called. Like plot_transition_correctness it was
        # therefore not split into an extractor + renderer pair under R8 and still needs a live
        # Session. Delete both, or give them a consumer.
        # plot session:
        mask = self.mask
        average_steps = self.get_average_step_matrix(maskd_special_params)
        corridor_order_indices = maskd_special_params.plot_corridor_order if mask.name == "D" else mask.corridors_shortest_path
        plot_utils.plot_steps_heatmap(ax, average_steps, corridor_order_indices,**kwargs)
        if mask.name == "D":  # if the mask is D, add the bottlenecks and home/out markers
            plot_utils.add_lines_to_matrix_plot(ax)
            shortest_path_order = maskd_special_params.shortest_path_corridor_order
            shortest_path_square_coordinates = [(x, shortest_path_order[i + 1]) for i, x in
                                            enumerate(shortest_path_order[:-1])]
            plot_utils.add_squares_to_matrix_plot(ax, shortest_path_square_coordinates, zorder=10)
        ax.set_title(f"{self.name} Session {self.idx}", )

    def plot_reward_interval_raster(self, ax, y_loc, color="black", markersize=10, y_increment=0.1,
                                    reverse=False, plot_end=True):
        """
        Plot this session's rewards as a raster along the in-maze clock.

        Rewards are delivered at traverse completions, so each reward is drawn at
        the cumulative in-maze time (sleep-thresholded ``get_bout_stats("duration")``,
        seconds) of its traverse bout. Outbound (home->out) rewards are up-triangles
        placed just above ``y_loc``; homebound (out->home) rewards are white
        down-triangles just below it.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        y_loc : float
            Row centre for this session's markers (e.g. an animal index).
        color : str, default "black"
            Marker colour (outbound fill / homebound edge).
        markersize : float, default 10
            Scatter marker size.
        y_increment : float, default 0.1
            Vertical offset separating the outbound row (``y_loc + y_increment``)
            from the homebound row (``y_loc - y_increment``).
        reverse : bool, default False
            If True, shift the clock so ``t = 0`` sits at the *end* of the session
            (all times become negative). Used to place a pre-swap session before a
            swap at ``t = 0`` while the post-swap session runs in positive time.
        plot_end : bool, default True
            If True, draw a short vertical line marking the session end
            (``times_in_maze[-1]``); if False, no line is drawn and ``end_line`` is
            ``None``. Set False when several sessions share one row and only one
            end marker is wanted.

        Returns
        -------
        tuple
            ``(out_scatter, home_scatter, end_line)`` matplotlib artists for legend
            building. ``end_line`` is the ``Line2D`` session-end marker, or ``None``
            when ``plot_end`` is False.

        See Also
        --------
        manhattan_maze.plot_data.get_reward_raster_data : the data half of this method.
        manhattan_maze.plot_behavior.plot_reward_raster : the render half.
        """
        return plot_utils.plot_reward_raster(
            ax, utils.get_reward_raster_data(self), y_loc, color=color, markersize=markersize,
            y_increment=y_increment, reverse=reverse, plot_end=plot_end)

    def get_correct_turns(self, *args, **kwargs):
        """
        Ground-truth correct turns for this session's mask.

        Returns
        -------
        dict
            Mapping ``(col, row)`` hole → correct allocentric direction
            (``'N'``/``'S'``/``'E'``/``'W'``). Delegates to
            :meth:`Mask.get_correct_turns`.
        """
        return self.mask.get_correct_turns(*args, **kwargs)

    def is_hole(self, *args, **kwargs):
        """Return whether the given position is a hole (delegates to the mask)."""
        return self.mask.is_hole(*args, **kwargs)

    def get_holes(self, *args, **kwargs):
        """Return the mask's hole positions (delegates to the mask)."""
        return self.mask.get_holes(*args, **kwargs)

    def extract_rwd_intervals_array(self, reward_df, end_frame_window=None):
        """
        Inter-reward intervals from raw LED-flash frames, in **seconds**.

        Flattens and sorts all reward-LED frames within the session's frame
        range, then takes successive differences divided by FPS.

        Parameters
        ----------
        reward_df : pandas.DataFrame
            Reward LED frames for the experiment (columns ``rwd_1``, ``rwd_2``).
            Frames are absolute video frame numbers.
        end_frame_window : int or None, default None
            If given, only consider frames within this many frames after
            ``first_frame`` (to avoid noise in late frames). If None, use the
            full session.

        Returns
        -------
        np.ndarray or list
            Intervals in **seconds** (frame differences divided by FPS). Empty
            list if ``reward_df`` is empty.

        Notes
        -----
        Units are **seconds**, not minutes. The previous code/comment labelled
        this "minutes" but the formula divides only by FPS (C7/R3). The first
        element is the gap from ``first_frame`` to the first LED frame.
        """
        # find all the flashes in the first bouts
        first_frame = self.first_frame
        if end_frame_window is None:
            last_frame = self.last_frame
        else:
            last_frame = self.first_frame+end_frame_window # only look at the frames within the window after the first frame, to avoid potential noise in the later frames
        # convert reward_df to numpy array and filter by first and last frame
        if reward_df.empty:
            return []
        else:
            reward_array = reward_df.to_numpy()
            reward_array = reward_array.flatten()
            reward_array = np.sort(reward_array)
            # sort from small to large
            reward_array = reward_array[(reward_array >= first_frame) & (reward_array <= last_frame)]
            # flatten this and extract the intervals
            intervals = np.diff(reward_array, prepend=first_frame) / self.FPS
        return intervals # seconds


    def __len__(self):
        return len(self.bouts)

    def __str__(self):
        return "Session with " + str(len(self.bouts)) + " bouts"

    def __repr__(self):
        return "\n".join([str(self.mask), str(self.bouts)])

class Bout:
    """
    A single continuous run between two port stops — the atomic trajectory unit.

    A bout holds three aligned representations of the same run: ``bout_df``
    (discrete (col,row) cells), ``tiles_df`` (per-floor tiles), and
    ``corridors_df`` (corridor runs). Its ``bout_type`` is one of ``"H-O"``,
    ``"O-H"`` (traverses) or ``"H-H"``, ``"O-O"`` (sorties).

    Parameters
    ----------
    bout : pandas.DataFrame or Bout
        Either a raw ``bout_df`` (columns ``in_frame``, ``out_frame``,
        ``discrete_loc``) or an existing :class:`Bout` to copy.
    mask : Mask, optional
        Maze geometry used to derive tiles/corridors and score turns.
    idx : int, optional
        Bout index within its session [bout_idx, 0-based].
    session, trajectory : optional
        Back-references to the owning containers.
    FPS : int, optional
        Frames per second, used for all duration/speed conversions.

    Raises
    ------
    ValueError
        If ``bout`` is neither a DataFrame nor a Bout.
    """
    def __init__(self, bout, mask=None, idx=None, session=None, trajectory=None, FPS=None):

        if isinstance(bout, pd.DataFrame):
            self.bout_df = bout # raw dataframe from discrete coordinates
            self.mask = mask # mask object
            self.tiles_df = self._build_tiles_df() # tiles df (tile index)
            self.corridors_df = self._build_corridors_df() # corridors df (corridor index)
            # self.decisions_df = self._build_decisions_df() # decisions df (egocentric turns) deprecated
            self.bout_type = self._get_bout_type() # bout type (H-O, O-H, H-H, O-O)

            self.idx = idx # index of the bout in the session, start from 0
            self.session = session # session object
            self.trajectory = trajectory # trajectory object
            self.FPS = FPS # frames per second of the session

        elif isinstance(bout, Bout):
            self.bout_df = bout.to_df()
            self.mask = bout.mask
            self.tiles_df = bout.tiles_df
            self.corridors_df = bout.corridors_df
            # self.decisions_df = bout.decisions_df
            self.bout_type = bout.bout_type

            self.idx = bout.idx
            self.session = bout.session
            self.trajectory = bout.trajectory
            self.FPS = bout.FPS

        else:
            raise ValueError(f"argument bout has type {type(bout)}, must be either Bout or pd.DataFrame")

    def satisfy(self, criteria):
        """
        Test whether the bout satisfies one or more criteria.

        Parameters
        ----------
        criteria : str, callable, or list
            A single criterion or list of criteria, ALL of which must hold.
            Strings select predefined bout types: ``"outbound"``, ``"homebound"``,
            ``"traverse"``, ``"sortie"``, ``"H-O"``, ``"O-H"``, ``"H-H"``,
            ``"O-O"``. A callable must accept a :class:`Bout` and return bool.

        Returns
        -------
        bool
            True if every criterion is satisfied.

        Raises
        ------
        ValueError
            If a string criterion is not one of the predefined names.
        """
        if not isinstance(criteria, list):
            criteria = [criteria]

        for ix, criterion in enumerate(criteria):

            if callable(criterion): # if the criterion is a function
                criteria[ix] = criterion
            elif isinstance(criterion, str):
                if criterion == "outbound":
                    criteria[ix] = lambda bout: bout.is_outbound()
                elif criterion == "homebound":
                    criteria[ix] = lambda bout: bout.is_homebound()
                elif criterion == "traverse":
                    criteria[ix] = lambda bout: bout.is_traverse()
                elif criterion == "sortie":
                    criteria[ix] = lambda bout: not bout.is_traverse()
                elif criterion == "H-O":
                    criteria[ix] = lambda bout: bout.is_traverse() and bout.is_outbound()
                elif criterion == "O-H":
                    criteria[ix] = lambda bout: bout.is_traverse() and bout.is_homebound()
                elif criterion == "H-H":
                    criteria[ix] = lambda bout: (not bout.is_traverse()) and bout.is_outbound()
                elif criterion == "O-O":
                    criteria[ix] = lambda bout: (not bout.is_traverse()) and bout.is_homebound()
                else:
                    raise ValueError(f"Unknown criterion string {criterion}")

        return all([criterion(self) for criterion in criteria])

    def to_df(self):
        """Return a deep copy of ``bout_df`` (columns: in_frame, out_frame, discrete_loc)."""
        return pd.DataFrame(columns=self.bout_df.columns, data=deepcopy(self.bout_df.to_numpy()))

    def to_tiles_df(self):
        """Return a deep copy of ``tiles_df`` (columns: in_frame, out_frame, tile, x, y, z)."""
        return pd.DataFrame(columns=self.tiles_df.columns, data=deepcopy(self.tiles_df.to_numpy()))

    def get_coords(self):
        """
        Return per-tile maze coordinates.

        Returns
        -------
        np.ndarray, shape (n_tiles, 3)
            Rows are ``[x, y, z]`` (column, row, floor).
        """
        return deepcopy(self.tiles_df[['x', 'y', 'z']].to_numpy())

    def get_tiles(self):
        """
        Return the tile-index sequence of the bout.

        Returns
        -------
        np.ndarray, shape (n_tiles,)
            Tile indices [tile index 0-241], in order of traversal.
        """
        return deepcopy(self.tiles_df['tile'].to_numpy())

    def get_corridors(self):
        """
        Return the corridor-index sequence of the bout.

        Returns
        -------
        np.ndarray, shape (n_corridors,)
            Corridor indices [corridor index 0-21], in order of traversal.
        """
        return deepcopy(self.corridors_df['corridor'].to_numpy())

    def get_corridor_distance_seq(self, goal_corridor=None,):
        """
        Graph distance from each visited corridor to a goal corridor.

        Parameters
        ----------
        goal_corridor : int or None
            Goal corridor index [corridor index 0-21]. If None, use the first
            corridor of the bout. A non-positive value indexes from the end of
            the corridor sequence.

        Returns
        -------
        list of float
            Shortest-path corridor distance at each step. Empty if the bout has
            no corridors.
        """
        corridor_seq = self.get_corridors()
        if len(corridor_seq) == 0:
            return np.array([]), np.array([])
        if goal_corridor is None:
            goal_corridor = corridor_seq[0]
        elif goal_corridor <= 0:
            goal_corridor = corridor_seq[goal_corridor]
        distance_seq = [self.mask.corridors_shortest_distance[goal_corridor, corridor] for corridor in corridor_seq]
        return distance_seq

    def get_transition_matrix(self, unit="corridor", normalize=True):
        """
        Transition-count (or probability) matrix for this bout.

        Parameters
        ----------
        unit : {"corridor", "tile"}, default "corridor"
            Whether to count transitions between corridors or tiles.
        normalize : bool, default True
            If True, column-normalise counts to probabilities.

        Returns
        -------
        np.ndarray
            ``mat[j, i]`` counts (or probability of) i→j steps. Shape is
            ``(22, 22)`` for corridors or ``(2·size², 2·size²)`` for tiles.

        Raises
        ------
        ValueError
            If ``unit`` is not "corridor" or "tile".
        """

        if unit == "corridor":
            transition_mat = np.zeros_like(self.mask.corridors_adj_mat)
            seq = self.get_corridors()
        elif unit == "tile":
            transition_mat = np.zeros_like(self.mask.tiles_adj_mat)
            seq  = self.get_tiles() # get the sequence of tiles in the session
        else:
            raise ValueError(f"Unknown unit: {unit}")

        for k, step in enumerate(seq[:-1]):
            transition_mat[seq[k + 1], step] += 1  # increment the transition matrix for the step

        if normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                transition_mat = transition_mat / np.sum(transition_mat, axis=0, keepdims=True)

        return transition_mat

    def get_farthest_tile(self, reverse=False, **kwargs):
        """
        Index and distance of the farthest tile from the goal in the bout.

        Parameters
        ----------
        reverse : bool, default False
            If True, scan the distance sequence from the end.
        **kwargs
            Forwarded to :meth:`get_tile_distance_seq` (e.g. ``goal_tile``).

        Returns
        -------
        tuple (int, float)
            ``(index, distance)`` of the first maximal-distance tile; ``np.nan``
            if the bout has no tiles.
        """
        distance_seq = self.get_tile_distance_seq(**kwargs)
        # find the farthest tile and its index (the first that appeared)
        if len(distance_seq) == 0:
            return np.nan
        else:
            if reverse:
                distance_seq = distance_seq[::-1]
            farthest_idx = np.argmax(distance_seq)

        return farthest_idx, distance_seq[farthest_idx]

    def get_tile_distance_seq(self, goal_tile=None):
        """
        Graph distance from each visited tile to a goal tile.

        Parameters
        ----------
        goal_tile : int or None
            Goal tile index [tile index 0-241]. If None, use the first tile of
            the bout. A non-positive value indexes from the end of the tile
            sequence.

        Returns
        -------
        list of float or np.ndarray
            Shortest-path tile distance at each step. Empty array if the bout has
            no tiles.
        """
        tile_seq = self.get_tiles()
        if len(tile_seq) == 0:
            return np.array([])
        else:
            if goal_tile is None:
                goal_tile = tile_seq[0]
            elif goal_tile <= 0: # if negative, count from the end
                goal_tile = tile_seq[goal_tile]

            distance_seq = [self.mask.tiles_shortest_distances[goal_tile, tile] for tile in tile_seq]
            return distance_seq

    def get_graph_error(self, unit="tile", goal=-1, error_type="count"):
        """
        Moves away from the goal (graph-distance-increasing steps), count or rate.

        Parameters
        ----------
        unit : {"corridor", "tile"}, default "tile"
            Distance space in which to score errors.
        goal : int, default -1
            Goal node; -1 means the last node of the bout.
        error_type : {"count", "rate"}, default "count"
            ``"count"`` returns the raw number of distance-increasing steps
            (strict ``> 0`` rule, unbounded); ``"rate"`` returns the per-step
            non-progress fraction in ``[0, 1]`` (non-decreasing ``>= 0`` rule,
            chance ~0.5), matching the ``error_propagation`` corridor error rate.

        Returns
        -------
        int or float
            Error count (``error_type="count"``) or per-step error rate
            (``error_type="rate"``).

        Raises
        ------
        ValueError
            If ``unit`` is not "corridor" or "tile".
        """
        if unit == "corridor":
            distance_seq = self.get_corridor_distance_seq(goal_corridor=goal)
        elif unit == "tile":
            distance_seq = self.get_tile_distance_seq(goal_tile=goal)
        else:
            raise ValueError(f"Unrecognized distance type {unit} for error sequence extraction")
        # calculate the errors (count of away-steps, or per-step non-progress rate)
        if error_type == "rate":
            return utils.calculate_seq_error_rate(distance_seq)
        return utils.calculate_seq_error(distance_seq)

    def get_speed(self, unit="tile/s"):
        """
        Average locomotion speed of the bout.

        Parameters
        ----------
        unit : {"tile/s"}, default "tile/s"
            Output unit. Currently only tiles per second is supported.

        Returns
        -------
        float
            ``n_tiles / duration`` in tiles per **second** (duration uses the
            default sleep-thresholded :meth:`get_duration_s`). NaN if duration is 0.

        Raises
        ------
        ValueError
            If ``unit`` is not "tile/s".
        """
        if unit == "tile/s":
            distance = len(self.tiles_df)
            duration = self.get_duration_s()
            return distance / duration if duration > 0 else np.nan
        else:
            raise ValueError(f"Unrecognized unit {unit} for speed calculation")

    def slice(self, start=None, end=None):
        """
        Return a new Bout restricted to rows ``[start:end]`` of ``bout_df``.

        Parameters
        ----------
        start : int or None
            First ``bout_df`` row to include (default 0).
        end : int or None
            First ``bout_df`` row to exclude (default len(bout_df)).

        Returns
        -------
        Bout
            New bout over the selected rows.
        """
        if start is None:
            start = 0
        if end is None:
            end = len(self.bout_df)
        return Bout(self.bout_df[start:end], self.mask, self.idx, self.session, self.trajectory, self.FPS,)

    def slice_by_frames(self, start_frame=None, end_frame=None):
        """
        Return a new Bout covering an absolute frame window.

        Parameters
        ----------
        start_frame : int or None
            First absolute video frame to include (default: bout start).
        end_frame : int or None
            Last absolute video frame to include (default: bout end).

        Returns
        -------
        Bout
            New bout over the rows whose frames fall in the window.
        """
        in_frames = self.bout_df['in_frame'].to_numpy()
        out_frames = self.bout_df['out_frame'].to_numpy()
        if start_frame is None:
            start = 0
        else:
            start = np.searchsorted(in_frames, start_frame, side="left")
        if end_frame is None:
            end = len(self.bout_df)
        else:
            end = np.searchsorted(out_frames, end_frame, side="right")
        return Bout(self.bout_df[start:end], self.mask, self.idx, self.session, self.trajectory, self.FPS,)

    def get_farthest_corridor(self, reverse=False, **kwargs):
        """
        Index and distance of the farthest corridor from the goal in the bout.

        Parameters
        ----------
        reverse : bool, default False
            If True, scan the distance sequence from the end.
        **kwargs
            Forwarded to :meth:`get_corridor_distance_seq`.

        Returns
        -------
        tuple (int, float)
            ``(index, distance)`` of the first maximal-distance corridor;
            ``np.nan`` if the bout has no corridors.
        """
        distance_seq = self.get_corridor_distance_seq(**kwargs)
        # find the farthest tile and its index
        if len(distance_seq) == 0:
            return np.nan
        else:
            if reverse:
                distance_seq = distance_seq[::-1]
            farthest_idx = np.argmax(distance_seq)
        return farthest_idx, distance_seq[farthest_idx]

    def get_corridor_transitions(self):
        """
        Count directed corridor transitions in the bout.

        Returns
        -------
        collections.Counter
            Keys are ``(corridor_from, corridor_to)`` pairs [corridor index
            0-21]; values are occurrence counts.
        """
        corridors = self.get_corridors()
        transitions_counters = Counter((a, b) for a, b in zip(corridors[:-1], corridors[1:]))
        return transitions_counters

    def get_corridor_transition_matrix(self, normalize=True):
        """
        Corridor transition matrix for the bout.

        Parameters
        ----------
        normalize : bool, default True
            If True, column-normalise to transition probabilities.

        Returns
        -------
        np.ndarray, shape (22, 22)
            ``mat[j, i]`` is the (normalised) count of corridor i→j transitions.

        Notes
        -----
        The unnormalised boolean form is the input to
        :func:`utils.transition_vec_similarity` for the adjusted-Jaccard
        retrace-similarity analysis (R15).
        """
        transition_counters = self.get_corridor_transitions()
        transition_matrix = np.zeros_like(self.mask.corridors_adj_mat, dtype=float)
        for counter, value in transition_counters.items():
            transition_matrix[counter[1], counter[0]] += value  # increment the transition matrix for the step
        if normalize:
            # ignore runtime warning for invalid divide
            with np.errstate(invalid="ignore"):
                transition_matrix = transition_matrix / transition_matrix.sum(axis=0, keepdims=True)
        return transition_matrix


    def get_frames(self):
        """
        Return per-tile frame bounds.

        Returns
        -------
        np.ndarray, shape (n_tiles, 2)
            Rows are ``(in_frame, out_frame)`` absolute video frames.
        """
        return deepcopy(self.tiles_df[['in_frame', 'out_frame']].to_numpy())

    def get_duration_s(self, sleep_threshold=5):
        """
        Sleep-thresholded bout duration in **seconds**.

        Parameters
        ----------
        sleep_threshold : float or None, default 5
            Per-tile dwell-time cap in **seconds**. Time spent longer than this in
            a single tile is capped at the threshold, to exclude resting/sleeping.
            If None, return raw wall-clock duration (last out_frame − first
            in_frame) / FPS instead.

        Returns
        -------
        float
            Duration in **seconds**.

        Notes
        -----
        The thresholded duration is the published quantity (sum of
        ``min(time_in_cell, 5 s)``), which differs from wall-clock time and is
        the basis for all D∞/D₀/k learning-curve fits.
        """
        in_times, out_times = self.get_frames().T
        if sleep_threshold is None:
            return (out_times[-1] - in_times[0]) / self.FPS
        else:
            # filter out time longer than sleep threshold
            time_in_cell = (out_times - in_times)/self.FPS
            time_in_cell = np.where(time_in_cell < sleep_threshold, time_in_cell, sleep_threshold)
            duration = np.sum(time_in_cell)
            return duration

    def get_xys(self):
        """
        Return the discrete (col, row) cell sequence of the bout.

        Returns
        -------
        list of tuple[int, int]
            ``(col, row)`` positions, one per ``bout_df`` row.
        """
        return self.bout_df['discrete_loc'].tolist()

    def is_outbound(self): # start with home_pos
        """Return True if the bout starts at the home port (home_pos)."""
        xy_seq = self.get_xys()
        return len(xy_seq) > 0 and xy_seq[0] == self.mask.home_pos

    def is_homebound(self): # start with out_pos
        """Return True if the bout starts at the out port (out_pos)."""
        xy_seq = self.get_xys()
        return len(xy_seq) > 0 and xy_seq[0] == self.mask.out_pos

    def is_traverse(self):
        """
        Return True if the bout crosses from one port to the other.

        Notes
        -----
        A traverse goes home→out or out→home; a bout that returns to its starting
        port is a sortie, not a traverse. This classification determines which
        bouts enter the learning-curve analysis.
        """
        xy_seq = self.get_xys()
        if len(xy_seq) == 0:
            return False
        if self.is_outbound():
            return xy_seq[-1] == self.mask.out_pos
        else: # homebound
            return xy_seq[-1] == self.mask.home_pos

    def is_sortie(self):
        """Return True if the bout is a sortie (does not traverse port-to-port)."""
        return not self.is_traverse()

    def get_allocentric_turns(self, tolerance=0):
        """
        Allocentric turn directions taken at maze holes during the bout.

        Parameters
        ----------
        tolerance : int, default 0
            Minimum Manhattan distance past a hole the animal must commit to
            before a turn is scored (smooths micro-reversals); 0 disables look-ahead.

        Returns
        -------
        list of tuple
            ``(hole, direction)`` pairs, direction in ``'N'``/``'S'``/``'E'``/``'W'``.

        Notes
        -----
        The sequence is padded with virtual home/goal endpoints so the first and
        last turns are well defined for outbound and homebound bouts.
        """
        home, goal = (-1, self.mask.size // 2), (self.mask.size // 2, self.mask.size - 1)
        if self.is_outbound():
            xy_seq = [home] + self.get_xys() + [goal]
        elif self.is_homebound():
            xy_seq = [goal] + self.get_xys() + [home]
        else: # likely segment of bout that is created
            xy_seq = self.get_xys()

        return utils.get_allocentric_turns(xy_seq, self.get_holes(), tolerance=tolerance)

    def get_hole_decisions(self):
        """
        Allocentric decision (chosen heading) at EVERY hole crossing during the bout.

        Unlike :meth:`get_allocentric_turns` (which records only direction *changes*),
        this records the heading the mouse leaves each hole on at every crossing,
        including straight-through passes — the per-hole action a turn-based RL agent
        trains on. Uses the same virtual home/goal padding so the first and last holes
        are scored. No tolerance smoothing applies (every crossing is a decision).

        Returns
        -------
        list of tuple
            ``(hole, direction)`` pairs, direction in ``'N'``/``'S'``/``'E'``/``'W'``,
            one entry per hole crossing in order.
        """
        home, goal = (-1, self.mask.size // 2), (self.mask.size // 2, self.mask.size - 1)
        if self.is_outbound():
            xy_seq = [home] + self.get_xys() + [goal]
        elif self.is_homebound():
            xy_seq = [goal] + self.get_xys() + [home]
        else:
            xy_seq = self.get_xys()

        return utils.get_hole_decisions(xy_seq, self.get_holes())

    def get_allocentric_turns_with_approach(self, tolerance=0):
        """
        Hole crossings as ``(hole, approach_dir, exit_dir)`` triples.

        Same as :meth:`get_allocentric_turns` (identical virtual home/goal
        padding) but also records the heading the mouse entered each hole on, so
        callers can condition on the approach corridor. Used by the
        approach-conditioned turn-error metric.

        Parameters
        ----------
        tolerance : int, default 0
            Only ``0`` is supported (the turn-error default).

        Returns
        -------
        list of tuple
            ``(hole, approach_dir, exit_dir)`` with each direction in
            ``'N'``/``'S'``/``'E'``/``'W'``.
        """
        home, goal = (-1, self.mask.size // 2), (self.mask.size // 2, self.mask.size - 1)
        if self.is_outbound():
            xy_seq = [home] + self.get_xys() + [goal]
        elif self.is_homebound():
            xy_seq = [goal] + self.get_xys() + [home]
        else:
            xy_seq = self.get_xys()

        return utils.allocentric_turns_with_approach(xy_seq, self.get_holes(), tolerance=tolerance)

    def get_egocentric_turns(self, tolerance=0):
        """
        Egocentric (left/right/back) turns at holes during the bout.

        Parameters
        ----------
        tolerance : int, default 0
            Look-ahead tolerance, forwarded to :meth:`get_allocentric_turns`.

        Returns
        -------
        list of tuple
            ``(hole, turn)`` pairs, turn in ``'L'``/``'R'``/``'B'``/``None``.
            The initial heading is East for outbound bouts and South for
            homebound bouts.
        """
        turns = self.get_allocentric_turns(tolerance=tolerance)
        prev_dir = 'E' if self.is_outbound() else 'S'
        egocentric_turns = []
        for hole, direction in turns:
            egocentric_direction = utils.to_egocentric_direction(prev_dir, direction)
            egocentric_turns.append((hole, egocentric_direction))
            prev_dir = direction
        return egocentric_turns

    def get_first_hole_dec_on_mask(self, tolerance=0):
        """
        First decision taken at each hole of the mask.

        Parameters
        ----------
        tolerance : int, default 0
            Look-ahead tolerance, forwarded to :meth:`get_allocentric_turns`.

        Returns
        -------
        dict
            Mapping of every mask hole ``(col, row)`` → first allocentric
            direction taken there, or ``None`` if the hole was not visited.

        Raises
        ------
        ValueError
            If a turn is recorded at a position not in the mask's holes.
        """
        seq = self.get_allocentric_turns(tolerance=tolerance)

        holes = self.get_holes()
        all_hole_dict = {hole: None for hole in holes}  # initialize a dictionary with all holes
        first_at_holes = utils.get_first_at_loc(seq)
        # fill the dictionary with None for not visited.
        for hole, direction in first_at_holes.items():
            if hole not in all_hole_dict:
                raise ValueError(f"Hole {hole} not found in the mask holes {holes}")
            all_hole_dict[hole] = direction

        return all_hole_dict

    def _approach_filtered_turns(self, **kwargs):
        """
        Hole crossings restricted to those entered on the correct corridor.

        Keeps only crossings whose approach corridor axis matches the
        shortest-path approach at that hole (see :meth:`get_correct_approach_map`),
        so the correct outgoing direction is actually reachable by the crossing.
        Returns ``[(hole, exit_dir)]`` — the same shape the raw scoring consumes.
        """
        cmap = self.get_correct_approach_map()
        seq = self.get_allocentric_turns_with_approach(**kwargs)
        return [(hole, exit_dir) for hole, approach, exit_dir in seq
                if hole in cmap and utils.turn_axis(approach) == utils.turn_axis(cmap[hole][0])]

    def get_seq_correctness(self, condition="approach", **kwargs):
        """
        Per-turn correctness over the allocentric hole-crossing sequence.

        Parameters
        ----------
        condition : {"approach", "raw"}, default "approach"
            ``"approach"`` scores only crossings entered on the shortest-path
            corridor (the accurate measure; chance level 0.5). ``"raw"`` scores
            every crossing, including wrong-corridor ones whose correct
            direction is unreachable — inflated and inaccurate.
        **kwargs
            Forwarded to the turn extraction.

        Returns
        -------
        list of int
            1 if a crossing matches the mask's correct direction at that hole,
            else 0, in turn order (all visits counted, including repeats).

        Notes
        -----
        This pools every crossing. The canonical published turn-error rate no
        longer uses it: :meth:`count_error` (default ``include="first"``) scores
        the first decision per hole via :meth:`get_hole_correctness_vec` to avoid
        the endogenous, reversal-inflated denominator of pooled crossings.
        """
        correct_turns = self.get_correct_turns()
        if condition == "approach":
            seq = self._approach_filtered_turns(**kwargs)
        else:
            seq = self.get_allocentric_turns(**kwargs)
        return [1 if direction == correct_turns[hole] else 0
                for hole, direction in seq if hole in correct_turns]

    def get_hole_correctness_vec(self, include="all", condition="approach", **kwargs):
        """
        Per-hole turn correctness vector for the bout.

        Parameters
        ----------
        include : {"all", "first"}, default "all"
            ``"all"`` averages correctness over every (scored) visit to each hole;
            ``"first"`` uses only the first.
        condition : {"approach", "raw"}, default "approach"
            ``"approach"`` scores only crossings entered on the shortest-path
            corridor (accurate); ``"raw"`` scores every crossing (inflated).
        **kwargs
            Forwarded to the turn extraction.

        Returns
        -------
        np.ndarray, shape (n_holes,)
            Mean correctness per hole in [0, 1]; NaN for unvisited holes.

        Notes
        -----
        ``include="first", condition="approach"`` is the canonical turn-error
        scoring reduced by :meth:`count_error` (its default). ``include="all"`` is
        used for the hole-resolved ``"error rate by hole"`` map.
        """
        correct_turns = self.get_correct_turns()
        if condition == "approach":
            seq = self._approach_filtered_turns(**kwargs)
        else:
            seq = self.get_allocentric_turns(**kwargs)
        return utils.get_hole_correctness(seq, correct_turns, include=include)

    def count_error(self, unit="turn", include="first", error_type="rate", **kwargs):
        """
        Turn-error count or rate for the bout.

        Parameters
        ----------
        unit : {"turn"}, default "turn"
            Only turn error is supported. Mask D has no turn-error measure.
        include : {"first", "approach", "all"}, default "first"
            ``"first"`` (default) is the canonical measure: one first-decision
            trial per decision hole, approach-conditioned, so the correct turn is
            one of two reachable outcomes — chance level exactly 0.5. Its
            denominator is the geometry-fixed count of distinct decision holes,
            immune to reversal inflation.
            ``"approach"`` / ``"all"`` are **deprecated** (they emit a
            ``DeprecationWarning``): ``"approach"`` pools every crossing, giving
            an endogenous denominator that reversals inflate; ``"all"`` also
            counts wrong-corridor forced errors on top of that.
        error_type : {"rate", "count"}, default "rate"
            ``"rate"`` returns errors / n_scored (Eq. 2); ``"count"`` the raw count.
        **kwargs
            Forwarded to the correctness computation.

        Returns
        -------
        float
            Turn error rate in [0, 1] or raw error count.

        Raises
        ------
        ValueError
            If the mask is D, if ``unit`` is not "turn", or if ``include`` is
            invalid.

        Notes
        -----
        Turn error = fraction of scored decision holes whose first crossing was
        made in the wrong allocentric direction. This is a directly-published
        quantity underlying every learning curve.
        """
        if unit == "turn" and self.mask.name == "D":
            raise ValueError("Mask D does not have turn error rate measured")
        elif unit != "turn":
            raise ValueError(f"Unknown unit {unit}, must be 'turn'")

        if include == "first":
            correctness_vec = self.get_hole_correctness_vec(include="first", condition="approach", **kwargs)
        elif include == "approach":
            _warn_deprecated_include("approach")
            correctness_vec = self.get_seq_correctness(condition="approach", **kwargs)
        elif include == "all":
            _warn_deprecated_include("all")
            correctness_vec = self.get_seq_correctness(condition="raw", **kwargs)
        else:
            raise ValueError(f"Unknown inclusion criterion: {include}")
        return utils.count_error(correctness_vec=correctness_vec, error_type=error_type)

    def get_turn_error_rate(self, include="first", **kwargs):
        """
        Turn error rate $E_{a,b}$ for the bout (manuscript Eq. 2).

        Convenience accessor for the published turn-error rate; equivalent to
        ``count_error(unit="turn", error_type="rate", include=include)``.

        Parameters
        ----------
        include : {"first", "approach", "all"}, default "first"
            ``"first"`` (default) is the canonical first-decision-per-hole,
            approach-conditioned rate (chance 0.5). ``"approach"`` / ``"all"``
            are deprecated (inflated denominator; see :meth:`count_error`) and
            emit a ``DeprecationWarning``.
        **kwargs
            Forwarded to :meth:`count_error`.

        Returns
        -------
        float
            Turn error rate in [0, 1]. Raises for Mask D (no turn-error measure).
        """
        return self.count_error(unit="turn", error_type="rate", include=include, **kwargs)

    def get_correct_turns(self):
        """
        Ground-truth correct turns for this bout's direction.

        Returns
        -------
        dict
            Hole ``(col, row)`` → correct allocentric direction. Uses the mask's
            outbound table for outbound bouts and the homebound table otherwise.
        """
        if self.is_outbound():
            return self.mask.get_correct_turns(homebound=False)
        else:
            return self.mask.get_correct_turns(homebound=True)

    def get_correct_approach_map(self):
        """
        Ground-truth ``{hole: (approach, exit)}`` for this bout's direction.

        Approach-aware analogue of :meth:`get_correct_turns`, used by the
        approach-conditioned turn-error metric. Uses the mask's outbound table
        for outbound bouts and the homebound table otherwise.
        """
        return self.mask.correct_approach_map(homebound=not self.is_outbound())

    def is_hole(self, *args, **kwargs):
        """Return whether a position is a hole (delegates to the mask)."""
        return self.mask.is_hole(*args, **kwargs)

    def get_holes(self, *args, **kwargs):
        """
        Return the mask's holes, ordered for this bout's direction.

        Returns
        -------
        list of tuple[int, int]
            Hole ``(col, row)`` positions. The order is reversed for homebound
            bouts so holes are listed in traversal order.
        """
        holes = self.mask.get_holes(*args, **kwargs)
        if self.is_homebound():
            holes.reverse()
        return holes

    def plot(self, ax=None, fig=None, noise=0.1, fig_size=(3, 3),
             cmap=plt.get_cmap('viridis'), linewidth=3,
             plot_colorbar=True, alpha=1.0, plot_mask=True, color=None, plot_start_time=False,
             plot_duration=False, plot_symbol=False, marker_size=10, marker_color="black"):
        """
        Plot the trajectory of the bout on a given ax.
        :param ax:
        :param fig:
        :param noise: jitter for the line
        :param fig_size: figure size
        :param cmap: color map for the trajectory, defaults to 'viridis'
        :param linewidth: linewidth of the trajectory line
        :param plot_colorbar: if true, plot a colorbar for the trajectory on the same fig
        :param alpha:
        :param plot_mask: if true, plot the mask with holes
        :param color: alternatively plot the trajectory with a single color
        :param plot_duration: add text of duration (s) to the plot
        :param plot_symbol: add symbols for the home and out ports
        :param plot_start_time: add a string for starting time
        :param marker_color
        :param marker_size
        :return:

        See Also
        --------
        manhattan_maze.plot_data.get_bout_path_data : the data half of this method.
        manhattan_maze.plot_behavior.plot_bout_path : the render half.
        """
        return plot_utils.plot_bout_path(
            ax, utils.get_bout_path_data(self), self.mask, fig=fig, noise=noise,
            fig_size=fig_size, cmap=cmap, linewidth=linewidth, plot_colorbar=plot_colorbar,
            alpha=alpha, plot_mask=plot_mask, color=color, plot_start_time=plot_start_time,
            plot_duration=plot_duration, plot_symbol=plot_symbol, marker_size=marker_size,
            marker_color=marker_color)

    def plot_tile_seq(self, ax, goal_tile=None, inverse=False, **plot_kwargs):
        """
        Plot the tile distance (trajectory, relative to the starting point based on the mask
        :param ax:
        :param goal_tile:
        :param inverse:
        :param plot_kwargs:
        :return:

        See Also
        --------
        manhattan_maze.plot_data.get_tile_seq_data : the data half of this method.
        manhattan_maze.plot_behavior.plot_tile_seq : the render half.
        """
        return plot_utils.plot_tile_seq(
            ax, utils.get_tile_seq_data(self, goal_tile=goal_tile), inverse=inverse,
            **plot_kwargs)

    def _build_tiles_df(self):
        """
        Converts bout_df to tiles_df: each tile is determined by its discrete_location.
        When the mouse makes a 90-degree turn, the discrete location is split evenly into two tiles of different floors
        :return:
        """
        bout_df = self.to_df()
        in_frames = bout_df['in_frame'].to_numpy()
        out_frames = bout_df['out_frame'].to_numpy()
        xys = np.array(self.get_xys()) # the xy locations
        if xys.shape[0] < 2:
            print(bout_df)
            print(xys)
            raise ValueError("Warning: Bout must have at least two frames to build tiles_df! Building pseudo tiles_df with one tile.")

        # determine the floor of the first tile, special case
        tiles_data = [[in_frames[0], out_frames[0], utils.xyz_to_ti([xys[0, 0], xys[0, 1], utils.z_xy(xys[1], xys[0])])]]
        for i in range(1, len(in_frames)-1):
            z0 = utils.z_xy(xys[i-1], xys[i])
            z1 = utils.z_xy(xys[i], xys[i+1])
            if z0 == z1: # same floor
                tiles_data.append([in_frames[i], out_frames[i], utils.xyz_to_ti([xys[i, 0], xys[i, 1], z0])])
            else: # split this cell into two for climbing through the hole
                frame_split = (in_frames[i] + out_frames[i]) // 2
                tiles_data.append([in_frames[i], frame_split, utils.xyz_to_ti([xys[i, 0], xys[i, 1], z0])])
                tiles_data.append([frame_split+1, out_frames[i], utils.xyz_to_ti([xys[i, 0], xys[i, 1], z1])])

        # determine the last tile of the bout, special case
        tiles_data.append([in_frames[-1], out_frames[-1], utils.xyz_to_ti([xys[-1, 0], xys[-1, 1], utils.z_xy(xys[-2], xys[-1])])])

        # Now convert tiles_data to dataframes
        tiles_df = pd.DataFrame(tiles_data, columns=['in_frame', 'out_frame', 'tile'])

        # Add x y z columns
        tiles_df[['x', 'y', 'z']] = utils.ti_to_xyz(tiles_df['tile'].to_numpy())
        return tiles_df

    def _build_corridors_df(self):
        """
        Convert tile_df to corridors_df: each corridor is determined by the tiles x, or y
        corridor indices are determined by xyz_to_ci(), time divided half by the hole occupied
        """
        bout_df = self.to_tiles_df()
        bout_df["corridor"] = utils.xyz_to_ci(bout_df[['x', 'y', 'z']].to_numpy(), self.mask.size)
        # Find the first and last frame of each corridor
        in_fr_df, out_fr_df = utils.df_condense_consecutive_repeats(bout_df, "corridor")
        corridors_df = in_fr_df.drop(columns=["x", "y", "z", "tile", "out_frame"]) # the outframes of the first rows are dropped
        corridors_df["out_frame"] = out_fr_df["out_frame"]
        return corridors_df

    def _is_turn(self, index):
        if index > 0 and index + 1 < len(self.bout_df):
            curr_loc = self.bout_df['discrete_loc'].iloc[index]
            prev_loc = self.bout_df['discrete_loc'].iloc[index - 1]
            next_loc = self.bout_df['discrete_loc'].iloc[index + 1]
            return utils.is_turn(prev_loc, curr_loc, next_loc)
        else:
            return False

    def _get_bout_type(self):
        """
        Generate string of bout types marked by start-end
        """
        if self.is_traverse() and self.is_outbound():
            return "H-O"
        elif self.is_traverse() and self.is_homebound():
            return "O-H"
        elif (not self.is_traverse()) and self.is_outbound():
            return "H-H"
        elif (not self.is_traverse()) and self.is_homebound():
            return "O-O"
        else:
            return "Unknown"


    def __getitem__(self, idx):
        """
        idx: int, index of the tile in the bout
        returns the idx-th tile, a tuple of [column, row] indices
        """
        return tuple(self.bout_df.iloc[idx].discrete_loc)

    def __len__(self):
        return len(self.bout_df)

    def __str__(self):
        return 'Bout(' + str(self.get_xys()) + ')'

    def __repr__(self):
        return str(self.bout_df)
