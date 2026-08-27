"""Plot-ready data extraction from live :class:`Session` / :class:`Bout` objects.

Each function here is the *data* half of a plotting method on
:class:`~manhattan_maze.trajectory.Session` or
:class:`~manhattan_maze.trajectory.Bout`: it takes the live object and returns a flat
DataFrame holding exactly what the matching ``plot_utils`` renderer needs.  The three
callers use it as follows:

- ``gen_*.py`` calls the extractor and caches the DataFrame to
  ``data/figure_data/*.parquet``,
- ``plot_*.py`` loads that cache and calls the renderer,
- the plotting method itself is a thin ``extract -> render`` wrapper.

Sharing one extractor between ``gen_*`` and the method is what stops the cache schema
from drifting away from the method's behaviour, and is why the figure caches no longer
need to embed live objects (R8).

Frames, not seconds
-------------------
Tables store absolute ``int64`` video frames plus ``fps``, never pre-converted seconds,
because different consumers use different time origins over the same rows: a sliced
segment's own first frame, the parent session's first frame, or the bout's *last* frame
(``plot_tile_seq(inverse=True)``).  Choosing the origin is the renderer's job.

Nothing is pre-binned or pre-selected either: histogram bin widths and the
``config.py`` example selectors stay authoritative at plot time.

Split out of trajectory.py (R8); re-exported through the ``utils`` facade.
"""

import numpy as np
import pandas as pd

__all__ = ['get_bout_path_data', 'get_tile_seq_data', 'get_reward_raster_data',
           'get_tile_distance_data', 'get_step_times_data', 'get_session_manifest_data',
           'get_example_session_tables', 'get_example_bout_tables',
           'derive_bout_path_table', 'derive_tile_distance_table', 'derive_step_times',
           'iter_example_bout_paths',
           'TRAVERSE_BOUT_TYPES']

#: Bout types that count as port-to-port traverses; the rest are sorties.  Storing
#: ``bout_type`` therefore reproduces every ``Bout.satisfy`` string exactly.
TRAVERSE_BOUT_TYPES = ("H-O", "O-H")


def get_bout_path_data(bout, reference_frame=None):
    """
    Discrete cell path of a bout, plus the scalars its path plot annotates with.

    Data half of :meth:`~manhattan_maze.trajectory.Bout.plot`.

    Parameters
    ----------
    bout : Bout
        Bout to extract.  Read-only.
    reference_frame : int or None, default None
        Origin for ``start_time_s``.  If None, fall back to ``bout.session.first_frame``
        when the bout still carries a session back-reference, else ``start_time_s`` is
        NaN (meaning "no start time available", and the renderer draws no start-time
        text).

    Returns
    -------
    pandas.DataFrame
        One row per ``bout_df`` row, ordered by ``step``:

        ``step`` : int64
            Position within the bout, 0-based.
        ``col``, ``row`` : int64
            Discrete maze cell (column, row) -- the two halves of ``discrete_loc``,
            split because a tuple column round-trips through Parquet as object cells.
        ``n_steps`` : int64
            Number of steps in the bout (constant); sets the colorbar tick labels.
        ``duration_s`` : float
            Sleep-thresholded bout duration in **seconds** (constant).
        ``bout_type`` : str
            One of ``"H-O"``, ``"O-H"``, ``"H-H"``, ``"O-O"``, ``"Unknown"`` (constant).
        ``start_frame`` : int64
            First absolute video frame of the bout (constant).
        ``start_time_s`` : float
            Bout start relative to ``reference_frame``, in **seconds** (constant); NaN
            if no reference is available.
        ``trajectory_name`` : str
            Experiment nickname, or ``""`` if the bout is detached (constant).
        ``session_idx``, ``bout_idx`` : int64
            Session and bout indices used in the auto-title; -1 when unknown (constant).

    Notes
    -----
    The mask is deliberately *not* encoded here.  Maze geometry is passed to the
    renderer as a live :class:`~manhattan_maze.mask.Mask` (the one legitimate object in
    the figure caches under R8's fallback clause), so hole layout and port positions
    cannot go stale relative to ``data/masks/``.
    """
    xys = bout.get_xys()
    frames = bout.get_frames()
    n_steps = len(xys)

    if reference_frame is None:
        session = getattr(bout, "session", None)
        reference_frame = getattr(session, "first_frame", None)

    start_frame = int(frames[0, 0]) if n_steps else -1
    if reference_frame is None or not n_steps:
        start_time_s = np.nan
    else:
        start_time_s = (frames[0, 0] - reference_frame) / bout.FPS

    trajectory = getattr(bout, "trajectory", None)
    session = getattr(bout, "session", None)

    return pd.DataFrame({
        "step": np.arange(n_steps, dtype=np.int64),
        "col": np.array([xy[0] for xy in xys], dtype=np.int64),
        "row": np.array([xy[1] for xy in xys], dtype=np.int64),
        "n_steps": np.int64(n_steps),
        "duration_s": float(bout.get_duration_s()) if n_steps else np.nan,
        "bout_type": bout.bout_type,
        "start_frame": np.int64(start_frame),
        "start_time_s": start_time_s,
        "trajectory_name": getattr(trajectory, "name", None) or "",
        "session_idx": np.int64(_or_missing(getattr(session, "idx", None))),
        "bout_idx": np.int64(_or_missing(getattr(bout, "idx", None))),
    })


def get_tile_seq_data(bout, goal_tile=None):
    """
    Per-tile graph distance to a goal over the course of a bout.

    Data half of :meth:`~manhattan_maze.trajectory.Bout.plot_tile_seq`.

    Parameters
    ----------
    bout : Bout
        Bout to extract.  Read-only.
    goal_tile : int or None, default None
        Goal tile index [tile index 0-241].  None means the mask's home tile, matching
        ``plot_tile_seq``'s default (note this is the *home* tile, not the bout's first
        tile, which is what :meth:`Bout.get_tile_distance_seq` would default to).

    Returns
    -------
    pandas.DataFrame
        One row per ``tiles_df`` row, ordered by ``step``:

        ``step`` : int64
            Position within the tile sequence, 0-based.
        ``in_frame``, ``out_frame`` : int64
            Absolute video frames bounding the tile visit.  Both are needed because the
            renderer's time origin is the first ``in_frame`` normally and the last
            ``out_frame`` when ``inverse=True``.
        ``tile`` : int64
            Tile index [tile index 0-241].
        ``tile_distance`` : float
            Shortest-path tile distance from ``goal_tile`` at this step.
        ``fps`` : float
            Frames per second (constant), so the renderer can convert to seconds.

    Notes
    -----
    Tile rows are *not* bout rows: :meth:`Bout._build_tiles_df` splits a cell in two
    when the animal changes floor within it, so this table is longer than
    :func:`get_bout_path_data`'s and the two must stay separate caches.
    """
    if goal_tile is None:
        goal_tile = bout.mask.home_tile
    distance_seq = np.asarray(bout.get_tile_distance_seq(goal_tile=goal_tile), dtype=float)
    frames = bout.tiles_df[["in_frame", "out_frame"]].to_numpy()

    return pd.DataFrame({
        "step": np.arange(len(distance_seq), dtype=np.int64),
        "in_frame": frames[:, 0].astype(np.int64),
        "out_frame": frames[:, 1].astype(np.int64),
        "tile": np.asarray(bout.get_tiles(), dtype=np.int64),
        "tile_distance": distance_seq,
        "fps": float(bout.FPS),
    })


def get_reward_raster_data(session):
    """
    Reward times of a session on the in-maze clock, as a per-bout table.

    Data half of :meth:`~manhattan_maze.trajectory.Session.plot_reward_interval_raster`.

    Parameters
    ----------
    session : Session
        Session to extract.  Read-only.

    Returns
    -------
    pandas.DataFrame
        One row per bout, in bout order:

        ``bout_idx`` : int64
            Position of the bout in the session, 0-based.
        ``cum_duration_s`` : float
            Cumulative sleep-thresholded in-maze time at the *end* of this bout, in
            **seconds**.  Rewards are delivered at traverse completions, so a traverse
            row's value is that reward's time.
        ``is_ho``, ``is_oh`` : bool
            Whether this bout is an outbound (home->out) or homebound (out->home)
            traverse, i.e. whether it earned an Out or Home reward.

    Notes
    -----
    ``bout_idx`` is the bout's *position*, which is how the method indexes
    ``np.cumsum(...)`` with :meth:`Session.get_traverse_indices` output.  On a session
    sliced with :meth:`Session.slice` the two only agree after
    :meth:`Session.reset_index`; this table reproduces the method's behaviour rather
    than silently diverging from it.
    """
    cum_duration_s = np.cumsum(session.get_bout_stats(unit="duration"))
    ho_indices, oh_indices = session.get_traverse_indices()
    positions = np.arange(len(cum_duration_s))

    return pd.DataFrame({
        "bout_idx": positions.astype(np.int64),
        "cum_duration_s": np.asarray(cum_duration_s, dtype=float),
        "is_ho": np.isin(positions, ho_indices),
        "is_oh": np.isin(positions, oh_indices),
    })


def get_tile_distance_data(session):
    """
    Per-tile distance-to-home traces for every bout of a session.

    Data half of
    :meth:`~manhattan_maze.trajectory.Session.plot_tile_distance_over_time`.

    Parameters
    ----------
    session : Session
        Session to extract.  Read-only.

    Returns
    -------
    pandas.DataFrame
        One row per (bout, tile step), sorted by ``bout_idx`` then ``step``:

        ``bout_idx`` : int64
            Position of the bout in the session, 0-based.
        ``step`` : int64
            Position within the bout's tile sequence, 0-based.
        ``in_frame``, ``out_frame`` : int64
            Absolute video frames bounding the tile visit.
        ``tile_distance`` : float
            Shortest-path tile distance from the mask's home tile.
        ``bout_type`` : str
            Bout type, which selects the highlight colour and reward marker.
        ``traverse_number`` : int64
            1-based traverse number for annotating H-O/O-H bouts, or -1 for sorties.
            An explicit sentinel is used rather than a null because an ``int64`` column
            with NaNs is promoted to ``float64`` and would render as ``Trav.#1.0``.
        ``fps`` : float
            Frames per second (constant).
        ``session_first_frame`` : int64
            The session's own first frame (constant), used as the default time origin.

    Notes
    -----
    Bouts with no tiles contribute no rows, matching the method, which draws an empty
    line for them.
    """
    traverse_indices = [b.idx for b in session.filter("traverse")]
    home_tile = session.mask.home_tile

    per_bout = []
    for position, bout in enumerate(session):
        distance_seq = np.asarray(bout.get_tile_distance_seq(goal_tile=home_tile), dtype=float)
        if not len(distance_seq):
            continue
        frames = bout.tiles_df[["in_frame", "out_frame"]].to_numpy()
        traverse_number = (traverse_indices.index(bout.idx) + 1
                           if bout.idx in traverse_indices else -1)
        per_bout.append(pd.DataFrame({
            "bout_idx": np.int64(position),
            "step": np.arange(len(distance_seq), dtype=np.int64),
            "in_frame": frames[:, 0].astype(np.int64),
            "out_frame": frames[:, 1].astype(np.int64),
            "tile_distance": distance_seq,
            "bout_type": bout.bout_type,
            "traverse_number": np.int64(traverse_number),
        }))

    if not per_bout:
        table = pd.DataFrame({
            "bout_idx": np.array([], dtype=np.int64),
            "step": np.array([], dtype=np.int64),
            "in_frame": np.array([], dtype=np.int64),
            "out_frame": np.array([], dtype=np.int64),
            "tile_distance": np.array([], dtype=float),
            "bout_type": np.array([], dtype=object),
            "traverse_number": np.array([], dtype=np.int64),
        })
    else:
        table = pd.concat(per_bout, ignore_index=True)

    table["fps"] = float(session.FPS)
    table["session_first_frame"] = np.int64(session.first_frame)
    return table


def get_step_times_data(session, unit="tile"):
    """
    Step-completion times of a session on the in-maze clock (a point process).

    Data half of :meth:`~manhattan_maze.trajectory.Session.plot_speed`, which
    histograms this point process to get steps per second.

    Parameters
    ----------
    session : Session
        Session to extract.  Read-only.
    unit : {"tile", "corridor"}, default "tile"
        Which step sequence to time.  Only ``"tile"`` is used by the manuscript
        figures; ``"corridor"`` is supported for parity with
        :meth:`Session.get_binned_hist` but is not cached anywhere.

    Returns
    -------
    pandas.DataFrame
        One row per step, in chronological order, with a single column
        ``step_time_s`` : float -- the step's completion time in **seconds** of in-maze
        time (per-bout frames measured from each bout's own first frame, accumulated
        across bouts).  May be empty if no bout has any step.

    Raises
    ------
    ValueError
        If ``unit`` is not "tile" or "corridor".

    Notes
    -----
    Deliberately *not* binned here.  Bin width and time limit are render parameters
    supplied per figure panel (3 s / 120 s for the Mask-D speed panel, 300 s / 7200 s
    for the Mask-A one), so pre-binning in ``gen_*`` would bake one panel's choice into
    the cache.  :func:`~manhattan_maze.plot_behavior.binned_step_rate` does the binning
    and additionally needs the two session scalars carried by
    :func:`get_session_manifest_data`.
    """
    step_times_s, elapsed_frames = [], 0
    for bout in session:
        if unit == "tile":
            steps = bout.tiles_df
        elif unit == "corridor":
            steps = bout.corridors_df
        else:
            raise ValueError("Unrecognized unit for histogram: {}".format(unit))
        first_frame = bout.get_frames()[0, 0]  # each bout is timed from its own start
        if len(steps) != 0:
            out_frames = steps.out_frame.to_numpy()
            step_times_s += ((out_frames - first_frame + elapsed_frames) / session.FPS).tolist()
        elapsed_frames += bout.get_frames()[-1, 1] - first_frame

    return pd.DataFrame({"step_time_s": np.asarray(step_times_s, dtype=float)})


def get_session_manifest_data(sessions, cache=None, cohort_rows=None):
    """
    One-row-per-session index of the example sessions behind a figure panel.

    Serves two purposes: it carries the session scalars the renderers need but that no
    per-step table owns (``fps``, frame bounds, ``in_maze_end_s``), and it lets the
    consumers that genuinely need live objects -- ``gen_endotaxis.py`` and
    ``report_panel_n.py`` -- reload them from a :class:`DataLoader` by name instead of
    unpickling them.

    Parameters
    ----------
    sessions : sequence of Session
        The example sessions, in the order the figure's ``config.py`` selector indexes
        them (``MASK_A_EXAMPLE_ID`` and friends are *positional*).
    cache : str or None, default None
        Name of the figure-data family these sessions belong to, recorded in the
        ``cache`` column so several panels can share one manifest key.
    cohort_rows : sequence of int or None, default None
        Row of the corresponding whole-cohort metric array (e.g.
        ``"Control Mask A duration"``) that each example session occupies -- normally the
        ``np.random.choice`` indices the producer used to pick them.  Recording these
        replaces the numerical identity join that plot scripts previously did to recover
        an example's cohort row.  None writes the -1 sentinel.

    Returns
    -------
    pandas.DataFrame
        One row per session:

        ``cache`` : str
            Figure-data family name, or ``""``.
        ``example`` : int64
            Positional index of the session within ``sessions``.
        ``animal_name`` : str
            Session nickname, e.g. ``"T3_a1"``.
        ``session_idx`` : int64
            Index of the session within its trajectory, for reloading it.
        ``first_frame``, ``last_frame`` : int64
            Session frame bounds.
        ``fps`` : float
            Frames per second.
        ``mask_name`` : str
            Mask used for the whole session.
        ``n_bouts`` : int64
            Number of bouts, which is the ``n`` reported for example-session panels.
        ``in_maze_end_s`` : float
            Total sleep-thresholded in-maze time in **seconds**; the speed histogram
            clamps its time axis to this.
        ``session_span_s`` : float
            Wall-clock session span in **seconds**, i.e.
            ``(last_frame - first_frame) / fps``; sets the number of speed-histogram
            bins.
    """
    rows = []
    for example, session in enumerate(sessions):
        cum_duration_s = np.cumsum(session.get_bout_stats(unit="duration"))
        rows.append({
            "cache": cache or "",
            "example": np.int64(example),
            "cohort_row": np.int64(-1 if cohort_rows is None else cohort_rows[example]),
            "animal_name": session.name or "",
            "session_idx": np.int64(_or_missing(session.idx)),
            "first_frame": np.int64(session.first_frame),
            "last_frame": np.int64(session.last_frame),
            "fps": float(session.FPS),
            "mask_name": getattr(session.mask, "name", None) or "",
            "n_bouts": np.int64(len(session)),
            "in_maze_end_s": float(cum_duration_s[-1]) if len(cum_duration_s) else np.nan,
            "session_span_s": (session.last_frame - session.first_frame) / session.FPS,
        })
    return pd.DataFrame(rows)


def _or_missing(value, missing=-1):
    """
    Return ``value`` as an int, or ``missing`` when it is None.

    Parameters
    ----------
    value : int or None
        Index that may be absent on a detached object.
    missing : int, default -1
        Sentinel written instead of a null, so the column stays ``int64`` (a null would
        promote it to ``float64`` and render indices as ``1.0``).

    Returns
    -------
    int
        ``int(value)``, or ``missing``.
    """
    return missing if value is None else int(value)


def get_example_session_tables(sessions, cache=None, cohort_rows=None):
    """
    Build the full set of flat tables for a list of example sessions.

    This is what ``gen_*.py`` calls to replace a pickled list of live
    :class:`~manhattan_maze.trajectory.Session` objects.  Every bout of every session is
    exported -- the *superset* -- because a long-format table of a whole session costs
    about two orders of magnitude less than the pickle did, and exporting everything
    keeps the ``config.py`` example selectors (``MASK_A_SEGMENT_BOUTS``,
    ``MASK_D_MOTIF_TRAVERSES``, ...) authoritative at *plot* time rather than baking one
    selection into the cache.

    Parameters
    ----------
    sessions : sequence of Session
        Example sessions in the order the figure's ``config.py`` selector indexes them
        (those selectors are positional).
    cache : str or None, default None
        Figure-data family name recorded in the manifest's ``cache`` column.
    cohort_rows : sequence of int or None, default None
        Row each example occupies in the corresponding whole-cohort metric array; passed
        through to :func:`get_session_manifest_data`.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        Keys are the figure-data key *suffixes*, so a caller writes each as
        ``f"{base} {suffix}"``:

        ``"bout steps"``
            Concatenated :func:`get_bout_path_data`, keyed by ``example``/``bout_idx``.
        ``"tile steps"``
            Concatenated :func:`get_tile_seq_data`, keyed by ``example``/``bout_idx``.
            Longer than "bout steps" because turn cells are split per floor.
        ``"bout meta"``
            One row per bout: the per-bout scalars from :func:`get_bout_path_data`
            merged with :func:`get_reward_raster_data`, plus ``traverse_idx`` so plot
            scripts can select the *n*-th traverse without a live ``filter("traverse")``.
        ``"manifest"``
            :func:`get_session_manifest_data`, carrying the session scalars and the
            ``(animal_name, session_idx)`` pairs that let object-needing consumers
            reload from a :class:`DataLoader`.

    Notes
    -----
    Only *primitive* tables are returned.  The distance-over-time frame and the
    speed-histogram point process are both fully derivable from ``"tile steps"`` (plus
    scalars already in ``"bout meta"`` and ``"manifest"``), so caching them too would
    store the same numbers three times; :func:`derive_tile_distance_table` and
    :func:`derive_step_times` reconstruct them at plot time instead.

    Manifests are deliberately per-family rather than one repo-wide key: the producer
    scripts run in parallel from ``batch_generate_figure_data.py``, so a single shared
    manifest file would be clobbered last-writer-wins.

    Consumes no ``np.random``, so inserting a call into a ``gen_*`` script cannot shift
    the RNG stream that selects which animals are the examples (R11).
    """
    bout_steps, tile_steps, bout_meta = [], [], []

    for example, session in enumerate(sessions):
        raster = get_reward_raster_data(session)
        traverse_indices = [bout.idx for bout in session.filter("traverse")]

        for position, bout in enumerate(session):
            path = get_bout_path_data(bout, reference_frame=session.first_frame)
            bout_steps.append(_keyed(path[["step", "col", "row"]], example, position))
            tiles = get_tile_seq_data(bout)
            tile_steps.append(_keyed(
                tiles[["step", "in_frame", "out_frame", "tile", "tile_distance"]],
                example, position))
            bout_meta.append({
                "example": np.int64(example),
                "bout_idx": np.int64(position),
                "traverse_idx": np.int64(traverse_indices.index(bout.idx)
                                         if bout.idx in traverse_indices else -1),
                "bout_type": bout.bout_type,
                "duration_s": float(path["duration_s"].iloc[0]),
                "start_frame": np.int64(path["start_frame"].iloc[0]),
                "start_time_s": float(path["start_time_s"].iloc[0]),
                "n_steps": np.int64(path["n_steps"].iloc[0]),
                "is_ho": bool(raster["is_ho"].iloc[position]),
                "is_oh": bool(raster["is_oh"].iloc[position]),
                "cum_duration_s": float(raster["cum_duration_s"].iloc[position]),
                "trajectory_name": path["trajectory_name"].iloc[0],
                "session_idx": np.int64(path["session_idx"].iloc[0]),
            })

    return {
        "bout steps": _concat(bout_steps),
        "tile steps": _concat(tile_steps),
        "bout meta": pd.DataFrame(bout_meta),
        "manifest": get_session_manifest_data(sessions, cache=cache, cohort_rows=cohort_rows),
    }


def derive_bout_path_table(bout_steps, bout_meta_row):
    """
    Rebuild one bout's path frame from the cached primitive tables.

    Produces exactly what :func:`get_bout_path_data` would have returned, so
    :func:`~manhattan_maze.plot_behavior.plot_bout_path` can be driven from
    ``data/figure_data``.  The per-step ``"bout steps"`` table is deliberately narrow, so
    the annotation scalars (duration, bout type, start time) come from the matching
    ``"bout meta"`` row rather than being repeated on every step.

    Parameters
    ----------
    bout_steps : pandas.DataFrame
        The ``"bout steps"`` rows for a single ``(example, bout_idx)``, ordered by
        ``step``.
    bout_meta_row : pandas.Series
        The matching ``"bout meta"`` row.

    Returns
    -------
    pandas.DataFrame
        Frame accepted by :func:`~manhattan_maze.plot_behavior.plot_bout_path`.

    See Also
    --------
    get_bout_path_data : the extractor this reproduces; pinned by regression test.
    """
    return pd.DataFrame({
        "step": bout_steps["step"].to_numpy(dtype=np.int64),
        "col": bout_steps["col"].to_numpy(dtype=np.int64),
        "row": bout_steps["row"].to_numpy(dtype=np.int64),
        "n_steps": np.int64(bout_meta_row["n_steps"]),
        "duration_s": float(bout_meta_row["duration_s"]),
        "bout_type": bout_meta_row["bout_type"],
        "start_frame": np.int64(bout_meta_row["start_frame"]),
        "start_time_s": float(bout_meta_row["start_time_s"]),
        "trajectory_name": bout_meta_row.get("trajectory_name", ""),
        "session_idx": np.int64(bout_meta_row.get("session_idx", -1)),
        "bout_idx": np.int64(bout_meta_row["bout_idx"]),
    })


def iter_example_bout_paths(bout_steps, bout_meta, label_column="label"):
    """
    Iterate the exported example bouts as ``(label, path_df)`` pairs.

    Convenience for the example-traverse strips, which previously iterated a pickled
    ``[(label, Bout), ...]`` list.  Bouts come back in export order, so the panels keep
    their original left-to-right ordering.

    Parameters
    ----------
    bout_steps : pandas.DataFrame
        A family's whole ``"bout steps"`` table.
    bout_meta : pandas.DataFrame
        The matching ``"bout meta"`` table, one row per exported bout.
    label_column : str, default "label"
        Meta column holding each bout's caption key -- ``"label"`` for the bout-keyed
        families (a traverse index, or a *day* for the memory panel), or
        ``"traverse_idx"`` when iterating traverses of a session-keyed family.

    Yields
    ------
    label : int
        Value of ``label_column`` for this bout.
    path_df : pandas.DataFrame
        Frame accepted by :func:`~manhattan_maze.plot_behavior.plot_bout_path`.
    """
    for _, meta_row in bout_meta.sort_values("example").iterrows():
        selector = ((bout_steps["example"] == meta_row["example"])
                    & (bout_steps["bout_idx"] == meta_row["bout_idx"]))
        yield int(meta_row[label_column]), derive_bout_path_table(bout_steps[selector], meta_row)


def derive_tile_distance_table(tile_steps, bout_meta, fps, session_first_frame):
    """
    Rebuild the distance-over-time frame from the cached primitive tables.

    Produces exactly what :func:`get_tile_distance_data` would have returned for the
    same session, so :func:`~manhattan_maze.plot_behavior.plot_tile_distance` can be fed
    from ``data/figure_data`` without caching a third copy of the tile rows.

    Parameters
    ----------
    tile_steps : pandas.DataFrame
        One session's rows of the ``"tile steps"`` table (already filtered to a single
        ``example``), with columns ``bout_idx``, ``step``, ``in_frame``, ``out_frame``,
        ``tile_distance``.
    bout_meta : pandas.DataFrame
        The matching rows of ``"bout meta"``, supplying ``bout_type`` and
        ``traverse_idx`` per ``bout_idx``.
    fps : float
        Frames per second, from the manifest.
    session_first_frame : int
        The session's own first frame, from the manifest.

    Returns
    -------
    pandas.DataFrame
        Frame accepted by :func:`~manhattan_maze.plot_behavior.plot_tile_distance`, with
        sorties carrying the -1 ``traverse_number`` sentinel.

    Notes
    -----
    ``traverse_number`` is ranked over *the rows passed in*, not taken from the
    session-wide ``traverse_idx``.  That reproduces the method, which numbers traverses
    within whatever session it is called on: a segment sliced from bout 91 labels its
    first traverse "1".  Passing a whole session gives ``traverse_idx + 1``, as expected.

    See Also
    --------
    get_tile_distance_data : the extractor this reproduces; a regression test asserts
        the two agree on a live session so they cannot drift.
    """
    types = bout_meta.set_index("bout_idx")
    table = tile_steps[["bout_idx", "step", "in_frame", "out_frame", "tile_distance"]].copy()
    table["bout_type"] = table["bout_idx"].map(types["bout_type"]).to_numpy()
    # number traverses within the supplied rows (see Notes)
    traverse_bouts = bout_meta[bout_meta["traverse_idx"] >= 0].sort_values("bout_idx")["bout_idx"]
    numbering = {bout_idx: rank + 1 for rank, bout_idx in enumerate(traverse_bouts)}
    table["traverse_number"] = table["bout_idx"].map(numbering).fillna(-1).astype(np.int64)
    table["fps"] = float(fps)
    table["session_first_frame"] = np.int64(session_first_frame)
    return table.reset_index(drop=True)


def derive_step_times(tile_steps, fps):
    """
    Rebuild the in-maze step-time point process from the cached tile rows.

    Produces exactly what :func:`get_step_times_data` would have returned, so the speed
    histogram needs no cached copy of its own.  Each bout is re-zeroed to its own first
    frame and the bouts are laid end to end, which is what makes the axis "time in maze"
    rather than wall-clock time.

    Parameters
    ----------
    tile_steps : pandas.DataFrame
        One session's rows of the ``"tile steps"`` table (a single ``example``), sorted
        by ``bout_idx`` then ``step``.
    fps : float
        Frames per second, from the manifest.

    Returns
    -------
    pandas.DataFrame
        Single column ``step_time_s`` : float, in **seconds** of in-maze time.

    See Also
    --------
    get_step_times_data : the extractor this reproduces; pinned by regression test.
    """
    if not len(tile_steps):
        return pd.DataFrame({"step_time_s": np.array([], dtype=float)})

    fps = float(fps)
    step_times, elapsed_frames = [], 0
    for _, bout_rows in tile_steps.groupby("bout_idx", sort=True):
        first_frame = bout_rows["in_frame"].iloc[0]
        out_frames = bout_rows["out_frame"].to_numpy()
        step_times.append((out_frames - first_frame + elapsed_frames) / fps)
        elapsed_frames += bout_rows["out_frame"].iloc[-1] - first_frame

    return pd.DataFrame({"step_time_s": np.concatenate(step_times).astype(float)})


def get_example_bout_tables(labelled_bouts, cache=None):
    """
    Build flat tables for a cache that holds individual bouts rather than sessions.

    Replaces the pickled ``[(label, Bout), ...]`` example-traverse caches.  ``label`` is
    whatever the producer keyed on -- a traverse index for the Mask-A/Mask-D traverse
    panels, a *day number* for the acortical memory panel -- so it is stored in a
    generic ``label`` column rather than something traverse-specific.

    Parameters
    ----------
    labelled_bouts : sequence of (label, Bout) or sequence of Bout
        The bouts to export, in draw order.  A bare sequence of bouts is labelled by
        position.
    cache : str or None, default None
        Figure-data family name recorded in the manifest's ``cache`` column.

    Returns
    -------
    dict of {str: pandas.DataFrame}
        ``"bout steps"``, ``"tile steps"`` and ``"bout meta"``, keyed by ``example``
        (the position in ``labelled_bouts``).  ``bout_idx`` carries the bout's own index
        within its session, and ``manifest`` gives one row per bout with the
        ``(animal_name, session_idx, bout_idx)`` needed to reload it live.

    Notes
    -----
    Unlike :func:`get_example_session_tables` there is nothing to take the superset of:
    the producer already chose these bouts, and the choice is a hardcoded index list in
    the ``gen_*`` script rather than a ``config.py`` selector.
    """
    bout_steps, tile_steps, bout_meta, manifest = [], [], [], []

    for example, item in enumerate(labelled_bouts):
        label, bout = item if isinstance(item, tuple) else (example, item)
        session = getattr(bout, "session", None)
        reference_frame = getattr(session, "first_frame", None)

        path = get_bout_path_data(bout, reference_frame=reference_frame)
        position = int(path["bout_idx"].iloc[0])
        bout_steps.append(_keyed(path[["step", "col", "row"]], example, position))
        tiles = get_tile_seq_data(bout)
        tile_steps.append(_keyed(
            tiles[["step", "in_frame", "out_frame", "tile", "tile_distance"]],
            example, position))
        bout_meta.append({
            "example": np.int64(example),
            "bout_idx": np.int64(position),
            "label": np.int64(label),
            "bout_type": bout.bout_type,
            "duration_s": float(path["duration_s"].iloc[0]),
            "start_frame": np.int64(path["start_frame"].iloc[0]),
            "start_time_s": float(path["start_time_s"].iloc[0]),
            "n_steps": np.int64(path["n_steps"].iloc[0]),
            "fps": float(bout.FPS),
            "trajectory_name": path["trajectory_name"].iloc[0],
            "session_idx": np.int64(path["session_idx"].iloc[0]),
        })
        manifest.append({
            "cache": cache or "",
            "example": np.int64(example),
            "label": np.int64(label),
            "animal_name": path["trajectory_name"].iloc[0],
            "session_idx": np.int64(path["session_idx"].iloc[0]),
            "bout_idx": np.int64(position),
            "fps": float(bout.FPS),
            "mask_name": getattr(bout.mask, "name", None) or "",
        })

    return {
        "bout steps": _concat(bout_steps),
        "tile steps": _concat(tile_steps),
        "bout meta": pd.DataFrame(bout_meta),
        "manifest": pd.DataFrame(manifest),
    }


def _keyed(table, example, bout_idx):
    """
    Prefix a per-bout table with its ``example`` and ``bout_idx`` key columns.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-step table for a single bout.
    example : int
        Positional index of the session (or bout) within the cached family.
    bout_idx : int
        Position of the bout within its session.

    Returns
    -------
    pandas.DataFrame
        A copy with ``example`` and ``bout_idx`` as the first two columns.
    """
    keyed = table.copy()
    keyed.insert(0, "bout_idx", np.int64(bout_idx))
    keyed.insert(0, "example", np.int64(example))
    return keyed


def _concat(tables):
    """
    Concatenate per-bout tables, returning an empty frame rather than failing on none.

    Parameters
    ----------
    tables : list of pandas.DataFrame
        Tables to stack, already carrying their key columns.

    Returns
    -------
    pandas.DataFrame
        The concatenation with a fresh RangeIndex, or an empty frame if ``tables`` is
        empty.
    """
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)
