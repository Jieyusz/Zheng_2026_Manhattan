"""Low-level shared utilities + facade re-export of the split modules.

Coordinate/turn/trajectory-format helpers live here. The curve-fit, bootstrap,
stats, similarity, geometry, graph, analysis, and io functions were split into
their own modules and are re-exported below so that `utils.X` keeps working.

`utils.X` is the intended import surface — keep calling through it rather than
importing the submodules directly. To find where a function lives, use this map:

    analysis    Behavioural metrics, turn-correctness, animal/session selection,
                and figure-data wrangling.
    curve_fit   Learning-curve models, nonlinear fitting, analytic confidence
                intervals, and fit quality.
    geometry    Maze coordinate system: tile/corridor encoding and L1 geometry.
    graph       Graph algorithms on the maze: Floyd-Warshall, shortest paths,
                and Markov-walk models.
    io          Figure-data serialization (npy/parquet/pkl) and file lookup.
    random_walk First-order Markov walker: expected completion time and corridor
                errors (forward-bias family; memoryless walk at beta=1/2).
    similarity  Path/transition similarity metrics (Jaccard, adjusted Jaccard,
                cosine).
    bootstrap   Animal-level bootstrap fitting, bootstrap curve CIs, and
                permutation testing.
    stats       Nonparametric group statistics (Friedman/Kruskal/Wilcoxon/
                Mann-Whitney/Levene) and helpers.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from collections import defaultdict


def df_condense_consecutive_repeats(df, column_name):
    """
    Condense consecutive repeated values in a column into first and last rows.

    Groups runs of identical consecutive values in ``column_name`` and returns
    two DataFrames: one with the first row of each run, one with the last.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with at least ``column_name`` column.
    column_name : str
        Column to detect consecutive repeats on.

    Returns
    -------
    first_row_df : pd.DataFrame
        One row per run — the first occurrence of each consecutive group.
    last_row_df : pd.DataFrame
        One row per run — the last occurrence of each consecutive group.

    Notes
    -----
    Index is reset to group number; original DataFrame index is not preserved.
    """
    first_row_df = df.groupby((df[column_name] != df[column_name].shift()).cumsum().values).first()
    last_row_df = df.groupby((df[column_name] != df[column_name].shift()).cumsum().values).last()
    return first_row_df, last_row_df


def add_turn_at_hole(pre_turn, post_turn, hole_list):
    """
    Identify the bend-point tile when a trajectory turns at a maze hole.

    Given a pre-turn position and a post-turn position, finds which of the two
    possible corner tiles (the L-shaped bend candidates) lies in ``hole_list``.

    Parameters
    ----------
    pre_turn : tuple[int, int]
        (col, row) of the position before the turn.
    post_turn : tuple[int, int]
        (col, row) of the position after the turn.
    hole_list : list of tuple[int, int]
        Maze hole positions, each as (col, row).

    Returns
    -------
    tuple[int, int] or None
        The bend-point tile (col, row) if exactly one candidate corner is in
        ``hole_list``; ``None`` if zero or both candidates qualify.
    """
    # find the bending point of this point
    alternative_1 = (pre_turn[0], post_turn[1])  # x1 y2,
    alternative_2 = (post_turn[0], pre_turn[1])  # x2 y1
    if alternative_1 in hole_list and alternative_2 not in hole_list:
        add_tile = alternative_1
    elif alternative_2 in hole_list and alternative_1 not in hole_list:
        add_tile = alternative_2
    else: # either we don't have any or both are in holes. need double checks
        add_tile = None
    return add_tile


def generate_cell_sequence_df(cell_by_frame_df, min_frame_per_cell, drop_low_likelihood=True,
                              drop_outside_ROIs=True):
    """
    Convert a per-frame cell annotation dataframe into a cell-sequence dataframe.

    Filters out low-likelihood and outside-ROI frames, then condenses consecutive
    identical cell labels into (in_frame, out_frame) rows and removes transitions
    shorter than ``min_frame_per_cell``.

    Parameters
    ----------
    cell_by_frame_df : pd.DataFrame
        Must contain columns ``cell`` (str label) and ``frame`` (int, absolute
        video frame number; FPS=30).
    min_frame_per_cell : int
        Minimum number of frames a cell occupancy must span to be retained.
        Shorter occupancies are dropped before a second de-duplication pass.
    drop_low_likelihood : bool, default True
        If True, remove rows where ``cell`` contains ``"low_likelihood"``.
    drop_outside_ROIs : bool, default True
        If True, remove rows where ``cell`` contains ``"outside_ROIs"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``cell`` (str), ``in_frame`` (int), ``out_frame`` (int).
        Each row is one contiguous cell occupancy episode surviving all filters.
    """

    # It could be that the mice went outside the maze
    if drop_low_likelihood or drop_outside_ROIs:
        # old way of processing the data
        pattern = "low_likelihood" if drop_low_likelihood else ""
        pattern += "|" if drop_low_likelihood and drop_outside_ROIs else ""
        pattern += "outside_ROIs" if drop_outside_ROIs else ""

        o_temp_df = cell_by_frame_df[~cell_by_frame_df.cell.str.contains(pattern)].reset_index(drop=True)
    else:
        o_temp_df = cell_by_frame_df

    # test function on finding in frame and out frame
    in_fr_df, out_fr_df = df_condense_consecutive_repeats(o_temp_df, "cell")

    in_fr_df = in_fr_df.rename(columns={"frame": "in_frame"})
    in_fr_df["out_frame"] = out_fr_df["frame"]

    # drop cells that are shorter than required, and then remove consecutive repeats again
    check_min_fr_df = in_fr_df[in_fr_df.out_frame - in_fr_df.in_frame >= min_frame_per_cell].reset_index(drop=True)
    checked_in_fr_df, checked_out_fr_df = df_condense_consecutive_repeats(check_min_fr_df, "cell")
    cs_df = checked_in_fr_df.drop(columns=["out_frame"])
    cs_df["out_frame"] = checked_out_fr_df["out_frame"]
    return cs_df


def bouts_to_tiles_format(cs_df):
    """
    Convert a cell-sequence bout dataframe to tile dataframe with discrete_loc tuples.

    Parses ``"col-row"`` strings in the ``cell`` column into (col, row) tuples
    stored in ``discrete_loc``.  Returns ``None`` on parse failure.

    Parameters
    ----------
    cs_df : pd.DataFrame
        Cell-sequence dataframe for a single bout.  Must have a ``cell`` column
        with ``"col-row"`` formatted strings.

    Returns
    -------
    pd.DataFrame or None
        Copy of ``cs_df`` with ``cell`` dropped and ``discrete_loc`` column
        added as tuples of int (col, row).  ``None`` if ``cell`` values cannot
        be split into integers.
    """
    tile_df = cs_df.reset_index(drop=True)
    try:
        tile_df["discrete_loc"] = tile_df.cell.apply(
            lambda x: tuple(int(coord) for coord in x.split('-')))  # convert string to discrete locations
    except ValueError:
        print("ignored bout:")
        print(tile_df["discrete_loc"])
        return None
    tile_df = tile_df.drop(columns=["cell"])
    return tile_df


def tiles_to_bouts_format(tile_df):
    """
    Collapse a tile dataframe back to bout (cell-sequence) format.

    Inverse of :func:`bouts_to_tiles_format`.  Reconstructs a ``cell`` column
    from ``x`` and ``y`` columns, de-duplicates consecutive repeated cells, and
    removes the per-tile spatial columns.

    Parameters
    ----------
    tile_df : pd.DataFrame
        Tile dataframe with columns ``x`` (int), ``y`` (int), ``z`` (int),
        ``tile`` (int), ``in_frame`` (int), ``out_frame`` (int).

    Returns
    -------
    pd.DataFrame
        Bout-format dataframe with columns ``in_frame``, ``out_frame``, and
        ``discrete_loc`` (tuple of int).  Spatial columns are dropped.
    """
    bout_df = tile_df.copy()
    bout_df["cell"] = tile_df.apply(lambda t: f"{t.x}-{t.y}", axis=1)
    bout_df["discrete_loc"] = bout_df.cell.apply(
            lambda x: tuple(int(coord) for coord in x.split('-')))
    # remove repeated from previous discrete loc
    first_row, last_row = df_condense_consecutive_repeats(bout_df, "cell")
    new_df = first_row.reset_index(drop=True)
    new_df = new_df.drop(columns=["x", "y", "z", "tile", "cell", "out_frame"])
    new_df["out_frame"] = last_row["out_frame"]
    return new_df


def format_sessions_with_tiles(sessions):
    """
    Apply :func:`bouts_to_tiles_format` across a nested list of sessions and bouts.

    Parameters
    ----------
    sessions : list of list of pd.DataFrame
        Outer list is sessions; inner list is bouts within each session.
        ``None`` bouts are skipped.

    Returns
    -------
    list of list of pd.DataFrame
        Same nested structure, with each bout converted to tile dataframe format.
    """
    return [[bouts_to_tiles_format(bout) for bout in session if bout is not None] for session in sessions]


def is_turn(prev_tile, curr_tile, next_tile):
    """
    Return True if the trajectory prev→curr→next constitutes a direction change.

    A turn occurs when motion along a column switches to motion along a row (or
    vice versa).  Identical consecutive tiles or zero-displacement steps are not
    considered turns.

    Parameters
    ----------
    prev_tile : tuple[int, int]
        (col, row) of the position before the candidate turn.
    curr_tile : tuple[int, int]
        (col, row) of the candidate turn position.
    next_tile : tuple[int, int]
        (col, row) of the position after the candidate turn.

    Returns
    -------
    bool
        True if the direction changes from prev→curr to curr→next.
    """
    if prev_tile == curr_tile or curr_tile == next_tile or prev_tile == next_tile:
        return False

    prev_col, prev_row = prev_tile
    curr_col, curr_row = curr_tile
    next_col, next_row = next_tile

    # moving along a column
    if prev_col == curr_col:
        # next tile is at same row as curr tile
        if curr_row == next_row:
            # check if it moves in any direction along the row
            return curr_col != next_col

    # moving along a row
    if prev_row == curr_row:
        # next tile is at same column as curr tile
        if curr_col == next_col:
            # check if it moves in any direction along the column
            return curr_row != next_row

    return False


def to_egocentric_direction(prev_direction, curr_direction):
    """
    Convert two consecutive allocentric directions to an egocentric turn label.

    Parameters
    ----------
    prev_direction : {'N', 'S', 'E', 'W'}
        Allocentric heading before the turn.
    curr_direction : {'N', 'S', 'E', 'W'}
        Allocentric heading after the turn.

    Returns
    -------
    {'L', 'R', 'B', None}
        Egocentric turn: 'L' = left, 'R' = right, 'B' = 180° back,
        None = straight (no turn).
    """
    turn_table = {
        'N': {'N': None, 'S': 'B', 'E': 'R', 'W': 'L'},
        'S': {'N': 'B', 'S': None, 'E': 'L', 'W': 'R'},
        'E': {'N': 'L', 'S': 'R', 'E': None, 'W': 'B'},
        'W': {'N': 'R', 'S': 'L', 'E': 'B', 'W': None}
    }
    return turn_table[prev_direction][curr_direction]


def get_vector_allocentric_direction(vector):
    """
    Return the cardinal (allocentric) direction of a 2-D displacement vector.

    Parameters
    ----------
    vector : tuple[int, int] or array-like of length 2
        (dx, dy) displacement.  Positive x = East, positive y = North.

    Returns
    -------
    {'N', 'S', 'E', 'W', None}
        Cardinal direction corresponding to the dominant axis of ``vector``.
        Returns None for zero-displacement vectors.
    """
    if vector[0] > 0:
        return 'E'  # East
    elif vector[0] < 0:
        return 'W'  # West
    elif vector[1] > 0:
        return 'N'  # North
    elif vector[1] < 0:
        return 'S'  # South
    else:
        return None  # No movement


def get_turn_direction(prev_tile, curr_tile, next_tile, check_is_turn=True):
    """
    Return the allocentric direction of a turn at a hole.

    Parameters
    ----------
    prev_tile : tuple[int, int]
        (col, row) before the turn.
    curr_tile : tuple[int, int]
        (col, row) at the turn position (hole).
    next_tile : tuple[int, int]
        (col, row) after the turn.
    check_is_turn : bool, default True
        If True, return None immediately when :func:`is_turn` is False.

    Returns
    -------
    {'N', 'S', 'E', 'W', None}
        Allocentric direction of the post-turn heading, or None if no turn
        or if the displacement is zero.
    """
    if check_is_turn and not is_turn(prev_tile, curr_tile, next_tile):
        return None

    curr_col, curr_row = curr_tile
    next_col, next_row = next_tile

    turn_vec = np.array([next_col - curr_col, next_row - curr_row])
    if np.all(turn_vec == 0):
        return None
    direction = get_vector_allocentric_direction(turn_vec)
    if direction is not None:
        return direction
    else:
        # If the direction is not recognized, return None
        print(f"Unrecognized turn from {prev_tile} to {curr_tile} to {next_tile}")
        return None


def make_line_segments(x, y):
    """
    Create line segments from x and y coordinate arrays for use with LineCollection.

    Parameters
    ----------
    x : array-like, shape (n,)
        X coordinates of the polyline vertices.
    y : array-like, shape (n,)
        Y coordinates of the polyline vertices.

    Returns
    -------
    np.ndarray, shape (n-1, 2, 2)
        Array of consecutive segments; each entry is [[x0, y0], [x1, y1]].
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def make_colorline(x, y, z=None, ax=None, cmap=plt.get_cmap('viridis'), norm=plt.Normalize(0.0, 1.0),
                   linewidth=3, alpha=1.0, color=None, **linekwargs):
    """
    Plot a polyline whose segments are colored by a scalar array.

    Parameters
    ----------
    x : array-like, shape (n,)
        X coordinates of the polyline.
    y : array-like, shape (n,)
        Y coordinates of the polyline.
    z : array-like, shape (n,) or scalar, optional
        Color values for each segment.  Defaults to a linear ramp from 0 to 1.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; defaults to the current axes.
    cmap : matplotlib.colors.Colormap, default plt.get_cmap('viridis')
        Colormap applied when ``color`` is not given.
    norm : matplotlib.colors.Normalize, default Normalize(0, 1)
        Normalization for mapping ``z`` values to the colormap.
    linewidth : float, default 3
        Width of the line segments in points.
    alpha : float, default 1.0
        Opacity of the line collection.
    color : color spec or None, default None
        If given, all segments are drawn with this color (overrides ``cmap``).
    **linekwargs
        Additional keyword arguments forwarded to :class:`~matplotlib.collections.LineCollection`.

    Returns
    -------
    matplotlib.collections.LineCollection
        The collection added to the axes.
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical original -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_line_segments(x, y)
    if color is not None:
        lc = LineCollection(segments, colors=[color]*len(x), linewidth=linewidth, alpha=alpha, **linekwargs)

    elif cmap is not None:
        lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha, **linekwargs)

    else:
        raise ValueError("Either color or cmap must be specified")
    lc.set_capstyle('round')

    if ax is None:
        ax = plt.gca()

    ax.add_collection(lc)

    return lc


def interpolate_coords_with_frames(prev_row, curr_row, axis):
    """
    Fill in missing intermediate tile positions between two non-adjacent rows.

    When two consecutive rows in a tile dataframe are separated by more than
    one Manhattan step along one axis, this function generates the intermediate
    tiles and distributes the total frame span evenly among them.  The final
    tile (``curr_row``) is not included; it is handled by the caller.

    Parameters
    ----------
    prev_row : pd.Series
        Row from a tile dataframe; must have ``discrete_loc`` (tuple of ints),
        ``in_frame`` (int, absolute video frame), and ``out_frame`` (int).
    curr_row : pd.Series
        Next row in the tile dataframe; same schema as ``prev_row``.
    axis : {'row', 'col'}
        Axis along which to interpolate.  ``'row'`` steps row index while
        holding column constant; ``'col'`` does the reverse.

    Returns
    -------
    list of dict
        Each dict has ``discrete_loc`` (tuple), ``in_frame`` (int), ``out_frame``
        (int), plus all other fields from ``prev_row``.
    """
    sc, sr = prev_row['discrete_loc']
    ec, er = curr_row['discrete_loc']

    # Total frame span to be redistributed
    f_start = prev_row['in_frame']
    f_end = curr_row['out_frame']

    dist = max(abs(ec - sc), abs(er - sr))
    n_tiles = dist + 1  # Total sequence: Start + Missing + End

    total_frames = f_end - f_start
    frames_per_tile = total_frames // n_tiles
    remainder = total_frames % n_tiles

    # Define all coordinates in the sequence (excluding the final end-point)
    step = 1 if (er > sr if axis == "row" else ec > sc) else -1
    if axis == "row":
        coords = [(sc, r) for r in range(sr, er, step)]
    else:
        coords = [(c, sr) for c in range(sc, ec, step)]

    full_interpolated_sequence = []
    current_cursor = f_start

    for idx, coord in enumerate(coords):
        # Distribute the remainder frames across the first few tiles
        duration = frames_per_tile + (1 if idx < remainder else 0)

        # Use metadata from prev_row
        new_tile = prev_row.to_dict()
        new_tile.update({
            'discrete_loc': coord,
            'in_frame': int(current_cursor),
            'out_frame': int(current_cursor + duration - 1)
        })

        full_interpolated_sequence.append(new_tile)
        current_cursor += duration

    return full_interpolated_sequence


def get_allocentric_turns(tiles_seq, holes, tolerance=0):
    """
    Return a list of (hole, direction) for all turns made at maze holes.

    Iterates over ``tiles_seq`` and records each position in ``holes`` where a
    direction change occurs.  An optional tolerance parameter allows smoothing
    of micro-reversals by looking ahead until the trajectory commits to a
    corridor beyond the specified Manhattan distance.

    Parameters
    ----------
    tiles_seq : list of tuple[int, int]
        Ordered (col, row) tile sequence for a single bout.
    holes : list of tuple[int, int] or set
        Maze hole positions at which turns are scored.
    tolerance : int, default 0
        Minimum Manhattan distance from the hole that the trajectory must
        travel before a direction is declared.  0 disables look-ahead.

    Returns
    -------
    list of tuple[tuple[int, int], str]
        Each element is (hole_position, allocentric_direction) where direction
        is one of 'N', 'S', 'E', 'W'.
    """
    turns = []
    idx = 1
    while idx < len(tiles_seq) - 1:
        prev, curr, next = tiles_seq[idx-1], tiles_seq[idx], tiles_seq[idx+1]
        if curr in holes:
            if is_turn(prev, curr, next):

                if tolerance > 0:
                    future_idx = idx + 1
                    while True:
                        future = tiles_seq[future_idx]
                        if same_corridor([curr, next, future]):
                            if manhattan_dist(curr, future) > tolerance:
                                turn_direction = get_turn_direction(prev, curr, future, check_is_turn=False)
                                turns.append((curr, turn_direction))
                                break
                        else:
                            if same_corridor([curr, prev, future]):
                                idx = future_idx - 1
                            else:
                                turn_direction = get_turn_direction(prev, curr, future, check_is_turn=False)
                                turns.append((curr, turn_direction))
                            break
                        future_idx += 1

                else:
                    turn_direction = get_turn_direction(prev, curr, next)
                    turns.append((curr, turn_direction))
        idx += 1
    return turns


def get_hole_decisions(tiles_seq, holes):
    """
    Return ``(hole, direction)`` for the decision made at EVERY hole crossing.

    Unlike :func:`get_allocentric_turns`, which records a hole only where the
    trajectory *changes* direction (a turn), this records the chosen heading at
    every crossing of a hole — including straight-through passes. It is the mouse's
    decision (allocentric action) at each hole, the signal a turn-based RL agent
    trains on.

    Consecutive identical tiles are collapsed first; the direction is the heading
    from the hole toward the next distinct tile
    (:func:`get_vector_allocentric_direction`). Endpoints of ``tiles_seq`` are not
    scored (a hole must have both a predecessor and a successor), so pad the
    sequence with virtual home/goal tiles exactly as
    :meth:`trajectory.Bout.get_allocentric_turns` does to score the first/last
    holes.

    Parameters
    ----------
    tiles_seq : list of tuple[int, int]
        Ordered ``(col, row)`` tile sequence for a single bout.
    holes : list of tuple[int, int] or set
        Maze hole positions at which decisions are scored.

    Returns
    -------
    list of tuple[tuple[int, int], str]
        Each element is ``(hole_position, direction)`` with direction in
        ``'N'``/``'S'``/``'E'``/``'W'``, one entry per hole crossing in order.
    """
    hole_set = set(holes)
    # collapse consecutive duplicate tiles so a "next distinct tile" is well defined
    path = []
    for tile in tiles_seq:
        tile = (int(tile[0]), int(tile[1]))
        if not path or tile != path[-1]:
            path.append(tile)

    decisions = []
    for idx in range(1, len(path) - 1):
        curr = path[idx]
        if curr in hole_set:
            nxt = path[idx + 1]
            direction = get_vector_allocentric_direction((nxt[0] - curr[0], nxt[1] - curr[1]))
            if direction is not None:
                decisions.append((curr, direction))
    return decisions


def turn_axis(direction):
    """
    Corridor axis of an allocentric direction.

    Parameters
    ----------
    direction : {'N', 'S', 'E', 'W'}
        Allocentric heading.

    Returns
    -------
    {'H', 'V', None}
        ``'H'`` for east/west (horizontal corridor), ``'V'`` for north/south
        (vertical corridor), ``None`` otherwise.
    """
    if direction in ('E', 'W'):
        return 'H'
    if direction in ('N', 'S'):
        return 'V'
    return None


def allocentric_turns_with_approach(tiles_seq, holes, tolerance=0):
    """
    Like :func:`get_allocentric_turns`, but also records the approach heading.

    For each hole crossing (direction change at a hole) this returns the
    incoming heading in addition to the outgoing one, so callers can condition
    on the corridor the mouse entered from. This is exactly the information
    :func:`get_allocentric_turns` discards.

    Parameters
    ----------
    tiles_seq : list of tuple[int, int]
        Ordered ``(col, row)`` tile sequence for a single bout.
    holes : list of tuple[int, int] or set
        Maze hole positions at which turns are scored.
    tolerance : int, default 0
        Only ``0`` is supported (the look-ahead used by the turn-error metric).

    Returns
    -------
    list of tuple[tuple[int, int], str, str]
        Each element is ``(hole, approach_dir, exit_dir)``; ``approach_dir`` is
        the heading of the segment into the hole and ``exit_dir`` the heading
        out of it, each one of ``'N'``/``'S'``/``'E'``/``'W'``.

    Raises
    ------
    NotImplementedError
        If ``tolerance != 0``.
    """
    if tolerance != 0:
        raise NotImplementedError("allocentric_turns_with_approach supports tolerance=0 only")

    holes = set(holes) if not isinstance(holes, set) else holes
    turns = []
    for idx in range(1, len(tiles_seq) - 1):
        prev, curr, nxt = tiles_seq[idx - 1], tiles_seq[idx], tiles_seq[idx + 1]
        if curr in holes and is_turn(prev, curr, nxt):
            approach = get_vector_allocentric_direction((curr[0] - prev[0], curr[1] - prev[1]))
            exit_dir = get_turn_direction(prev, curr, nxt, check_is_turn=False)
            turns.append((curr, approach, exit_dir))
    return turns


def opposite_direction(direction):
    """
    Return the opposite cardinal direction.

    Parameters
    ----------
    direction : {'N', 'S', 'E', 'W'}
        Input allocentric direction.

    Returns
    -------
    {'N', 'S', 'E', 'W', None}
        Opposite direction, or None for unrecognized input.
    """
    if direction == 'N':
        return 'S'
    elif direction == 'S':
        return 'N'
    elif direction == 'E':
        return 'W'
    elif direction == 'W':
        return 'E'
    else:
        return None


def moving_average(x, window_size, mode="valid"):
    """
    Compute a moving average of ``x`` using numpy convolution.

    Parameters
    ----------
    x : array-like
        Input 1-D signal.
    window_size : int
        Number of samples in the averaging window.
    mode : {'full', 'valid', 'same'}, default 'valid'
        Convolution mode passed to :func:`numpy.convolve`.
        'valid' returns only the positions whose whole window fits inside ``x``, so the
        output has length ``len(x) - window_size + 1``; 'same' returns output the same
        length as ``x``, with the windows at either end truncated.

    Returns
    -------
    np.ndarray
        Smoothed signal.  Length depends on ``mode``.

    Notes
    -----
    NaN-robust normalized convolution: NaNs are excluded from each window and every
    position is divided by the count of valid samples actually in its window (not by the
    full ``window_size``).  This both ignores isolated NaNs and avoids the edge dampening
    that fixed-divisor smoothing produces near the start/end.  Positions whose window has
    no valid sample are NaN.

    ``mode='valid'`` is the default because a truncated end window rests on as few as half
    the samples of an interior one, which makes the first and last points swing enough to
    read as a spurious hook.  **Callers must align x themselves**: output position ``j``
    corresponds to input position ``j + (window_size - 1) // 2``, so plot against
    ``np.arange(len(out)) + (window_size - 1) // 2`` (see
    :func:`~manhattan_maze.plot_behavior.plot_individual_memory`, which draws its width-5
    average against ``xs[2:-2]``).  Pass ``mode='same'`` when equal input/output length
    matters more than dropping under-supported ends.
    """
    x = np.asarray(x, dtype=float)
    kernel = np.ones(window_size)
    valid = ~np.isnan(x)
    weighted = np.convolve(np.where(valid, x, 0.0), kernel, mode=mode)
    counts = np.convolve(valid.astype(float), kernel, mode=mode)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, weighted / counts, np.nan)


def drop_unnamed_column(df):
    """
    Drop the ``Unnamed: 0`` column from a DataFrame if it is present.

    This spurious column is commonly introduced when a CSV is written without
    suppressing the index (e.g., via LibreOffice Calc or :func:`pandas.DataFrame.to_csv`
    without ``index=False``).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, possibly containing an ``Unnamed: 0`` column.

    Returns
    -------
    pd.DataFrame
        Dataframe with ``Unnamed: 0`` removed, or unchanged if not present.
    """
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df


def split_dataframe(df, column, anchors):
    """
    Split a DataFrame into sub-DataFrames at rows matching anchor values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column to scan for anchor values.
    anchors : list
        Values in ``column`` that mark split points.  Each anchor row begins a
        new sub-dataframe; the row itself is included in the sub-dataframe that
        starts at that anchor.

    Returns
    -------
    list of pd.DataFrame
        Sub-DataFrames between consecutive anchor positions.  Empty slices are
        dropped.  Each sub-DataFrame has a reset index.
    """
    # Find indices where column value is in the anchors list
    split_indices = df.index[df[column].isin(anchors)].tolist()

    # Add start and end indices
    split_indices = [0] + split_indices + [len(df)]

    # Create sub-dataframes
    # issues here: the bouts don't start and end at the same locations, and the traverse function falls apart
    dataframes = [df.iloc[split_indices[i]: split_indices[i + 1]].reset_index(drop=True)
                  for i in range(len(split_indices) - 1)]

    # Remove empty DataFrames (e.g., when an anchor appears at the beginning)
    dataframes = [df_part for df_part in dataframes if not df_part.empty]

    return dataframes


def extract_array(data_list, size, align_tail=False):
    """
    Stack a list of 1-D arrays into a 2-D NaN-padded array.

    Parameters
    ----------
    data_list : list of array-like
        Each element is a 1-D sequence.  May be shorter than ``size``.
    size : int
        Length of each row in the output array.  Values beyond an element's
        length are filled with NaN.
    align_tail : bool, default False
        If True, align each sequence to the right (tail) of its row;
        leading entries are NaN.  If False, align to the left (head).

    Returns
    -------
    np.ndarray, shape (len(data_list), size)
        2-D float array with each input sequence in one row.
    """
    data_array = np.full((len(data_list), size), np.nan)
    for i, df in enumerate(data_list):
        # use the shortest length to avoid data error
        length = min(len(df), size)
        if align_tail:
            data_array[i, -length:] = df[-length:]
        else:
            data_array[i, :length] = df[:length]
    return data_array


def map_array_based_on_ref(ref, original):
    """
    Map array values to their sorted positions in a reference array.

    Finds where each element of ``original`` falls in the sorted ordering of
    ``ref``, and returns the original (unsorted) indices of those positions.

    Parameters
    ----------
    ref : array-like
        Reference array whose sort order defines the mapping.
    original : array-like
        Values to map; each must appear in ``ref``.

    Returns
    -------
    np.ndarray
        Array of indices into ``ref`` corresponding to ``original`` values,
        ordered by sorted position in ``ref``.
    """
    ref = np.array(ref)
    sort_idx = np.argsort(ref)
    sorted_ref = ref[sort_idx]
    # Find where each original point fits in sorted_ref
    positions = np.searchsorted(sorted_ref, original)

    # Map to original index
    mapped = sort_idx[positions]
    return mapped

def condense_by_temporal(sequence):
    """
    Given a list of (location, turn) tuples, group consecutive turns
    that occur at the same location. This preserve the order of turns; and a location will appear multiple time

    Parameters:
        sequence (list of tuple): List of (location, turn) pairs.

    Returns:
        list of tuple: Condensed list of (location, [turns]) blocks.
    """
    condensed = []
    if not sequence:
        return condensed

    current_loc = sequence[0][0]
    current_turns = []

    for loc, turn in sequence:
        if loc == current_loc:
            current_turns.append(turn)
        else:
            condensed.append((current_loc, current_turns))
            current_loc = loc
            current_turns = [turn]

    # Append the last group
    condensed.append((current_loc, current_turns))
    return condensed


def condense_by_location(sequence):
    """
    Group all (location, turn) pairs by location using a defaultdict.

    Unlike :func:`condense_by_temporal`, each location appears only once and all
    turns at that location are aggregated regardless of visit order.

    Parameters
    ----------
    sequence : list of tuple[any, any]
        Ordered list of (location, turn) pairs.

    Returns
    -------
    list of tuple[any, list]
        List of (location, [turn, …]) tuples, one per unique location,
        in insertion order.
    """
    condensed = defaultdict(list)
    for hole, turn in sequence:
        condensed[hole].append(turn)
    # convert to list of tuples
    condensed_list = [(hole, turns) for hole, turns in condensed.items()]
    return condensed_list


def look_up_segment_indices(seq, start, end=None):
    """
    Find (start_idx, end_idx) index pairs for segments in a (possibly nested) sequence.

    Parameters
    ----------
    seq : list
        Sequence of elements, which may be scalars or nested lists/tuples/arrays.
    start : any
        Element marking the beginning of each segment.
    end : any or None, default None
        Element marking the end of each segment.  Defaults to ``start``.

    Returns
    -------
    list of tuple[int, int]
        Each tuple is (start_idx, end_idx) where end_idx is the index of the
        first occurrence of ``end`` after each ``start``.  Returns an empty
        list if ``start`` or ``end`` is not found.
    """
    if end is None:
        end = start

    # Handle nested elements robustly (convert each element to tuple for comparison)
    def normalize(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return tuple(np.asarray(x).ravel())
        return x

    seq_norm = [normalize(x) for x in seq]
    start_norm = normalize(start)
    end_norm = normalize(end)

    # Find matching indices
    start_indices = [i for i, x in enumerate(seq_norm) if x == start_norm]
    end_indices = [i for i, x in enumerate(seq_norm) if x == end_norm]

    if len(start_indices) == 0 or len(end_indices) == 0:
        return []

    segment_indices = []
    for s in start_indices:
        # Find the first end index after the start
        following_end = [e for e in end_indices if e > s]
        if following_end:
            segment_indices.append((s, following_end[0]))
        else:
            segment_indices.append((s, s))
    return segment_indices


def to_traverse_number(traverse_idx):
    """
    Convert a 0-based internal traverse index to the 1-based manuscript number (C8).

    Parameters
    ----------
    traverse_idx : int or array-like
        0-based traverse index used for internal array indexing.

    Returns
    -------
    int or numpy.ndarray
        1-based ``traverse_number`` for figure axis labels and panel titles.

    Notes
    -----
    Indexing convention (see ``docs/data_contracts.md`` §"Indexing conventions"):
    ``traverse_idx`` is 0-based; ``traverse_number`` is 1-based, i.e. traverse 1 is
    the first completed traverse. This is the single helper figure scripts use for
    traverse-axis labels so the +1 is not scattered across plot code.
    """
    return traverse_idx + 1


# --- facade: re-export the split modules so existing `utils.X` calls keep working ---
from manhattan_maze.analysis import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.curve_fit import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.geometry import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.graph import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.random_walk import *  # noqa: F401,F403  (facade re-export; imports graph, keep after it)
from manhattan_maze.io import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.plot_data import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.similarity import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.bootstrap import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.stats import *  # noqa: F401,F403  (facade re-export)
