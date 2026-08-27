"""Maze coordinate system: tile/corridor encoding and L1 geometry.

Split out of utils.py; see docs.
"""
import numpy as np

__all__ = ['xyz_to_ti', 'ti_x', 'ti_y', 'ti_z', 'z_xy', 'ti_to_xyz', 'ti_to_ci', 'xyz_to_ci', 'same_col', 'same_row', 'is_diagonal', 'same_corridor', 'manhattan_dist']

def xyz_to_ti(xyz, maze_size=11):
    """
    Convert (x, y, z) maze coordinates to a tile index.

    Parameters
    ----------
    xyz : array-like, shape (3,)
        Maze coordinates ``[x, y, z]`` where x ∈ [0, maze_size−1] (column,
        East), y ∈ [0, maze_size−1] (row, North), z ∈ {0, 1} (floor).
    maze_size : int, default 11
        Grid dimension of one floor.

    Returns
    -------
    int
        Tile index = x + y·maze_size + z·maze_size².
        Range [0, 2·maze_size²−1].  Home tile (0,5,0)=55; out tile (5,9,1)=225.
    """
    return np.array(xyz) @ [1, maze_size, maze_size ** 2]


def ti_x(ti, maze_size=11):
    """
    Return the x (column) coordinate of tile index *ti*.

    Parameters
    ----------
    ti : int or array-like
        Tile index [tile index 0–2·maze_size²−1].
    maze_size : int, default 11

    Returns
    -------
    int or ndarray
        Column index x ∈ [0, maze_size−1].
    """
    return ti % maze_size


def ti_y(ti, maze_size=11):
    """
    Return the y (row) coordinate of tile index *ti*.

    Parameters
    ----------
    ti : int or array-like
        Tile index [tile index 0–2·maze_size²−1].
    maze_size : int, default 11

    Returns
    -------
    int or ndarray
        Row index y ∈ [0, maze_size−1].
    """
    return (ti % (maze_size ** 2)) // maze_size


def ti_z(ti, maze_size=11):
    """
    Return the z (floor) of tile index *ti*.

    Parameters
    ----------
    ti : int or array-like
        Tile index [tile index 0–2·maze_size²−1].
    maze_size : int, default 11

    Returns
    -------
    int or ndarray
        Floor index: 0 = bottom (horizontal corridors), 1 = top (vertical corridors).
    """
    return ti // (maze_size ** 2)


def z_xy(xy1, xy2):
    """
    Infer the maze floor from two consecutive (x, y) positions in the same corridor.

    Parameters
    ----------
    xy1 : tuple[int, int]
        First position (col, row).
    xy2 : tuple[int, int]
        Second position (col, row).

    Returns
    -------
    int
        1 if both positions share the same x (vertical corridor, floor 1);
        0 if both share the same y (horizontal corridor, floor 0).
    """
    return 1 * (xy1[0] == xy2[0])  # top floor if same x coordinate


def ti_to_xyz(ti, maze_size=11):
    """
    Convert a tile index to (x, y, z) maze coordinates.

    Parameters
    ----------
    ti : int or array-like
        Tile index or array of tile indices [tile index 0–2·maze_size²−1].
    maze_size : int, default 11

    Returns
    -------
    np.ndarray, shape (..., 3)
        Columns are [x, y, z].  Inverse of :func:`xyz_to_ti`.
    """
    xyz = np.array([ti_x(ti, maze_size), ti_y(ti, maze_size), ti_z(ti, maze_size)]).T
    return xyz


def ti_to_ci(ti, maze_size=11):
    """
    Convert a tile index to its corridor index.

    Parameters
    ----------
    ti : int or array-like
        Tile index [tile index 0–2·maze_size²−1].
    maze_size : int, default 11

    Returns
    -------
    np.ndarray
        Corridor index array.  See :func:`xyz_to_ci` for encoding.
    """
    xyz = ti_to_xyz(ti, maze_size=maze_size)
    return xyz_to_ci(xyz, maze_size=maze_size)


def xyz_to_ci(xyz, maze_size=11):
    """
    Convert (x, y, z) maze coordinates to corridor indices.

    Parameters
    ----------
    xyz : array-like, shape (3,) or (n, 3)
        Maze coordinates.  Each row is ``[x, y, z]``.
    maze_size : int, default 11

    Returns
    -------
    np.ndarray, shape (n,)
        Corridor indices.  Encoding:
        - z=0 (horizontal corridors): corridor = y  [0–10]
        - z=1 (vertical corridors):   corridor = x + maze_size  [11–21]

    Notes
    -----
    Only supports two-floor mazes (z ∈ {0, 1}).
    """
    xyz = np.array(xyz)
    if len(xyz.shape) == 1:
        xyz = xyz[np.newaxis, :]

    return np.apply_along_axis(lambda x: x[1]*(1-x[2]) + (x[0]+maze_size)*(x[2]), axis=1, arr=xyz)


def same_col(coords):
    """
    Return True if all (col, row) coordinates share the same column.

    Parameters
    ----------
    coords : list of tuple[int, int]
        Sequence of (col, row) positions.

    Returns
    -------
    bool
        True if every coordinate has the same column value as the first.
    """
    first_col, _ = coords[0]

    all_same_col = True
    for col, row in coords:
        if col != first_col:
            all_same_col = False
    return all_same_col


def same_row(coords):
    """
    Return True if all (col, row) coordinates share the same row.

    Parameters
    ----------
    coords : list of tuple[int, int]
        Sequence of (col, row) positions.

    Returns
    -------
    bool
        True if every coordinate has the same row value as the first.
    """
    _, first_row = coords[0]

    all_same_row = True
    for col, row in coords:
        if row != first_row:
            all_same_row = False
    return all_same_row


def is_diagonal(pre, curr):
    """
    Return True if the step from pre to curr is diagonal (both column and row change).

    Parameters
    ----------
    pre : tuple[int, int]
        (col, row) of the prior position.
    curr : tuple[int, int]
        (col, row) of the current position.

    Returns
    -------
    bool
        True if neither column nor row is shared between ``pre`` and ``curr``.
    """
    all_same_col = same_col([pre, curr])
    all_same_row = same_row([pre, curr])
    return not all_same_col and not all_same_row


def same_corridor(coords):
    """
    Return True if all coordinates lie in the same maze corridor.

    A set of positions is in the same corridor if they all share the same column
    (vertical corridor) or all share the same row (horizontal corridor).

    Parameters
    ----------
    coords : list of tuple[int, int]
        Sequence of (col, row) positions.

    Returns
    -------
    bool
        True if :func:`same_col` or :func:`same_row` holds for ``coords``.
    """
    all_same_col = same_col(coords)
    all_same_row = same_row(coords)

    return all_same_col or all_same_row


def manhattan_dist(coords1, coords2):
    """
    Compute Manhattan (L1) distance between two (col, row) maze positions.

    Parameters
    ----------
    coords1 : tuple[int, int]
        First position (col, row).
    coords2 : tuple[int, int]
        Second position (col, row).

    Returns
    -------
    int
        |col1 − col2| + |row1 − row2|, in tile units.
    """
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])
