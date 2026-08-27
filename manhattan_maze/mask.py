"""
Maze mask geometry: hole positions, tile/corridor adjacency graphs, and shortest paths.

A ``Mask`` encodes one experimental configuration of the Manhattan Maze by specifying which
floor-holes are open (i.e., which stairways connect the two floors).  All graph-theoretic
quantities (tile adjacency, corridor adjacency, Floyd–Warshall distances, shortest paths) are
computed at construction time and stored as attributes.

``MaskDSpecial`` is a ``Mask`` subclass used for Mask D sessions.  Mask D has a biclique
bottleneck topology in which three transitions are topologically mandatory on the shortest
path; this requires a correction of 3 in the adjusted Jaccard similarity (R15 / C10).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from copy import deepcopy

from manhattan_maze import utils, plot_utils


class Mask:
    """
    Maze mask: hole coordinates, tile/corridor graphs, and shortest-path geometry.

    A mask defines one experimental layout of the two-floor 11×11 Manhattan Maze by listing
    the (x, y) positions of floor holes (stairways between floors).  All graph-theoretic
    attributes are computed at construction and are read-only after that.

    Parameters
    ----------
    holes_coords : str or array-like of shape (n_holes, 2)
        Path to a ``.npy`` file of hole (x, y) coordinates, or an array / list of
        ``[x, y]`` pairs directly.  Coordinates are 0-based integer column (x) and
        row (y) indices within one floor.
    size : int
        Grid dimension of one floor (e.g. 11 for the standard 11×11 maze).  Tile index
        formula: ``tile = x + y * size + z * size**2`` where z ∈ {0, 1}.
    name : str
        Single-letter mask identifier (``'A'``, ``'B'``, ``'D'``, …).
    home_coordinates : tuple of (x, y, z)
        (x, y, z) position of the home port.  Standard value: ``(0, 5, 0)``.
    out_coordinates : tuple of (x, y, z)
        (x, y, z) position of the out port.  Standard value: ``(5, 9, 1)``.

    Attributes
    ----------
    n_guaranteed_transitions_for_adjusted_jaccard : int
        Number of topologically mandatory transitions subtracted from both numerator and
        denominator of the Jaccard similarity (R15 / C10).  ``0`` for all non-D masks
        (standard Jaccard).  Overridden to ``3`` in ``MaskDSpecial``.
    home_tile : int
        Tile index of the home port [0–241 for size=11].
    out_tile : int
        Tile index of the out port [0–241 for size=11].
    home_corridor : int
        Corridor index of the home port [0–21 for size=11].
    out_corridor : int
        Corridor index of the out port [0–21 for size=11].
    tiles_adj_mat : ndarray of shape (2*size**2, 2*size**2)
        Undirected adjacency matrix over all tiles on both floors.
    tiles_shortest_distances : ndarray of shape (2*size**2, 2*size**2)
        All-pairs shortest distances via Floyd–Warshall (directed).
    tiles_shortest_path : list of int
        Tile-level shortest path from ``home_tile`` to ``out_tile``.
    corridors_adj_mat : ndarray of shape (2*size, 2*size)
        Undirected adjacency matrix over all corridors (rows 0–size−1; columns size–2*size−1).
    corridors_shortest_path : list of int
        Corridor-level shortest path from ``home_corridor`` to ``out_corridor``.

    Notes
    -----
    Corridor index convention (size=11):
    - Corridors 0–10: horizontal (floor z=0), indexed by row ``y``.
    - Corridors 11–21: vertical (floor z=1), indexed by column ``x + 11``.

    Tile index convention: ``tile = x + y * size + z * size**2``.
    """

    n_guaranteed_transitions_for_adjusted_jaccard = 0

    def __init__(self, holes_coords, size, name, home_coordinates, out_coordinates):
        self.name = name
        self.size = size

        if isinstance(holes_coords, str):
            self.holes_coords = np.load(holes_coords)
        else:
            self.holes_coords = np.array(holes_coords)

        self.holes_list = list(tuple(hole) for hole in self.holes_coords)
        self.home_coordinates = home_coordinates
        self.home_pos = (home_coordinates[0], home_coordinates[1])
        self.home_tile = utils.xyz_to_ti(home_coordinates, maze_size=size)
        self.out_coordinates = out_coordinates
        self.out_pos = (out_coordinates[0], out_coordinates[1])
        self.out_tile = utils.xyz_to_ti(out_coordinates, maze_size=size)
        self.home_corridor = int(utils.xyz_to_ci(home_coordinates, maze_size=size)[0])
        self.out_corridor = int(utils.xyz_to_ci(out_coordinates, maze_size=size)[0])

        self.tiles_adj_mat = self._get_tiles_adj_mat()
        self.tiles_indices = self._get_tiles_indices()
        self.tiles_shortest_distances = utils.floyd_warshall(self.tiles_adj_mat)
        self.tiles_shortest_path = utils.find_shortest_path(
            self.tiles_adj_mat, self.home_tile, self.out_tile
        )

        self.corridors_adj_mat = self._get_corridors_adj_mat()
        self.corridors_shortest_distance = utils.floyd_warshall(self.corridors_adj_mat)
        self.corridors_shortest_path = utils.find_shortest_path(
            self.corridors_adj_mat, self.home_corridor, self.out_corridor
        )

    def is_hole(self, col, row=None):
        """
        Return True if (col, row) is an open floor-hole in this mask.

        Parameters
        ----------
        col : int or tuple
            Column index (x), or a ``(col, row)`` tuple if ``row`` is None.
        row : int, optional
            Row index (y).  If None, ``col`` is interpreted as a ``(col, row)`` pair.

        Returns
        -------
        bool
            True when the coordinate is in ``self.holes_list``.
        """
        if row is None:
            col, row = col
        return (col, row) in self.holes_list

    def get_holes(self):
        """
        Return the list of open floor-hole coordinates for this mask.

        Returns
        -------
        list of tuple
            Each element is an ``(x, y)`` integer pair.
        """
        return list(self.holes_list)

    def get_correct_turns(self, homebound=False, home=None, goal=None):
        """
        Return the allocentrically correct turn direction at each hole on the shortest path.

        The turn direction is the cardinal direction (``'N'``, ``'S'``, ``'E'``, ``'W'``) a
        mouse must take at each hole position to follow the shortest-path trajectory from home
        to out (outbound) or out to home (homebound).

        Parameters
        ----------
        homebound : bool, optional
            If True, compute the homebound (out → home) correct turns.
            If False (default), compute the outbound (home → out) correct turns.
        home : tuple of (x, y), optional
            Override home position.  Defaults to ``self.home_pos``.
        goal : tuple of (x, y), optional
            Override goal position.  Defaults to ``self.out_pos``.

        Returns
        -------
        dict
            Mapping ``{(x, y): direction}`` for each hole.  ``direction`` is one of
            ``'N'``, ``'S'``, ``'E'``, ``'W'``.

        Notes
        -----
        This is the ground truth for turn-error counting.  See also
        ``utils.get_allocentric_turns`` and :meth:`correct_approach_map` (which
        additionally records the correct approach corridor at each hole).
        """
        if home is None:
            home = self.home_pos
        if goal is None:
            goal = self.out_pos
        xy_seq = [home] + self.get_holes() + [goal]
        if homebound:
            xy_seq.reverse()
        correct_turns_seq = utils.get_allocentric_turns(xy_seq, self.get_holes())
        return {hole: direction for hole, direction in correct_turns_seq}

    def correct_approach_map(self, homebound=False, home=None, goal=None):
        """
        Ground-truth approach and exit direction at each shortest-path turn-hole.

        Like :meth:`get_correct_turns`, but each hole maps to the ``(approach,
        exit)`` heading pair the shortest path takes through it, where
        ``approach`` is the heading into the hole and ``exit`` the correct
        outgoing heading. The keys and exit directions match
        :meth:`get_correct_turns` exactly; the extra ``approach`` entry is what
        the approach-conditioned turn-error metric keys on (a crossing is only
        scored when the mouse enters on the same corridor axis).

        Parameters
        ----------
        homebound : bool, optional
            If True, compute the homebound (out -> home) map; if False (default),
            the outbound (home -> out) map.
        home, goal : tuple of (x, y), optional
            Override the home/out positions. Default to ``self.home_pos`` /
            ``self.out_pos``.

        Returns
        -------
        dict
            Mapping ``{(x, y): (approach, exit)}``; each heading is one of
            ``'N'``/``'S'``/``'E'``/``'W'``.
        """
        if home is None:
            home = self.home_pos
        if goal is None:
            goal = self.out_pos
        xy_seq = [home] + self.get_holes() + [goal]
        if homebound:
            xy_seq.reverse()
        seq = utils.allocentric_turns_with_approach(xy_seq, self.get_holes())
        return {hole: (approach, exit_dir) for hole, approach, exit_dir in seq}

    def remove_outskirts(self):
        """
        Return a reduced mask with the outermost row/column of corridors removed.

        Shrinks the maze from ``size × size`` to ``(size-2) × (size-2)`` by shifting all
        hole coordinates by ``-1`` and adjusting the home/out port coordinates accordingly.
        Used for graph-theoretic analyses where boundary corridors are excluded.

        Returns
        -------
        Mask
            New ``Mask`` of size ``self.size - 2`` named ``'<original>_reduced'``.

        Notes
        -----
        The home port x-coordinate becomes 0 and out port x-coordinate becomes ``new_size-1``.
        Row indices are shifted down by 1.  Holes that would fall outside the reduced grid are
        kept (the caller is responsible for filtering if needed).
        """
        old_mask = deepcopy(self)
        new_size = old_mask.size - 2
        new_home_coords = (
            0,
            old_mask.home_coordinates[1] - 1,
            self.home_coordinates[-1],
        )
        new_out_coords = (
            new_size - 1,
            old_mask.out_coordinates[1] - 1,
            self.out_coordinates[-1],
        )
        new_holes = old_mask.holes_coords - 1
        return Mask(new_holes, new_size, f"{old_mask.name}_reduced", new_home_coords, new_out_coords)

    def plot(self, ax=None, origin=None, linewidth=0.5, color="grey", **kwargs):
        """
        Draw the maze grid and open holes on a matplotlib axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw on.  A new figure is created if None.
        origin : list of [x, y], optional
            Bottom-left corner of the plot in data coordinates.  Defaults to ``[0, 0]``.
        linewidth : float, optional
            Width of grid lines.  Default 0.5.
        color : str, optional
            Color for grid lines and hole circles.  Default ``'grey'``.
        **kwargs
            Additional keyword arguments forwarded to ``ax.plot``.
        """
        if origin is None:
            origin = [0, 0]
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(origin[0], self.size)
        ax.set_ylim(origin[1], self.size)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for row in range(self.size):
            ax.plot(
                [origin[0], self.size + origin[0]],
                [row + origin[1], row + origin[1]],
                color=color, linewidth=linewidth, **kwargs,
            )

        # Top boundary: gap for the out port
        ax.plot([origin[0], self.size // 2 + origin[0]],
                [self.size + origin[1], self.size + origin[1]],
                color=color, linewidth=linewidth, **kwargs)
        ax.plot([self.size // 2 + 1 + origin[0], self.size + origin[0]],
                [self.size + origin[1], self.size + origin[1]],
                color=color, linewidth=linewidth, **kwargs)

        # Left boundary: gap for the home port
        ax.plot(origin, [origin[1], self.size // 2 + origin[1]],
                color=color, linewidth=linewidth, **kwargs)
        ax.plot(origin, [self.size // 2 + 1 + origin[1], self.size + origin[1]],
                color=color, linewidth=linewidth, **kwargs)

        for col in range(1, self.size + 1):
            ax.plot([col + origin[0], col + origin[0]],
                    [0 + origin[1], self.size + origin[1]],
                    color=color, linewidth=linewidth, **kwargs)

        for col, row in self.get_holes():
            ax.add_artist(plt.Circle(
                xy=(col + 0.5 + origin[0], row + 0.5 + origin[1]),
                radius=0.35, color=color, fill=False, linewidth=linewidth,
            ))

    def plot_with_shortest_path(self, ax=None, zorder=5, path_linewidth=2, holes_list=None,
                                home_xy=None, out_xy=None, maskd_bottleneck=False, plot_ho=False):
        """
        Draw the maze with the shortest-path trajectory overlaid as a colour-coded line.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw on.  A new figure is created if None.
        zorder : int, optional
            Drawing order for the path line.  Default 5.
        path_linewidth : float, optional
            Width of the path line.  Default 2.
        holes_list : list of [x, y], optional
            Explicit waypoint sequence.  If None, uses ``self.holes_list`` (or a Mask D
            hardcoded sequence when ``self.name == 'D'``).
        home_xy : list of [x, y], optional
            Starting point(s) prepended to the path.  Defaults to ``[[home_x - 1, home_y]]``.
        out_xy : list of [x, y], optional
            Ending point(s) appended to the path.  Defaults to ``[[out_x, out_y + 0.5]]``.
        maskd_bottleneck : bool, optional
            If True, draw horizontal and vertical red lines marking the Mask D bottleneck.
        plot_ho : bool, optional
            If True, annotate home (``'H'``) and out (``'O'``) ports with text labels.

        Returns
        -------
        ax : matplotlib.axes.Axes
        lc : matplotlib.collections.LineCollection
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(3, 3))
        if home_xy is None:
            home_xy = [[self.home_pos[0] - 1, self.home_pos[1]]]
        if out_xy is None:
            out_xy = [[self.out_pos[0], self.out_pos[1] + 0.5]]
        self.plot(ax)
        ax.set_title(
            f"Mask {self.name}",
            fontsize=plot_utils.FONT_SIZE,
            color=plot_utils.mask_colors[self.name],
        )

        if holes_list is None:
            if self.name == "D":
                # Hardcoded Mask D display path (biclique representative route)
                holes_arr = np.array(home_xy + [[8, 5], [8, 1], [1, 1], [1, 2], [5, 2]] + out_xy)
            else:
                holes_arr = np.array(home_xy + self.holes_list + out_xy)
        else:
            holes_arr = np.array(holes_list)

        n_positions = holes_arr.shape[0] - 1
        line_ends = holes_arr.reshape(-1, 1, 2)
        segments = np.concatenate([line_ends[:-1], line_ends[1:]], axis=1) + 0.5

        color_list = np.array(range(n_positions))
        norm = plt.Normalize(0, n_positions)
        lc = LineCollection(segments, cmap="viridis", norm=norm, zorder=zorder)
        lc.set_capstyle("round")
        lc.set_array(color_list)
        lc.set_linewidth(path_linewidth)
        ax.add_collection(lc)
        ax.set_aspect("equal", "box")

        if maskd_bottleneck:
            ax.plot([0, self.size], [1, 1], color="red", linewidth=path_linewidth / 2)
            ax.plot([0, self.size], [2, 2], color="red", linewidth=path_linewidth / 2)
            ax.plot([self.size // 2, self.size // 2], [0, self.size],
                    color="red", linewidth=path_linewidth / 2)
            ax.plot([self.size // 2 + 1, self.size // 2 + 1], [0, self.size],
                    color="red", linewidth=path_linewidth / 2)

        if plot_ho:
            ax.text(home_xy[0][0], home_xy[0][1], s="H",
                    fontsize=plot_utils.FONT_SIZE, color="tab:blue",
                    horizontalalignment="center")
            ax.text(out_xy[0][0] + 0.5, out_xy[0][1] + 1, s="O",
                    fontsize=plot_utils.FONT_SIZE, color="tab:orange",
                    horizontalalignment="center")
        return ax, lc

    def __repr__(self):
        holes = set(tuple(hole) for hole in self.holes_coords)
        _str = "\n"
        for row in range(self.size):
            for col in range(self.size):
                if (row, col) == (5, 0):
                    _str += " "
                elif (row, col) == (0, 5):
                    _str += " "
                elif (col, self.size - row - 1) in holes:
                    _str += "O"
                else:
                    _str += "+"
            _str += "\n"
        return _str

    def __str__(self):
        return f"size={self.size}, holes={self.holes_coords}"

    def _get_tiles_adj_mat(self):
        """
        Build the undirected tile adjacency matrix for the two-floor maze.

        Returns
        -------
        ndarray of shape (2 * size**2, 2 * size**2)
            Entry ``[i, j] = 1`` iff tiles ``i`` and ``j`` are directly connected by a
            corridor or a floor hole; 0 otherwise.

        Notes
        -----
        Tile index: ``x + y * size + z * size**2``.
        Floor z=0: horizontal corridors (left–right adjacency within each row).
        Floor z=1: vertical corridors (up–down adjacency within each column).
        Holes connect the same (x, y) tile on both floors.
        """
        n = self.size ** 2 * 2
        A = np.zeros((n, n))
        for y in range(self.size):
            for x in range(self.size - 1):
                i = x + y * self.size
                j = (x + 1) + y * self.size
                A[i, j] = A[j, i] = 1
        for x in range(self.size):
            for y in range(self.size - 1):
                i = x + y * self.size + self.size ** 2
                j = x + (y + 1) * self.size + self.size ** 2
                A[i, j] = A[j, i] = 1
        for x, y in self.holes_coords:
            i = x + y * self.size
            j = i + self.size ** 2
            A[i, j] = A[j, i] = 1
        return A

    def _get_tiles_indices(self):
        """
        Compute tile indices (bottom and top floor) for each hole position.

        Returns
        -------
        ndarray of shape (n_holes, 2)
            Column 0: tile index on floor z=0.  Column 1: tile index on floor z=1.
        """
        t_idx_arr = np.zeros_like(self.holes_coords)
        bottom = np.zeros((self.holes_coords.shape[0], 3))
        bottom[:, :2] = self.holes_coords
        t_idx_arr[:, 0] = utils.xyz_to_ti(bottom, self.size)
        t_idx_arr[:, 1] = t_idx_arr[:, 0] + self.size ** 2
        return t_idx_arr

    def _get_corridors_adj_mat(self):
        """
        Build the undirected corridor adjacency matrix.

        Returns
        -------
        ndarray of shape (2 * size, 2 * size)
            Entry ``[i, j] = 1`` iff corridors ``i`` and ``j`` are connected through a hole;
            0 otherwise.

        Notes
        -----
        Corridor indices 0–(size−1) are horizontal (floor z=0, indexed by row ``y``).
        Corridor indices size–(2*size−1) are vertical (floor z=1, indexed by ``x + size``).
        Each hole at ``(x, y)`` connects corridor ``y`` (horizontal) with corridor ``x + size``
        (vertical).
        """
        co_adj_mat = np.zeros((self.size * 2, self.size * 2))
        for hole in self.holes_coords:
            x, y = hole
            co_adj_mat[y, x + self.size] = co_adj_mat[x + self.size, y] = 1
        return co_adj_mat


class MaskDSpecial(Mask):
    """
    Mask D variant with the adjusted Jaccard correction for 3 mandatory transitions.

    Mask D has a biclique bottleneck topology: the single bottleneck corridor must be
    traversed exactly 3 times on every shortest path from home to out.  These 3 transitions
    are topologically mandatory, so the Jaccard similarity denominator and numerator are each
    reduced by 3 (R15 / C10).

    Attributes
    ----------
    n_guaranteed_transitions_for_adjusted_jaccard : int
        Set to 3 for Mask D (overrides the base ``Mask`` default of 0).

    Notes
    -----
    Used by ``DataLoader._load_mask`` for masks named ``'D'`` or ``'D_flipped'``.
    Passed to ``utils.transition_vec_similarity`` via
    ``mask.n_guaranteed_transitions_for_adjusted_jaccard``.
    """

    n_guaranteed_transitions_for_adjusted_jaccard = 3
