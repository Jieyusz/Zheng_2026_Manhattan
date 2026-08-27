"""Maze / graph / model schematic figures and their annotations.

Split out of plot_utils.py.
"""
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from itertools import permutations
from manhattan_maze.plot_constants import (FONT_SIZE, LW_DATA, LW_HAIRLINE, MS_AREA_LARGE,
                                            TICK_SIZE)
from manhattan_maze.plot_utils import add_direction_arrows, draw_arrow, draw_ellipse, draw_rhombus, draw_vertical_parallelogram, format_xs_ys, get_normalized_color_seq, ob_condition_color_dict, plot_illustrative_cbar

__all__ = ['plot_schematic_3d_maze', 'plot_schematic_swap_maze', 'plot_schematic_cage_swap', 'plot_schematic_path_graph', 'plot_tile_path_graph', 'plot_schematic_d_graph', 'plot_maskd_corridor_interval', 'plot_maskd_similarity_matrix', 'plot_allocentric_turn_seq', 'plot_biclique_transitions_colormap', 'plot_markov_schematics', 'format_mask_d_zones', 'format_path_graph_zones', 'plot_hole_decision_schematic', 'add_biclique_arrows', 'node_position', 'plot_corridor_transition_schematic', 'plot_circle_with_signal_values', 'plot_edges_based_on_adj_mat', 'plot_goal_signal', 'plot_exponential_schematic', 'plot_d2_session_timeline', 'plot_ablation_timeline']

def _draw_tray_outline(ax, x, y, fig_width, corr_height, y_increment, linewidth, zorder):
    """
    Draw the five black boundary edges of one isometric maze tray (floor).

    Shared by the bottom floor and the top tray of :func:`plot_schematic_3d_maze` so
    both outlines use exactly the same edge geometry. ``(x, y)`` is the tray's lower-left
    corner in data coordinates; the tray is a rhombus of width ``fig_width`` slanted by
    ``y_increment`` over its half-width, with vertical walls of height ``corr_height``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : float
        Lower-left corner of the tray in data coordinates.
    fig_width : float
        Full horizontal extent of the tray.
    corr_height : float
        Height of the vertical (parallelogram) walls.
    y_increment : float
        Vertical slant of the isometric projection across the tray half-width.
    linewidth : float
    zorder : int
        Draw order for the outline edges.
    """
    ax.plot([x, x], [y + corr_height, y], linewidth=linewidth, c="black", zorder=zorder)
    ax.plot([x + fig_width / 2, x + fig_width / 2], [y + corr_height - y_increment, y - y_increment],
            linewidth=linewidth, c="black", zorder=zorder)
    ax.plot([x + fig_width, x + fig_width], [y + corr_height, y], linewidth=linewidth, c="black", zorder=zorder)
    ax.plot([x, x + fig_width / 2], [y, y - y_increment], linewidth=linewidth, c="black", zorder=zorder)
    ax.plot([x + fig_width / 2, x + fig_width], [y - y_increment, y], linewidth=linewidth, c="black", zorder=zorder)


def plot_schematic_3d_maze(ax, zorder_list=None,
                           bottom_properties=None,
                           mask_properties=None,
                           top_properties=None,
                           alpha_increment=0.1, hole_color="white", arrow_color="tab:purple", maze_size=11, linewidth=LW_HAIRLINE, add_text=True,
                           reverse_arrow=False, arrow_width_scale=1.0, arrow_zorder=None,
                           arrow_length_scale=1.0, arrow_head_width=None, arrow_head_length=None):
    '''
    Draw the isometric three-layer schematic of the physical maze.

    Renders, bottom to top: the entrance ("H") floor with its corridor walls, the
    occluding mask layer in the middle, and the exit ("O") tray on top, plus the
    purple navigation arrows linking entrance hole -> mask hole -> exit hole. Used as
    the building block for the swap/cage schematics.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis; set to equal aspect and turned off by this function.
    zorder_list : list of int or None
        Five ascending z-orders ``[floor, floor_edges, mask, top_fill, top_edges]``.
        Defaults to ``[0, 5, 10, 15, 20]``.
    bottom_properties, mask_properties, top_properties : dict or None
        ``{"color": ..., "alpha": ...}`` fills for the entrance floor, the mask, and the
        exit tray. Default to blue / black / orange respectively.
    alpha_increment : float
        Extra alpha added to interior/near walls so they read as darker than far walls.
    hole_color : color
        Fill color for the entrance/exit/mask holes.
    arrow_color : color
        Color of the navigation arrows.
    maze_size : int
        Number of corridor walls drawn per tray (11 for the 11x11 maze).
    linewidth : float
    add_text : bool
        If True, label the entrance hole "H" and the exit hole "O".
    reverse_arrow : bool
        If True, reverse the arrow directions (exit -> entrance) for the swapped/cage
        configuration.
    arrow_width_scale : float
        Multiplier on the navigation-arrow width; ``1.0`` keeps the default appearance,
        larger values make the run-direction arrows more prominent.
    arrow_zorder : int or None
        If given, draw all navigation arrows at this z-order (e.g. above the mask/trays
        so they stay visible); ``None`` keeps each arrow's original per-layer z-order.
    arrow_length_scale : float
        Shorten (<1) or lengthen (>1) the arrow shafts about their midpoint; ``1.0``
        keeps the original lengths.
    arrow_head_width, arrow_head_length : float or None
        If either is given, draw the arrows as ``FancyArrow`` with a fixed head size (in
        data units) independent of shaft length, so a short shaft still shows a bold head;
        ``None`` (default) keeps the equilateral arrow whose head scales with the shaft.

    Returns
    -------
    matplotlib.axes.Axes
        The same ``ax``, for chaining.

    Notes
    -----
    Lengths are in inches scaled to the physical construction; the geometry constants
    (corridor width 1.5", tray height 2", 1.75x visualization scale) reproduce the real
    maze proportions and must not be changed without re-checking every dependent figure.
    '''
    if zorder_list is None:
        zorder_list = [0, 5, 10, 15, 20]
    if bottom_properties is None:
        bottom_properties = {"color": "tab:blue", "alpha": 0.3}
    if mask_properties is None:
        mask_properties = {"color": "black", "alpha": 0.5}
    if top_properties is None:
        top_properties = {"color": "tab:orange", "alpha": 0.3}

    # Keep these parameters the same for all plots
    # Maze dimensions based on experiments
    fig_width = maze_size * 2  # inch, for y ranges
    corr_width = 1.5  # inch, actual width of the corridor
    tray_height = 2  # inch, actual height of the tray
    scaling_factor = 1.75  # scale to the actual construction visualization
    corr_height = np.sqrt(1 + (2 * tray_height) ** 2) / (
                2 * tray_height) / corr_width * scaling_factor  # parallellogram height

    # scale layer and height
    y_increment = fig_width / 8  # vertical increment defines the space between different structures, 1/8 is arbitrary
    fig_height = (y_increment * 7 + corr_height / 2)  # arbitrary number to remove blank space
    # scaled hole
    hole_height = corr_height / scaling_factor * corr_width  # scale to 1.5 inch
    hole_width = hole_height / np.sqrt(tray_height)  # 45 degree angle

    xrange = (-0.1, fig_width + 0.1)
    yrange = (0, fig_height)
    # ax.set_position([0, 0, 0.5, 0.5])

    ## Plot bottom floor:
    origin_x, origin_y = (0, y_increment)
    # for all tray, the weidth to height ration is 1:4
    draw_rhombus(ax, origin_x, origin_y, fig_width, fig_width / 4, bottom_properties["color"],
                       alpha=bottom_properties["alpha"])

    ## Exterior walls of the bottom tray
    # slant up — previously drawn twice at base alpha, which compounded; draw once at 2x
    # alpha to keep the same darker appearance with a single patch.
    draw_vertical_parallelogram(ax, origin_x, origin_y, fig_width / 2, corr_height, y_increment,
                                      bottom_properties["color"], alpha=2 * bottom_properties["alpha"])
    # slant down
    draw_vertical_parallelogram(ax, origin_x + fig_width / 2, origin_y - y_increment,
                                      fig_width / 2, corr_height, y_increment, bottom_properties["color"],
                                      alpha=bottom_properties["alpha"] + alpha_increment)

    # Entrance hole of the bottom floor (left, middle corridor:
    draw_ellipse(ax, origin_x + fig_width / 4, origin_y + y_increment / 2 + corr_height / 2, hole_width,
                       hole_height, hole_color)
    # Add text

    if add_text:
        ax.text(origin_x + fig_width / 4, origin_y + y_increment + corr_height / 2, s="H",
            fontweight="bold", color=bottom_properties["color"],
            horizontalalignment="center", verticalalignment="center")

    def _nav_arrow(x, y, dx, dy, default_zorder=None):
        # Shorten the shaft about its midpoint (arrow_length_scale) while keeping the
        # start/end centered on the original path; head size is fixed when arrow_head_*
        # are set so a short shaft still shows a bold, direction-revealing head.
        cx, cy = x + dx / 2, y + dy / 2
        sdx, sdy = dx * arrow_length_scale, dy * arrow_length_scale
        z = arrow_zorder if arrow_zorder is not None else default_zorder
        draw_arrow(ax, cx - sdx / 2, cy - sdy / 2, sdx, sdy, corr_height * arrow_width_scale,
                   color=arrow_color, zorder=z, head_width=arrow_head_width, head_length=arrow_head_length)

    # Arrow on the bottom floor, point to the center of the tray
    b_arrow_x, b_arrow_y = origin_x + fig_width / 4, origin_y + y_increment / 2 + corr_height / 2
    b_dx = fig_width / 4
    b_dy = -y_increment / 2
    if reverse_arrow:
        _nav_arrow(b_arrow_x+b_dx, b_arrow_y+b_dy, -b_dx, -b_dy)
    else:
        _nav_arrow(b_arrow_x, b_arrow_y, b_dx, b_dy)

    # Interior walls
    # First wall is exterior, add alpha to make it darker
    draw_vertical_parallelogram(ax, origin_x, origin_y, fig_width / 2, corr_height, -y_increment,
                                      bottom_properties["color"], alpha=bottom_properties["alpha"] + alpha_increment, )
    # The rest of the walls
    for i in np.arange(maze_size) + 1:
        wall_x = origin_x + i
        wall_y = origin_y + i / 4
        draw_vertical_parallelogram(ax, wall_x, wall_y, fig_width / 2, corr_height, -y_increment,
                                          bottom_properties["color"], alpha=bottom_properties["alpha"])
        # draw rims of the walls
        ax.plot([wall_x, wall_x], [wall_y + corr_height, wall_y + corr_height - 1 / 2], linewidth=linewidth, c="black",
                zorder=zorder_list[1])
        ax.plot([wall_x, wall_x + fig_width / 2], [wall_y + corr_height, wall_y + corr_height - y_increment],
                linewidth=linewidth, c="black", zorder=zorder_list[1])
    # Plot edges of the bottom floor
    draw_rhombus(ax, origin_x, origin_y + corr_height, fig_width, fig_width / 4, "white", ec="black",
                       linewidth=linewidth, fill=False)

    # Plot edges of the walls
    _draw_tray_outline(ax, origin_x, origin_y, fig_width, corr_height, y_increment, linewidth, zorder_list[1])

    ## Plot Mask in the middle (Mask O)
    mask_x, mask_y = (0, y_increment * 3)
    # Plot bottom arrow pointing up
    if reverse_arrow:
        _nav_arrow(mask_x + fig_width / 2, mask_y + fig_width/4, 0, corr_width-fig_width / 4, default_zorder=zorder_list[1])
    else:
        _nav_arrow(mask_x + fig_width / 2, mask_y - fig_width / 4 + 1, 0, fig_width / 4 - corr_width, default_zorder=zorder_list[1])
    # Plot mask
    draw_rhombus(ax, mask_x, mask_y, fig_width, fig_width / 4, mask_properties["color"],
                       alpha=mask_properties["alpha"], zorder=zorder_list[2])
    # add an edge to the mask
    draw_rhombus(ax, mask_x, mask_y, fig_width, fig_width / 4, "white", fill=False, ec="black",
                       linewidth=linewidth, zorder=zorder_list[2])
    # Draw the hole on the mask
    draw_ellipse(ax, mask_x + fig_width / 2, mask_y, 1, 0.5, hole_color, ec="black", linewidth=linewidth,
                       zorder=zorder_list[2])
    # Plot another vertical arrow
    if reverse_arrow:
        _nav_arrow(mask_x + fig_width / 2, mask_y-1/2, 0, corr_width - fig_width/4)
    else:
        _nav_arrow(mask_x + fig_width / 2, mask_y + 1, 0, fig_width / 4 - corr_width, default_zorder=zorder_list[2])

    # Plot top tray
    top_x, top_y = (0, y_increment * 5 - corr_height / 2)
    # Plot exterior walls
    draw_vertical_parallelogram(ax, top_x, top_y, fig_width / 2, corr_height, -y_increment,
                                      top_properties["color"], alpha=0.3)
    draw_vertical_parallelogram(ax, top_x + fig_width / 2, top_y + y_increment, fig_width / 2, corr_height,
                                      -y_increment, top_properties["color"], alpha=0.3)
    # Plot exit hole (top right, middle corridor)
    draw_ellipse(ax, top_x + fig_width * 0.75, top_y + y_increment / 2 + corr_height / 2, hole_width, hole_height,
                       hole_color)
    # add text
    if add_text:
        ax.text(top_x + fig_width * 0.75, top_y + y_increment + corr_height / 2, s="O",
            fontweight="bold", color=top_properties["color"],
            horizontalalignment="center", verticalalignment="center")

    # top arrow:
    if reverse_arrow:
        _nav_arrow(top_x + fig_width * 0.75, top_y + y_increment / 2 + corr_height / 2, -b_dx, b_dy)
    else:
        _nav_arrow(top_x + fig_width / 2, top_y + corr_height / 2, b_dx, -b_dy)

    for i in np.arange(maze_size + 1):
        wall_x = top_x + i
        wall_y = top_y - i / 4
        draw_vertical_parallelogram(ax, wall_x, wall_y, fig_width / 2, corr_height, y_increment,
                                          top_properties["color"], alpha=top_properties["alpha"], zorder=zorder_list[3])

    draw_rhombus(ax, top_x, top_y + corr_height, fig_width, fig_width / 4, top_properties["color"],
                       alpha=top_properties["alpha"], zorder=zorder_list[3])
    draw_rhombus(ax, top_x, top_y + corr_height, fig_width, fig_width / 4, "white", ec="black",
                       linewidth=linewidth, fill=False, zorder=zorder_list[4])

    # Plot edges of the top box:
    _draw_tray_outline(ax, top_x, top_y, fig_width, corr_height, y_increment, linewidth, zorder_list[4])

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax


def plot_schematic_swap_maze(axes, **schematic_kwargs):
    '''
    Draw the entrance/exit-swap protocol as two mazes with a "Swap" arrow between them.

    The left maze uses the default entrance/exit color coding; the right maze swaps the
    top/bottom colors to depict the reversed reward configuration after 10-15 roundtrips.

    Parameters
    ----------
    axes : sequence of 3 matplotlib.axes.Axes
        ``[before, annotation, after]``. The middle axis holds the swap arrow and text.
    **schematic_kwargs
        Forwarded to :func:`plot_schematic_3d_maze` for both mazes.

    Raises
    ------
    AssertionError
        If exactly three axes are not provided.
    '''
    assert len(axes) == 3, "You need 3 axes to plot the swap maze"
    # plot the maze
    plot_schematic_3d_maze(axes[0], **schematic_kwargs)
    plot_schematic_3d_maze(axes[-1], top_properties={"color":"tab:blue", "alpha":0.3}, bottom_properties={"color":"tab:orange", "alpha":0.3}, **schematic_kwargs)
    # add a horizontal from left to right
    draw_arrow(axes[1], x=0.25, y=0.5, dx=0.5, dy=0, w=0.1, color="black")
    axes[1].text(0.5, 0.4, s="Swap", horizontalalignment="center", verticalalignment="center", fontsize=TICK_SIZE)
    axes[1].text(0.5, 0.6, s=r"$\sim20$ rewards", fontsize=TICK_SIZE, horizontalalignment="center", verticalalignment="center")
    axes[1].axis("off")


def _add_return_route_label(ax, second_line):
    '''Two-line caption below a cage-swap maze: "Return route: <hollow triangle>" over
    ``second_line`` (e.g. "Homebound" or "To H"). The triangle is drawn as an open
    down-triangle scatter marker to match the To-O glyph used in the turn-error panels.
    Placed a little below the axes so a gap separates it from the schematic. The open
    ``$\triangledown$`` glyph keeps the first line centered as a single text unit.'''
    ax.text(0.5, -0.10, s=r"Return route: $\triangledown$", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, -0.24, s=second_line, ha="center", va="center", transform=ax.transAxes)


def plot_schematic_cage_swap(axes, **schematic_kwargs):
    '''
    Draw the home-cage relocation protocol as two mazes labeled by cage position.

    The left maze marks the cage on the West side (entrance default colors); the right
    maze marks the cage on the North side with reversed navigation arrows. A "vs." label
    sits between them.

    Parameters
    ----------
    axes : sequence of 3 matplotlib.axes.Axes
        ``[west_cage, annotation, north_cage]``.
    **schematic_kwargs
        Forwarded to :func:`plot_schematic_3d_maze` for both mazes.

    Raises
    ------
    AssertionError
        If exactly three axes are not provided.
    '''
    assert len(axes) == 3, "You need 3 axes to plot the swap cage"
    # plot the mazes; the two cage configs have opposite return-route geometries, so the
    # West maze points exit -> entrance while the North maze keeps the default direction.
    plot_schematic_3d_maze(axes[0], **schematic_kwargs, reverse_arrow=True)
    # plot the cage location
    axes[0].text(-0.1, 0.25, s="Cage", color="tab:blue", horizontalalignment="center", verticalalignment="center", transform=axes[0].transAxes,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="none", edgecolor="tab:blue", linewidth=LW_HAIRLINE))
    axes[0].text(0.5, 1, s="West", horizontalalignment="center", verticalalignment="center", transform=axes[0].transAxes)
    _add_return_route_label(axes[0], "Homebound")
    plot_schematic_3d_maze(axes[-1], **schematic_kwargs, reverse_arrow=False)
    axes[-1].text(1, 0.8, s="Cage", color="tab:orange",horizontalalignment="center", verticalalignment="center", transform=axes[-1].transAxes,
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="none", edgecolor="tab:orange", linewidth=LW_HAIRLINE))
    axes[-1].text(0.5, 1, s="North", horizontalalignment="center", verticalalignment="center", transform=axes[-1].transAxes)
    _add_return_route_label(axes[-1], "To O (cage)")
    # add a horizontal from left to right
    axes[1].text(0.5, 0.5, s="vs.", horizontalalignment="center", verticalalignment="center")
    axes[1].axis("off")


def plot_schematic_path_graph(ax, H_circle_x=1, V_circle_x=3, n_nodes=5, radius=0.25, zorder_list=None, path_linewidth=LW_DATA, arrow_length=0.8,
                              top_color="tab:orange", bottom_color="tab:blue", arrow_color="tab:purple",
                              y_scale=1, plot_shortest_path=False, anchor=None):
    '''
    Draw the P10 corridor graph as two columns of nodes (horizontal vs vertical corridors).

    Left column ("Hori.") and right column ("Vert.") each hold ``n_nodes`` corridor nodes;
    edges connect each horizontal node to the vertical node at the same level and to the
    next level up, reproducing the path-graph adjacency. Optionally overlays the shortest
    H -> O path as a viridis-colored line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    H_circle_x, V_circle_x : float
        Data x-coordinates of the horizontal- and vertical-corridor columns.
    n_nodes : int
        Number of nodes per column.
    radius : float
        Node circle radius in data units.
    zorder_list : list of int or None
        Five ascending z-orders ``[edges_bg, edges, nodes, _, _]``; defaults to
        ``[0, 5, 10, 15, 20]``.
    path_linewidth : float
        Linewidth of the shortest-path overlay; graph edges use half this.
    arrow_length : float
        Size of the direction arrows.
    top_color, bottom_color : color
        Fill colors for the vertical (top) and horizontal (bottom) nodes.
    arrow_color : color
        Color of the navigation direction arrows.
    y_scale : float
        Vertical spacing between successive node rows.
    plot_shortest_path : bool
        If True, overlay the shortest H -> O traversal colored by step order.

    Returns
    -------
    matplotlib.axes.Axes
    '''

    # Connect H and V
    if zorder_list is None:
        zorder_list = [0, 5, 10, 15, 20]

    # Connect the V to next H
    for i in range(n_nodes - 1):
        ax.plot([H_circle_x, V_circle_x], [(i+0.5) * y_scale, ((i + 1.5) * y_scale)], color="black", zorder=zorder_list[1], linewidth=path_linewidth / 2)
    for i in range(n_nodes):
        ax.add_patch(mpatches.Circle((H_circle_x, (i+0.5) * y_scale), radius=radius, color=bottom_color, zorder=zorder_list[2]))
        ax.add_patch(mpatches.Circle((V_circle_x, (i+0.5) * y_scale), radius=radius, color=top_color, zorder=zorder_list[2]))
        ax.plot([H_circle_x, V_circle_x], [(i+0.5) * y_scale, (i + 0.5) * y_scale], color="black", zorder=zorder_list[1], linewidth=path_linewidth / 2)

    if plot_shortest_path:
        # highlight the segments that are the shortest, with viridis color
        # number of positions in the path for line collection
        n_positions = n_nodes * 2 - 1
        line_ends = []
        for i in range(n_nodes):
            line_ends.append((H_circle_x, (n_nodes-i-0.5) * y_scale))  # H nodes
            line_ends.append((V_circle_x, (n_nodes-i-0.5) * y_scale))
        line_ends = np.array(line_ends).reshape(-1, 1, 2)  # reshape to (n, 1, 2) for LineCollection
        segments = np.concatenate([line_ends[:-1], line_ends[1:]], axis=1)
        # select color for each step using viridis
        color_list = np.array(range(n_positions))
        norm = plt.Normalize(0, n_positions)  # normalize
        lc = LineCollection(segments, cmap='viridis', norm=norm, zorder=zorder_list[2])
        lc.set_array(color_list)
        lc.set_linewidth(path_linewidth)  # define linewidth
        ax.add_collection(lc)

    ax.set_aspect("equal", adjustable="box")
    ax.text(H_circle_x, n_nodes * y_scale, s="Hori.", fontsize=TICK_SIZE, horizontalalignment="center")
    ax.text(V_circle_x, n_nodes * y_scale, s="Vert.", fontsize=TICK_SIZE, horizontalalignment="center")

    # Arrows
    add_direction_arrows(ax, H_circle_x, V_circle_x, y_scale, n_nodes, radius, top_color=top_color, bottom_color=bottom_color, arrow_color=arrow_color, arrow_size=arrow_length)
    ax.axis("off")
    ax.set_xlim(H_circle_x-1.5, V_circle_x + 1.6)
    # `anchor` places the equal-aspect box within its (usually taller) cell — e.g. "S" sinks it to the
    # bottom, "N" raises it to the top — to reduce whitespace between stacked graphs. Default None keeps
    # matplotlib's centered placement, so existing callers are unaffected.
    if anchor is not None:
        ax.set_anchor(anchor)
    return ax


def plot_tile_path_graph(ax, holes_list, tile_width=0.8, hole_width=1, linewidth=LW_HAIRLINE, colors=None, maze_size=11, hole_color="tab:grey",
                         plot_HO=False, anchor=None):
    """
    Draw the unrolled path-graph map as a strip of tiled corridors (cf. Tolman's T-maze).

    Each corridor is a row of ``maze_size`` tiles; successive corridors are offset
    horizontally and stacked downward, connected through holes whose x-positions are
    derived from ``holes_list``. Arrows colored by viridis trace the turn sequence from
    the first corridor to the last.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    holes_list : list of (int, int)
        Ordered hole coordinates ``(x, y)`` (tile indices, 0-based) linking consecutive
        corridors; ``len(holes_list) + 1`` corridors are drawn.
    tile_width : float
        Side length of one tile in data units.
    hole_width : float
        Vertical gap between corridors that the connecting hole spans.
    linewidth : float
    colors : list of color or None
        Two colors ``[horizontal, vertical]`` alternated per corridor by parity; defaults
        to ``["tab:blue", "tab:orange"]``.
    maze_size : int
        Tiles per corridor (11 for the 11x11 maze).
    hole_color : color
        Edge color of the hole circles.
    plot_HO : bool
        If True, label the first corridor "H" and the last corridor "O".

    Notes
    -----
    Corridor parity (``k % 2``) determines both the alternating color and the direction
    of the horizontal offset, so the strip zig-zags the way the real corridors alternate
    between horizontal and vertical orientation.
    """
    # color vertical and horizontal corridors separately
    if colors is None:
        colors = ["tab:blue", "tab:orange"]
    n_corridors = len(holes_list) + 1 # number of corridors is number of holes + 1
    # add grids of the tiles
    # determine the starting x of each corridor based on the hole list
    corridor_xs = [0]
    hole_xs = []
    for k, hole in enumerate(holes_list):
        z = k%2
        step = (hole[0] - (maze_size-1-hole[1]))*(-1)**z*tile_width
        corridor_xs.append(step+corridor_xs[-1])
        hole_x = corridor_xs[k] + (z * (maze_size - 1) + (-1) ** z * hole[z]) * tile_width
        hole_xs.append(hole_x)

    # add the last hole_xs for the last corridor
    hole_xs.append(corridor_xs[-1])

    # add holes based on coordinates in the hole lists
    for k, hole in enumerate(holes_list):
        hole_x = hole_xs[k]
        ax.plot([hole_x, hole_x], [-(k+1)*tile_width-k*hole_width, -(k+1)*(tile_width+hole_width)], color="tab:grey", linewidth=linewidth)
        ax.plot([hole_x+tile_width, hole_x+tile_width], [-(k + 1) * tile_width -k * hole_width, -(k + 1) * (tile_width + hole_width)],
                color="tab:grey", linewidth=linewidth)
        # add a circle as the hole
        diameter = min(hole_width, tile_width)
        draw_ellipse(ax, hole_x+tile_width/2, -(k+1)*tile_width-k*hole_width-hole_width/2,
                     diameter*0.9, diameter*0.9, color="white", ec=hole_color, linewidth=linewidth)
        # add an arrow to indicate the turns, depending on the corridor_xs value
        dx = hole_xs[k+1] - hole_xs[k]
        draw_arrow(ax, hole_x+tile_width/2, -(k+1)*(tile_width+hole_width)-tile_width/2, dx, 0, tile_width,
                   color=plt.cm.viridis((k+1)/len(holes_list)), zorder=5)

    # add one arrow from the beginning
    draw_arrow(ax, corridor_xs[0]+tile_width/2, -tile_width/2, hole_xs[0]-corridor_xs[0], 0, tile_width/2, color=plt.cm.viridis(0), zorder=5)


    for i in range(n_corridors): # plot the corridor grids
        # horizontal lines:
        ax.plot([corridor_xs[i], maze_size*tile_width+corridor_xs[i]], [-i*(tile_width+hole_width), -i*(tile_width+hole_width)], color=colors[i%2], linewidth=linewidth)
        ax.plot([corridor_xs[i], maze_size*tile_width+corridor_xs[i]], [-(i+1)*tile_width-i*hole_width, -(i+1)*tile_width-i*hole_width], color=colors[i%2], linewidth=linewidth)
        # vertical lines:
        for j in range(maze_size+1):
            ax.plot([j*tile_width+corridor_xs[i], j*tile_width+corridor_xs[i]], [-i*(tile_width+hole_width), -(i+1)*tile_width-i*hole_width], color=colors[i%2], linewidth=linewidth)

    # Add H and O on the first corridor and last corridor
    if plot_HO:
        ax.text(corridor_xs[0]-tile_width, -tile_width/2, s="H", horizontalalignment="right", verticalalignment="center", color=colors[0])
        ax.text(corridor_xs[-1]-tile_width/2, -(n_corridors-1)*(tile_width+hole_width)-tile_width/2, s="O", color=colors[1], horizontalalignment="right", verticalalignment="center")

    # formating
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    # `anchor` places the equal-aspect box within its cell (e.g. "N" to top) to close the gap to a
    # graph stacked above it; default None preserves matplotlib's centered placement for existing callers.
    if anchor is not None:
        ax.set_anchor(anchor)


def plot_schematic_d_graph(ax, n_biclique=4, H_circle_x=1, V_circle_x=2, n_nodes=9, radius=0.18,
                           zorder_list=None, biclique_colors=None, y_scale=0.5,
                           top_color="tab:orange", bottom_color="tab:blue", arrow_color="tab:purple",
                           bottleneck_color="red", path_linewidth=LW_DATA, highlight_bottleneck=True, highlight_keynodes=False,
                           keynode_color="tab:purple", plot_shortest_path=True, outbound=True,
                           plot_direction_arrows=True, involved_positions=None, uninvolved_color="tab:grey"):
    """
    Draw the Mask-D corridor graph: two bicliques joined by a single bottleneck (P2) edge.

    The graph is laid out as two columns (horizontal vs vertical corridors). The bottom
    ``n_biclique`` rows form the first complete bipartite block, a single P2 bottleneck
    connects to the top ``n_biclique`` rows forming the second block. Optional overlays
    mark the bottleneck nodes, the high-degree key nodes, and the shortest H -> O path.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    n_biclique : int
        Number of nodes per side in each bipartite block.
    H_circle_x, V_circle_x : float
        Data x-coordinates of the horizontal and vertical corridor columns.
    n_nodes : int
        Total node rows per column (``2 * n_biclique + 1`` for the two blocks plus P2).
    radius : float
        Node circle radius.
    zorder_list : list of int or None
        Five ascending z-orders ``[edges, _, nodes, path, _]``; defaults to
        ``[0, 5, 10, 15, 20]``.
    biclique_colors : list of color or None
        Two edge colors ``[top_block, bottom_block]``; defaults to grey/grey.
    y_scale : float
        Vertical spacing between node rows.
    top_color, bottom_color : color
        Fill colors for vertical (top) and horizontal (bottom) nodes.
    arrow_color : color
        Direction arrow color.
    bottleneck_color : color
        Outline color for the bottleneck (and key) nodes.
    path_linewidth : float
        Linewidth for the shortest-path overlay; edges use half this.
    highlight_bottleneck : bool
        If True, outline the two topologically mandatory bottleneck nodes.
    highlight_keynodes : bool
        If True, fill and emphasize the high-degree corridors and their edge fans.
    keynode_color : color
        Fill color for key nodes when ``highlight_keynodes`` is True.
    plot_shortest_path : bool
        If True, overlay the optimal H -> O path colored by step order.
    outbound : bool
        Direction of travel; selects which key nodes/arrows are emphasized.
    involved_positions : iterable of int or None
        Display-position indices (``node_position`` / ``add_biclique_arrows`` index
        space) of the nodes to keep colored; every other node is drawn in
        ``uninvolved_color``. ``None`` colors all nodes normally.
    uninvolved_color : color, default 'tab:grey'
        Fill color for nodes not in ``involved_positions``.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    AssertionError
        If ``biclique_colors`` is given but is not length 2.

    Notes
    -----
    The ``+1.5`` row offsets in the top block leave room for the central P2 bottleneck
    edge that is the defining feature of the Mask-D topology.
    """
    if zorder_list is None:
        zorder_list = [0, 5, 10, 15, 20]

    if biclique_colors is None:
        biclique_colors = ["tab:grey", "tab:grey"]
    else:
        assert len(biclique_colors) == 2, "You need to provide two colors for the biclique"

    def draw_biclique_edges(i_offset, j_offset, color):
        """Draw all H->V edges of one bipartite block, rows offset by i_offset/j_offset."""
        for i in range(n_biclique):
            for j in range(n_biclique):
                ax.plot([H_circle_x, V_circle_x], [(i + i_offset) * y_scale, (j + j_offset) * y_scale],
                        color=color, zorder=zorder_list[0], linewidth=path_linewidth / 2)

    # top block: +1.5 leaves room for the central P2 bottleneck edge
    draw_biclique_edges(n_biclique + 1.5, n_biclique + 1.5, biclique_colors[0])

    # connect the bottleneck P2
    ax.plot([H_circle_x, V_circle_x], [(n_biclique + 0.5)*y_scale, (n_biclique + 1.5)*y_scale], color="tab:grey", zorder=zorder_list[0],
            linewidth=path_linewidth / 2)
    ax.plot([H_circle_x, V_circle_x], [(n_biclique + 0.5)*y_scale, (n_biclique + 0.5)*y_scale], color="tab:grey", zorder=zorder_list[0],
            linewidth=path_linewidth / 2)

    # bottom block
    draw_biclique_edges(0.5, 1.5, biclique_colors[1])

    ax.plot([H_circle_x, V_circle_x], [0.5*y_scale, 0.5*y_scale], color="tab:grey", zorder=zorder_list[0], linewidth=path_linewidth / 2)

    if plot_shortest_path:
        # Highlight optimal path
        optimal_nodes = [(H_circle_x, (n_nodes - 0.5)*y_scale), (V_circle_x, (n_nodes / 2 + 1)*y_scale), (H_circle_x, (n_nodes / 2 )*y_scale),
                         (V_circle_x, (n_nodes / 2)*y_scale),
                         (H_circle_x, 0.5*y_scale), (V_circle_x, 0.5*y_scale)]
        line_ends = np.array(optimal_nodes).reshape(-1, 1, 2)  # reshape to (n, 1, 2) for LineCollection
        # select color for each step using viridis
        color_list = np.array(range(len(optimal_nodes) - 1))
        norm = plt.Normalize(0, len(optimal_nodes) - 1)  # normalize
        segments = np.concatenate([line_ends[:-1], line_ends[1:]], axis=1)  # +0.5 to center the lines
        lc = LineCollection(segments, cmap='viridis', norm=norm, zorder=zorder_list[3])
        lc.set_array(color_list)
        lc.set_linewidth(path_linewidth)  # define linewidth
        ax.add_collection(lc)

    # Map a drawn node (column, row i) to its display-position index (the same
    # index space as node_position / add_biclique_arrows), then grey it out when an
    # ``involved_positions`` set is given and the node is not in it.
    def node_fill(is_v, i, base):
        if involved_positions is None:
            return base
        pos = 2 * (n_nodes - 1 - i) + (1 if is_v else 0)
        return base if pos in involved_positions else uninvolved_color

    # Draw all nodes
    for i in range(n_nodes):
        ax.add_patch(mpatches.Circle((H_circle_x, (i+0.5)*y_scale), radius=radius, color=node_fill(False, i, bottom_color), zorder=zorder_list[2]))
        ax.add_patch(mpatches.Circle((V_circle_x, (i+0.5)*y_scale), radius=radius, color=node_fill(True, i, top_color), zorder=zorder_list[2]))

    # add the ones with edge for bottleneck (fill greyed too when uninvolved)
    if highlight_bottleneck:
        ax.add_patch(mpatches.Circle((H_circle_x, (n_nodes / 2)*y_scale), radius=radius+0.02, facecolor=node_fill(False, n_nodes // 2, bottom_color),
                                     edgecolor=bottleneck_color, zorder=zorder_list[2], linewidth=path_linewidth))
        ax.add_patch(mpatches.Circle((V_circle_x, 0.5*y_scale), radius=radius+0.02, facecolor=node_fill(True, 0, top_color),
                                     edgecolor=bottleneck_color, zorder=zorder_list[2], linewidth=path_linewidth))

    if highlight_keynodes: # color the highly connected nodes in the biclique
        if outbound:
            ax.add_patch(mpatches.Circle((V_circle_x, (n_nodes / 2 + 2)*y_scale), radius=radius+0., facecolor=keynode_color,
                                         edgecolor=bottleneck_color, zorder=zorder_list[2], linewidth=path_linewidth))
            # also highlight its edge choices
            for j in range(n_biclique+1):
                # Add 1.5 to account for the P2 in the middle
                ax.plot([V_circle_x, H_circle_x], [(n_biclique + 1.5)*y_scale, (j + n_biclique + 0.5)*y_scale], color=keynode_color,
                        zorder=zorder_list[0], linewidth=path_linewidth)

            ax.add_patch(mpatches.Circle((H_circle_x, 0.5*y_scale), radius=radius, facecolor=keynode_color,
                                         edgecolor=bottleneck_color, zorder=zorder_list[2], linewidth=path_linewidth))
            # also highlight its edge choices
            for j in range(n_biclique+1):
                # Add 1.5 to account for the P2 in the middle
                ax.plot([H_circle_x, V_circle_x], [0.5*y_scale, (j + 0.5)*y_scale], color=keynode_color,
                        zorder=zorder_list[0], linewidth=path_linewidth)
        else:
            ax.add_patch(mpatches.Circle((V_circle_x, (n_nodes / 2 + 1)*y_scale), radius=radius, facecolor=keynode_color,
                                         edgecolor=bottleneck_color, zorder=zorder_list[2], linewidth=path_linewidth))


    ax.set_aspect("equal", adjustable="box")

    # Arrows (and the H/O direction labels)
    if plot_direction_arrows:
        add_direction_arrows(ax, H_circle_x, V_circle_x, y_scale, n_nodes, radius, top_color=top_color,
                             bottom_color=bottom_color, arrow_color=arrow_color, arrow_size=1, outbound=outbound)

    ax.set_xlim(H_circle_x-2*y_scale, V_circle_x + 2*y_scale)
    ax.set_ylim([0, n_nodes*y_scale])  # hide unrelated corridors
    ax.axis("off")
    return ax


def plot_maskd_corridor_interval(ax, interval_array, maskd_special_params,
                                 corridor_groups=None, marker_types=None, linecolor="tab:blue", **kwargs):

    '''
    Plot per-corridor reward/visit intervals for the Mask-D special layout, grouped by zone.

    Draws the interval-vs-corridor line, then overlays scatter markers colored and shaped
    by corridor zone (bottleneck / out / home) so the bottleneck corridor stands out.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    interval_array : ndarray, shape (n_corridors, 3)
        Columns are ``(corridor_index, interval, n_sorties)``. Only the corridor index
        (col 0) and interval (col 1) are plotted; corridor indices are display positions
        in the reduced Mask-D corridor order.
    maskd_special_params : MaskDSpecial
        Provides ``corridor_color_indices_dict[group] -> (facecolor, indices)`` mapping
        each zone name to its scatter color and the corridor indices in that zone.
    corridor_groups : list of str or None
        Zone names to overlay; defaults to ``["Bottleneck", "Out", "Home"]``.
    marker_types : list of str or None
        Matplotlib markers, one per group; defaults to ``["X", "o", "o"]``.
    linecolor : color
        Color of the connecting line and marker edges.
    **kwargs
        Forwarded to ``ax.plot`` for the line.

    Returns
    -------
    list of matplotlib.collections.PathCollection
        The scatter artists, one per corridor group (for legend construction).
    '''
    if marker_types is None:
        marker_types = ["X", "o", "o"]
    if corridor_groups is None:
        corridor_groups = ["Bottleneck", "Out", "Home"]
    n_corridors = interval_array.shape[0]
    corridor_seq = interval_array[:, 0]
    intervals = interval_array[:, 1]
    xs = np.arange(n_corridors) + 1
    ax.plot(xs, intervals, color=linecolor, **kwargs)
    scatter_objects = []
    for group_name, marker_type in zip(corridor_groups, marker_types):
        # get the group index
        facecolor, indices = maskd_special_params.corridor_color_indices_dict[group_name]
        plot_indices = np.isin(corridor_seq, indices)
        # plot the scatter
        scatter_objects.append(ax.scatter(xs[plot_indices], intervals[plot_indices], marker=marker_type, s=MS_AREA_LARGE,
        facecolor=facecolor, edgecolor=linecolor, label=group_name, zorder=10))

    return scatter_objects


def plot_maskd_similarity_matrix(axes, j_oo, j_hh, j_oh_prime, labels=None, axis_labels=None, plot_colorbar=False, cmap="plasma", label_loc="right", label_fontsize=None, **kwargs):
    """
    Render the three Mask-D Jaccard similarity matrices side by side with a shared colorbar.

    Panels, in order: outbound-outbound ``J(O, O)``, homebound-homebound ``J(H, H)``, and
    outbound vs reversed-homebound ``J(O, H')``. Each matrix is shown with a fixed [0, 1]
    color scale so panels are directly comparable.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        Three matrix axes, plus one colorbar axis when ``plot_colorbar`` is True. The
        colorbar axis is taken from the end (``label_loc="right"``) or the start
        (``label_loc="left"``) of the sequence.
    j_oo, j_hh, j_oh_prime : ndarray
        Square similarity matrices in [0, 1] — $J_{O,O}$ (outbound-outbound),
        $J_{H,H}$ (homebound-homebound), and $J_{O,H'}$ (outbound vs reversed-homebound).
    labels : list of str or None
        Three panel titles; defaults to plain-text ``["J(O, O)", "J(H, H)", "J(O, H')"]``.
        Pass manuscript LaTeX (e.g. ``config.SIMILARITY_LATEX.values()``) for math notation.
    axis_labels : dict or None
        Map ``{bout-type: label}`` (e.g. ``config.TRAVERSE_LATEX``) for the x/y axis
        labels; missing keys fall back to the plain-text key. Defaults to ``{}``.
    plot_colorbar : bool
        If True, draw a shared "Jaccard similarity" colorbar on the dedicated axis.
    cmap : str
        Colormap name for ``imshow`` and the colorbar.
    label_loc : {"right", "left"}
        Side of ``axes`` holding the colorbar axis.
    label_fontsize : float or None
        Font size for the three panel titles and the x/y axis labels. None keeps the
        rcParams default. Pass e.g. ``TICK_SIZE`` where the triplet shares a row with
        other panels: the labels are then a smaller share of the cell, which leaves more
        width for the aspect-equal matrices themselves (see plot_d_gen_supp.py).
    **kwargs
        Forwarded to each ``imshow`` call.

    Notes
    -----
    The Jaccard values are produced upstream with the Mask-D-specific guaranteed-transition
    correction (see ``MaskDSpecial.n_guaranteed_transitions_for_adjusted_jaccard``); this
    function only displays them.
    """
    if labels is None:
        labels = ["J(O, O)", "J(H, H)", "J(O, H')"]
    if axis_labels is None:
        axis_labels = {}
    # Select which axes hold the matrices vs the colorbar. mat_axes is always defined so
    # the function works with or without a colorbar (the no-colorbar default previously
    # left mat_axes unbound and raised NameError).
    if plot_colorbar and label_loc == "left":
        mat_axes, cbar_ax = axes[1:], axes[0]
    elif plot_colorbar:
        mat_axes, cbar_ax = axes[:3], axes[-1]
    else:
        mat_axes, cbar_ax = axes, None

    # empty when label_fontsize is None, so the rcParams default is left untouched
    fs = {} if label_fontsize is None else {"fontsize": label_fontsize}

    mat_axes[0].imshow(j_oo, cmap=cmap, vmin=0, vmax=1, **kwargs)
    mat_axes[0].text(0.5, 1, s=labels[0], ha="center", va="bottom", transform=mat_axes[0].transAxes, **fs)
    mat_axes[1].imshow(j_hh,cmap=cmap, vmin=0, vmax=1, **kwargs)
    mat_axes[1].text(0.5, 1, s=labels[1], ha="center", va="bottom", transform=mat_axes[1].transAxes, **fs)
    mat_axes[2].imshow(j_oh_prime, cmap=cmap, vmin=0, vmax=1, **kwargs)
    mat_axes[2].text(0.5, 1, s=labels[2], ha="center", va="bottom", transform=mat_axes[2].transAxes, **fs)
    for k, ax in enumerate(mat_axes):
        ax.set_aspect("equal", adjustable="box")


    mat_axes[0].set_xlabel(axis_labels.get("H-O", "H-O"), **fs)
    mat_axes[2].set_xlabel(axis_labels.get("H-O", "H-O"), **fs)
    mat_axes[1].set_ylabel(axis_labels.get("O-H", "O-H"), **fs)
    mat_axes[1].set_xlabel(axis_labels.get("O-H", "O-H"), **fs)
    mat_axes[2].set_ylabel(axis_labels.get("O-H'", "O-H'"), **fs)
    mat_axes[0].set_ylabel(axis_labels.get("H-O", "H-O"), **fs)


    # add cbar to the last ax
    if plot_colorbar:
        cbar = plot_illustrative_cbar(ax=cbar_ax, cmap=cmap,
                               ticks=[0, 0.5, 1], ticklabels=[0, 0.5, 1], label_loc=label_loc)
        # set the y label
        cbar.ax.set_ylabel("Jaccard similarity", fontsize=TICK_SIZE)


def plot_allocentric_turn_seq(ax, allocentric_turn_seq, cmap=None, turn_vectors=None, size=0.3, jitter=0.1, headsize=0.2, add_text=False, seed=None, **kwargs):
    '''
    Draw an ordered sequence of allocentric turns as colored arrows on the maze grid.

    Each entry is a hole location and the compass direction taken there; arrows are
    colored in temporal order so the reader can follow the path. A small random
    positional jitter is added so overlapping turns at the same hole remain visible.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis, in maze tile coordinates (x to the right, y up).
    allocentric_turn_seq : list of ((int, int), str)
        Ordered turns ``[((hole_x, hole_y), direction), ...]`` where ``direction`` is
        one of ``"N"/"E"/"S"/"W"`` (allocentric compass). Index 0 is the first turn.
    cmap : None, str, or array of RGBA, optional
        Per-arrow colors. ``None`` -> viridis sampled over the sequence length; a
        colormap name -> that colormap sampled likewise; or a precomputed color array
        of length ``len(allocentric_turn_seq)``.
    turn_vectors : dict, optional
        Map from direction letter to unit displacement vector. Defaults to the standard
        N/E/S/W axis-aligned unit vectors.
    size : float, optional
        Arrow length in tile units.
    jitter : float, optional
        Half-width of the uniform positional jitter (tile units) added to each arrow
        base so coincident turns do not overplot.
    headsize : float, optional
        Arrow head width/length in tile units.
    add_text : bool, optional
        If True, annotate each arrow with its 1-based order number.
    seed : int or None, optional
        Seed for the jitter RNG (R11). ``None`` reproduces the previous nondeterministic
        behavior; any fixed seed gives a reproducible jitter pattern. Jitter is cosmetic
        only and does not affect which turns are drawn.

    Notes
    -----
    Arrows are centered on the tile center ``(hole + 0.5)`` before jitter is applied.
    '''
    rng = np.random.default_rng(seed)
    if cmap is None:
        # use a color map the viridis colormap
        cmap = plt.cm.viridis(np.linspace(0, 1, len(allocentric_turn_seq)))
    elif isinstance(cmap, str):
        # create a color map from the string
        cmap = plt.get_cmap(cmap, len(allocentric_turn_seq))

    if turn_vectors is None:
        turn_vectors = {
            "N": np.array([0, 1]),
            "E": np.array([1, 0]),
            "S": np.array([0, -1]),
            "W": np.array([-1, 0])
        }

    for i, (loc, turn) in enumerate(allocentric_turn_seq):
        # plot the turn
        vec = turn_vectors[turn]
        vec = vec*size # scale the vector
        loc = np.array(loc) + rng.uniform(-jitter, jitter, size=2) # add jitter
        ax.arrow(loc[0]+0.5, loc[1]+0.5, vec[0], vec[1], head_width=headsize, head_length=headsize, fc=cmap[i], ec=cmap[i], alpha=1, zorder=5)
        if add_text:
            # add the text
            ax.text(loc[0]+0.5, loc[1]+0.5, s=f"{i+1}", fontsize=TICK_SIZE, color=cmap[i], ha="right", va="top", zorder=10, **kwargs)


def plot_biclique_transitions_colormap(ax, maskd_special_params, transition_counts, biclique_group=None, width_scale=5,
                                       cmap="viridis", node_colors=None, normalize=False, shortest_path_only=False, **kwargs):
    """
    Draw observed corridor-to-corridor transitions within a biclique as a weighted digraph.

    Builds a directed graph of transition counts between corridors of the requested
    biclique group, then draws it with edge width and color encoding transition frequency.
    Edges may be color-normalized either to the per-group min-max range or to the max count.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    maskd_special_params : MaskDSpecial
        Supplies the node layout (``biclique_column_layout``), the per-node colors
        (``biclique_column_colors``), and the shortest-path corridor indices.
    transition_counts : collections.Counter
        Maps ``(from_corridor, to_corridor)`` -> observed count. Corridor indices are
        display positions in the reduced Mask-D corridor order.
    biclique_group : hashable or None
        Which biclique group to lay out and restrict edges to; passed through to
        ``biclique_column_layout``.
    width_scale : float
        Maximum edge width after min-max normalization (only used when ``normalize`` True).
    cmap : str
        Colormap name for edge colors.
    node_colors : list of color or None
        Explicit per-node colors; if None, taken from ``biclique_column_colors``.
    normalize : bool
        If True, rescale edge widths to ``[0, width_scale]`` by the group min-max; if
        False, color edges by ``count / max_count`` at native width.
    shortest_path_only : bool
        If True, keep only edges touching a shortest-path corridor.
    **kwargs
        Forwarded to ``biclique_column_colors`` when deriving node colors.
    """
    biclique_group_pos = maskd_special_params.biclique_column_layout(biclique_group)
    # Filter out the transitions not in the partite dict (nodes are not inspected)
    keys = biclique_group_pos.keys()
    count_subset = {k:val for k, val in transition_counts.items() if k[0] in keys and k[1] in keys}

    # create a graph
    G = nx.DiGraph()
    shortest_path_corridors = maskd_special_params.shortest_path_corridor_indices
    for (u, v), w in count_subset.items():
        if shortest_path_only and (u not in shortest_path_corridors and v not in shortest_path_corridors):
            continue
        G.add_edge(u, v, weight=w)
    # positions for the graph defined by the partite dict
    # Extract edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    norm = mpl.colors.Normalize(vmin=0, vmax=1) # color normalization
    cmap_func = plt.colormaps[cmap]
    if normalize:
        # normalize based on the starting point:
        weight_min, weight_max = min(edge_weights), max(edge_weights)
        # normalize edge widths to [0, 1]
        edge_weights = [width_scale * (w - weight_min) / (weight_max - weight_min) for w in edge_weights]
        edge_colors = [cmap_func(norm(w)) for w in edge_weights]
    else:
        weight_max = max(edge_weights) # just use the max value for color normalization
        edge_colors = [cmap_func(norm(w / weight_max)) for w in edge_weights] # normalize to the max weight

    if node_colors is None:
        node_color_dict = maskd_special_params.biclique_column_colors(**kwargs)
        node_colors = [node_color_dict[n] for n in G.nodes()]

    # Draw graph
    nx.draw(G, pos=biclique_group_pos, ax=ax, with_labels=True,
            node_color=node_colors, node_size=200,
            arrows=True,
            arrowsize=5,
            width=edge_weights,
            edge_color=edge_colors, alpha=0.8,
            connectionstyle="arc3,rad=0.2",
            font_size=TICK_SIZE)


def plot_markov_schematics(ax, origin=None, horizontal_color="tab:blue", vertical_color="tab:orange", y_offset=1,
                           x_offset=0.8, radius=0.2, arrow_color="black", markov_colors=None,):
    """
    Draw the zero-order vs first-order Markov corridor-choice models as two node rows.

    Each row shows three corridor states (C_{t-1}, C_t, C_{t+1}) with transition arrows
    annotated by probability: the top row (zero-order) splits 0.5/0.5 left/right; the
    bottom row (first-order) is deterministic (p=0 one way, p=1 the other), illustrating
    that knowing the previous corridor fully determines the next.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    origin : (float, float) or None
        Lower-left anchor of the schematic in data coordinates; defaults to ``(0, 0)``.
    horizontal_color, vertical_color : color
        Fills for horizontal- and vertical-corridor state nodes (alternated across the
        three states).
    y_offset : float
        Vertical gap between the zero-order (top) and first-order (bottom) rows.
    x_offset : float
        Horizontal spacing between successive state nodes.
    radius : float
        State-node circle radius.
    arrow_color : color
        Color of the transition-probability arrows and labels.
    markov_colors : list of color or None
        Two title colors ``[zero_order, first_order]``; defaults to pink/grey.
    """
    # plot three circles for each model:
    if origin is None:
        origin = (0, 0)  # default origin at (0, 0)
    if markov_colors is None:
        markov_colors = ["tab:pink", "tab:grey"]

    ys = [origin[1], origin[1]-y_offset]
    xs = [origin[0], origin[0]+x_offset, origin[0]+x_offset*2]
    colors= [horizontal_color, vertical_color, horizontal_color]  # colors for the circles
    texts = ["${C}_{t-1}$", "${C}_{t}$", "${C}_{t+1}$"]
    # plot the circles
    for y in ys:
        for k, x in enumerate(xs):
            ax.add_patch(mpatches.Circle((x, y), radius=radius, color=colors[k]))
            ax.text(x, y, texts[k], fontsize=TICK_SIZE, ha="center", va="center", color="white", zorder=10)
    arrow_length = x_offset-2*radius
    # Add arrows for probabilities
    # zero-order Markov
    # draw_arrow(ax, 0.5, n_nodes - 0.5, -1, 0, 1, arrow_color)
    draw_arrow(ax, xs[1]-radius, ys[0], dx=-arrow_length, dy=0, w=0.2, color=arrow_color,)
    ax.text(xs[1]-x_offset/2, ys[0]+y_offset/8, s="$p=0.5$", fontsize=TICK_SIZE, ha="center", va="bottom", color=arrow_color, zorder=10)
    draw_arrow(ax, xs[1]+radius, ys[0], dx=arrow_length, dy=0, w=0.2, color=arrow_color)
    ax.text(xs[1]+x_offset/2, ys[0]+y_offset/8, s="$p=0.5$", fontsize=TICK_SIZE, ha="center", va="bottom", color=arrow_color, zorder=10)
    # first-order Markov
    draw_arrow(ax, xs[1]-radius, ys[1], dx=-arrow_length, dy=0, w=0.1, color=arrow_color, alpha=0.1)
    ax.text(xs[1]-x_offset/2, ys[1]+y_offset/8, s="$p=0$", fontsize=TICK_SIZE, ha="center", va="bottom", color=arrow_color, zorder=10)
    draw_arrow(ax, xs[1]+radius, ys[1], dx=arrow_length, dy=0, w=0.4, color=arrow_color)
    ax.text(xs[1]+x_offset/2, ys[1]+y_offset/8, s="$p=1$", fontsize=TICK_SIZE, ha="center", va="bottom", color=arrow_color, zorder=10)

    # add titles
    ax.text(xs[0]-radius, ys[0]+y_offset/2, s="Zero-order Markov", ha="left", va="top", color=markov_colors[0], zorder=10)
    ax.text(xs[0]-radius, ys[1]+y_offset/2, s="First-order Markov", ha="left", va="top", color=markov_colors[1], zorder=10)
    # format axis
    ax.set_xlim(xs[0]-2*radius, xs[-1]+2*radius)
    ax.set_ylim(ys[-1]-2*radius, ys[0]+2*radius)
    ax.set_aspect("equal", adjustable="box")
    ax.axis('off')


def format_mask_d_zones(ax, vspan_colors=None, alpha=0.2, zorder=-5, **kwargs):
    """
    Shade the x-axis of a Mask-D corridor-sequence plot into its four topological zones.

    Adds background ``axvspan`` bands and labels for Biclique 1 (corridors 0.5-7.5), the
    Bottleneck (7.5-8.5), Biclique 2 (8.5-16.5), and Out (16.5-17.5). The x-units are
    display-position corridor indices in the reduced Mask-D corridor order.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis whose x-axis is the corridor sequence; modified in place.
    vspan_colors : list of color or None
        Four band colors ``[biclique1, bottleneck, biclique2, out]``; defaults to
        olive/red/olive/red.
    alpha : float
        Band transparency.
    zorder : int
        Draw order of the bands (negative keeps them behind data).
    **kwargs
        Forwarded to ``axvspan`` and the zone-label ``text`` calls.
    """
    if vspan_colors is None:
        vspan_colors = ["tab:olive", "tab:red", "tab:olive", "tab:red"]

    ax.axvspan(0.5, 7.5, alpha=alpha, facecolor=vspan_colors[0], zorder=zorder,  **kwargs)
    ax.text(0.1, 0.1, "Biclique 1", fontsize=TICK_SIZE, ha="left", va="top", transform=ax.transAxes, color=vspan_colors[0], **kwargs)
    ax.axvspan(7.5, 8.5, alpha=alpha, facecolor=vspan_colors[1], zorder=zorder, **kwargs)
    ax.text(0.4, 0.1, "Bottleneck", fontsize=TICK_SIZE, ha="left", va="top", transform=ax.transAxes, color=vspan_colors[1], **kwargs)
    ax.axvspan(8.5, 16.5, alpha=alpha, facecolor=vspan_colors[2], **kwargs)
    ax.text(0.6, 0.1, "Biclique 2", fontsize=TICK_SIZE, ha="left", va="top", transform=ax.transAxes, color=vspan_colors[2], **kwargs)
    ax.axvspan(16.5, 17.5, alpha=alpha, facecolor=vspan_colors[3], zorder=zorder, **kwargs)
    ax.text(0.95, 0.1, "Out", fontsize=TICK_SIZE, ha="left", va="top", transform=ax.transAxes, color=vspan_colors[3], **kwargs)


def format_path_graph_zones(ax, vspan_colors=None, alpha=0.2, zorder=-5, **kwargs):
    """
    Shade the x-axis of a path-graph plot into Home / corridor / Out zones.

    Adds background ``axvspan`` bands for Home (0.5-1.5), the intervening corridors
    (1.5-8.5), and Out (8.5-9.5), labeling the Home and Out ends. X-units are corridor
    positions along the path graph.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to shade; modified in place.
    vspan_colors : list of color or None
        Three band colors ``[home, middle, out]``; defaults to red/olive/red.
    alpha : float
        Band transparency.
    zorder : int
        Draw order of the bands (negative keeps them behind data).
    **kwargs
        Forwarded to ``axvspan`` and the label ``text`` calls.
    """
    if vspan_colors is None:
        vspan_colors = ["tab:red", "tab:olive", "tab:red"]

    ax.axvspan(0.5, 1.5, alpha=alpha, facecolor=vspan_colors[0], zorder=zorder, **kwargs)
    ax.text(0.01, 0.8, "Home", fontsize=TICK_SIZE, ha="left", va="top", transform=ax.transAxes, color=vspan_colors[0], **kwargs)
    ax.axvspan(1.5, 8.5, alpha=alpha, facecolor=vspan_colors[1], zorder=zorder, **kwargs)
    ax.axvspan(8.5, 9.5, alpha=alpha, facecolor=vspan_colors[2], zorder=zorder, **kwargs)
    ax.text(0.9, 0.8, "Out", fontsize=TICK_SIZE, ha="left", va="top", transform=ax.transAxes, color=vspan_colors[2], **kwargs)


def plot_hole_decision_schematic(ax,
    bottom_color="tab:blue",
    top_color="tab:orange",
    linewidth=LW_HAIRLINE,
    arrow_width=0.24,
):
    """
    Draw an exploded isometric schematic of the four decisions at a single hole.

    The horizontal (``bottom_color``) corridor and the vertical (``top_color``) corridor
    are drawn as thin slabs running along the maze's two ground axes (slope +/-1/4, the
    same isometric as :func:`plot_schematic_3d_maze`) and lifted apart so the two levels
    read unambiguously.  Each corridor carries a centered round hole, drawn as a horizontal
    ellipse -- a round opening in a horizontal plane projects that way under this isometric
    -- and the two holes are joined by a dashed "climb through hole" link.  Four arrows
    emanate symmetrically from the holes: proceed / retract along the lower corridor and
    turn-left / turn-right along the upper corridor.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis; set to equal aspect and turned off by this function.
    bottom_color, top_color : color
        Fills and labels for the horizontal (lower) and vertical (upper) corridor domains.
    linewidth : float
        Linewidth of the slab and hole edges.
    arrow_width : float
        Shaft width of the four move arrows.

    Notes
    -----
    Geometry is in arbitrary data units chosen for visual clarity; the +/-1/4 ground-axis
    slope matches the physical-maze isometric of :func:`plot_schematic_3d_maze`.
    """
    ax.set_aspect("equal")
    ax.axis("off")

    def _arrow(start, vec, color, zorder):
        draw_arrow(ax, start[0], start[1], vec[0], vec[1], w=arrow_width, color=color, zorder=zorder)
        return start + vec

    # maze ground axes: slope +/-1/4 as in plot_schematic_3d_maze (4:1 rhombus)
    d1 = np.array([1.0, -0.25])   # horizontal corridor (lower) long axis, down-right
    d2 = np.array([1.0, 0.25])    # vertical corridor (upper) long axis, up-right
    UP = np.array([0.0, 3.0])     # exploded vertical lift between levels
    HALF, WID = 2.2, 0.8          # corridor half-length / width (multipliers on d1, d2)

    # ---- lower horizontal corridor, long along d1, hole + arrows centered ----
    hole_lo = np.array([2.8, 1.4])
    lo0 = hole_lo - d1 * HALF - d2 * (WID / 2)
    lower = [lo0, lo0 + d1 * 2 * HALF, lo0 + d1 * 2 * HALF + d2 * WID, lo0 + d2 * WID]
    ax.add_patch(mpatches.Polygon(lower, closed=True, facecolor=bottom_color, alpha=0.22,
                                  edgecolor="k", linewidth=linewidth, zorder=0))
    draw_ellipse(ax, hole_lo[0], hole_lo[1], 0.6, 0.3, color="white", alpha=0.95,
                 edgecolor="black", linewidth=linewidth * 1.5, zorder=2)
    tip = _arrow(hole_lo + d1 * 0.5, d1 * 1.4, bottom_color, 3)
    ax.text(tip[0] + 0.12, tip[1] - 0.1, "proceed", fontsize=FONT_SIZE, ha="center", va="top", color=bottom_color)
    tip = _arrow(hole_lo - d1 * 0.5, -d1 * 1, bottom_color, 3)
    # port letter "H" sits at the arrow's end (the home the retract move leads to); "retract" labels the move
    ax.text(tip[0] - 0.12, tip[1] + 0.06, "H", fontsize=FONT_SIZE, ha="right", va="center", color=bottom_color)
    ax.text(tip[0] - 0.12, tip[1] - 0.30, "retract", fontsize=FONT_SIZE, ha="right", va="top", color=bottom_color)
    ax.text(hole_lo[0], hole_lo[1] - 1.15, "horizontal corridor", fontsize=TICK_SIZE,
            ha="center", va="top", color=bottom_color, alpha=0.85)

    # ---- climb link (dashed, up through both holes) ----
    center_up = hole_lo + UP
    ax.add_patch(mpatches.FancyArrowPatch((hole_lo[0], hole_lo[1] + 0.2),
                                          (center_up[0], center_up[1] - 0.2),
                                          arrowstyle="-|>", mutation_scale=7, lw=linewidth * 1.8,
                                          ls=(0, (2, 1.5)), color="0.35", zorder=1))
    ax.text(center_up[0] + 0.2, (hole_lo[1] + center_up[1]) / 2, "climb\nthrough\nhole",
            fontsize=TICK_SIZE, ha="left", va="center", color="0.35")

    # ---- upper vertical corridor, long along d2, hole + arrows centered ----
    up0 = center_up - d2 * HALF - d1 * (WID / 2)
    upper = [up0, up0 + d2 * 2 * HALF, up0 + d2 * 2 * HALF + d1 * WID, up0 + d1 * WID]
    ax.add_patch(mpatches.Polygon(upper, closed=True, facecolor=top_color, alpha=0.22,
                                  edgecolor="k", linewidth=linewidth, zorder=3))
    draw_ellipse(ax, center_up[0], center_up[1], 0.6, 0.3, color="white", alpha=0.95,
                 edgecolor="black", linewidth=linewidth * 1.5, zorder=4)
    tip = _arrow(center_up + d2 * 0.5, d2 * 1.4, top_color, 5)
    # port letter "O" sits at the arrow's end (the out the turn-left move leads to); "turn left" labels the move
    ax.text(tip[0] + 0.15, tip[1] + 0.06, "O", fontsize=FONT_SIZE, ha="left", va="center", color=top_color)
    ax.text(tip[0] + 0.15, tip[1] - 0.30, "turn left", fontsize=FONT_SIZE, ha="center", va="top", color=top_color)
    tip = _arrow(center_up - d2 * 0.5, -d2 * 1.4, top_color, 5)
    ax.text(tip[0] - 0.15, tip[1] - 0.1, "turn right",
            fontsize=FONT_SIZE, ha="center", va="top", color=top_color)
    ax.text(center_up[0], center_up[1] + 1.25, "vertical corridor", fontsize=TICK_SIZE,
            ha="center", va="bottom", color=top_color, alpha=0.85)

    ax.autoscale_view()
    ax.margins(0.16)


def add_biclique_arrows(ax, y_scale, transitions, start_node=19, maze_size=11, H_circle_x=1, V_circle_x=2,
                         goal_color=plt.cm.RdGy(0.1),  transition_colors=None,
                         n_nodes=9, arrow_width=0.5, head_width=None, head_length=None):
    """
    Overlay transition arrows on a :func:`plot_schematic_d_graph` layout.

    Two modes.  When ``transitions`` is given, draw one colored arrow per
    ``(start, end)`` **display-position** index pair (positions from
    :func:`node_position`, the same index space as the reduced transition matrices) — used
    for arbitrary transition sets such as the off-path biclique transitions.  Otherwise
    fall back to the legacy single-start contrast: from a fixed start corridor, three
    control arrows (toward control corridors) and one goal arrow (toward the
    bottleneck-leading corridor), color-coded so the goal stands out, with the start column
    and vertical offsets chosen by which biclique ``start_node`` belongs to.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis already holding a :func:`plot_schematic_d_graph` layout.
    y_scale : float
        Vertical row spacing, matching the underlying d-graph.
    control_colors : list of color or None
        Three control-arrow colors (legacy mode); defaults to a light-to-mid RdGy ramp.
    start_node : int
        Corridor index the arrows originate from (legacy mode). ``> maze_size - 1`` means a
        vertical corridor (arrows point left to the horizontal column), else a horizontal
        corridor (arrows point right). ``19`` selects the first-biclique offsets, otherwise
        the second-biclique offsets.
    maze_size : int
        Corridors per orientation (11), used to decide the start column (legacy mode).
    H_circle_x, V_circle_x : float
        Data x-coordinates of the horizontal and vertical columns.
    goal_color : color
        Color of the single goal arrow (legacy mode); defaults to dark RdGy.
    transitions : iterable of (int, int) or None
        Display-position ``(start, end)`` index pairs to draw arrows for.  When given, the
        general path is taken and the legacy ``start_node`` fan is skipped.
    transition_colors : iterable of color or None
        One color per entry in ``transitions`` (e.g. matched to the line panel).
    n_nodes : int, default 9
        Nodes per column, forwarded to :func:`node_position`.
    arrow_width : float, default 0.5
        Arrow body width; smaller (~0.1) gives thin, small-headed arrows.
    head_width, head_length : float or None
        When given (transitions mode), draw FancyArrow arrows whose head is sized
        independently of the shaft — e.g. a wide head on a thin body. ``None`` keeps
        the equilateral arrow whose head scales with ``arrow_width``.

    Notes
    -----
    The ``start_node == 19`` special case encodes the first biclique's key corridor in the
    reduced Mask-D corridor order; other values fall through to the second-biclique layout.
    """
    for (start, end), color in zip(transitions, transition_colors):
        x0, y0 = node_position(start, H_circle_x, V_circle_x, y_scale, n_nodes)
        x1, y1 = node_position(end, H_circle_x, V_circle_x, y_scale, n_nodes)
        draw_arrow(ax, x=x0, y=y0, dx=x1 - x0, dy=y1 - y0,
                   w=arrow_width, color=color, zorder=20,
                   head_width=head_width, head_length=head_length)


def node_position(node, H_circle_x=1, V_circle_x=3, y_scale=1, n_nodes=5):
    """
    (x, y) of a corridor node in the two-column corridor schematic.

    Even node index -> left (``H_circle_x``) column, odd -> right (``V_circle_x``)
    column; the row descends with ``node // 2`` (node 0 at the top).  Shared by the
    signal/adjacency schematics and the Mask-D transition arrows so they all use one
    index -> position convention.
    """
    x = H_circle_x if node % 2 == 0 else V_circle_x
    y = (n_nodes - 0.5 - node // 2) * y_scale
    return x, y


def plot_corridor_transition_schematic(ax, start, goal, controls, goal_color, control_colors, adj=None,
                                       n_nodes=9, H_circle_x=1, V_circle_x=2, y_scale=0.5, radius=0.18,
                                       node_color="tab:grey", bottleneck_color="red", linewidth=LW_DATA,
                                       column_colors=None, outbound=True,
                                       plot_direction_arrows=False, arrow_color="tab:purple",
                                       grey_uninvolved_nodes=False, red_outline_orange=False):
    """
    Schematic of a corridor transition contrast in the parity node layout.

    Draws the corridor nodes (and, if ``adj`` is given, the corridor graph edges) at
    their :func:`node_position` coordinates, then arrows from the ``start`` corridor to
    the ``goal`` (in ``goal_color``) and each ``control`` (in ``control_colors``).  All
    indices are display positions in the reduced corridor order — the same space as
    :func:`manhattan_maze.analysis.get_d_transition_matrices` and the endotaxis
    schematic — so any combination of corridors maps naturally.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    start, goal : int
        Display-position indices of the start and goal (bottleneck) corridors.
    controls : list of int
        Display-position indices of the control corridors.
    goal_color : color
    control_colors : list of color
        One per control corridor.
    adj : ndarray or None
        Reduced corridor adjacency (display order) for the backdrop graph edges; if
        None, only nodes and arrows are drawn.
    n_nodes : int, default 9
        Nodes per column (2*n_nodes total).
    column_colors : (left, right) or None
        Per-column node fill colors (even index -> left, odd -> right). If None,
        all nodes use ``node_color``.
    outbound : bool, default True
        Travel direction for the purple direction arrows only (see
        ``plot_direction_arrows``); ``False`` reverses them for the homebound return.
        The transition arrows always emanate from ``start`` regardless of this flag.
    plot_direction_arrows : bool, default False
        If True, overlay the purple ``H->``/``->O`` navigation arrows via
        :func:`add_direction_arrows`, reversed for ``outbound=False`` (homebound).
    arrow_color : color, default 'tab:purple'
        Color of the direction arrows.
    grey_uninvolved_nodes : bool, default False
        If True (and ``column_colors`` is given), only the nodes involved in the
        contrast (``start``, ``goal`` and ``controls``) keep their column color;
        every other node is drawn in ``node_color`` (grey).
    red_outline_orange : bool, default False
        If True, involved nodes in the orange (odd-index) column get a red outline
        (keeping their orange fill); the goal keeps its ``bottleneck_color`` outline.
    """
    involved = {start, goal, *controls}

    def fill_color(node):
        if column_colors is None or (grey_uninvolved_nodes and node not in involved):
            return node_color
        return column_colors[0] if node % 2 == 0 else column_colors[1]

    if adj is not None:
        plot_edges_based_on_adj_mat(ax, adj, H_circle_x=H_circle_x, V_circle_x=V_circle_x,
                                    y_scale=y_scale, n_nodes=n_nodes, edge_color=node_color, linewidth=linewidth / 2)
    # corridor nodes
    for node in range(n_nodes * 2):
        x, y = node_position(node, H_circle_x, V_circle_x, y_scale, n_nodes)
        ax.add_patch(mpatches.Circle((x, y), radius=radius, color=fill_color(node), zorder=15))
    # outline the start (black) and goal/bottleneck (bottleneck_color); optionally
    # give the orange (odd-column) involved nodes a red outline instead (keeps fill).
    outlines = {start: "black", goal: bottleneck_color}
    if red_outline_orange:
        for node in involved:
            if node % 2 == 1:
                outlines[node] = "red"
    for node, edgecolor in outlines.items():
        x, y = node_position(node, H_circle_x, V_circle_x, y_scale, n_nodes)
        ax.add_patch(mpatches.Circle((x, y), radius=radius + 0.02, facecolor=fill_color(node),
                                     edgecolor=edgecolor, linewidth=linewidth, zorder=16))
    # arrows always emanate from the start node toward goal + controls (the choice
    # point); the travel direction is conveyed separately by the purple arrows.
    x0, y0 = node_position(start, H_circle_x, V_circle_x, y_scale, n_nodes)
    for node, color in [(goal, goal_color)] + list(zip(controls, control_colors)):
        x1, y1 = node_position(node, H_circle_x, V_circle_x, y_scale, n_nodes)
        draw_arrow(ax, x=x0, y=y0, dx=x1 - x0, dy=y1 - y0, w=0.5, color=color, zorder=20)
    # purple H->/->O direction arrows (reversed for homebound), matching panel A
    if plot_direction_arrows:
        add_direction_arrows(ax, H_circle_x, V_circle_x, y_scale, n_nodes, radius,
                             outbound=outbound, arrow_color=arrow_color)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def plot_circle_with_signal_values(ax, signals, H_circle_x=1, V_circle_x=3, y_scale=1, n_nodes=5, radius=0.2, cmap=plt.cm.plasma):
    """
    Color the two-column corridor nodes by a per-node scalar signal.

    Places ``2 * n_nodes`` node circles at their :func:`node_position` coordinates and
    fills each with a colormap value taken from ``signals`` (e.g. a learned goal signal
    or value function over corridors).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    signals : sequence of float
        Per-node values in [0, 1] (already normalized for the colormap); length must be
        ``2 * n_nodes``, indexed in the same order as :func:`node_position`.
    H_circle_x, V_circle_x : float
        Data x-coordinates of the two node columns.
    y_scale : float
        Vertical row spacing.
    n_nodes : int
        Nodes per column.
    radius : float
        Node circle radius.
    cmap : matplotlib.colors.Colormap
        Colormap mapping signal values to fill colors.

    Raises
    ------
    AssertionError
        If any signal exceeds 1 (not normalized) or ``len(signals) != 2 * n_nodes``.
    """
    assert max(signals) <= 1, "signals must be normalized for cmap"
    assert len(signals) == n_nodes*2, f"Length of signals {len(signals)} must be equal to n_nodes*2 {n_nodes*2}"
    # normalize signal to 0 and 1 for colormap
    colors = get_normalized_color_seq(signals, cmap=cmap)

    for signal_index in range(n_nodes * 2):
        x, y = node_position(signal_index, H_circle_x, V_circle_x, y_scale, n_nodes)
        color = colors[signal_index]
        ax.add_patch(mpatches.Circle((x, y), radius=radius, facecolor=color, edgecolor=color, alpha=1,
                                     zorder=20))


def plot_edges_based_on_adj_mat(ax, adj, H_circle_x=1, V_circle_x=3, y_scale=1, n_nodes=5, edge_color="tab:gray", linewidth=LW_DATA):
    """
    Draw corridor-graph edges from an adjacency matrix in the two-column node layout.

    For every below-diagonal nonzero entry of ``adj`` (treated as an undirected edge),
    connects the two corridor nodes at their :func:`node_position` coordinates. Used as the
    backdrop graph for the signal/transition schematics.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    adj : ndarray, shape (2*n_nodes, 2*n_nodes)
        Symmetric corridor adjacency; only the strict lower triangle is read so each edge
        is drawn once. Row/column indices are node indices in :func:`node_position` order.
    H_circle_x, V_circle_x : float
        Data x-coordinates of the two node columns.
    y_scale : float
        Vertical row spacing.
    n_nodes : int
        Nodes per column.
    edge_color : color
    linewidth : float
    """
    # convert the values into col and row for the holes.
    # get the below diagnoal of the adjacency matrix
    adj_below = np.tril(adj, k=-1)
    # get the indices for plotting
    row_indices, col_indices = np.where(adj_below == 1)
    for row, col in zip(row_indices, col_indices):
        x1, y1 = node_position(row, H_circle_x, V_circle_x, y_scale, n_nodes)
        x2, y2 = node_position(col, H_circle_x, V_circle_x, y_scale, n_nodes)
        ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=linewidth, zorder=10)


def plot_goal_signal(ax, goal_signal, xs=None, plot_values=False, cmap=plt.cm.plasma, **kwargs):
    """
    Plot the per-corridor goal signal as a colored line-and-scatter trace.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    goal_signal : sequence of float
        Goal signal per corridor (log scale, as labeled on the y-axis), ordered along the
        corridor sequence.
    xs : sequence or None
        X positions; defaults to ``0..len(goal_signal)-1`` (corridor index).
    plot_values : bool
        If True, annotate each point with its value to two decimals.
    cmap : matplotlib.colors.Colormap
        Colormap used to color the scatter points by signal magnitude.
    **kwargs
        Forwarded to ``ax.plot`` for the connecting line.
    """
    if xs is None:
        xs = np.arange(len(goal_signal))
    ax.plot(xs, goal_signal, **kwargs)
    colors = get_normalized_color_seq(goal_signal, cmap)
    ax.scatter(xs, goal_signal, color=colors, zorder=10, s=MS_AREA_LARGE)
    if plot_values:
        for x, y in zip(xs, goal_signal):
            ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=TICK_SIZE)
    ax.set_xlabel("Corridor")
    ax.set_ylabel("Goal signal (log)")


def plot_exponential_schematic(ax, func, parameter_names=None, equation_string=None, xs=None,
                               xlabel=None, ylabel=None,**kwargs):
    """
    Plot an illustrative exponential learning curve annotated with its model equation.

    Draws ``func`` over ``xs`` with fixed illustrative parameters (D0=50, D_inf=5, k=0.3)
    and labels the asymptote and initial value on the y-axis, rendering the manuscript
    equation as a LaTeX annotation. This is a schematic of the model form (Eq. 2), not a
    fit to data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    func : callable
        Exponential model ``func(xs, D_infty, D_0, k)`` (e.g. ``utils.exponential_func``);
        traverse number ``b`` is 1-based, so ``b = 1`` gives no decay.
    parameter_names : list of str or None
        LaTeX labels ``[D_0, D_inf, delta]``; the first two label the y-ticks. Defaults to
        ``[$D_0$, $D_\\infty$, $\\delta$]``.
    equation_string : str or None
        LaTeX equation annotation; defaults to the manuscript duration-decay equation.
    xs : ndarray or None
        Traverse-number grid; defaults to ``linspace(1, 20, 100)``.
    xlabel, ylabel : str or None
        Axis labels; default to "Traverse #" and "Duration".
    **kwargs
        Forwarded to ``ax.plot``.
    """
    if parameter_names is None:
        parameter_names = [r"$D_0$", r"$D_{\infty}$", r"$\delta$"]
    if equation_string is None:
        equation_string = r"$D_{a,b} = D_{\infty} + \left(D_{0} - D_{\infty}\right)\exp\left[-\delta(b - 1)\right] + \xi^D_{a,b}$"
    if xs is None:
        xs = np.linspace(1, 20, 100)

    if xlabel is None:
        xlabel="Traverse #"
    if ylabel is None:
        ylabel="Duration"
    # exponential function utils.exponential
    d0 = 50
    d_infty = 5
    k = 0.3
    ys = func(xs, d_infty, d0, k)
    ax.plot(xs, ys, **kwargs)
    ax.text(1, 0.9, equation_string, ha="right", va="top", fontsize=TICK_SIZE, transform=ax.transAxes)
    format_xs_ys(ax, xs, ylim=d0*1.2,xlabel=xlabel, ylabel=ylabel)
    ax.set_xticklabels([])
    ax.set_yticks([d_infty, d0])
    ax.set_yticklabels([parameter_names[1], parameter_names[0]])


def plot_d2_session_timeline(ax, origin=None, mask_list=None, increment=None,
                             color_dict=None):
    """
    Draw the Day-2 counterbalancing design as a text grid of mask sequences per group.

    Each row is one experimental group's mask schedule: a fixed Mask A on Day 1, then a
    Day-2 permutation rendered in the pattern ``x y x z`` (where ``(x, y, z)`` is one
    permutation of ``mask_list``), so every group sees a different Day-2 order. Repeated
    columns (positions 1 and 3) are boxed to highlight the within-day repeat.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    origin : (float, float) or None
        Anchor (x, y) for the grid in data coordinates; defaults to ``(1, 0)``.
    mask_list : list of str or None
        The three mask labels to permute; defaults to ``["A", "B", "C"]``. Must have
        exactly three entries (each row unpacks a 3-tuple permutation).
    increment : (float, float) or None
        ``(row_step, col_step)`` spacing in data units; defaults to ``[0.1, 0.06]``.
    color_dict : dict or None
        Map mask label -> color; defaults to A=blue, B=red, C=green.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    Groups are numbered in descending order from top to bottom so "Group 1" is the last
    permutation row, matching the manuscript's group table.
    """
    if origin is None:
        origin = (1, 0)
    if mask_list is None:
        mask_list = ["A", "B", "C"]
    if increment is None:
        increment = [0.1, 0.06]
    if color_dict is None:
        color_dict = {"A": "tab:blue", "B": "tab:red", "C": "tab:green"}

    all_permutations = list(permutations(mask_list))
    for i, (x, y, z) in enumerate(all_permutations):
        # each line is the text of xyxz format
        # Add group count
        ax.text(origin[0] - increment[1] * 2, i * increment[0], s=f"Group {len(all_permutations) - i}",
                fontsize=TICK_SIZE, horizontalalignment="center")
        # Add text for Day 1 (Mask A)
        ax.text(origin[0] - increment[1], i * increment[0], s="A", fontsize=TICK_SIZE,
                horizontalalignment="center", color="tab:grey")
        ax.text(origin[0], i * increment[0], s=f"{x}", color=color_dict[x],
                fontsize=TICK_SIZE, horizontalalignment="center")
        ax.text(origin[0] + increment[1], i * increment[0], s=f"{y}", color=color_dict[y],
                fontsize=TICK_SIZE, horizontalalignment="center")
        ax.text(origin[0] + 2 * increment[1], i * increment[0], s=f"{x}", color=color_dict[x],
                fontsize=TICK_SIZE, horizontalalignment="center")
        ax.text(origin[0] + 3 * increment[1], i * increment[0], s=f"{z}", color=color_dict[z],
                fontsize=TICK_SIZE, horizontalalignment="center")
        # Add a horizontal dashed line
        ax.plot([origin[0] - 0.5 * increment[1], 3.5 * increment[1]],
                [(i - increment[0]) * increment[0], (i - increment[0]) * increment[0]], linestyle="--", color="black",
                linewidth=LW_HAIRLINE)

    for k in range(4):
        ax.text(origin[0] + k * increment[1], (i + 1) * increment[0], s=f"Day 2.{k + 1}", fontsize=FONT_SIZE,
                horizontalalignment="center", rotation=45)

    ax.text(origin[0]-increment[1], (i + 1) * increment[0], s="Day 1", fontsize=FONT_SIZE, horizontalalignment="center",
            rotation=45)
    ax.axis("off")
    ax.set_xlim(left=-increment[1], right=5 * increment[1])
    ax.set_ylim(bottom=-increment[0] / 2, top=(len(all_permutations) + 3) * increment[0])

    # Add two rectangles to show 1 and 3 are repeats
    ax.add_patch(mpatches.Rectangle((-increment[1] / 2, -increment[1] / 2), increment[1],
                                    (len(all_permutations)) * increment[0], fill=False, color="black", alpha=0.3))
    ax.add_patch(mpatches.Rectangle((1.5 * increment[1], -increment[1] / 2), increment[1],
                                    (len(all_permutations)) * increment[0], fill=False, color="black", alpha=0.3))
    return ax


def plot_ablation_timeline(ax, increments=None, rotation=30,
                           condition_color_dict=None):
    """
    Draw the olfactory-bulb ablation experimental design as a labeled condition table.

    One row per condition (e.g. Ablated / Sham / Rest, excluding "Recovered"), with columns
    for treatment (ZnSO4 vs saline) and testing windows (<1 week, >1 month), each cell
    filled with the condition's text and color.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    increments : (float, float) or None
        ``(col_step, row_step)`` spacing in data units; defaults to ``(0.5, 0.02)``.
    rotation : float
        Rotation (degrees) of the column header text.
    condition_color_dict : dict or None
        Map condition name -> color; defaults to the shared ``ob_condition_color_dict``.
        The "Recovered" key is excluded from the rows.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    AssertionError
        If ``increments`` is given but is not a length-2 list/tuple.
    """
    if condition_color_dict is None:
        condition_color_dict = ob_condition_color_dict
    if increments is None:
        increments = (0.5, 0.02)
    else:
        # make sure increments is a tuple of length 2
        assert isinstance(increments, (list, tuple)) and len(increments) == 2, "increments must be a tuple or list of length 2"

    xi, yi = increments[0], increments[1]

    group_names = [key for key in condition_color_dict.keys() if key != "Recovered"]
    group_colors = [condition_color_dict[key] for key in group_names]
    for i, group in enumerate(group_names):
        ax.text(-0.2 * xi, i * yi, s=f"{group}", color=group_colors[i], fontsize=TICK_SIZE,
                horizontalalignment="center")
        # Add a horizontal dashed line
        ax.plot([-0.5 * xi, 4 * increments[0]],
                [(i - yi) * yi, (i - yi) * increments[1]], linestyle="--",
                color="black", linewidth=LW_HAIRLINE)

    # Add the timeline for each group
    ## Ablated
    ax.text(xi, 0, s=r"$\mathregular{ZnSO_4}$", fontsize=TICK_SIZE, horizontalalignment="center")
    ax.text(2 * xi, 0, s="Yes", fontsize=TICK_SIZE, horizontalalignment="center")
    ax.text(3 * xi, 0, s="Yes", fontsize=TICK_SIZE, horizontalalignment="center")

    ## Sham
    ax.text(xi, yi, s="Saline", fontsize=TICK_SIZE, horizontalalignment="center")
    ax.text(2 * xi, yi, s="Yes", fontsize=TICK_SIZE, horizontalalignment="center")
    ax.text(3 * xi, yi, s="Yes", fontsize=TICK_SIZE, horizontalalignment="center")

    ## Rest
    ax.text(xi, 2 * yi, s=r"$\mathregular{ZnSO_4}$", fontsize=TICK_SIZE,
            horizontalalignment="center")
    ax.text(2 * xi, 2 * yi, s="No", fontsize=TICK_SIZE, horizontalalignment="center")
    ax.text(3 * xi, 2 * yi, s="Yes", fontsize=TICK_SIZE, horizontalalignment="center")

    # add column heading
    ax.text(-0.2 * xi, (len(group_names)-0.5) * yi, s="Group", rotation=rotation, fontsize=TICK_SIZE,
            horizontalalignment="center")
    # slant the text to fit the space
    ax.text(xi, (len(group_names)-0.5) * yi, s="Treatment", rotation=rotation, fontsize=TICK_SIZE,
            horizontalalignment="center")
    ax.text(2 * xi, (len(group_names)-0.5) * yi, s="$<$1 week", rotation=rotation, fontsize=TICK_SIZE,
            horizontalalignment="center")
    ax.text(3 * xi, (len(group_names)-0.5) * yi, s="$>$1 month", rotation=rotation, fontsize=TICK_SIZE,
            horizontalalignment="center")

    # Add a vertical line
    ax.plot([0.3 * xi, 0.3 *xi], [0, len(group_names) * yi], linestyle="--",
            color="black", linewidth=LW_HAIRLINE)

    # format axis
    ax.axis("off")
    ax.set_xlim(left=-xi, right=4 * xi)
    ax.set_ylim(bottom=-yi / 2, top=(len(group_names) + 1) * yi)
    return ax
