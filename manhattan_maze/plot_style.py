"""Matplotlib style, drawing primitives, axis/legend/significance/colour formatting.

Split out of plot_utils.py.
"""
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation
from manhattan_maze.plot_constants import (FONT_SIZE, LABEL_SIZE, LW_DATA, LW_HAIRLINE,
                                            MS_AREA_LARGE, MS_PT_SMALL, TICK_SIZE, TITLE_PAD,
                                            Z_REFERENCE)
from manhattan_maze.plot_utils import bout_type_color_dict

__all__ = ['set_style', 'draw_vertical_parallelogram', 'draw_rhombus', 'draw_ellipse', 'draw_arrow', 'add_direction_arrows', 'set_distance_plot_yaxis', 'format_xs_ys', 'plot_illustrative_cbar', 'plot_phase_lines', 'set_corridor_steps_axis', 'add_lines_to_matrix_plot', 'add_squares_to_matrix_plot', 'add_symbol_for_p_value', 'add_signficance_bracket', 'format_yaxis_color', 'add_letter_labels', 'add_panel_title', 'get_legend_objects_as_dict', 'distance_reward_marker', 'session_distance_plot_label', 'axis_format_with_color', 'create_legend_for_double_axes', 'gap_to_str', 'color_bouts_from_indices', 'get_normalized_color_seq', 'format_value_str']

def set_style():
    """
    Apply the manuscript-wide matplotlib style to the global ``rcParams``.

    Sets sans-serif (Arial) annotation, Computer-Modern maths, 300 dpi, and the
    module font sizes (``FONT_SIZE`` / ``TICK_SIZE``).  Call this once at the top
    of any figure-producing script before plotting.

    Typography, and why it is split this way
    ----------------------------------------
    *Annotation* -- axis labels, tick numbers, legends, panel letters, in-panel
    text -- is sans-serif, on reviewer request, because it is easier to read at
    small size: Liberation Sans has a 22.5% larger x-height than ``cmr10`` at the
    same point size and the same advance width to within 0.2%, so the legibility
    is gained without widening anything (measured page-size change across all
    figures: -0.11% to 0.00%).

    *Mathematics* stays Computer Modern.  ``mathtext.fontset`` is deliberately
    left at ``'cm'`` rather than switched to a sans fontset, so every ``$...$``
    symbol -- Latin and Greek alike -- is unchanged, and matches the
    Computer-Modern inline maths of ``main.tex``.  Rebuilding the maths font
    instead (``mathtext.fontset='custom'`` with ``mathtext.it='cmmi10'``) looks
    equivalent but is not: ``cmmi10.ttf`` carries a 148-entry legacy TeX cmap with
    no Unicode entries for Greek, so ``\\delta`` and friends silently fall through
    to a substitute italic while ``D`` and ``E`` do not.

    Notes
    -----
    ``axes.formatter.use_mathtext`` is what routes numeric tick labels through the
    maths font, so it must be off for tick digits to pick up the sans family.  That
    is safe here only because no figure uses a log axis or scientific-notation
    offset -- log formatters emit mathtext regardless of this flag.  Re-enable it
    and tick digits revert to serif.

    ``axes.unicode_minus`` is now True.  It was False as a ``cmr10`` workaround
    (cmr10 has no U+2212); with mathtext off the minus sign would otherwise
    degrade to an ASCII hyphen.

    These settings were previously applied at import time, which mutated global
    matplotlib state as a side effect of ``import plot_utils`` (R6).  They now
    live here so that importing the module is side-effect-free; scripts opt in
    explicitly via ``plot_utils.set_style()``.
    """
    # matplotlib's bundled Computer Modern faces (cmr10, cmmi10, cmsy10, cmex10) carry
    # head.created == head.modified == 0. fontTools reads that as a pre-1970 date, 'corrects'
    # it, and logs two lines about it for every font of every savefig -- 42 lines across the
    # figure set. The correction is right and the embedded font is fine, so this is pure
    # noise, but it buries real output. It reaches us because the two settings below pull
    # every $...$ glyph out of those faces and subset them through fontTools on PDF save.
    # Note this is a logging record, not a warnings warning: -W / PYTHONWARNINGS filters do
    # nothing to it, and it is never deduplicated.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    mpl.rcParams['mathtext.fontset'] = 'cm'      # maths untouched; see docstring
    mpl.rcParams['font.family'] = 'sans-serif'
    # Ordered fallback chain: only the first entry that resolves is ever embedded.
    # All four share Arial metrics (Arial clones Helvetica, Nimbus clones Helvetica;
    # Liberation Sans is Arial-metric), so any substitution among them is layout-safe.
    # DejaVu Sans is deliberately absent: it is 13% wider, and matplotlib already falls
    # back to it when nothing resolves *with a findfont warning* -- listing it would
    # silence that warning and quietly rescale every label.
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans']
    mpl.rcParams['axes.formatter.use_mathtext'] = False
    mpl.rcParams['axes.unicode_minus'] = True
    # Default 3 embeds each glyph as a procedural drawing; Type 42 subsets are compact
    # and are what journal production systems expect.
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['figure.max_open_warning'] = False
    mpl.rcParams['figure.dpi'] = 300
    # Font sizes
    mpl.rcParams['font.size'] = FONT_SIZE
    mpl.rcParams['xtick.labelsize'] = TICK_SIZE
    mpl.rcParams['ytick.labelsize'] = TICK_SIZE
    mpl.rcParams['axes.labelsize'] = FONT_SIZE
    mpl.rcParams['legend.fontsize'] = TICK_SIZE
    # Spine / tick geometry. Matplotlib defaults these to 0.8, which left the axis frame
    # heavier than the 0.5 hairline most data and reference lines are drawn at; pin them to
    # the same tier so the frame stops out-weighing the data it frames.
    mpl.rcParams['axes.linewidth'] = LW_HAIRLINE
    mpl.rcParams['xtick.major.width'] = LW_HAIRLINE
    mpl.rcParams['ytick.major.width'] = LW_HAIRLINE
    mpl.rcParams['xtick.minor.width'] = LW_HAIRLINE
    mpl.rcParams['ytick.minor.width'] = LW_HAIRLINE


def draw_vertical_parallelogram(ax, x,y, l, w, h, color, fill=True, alpha=1, **kwargs):
    """
    draw parallelogram to an axis with two vertical edges parallel to y-axis
    inputs:
        x, y are the coordinates of the bottom left corner
        l is the length of the bottom edge, in x direction
        w is the width of the parallelogram, in y direction, strictly positive
        h is the height of the parallelogram, in y direction, can be negative, so the shape slants in different ways
    """
    xs = [x, x+l, x+l, x]  # bottom left, bottom right, top right, top left
    ys = [y, y+h, y+h+w, y+w] # bottom left, bottom right, top right, top left

    paral = mpatches.Polygon(xy=list(zip(xs, ys)), facecolor=color, fill=fill, alpha=alpha, **kwargs)
    ax.add_patch(paral)


def draw_rhombus(ax, x,y, w, h, color, fill=True, alpha=1,**kwargs):
    '''
    draw rhombus to an axis with two vertical edges parallel to y axis
    inputs:
        x, y are the coordinates of the bottom left corner
        w is the width of the rhombus, in x direction
        h is the height of the rhombus, in y direction
    '''
    xs = [x, x+w/2, x+w, x+w/2] # bottom left, bottom right, top right, top left
    ys = [y, y-h/2, y, y+h/2] # bottom left, bottom right, top right, top left
    rhom = mpatches.Polygon(xy=list(zip(xs, ys)), facecolor=color, fill=fill, alpha=alpha, **kwargs)
    ax.add_patch(rhom)


def draw_ellipse(ax, x, y, w, h, color, alpha=1,**kwargs):
    '''
    draw ellipse to an axis (no slanted axis
    inputs:
        x, y are the coordinates of the center
        w is the width of the ellipse, in x direction
        h is the height of the ellipse, in y direction
    '''
    ellipse = mpatches.Ellipse((x, y), w, h, facecolor=color, alpha=alpha, **kwargs)
    ax.add_patch(ellipse)


def draw_arrow(ax, x, y, dx, dy, w, color, alpha=1, head_width=None, head_length=None, **kwargs):
    '''
    Draw arrow using matplotlib's patches
    inputs:
        x, y are the coordinates of the starting point
        dx, dy are the change in x and y, respectively
        w is the width of the arrow (shaft/body width)
        head_width, head_length: if either is given, draw a FancyArrow whose head
            can be sized independently of the shaft (wider head, thinner body);
            otherwise fall back to the equilateral mpatches.Arrow (head scales with w).
    '''
    if head_width is None and head_length is None:
        arrow = mpatches.Arrow(x, y, dx, dy, width=w, color=color, fill=True, alpha=alpha, **kwargs)
    else:
        arrow = mpatches.FancyArrow(x, y, dx, dy, width=w, head_width=head_width, head_length=head_length,
                                    length_includes_head=True, color=color, alpha=alpha, **kwargs)
    ax.add_patch(arrow)


def add_direction_arrows(ax, H_circle_x, V_circle_x,y_scale, n_nodes, radius, outbound=True, top_color="tab:orange",
                         bottom_color="tab:blue", arrow_color="tab:purple", arrow_size=1, ):
    # Arrows
    if outbound:
        draw_arrow(ax, H_circle_x-2*radius-y_scale, (n_nodes - 0.5)*y_scale, y_scale*arrow_size, 0, y_scale*arrow_size, arrow_color)
        draw_arrow(ax, (V_circle_x+radius*2), 0.5*y_scale, y_scale*arrow_size, 0, y_scale*arrow_size, arrow_color)
    else:
        # invert the arrow directons
        draw_arrow(ax, H_circle_x-2*radius, (n_nodes - 0.5)*y_scale, -y_scale*arrow_size, 0, y_scale*arrow_size, arrow_color)
        draw_arrow(ax, V_circle_x+radius*2+y_scale, 0.5*y_scale, -y_scale*arrow_size, 0, y_scale*arrow_size, arrow_color)
    ax.text(H_circle_x-y_scale, n_nodes*y_scale, s="H", fontsize=TICK_SIZE, color=bottom_color, horizontalalignment="center")

    ax.text(V_circle_x + y_scale, y_scale, s="O", fontsize=TICK_SIZE, color=top_color, horizontalalignment="center")


def set_distance_plot_yaxis(ax, mask, top=None):
    if top is None:
        top = mask.tiles_shortest_distances[mask.home_tile, mask.out_tile] + 1 # move the upper limit a bit to show the entire path
    ax.set_ylim(bottom=0, top=top)
    ax.set_yticks([0, 15, 30, mask.tiles_shortest_distances[mask.home_tile, mask.out_tile]])
    ax.set_yticklabels(["Home", "15", "30", "Out"])
    ax.set_ylabel("Distance (tiles)")
    ax.axhline(y=mask.tiles_shortest_distances[mask.home_tile, mask.out_tile], color="black", linestyle="--",
               linewidth=LW_HAIRLINE, zorder=Z_REFERENCE)  # mark the Out port


def format_xs_ys(ax, xs, ylim=None, xlabel="Reward #", ylabel="Interval (s)"):

    # if xticks is None:
    #     xticks = np.arange(1, len(xs) + 1, 5)
    # format x axis
    ax.set_xlim(xs[0]-0.5, xs[-1]+0.5)
    # also set the major tics only at the integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(xlabel)

    # format y axis
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)  # always start your y axis from 0!
    if ylim is not None:
        ax.set_ylim(top=ylim)


def plot_illustrative_cbar(ax, aspect=11, cmap="viridis", ticks=None, ticklabels=None, label_loc="left", **kwargs):
    cbar = Colorbar(ax=ax, cmap=cmap, orientation="vertical", **kwargs)
    if label_loc == "left":
        labelleft, labelright = True, False
    elif label_loc == "right":
        labelleft, labelright = False, True
    else:
        raise ValueError("Unsupported label location. Use 'left' or 'right'.")
    if ticks is None:
        ticks = np.arange(0, 2, 1)
    if ticklabels is None:
        ticklabels = ["Start", "End"]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels, fontsize=TICK_SIZE)
    cbar.ax.tick_params(length=0, labelleft=labelleft, labelright=labelright)
    cbar.ax.set_aspect(aspect)
    return cbar


def plot_phase_lines(ax, phases=[], color="black", linewidth=LW_HAIRLINE, zorder=Z_REFERENCE, **kwargs):
    '''
    Plot the vertical lines for the phases
    :param ax: figure ax
    :param phases: list of phases
    :param linewidth: defaults to the LW_HAIRLINE reference-line tier. Previously absent, so
        the lines silently inherited the rcParams default and every caller had to pass 0.5.
    :return:
    '''
    for phase in phases:
        ax.axvline(phase, color=color, linestyle="--", linewidth=linewidth, zorder=zorder, **kwargs)


def set_corridor_steps_axis(ax, n_corridors):
    '''
    Formating the x axis for the corridor steps
    :param ax:
    :return:
    '''
    ax.set_xlabel("New corridors")
    ax.set_xlim(left=0.5, right=n_corridors+0.5) # for aesthetics
    ax.set_xticks(np.arange(1, n_corridors+1, 2))
    ax.set_xticklabels(np.arange(1, n_corridors+1, 2), )
    ax.set_ylabel("Steps")

    ax.set_ylim(bottom=0)


def add_lines_to_matrix_plot(ax, coordinates=None, color="white", linestyle="--",
                             linewidth=LW_HAIRLINE, **kwargs):
    # add lines to matrix plot to segment the corridors (biclique 1, bottleneck, biclique 2, goal)
    # The body used to hard-code color="white"/linewidth=0.5/linestyle="--", silently ignoring
    # all three arguments; it now honours them (same defaults, so no figure changes).
    if coordinates is None:
        coordinates = [8, 9, 17] # default for the Mask D
    for i in coordinates:
        ax.axvline(x=i-0.5, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)
        ax.axhline(y=i-0.5, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)


def add_squares_to_matrix_plot(ax, xy_coordinates, color="red", linewidth=LW_DATA, **kwargs):
    # add squares to matrix plot to the shortest path
    for x, y in xy_coordinates:
        ax.add_patch(mpatches.Rectangle((x-0.5, y-0.5), 1, 1, facecolor="none", edgecolor=color, linewidth=linewidth, **kwargs))


def add_symbol_for_p_value(ax, p, loc, marker_type="*", color="black", markersize=MS_PT_SMALL, fontsize=TICK_SIZE, plot_ns=False, loc_type="axis", **kwargs):
    """
    Add a symbol for the p-value to the plot. only when p < 0.05
    :param ax: the axis to plot on
    :param p: the p-value to plot
    :param loc: the (x, y) location. Interpretation depends on ``loc_type``:
        ``"axis"`` (default) both x and y are axis fractions in [0, 1];
        ``"numeric"`` both are data coordinates;
        ``"blended"`` x is a data coordinate and y is an axis fraction (via
        ``ax.get_xaxis_transform()``), so the height is fixed relative to the axes
        regardless of the data ``ylim``.
    """
    transform = ax.transData
    if loc_type == "axis":
        y_lim = ax.get_ylim()
        x_lim = ax.get_xlim()
        # check if loc range is in [0, 1]
        if not (0 <= loc[0] <= 1 and 0 <= loc[1] <= 1):
            raise Warning(f"Location {loc} is not in the range [0, 1]. "
                          f"It should be in the format (x, y) where x and y are in [0, 1] range, as it scales with the axis limits.")
        loc = (loc[0] * (x_lim[1] - x_lim[0]) + x_lim[0], loc[1] * (y_lim[1] - y_lim[0]) + y_lim[0]) # convert to data coordinates
    elif loc_type == "blended":
        # x in data coordinates, y as an axes fraction (0-1). Robust to ylim changes.
        transform = ax.get_xaxis_transform()
    # else use the coordinates based on the number in the graph.
    else:
        assert loc_type == "numeric", f"Unsupported loc_type {loc_type}. Use 'axis', 'numeric', or 'blended'."

    if marker_type == "multiple": # add markers for different significance levels
        if p < 0.001:
            marker_str = "***"
        elif p < 0.01:
            marker_str = "**"
        elif p < 0.05:
            marker_str = "*"
        else:
            marker_str = "ns"
        # Asterisk glyphs sit near the top of their font cell, so anchoring the cell
        # center on the bracket line ("center") places the visible star just above and
        # close to the line. "ns" letters fill the lower/middle of the cell, so anchor
        # the box bottom ("bottom") to sit cleanly just above the line. Both are
        # font/axis-relative (no ylim math), robust to ylim changes.
        va_loc = "bottom" if marker_str == "ns" else "center"
        ax.text(loc[0], loc[1], s=marker_str, fontsize=fontsize, color=color, ha="center", va=va_loc, transform=transform, **kwargs)
    elif p < 0.05: # just add a single marker
        ax.scatter(loc[0], loc[1], marker=marker_type, zorder=10, color=color, s=markersize, transform=transform, **kwargs)
    elif plot_ns:
        ax.text(loc[0], loc[1], s="ns", fontsize=fontsize, color=color, ha="center", va="center", transform=transform, **kwargs)


def add_signficance_bracket(ax, loc1, loc2, p_value, color="black", linewidth=LW_HAIRLINE, fontsize=TICK_SIZE, plot_ns=False, loc_type="axis", **kwargs):
    """
    Add a significance bracket between two locations
    :param ax:
    :param loc1:
    :param loc2:
    :param p_value:
    :param y_offset:
    :param color:
    :param fontsize:
    :param plot_ns:
    :param kwargs:
    :return:
    """
    # add a line between the two locations
    transform = ax.transData
    if loc_type == "axis": # make this relative to the axis limits
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        loc1 = (loc1[0] * (x_lim[1] - x_lim[0]) + x_lim[0], loc1[1] * (y_lim[1] - y_lim[0]) + y_lim[0])
        loc2 = (loc2[0] * (x_lim[1] - x_lim[0]) + x_lim[0], loc2[1] * (y_lim[1] - y_lim[0]) + y_lim[0])
    elif loc_type == "blended":
        # x in data coordinates, y as an axes fraction (0-1). Robust to ylim changes.
        transform = ax.get_xaxis_transform()
    else:
        assert loc_type == "numeric", f"Unsupported loc_type {loc_type}. Use 'axis', 'numeric', or 'blended'."

    if loc1[1] != loc2[1]: # if the y values are not the same raise warning
        raise Warning(f"loc1 {loc1} and loc2 {loc2} do not have the same y value. The bracket will be slanted")
    if loc1[0] == loc2[0]: # if the x values are the same raise warning
        raise Warning(f"loc1 {loc1} and loc2 {loc2} have the same x value. The bracket will be vertical")

    ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], color=color, linewidth=linewidth, zorder=5, transform=transform)
    # add the p-value symbol
    add_symbol_for_p_value(ax, p_value, loc=((loc1[0]+loc2[0])/2, loc1[1]), color=color, fontsize=fontsize, plot_ns=plot_ns,
                           loc_type=loc_type, **kwargs)


def format_yaxis_color(ax, color, spine_loc="left"):
    """
    Format the axis for text, spine and lines all have the same color
    :param ax:
    :param color:
    :return:
    """
    ax.yaxis.label.set_color(color)
    ax.spines[spine_loc].set_color(color)
    ax.tick_params(axis='y', colors=color)


def add_letter_labels(FIG, xys, string_sequence="ABCDEFGHIJK"):
    """
    Add letter labels for subplots
    :param FIG:
    :param xys: list of (x, y) coordinates for the labels in figure coordinates

    """
    assert len(xys) <= len(string_sequence), (f"Number of labels {len(xys)} exceeds the length of string sequence {len(string_sequence)}. "
                                              f"Do you need as many panels in one figure?")
    for i, (x, y) in enumerate(xys):
        FIG.text(x, y, string_sequence[i], fontsize=LABEL_SIZE, fontweight="bold", va="top", ha="left")


def add_panel_title(ax, text, pad=TITLE_PAD, fontsize=FONT_SIZE, anchor=1.0, **kwargs):
    """
    Draw a centred heading above ``ax``, offset by a fixed gap measured in points.

    The house replacement for ``ax.set_title``.  Every panel heading in the manuscript
    figures goes through here so that the gap between the text and the top of the panel is
    the same everywhere.

    Why points, not an axes fraction
    --------------------------------
    ``ax.text(0.5, 1.15, ...)`` offsets by 15% of the *panel height*, so the identical line
    of code yields a large gap over a tall panel and a small one over a short one; these
    figures mix panels that differ four-fold in height on a single page.  Composing a
    ``ScaledTranslation`` (inches, via ``dpi_scale_trans``) onto ``ax.transAxes`` instead
    pins the gap to ``pad`` points regardless of panel geometry or ``fontsize``.

    Relationship to constrained layout
    ----------------------------------
    The returned ``Text`` is a child of the axes, so ``Axes.get_tightbbox`` picks it up and
    constrained layout reserves room for it, multi-line headings included -- ``set_title``
    is *not* required for that.  What ``set_title`` does buy is a pad that cannot feed back
    into the layout solver, and the points-based offset here has that property too: an
    axes-fraction offset shrinks as the solver shrinks the axes to accommodate it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    text : str
        Heading, ``\n`` for a second line.
    pad : float
        Gap in points between ``anchor`` and the heading's text box.
    anchor : float
        Axes fraction the gap is measured from; ``1.0`` (default) is the top spine.  Raise
        it for a panel that already draws something above its spine, so the heading clears
        that instead of colliding with it -- the parameter-comparison panels pass
        ``anchor=PARAM_ANNOTATION_Y`` to sit above their value/CI band.  The *gap* stays
        ``pad`` either way; only what it is measured from changes.
    fontsize : float
        Defaults to ``FONT_SIZE``; small sub-panel grids pass ``TICK_SIZE``.  The gap is
        unaffected either way, being measured from the text box rather than the glyphs.
    **kwargs
        Forwarded to ``ax.text`` (``color``, ``fontweight``, ...).

    Returns
    -------
    matplotlib.text.Text
    """
    offset = ScaledTranslation(0, pad / 72, ax.figure.dpi_scale_trans)
    return ax.text(0.5, anchor, text, transform=ax.transAxes + offset,
                   ha="center", va="bottom", fontsize=fontsize, **kwargs)


def get_legend_objects_as_dict(ax, sort=True):
    """
    Get the legend objects as a dictionary with label as key and list of handles as value
    """
    handles, labels = ax.get_legend_handles_labels()  # do not show repeated labels
    label_object_dict = {label: [] for label in set(labels)}
    for handle, label in zip(handles, labels):
        label_object_dict[label].append(handle)
    by_labels = {label: tuple(value) for label, value in label_object_dict.items()}
    if sort: # sort the dictionary by labels
        by_labels = dict(sorted(by_labels.items(), key=lambda item: item[0]))
    return by_labels


def distance_reward_marker(ax, b_type, t_end, distance_seq, color):
    """Helper to handle the scatter logic."""
    if b_type == "H-O":
        ax.scatter(t_end, distance_seq[-1]-1, marker="^", s=MS_AREA_LARGE, color=color, label="Out")
    elif b_type == "O-H":
        ax.scatter(t_end, distance_seq[-1]+1, marker="v", s=MS_AREA_LARGE, color="white",
                   edgecolor=color, label="Home")
    else:
        return


def session_distance_plot_label(ax, x_min, x_max):
    by_label = get_legend_objects_as_dict(ax, sort=True)

    # only use one object for each category
    ax.legend([by_label[k][0] for k in by_label.keys()], by_label.keys(),
              bbox_to_anchor=(-0.02, 0.95), loc="lower left",
              fontsize=TICK_SIZE, ncol=6, frameon=False)

    ax.set_xlim(left=x_min, right=x_max)
    ax.set_xlabel("Session time (s)")


def axis_format_with_color(ax, color, spine_loc="left"):
    """
    Format the axis with the given color for spine, ticks and label
    """
    ax.spines[spine_loc].set_color(color)
    ax.tick_params(axis='y', colors=color)
    ax.yaxis.label.set_color(color)


def create_legend_for_double_axes(ax, ax2, **kwargs):
    by_label1 = get_legend_objects_as_dict(ax, sort=True)
    by_label2 = get_legend_objects_as_dict(ax2, sort=True)

    # only use one object for each category
    handles = [by_label1[k][0] for k in by_label1.keys()] + [by_label2[k][0] for k in by_label2.keys()]
    labels = list(by_label1.keys()) + list(by_label2.keys())
    # merge the same labels:
    unique_labels = set(labels)
    final_handles = []
    for ul in unique_labels:
        ul_handles = [h for h, l in zip(handles, labels) if l == ul]
        final_handles.append(tuple(ul_handles))
    legend = ax.legend(final_handles, unique_labels, fontsize=TICK_SIZE, **kwargs)
    # A legend key has to stay readable however faint the data it stands for. Matplotlib
    # copies the source artist's alpha into the key, so de-emphasised data (e.g. the faded
    # per-traverse markers in plot_individual_memory) would wash out the key too.
    # ``legend_handles`` are the legend's own copies, so this does not touch the plotted data.
    for handle in legend.legend_handles:
        if handle is not None:
            handle.set_alpha(1.0)
    return legend


def gap_to_str(gap):
    """
    Convert a day range tuple to a string representation.
    :param gap: tuple of (start_day, end_day)
    :return: string representation of the day range
    """
    if gap[0] + 1 == gap[1]:
        return "Overnight"
    elif gap[0] == gap[1]:
        return "Day 1"
    else:
        return f"{gap[0]}-{gap[1]-1} days"


def color_bouts_from_indices(
    ax,
    h_indices,
    o_indices,

    *,
    alpha=0.25,
    zorder=0,
    h_first_on_tie=True,
    bout_color_dict=None,
):
    """
    Color spans between adjacent index appearances.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    h_indices, o_indices : sequence of int/float
        Positions where H and O appear.
    bout_colors : dict
        Mapping like {"h-h": "red", "h-o": "orange", "o-h": "blue", "o-o": "green"}.
    alpha : float
        Span transparency.
    zorder : int/float
        Drawing order for axvspan.
    h_first_on_tie : bool
        If an h and o share the same index, choose whether h appears first.

    Returns
    -------
    patches : list
        List of created axvspan patches.
    """
    # Build event list: (position, label)
    events = [(x, "H") for x in h_indices] + [(x, "O") for x in o_indices]

    if not events:
        return []
    if bout_color_dict is None:
        bout_color_dict = bout_type_color_dict
    # Stable tie-breaking for same position
    tie_rank = {"H": 0, "O": 1} if h_first_on_tie else {"H": 1, "O": 0}
    events.sort(key=lambda t: (t[0], tie_rank[t[1]]))

    bout_type_counts = []
    bout_change_locs = []
    patches = []
    for (x0, a), (x1, b) in zip(events[:-1], events[1:]):
        if x1 == x0:
            continue  # zero-width span
        bout_type = f"{a}-{b}"
        color = bout_color_dict.get(bout_type)
        if bout_type not in bout_type_counts:
            bout_type_counts.append(bout_type)
            ax.text(0.5, x0, bout_type, ha="center", va="top", fontsize=FONT_SIZE, color=color)
            bout_change_locs.append((x0, color))
        p = ax.axhspan(x0, x1, color=color, alpha=alpha, zorder=zorder, linewidth=0)
        patches.append(p)
    ax.set_xlim(0, 1)
    return patches, bout_change_locs


def get_normalized_color_seq(signals, cmap=plt.cm.plasma):
    if np.max(signals)-np.min(signals)< 1e-5:
        norm_signal = np.full(signals.shape, np.nan)
    else:
        norm_signal = (signals - np.min(signals))/(np.max(signals)-np.min(signals))
    colors = [cmap(s) for s in norm_signal]
    return colors


def format_value_str(value):
    """
    Return the number of significant digits appropriate for a float value.

    Parameters
    ----------
    value : float
        Non-zero numeric value.

    Returns
    -------
    int
        Number of significant digits: 1 for |value| < 1; floor(log10(|value|))+1
        for |value| >= 1.
    """
    # determine the number of digits in value
    mag = int(np.floor(np.log10(np.abs(value))))
    if mag < 0:
        prec = 1
    else:
        prec = mag+1
    return prec
