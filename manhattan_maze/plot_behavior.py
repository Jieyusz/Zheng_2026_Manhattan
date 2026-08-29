"""Rasters, corridor-discovery, Markov-step, memory, and single-session/bout plots.

Split out of plot_utils.py.

The ``plot_bout_path`` / ``plot_tile_seq`` / ``plot_reward_raster`` /
``plot_tile_distance`` / ``plot_speed_hist`` family are the *render* halves of the
:class:`~manhattan_maze.trajectory.Session` and
:class:`~manhattan_maze.trajectory.Bout` plotting methods: they take the flat tables
produced by :mod:`manhattan_maze.plot_data` instead of live objects, so ``plot_*.py``
can draw these panels straight from ``data/figure_data/*.parquet`` (R8).  The methods
themselves now delegate here, which keeps one copy of the drawing logic (R2).
"""
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from manhattan_maze.plot_constants import (ALPHA_FAINT, CAPSIZE, FONT_SIZE, LW_DATA, LW_HAIRLINE,
                                            LW_TRAJECTORY, MARKER_SIZE, MS_AREA_LARGE, TICK_SIZE,
                                            Z_DATA, Z_RAW_TRACE, Z_REFERENCE)
from manhattan_maze.plot_utils import add_panel_title, bout_type_color_dict, distance_reward_marker, format_yaxis_color, genotype_colors, plot_jittered_scatter, plot_maskd_corridor_interval, plot_oh_scatter_line, session_distance_plot_label, set_corridor_steps_axis
from manhattan_maze.plot_data import TRAVERSE_BOUT_TYPES
from manhattan_maze.utils import make_colorline

__all__ = ['plot_example_rasters', 'plot_example_rasters_from_data', 'plot_markov_in_p10',
           'plot_top_patterns', 'plot_corridor_discovery_pair', 'plot_markov_comparisons_average_steps',
            'plot_individual_memory', 'plot_relative_memory',
            'plot_bout_path', 'plot_tile_seq', 'plot_reward_raster', 'plot_tile_distance',
            'binned_step_counts', 'binned_step_rate', 'plot_speed_hist']


def plot_example_rasters(ax, sessions, cmap=plt.cm.tab10, y_increment=0.1, keep_original_names=False, markersize=MARKER_SIZE):
    '''
    Plot example rasters for the reward intervals any number
    :param ax:
    :param sessions:
    :param cmap:
    :param y_increment:
    :param markersize:
    :return:
    '''
    if isinstance(cmap, mpl.colors.Colormap):
        cmap = [cmap(i) for i in range(len(sessions))]
    else:
        assert isinstance(cmap, (list, tuple)) and len(cmap) >= len(sessions), "cmap must be a list of colors with length at least equal to the number of sessions"

    out_scatters, home_scatters = [], []
    for s, session in enumerate(sessions):
        out_s, home_s, end_line = session.plot_reward_interval_raster(ax, y_loc=s + 1, color=cmap[s],
                                                                      markersize=markersize, y_increment=y_increment)
        out_scatters.append(out_s)
        home_scatters.append(home_s)

    # Change y axis into animal names
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.3, top=len(sessions) + 0.3)
    ax.set_yticks(np.arange(len(sessions)) + 1)
    if keep_original_names:
        animal_names = [s.name.split("_")[0] for s in sessions]
        ax.set_yticklabels(animal_names)


    # merge the legend with the same name using handlertuples
    ax.legend(handles=[tuple(out_scatters), tuple(home_scatters), end_line],
              labels=["Out Reward", "Home Reward", "Session End"], handler_map={tuple: HandlerTuple(ndivide=None)},
              bbox_to_anchor=(0, 1), loc="lower left", fontsize=TICK_SIZE, ncol=3)

    ax.set_ylabel("Mouse")
    ax.set_xlabel("Time in maze (s)")


def plot_markov_in_p10(ax, color_list=["tab:blue", "tab:orange"], **kwargs):
    '''
    Plot Markov (zero order and first order) interval
    :return:
    '''
    # first order
    zero = ax.plot(np.arange(1, 10, 1), np.diff(np.arange(1, 10, 1)**2, prepend=0), color=color_list[0], label="Zero-order", **kwargs)
    first = ax.plot(np.arange(1, 10, 1), np.ones((9)), color=color_list[1], label="1st-order", **kwargs)
    set_corridor_steps_axis(ax, n_corridors=9)
    return *zero, *first


def plot_top_patterns(ax, top_patterns):
    for i, (pattern, count) in enumerate(top_patterns):
        ax.bar(i, count, label=str(pattern), color=plt.cm.viridis(i / len(top_patterns)))
    ax.set_xticks(range(len(top_patterns)))
    ax.set_xticklabels([str(pattern) for pattern, _ in top_patterns], rotation=45, ha='right',
                       )
    ax.set_ylabel("Frequency", )


def plot_corridor_discovery_pair(session, axes=None, fig=None, figsize=(2, 6), n_corridors=None, maskd_special_params=None, **kwargs):
    """
    Plot the corridor discovery pair for the session
    :param session: session object containing the corridor discovery data
    :return: fig, axes, and plot_objects tuple
    """
    if n_corridors is None:
        n_corridors = 17 if session.mask.name == "D" else 9

    slices = session.slice_to_journeys()

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey="row", height_ratios=[1, 0.2])

    plot_objects = []

    for s in slices:
        if not s:
            continue
        interval_array = s.get_corridor_sortie_intervals()
        is_outbound = s[-1].is_outbound()
        col = 0 if is_outbound else 1

        line, scatter, _ = s.plot_corridor_seq_ordered(axes[0, col], **kwargs)

        ax_plot = axes[1, col] if is_outbound else axes[1, col]
        if session.mask.name == "D":
            plot_maskd_corridor_interval(ax_plot, interval_array, maskd_special_params, alpha=0.7, linewidth=LW_DATA, linecolor="tab:brown")
        else:
            ax_plot.plot(np.arange(n_corridors) + 1, interval_array[:, 1], color="tab:brown", linewidth=LW_DATA)

        plot_objects.append((line, scatter))

    add_panel_title(axes[0, 0], "Outbound")
    add_panel_title(axes[0, 1], "Homebound")
    # add theoretical
    if session.mask.name != "D":
        plot_markov_in_p10(axes[1, 0], color_list=["tab:pink", "tab:grey"], linestyle="--", linewidth=LW_HAIRLINE)
        plot_markov_in_p10(axes[1, 1], color_list=["tab:pink", "tab:grey"], linestyle="--", linewidth=LW_HAIRLINE)
    for row in axes:
        for ax in row:
            set_corridor_steps_axis(ax, n_corridors=n_corridors)
    axes[0, 1].yaxis.set_visible(False)
    axes[1, 1].yaxis.set_visible(False)
    axes[1, 0].set_ylabel("Interval (steps)", )
    fig.set_tight_layout(True)

    return fig, axes, plot_objects


def plot_markov_comparisons_average_steps(session, axes=None, fig=None, figsize=(12, 5), maskd_special_params=None):
    """
    Plot the Markov comparisons for the session
    :param session:
    :return: fig, axes, and plot_objects tuple
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)

    mask = session.mask
    corridor_order_indices = maskd_special_params.plot_corridor_order if mask.name == "D" else None
    mask.plot_corridor_average_steps(axes[0], order_by_corridor_indices=True, model_type="Zero order",
                                     corridor_order_indices=corridor_order_indices,
                                     plot_numbers=True)
    add_panel_title(axes[0], "Zero-order Markov")
    mask.plot_corridor_average_steps(axes[1], order_by_corridor_indices=True, model_type="First order",
                                     corridor_order_indices=corridor_order_indices, probability=1,
                                     plot_numbers=True)
    add_panel_title(axes[1], "1st-order Markov (forward p=1)")
    # plot the session
    session.plot_corridor_average_steps(axes[2], maskd_special_params=maskd_special_params)
    # hide y axis
    axes[1].yaxis.set_visible(False)
    axes[2].yaxis.set_visible(False)
    fig.set_tight_layout(True)

    return fig, axes

def plot_individual_memory(ax, metric_dict, averaging_func, a_days, axes_colors=None, unit="traverse",
                           scatter_alpha=ALPHA_FAINT, scatter_zorder=Z_RAW_TRACE, **kwargs):
    """
    Per-traverse points for one animal, with a width-5 moving average over each day.

    The raw points are deliberately de-emphasised -- faded to ``scatter_alpha`` and dropped
    to ``scatter_zorder`` -- so the smoothed line reads as the primary signal. Both are
    needed: at full z-order the markers are drawn over the line and break it up at every
    crossing, so fading alone is not enough.

    ``scatter_alpha`` / ``scatter_zorder`` are separate parameters rather than part of
    ``**kwargs`` because ``**kwargs`` is forwarded to the mean line as well as to the
    markers -- an ``alpha`` passed there would fade the line by the same amount, which is
    the opposite of the intent.

    Callers: fig:ac_mem_gen A (duration/turn error) and fig:ac_mem_supp B (speed), plus
    ``scripts/defense_plots.py::fig_ac_memory_example``, which inherits these defaults.
    """
    # one axis per metric; a twin y-axis is added only when a second metric is given
    metric_names = list(metric_dict.keys())
    if len(metric_names) == 1:
        # only one axis is enough
        axes = [ax]
    else:
        axes = [ax, ax.twinx()]

    if axes_colors is None:
        axes_colors = ["tab:grey", "tab:red"]


    for k, metric_name in enumerate(metric_names):
        metric_array = metric_dict[metric_name]
        n_sessions, n_traverses = metric_array.shape
        for j in range(n_sessions):
            xs = np.arange(n_traverses)+ j*n_traverses+1
            ys = metric_array[j]
            os, hs, _ = plot_oh_scatter_line(axes[k], ys=ys, xs=xs, scatter_colors=[axes_colors[k], axes_colors[k]],
                                             line_color=None, format_xy=False, alpha=scatter_alpha,
                                             scatter_zorder=scatter_zorder, **kwargs,)
            ays = averaging_func(ys, window_size=5, mode="valid")
            # Explicit z-order: without it the line lands on the Line2D default of 2, which
            # is below the markers' old Z_MARKER tier.
            axes[k].plot(xs[2:-2], ays, color=axes_colors[k], zorder=Z_DATA, **kwargs)
            axes[k].axvline(x=(j+1)*n_traverses+0.5, color="black", linestyle="--", linewidth=LW_HAIRLINE,
                            zorder=Z_REFERENCE)

    assert len(a_days) == n_sessions, f"Length of a_days {len(a_days)} must match number of sessions {n_sessions}"
    for i, day in enumerate(a_days):
        ax.text((i+0.5)/len(a_days), 0.9, f"Day{day+1}", fontsize=TICK_SIZE,
            color="black", ha="center", va="center", transform=ax.transAxes, zorder=15)
    # format
    ax.set_xlabel(f"First {n_traverses} {unit}s each day")
    ax.set_xlim(0, n_sessions*n_traverses+1)
    for i, (a, loc) in enumerate(zip(axes, ["left", "right"])):
        a.set_ylabel(metric_names[i])
        a.set_ylim(bottom=0)
        format_yaxis_color(a, axes_colors[i], spine_loc=loc)

    return tuple(axes)


# of different day ranges
def plot_relative_memory(axes, ratio_dict, raw_dict, day_gaps=None, format_title=True,
                           colordict=None, ylabel="Relative metric"):
    """Plot per-genotype relative-memory ratios across day-gap ranges.

    For each day-gap range (one column of axes) and each genotype, draws the
    observed ratio with its bootstrap confidence interval (errorbar + centre bar)
    and overlays the per-session raw ratios as jittered scatter points.

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        One axis per day-gap range; ``len(axes)`` must equal ``len(day_gaps)``.
    ratio_dict : dict[str, list of tuple]
        Maps genotype label to a list of ``(gap, observed_ratio, low, high)``
        tuples, one per day-gap range. ``observed_ratio`` is the Day-N/Day-1
        metric ratio (dimensionless, [0, ~1.1]); ``low``/``high`` are the
        confidence-interval bounds.
    raw_dict : dict[str, list of tuple]
        Maps genotype label to a list of ``(gap, raw_df)`` tuples, one per
        day-gap range. ``raw_df`` has a ``Day`` column (0-based; ``Day == 0``
        rows are excluded) and a ``SessionRatio`` column (dimensionless).
    day_gaps : list of tuple of int, optional
        Inclusive-exclusive day-gap ranges ``(start, end)``; defaults to
        ``[(1, 2), (2, 8), (8, 61)]``. These are day numbers, not traverse
        numbers (no C8 conversion).
    format_title : bool, optional
        If True, write the per-column gap titles ("Overnight", "N- to M-day Gap").
    colordict : dict[str, color], optional
        Maps genotype to plot color; defaults to ``genotype_colors``.
    ylabel : str, optional
        Y-axis label for the leftmost (visible-axis) column.

    Returns
    -------
    None
        Draws onto ``axes`` in place.

    Notes
    -----
    Only the leftmost axis keeps a visible y-axis; the remaining columns share
    its scale with the y-axis hidden. A dashed reference line at ratio = 1
    marks no change between days.
    """
    if day_gaps is None:
        day_gaps = [(1, 2), (2, 8), (8, 61)]
    # for each range of day gaps, determine the selected dataset.

    # check axes length
    assert len(day_gaps) == len(axes), "Number of day gap ranges must match number of axes"
    categories = list(ratio_dict.keys()) # genotype

    if colordict is None:
        colordict = genotype_colors
    for label_index, cat in enumerate(categories):
        results = ratio_dict[cat]
        color = colordict[cat]
        for i, sub_result in enumerate(results):
            gap, observed_ratio, low, high = sub_result
            yerr = np.array([[observed_ratio - low], [high - observed_ratio]])
            # Plot errorbar with smaller caps
            axes[i].errorbar(label_index, observed_ratio, yerr=yerr, color=color, fmt="none", capsize=CAPSIZE, elinewidth=LW_HAIRLINE, zorder=5)
            # Add longer horizontal line through the center point
            center_linelength = 0.12
            axes[i].plot([label_index - center_linelength, label_index + center_linelength],
                        [observed_ratio, observed_ratio], color=color, linewidth=LW_DATA, zorder=6)
            value_str = rf"{observed_ratio:.1g}$\pm${high - observed_ratio:.1g}"
            # axes[i].text(label_index, high, value_str, va="bottom", color=color,
                    # fontsize=TICK_SIZE, ha="center")
            # add raw data points
            _, raw_df = raw_dict[cat][i] # find the correct day gap. first of the tuple is day gap
            raw_ratios = raw_df[raw_df["Day"]!=0].SessionRatio.values
            # plot scatter
            plot_jittered_scatter(axes[i], label_index, raw_ratios, color)

    if format_title:
        add_panel_title(axes[0], "Overnight", fontsize=TICK_SIZE)

    for i, ax in enumerate(axes):
        ax.set_ylim([0, 1.1])
        ax.axhline(1, color="black", linestyle="--", linewidth=LW_HAIRLINE, zorder=Z_REFERENCE)
        if i > 0:
            gap = day_gaps[i]
            axes[i].yaxis.set_visible(False)
            if format_title:
                add_panel_title(axes[i], f"{gap[0]}- to {gap[1] - 1}-day Gap", fontsize=TICK_SIZE)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(categories[:-1])
            ax.set_xlim([-0.4, 1.4])
        else:
            ax.set_ylabel(ylabel)
            # General over the number of categories (3 genotypes for the memory
            # panels; 2 directions for the swap ratio) so xticks/labels always match.
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories)
            ax.set_xlim([-0.4, len(categories) - 0.6])


# --- render halves of the Session/Bout plotting methods (see manhattan_maze.plot_data) ---

def plot_bout_path(ax, path_df, mask, fig=None, noise=0.1, fig_size=(3, 3),
                   cmap=None, linewidth=LW_TRAJECTORY, plot_colorbar=True, alpha=1.0,
                   plot_mask=True, color=None, plot_start_time=False, plot_duration=False,
                   plot_symbol=False, marker_size=MS_AREA_LARGE, marker_color="black", title=None):
    """
    Draw a bout's discrete cell path on the maze, coloured by progress through the bout.

    Render half of :meth:`~manhattan_maze.trajectory.Bout.plot`; consumes
    :func:`~manhattan_maze.plot_data.get_bout_path_data`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  If None, a new figure of ``fig_size`` is created.
    path_df : pandas.DataFrame
        Table from :func:`~manhattan_maze.plot_data.get_bout_path_data`.
    mask : Mask
        Mask whose geometry is drawn and whose ports the symbols mark.
    fig : matplotlib.figure.Figure or None, default None
        Figure used for the colorbar; required when ``plot_colorbar`` is True.
    noise : float, default 0.1
        Gaussian jitter (in cells) added to the path so repeated visits stay visible.
    fig_size : tuple, default (3, 3)
        Size of the created figure when ``ax`` is None.
    cmap : matplotlib.colors.Colormap or None, default None
        Colormap for the path; None means ``viridis``.
    linewidth : float, default 3
        Path line width.
    plot_colorbar : bool, default True
        Draw a colorbar whose ticks are labelled with step counts.
    alpha : float, default 1.0
        Path opacity.
    plot_mask : bool, default True
        Draw the mask underneath the path.
    color : color spec or None, default None
        Single colour for the whole path, overriding ``cmap``.
    plot_start_time : bool, default False
        Annotate the bout's start time (needs a non-NaN ``start_time_s``).
    plot_duration : bool, default False
        Annotate the sleep-thresholded bout duration in seconds.
    plot_symbol : bool, default False
        Mark the destination port: up-triangle at Out for H-O, white down-triangle at
        Home for O-H.
    marker_size : float, default 10
        Port marker size.
    marker_color : str, default "black"
        Port marker fill (H-O) or edge (O-H) colour.
    title : str or None, default None
        Axes title.  None composes the default ``"Mask <name>\\n<animal>, Session
        <idx>, Bout <idx>"`` from ``path_df``, falling back to just the mask name when
        the table carries no animal identity.

    Notes
    -----
    The jitter is drawn as a single ``np.random.randn(n, 2)`` call, matching the
    original method's consumption of the global RNG stream so that seeding reproduces
    identical panels.  These panels are therefore *not* byte-reproducible without an
    explicit seed.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    if cmap is None:
        cmap = plt.get_cmap('viridis')

    if plot_mask:
        mask.plot(ax=ax)

    # Jitter as one (n, 2) draw, exactly as Bout.plot did, to preserve the RNG stream.
    xy = path_df[["col", "row"]].to_numpy(dtype=float)
    x, y = xy.T + 0.5 + noise * np.random.randn(len(xy), 2).T
    traj = make_colorline(x, y, ax=ax, cmap=cmap, linewidth=linewidth, alpha=alpha, color=color)

    n_steps = int(path_df["n_steps"].iloc[0]) if len(path_df) else 0
    if plot_colorbar:
        cbar = fig.colorbar(traj, ax=ax, ticks=np.linspace(0, 1, 5))
        cbar.ax.set_yticklabels(np.linspace(0, n_steps, 5).astype(int), fontsize=TICK_SIZE)

    if plot_duration:
        ax.text(x=0.5, y=-0.05, s=f"{float(path_df['duration_s'].iloc[0]):.2f}s",
                ha="center", va="top", fontsize=TICK_SIZE, transform=ax.transAxes)
    if plot_start_time:
        start_time = float(path_df["start_time_s"].iloc[0])
        ax.text(x=0.01, y=-0.05,
                s=f"Start: {time.strftime('%H:%M:%S', time.gmtime(round(start_time)))}",
                ha="left", va="top", fontsize=TICK_SIZE, transform=ax.transAxes)

    bout_type = path_df["bout_type"].iloc[0] if len(path_df) else "Unknown"
    if plot_symbol:
        if bout_type == "H-O":  # outbound: mark the Out port it reached
            port_pos = mask.out_pos
            ax.scatter(port_pos[0] + 0.5, port_pos[1] + 0.5, marker="^", s=marker_size,
                       color=marker_color, zorder=10)
        elif bout_type == "O-H":  # homebound: mark the Home port it reached
            port_pos = mask.home_pos
            ax.scatter(port_pos[0] + 0.5, port_pos[1] + 0.5, marker="v", s=marker_size,
                       color="white", linewidth=LW_DATA, edgecolor=marker_color, zorder=10)

    ax.set_aspect("equal", "box")
    ax.set_xlim(left=0, right=mask.size)
    if title is None:
        title = _bout_path_title(path_df, mask)
    # Callers pass title="" to suppress the heading. set_title("") was a no-op; an empty
    # ax.text is not -- zero width but a full line height, which it would claim in the
    # axes' tight bbox and so shift the surrounding layout.
    if title:
        add_panel_title(ax, title)


def _bout_path_title(path_df, mask):
    """
    Compose the default bout-path title from a path table.

    Parameters
    ----------
    path_df : pandas.DataFrame
        Table from :func:`~manhattan_maze.plot_data.get_bout_path_data`.
    mask : Mask
        Mask supplying the configuration name.

    Returns
    -------
    str
        ``"Mask <name>\\n<animal>, Session <idx>, Bout <idx>"``, or just
        ``"Mask <name>"`` when the table has no animal identity (a detached bout
        reconstructed from a figure cache).
    """
    if not len(path_df):
        return f"Mask {mask.name}"
    animal = path_df["trajectory_name"].iloc[0]
    if not animal:
        return f"Mask {mask.name}"
    return (f"Mask {mask.name}\n"
            f"{animal}, Session {int(path_df['session_idx'].iloc[0])}, "
            f"Bout {int(path_df['bout_idx'].iloc[0])}")


def plot_tile_seq(ax, tile_df, inverse=False, fps=None, **plot_kwargs):
    """
    Plot a bout's tile-graph distance to its goal against time.

    Render half of :meth:`~manhattan_maze.trajectory.Bout.plot_tile_seq`; consumes
    :func:`~manhattan_maze.plot_data.get_tile_seq_data`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  If None, a new 3x3 inch figure is created.
    tile_df : pandas.DataFrame
        Table from :func:`~manhattan_maze.plot_data.get_tile_seq_data`.
    inverse : bool, default False
        Put ``t = 0`` at the *end* of the bout (times become negative), so several
        traverses of different lengths align on their reward.
    fps : float or None, default None
        Frames per second.  None reads it from the table's ``fps`` column, which
        :func:`~manhattan_maze.plot_data.get_tile_seq_data` provides.  Pass it explicitly
        when driving this from a cached ``"tile steps"`` table, where ``fps`` lives in the
        accompanying meta/manifest table instead of being repeated per step.
    **plot_kwargs
        Forwarded to ``ax.plot`` (``color``, ``alpha``, ``linewidth``, ``label``, ...).
        ``label`` must keep flowing through: the callers build legends by reading
        artist labels back off the axes.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The drawn line, as returned by ``ax.plot``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    frames = tile_df[["in_frame", "out_frame"]].to_numpy()
    fps = float(tile_df["fps"].iloc[0]) if fps is None else float(fps)
    origin = frames[-1, -1] if inverse else frames[0, 0]
    times_s = (frames - origin) / fps
    return ax.plot(times_s[:, -1], tile_df["tile_distance"].to_numpy(dtype=float), **plot_kwargs)


def plot_reward_raster(ax, raster_df, y_loc, color="black", markersize=MS_AREA_LARGE, y_increment=0.1,
                       reverse=False, plot_end=True):
    """
    Draw one session's rewards as a raster row along the in-maze clock.

    Render half of
    :meth:`~manhattan_maze.trajectory.Session.plot_reward_interval_raster`; consumes
    :func:`~manhattan_maze.plot_data.get_reward_raster_data`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    raster_df : pandas.DataFrame
        Table from :func:`~manhattan_maze.plot_data.get_reward_raster_data`.
    y_loc : float
        Row centre for this session's markers (e.g. an animal index).
    color : str, default "black"
        Marker colour (outbound fill / homebound edge).
    markersize : float, default 10
        Scatter marker size.
    y_increment : float, default 0.1
        Vertical offset separating the outbound row (``y_loc + y_increment``) from the
        homebound row (``y_loc - y_increment``).
    reverse : bool, default False
        Shift the clock so ``t = 0`` sits at the *end* of the session (all times become
        negative), to place a pre-swap session before a swap at ``t = 0``.
    plot_end : bool, default True
        Draw a short vertical line at the session end.  When False, ``end_line`` is
        None -- used when several sessions share one row and only one end marker is
        wanted.

    Returns
    -------
    tuple
        ``(out_scatter, home_scatter, end_line)`` matplotlib artists for legend
        building; ``end_line`` is None when ``plot_end`` is False.
    """
    times_in_maze = raster_df["cum_duration_s"].to_numpy(dtype=float)
    if reverse:
        times_in_maze = times_in_maze - times_in_maze[-1]  # have t=0 at the end of the session
    out_times = times_in_maze[raster_df["is_ho"].to_numpy(dtype=bool)]
    home_times = times_in_maze[raster_df["is_oh"].to_numpy(dtype=bool)]

    out_scatter = ax.scatter(out_times, [y_loc + y_increment] * len(out_times), linewidths=LW_HAIRLINE,
                             marker="^", s=markersize, color=color, label="Out reward")
    home_scatter = ax.scatter(home_times, [y_loc - y_increment] * len(home_times), linewidths=LW_HAIRLINE,
                              marker="v", s=markersize, color="white", edgecolor=color,
                              label="Home reward")
    if plot_end:
        end_line = ax.plot([times_in_maze[-1], times_in_maze[-1]],
                           [y_loc - y_increment, y_loc + y_increment],
                           color="black", linewidth=LW_DATA, label="Session End")[0]
    else:
        end_line = None
    return out_scatter, home_scatter, end_line


def plot_tile_distance(ax, dist_df, reference_frame=None, figsize=None, reward_color="black",
                       linewidth=LW_DATA, plot_bout_types=True, bout_type_colors=None):
    """
    Plot distance-to-home over session time, one grey trace per bout.

    Render half of
    :meth:`~manhattan_maze.trajectory.Session.plot_tile_distance_over_time`; consumes
    :func:`~manhattan_maze.plot_data.get_tile_distance_data`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  If None, a new figure of ``figsize`` is created.
    dist_df : pandas.DataFrame
        Table from :func:`~manhattan_maze.plot_data.get_tile_distance_data`.
    reference_frame : int or None, default None
        Time origin.  None uses the table's own ``session_first_frame``; pass the parent
        session's first frame to place a sliced segment on the full session's clock.
    figsize : tuple or None, default None
        Size of the created figure when ``ax`` is None.
    reward_color : color spec or None, default "black"
        Colour of the reward markers at traverse ends; falsy disables them.
    linewidth : float, default 1
        Trace line width.
    plot_bout_types : bool, default True
        Shade each bout by type and annotate traverse numbers.
    bout_type_colors : dict or None, default None
        Bout-type to colour mapping; None uses
        :data:`~manhattan_maze.plot_utils.bout_type_color_dict`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if not len(dist_df):
        return ax

    fps = float(dist_df["fps"].iloc[0])
    ref_frame = (reference_frame if reference_frame is not None
                 else int(dist_df["session_first_frame"].iloc[0]))
    colors = bout_type_colors or bout_type_color_dict
    to_s = lambda frame: (frame - ref_frame) / fps

    start_s = to_s(dist_df["in_frame"].iloc[0])
    end_s = to_s(dist_df["out_frame"].iloc[-1])

    first_bout_type = dist_df["bout_type"].iloc[0]
    for _, bout_rows in dist_df.groupby("bout_idx", sort=True):
        distance_seq = bout_rows["tile_distance"].to_numpy(dtype=float)
        t_start = to_s(bout_rows["in_frame"].iloc[0])
        t_end = to_s(bout_rows["out_frame"].iloc[-1])

        ax.plot(to_s(bout_rows["out_frame"].to_numpy()), distance_seq,
                color="tab:grey", linewidth=linewidth, zorder=1)

        if not plot_bout_types:
            continue

        # Exactly one bout type matches per bout, so this replaces the original's
        # loop over every type calling Bout.satisfy.
        b_type = bout_rows["bout_type"].iloc[0]
        color = colors.get(b_type)
        if color is None:
            continue
        ax.axvspan(t_start, t_end, alpha=0.2, facecolor=color, label=b_type)

        if b_type in TRAVERSE_BOUT_TYPES:
            ax.text(t_start, 0, f"Trav.#{int(bout_rows['traverse_number'].iloc[0])}",
                    color=color, fontweight="bold", ha="left", va="bottom", fontsize=TICK_SIZE)
            if reward_color:
                distance_reward_marker(ax, b_type, t_end, distance_seq, reward_color)

    if plot_bout_types and first_bout_type not in TRAVERSE_BOUT_TYPES:
        s_color = colors["H-H"] if first_bout_type == "H-H" else colors["O-O"]
        ax.text(start_s, 10, "Sorties", color=s_color, va="bottom", fontweight="bold",
                fontsize=TICK_SIZE)

    session_distance_plot_label(ax, start_s, end_s)
    return ax


def binned_step_counts(step_times_s, session_span_s, in_maze_end_s=np.inf, bw=3, tm=None):
    """
    Histogram a step-time point process into per-bin step counts.

    Binning happens here, at render time, rather than in ``gen_*``: the bin width and
    time limit differ per figure panel (3 s / 120 s for the Mask-D speed panel,
    300 s / 7200 s for the Mask-A one), so pre-binning would bake one panel's choice
    into the cache.

    Parameters
    ----------
    step_times_s : array_like
        Step-completion times in **seconds** of in-maze time, from
        :func:`~manhattan_maze.plot_data.get_step_times_data`.
    session_span_s : float
        Wall-clock session span in **seconds**, ``(last_frame - first_frame) / fps``.
        Caps the number of bins.
    in_maze_end_s : float, default numpy.inf
        Total sleep-thresholded in-maze time in **seconds**, used to clamp ``tm`` to
        the session end.  The default disables the clamp, reproducing
        :meth:`~manhattan_maze.trajectory.Session.get_binned_hist`, which never applied
        it -- the clamp belonged to ``plot_speed``.
    bw : float, default 3
        Bin width in **seconds**.
    tm : float or None, default None
        Maximum time to bin in **seconds**; None means ``in_maze_end_s``.

    Returns
    -------
    edges : np.ndarray
        Bin edges [seconds].
    counts : np.ndarray
        Number of steps completed in each bin.

    Notes
    -----
    The bin count deliberately mixes the two clocks exactly as the original
    ``get_binned_hist`` did -- ``min(session_span_s, tm) / bw``, where the first term is
    wall-clock and the second in-maze time -- so published panels keep their axis
    limits.
    """
    if tm is None:
        tm = in_maze_end_s
    tm = min(tm, in_maze_end_s)  # end at session end
    n_bins = int(np.min((session_span_s, tm)) / bw)
    bins = np.linspace(0, n_bins * bw, n_bins + 1, endpoint=True)
    counts, edges = np.histogram(np.asarray(step_times_s, dtype=float), bins)
    return edges, counts


def binned_step_rate(step_times_s, session_span_s, in_maze_end_s, bw=3, tm=None):
    """
    Histogram a step-time point process into a steps-per-second rate.

    Thin wrapper over :func:`binned_step_counts` dividing by the bin width.

    Parameters
    ----------
    step_times_s : array_like
        Step-completion times in **seconds** of in-maze time.
    session_span_s : float
        Wall-clock session span in **seconds**.
    in_maze_end_s : float
        Total sleep-thresholded in-maze time in **seconds**; clamps ``tm``.
    bw : float, default 3
        Bin width in **seconds**.
    tm : float or None, default None
        Maximum time to bin in **seconds**; None means ``in_maze_end_s``.

    Returns
    -------
    edges : np.ndarray
        Bin edges [seconds].
    rate : np.ndarray
        Steps per second in each bin.
    """
    edges, counts = binned_step_counts(step_times_s, session_span_s,
                                       in_maze_end_s=in_maze_end_s, bw=bw, tm=tm)
    return edges, counts / bw


def plot_speed_hist(ax, step_df, color, session_span_s, in_maze_end_s, bw=3, tm=None,
                    unit="tile", plot_hist=True, **kwargs):
    """
    Plot session speed (steps per second) over in-maze time.

    Render half of :meth:`~manhattan_maze.trajectory.Session.plot_speed`; consumes
    :func:`~manhattan_maze.plot_data.get_step_times_data` plus the two session scalars
    carried by :func:`~manhattan_maze.plot_data.get_session_manifest_data`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    step_df : pandas.DataFrame
        Table from :func:`~manhattan_maze.plot_data.get_step_times_data`.
    color : color spec
        Bar/line colour.
    session_span_s : float
        Wall-clock session span in **seconds**.
    in_maze_end_s : float
        Total sleep-thresholded in-maze time in **seconds**.
    bw : float, default 3
        Bin width in **seconds**.
    tm : float or None, default None
        Maximum time in **seconds**; None means the session end.
    unit : {"tile", "corridor"}, default "tile"
        Step unit, used only for the y-axis label.
    plot_hist : bool, default True
        Draw as a histogram (True) or a line through bin centres (False).
    **kwargs
        Forwarded to the matplotlib plotting call.

    Returns
    -------
    tuple or list
        Whatever ``ax.hist`` or ``ax.plot`` returned.
    """
    edges, rate = binned_step_rate(step_df["step_time_s"].to_numpy(dtype=float),
                                   session_span_s, in_maze_end_s, bw=bw, tm=tm)
    if plot_hist:
        hist = ax.hist(edges[:-1], edges, weights=rate, color=color, **kwargs)
    else:
        bin_centers = (edges[:-1] + edges[1:]) / 2
        hist = ax.plot(bin_centers, rate, color=color, **kwargs)

    ax.set_xlabel("Time in maze (s)", )
    ax.set_ylabel(f"{unit.capitalize()}s/s", )
    return hist


def plot_example_rasters_from_data(ax, raster_dfs, cmap=plt.cm.tab10, y_increment=0.1,
                                  animal_names=None, markersize=MARKER_SIZE):
    """
    Stack several sessions' reward rasters, one row per session.

    Array-based sibling of :func:`plot_example_rasters`, for ``plot_*.py`` scripts that
    load cached raster tables instead of holding live sessions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    raster_dfs : sequence of pandas.DataFrame
        One :func:`~manhattan_maze.plot_data.get_reward_raster_data` table per session,
        in the order the rows should be stacked (row 1 at the bottom).
    cmap : matplotlib.colors.Colormap or sequence, default matplotlib.cm.tab10
        Per-session colours; a Colormap is sampled at integer indices.
    y_increment : float, default 0.1
        Vertical offset between the outbound and homebound sub-rows.
    animal_names : sequence of str or None, default None
        Y tick labels.  None leaves numeric ticks.
    markersize : float, default MARKER_SIZE
        Scatter marker size.
    """
    if isinstance(cmap, mpl.colors.Colormap):
        cmap = [cmap(i) for i in range(len(raster_dfs))]
    else:
        assert isinstance(cmap, (list, tuple)) and len(cmap) >= len(raster_dfs), "cmap must be a list of colors with length at least equal to the number of sessions"

    out_scatters, home_scatters = [], []
    for row, raster_df in enumerate(raster_dfs):
        out_s, home_s, end_line = plot_reward_raster(ax, raster_df, y_loc=row + 1, color=cmap[row],
                                                    markersize=markersize, y_increment=y_increment)
        out_scatters.append(out_s)
        home_scatters.append(home_s)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.3, top=len(raster_dfs) + 0.3)
    ax.set_yticks(np.arange(len(raster_dfs)) + 1)
    if animal_names is not None:
        ax.set_yticklabels(animal_names)

    # merge the legend with the same name using handlertuples
    ax.legend(handles=[tuple(out_scatters), tuple(home_scatters), end_line],
              labels=["Out Reward", "Home Reward", "Session End"],
              handler_map={tuple: HandlerTuple(ndivide=None)},
              bbox_to_anchor=(0, 1), loc="lower left", fontsize=TICK_SIZE, ncol=3)

    ax.set_ylabel("Mouse")
    ax.set_xlabel("Time in maze (s)")
