"""Learning-curve, array-data, and fitted-curve plots.

Split out of plot_utils.py.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from manhattan_maze.utils import moving_average
from manhattan_maze.plot_constants import (CAPSIZE_NONE, LW_DATA, LW_EMPHASIS, LW_HAIRLINE,
                                            MARKER_SIZE, MS_AREA_LARGE, MS_AREA_SMALL, TICK_SIZE,
                                            Z_RAW_TRACE, Z_REFERENCE)
from manhattan_maze.plot_utils import format_xs_ys, genotype_colors, mask_colors

__all__ = ['plot_oh_scatter_line', 'plot_array_data', 'plot_direction_mean', 'plot_fitted_curve_and_confidence',
           'plot_level_error_array', 'plot_two_day_data', 'plot_curve_fit_comparison', 'plot_array_comparison',
           'distance_scalar_mappable', 'add_distance_colorbar']

def plot_oh_scatter_line(ax, ys, xs=None, scatter_colors=None, markersize=MS_AREA_LARGE, labels=None, line_color="tab:grey",
                         alpha=1, linewidth=LW_HAIRLINE, format_xy=True, **kwargs):
    """
    Plot the scatter and line for the outbound and homebound data
    :param ax:
    :param ys:
    :param xs:
    :param scatter_colors:
    :param markersize:
    :param labels:
    :param line_color:
    :param kwargs:
    :return: out_scatter, home_scatter, line (for legend)
    """
    if len(ys) == 0:
        return None, None, None
    if xs is None:
        xs = np.arange(len(ys)) + 1
    if labels is None:
        labels = ["Outbound", "Homebound"]
    if scatter_colors is None:
        scatter_colors = ["tab:blue", "tab:orange"]

    if len(scatter_colors) ==  1:
        scatter_colors = [scatter_colors[0]]*2
    if len(labels) == 1:
        labels = [labels[0]]*2
    out_scatter = ax.scatter(xs[::2], ys[::2], marker="^", s=markersize, color=scatter_colors[0], alpha=alpha,
                         label=labels[0], zorder=10)
    home_scatter = ax.scatter(xs[1::2], ys[1::2], marker="v", s=markersize, color="white", alpha=alpha,
                          edgecolor=scatter_colors[1], label=labels[1], zorder=10)

    # plot line
    if line_color is not None:
        line = ax.plot(xs, ys, color=line_color, linewidth=linewidth, zorder=5, label=labels[0], alpha=alpha)
    else:
        line = [None]
    if format_xy:
        format_xs_ys(ax, xs, **kwargs)

    return out_scatter, home_scatter, *line


def plot_direction_mean(ax, array, direction, color, xs=None, markersize=MARKER_SIZE,
                        connect=False, plot_errorbar=False, label=None, linewidth=LW_HAIRLINE):
    """Plot one traverse direction's across-animal mean turn-error series.

    Draws the mean over animals with the codebase's direction glyph (filled up-triangle
    for ``"outbound"``, open/white down-triangle for ``"homebound"``), optionally adding a
    connecting line and/or mean+/-SE error bars. This is the single-direction counterpart
    to :func:`plot_array_data`/:func:`plot_oh_scatter_line`, which take one interleaved
    array and split it by column parity into a single zig-zag; here each direction is
    already sliced to its own columns so it can be connected (or fit) as its own series.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    array : np.ndarray
        ``(n_animals, n_traverses)`` per-direction array (e.g. turn-error rate in [0, 1]);
        NaN-padded for animals with fewer traverses. Reduced over ``axis=0`` (animals).
    direction : {"outbound", "homebound"}
        Selects the marker/fill glyph (filled ^ vs white v).
    color : color
        Edge/line/error-bar color; also the marker fill for ``"outbound"``.
    xs : np.ndarray or None
        1-based x positions; defaults to ``arange(n_traverses) + 1``. Pass explicit values
        to interleave two directions along a shared axis.
    markersize : float
        Scatter marker size (``s=``).
    connect : bool
        If True, draw a connecting line through the mean.
    plot_errorbar : bool
        If True, draw mean+/-SE (SE = nanstd / sqrt(n_animals)) error bars.
    label : str or None
        Legend label; defaults to ``direction.capitalize()``.
    linewidth : float
        Connecting-line width (used only when ``connect``).

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter handle (for building the direction legend).
    """
    if xs is None:
        xs = np.arange(array.shape[1]) + 1
    mean = np.nanmean(array, axis=0)
    if plot_errorbar:
        se = np.nanstd(array, axis=0) / np.sqrt(array.shape[0])
        ax.errorbar(xs, mean, yerr=se, linewidth=0, elinewidth=LW_HAIRLINE, ecolor=color,
                    capsize=CAPSIZE_NONE, zorder=0)
    if connect:
        ax.plot(xs, mean, color=color, linewidth=linewidth, zorder=5)
    marker = "^" if direction == "outbound" else "v"
    fill = color if direction == "outbound" else "white"
    return ax.scatter(xs, mean, marker=marker, s=markersize, linewidths=LW_HAIRLINE,
                      color=fill, edgecolor=color, label=label or direction.capitalize(), zorder=10)


def _summary_stats(array_data, stats_type):
    """
    Central tendency and spread bounds for one ``(n_animals, n_points)`` array.

    The central-tendency and spread are coupled by design: ``"mean"`` always
    pairs with the standard error, ``"median"`` always with the inter-quartile
    range. Columns are reduced over ``axis=0`` (animals), ignoring NaNs.

    Parameters
    ----------
    array_data : np.ndarray
        ``(n_animals, n_points)`` metric array; rows are animals.
    stats_type : {'mean', 'median'}
        ``'mean'`` -> (mean, mean-SE, mean+SE, 'SE');
        ``'median'`` -> (median, 25th pct, 75th pct, 'IQR').

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray, str)
        ``(line_values, low_bound, high_bound, shade_name)``.
    """
    if stats_type == "mean":
        line_values = np.nanmean(array_data, axis=0)
        se = np.nanstd(array_data, axis=0) / np.sqrt(array_data.shape[0])
        return line_values, line_values - se, line_values + se, "SE"
    elif stats_type == "median":
        line_values = np.nanmedian(array_data, axis=0)
        low_bound = np.nanpercentile(array_data, 25, axis=0)
        high_bound = np.nanpercentile(array_data, 75, axis=0)
        return line_values, low_bound, high_bound, "IQR"
    raise ValueError("stats_type must be 'median' or 'mean'.")


# Fixed alpha for the median IQR fill band (mean SE error bars use ``shade_alpha``).
_MEDIAN_SHADE_ALPHA = 0.2


def plot_array_data(ax, array_data, xs=None, stats_type="mean", markersize=MS_AREA_LARGE,
                    scatter_colors=None, add_shade_label=False, labels=None,
                    plot_scatter=True, linewidth=LW_HAIRLINE, shade_alpha=1,
                    line_alpha=1, bar_displacement=0, plot_shade=True,
                    connect_scatters=True, line_color="tab:grey", **axis_kwargs):
    """
    Plot a summary curve (central tendency + spread) for an array of animals.

    Reduces ``array_data`` over animals to a per-point central tendency, then
    draws (optionally) a spread indicator, the alternating outbound/homebound
    scatter markers, and a connecting line. ``format_xs_ys`` is applied last
    with any extra ``**axis_kwargs`` (e.g. ``xlabel``, ``ylabel``, ``ylim``).

    Data contract
    -------------
    Columns alternate **outbound** (even indices ``xs[::2]``, filled ``^``) and
    **homebound** (odd indices ``xs[1::2]``, white ``v``); ``labels`` /
    ``scatter_colors`` are the (outbound, homebound) pair. A single-element list
    is duplicated to both.

    stats_type coupling
    -------------------
    ``"mean"`` -> central=mean, spread=±SE drawn as error bars (honouring
    ``shade_alpha``). ``"median"`` -> central=median, spread=IQR drawn as a
    ``fill_between`` band at a fixed light alpha (``_MEDIAN_SHADE_ALPHA``;
    ``shade_alpha`` does not apply to the median band). The two are always
    coupled this way.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    array_data : np.ndarray
        ``(n_animals, n_points)`` metric array; rows are animals.
    xs : array-like, optional
        X positions; defaults to ``1..n_points``.
    stats_type : {'mean', 'median'}, default 'mean'
        Central tendency + coupled spread (see above).
    markersize : float, default 10
        Scatter marker size.
    scatter_colors : list of str, optional
        ``[outbound, homebound]`` marker colours (default blue/orange); a
        single-element list is duplicated.
    add_shade_label : bool, default False
        If True, label the spread artist ``"SE"``/``"IQR"`` (for legends).
    labels : list of str, optional
        ``[outbound, homebound]`` legend labels (default Outbound/Homebound).
    plot_scatter : bool, default True
        Draw the scatter markers. If False, the connecting line is forced on.
    linewidth : float, default 0.5
        Connecting-line width.
    shade_alpha : float, default 1
        Alpha for the **mean** SE error bars (ignored for the median band).
    line_alpha : float, default 1
        Alpha for the scatters and (when connected) the line.
    bar_displacement : float, default 0
        X shift applied to the mean error bars (for side-by-side groups).
    plot_shade : bool, default True
        Draw the spread indicator (band/error bars).
    connect_scatters : bool, default True
        If False the line is drawn transparently (alpha 0) so the returned
        line handle still exists.
    line_color : str, default 'tab:grey'
        Colour of the line and the spread indicator.
    **axis_kwargs
        Forwarded to ``format_xs_ys`` (axis labels/limits/ticks).

    Returns
    -------
    tuple
        ``(out_scatter, home_scatter, line, shade)`` matplotlib artists;
        ``out_scatter``/``home_scatter`` are ``None`` when ``plot_scatter`` is
        False, and ``shade`` is ``None`` when ``plot_shade`` is False.
    """
    # 1. Initialize Defaults
    if labels is None:
        labels = ["Outbound", "Homebound"]
    if scatter_colors is None:
        scatter_colors = ["tab:blue", "tab:orange"]
    if xs is None:
        xs = np.arange(array_data.shape[1]) + 1

    # Ensure lists have enough elements
    scatter_colors = scatter_colors * 2 if len(scatter_colors) == 1 else scatter_colors
    labels = labels * 2 if len(labels) == 1 else labels

    # 2. Calculate statistics (central tendency + coupled spread)
    line_values, low_bound, high_bound, shade_name = _summary_stats(array_data, stats_type)
    shade_label = shade_name if add_shade_label else None

    # 3. Plot spread: median -> IQR band (fixed alpha); mean -> SE error bars.
    shade = None
    if plot_shade:
        if stats_type == "median":
            shade = ax.fill_between(xs, low_bound, high_bound, color=line_color,
                                    alpha=_MEDIAN_SHADE_ALPHA, zorder=0, label=shade_label)
        else:
            shade = ax.errorbar(xs + bar_displacement, line_values, yerr=(high_bound - line_values),
                                linewidth=0, elinewidth=LW_HAIRLINE, alpha=shade_alpha,
                                ecolor=line_color, capsize=CAPSIZE_NONE, zorder=0, label=shade_label)

    # 4. Plot Scatters
    out_scatter, home_scatter = None, None
    if plot_scatter:
        out_scatter = ax.scatter(xs[::2], line_values[::2], marker="^", s=markersize, linewidths=LW_HAIRLINE,
                                 alpha=line_alpha, color=scatter_colors[0],
                                 label=labels[0], zorder=10)
        home_scatter = ax.scatter(xs[1::2], line_values[1::2], marker="v", s=markersize, linewidths=LW_HAIRLINE,
                                  alpha=line_alpha, color="white", edgecolor=scatter_colors[1],
                                  label=labels[1], zorder=10)
    else:
        connect_scatters = True  # Force line if no scatter

    # 5. Plot Connecting Line (drawn transparently rather than skipped so the
    #    returned line handle is always valid).
    line_alpha_actual = line_alpha if connect_scatters else 0
    line = ax.plot(xs, line_values, color=line_color, linewidth=linewidth,
                   zorder=5, alpha=line_alpha_actual)

    # 6. Formatting
    format_xs_ys(ax, xs, **axis_kwargs)

    return out_scatter, home_scatter, line[0], shade


def plot_fitted_curve_and_confidence(ax, xs, ci_lower, y_fit, ci_upper, label=None, color="tab:blue", linewidth=LW_EMPHASIS, alpha=0.2, plot_confidence=True, **line_kwargs):
    '''
    Plot the fitted curve with confidence interval
    :param ax: figure ax
    :param xs: x values for the fitted curve
    :param popt: optimized parameters
    :param pcov: covariance matrix
    :return:
    '''
    line = ax.plot(xs, y_fit, color=color, linewidth=linewidth, label=label, **line_kwargs) # fitted line
    if plot_confidence:
        ci_plot = ax.fill_between(xs, ci_lower, ci_upper, color=color, label=label, alpha=alpha)
    else:
        ci_plot = None
    return *line, ci_plot


def plot_level_error_array(ax, hole_data_array_dict, data_type="mean", plot_scatter=False, plot_shade=False, connect_scatters=True, xlabel="Traverse #",
                           ylabel="Error rate", ylim=0.8, cmap=plt.cm.cividis, smooth_window=0,
                           chance_level=0.5, **kwargs):
    """One colour-coded summary line per path position vs traverse (close -> far).

    Each entry of ``hole_data_array_dict`` maps a position key ``k`` to an
    ``(n_animals, n_traverses)`` array that is reduced (NaN-robustly) to a single
    line via :func:`plot_array_data`. This works for turn-error holes and, equally,
    for corridor positions: a pre-averaged corridor row is just the ``(1,
    n_traverses)`` special case (feed ``{k: row[None, :]}``).

    ``cmap(k / len(keys))`` sets each line's colour, so pass a discrete colormap
    matching the distance colorbar (e.g. ``plt.get_cmap("viridis", n_positions)``).
    ``smooth_window`` applies a NaN-robust per-animal moving average before the
    mean. ``chance_level`` draws a dashed horizontal chance line (default 0.5);
    pass ``None`` to omit it (e.g. for the corridor rows, which have no 0.5 chance).
    """
    keys = list(hole_data_array_dict.keys())
    plot_objects = []
    for k in keys:
        arr = hole_data_array_dict[k]
        # NaN-robust per-row (per-animal) moving average before the mean is taken.
        # mode="same" (truncated windows at the ends) rather than the "valid" default, so
        # the smoothed curve keeps one value per traverse and starts at traverse 1 like the
        # unsmoothed panels -- otherwise a smoothed panel starts at traverse
        # 1 + (smooth_window - 1) // 2 and sits offset from an unsmoothed one in the same
        # figure. moving_average normalizes by the count actually in each window, so the
        # truncated end points are not dampened; they simply rest on fewer traverses.
        # Interior values are identical to the "valid" result.
        xs = None
        if smooth_window and smooth_window > 1:
            arr = np.vstack([moving_average(row, smooth_window, mode="same") for row in arr])
            xs = np.arange(arr.shape[1]) + 1
        # axes[i].plot(np.nanmean(arr, axis=0), color=cmap(k/9))
        _, _, line, _= plot_array_data(ax, arr, xs=xs, stats_type=data_type,
                                       plot_scatter=plot_scatter, plot_shade=plot_shade, connect_scatters=connect_scatters,
                                       xlabel=xlabel, ylabel=ylabel, ylim=ylim, line_color=cmap(k / len(keys)), **kwargs)
        plot_objects.append(line)
    # Chance line: the approach-conditioned turn error rate has an exact 0.5 chance
    # level (two reachable turn outcomes per scored decision, one correct). Corridor
    # rows have no such chance level, so callers pass chance_level=None there.
    if chance_level is not None:
        ax.axhline(y=chance_level, color="black", linestyle="--", linewidth=LW_HAIRLINE,
                   zorder=Z_REFERENCE, label="Random")
    return plot_objects


def plot_two_day_data(x, y, axes=None, figsize=(4, 2), x1=50, x2=30, xlabel="Traverse #", ylabel="Duration (s)", **kwargs):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize, width_ratios=[x1, x2, x2], sharey=True)

    b, sg1, sg2, maskb, maskc = x
    # iterate each column of x
    # day 1 data sg1==0, sg2 ==0
    b1 = b[(sg1 == 0) & (sg2 == 0)] # day 1
    y1 = y[(sg1 == 0) & (sg2 == 0)]
    # differentiate outbound and homebound
    plot_oh_scatter_line(axes[0], ys=y1, xs=b1, scatter_colors=["tab:brown"], format_xy=False, line_color=None, **kwargs)

    # day 2 data sg1==1, sg2 ==0 for early day 2, sg1==0, sg2 ==1 for late day 2


    # differentiate mask b and mask c
    for k in range(3):
        if k == 0:
            additional_idx = (maskb == 1) & (maskc == 0)
            mask_name = "B"
            color = mask_colors[mask_name]
        elif k == 1:
            additional_idx = (maskb == 0) & (maskc == 1)
            mask_name = "C"
            color = mask_colors[mask_name]
        else:
            # "A"
            additional_idx = (maskb == 0) & (maskc == 0)
            mask_name = "A"
            color = mask_colors[mask_name]
        b2 = b[(sg1 == 1) & (sg2 == 0) & additional_idx]  # day 2 early
        y2 = y[(sg1 == 1) & (sg2 == 0) & additional_idx]  # day 2 early
        b3 = b[(sg1 == 0) & (sg2 == 1) & additional_idx]  # day 2 late
        y3 = y[(sg1 == 0) & (sg2 == 1) & additional_idx]  # day 2 late
        plot_oh_scatter_line(axes[1], ys=y2, xs=b2, scatter_colors=[color], format_xy=False, line_color=None,**kwargs)
        plot_oh_scatter_line(axes[2], ys=y3, xs=b3, scatter_colors=[color], format_xy=False, line_color=None, **kwargs)

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[1].yaxis.set_visible(False)
    axes[2].yaxis.set_visible(False)
    axes[0].set_ylim(bottom=0)
    return axes


def plot_curve_fit_comparison(ax, result_dict, colordict=genotype_colors,
                              xlim=40, upper_y=300, plot_scatter=True, linewidth=LW_DATA,
                              raw_traces_dict=None, raw_trace_cmap=None,
                              raw_trace_color_start=0, raw_trace_colors=None, **kwargs):
    """Plot fitted learning curves with bootstrap CI bands for each group.

    ``raw_traces_dict`` optionally overlays per-animal raw traces for groups
    whose sample size is too small for a meaningful bootstrap CI (e.g. the
    n=3 acortical-control cohort): ``{group_label: array of shape
    (n_animals, n_traverses)}``. Each animal is drawn as a thin line
    (NaN-padded tails are skipped) and the group gets one combined legend entry
    holding a colour swatch per animal (``HandlerTuple(ndivide=None)``), instead
    of a fitted curve + shaded band.

    By default every animal in a raw group shares ``colordict[group_label]``.
    Pass ``raw_trace_cmap`` (a matplotlib colormap) to instead colour animal
    ``i`` with ``raw_trace_cmap(raw_trace_color_start + i)`` so overlapping,
    discretised traces are distinguishable; ``raw_trace_color_start`` offsets
    the index (e.g. to skip qualitative slots that clash with the fitted
    curves). For full control pass ``raw_trace_colors`` (``{group_label: list
    of colours, one per animal}``), which overrides the colormap. Each
    animal's colour then appears in the group's legend entry.
    """
    # check if color dict has the component
    for key in list(result_dict.keys()) + list((raw_traces_dict or {}).keys()):
        if key not in colordict.keys():
            raise ValueError("the result dict must have its keys in the colordict")
    xs = np.linspace(1, xlim, 100)
    at_objects = []
    for gt, fit_results in result_dict.items():
        color = colordict[gt]
        bs, ds, summary_df, bootstrap_curves = fit_results
        line, ci_plot = plot_fitted_curve_and_confidence(ax, *bootstrap_curves, label=gt, color=color, linewidth=linewidth)
        if gt == "Wildtype": # skip mean
            at_objects.append((line, ci_plot))
            continue
        else: # plot mean
            xs_orig = np.arange(1, xlim)
            mean_ds = [np.mean(ds[bs == i]) for i in xs_orig]
            if plot_scatter:
                out, home, _ = plot_oh_scatter_line(
                    ax, alpha=0.5,
                    ys=mean_ds,
                    xs=xs_orig,
                    line_color=None,
                    scatter_colors=[color],
                    format_xy=False,
                    markersize=MS_AREA_SMALL,
                    labels=["Mean", "Mean"],
                )
                at_objects.append((line, ci_plot, out, home))
            else:
                at_objects.append((line, ci_plot))

    none_removed_legend = []
    for group in at_objects:
        none_removed_legend.append(tuple([item for item in group if item is not None]))

    legend_labels = list(result_dict.keys())
    # Overlay per-animal raw traces (no fitted curve / CI band) for small-n groups.
    for gt, traces in (raw_traces_dict or {}).items():
        traces = np.asarray(traces)
        xs_raw = np.arange(1, traces.shape[1] + 1)
        if raw_trace_colors is not None and gt in raw_trace_colors:
            colors = raw_trace_colors[gt]
        elif raw_trace_cmap is not None:
            colors = [raw_trace_cmap(raw_trace_color_start + i) for i in range(len(traces))]
        else:
            colors = [colordict[gt]] * len(traces)
        proxies = []
        for row, color in zip(traces, colors):
            ax.plot(xs_raw, row, color=color, alpha=0.45, linewidth=LW_HAIRLINE, zorder=Z_RAW_TRACE)
            proxies.append(Line2D([], [], color=color, alpha=0.45, linewidth=LW_HAIRLINE))
        # One legend entry per group, packing every animal's colour swatch
        # (HandlerTuple below renders them side by side, undivided).
        none_removed_legend.append(tuple(proxies))
        legend_labels.append(gt)

    ax.legend(none_removed_legend, legend_labels, fontsize=TICK_SIZE,
                  handler_map={tuple: HandlerTuple(ndivide=None)}, )
    format_xs_ys(ax, xs, ylim=upper_y, **kwargs)


def plot_array_comparison(ax, result_dict, colordict=genotype_colors, displace_bars=False,
                          legend_ax=None, **kwargs):
    """
    Overlay one per-animal metric array per group on a single axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw the curves on.
    result_dict : dict of {str: np.ndarray}
        Group label -> ``(n_animals, n_points)`` metric array.
    colordict : dict, default genotype_colors
        Group label -> colour. Every key of ``result_dict`` must appear here.
    displace_bars : bool, default False
        Spread the groups' error bars horizontally so they do not overprint.
    legend_ax : matplotlib.axes.Axes or None, default None
        Where to put the legend. None draws it inside ``ax`` (the usual case). Pass a
        separate blank axes to hoist the legend out of the data panel -- useful when a
        multi-entry legend would otherwise cover the curves in a small panel, or when
        several rows share one legend.
    **kwargs
        Forwarded to :func:`plot_array_data` per group.
    """
    # check if color dict has the component
    for key in result_dict.keys():
        if key not in colordict.keys():
            raise ValueError("the result dict must have its keys in the colordict")

    result_keys = list(result_dict.keys())

    at_objects = []
    if displace_bars:
        bar_displacements = np.linspace(-0.2, 0.2, len(result_keys))
    else:
        bar_displacements = [0] * len(result_keys)
    for k, gt in enumerate(result_keys):
        array = result_dict[gt]
        color = colordict[gt]
        displace = bar_displacements[k]
        out, home, line, shade = plot_array_data(ax, array, scatter_colors=[color], bar_displacement=displace,
                                                 line_color=color, **kwargs)
        at_objects.append((out, home, line, shade))
    # remove the none types in the at_objects
    none_removed_legend = []
    for group in at_objects:
        none_removed_legend.append(tuple([item for item in group if item is not None]))

    if legend_ax is None:
        ax.legend(none_removed_legend, result_dict.keys(), fontsize=TICK_SIZE,
                      handler_map={tuple: HandlerTuple(ndivide=None)}, )
    else:  # hoisted out of the data panel; centre it in the axes given to it
        legend_ax.legend(none_removed_legend, result_dict.keys(), fontsize=TICK_SIZE,
                         handler_map={tuple: HandlerTuple(ndivide=None)},
                         loc="center", frameon=False)


# ---------------------------------------------------------------------------
# Error-by-position line plots (distance-to-reward colour coding)
# ---------------------------------------------------------------------------
# Companion plots to manhattan_maze.analysis.cohort_position_error_rate /
# hole_error_rate_by_direction: one line per path position (corridor distance or
# decision hole) vs slice index, coloured close->far from reward. Used by
# scripts/plot_error_propagation_supp.py.

def distance_scalar_mappable(n_pos, cmap="viridis"):
    """Discrete ScalarMappable over positions at distances 1..n_pos-1.

    The reward position (distance 0) is dropped; distance 1 (closest to reward)
    maps to the dark end of ``cmap``, distance n_pos-1 (start) to the light end.
    Defaults to viridis (animal data); pass e.g. ``cmap="plasma"`` for the RL
    columns to keep the two colour groups visually distinct.
    """
    disc = plt.get_cmap(cmap, n_pos - 1)
    norm = mpl.colors.BoundaryNorm(np.arange(0.5, n_pos + 0.5), n_pos - 1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=disc)
    sm.set_array([])
    return sm


def add_distance_colorbar(fig, axes, sm, n_pos, fraction=0.03, pad=0.02, cax=None,
                          show_ticklabels=True, label="Distance to reward",
                          ticklocation="right"):
    """Shared discrete colorbar keyed to distance-to-reward (close -> far).

    Pass ``cax`` to draw into an explicit colorbar axis (e.g. a dedicated gridspec
    cell so several colorbars share an identical height) instead of stealing space
    from ``axes``. ``ticklocation`` ("left"/"right") sets which side the ticks (and,
    if shown, their labels) face -- handy for a side-by-side double colorbar whose
    two bars flank one shared, central label set. ``show_ticklabels=False`` keeps
    the two ticks but hides their text; ``label`` is the colorbar's main label
    (pass ``None`` to omit it).
    """
    if cax is not None:
        cb = fig.colorbar(sm, cax=cax, ticks=[1, n_pos - 1], ticklocation=ticklocation)
    else:
        cb = fig.colorbar(sm, ax=list(np.ravel(axes)), fraction=fraction, pad=pad,
                          ticks=[1, n_pos - 1])
    if show_ticklabels:
        cb.ax.set_yticklabels(["close\n(reward)", "far\n(start)"], fontsize=TICK_SIZE)
    else:
        cb.ax.set_yticklabels([])
    if label:
        cb.set_label(label)
    return cb
