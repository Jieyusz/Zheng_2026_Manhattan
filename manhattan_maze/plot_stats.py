"""Group / memory / gap comparison plots and statistical-result annotations.

Split out of plot_utils.py.
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from scipy import stats
from manhattan_maze.plot_constants import (CAPSIZE, FONT_SIZE, LW_EMPHASIS, LW_HAIRLINE,
                                            MS_AREA_SMALL, MS_PT_SMALL, PARAM_ANNOTATION_Y,
                                            TICK_SIZE, Z_REFERENCE)
from manhattan_maze.plot_utils import add_signficance_bracket, add_symbol_for_p_value, format_value_str, format_xs_ys, gap_to_str, genotype_colors, plot_array_data, plot_box, plot_jittered_scatter

__all__ = ['plot_time_series_kruskal_results', 'plot_kruskal_results_at_single_point', 'plot_aggregated_choice_ratios', 'plot_offpath_choice_ratios', 'plot_grouped_memory', 'plot_group_scatter_box_comparison', 'plot_pairwise_results_across_bars', 'plot_late_early_comparison', 'plot_gap_comparison_series', 'get_param_ci', 'annotate_param_estimate', 'plot_param_estimate_with_ci', 'plot_ci_shade', 'plot_ci_ratios']

def plot_time_series_kruskal_results(ax, kruskal_result_list, **kwargs):
    time_range = len(kruskal_result_list)
    for t in range(time_range):
        loc = ((t+0.5)/time_range, 0.96)
        res = kruskal_result_list[t]
        plot_kruskal_results_at_single_point(ax, res, default_loc=loc, **kwargs)


def plot_kruskal_results_at_single_point(ax, kruskal_results, default_loc, default_color="black", colors_dict=None, y_offset=0.03,
                                         x_offset=0.03, plot_pairwise=True, **marker_kwargs):
    """
    Plot the Kruskal results on the ax
    :param ax:
    :param kruskal_results: format in the (kruskal_stat, kruskal_p, pairwise_results)
    pairwise_results: should be none if kruskal_p < 0.05
    results should be a dataframe with column (group, stats, p_value)
    :param default_loc: the default location to plot the p_value for kruskal test
    :param default_color: the default color for the kruskal p_values
    :param colors_dict: a dictionary of colors for the pairwise results, need to have the same keys as the group names
    :param y_offset: the y_offset for the y location of the pairwise results
    """
    kruskal_stat, kruskal_p, pairwise_results = kruskal_results
    # plot the kruskal p-value
    if kruskal_stat is None and pairwise_results is not None: # only one pair of Mannwhitney u test
        assert len(pairwise_results) == 1, (f"No kruskal stats means this is only Mann Whitney U test, "
                                            f"but the pairwise result is {pairwise_results}")
        pairwise_p = pairwise_results.p_value.values[0]
        add_symbol_for_p_value(ax, pairwise_p, loc=default_loc, color=default_color, **marker_kwargs)

    elif not plot_pairwise:
        add_symbol_for_p_value(ax, kruskal_p, default_loc, color=default_color, **marker_kwargs)
    elif pairwise_results is not None and plot_pairwise: # significant pairwise results
        group_names = pairwise_results.group1.tolist() + pairwise_results.group2.tolist()
        if colors_dict is None: # significant p-vale
            colors_dict = {group: plt.cm.tab10(i) for i, group in enumerate(group_names)}
        else:
            # check if the colors_dict includes the keys from the group names
            assert all(group in colors_dict for group in group_names), \
                f"colors_dict keys {list(colors_dict.keys())} do not match the group names {group_names}"

        # plot the significant pairwise values
        groups_to_plot = pairwise_results[pairwise_results.p_value< 0.05]
        for k, row in groups_to_plot.iterrows():
            group1 = row.group1
            group2 = row.group2
            p_value = row.p_value
            loc1 = (default_loc[0]-x_offset, default_loc[1] - (k+1) * y_offset)
            loc2 = (default_loc[0]+x_offset, default_loc[1] - (k+1) * y_offset)
            add_symbol_for_p_value(ax, p_value, loc1, color=colors_dict[group1], **marker_kwargs)
            add_symbol_for_p_value(ax, p_value, loc2, color=colors_dict[group2], **marker_kwargs)
            # add a dashed line between the two groups
            ax.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], color="black", linewidth=LW_HAIRLINE, transform=ax.transAxes)


def plot_aggregated_choice_ratios(ax, transition_to_node_dict, node_set, bottleneck_color, control_colors=None,
                              random_value=0.2, xlabel=None, ylabel=None, start_idx=0, plot_control=True, **kwargs):
    """
    start_idx: the data array do not specify outbound vs. homebound. so we only plot every other column starting from start_idx (0 for outbound, 1 for homebound)
    """

    start_node, goal_node, control_nodes = node_set
    if plot_control:
        assert len(control_colors) == len(control_nodes), f"Number of control colors {len(control_colors)} must match number of control nodes {len(control_nodes)}"
    if xlabel is None:
        xlabel="Reward"
    if ylabel is None:
        ylabel="Choice ratio"

    for node, transition_array in transition_to_node_dict.items():
        if node == goal_node:
            plot_array_data(ax,
                            transition_array[:, start_idx::2],
                            stats_type="mean",
                            scatter_colors=[bottleneck_color],
                            plot_scatter=False,
                            shade_alpha=1,
                            line_color=bottleneck_color,
                            plot_shade=True,
                            connect_scatters=True,
                            labels=["Bottleneck"],
                            xlabel=xlabel,
                            ylabel=ylabel,
                            ylim=1.05, **kwargs)
        elif plot_control and node in control_nodes:
                plot_array_data(ax,
                                transition_array[:, start_idx::2],
                                stats_type="mean",
                                scatter_colors=[control_colors[control_nodes.index(node)]],
                                plot_scatter=False,
                                shade_alpha=1,
                                line_color=control_colors[control_nodes.index(node)],
                                plot_shade=True,
                                connect_scatters=True,
                                labels=[f"Control {node}"],
                                xlabel=xlabel,
                                ylabel=ylabel,
                                ylim=1.05, **kwargs)
        else:
            continue
    ax.axhline(random_value, linewidth=LW_HAIRLINE, color="black", linestyle="--",
               zorder=Z_REFERENCE, label="Random")


def plot_offpath_choice_ratios(ax, transition_dict, colors, smooth_func=None, smooth_window=5,
                               start_idx=None, chance_level=0.25, xlabel="Reward",
                               ylabel="Choice ratio", ylim=0.5, linewidth=LW_HAIRLINE):
    """
    Population-average choice ratio over rewards, one line per transition (no spread).

    Companion to :func:`plot_aggregated_choice_ratios` for the off-path biclique
    transitions: for each ``(start, end)`` key it plots the across-session mean choice ratio
    (``np.nanmean`` over sessions, so transitions missing in some sessions are skipped per
    point) as a single colored line — no error bars or scatter — optionally smoothed, then
    adds a dashed chance-level reference.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    transition_dict : dict of {(int, int): ndarray (n_sessions, size)}
        Per-transition choice ratios (e.g. from
        :func:`manhattan_maze.analysis.select_biclique_offpath_transitions`).
    colors : dict of {(int, int): color}
        Line color per transition key (shared with the schematic panel).
    smooth_func : callable or None
        Optional ``f(curve, window) -> curve`` applied to each per-transition mean (e.g.
        :func:`manhattan_maze.utils.moving_average`); ``None`` plots the raw mean.
    smooth_window : int, default 5
        Window passed to ``smooth_func``.
    start_idx : {None, 0, 1}, default None
        ``None`` pools all columns (outbound + homebound); ``0``/``1`` keeps one journey
        phase via ``[:, start_idx::2]`` (outbound / homebound), matching the data contract
        of :func:`manhattan_maze.plot_curves.plot_array_data`.
    chance_level : float, default 0.25
        Y of the dashed black chance line (uniform choice over the 4 opposite-group arms).
    xlabel, ylabel : str
    ylim : float, default 1.05
        Upper y limit.
    linewidth : float, default 1.0
    """
    xs = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nanmean of all-NaN slices -> NaN
        for key, array in transition_dict.items():
            data = array if start_idx is None else array[:, start_idx::2]
            mean = np.nanmean(data, axis=0)
            offset = 0
            if smooth_func is not None:
                mean = smooth_func(mean, smooth_window)
                # moving_average defaults to mode="valid": the curve shortens and output j
                # sits at input j + (smooth_window - 1) // 2, so shift xs to keep the true
                # reward numbers on the axis.
                offset = (smooth_window - 1) // 2
            xs = np.arange(len(mean)) + 1 + offset
            ax.plot(xs, mean, color=colors[key], linewidth=linewidth)
    if xs is not None:
        format_xs_ys(ax, xs, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    ax.axhline(chance_level, linewidth=LW_HAIRLINE, color="black", linestyle="--",
               zorder=Z_REFERENCE, label="Chance")


def plot_grouped_memory(ax, gap_array_list, xunit="traverse", yunit="Duration (s)", upper_y=None, plot_day=True, **kwargs):
    """
    plot arrays of data, that come in the formats of gap_array_list: a list of tuples of (gap, data_array),
    where gap is a tuple of (start_day, end_day) and data_array is a 2D array of shape (n_sessions, n_traverses)
    """
    x0 = 1
    day_xs = []
    for j, (gap, data_array) in enumerate(gap_array_list):
        _, n_traverses = data_array.shape
        xs = np.arange(n_traverses)+x0
        plot_array_data(ax, data_array, xs=xs, **kwargs)
        x0 += n_traverses
        ax.axvline(x=x0-0.5, color="black", linestyle="--", linewidth=LW_HAIRLINE, zorder=Z_REFERENCE)
        gap_str = gap_to_str(gap)
        day_xs.append((gap_str, x0 - n_traverses//2-1))

    upper_y = ax.get_ylim()[1] if upper_y is None else upper_y
    ax.set_ylim(bottom=0, top=upper_y)
    # add day range labels
    if plot_day:
        for gap_str, day_x in day_xs:
            ax.text(day_x, upper_y*0.9, gap_str, fontsize=TICK_SIZE,
            color="black", ha="center", va="center", zorder=15)

     # format
    ax.set_xlabel(f"First {n_traverses} {xunit}s after gaps")
    ax.set_ylabel(f"{yunit}", )
    # make x ticks factors of 5
    ax.set_xticks(np.arange(0, x0, 5))
    ax.set_xlim(0.5, x0)


def plot_group_scatter_box_comparison(ax, data_dict, kruskal_results, upper_y=None,
                                      colordict=None, markersize=MS_AREA_SMALL, ylabel="N(reward) 1st hour", plot_ns=False,
                                      plot_pairwise=True, plot_scatter=True, scatter_only=None,
                                      markerdict=None, open_markers=None):
    """

    ``scatter_only`` is an optional collection of category labels drawn as scatter
    points without a box (e.g. a small-n group whose box quartiles are not
    meaningful); every other category still gets scatter + box.

    ``markerdict`` optionally maps a category label to a scatter marker (default ``"o"``),
    and ``open_markers`` is an optional collection of labels drawn as hollow glyphs. Together
    they let two categories sharing a single ``colordict`` color be told apart by symbol
    (e.g. filled ``"^"`` vs. open ``"v"`` triangles).
    """
    if colordict is None:
        colordict = genotype_colors
    scatter_only = set(scatter_only or [])
    open_markers = set(open_markers or [])
    categories = list(data_dict.keys()) # depending on the keys, make pairwise comparison
    # check if category in colordict:
    if isinstance(colordict, str):
        # if colordict is a string, use it for all categories
        colordict = {cat: colordict for cat in categories}
    assert all([cat in colordict for cat in categories]), f"All categories {categories} must be in colordict {colordict.keys()}"

    for k, gt in enumerate(categories):
        counts = data_dict[gt]
        if plot_scatter:
            marker = markerdict.get(gt, "o") if markerdict else "o"
            plot_jittered_scatter(ax, k, counts, color=colordict[gt], markersize=markersize,
                                  marker=marker, open_marker=gt in open_markers)
        if gt not in scatter_only:
            plot_box(ax, k, counts, color=colordict[gt], box_width=0.5)
    max_rewards = max([max(counts) for counts in data_dict.values()])
    if upper_y is None:
        # leave headroom so the tallest box/scatter clears the significance-bracket
        # band, which now starts at axes fraction bracket_base (~0.78) in
        # plot_pairwise_results_across_bars.
        upper_y = max_rewards / 0.7
    # add significance bracket
    ax.set_ylim(0, upper_y)
    plot_pairwise_results_across_bars(ax, kruskal_results, categories, upper_y, plot_ns=plot_ns, plot_pairwise=plot_pairwise)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=TICK_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)


def plot_pairwise_results_across_bars(ax, kruskal_results, categories, upper_y=None, plot_ns=False, plot_pairwise=True,
                                      bracket_base=0.78, bracket_step=0.07):
    """
    Stack pairwise significance brackets in the top band of the axes.

    Bracket heights are placed as **axes fractions** (``loc_type="blended"``: x in
    data / category coordinates, y as an axes fraction), so the stack sits at a fixed
    position relative to the axes regardless of the data ``ylim``. The lowest bracket
    is at ``bracket_base`` and each subsequent one is ``bracket_step`` higher; the
    caller must leave enough headroom above the data (see
    :func:`plot_group_scatter_box_comparison`) so the ``bracket_base`` line clears the
    tallest box/scatter.

    Parameters
    ----------
    upper_y : float or None
        Unused for placement (kept for backward compatibility); heights are now
        axes-relative via ``bracket_base``/``bracket_step``.
    bracket_base, bracket_step : float
        Axes-fraction height of the lowest bracket and the gap between stacked ones.
    """
    kruskal_stat, kruskal_p, pairwise_results = kruskal_results
    if not plot_pairwise: # plot only kruskal resutls
        add_symbol_for_p_value(ax, kruskal_p, (0.5, 0.9), loc_type="axis", plot_ns=plot_ns)
    if plot_pairwise and pairwise_results is None: # not significant kruskal results.
        add_symbol_for_p_value(ax, kruskal_p, (0.5, 0.9), loc_type="axis", plot_ns=True)
    else: # significant pairwise results
        if not plot_ns:
            groups_to_plot = pairwise_results[pairwise_results.p_value < 0.05].reset_index(drop=True)
        else:
            groups_to_plot = pairwise_results
        for k, row in groups_to_plot.iterrows():
            group1 = row.group1
            group2 = row.group2
            p_value = row.p_value
            x1 = categories.index(group1)
            x2 = categories.index(group2)
            y = bracket_base + k * bracket_step  # axes fraction (blended transform)
            loc1 = (x1, y)
            loc2 = (x2, y)
            add_signficance_bracket(ax, loc1, loc2, p_value, plot_ns=True, marker_type="multiple", loc_type="blended")


def plot_late_early_comparison(axes, durations, midpoint=10, endpoint=20, **plot_array_data_kwargs):
    assert len(axes) == len(durations) ==2, "Only supports two group comparison"
    for k, duration in enumerate(durations):
        sub_duration = duration[:, :midpoint] if k == 1 else duration[:, midpoint:]
        xs = np.arange(midpoint)+1 if k == 1 else np.arange(midpoint, endpoint)+1
        plot_array_data(axes[k], sub_duration, xs=xs, **plot_array_data_kwargs)


def plot_gap_comparison_series(axes, gap_durations, color="tab:grey", midpoint=10, upper_y=200, stats_type="median",
                               ylabel="Duration (s)", **plot_array_data_kwargs):
    """
    Compare the late previous session with early next session with a certain size of gap.
    """
    assert len(axes) == len(gap_durations)+1, f"the axes and dataset lengths must match, got {len(axes)} axes and {len(gap_durations)} datasets"
    
    _, duration_day1_comparison = gap_durations[0]
    endpoint = duration_day1_comparison[0].shape[1]
    assert midpoint < endpoint, f"midpoint {midpoint} is out of range"

    # plot prior as a box scatter plot
    plot_late_early_comparison(axes[:2], duration_day1_comparison, midpoint=midpoint, endpoint=endpoint,
                               scatter_colors=[color], stats_type=stats_type,
                        default_color=color, **plot_array_data_kwargs)

    axes[0].text(0.5, 0.9, "Day1 late", ha="center", va="center", transform=axes[0].transAxes,
                        fontsize=TICK_SIZE)
    # draw a line for the median from day1 late
    axes[1].text(0.5, 0.9, "Day2 early", ha="center", va="center", transform=axes[1].transAxes,
                        fontsize=TICK_SIZE)
    # next two only plot duration box plot
    for k, (gap_range, duration_comparison) in enumerate(gap_durations[1:]):
        duration_group = []
        for j, duration in enumerate(duration_comparison):
            sub_duration = duration[:, :midpoint] if j == 1 else duration[:, midpoint:]
            sub_duration = sub_duration.flatten()
            duration_group.append(sub_duration)
            # plot box if median,
            if stats_type == "median":
                plot_box(axes[k + 2], x=j, ys=sub_duration, color=color)
                plot_jittered_scatter(axes[k + 2], x=j, ys=sub_duration, markersize=MS_AREA_SMALL, scatter_alpha=0.5,
                                             color=color)
            elif stats_type == "mean": # plot error bar with whiskers only
                mean = np.nanmean(sub_duration)
                sem = stats.sem(sub_duration, nan_policy="omit")
                axes[k + 2].errorbar(x=j, y=mean, yerr=sem, color=color, capsize=CAPSIZE,)
            else:
                raise ValueError("stats_type must be either 'median' or 'mean'")

        # format and stats test
        axes[k + 2].set_xticks([0, 1])
        axes[k + 2].set_xticklabels(["Pre-late", "Post-early"], rotation=45)
        axes[k + 2].text(0.5, 0.9, f"{gap_range[0]}- to {gap_range[1] - 1}-day", ha="center", va="center",
                                transform=axes[k + 2].transAxes, fontsize=TICK_SIZE)
        axes[k+2].set_xlim(-0.3, 1.3)

        # perform within-subject wilcoxon signed-rank test and add significance bracket
        stat, p_value = stats.wilcoxon(duration_group[0], duration_group[1])
        y = upper_y * 0.8
        loc1 = (0, y)
        loc2 = (1, y)
        add_signficance_bracket(axes[k+2], loc1, loc2, p_value, plot_ns=True, marker_type="multiple", loc_type="numeric")

    for i, ax in enumerate(axes):  # format all axes
        ax.set_ylim(0, upper_y)
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.yaxis.set_visible(False)


def get_param_ci(summary_df, param_name):
    """
    Pull one parameter's bootstrap estimate and CI bounds from a fit summary frame.

    Single source for reading a ``fit_traverse_data_df_with_bootstrap`` summary so
    the curve-fit supplementary plots all extract values the same way. Raises a
    clear error naming the available parameters instead of the silent empty-slice
    ``IndexError`` the supp plots used to hit when a name drifted.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Bootstrap summary with columns ``Parameter``, ``Estimate``, ``CI_lower``,
        ``CI_upper``.
    param_name : str
        Parameter to look up (e.g. ``"D_infty"``, ``"delta"``).

    Returns
    -------
    tuple of float
        ``(estimate, ci_lower, ci_upper)``.
    """
    sub_df = summary_df[summary_df.Parameter == param_name]
    if sub_df.empty:
        raise KeyError(
            f"Parameter {param_name!r} not found in summary_df; "
            f"available parameters: {list(summary_df.Parameter)}"
        )
    row = sub_df.iloc[0]
    return row["Estimate"], row["CI_lower"], row["CI_upper"]


def annotate_param_estimate(ax, x, estimate, ci_lower, ci_upper, y, color, zorder, prec=None):
    """
    Draw the two-line ``$estimate^{+hi}_{-lo}$`` annotation above an estimate marker.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x : float
        X position of the annotation in data coordinates (matches the estimate marker).
    estimate, ci_lower, ci_upper : float
        Point estimate and CI bounds.
    y : float
        Y position of the annotation as an axes fraction (top-aligned), so the label
        stays put relative to the axes regardless of the data ``ylim`` (``1.0`` is the
        top spine; ``> 1`` sits above it). Drawn via ``ax.get_xaxis_transform()``
        (data x, axes-fraction y).
    color : color
        Text color.
    prec : int or None
        Significant digits; defaults to ``format_value_str(estimate)`` so precision
        tracks the printed value. Callers keying off the axis scale pass ``prec``
        explicitly (see :func:`plot_param_estimate_with_ci`).
    """
    if prec is None:
        prec = format_value_str(estimate)
    lo_err = estimate - ci_lower
    hi_err = ci_upper - estimate
    # These digits are annotation, not notation, so they follow the sans body font. The
    # mathtext wrapper is still needed to stack the +hi/-lo superscript, so the digits go
    # through \mathregular, which binds to font.family instead of the maths font.
    value_str = (
        rf"$\mathregular{{{estimate:.{prec}g}}}$" + "\n" +
        rf"$^{{\mathregular{{+{hi_err:.{prec}g}}}}}_{{\mathregular{{-{lo_err:.{prec}g}}}}}$"
    )
    # white outline keeps the label readable where it overlaps the error bars
    ax.text(x, y, value_str, va="top", color=color, fontweight="bold",
            fontsize=TICK_SIZE, ha="center", zorder=zorder,
            transform=ax.get_xaxis_transform(),
            path_effects=[patheffects.withStroke(linewidth=LW_EMPHASIS, foreground="white")])


def plot_param_estimate_with_ci(ax, x, summary_df, param_name, color, ylim,
                                markersize=MS_PT_SMALL, capsize=CAPSIZE, annotate=True):
    """
    Plot a single parameter estimate as an asymmetric-CI error bar, optionally annotated.

    Shared core of the curve-fit supplementary panels: looks the parameter up via
    :func:`get_param_ci`, draws the error bar at ``x``, and (by default) writes the
    ``annotate_param_estimate`` value label just above the axes (``PARAM_ANNOTATION_Y``),
    so it tracks the axes rather than the data ``ylim``.

    Returns
    -------
    tuple of float
        ``(estimate, ci_lower, ci_upper)`` so callers can reuse the bounds (e.g. for a
        Day-1 reference band).
    """
    estimate, ci_lower, ci_upper = get_param_ci(summary_df, param_name)
    yerr = np.array([[estimate - ci_lower], [ci_upper - estimate]])
    ax.errorbar(x, estimate, yerr=yerr, color=color, fmt=".", markersize=markersize, capsize=capsize)
    if annotate:
        prec = format_value_str(ylim[1])
        annotate_param_estimate(ax, x, estimate, ci_lower, ci_upper, PARAM_ANNOTATION_Y, color,
                                prec=prec, zorder=10)
    return estimate, ci_lower, ci_upper


def plot_ci_shade(ax, x, ci_lower, ci_upper, color, width=0.3, alpha=0.3):
    """
    Shade a horizontal CI band centred on ``x`` (used for the reference/control overlays).
    """
    ax.fill_between([x - width, x + width], [ci_lower, ci_lower], [ci_upper, ci_upper],
                    color=color, alpha=alpha, zorder=0)

def plot_ci_ratios(ax, ratio_df, param_latex=None, color="black", reference=1.0,
                   markersize=MS_PT_SMALL, capsize=CAPSIZE, linewidth=LW_HAIRLINE, xlabel="Relative value", annotate=False,
                   offset=0.0):
    """
    Forest-style plot of Day-2/Day-1 curve-derived ratios with bootstrap CIs.

    One horizontal error bar per row of ``ratio_df``: the y-axis lists the
    curve-derived quantities (LaTeX labels) and the x-axis shows each ratio with its
    ``[CI_lower, CI_upper]`` interval.  Duration (D) and turn-error (E) rows
    share one axis — concatenate the two metrics' rows before calling, e.g.::

        ratios = figure_data_dict   # from utils.load_all_figure_data()
        d = ratios["Wildtype two day duration param ratios"]
        e = ratios["Wildtype two day turn error rate param ratios"]
        summary = pd.concat([d, e])
        summary = summary[summary.Session == "all"]   # across-combinations summary rows
        plot_ci_ratios(ax, summary, param_latex=config.PARAM_LATEX)   # config from scripts/

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ratio_df : pd.DataFrame
        One row per curve-derived quantity with columns ``Parameter``, ``Ratio``,
        ``CI_lower``, ``CI_upper``.  Rows are drawn top-to-bottom in the given order.
    param_latex : dict or None
        Maps parameter name to its LaTeX tick label (e.g. ``config.PARAM_LATEX``
        from ``scripts/config.py``, derived from ``CURVE_FIT_SPECS``).  When None,
        each label falls back to ``f"${name}$"``; an unmapped name uses the same
        fallback.
    color : color or dict
        Single color, or a ``{parameter_name: color}`` mapping.
    reference : float, default 1.0
        x position of the dashed reference line (1 = no Day-2 change).
    markersize, capsize : float
        Error-bar marker and cap sizes.
    linewidth : float, default 0.5
        Thickness of the error-bar line and caps.
    xlabel : str
        X-axis label.
    annotate : bool, default False
        If True, write each ratio's value beside its marker.
    offset : float, default 0.0
        Vertical shift applied to every bar's y position.  Call twice with, e.g.,
        ``offset=-0.15`` and ``offset=+0.15`` to draw two series (different days or
        genotypes) side by side at the same parameter rows without overlap.
    """
    if param_latex is None:
        param_latex = {}
    df = ratio_df.reset_index(drop=True)
    ys = np.arange(len(df)) + offset
    for y, (_, row) in zip(ys, df.iterrows()):
        estimate, ci_lower, ci_upper = row["Ratio"], row["CI_lower"], row["CI_upper"]
        xerr = np.array([[estimate - ci_lower], [ci_upper - estimate]])
        c = color[row["Parameter"]] if isinstance(color, dict) else color
        ax.errorbar(estimate, y, xerr=xerr, fmt="o", color=c,
                    markersize=markersize, capsize=capsize,
                    elinewidth=linewidth, capthick=linewidth, zorder=5)
        if annotate:
            prec = format_value_str(estimate) + 1
            ax.text(ci_upper, y, f"  {estimate:.{prec}g}", va="center", ha="left",
                    color=c, fontsize=TICK_SIZE)

    ax.axvline(reference, color="black", linestyle="--", linewidth=LW_HAIRLINE, zorder=Z_REFERENCE)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([param_latex.get(p, rf"${p}$") for p in df["Parameter"]])
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.invert_yaxis()  # first row at the top, forest-plot convention
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 1.2)