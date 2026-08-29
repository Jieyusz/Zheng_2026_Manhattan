from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)

save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 5.5

figure_data_dict = utils.load_all_figure_data()

# combine memory plots and generalization

FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(3, 1, height_ratios=[1, 1.4, 1], hspace=0.05)
gs00 = gs0[0].subgridspec(1, 1,)

# memory of Mask A
ax_a = FIG.add_subplot(gs00[0])
a_metric_dict = {"Duration (s)": figure_data_dict["Acortical Mask A example memory duration"],
               "Turn error rate": figure_data_dict["Acortical Mask A example memory turn error rate"]}
a_days = figure_data_dict["Acortical Mask A example memory days"]
_, ax_a2 = plot_utils.plot_individual_memory(ax_a, a_metric_dict, utils.moving_average, a_days, linewidth=config.LW_DATA, markersize=config.MS_AREA_DEFAULT)
ax_a2.axhline(0.5, color="tab:red", linestyle="--", label="Random", linewidth=config.LW_HAIRLINE,
              zorder=config.Z_REFERENCE)
ax_a.set_ylim(top=400)
ax_a2.set_ylim(top=0.6)
# add legend
plot_utils.create_legend_for_double_axes(ax_a, ax_a2, bbox_to_anchor=(-0.02, 0.95), loc="lower left",
            ncol=6, frameon=False, handler_map={tuple: HandlerTuple(ndivide=None)})

gs10 = gs0[1].subgridspec(1, 4, width_ratios=[0.06, 1, 0.9, 0.3], hspace=0.05, wspace=0)
gs11 = gs10[1].subgridspec(2, 3, width_ratios=[1.5, 1, 1], hspace=0, wspace=0,
                          height_ratios=[1, 1.1])
# plot results
duration_axes = [FIG.add_subplot(gs11[0, i]) for i in range(3)]
ratio_dict = utils.select_by_prefix(figure_data_dict, config.GENOTYPES, "Mask A duration relative ratio")

raw_mean_dict = utils.select_by_prefix(figure_data_dict, config.GENOTYPES, "Mask A duration gap data points")
plot_utils.plot_relative_memory(duration_axes, ratio_dict, raw_mean_dict, format_title=True,
                       ylabel="Relative duration")
for ax in duration_axes:
    ax.set_xticklabels([])

# plot errors
error_axes = [FIG.add_subplot(gs11[1, i]) for i in range(3)]
error_dict = utils.select_by_prefix(figure_data_dict, config.GENOTYPES, "Mask A turn error rate relative ratio")
raw_error_dict = utils.select_by_prefix(figure_data_dict, config.GENOTYPES, "Mask A turn error rate gap data points")
plot_utils.plot_relative_memory(error_axes, error_dict, raw_error_dict, format_title=False,
                       ylabel="Relative error")
for ax in error_axes:
    ax.xaxis.set_tick_params(rotation=45)

# generalization in A on the right
# First plot Later masks
mask_color_dict = {"First A": "tab:grey",
                "Repeat A": plot_utils.mask_colors["A"],
                    "Mask B": plot_utils.mask_colors["B"],
                   "Mask C": plot_utils.mask_colors["C"],}
gs12 = gs10[2].subgridspec(2, 1)

# curve-derived relative ratio
ax_ratio = FIG.add_subplot(gs10[3])
summary = figure_data_dict["Acortical generalization param ratios"]
gen_keys = [key for key in mask_color_dict if key != "First A"]
for i, key in enumerate(gen_keys):
    sub_df = summary[summary.Comparison == f"{key}/First A"]
    plot_utils.plot_ci_ratios(ax_ratio, sub_df, param_latex=config.PARAM_LATEX,
                              color=mask_color_dict[key], offset=0.3 * (i - 1))
plot_utils.add_panel_title(ax_ratio, "Generalization\nMask/First A",
                           fontsize=plot_utils.TICK_SIZE, color="black")

def acortical_gen_fit_dict(metric):
    """Build the {First A, Repeat A, Mask B, Mask C} fit-results dict for a metric.

    Assembles the acortical-mouse generalization comparison for one metric by
    keying into ``figure_data_dict``. "First A" is the un-generalized reference
    ("Acortical A {metric} fit results"); the others use the generalization keys
    ("... repeat Gen ..." / "... Gen ...").

    Parameters
    ----------
    metric : str
        Metric name embedded in the figure-data key, e.g. ``"duration"``
        (seconds) or ``"turn error rate"`` (fraction [0, 1]).

    Returns
    -------
    dict[str, tuple]
        Maps each comparison label to its fitted-curve tuple
        ``(bs, ds, summary_df, bootstrap_curves)`` (see data_contracts.md §12).

    Notes
    -----
    Reads the module-level ``figure_data_dict``; raises ``KeyError`` if any
    expected key is absent for ``metric``.
    """
    return {"First A": figure_data_dict[f"Acortical A {metric} fit results"],
            "Repeat A": figure_data_dict[f"Acortical A repeat Gen {metric} fit results"],
            "Mask B": figure_data_dict[f"Acortical B Gen {metric} fit results"],
            "Mask C": figure_data_dict[f"Acortical C Gen {metric} fit results"]}


bc_durations = acortical_gen_fit_dict("duration")
ax_duration = FIG.add_subplot(gs12[0])
plot_utils.plot_curve_fit_comparison(ax_duration, result_dict=bc_durations,
                                     xlim=20, upper_y=200, xlabel="Traverse #", ylabel="Duration (s)", colordict=mask_color_dict,
                                     plot_scatter=False)
ax_duration.set_xticklabels([])
ax_duration.set_xlabel("")

bc_error_rate = acortical_gen_fit_dict("turn error rate")
ax_error = FIG.add_subplot(gs12[1])
plot_utils.plot_curve_fit_comparison(ax_error, result_dict=bc_error_rate,
                                     xlim=20, upper_y=0.5, xlabel="Traverse #", ylabel="Turn error rate", colordict=mask_color_dict,
                                     plot_scatter=False)
# hide legend
ax_error.get_legend().remove()

# row 2: acortical in Mask D
gs20 = gs0[-1].subgridspec(1, 3, width_ratios=[2, 2, 1], hspace=0.05)
ax_d_tpc = FIG.add_subplot(gs20[-1])
tiles_per_corridor = utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:2], "Mask D tiles per corridor")
tpc_for_kruskals = {"Acor. -": [tcp for j, tcp in tiles_per_corridor["Acortical"] if j ==0],
    "Acor. +": [tcp for j, tcp in tiles_per_corridor["Acortical"] if j ==1],
                    "Control":tiles_per_corridor["Control"]}
kruskal_results =  utils.kruskal_with_pairwise_mann_whitney(tpc_for_kruskals)

def plot_unsuccessful_scatter_box(ax, tpc_for_kruskals, kruskal_results, plot_ns=True, ylabel="Tiles/corridor",
                                  plot_pairwise=True, markersize=config.MS_AREA_SMALL):
    """Plot per-group jittered scatter + box with pairwise-significance brackets.

    Draws one box-and-scatter column per group (acortical successful/unsuccessful
    and control) for the tiles-per-corridor metric, then overlays the
    Kruskal-Wallis pairwise comparison brackets.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    tpc_for_kruskals : dict[str, array-like of float]
        Maps group label to its per-animal scalar values (tiles per corridor;
        dimensionless count per animal, see data_contracts.md §12). The
        "Acor. -" (unsuccessful) group is drawn with a downward-triangle marker.
    kruskal_results : object
        Output of ``utils.kruskal_with_pairwise_mann_whitney`` for the same
        groups; consumed by ``plot_utils.plot_pairwise_results_across_bars``.
    plot_ns : bool, optional
        If True, annotate non-significant pairwise comparisons.
    ylabel : str, optional
        Y-axis label.
    plot_pairwise : bool, optional
        If True, draw the pairwise comparison brackets.
    markersize : int, optional
        Scatter marker size.

    Returns
    -------
    None
        Draws onto ``ax`` in place.

    Notes
    -----
    The y-limit is set so the global maximum value across all groups reaches ~0.7 of
    the axes height, leaving the top band clear for the significance brackets.
    """
    categories = list(tpc_for_kruskals.keys()) # depending on the keys, make pairwise comparison
    # check if category in colordict:
    colordict = {"Acor. +": plot_utils.genotype_colors["Acortical"],
                 "Acor. -": plot_utils.genotype_colors["Acortical"],
                 "Control": plot_utils.genotype_colors["Control"]}

    for k, gt in enumerate(categories):
        counts = tpc_for_kruskals[gt]
        if gt == "Acor. -":
            marker = "v"
        else:
            marker ="o"
        plot_utils.plot_jittered_scatter(ax, k, counts, color=colordict[gt], markersize=markersize, marker=marker)
        plot_utils.plot_box(ax, k, counts, color=colordict[gt], box_width=0.5)
    max_rewards = max([max(counts) for counts in tpc_for_kruskals.values()])
    upper_y = max_rewards / 0.7  # headroom so boxes clear the bracket band (base ~0.78 axes fraction)
    # add significance bracket
    ax.set_ylim(0, upper_y)
    plot_utils.plot_pairwise_results_across_bars(ax, kruskal_results, categories, upper_y, plot_ns=plot_ns, plot_pairwise=plot_pairwise)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=plot_utils.FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=plot_utils.FONT_SIZE)

plot_unsuccessful_scatter_box(ax_d_tpc, tpc_for_kruskals, kruskal_results)
plot_utils.add_panel_title(ax_d_tpc, "Mask D")
ax_d_tpc.tick_params(axis="x", rotation=45)


ax_d_traverse = FIG.add_subplot(gs20[0])
d_durations = {"Acortical": figure_data_dict["Acortical D Gen duration fit results"],
    "Control": figure_data_dict["Control D duration fit results"],
               "Wildtype": figure_data_dict["Wildtype D duration fit results"]}
plot_utils.plot_curve_fit_comparison(ax_d_traverse, result_dict=d_durations,
                                     xlim=20, upper_y=200, xlabel="Traverse #", ylabel="Duration (s)",
                                     plot_scatter=False)

ax_d_bottleneck = FIG.add_subplot(gs20[1])
d_bottleneck = utils.select_by_prefix(figure_data_dict, config.GENOTYPES, "Mask D goal transition array")
plot_utils.plot_array_comparison(ax_d_bottleneck, d_bottleneck, stats_type="mean", displace_bars=True, plot_shade=True, connect_scatters=True,
                                 plot_scatter=False, shade_alpha=1, linewidth=config.LW_DATA,
                                 xlabel="Reward #", ylabel="Choice of bottleneck", ylim=1.05)
# hide legend
ax_d_bottleneck.get_legend().remove()
# add random:
ax_d_bottleneck.axhline(0.2, color='k', linestyle='--', linewidth=config.LW_HAIRLINE,
                        zorder=config.Z_REFERENCE, label="Chance")

# Row 3: traverse similarity


plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.01, 0.71), (0.01, 0.53), (0.44, 0.71), (0.44, 0.53), (0.82, 0.71),
                                   (0.01, 0.3), (0.38, 0.3), (0.77, 0.3)])
config.save_figure(FIG, "ac_mem_gen.pdf", save_path)