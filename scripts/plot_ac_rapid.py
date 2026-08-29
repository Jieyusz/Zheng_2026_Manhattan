from manhattan_maze import plot_utils, utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)

save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 5.8

figure_data_dict = utils.load_all_figure_data()

genotypes = list(plot_utils.genotype_colors)

# Plot the number of rewards within the first session:

FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
# gs0 = FIG.add_gridspec(4, 1, height_ratios=[1, 1, 1.2, 1], hspace=0.05)
gs0 = FIG.add_gridspec(4, 1, height_ratios=[0.8, 1, 1, 1], hspace=0.05)

# Row 1: example of session behavior in Mask A (by acortical mouse)
gs00 = gs0[1].subgridspec(1, 1)
ax_e = FIG.add_subplot(gs00[0])
# The example sessions are exported as flat per-bout/per-tile tables (R8), so the segment
# is a bout_idx range filter and the panel keeps the parent session's clock.
acortical_meta = figure_data_dict["Acortical A example bout meta"]
acortical_tiles = figure_data_dict["Acortical A example tile steps"]
acortical_manifest = figure_data_dict["Acortical A example manifest"]
mask_a = figure_data_dict["masks"]["A"]
example_idx = config.ACORTICAL_A_EXAMPLE_ID
seg_start, seg_end = config.ACORTICAL_A_SEGMENT_BOUTS
example_rows = acortical_meta[acortical_meta.example == example_idx]
segment_meta = example_rows[(example_rows.bout_idx >= seg_start) & (example_rows.bout_idx < seg_end)]
segment_tiles = acortical_tiles[(acortical_tiles.example == example_idx)
                               & (acortical_tiles.bout_idx >= seg_start)
                               & (acortical_tiles.bout_idx < seg_end)]
acortical_row = acortical_manifest[acortical_manifest.example == example_idx].iloc[0]
reference_frame = acortical_row.first_frame
plot_utils.plot_tile_distance(
    ax_e, utils.derive_tile_distance_table(segment_tiles, segment_meta, acortical_row.fps,
                                          reference_frame),
    reward_color=plt.cm.Dark2(example_idx), linewidth=config.LW_HAIRLINE,
    reference_frame=reference_frame) # plot distance over time for the example session
plot_utils.set_distance_plot_yaxis(ax_e, mask_a) # set y-axis for distance plot
FIG.text(0.01, 0.99, "A", fontsize=plot_utils.LABEL_SIZE, fontweight="bold", va="top", ha="left")


gs10 = gs0[0].subgridspec(1, 1)
# row 2: raster of rewards
ax_raster = FIG.add_subplot(gs10[0])
control_meta = figure_data_dict["Control A example bout meta"]
control_manifest = figure_data_dict["Control A example manifest"]

# Shared per-mouse control colours across the raster (A) and the traverse reference (D).
# The control cohort arrays ("Control Mask A ...") are built in control_first_a order, but
# the raster examples are a random subset of that cohort. Each example's cohort row is recorded in the manifest by
# gen_acortical_learning.py, which knows the selection indices directly -- this replaces
# the numerical identity join on per-traverse durations that used to recover it here.
# Colours are chosen for visibility as thin lines / small markers (all from Dark2 for
# consistency, avoiding the acortical raster hues Dark2(0-1)=teal/orange, the
# wildtype-like purple Dark2(2), the light Dark2 gold, and brown (too close to acortical
# orange)): the raster examples take grey, pink/magenta in order (so "Control 1" is grey,
# "Control 2" is the Dark2 pink/magenta), and the remaining cohort mouse takes green.
CONTROL_PALETTE = [plt.cm.Dark2(7), plt.cm.Dark2(3), plt.cm.Dark2(4)]  # grey, pink/magenta, green
# Loaded only for its row count and control_first_a ordering, which the colour map below
# is defined against -- the array itself is no longer plotted (E/F show no control).
control_cohort_duration = figure_data_dict["Control Mask A duration"]

# Map each cohort row -> colour: raster examples (in raster order) get the leading
# palette colours, non-example mice get the rest, so trace i and raster mouse i match.
control_example_cohort_idx = [int(row) for row in
                              control_manifest.sort_values("example")["cohort_row"]]
control_color_by_cohort = [None] * control_cohort_duration.shape[0]
for j, ci in enumerate(control_example_cohort_idx):
    control_color_by_cohort[ci] = CONTROL_PALETTE[j]
leftover = iter(CONTROL_PALETTE[len(control_example_cohort_idx):])
control_color_by_cohort = [c if c is not None else next(leftover) for c in control_color_by_cohort]

n_acortical = len(acortical_manifest)
n_control = len(control_manifest)
raster_colors = [plt.cm.Dark2(i) for i in range(n_acortical)] + \
                [control_color_by_cohort[ci] for ci in control_example_cohort_idx]
raster_tables = [acortical_meta[acortical_meta.example == k] for k in acortical_manifest["example"]] + \
                [control_meta[control_meta.example == k] for k in control_manifest["example"]]
plot_utils.plot_example_rasters_from_data(ax_raster, raster_tables, cmap=raster_colors, y_increment=0.13, markersize=config.MS_AREA_RASTER)
# set yticks to show genotype labels
animal_labels = [f"Acortical {idx+1}" for idx in range(n_acortical)] + [f"Control {idx+1}" for idx in range(n_control)]
ax_raster.set_yticklabels(animal_labels)
ax_raster.set_xlim(0, 220*60)

gs20 = gs0[2].subgridspec(1, 4, width_ratios=[0.7, 1, 1, 0.1], hspace=0.05)
# plot example traverses: (initial traverses)
# tpc
ax_tpc = FIG.add_subplot(gs20[0])
tiles_per_corridor = utils.select_by_prefix(figure_data_dict, config.GENOTYPES, "Mask A tiles per corridor")

kruskal_results =  utils.kruskal_with_pairwise_mann_whitney(tiles_per_corridor, alternative="greater")
plot_utils.plot_group_scatter_box_comparison(ax_tpc, tiles_per_corridor, kruskal_results,
                                             ylabel="Tiles/corridor", scatter_only=["Control"])

axes_traverses = [FIG.add_subplot(gs20[i+1]) for i in range(2)]


def traverse_tile_steps(meta, tiles, example, traverse_idx):
    """
    Return the tile rows of one traverse of an exported example session.

    Parameters
    ----------
    meta : pandas.DataFrame
        A family's ``"bout meta"`` table.
    tiles : pandas.DataFrame
        The matching ``"tile steps"`` table.
    example : int
        Positional index of the example session.
    traverse_idx : int
        0-based traverse number within that session, i.e. the index the old
        ``session.filter("traverse")`` list used.

    Returns
    -------
    pandas.DataFrame
        The traverse's rows of ``tiles``, ready for ``plot_utils.plot_tile_seq``.
    """
    rows = meta[(meta.example == example) & (meta.traverse_idx == traverse_idx)]
    bout_idx = rows["bout_idx"].iloc[0]
    return tiles[(tiles.example == example) & (tiles.bout_idx == bout_idx)]


# The acortical example whose traverses are drawn is raster row 0 (as before).
traverse_example, control_ref = 0, config.CONTROL_TRAVERSE_REF_ID
control_fps = control_manifest[control_manifest.example == control_ref].iloc[0].fps
control_tiles = figure_data_dict["Control A example tile steps"]
for i, ax in enumerate(axes_traverses):
    for k in range(10):
        color = plt.cm.viridis(k/10)
        plot_utils.plot_tile_seq(ax, traverse_tile_steps(acortical_meta, acortical_tiles,
                                                        traverse_example, k*2+i),
                                inverse=True, fps=acortical_row.fps, color=color,
                                alpha=1, linewidth=config.LW_DATA,)
    # add a reference from the chosen control (same per-mouse colour it gets in the raster / curves)
    plot_utils.plot_tile_seq(ax, traverse_tile_steps(control_meta, control_tiles,
                                                    control_ref, 8+i),
                            inverse=True, fps=control_fps,
                            color=control_color_by_cohort[control_example_cohort_idx[config.CONTROL_TRAVERSE_REF_ID]],
                            alpha=1, linewidth=config.LW_DATA)
for i, ax in enumerate(axes_traverses):
# format ax
    plot_utils.set_distance_plot_yaxis(ax, mask=mask_a, )
    ax.set_xlabel("Time to reward (s)")
    ax.set_xlim(left=-300, right=0)
# hide the y axis of the right plot
axes_traverses[1].yaxis.set_visible(False)
plot_utils.add_panel_title(axes_traverses[0], "Outbound traverses")
plot_utils.add_panel_title(axes_traverses[1], "Homebound traverses")

# add colorbar to the right
ax_cbar = FIG.add_subplot(gs20[-1])
plot_utils.plot_illustrative_cbar(ax_cbar, ticklabels=["Early", "Late"], label_loc="right")

# row 2: traverse duration and error plot.
# Only Acortical (n=4) and Wildtype (n=25) appear, as fitted curves + bootstrap CI bands.
# The n=3 control cohort is not summarised here at all: its animal-level bootstrap has only
# ~10 distinct resamples, so n=3 supports neither a meaningful CI nor a cohort exponential
# fit (which would overfit). Faint per-animal control traces used to stand in for that
# summary, but they obscured the acortical/wildtype contrast these panels exist to show.
# The control's Mask-A trajectory lives in the ac_oa_supp cohort means, and its per-animal
# fit parameters in ac_curve.
wildtype_fit_result = figure_data_dict["Wildtype two day duration fit results"][(1, "A")] # Day-1 Mask A reference
duration_fit_results = {**utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:1], "A duration fit results"),
                        "Wildtype": wildtype_fit_result}
gs30 = gs0[3].subgridspec(1, 3, width_ratios= [1, 1, 0.5])
ax_duration = FIG.add_subplot(gs30[0])
plot_utils.plot_curve_fit_comparison(ax_duration, result_dict=duration_fit_results,
                                     xlim=20, upper_y=400, xlabel="Traverse #", ylabel="Duration (s)", plot_scatter=False)
# error rate
wildtype_error = figure_data_dict["Wildtype two day turn error rate fit results"][(1, "A")] # Day-1 Mask A reference
error_fit_results = {**utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:1], "A turn error rate fit results"),
                     "Wildtype": wildtype_error}
ax_error = FIG.add_subplot(gs30[1])
plot_utils.plot_curve_fit_comparison(ax_error, result_dict=error_fit_results,
                                     xlim=20, upper_y=0.6, xlabel="Traverse #", ylabel="Turn error rate", plot_scatter=False)
ax_error.get_legend().remove()
# Add curve-derived ratio confidence intervals. Only the Wildtype/Acortical ratio
# is shown: the Control/Acortical ratio inherits the degenerate n=3 control bootstrap
# CI, so it is dropped along with the control fitted curve above.
ax_ratio = FIG.add_subplot(gs30[2])
summary = figure_data_dict["Acortical A genotype param ratios"]
sub_df = summary[summary.Comparison.str.contains("Wildtype")]
plot_utils.plot_ci_ratios(ax_ratio, sub_df,
                          param_latex=config.PARAM_LATEX, color=plot_utils.genotype_colors["Wildtype"])

plot_utils.add_panel_title(ax_ratio, "Wildtype/Acortical", color="black")

plot_utils.add_letter_labels(FIG, [(0.01, 0.99),  (0.01, 0.78), (0.01, 0.52), (0.26, 0.52), (0.01, 0.26), (0.39, 0.26), (0.77, 0.26)])
# # add a dashed line on FIG to divde mask D and Mask A panels
# FIG.add_artist(plt.Line2D([0.01, 0.99], [0.36, 0.36], color="black", linestyle="--", linewidth=0.5))
config.save_figure(FIG, "acortical_rapid.pdf", save_path)


# Supplementary D: select examples of Mask D traverses and reweard intervals
