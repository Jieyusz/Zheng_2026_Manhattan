import numpy as np
from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)

save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 5
n_sessions = 4  # number of Day 2 sessions

## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()

# Plot the figure
FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.15)
gs0a = gs0[0].subgridspec(3, 1, hspace=0.05,  height_ratios=[0.9, 1, 1.8])

# row 1 panel A: Mask B and C top view and timeline
gs00 = gs0a[0].subgridspec(1, 3, width_ratios=[1, 1, 1.3])
# Panel A: Tile and corridor errors
axes_a = [FIG.add_subplot(gs00[i]) for i in range(2)]
for i, mask_name in enumerate(["B", "C"]):
    mask = figure_data_dict["masks"][mask_name]
    ax = axes_a[i]
    # plot_with_shortest_path draws the "Mask <name>" heading itself, in the mask colour
    _, lc = mask.plot_with_shortest_path(ax, plot_ho=True, holes_list=None) # plot Home and Out for marking
    ax.axis("off")
axes_a[0].text(0.5, -0.05, "LRLRRRLLL", fontsize=plot_utils.FONT_SIZE, color=plot_utils.mask_colors["B"], transform=axes_a[0].transAxes,
                fontweight="bold", va="top", ha="center")
axes_a[1].text(0.5, -0.05, "LRRRLLRLL", fontsize=plot_utils.FONT_SIZE, color=plot_utils.mask_colors["C"], transform=axes_a[1].transAxes,
                fontweight="bold", va="top", ha="center")# Add colorbar to the right of the last panel

# Row 1 Panel B: schematic for the two-day experiment
ax_b = FIG.add_subplot(gs00[-1])
plot_utils.plot_d2_session_timeline(ax_b, origin=[0, 0])

# Row 2: compare early and late traverses of Day1 and Day2-1 (Mask A)

gs10 = gs0a[1].subgridspec(1, 2, width_ratios=[1, 1])
axes_c = [FIG.add_subplot(gs10[i]) for i in range(2)]
# left: Day1 1st and 20th
# find the Day1 Mask A based on Day2-1's A (examine repetition)

trajectory_colors = [("Day 1 #",plt.cm.Dark2(0)), ("Day 1 #", plt.cm.Dark2(1)), ("Day 2.1 #",plt.cm.Dark2(2))]
# separate outbound and homebound parameters
# The overnight traverses are exported as flat tables (R8). The nested
# [direction][set][animal] structure survives as index columns, and only animal_idx 0 is
# drawn (as before).
overnight_meta = figure_data_dict["Overnight traverse example bout meta"]
overnight_tiles = figure_data_dict["Overnight traverse example tile steps"]
for i, (ax, direction) in enumerate(zip(axes_c, ["outbound", "homebound"])):
    traverse_set = overnight_meta[(overnight_meta.direction == direction)
                                  & (overnight_meta.animal_idx == 0)] # only show one animal
    for _, bout_row in traverse_set.sort_values("set_idx").iterrows():
        label, color = trajectory_colors[int(bout_row.set_idx)]
        plot_utils.plot_tile_seq(ax, overnight_tiles[overnight_tiles.example == bout_row.example],
                                inverse=True, fps=bout_row.fps, color=color, alpha=1.0,
                                linewidth=config.LW_DATA, label=f"{label}{int(bout_row.label)+1}")
    # format ax
    plot_utils.set_distance_plot_yaxis(ax, mask=figure_data_dict["masks"]["A"], )
    ax.set_xlabel("Time to reward (s)")
    ax.set_xlim(left=-200, right=0)
    by_labels = plot_utils.get_legend_objects_as_dict(ax)
    ax.legend(by_labels.values(), by_labels.keys(), fontsize=plot_utils.TICK_SIZE, handler_map={tuple: HandlerTuple(ndivide=None)},)

# hide the y axis of the right plot
axes_c[1].yaxis.set_visible(False)
plot_utils.add_panel_title(axes_c[0], "Outbound traverses")
plot_utils.add_panel_title(axes_c[1], "Homebound traverses")


# Row 4: curve fit results for predictions
day1_x = 30
day2_x = 12
gs30 = gs0a[2].subgridspec(2, 5, hspace=0.05, height_ratios=[1, 1], width_ratios=[day1_x, day2_x, day2_x, day2_x, day2_x,])

def plot_two_day_curve_fit(axes, fit_results, upper_y=150, ylabel="Duration (s)"):
    """
    Plot fitted learning curves with CI bands across two-day sessions.

    Each ``(session, mask)`` fit-results entry draws its bootstrap-median curve and CI
    band on the axis for that session (see ``docs/data_contracts.md`` §"Fitted curve
    tuple"). The x-axis is the 1-based traverse number; Day1 spans a wider traverse
    range (``day1_x``) than the Day-2 sessions (``day2_x``).

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        One axis per session; ``axes[s-1]`` receives session ``s`` (1-based session
        index). Index 0 is Day1; indices 1-4 are the four Day-2 sessions.
    fit_results : dict
        Maps ``(session, mask)`` -> fit-results tuple
        ``(bs, ds, summary_df, bootstrap_curves)``. ``session`` is 1-based; ``mask`` is
        one of ``"A"``, ``"B"``, ``"C"``.
    upper_y : float, optional
        Upper y-axis limit, in the metric's units (seconds for duration).
    ylabel : str, optional
        Y-axis label for the Day-1 axis.

    Returns
    -------
    None
        Draws onto ``axes`` in place.
    """
    for (s, m), (bs, ds, summary_df, bootstrap_curves) in fit_results.items():
        ax = axes[s - 1]
        color = plot_utils.mask_colors[m]
        # --- Part 1: Fit and Confidence Intervals ---
        plot_utils.plot_fitted_curve_and_confidence(ax, *bootstrap_curves, label=f"Mask {m}", color=color, linewidth=config.LW_EMPHASIS)

    axes[0].set_xlabel("Traverse #")
    axes[0].set_xlim(0.5, day1_x + 0.5)
    axes[0].set_ylim(0, upper_y)
    axes[0].set_ylabel(ylabel)
    for i in range(4):
        axes[i + 1].set_xlim(0.5, day2_x + 0.5)
        axes[i+1].set_ylim(0, upper_y)
        axes[i+1].set_ylabel("")
        axes[i+1].set_yticks([])

# first two panels: plot fit results for traverse duration
fit_axes = [[FIG.add_subplot(gs30[i, j]) for j in range(n_sessions+1)] for i in range(2)]
duration_fit_results = figure_data_dict["Wildtype two day duration fit results"]
plot_two_day_curve_fit(fit_axes[0], duration_fit_results, upper_y=150)
plot_utils.add_panel_title(fit_axes[0][0], "Day 1")
# add equation
fit_axes[0][0].text(1, 0.8, r"$D= D_{\infty} + \left(D_0 - D_{\infty}\right)\exp\left[-\delta(b-1)\right]$",
                    fontsize=plot_utils.TICK_SIZE, ha="right", va="bottom", transform=fit_axes[0][0].transAxes)
for ax in fit_axes[0]:
    ax.xaxis.set_visible(False)
by_labels = plot_utils.get_legend_objects_as_dict(fit_axes[0][-1], sort=True)
FIG.legend(handles=by_labels.values(), labels=by_labels.keys(), bbox_to_anchor=(0.08, 0.24), loc="lower left", ncol=6, frameon=True,
                      fontsize=plot_utils.TICK_SIZE,
                      handler_map={tuple: HandlerTuple(ndivide=None)},)
for i in range(n_sessions):
    ax = fit_axes[0][i + 1]
    plot_utils.add_panel_title(ax, f"Day 2.{i + 1}")

te_fit_results = figure_data_dict["Wildtype two day turn error rate fit results"]
plot_two_day_curve_fit(fit_axes[1], te_fit_results, upper_y=0.5, ylabel="Turn error rate")
# add equation
fit_axes[1][0].text(1, 0.8, r"$E = E_{\infty} + \left(E_0 - E_{\infty}\right)\exp\left[-\epsilon(b-1)\right]$",
                    fontsize=plot_utils.TICK_SIZE, ha="right", va="bottom", transform=fit_axes[1][0].transAxes)

# curve-derived ratios:
# right column: add the curve-derived summary ratios to explain generalization (panel G)
gs0b = gs0[1].subgridspec(3, 1) # curve-derived ratio
ax_ratio = FIG.add_subplot(gs0b[0])
summary = figure_data_dict["Wildtype two day param ratios"]
overnight = summary[(summary.Session=="2")&(summary.Mask=="A")]
plot_utils.plot_ci_ratios(ax_ratio, overnight, param_latex=config.PARAM_LATEX, color=plot_utils.mask_colors["A"])
plot_utils.add_panel_title(ax_ratio, "Overnight\nMask A Day 2.1/Day 1", color=plot_utils.mask_colors["A"])
ax_ratio.set_xlabel("")

ax_bc = FIG.add_subplot(gs0b[2])
bvc_df = figure_data_dict["Wildtype day21 mask BC param ratios"]
plot_utils.plot_ci_ratios(ax_bc, bvc_df, color="black", param_latex=config.PARAM_LATEX)
plot_utils.add_panel_title(ax_bc, "Turn sequence\nDay 2.1 B/C", color="black")
ax_bc.set_xlabel("")

# generaliation
ax_gen = FIG.add_subplot(gs0b[1])
gen_median = figure_data_dict["Wildtype two day mask param ratios"]
for i, mask in enumerate(["A", "B", "C"]):
    sub_df = gen_median[gen_median.Mask==mask]
    plot_utils.plot_ci_ratios(ax_gen, sub_df, color=plot_utils.mask_colors[mask],
                              param_latex=config.PARAM_LATEX, offset=0.2*(i-1))
plot_utils.add_panel_title(ax_gen, "Generalization\nDay 2.2+/Day 1", color="black")


plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.42, 0.99), (0.01, 0.75),
                                    (0.01, 0.48), (0.01, 0.28), (0.76, 0.99), (0.76, 0.67), (0.76, 0.31)],)

config.save_figure(FIG, "day2.pdf", save_path)
