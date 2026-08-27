import numpy as np
from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt

import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)
save_path = config.parse_save_path()

## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()
mask_colors = plot_utils.mask_colors


fig_width = 5.8
fig_height = 3.8
# Figure setup: two rows. Row 1 = cage-swap schematic + sorties per journey for the Mask A
# (West) and North (pooled) cohorts, one panel each. Row 2 = Mask A turn-error rate before
# and after cage relocation (West vs North), so the reader can judge whether the direction
# asymmetry survives the swap.
FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0)
gs00 = gs0[0].subgridspec(1, 5, width_ratios=[0.8, 0.05, 0.8, 1.1, 1.1])

# Row 1, Panel A: cage-relocation schematic (boxes drawn around each "Cage" label).
axes_a = [FIG.add_subplot(gs00[i]) for i in range(3)]
plot_utils.plot_schematic_cage_swap(axes_a, arrow_color="darkviolet", arrow_zorder=25,
                                    arrow_width_scale=1.1, arrow_length_scale=0.75,
                                    arrow_head_width=2, arrow_head_length=1.2)

# Row 1, Panels B & C: sorties per journey split by starting port (H-H = leaves/returns
# home, O-O = leaves/returns out), one panel per cohort. H-H vs O-O is compared within
# cohort with a paired Wilcoxon (unit = session); "(cage)" marks the cage-side port, which
# is the home (H) side in West but relocated to the O side under North. Boxes are colored by
# cohort; H-H vs O-O individual points are told apart by marker (filled up- vs open
# down-triangle). Independent y-axes: the two cohorts have very different baseline rates.
# (title, figure_data key, box color, x-tick labels marking the cage-side port).
sortie_panels = [
    ("Mask A (West)", "Wildtype A sortie count by direction", mask_colors["A"], ["H-H (cage)", "O-O"]),
    ("North (pooled)", "Single north pooled sortie count by direction", "black", ["H-H", "O-O (cage)"]),
]
sortie_markers = {"H-H": "^", "O-O": "v"}  # H-H filled up-, O-O open down-triangle
for j, (title, key, color, xticklabels) in enumerate(sortie_panels):
    ax = FIG.add_subplot(gs00[3 + j])
    arr = figure_data_dict[key]
    arr = arr[~np.isnan(arr).any(axis=1)]  # listwise deletion for the paired test
    data_dict = {"H-H": arr[:, 0], "O-O": arr[:, 1]}
    results = utils.friedman_with_pairwise_wilcoxon(data_dict)
    plot_utils.plot_group_scatter_box_comparison(
        ax, data_dict, results, colordict=color, markerdict=sortie_markers,
        open_markers={"O-O"}, markersize=config.MS_AREA_LARGE,
        ylabel="Sorties per journey" if j == 0 else "", plot_ns=True, upper_y=10)
    ax.set_xticklabels(xticklabels, fontsize=plot_utils.TICK_SIZE)  # annotate the cage-side port
    ax.text(0.5, 1, title, color=color, fontsize=plot_utils.FONT_SIZE,
            ha="center", va="bottom", transform=ax.transAxes)

# Row 2: Mask A turn error rate, West (main-text cohort) vs North (post cage relocation).
# The outbound (away from cage, filled ^) and homebound (toward cage, open v) legs are drawn
# as separate connected lines so the direction asymmetry is legible, interleaved along the
# shared x-axis (odd traverses = outbound, even = homebound). Both arrays use the same
# parity->direction convention (even cols = outbound); the legend labels are per-cohort so
# the "(cage)" tag lands on whichever port currently holds the relocated cage -- H in West
# (so homebound is toward the cage) and O in North (so the return leg is "To Out (cage)").
gs10 = gs0[1].subgridspec(1, 2, hspace=0.01)
axes_turns = [FIG.add_subplot(gs10[j]) for j in range(2)]
turn_panels = [("West", "Wildtype A turn error rate", ["Outbound", "Homebound (cage)"]),
               ("North", "Single north Day 2 A traverse turn error rate", ["To Home", "To Out (cage)"])]
max_traverses = 40  # cap the shared x-range so West (50 traverses) matches North (extracted to 40)
for j, (cond, key, dir_labels) in enumerate(turn_panels):
    ax = axes_turns[j]
    error_array = figure_data_dict[key][:, :max_traverses]
    out_array, home_array = error_array[:, ::2], error_array[:, 1::2]
    xs_out = np.arange(out_array.shape[1]) * 2 + 1   # odd traverse numbers
    xs_home = np.arange(home_array.shape[1]) * 2 + 2  # even traverse numbers
    color = mask_colors["A"]
    h_out = plot_utils.plot_direction_mean(ax, out_array, "outbound", color, xs=xs_out, markersize=config.MS_AREA_EMPHASIS, connect=True, label=dir_labels[0])
    h_home = plot_utils.plot_direction_mean(ax, home_array, "homebound", color, xs=xs_home, markersize=config.MS_AREA_EMPHASIS, connect=True, label=dir_labels[1])
    plot_utils.format_xs_ys(ax, utils.to_traverse_number(np.arange(error_array.shape[1])), xlabel="Traverse #", ylabel="Turn error rate", ylim=0.5)
    # Chance level: the approach-conditioned turn error rate has an exact 0.5 chance.
    ax.axhline(0.5, color="black", linestyle="--", linewidth=config.LW_HAIRLINE, zorder=config.Z_REFERENCE)
    ax.text(0.5, 1, f"Mask A ({cond})", color=color, fontsize=plot_utils.FONT_SIZE, ha="center", va="bottom", transform=ax.transAxes)
    ax.legend(handles=[h_out, h_home], labels=dir_labels,
              loc="upper right", bbox_to_anchor=(1, 1), fontsize=plot_utils.TICK_SIZE)
    if j == 1:  # share the West y-axis
        ax.set_yticklabels([])
        ax.set_ylabel("")

plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.45, 0.99), (0.73, 0.99),
                                   (0.01, 0.51), (0.52, 0.51)])

config.save_figure(FIG, "north_supp.pdf", save_path)
