from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)
save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 5.5

# data loading
figure_data_dict = utils.load_all_figure_data()
mask = figure_data_dict["masks"]["O"]

FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(3, 1, height_ratios=[1.1, 1, 1])

# === Row 1: maze schematics — A maze photo (+zoom inset), B Mask O top view + graphs, C hole decision ===
gs00 = gs0[0].subgridspec(1, 5, width_ratios=[1.8, 0.3, 0.6, 0.6, 1], wspace=0.05)
# Panel A: full maze photo, with a red dashed box marking the zoomed region and an inset of that region
ax_a = FIG.add_subplot(gs00[0])
ax_a.imshow(plt.imread(f"{config.DATA_DIR}/Top_view_labeled.png"), aspect="equal")
ax_a.axis("off")
x_min, y_min = 890, 260  # top-left corner of the region to zoom
width, height = 80, 190  # width and height of the region to zoom
ax_inset = FIG.add_subplot(gs00[1])
img = plt.imread(f"{config.DATA_DIR}/Top_view_labeled.png")
ax_inset.imshow(img[int(y_min):int(y_min + height), int(x_min):int(x_min + width)], aspect="equal")
ax_inset.axis("off")
# red dashed frame around the zoomed mouse, echoing the zoom box drawn on the full photo (panel A).
# inset coords are local to the crop (x: 0..width, y: 0..height); the pixel-edge extent is [-0.5, size-0.5].
ax_inset.add_patch(Rectangle((-0.5, -0.5), width, height, linewidth=config.LW_HAIRLINE, edgecolor="red",
                             facecolor="none", linestyle="--", clip_on=False))
ax_inset.margins(0)
rect = Rectangle((x_min, y_min), width, height, linewidth=config.LW_HAIRLINE, edgecolor="red", facecolor="none", linestyle="--")
ax_a.add_patch(rect)

# Panel B: Mask O top view + illustrative Home/Out colorbar + P10 corridor graph + tile graph
ax_b = FIG.add_subplot(gs00[2])
# draw the path without the built-in H/O labels (plot_ho=False); the big FONT_SIZE glyphs would land on
# top of the corridor tick labels in this narrow panel, so we place H/O ourselves clear of that lane
_, lc = mask.plot_with_shortest_path(ax_b, plot_ho=False, holes_list=None)
# stretch the path's viridis coloring across the full purple->yellow range so the top-view path matches
# the tile-graph turn arrows below (plot_tile_path_graph uses viridis 0..1); default norm stops short of yellow
lc.set_norm(plt.Normalize(0, len(lc.get_array()) - 1))
for i in range(11):  # corridor index labels 0-10, centered on each tile (i+0.5) so each aligns with its
    # corridor and sits just outside the grid rather than on the grid lines
    ax_b.text(i + 0.5, -0.55, str(i), fontsize=plot_utils.TICK_SIZE, color=plot_utils.mask_colors["O"],
              ha="center", va="center")
    ax_b.text(-0.55, i + 0.5, str(i), fontsize=plot_utils.TICK_SIZE, color=plot_utils.mask_colors["O"],
              ha="center", va="center", rotation=90)
# Home marker to the left of the tick lane (row 5 = home gap); Out marker at the top of the vertical path
ax_b.text(-1.6, 5.5, "H", fontsize=plot_utils.FONT_SIZE, color="tab:blue", ha="center", va="center")
ax_b.text(5.5, 10.3, "O", fontsize=plot_utils.FONT_SIZE, color="tab:orange", ha="center", va="center")
gs01 = gs00[3].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.0)
ax_c = FIG.add_subplot(gs01[0])
plot_utils.plot_schematic_path_graph(ax_c, n_nodes=1, plot_shortest_path=True, anchor="S")  # P10 graph, sunk to cell bottom
ax_c2 = FIG.add_subplot(gs01[1])
plot_utils.plot_tile_path_graph(ax_c2, holes_list=mask.holes_list, anchor="N")  # tile graph, raised to cell top

# Panel C: the four decisions at a hole
ax_d = FIG.add_subplot(gs00[4])
plot_utils.plot_hole_decision_schematic(ax_d, linewidth=config.LW_HAIRLINE)  # H/O labels sit at the arrow ends (see plot_schematics)

# === Row 2: D Mask O reward intervals, E Mask A reward intervals, F Mask A example speed ===
# Panel order follows the Results narrative (reward intervals before speed).
gs10 = gs0[1].subgridspec(1, 3, width_ratios=[1, 1, 1])
# Panel D: Mask O reward intervals
ax_o_int = FIG.add_subplot(gs10[0])
interval_array = figure_data_dict["Wildtype O reward intervals"]
plot_utils.plot_array_data(ax_o_int, interval_array, stats_type="mean", scatter_colors=[plot_utils.mask_colors["O"]],
                           plot_shade=True, connect_scatters=True, ylim=15 * 60, markersize=config.MS_AREA_SMALL)
ax_o_int.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=plot_utils.TICK_SIZE)
# Panel E: Mask A average reward intervals
ax_int = FIG.add_subplot(gs10[1])
interval_array = figure_data_dict["Wildtype A reward intervals"]
plot_utils.plot_array_data(ax_int, interval_array, stats_type="mean", scatter_colors=[plot_utils.mask_colors["A"]],
                           line_color=plot_utils.mask_colors["A"], plot_shade=True, connect_scatters=True, ylim=20 * 60)
# Panel F: example-session speed profile (Mask A)
ax_speed = FIG.add_subplot(gs10[2])
# Step times are derived from the exported tile rows (R8); binning stays a panel choice.
example_tiles = figure_data_dict["Mask A example tile steps"]
example_manifest = figure_data_dict["Mask A example manifest"]
speed_row = example_manifest[example_manifest.example == config.MASK_A_EXAMPLE_ID].iloc[0]
plot_utils.plot_speed_hist(
    ax_speed, utils.derive_step_times(example_tiles[example_tiles.example == config.MASK_A_EXAMPLE_ID],
                                     speed_row.fps),
    plot_utils.mask_colors["A"], session_span_s=speed_row.session_span_s,
    # bw/tm in SECONDS. 3-minute bins, matching the caption and plot_d_supp.py's
    # sibling panel; in_maze_end_s (~91 min here) clamps tm, so the panel spans the
    # whole in-maze session.
    in_maze_end_s=speed_row.in_maze_end_s, bw=60 * 3, tm=120 * 60)

# === Row 3: G Mask A group traverse speed, H Mask A sortie counts, I first-journey forward bias ===
gs20 = gs0[2].subgridspec(1, 3)
# Panel G: Mask A group traverse speed (all animals)
ax_group_sp = FIG.add_subplot(gs20[0])
speed_array = figure_data_dict["Wildtype A traverse speed"]
plot_utils.plot_array_data(ax_group_sp, speed_array, stats_type="mean", scatter_colors=[plot_utils.mask_colors["A"]],
                           line_color=plot_utils.mask_colors["A"], plot_shade=True, connect_scatters=True,
                           xlabel="Traverse #", ylabel="Speed (tiles/s)", ylim=3)
ax_group_sp.legend(loc="lower right", bbox_to_anchor=(1, 0), fontsize=plot_utils.TICK_SIZE)
# Panel H: sortie counts
ax_sorties = FIG.add_subplot(gs20[1])
sortie_array = figure_data_dict["Wildtype A sortie counts"]
plot_utils.plot_array_data(ax_sorties, sortie_array, stats_type="mean", scatter_colors=[plot_utils.mask_colors["A"]],
                           line_color=plot_utils.mask_colors["A"], plot_shade=True, connect_scatters=True,
                           ylabel="N(sorties)", ylim=30)

# Panel I: empirical forward bias beta_hat over the merged first journey (Mask A). Reversal-based estimator
# (eq:betahat); dashed black = the memoryless random walker (beta=0.5). Cohort mean +/- SE, computed in
# gen_wildtype_two_day_data.py from one 20%-of-journey window per point, with the window as the only
# smoothing (no moving average on top). Only fully-supported positions are reported (mode="valid", as for
# the smoothed lines of fig:ac_mem_gen A), so the line runs x=0.118..0.882 inside the 0-1 journey axis
# rather than reaching the ends on half-width windows.
ax_beta = FIG.add_subplot(gs20[2])
bx, bmean, bse = figure_data_dict["Wildtype A first journey forward bias"]
ok = ~np.isnan(bmean)
maskA_color = plot_utils.mask_colors["A"]
ax_beta.axhline(0.5, ls="--", color="black", lw=config.LW_HAIRLINE, zorder=config.Z_REFERENCE, label="random")
ax_beta.plot(bx[ok], bmean[ok], color=maskA_color, lw=config.LW_DATA)
ax_beta.fill_between(bx[ok], (bmean - bse)[ok], (bmean + bse)[ok], color=maskA_color, alpha=0.25, lw=0)
ax_beta.set_xlim(0.1, 0.9)
ax_beta.set_ylim(0, 0.8)
ax_beta.set_xlabel("fraction of the 1st journey")
ax_beta.set_ylabel(r"Forward bias $\hat{\beta}$")
ax_beta.legend(loc="upper left", fontsize=plot_utils.TICK_SIZE)

plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.49, 0.99), (0.76, 0.99),
                                   (0.01, 0.65), (0.34, 0.65), (0.67, 0.65),
                                   (0.01, 0.32), (0.34, 0.32), (0.67, 0.32)])

config.save_figure(FIG, "oa_supp.pdf", save_path)
