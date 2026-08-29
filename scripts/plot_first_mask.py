import numpy as np
from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib import patches as mpatches
import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)

figure_data_dict = utils.load_all_figure_data()
save_path = config.parse_save_path()
fig_width = 5
fig_height = 6.3
example_id = config.MASK_A_EXAMPLE_ID

# Plot the figure
FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(5, 1, hspace=0.2,  height_ratios=[0.8, 0.8, 1, 0.75,  1])

# Row 1: all schematics of Mask A
gs00 = gs0[0].subgridspec(1, 5, wspace=0.1, width_ratios=[3,3,3,3, 0.2])
# Panel A: 3D maze
ax_a = FIG.add_subplot(gs00[0])
ax_a = plot_utils.plot_schematic_3d_maze(ax_a) # plot 3D maze schematic
# Panel B: top view
mask = figure_data_dict["masks"]["A"]
ax_b = FIG.add_subplot(gs00[1])
ax_b, lc = mask.plot_with_shortest_path(ax=ax_b, holes_list=None, plot_ho=True, path_linewidth=config.LW_EMPHASIS) # plot with shortest path
ax_b.text(0.5, -0.05, "LRLRRRLLL", fontsize=plot_utils.FONT_SIZE,
          color=plot_utils.mask_colors["A"], transform=ax_b.transAxes,
               fontweight="bold", va="top", ha="center")
# add text for H and O
cax = FIG.add_subplot(gs00[-1])
plot_utils.plot_illustrative_cbar(cax, ticks=np.arange(0, mask.size, 2)/mask.size, ticklabels=["Home", 2, 4, 6, 8, "Out"],
                                  label_loc="right", aspect=20) # plot colorbar for the masks
# panel C: P10 graph
ax_c = FIG.add_subplot(gs00[2])
ax_c = plot_utils.plot_schematic_path_graph(ax_c, plot_shortest_path=True, path_linewidth=config.LW_EMPHASIS) # plot P10 graph schematic
# Panel c2; tile graph
ax_c2 = FIG.add_subplot(gs00[3])
plot_utils.plot_tile_path_graph(ax_c2, holes_list=mask.holes_list, tile_width=0.8, plot_HO=True)
# Row 2: reward interval raster
# The example sessions are held as flat per-bout/per-tile tables (R8): "bout meta" has
# one row per bout (durations, traverse index, reward flags), "bout steps"/"tile steps"
# the per-step paths, and "manifest" the session scalars. Every bout of every example is
# exported, so the panels below are row selections rather than separate caches.
example_meta = figure_data_dict["Mask A example bout meta"]
example_steps = figure_data_dict["Mask A example bout steps"]
example_tiles = figure_data_dict["Mask A example tile steps"]
example_manifest = figure_data_dict["Mask A example manifest"]

gs10 = gs0[1].subgridspec(1,1)
ax_d = FIG.add_subplot(gs10[0])
plot_utils.plot_example_rasters_from_data(
    ax_d, [example_meta[example_meta.example == k] for k in example_manifest["example"]],
    cmap=plt.cm.tab10, y_increment=0.13, markersize=config.MS_AREA_RASTER)
ax_d.set_xlim(right=6300)
# Box the example segment (shown in panel E) on the example animal's raster row.
# The raster x-axis is cumulative in-maze time (sleep-thresholded bout durations),
# so the box must end at the segment's in-maze duration -- not a fixed wall-clock
# width. The segment starts at bout 0, so the cumulative duration already recorded at
# its last bout is the right edge in the raster's own time base.
example_rows = example_meta[example_meta.example == example_id]
segment_meta = example_rows[example_rows.bout_idx < config.MASK_A_SEGMENT_BOUTS]
box_end_s = segment_meta["cum_duration_s"].iloc[-1]
ax_d.add_patch(mpatches.Rectangle((0, example_id+1-0.3), box_end_s, 0.6, fill=False, color="black", linewidth=config.LW_HAIRLINE))

# Row 3: trajectory of the example mouse of mask A
# Panel E: the distance over time for the first config.MASK_A_SEGMENT_BOUTS (29)
# bouts using one example animal
gs00 = gs0[2].subgridspec(1, 1)
ax_e = FIG.add_subplot(gs00[0])
segment_tiles = example_tiles[(example_tiles.example == example_id)
                             & (example_tiles.bout_idx < config.MASK_A_SEGMENT_BOUTS)]
# Session.slice() takes its first_frame from the first bout's first tile, not from the
# parent session, so the segment's clock starts there rather than at the session start.
segment_ref_frame = segment_tiles["in_frame"].iloc[0]
example_fps = example_manifest.loc[example_manifest.example == example_id, "fps"].iloc[0]
plot_utils.plot_tile_distance(
    ax_e, utils.derive_tile_distance_table(segment_tiles, segment_meta, example_fps,
                                          segment_ref_frame),
    reward_color=plt.cm.tab10(example_id)) # plot distance over time for the example session
plot_utils.set_distance_plot_yaxis(ax_e, mask) # set y-axis for distance plot
ax_e.set_xlim(left=0)

# Row 4: example traverses
gs20 = gs0[3].subgridspec(1, 5, width_ratios=[0.2, 3, 3, 3, 3], wspace=0.1)
axes_f = [FIG.add_subplot(gs20[i]) for i in range(5)]
for ax, tr_idx in zip(axes_f[1:], config.EXAMPLE_TRAVERSE_INDICES):
    bout_row = example_rows[example_rows.traverse_idx == tr_idx].iloc[0]
    path_df = example_steps[(example_steps.example == example_id)
                            & (example_steps.bout_idx == bout_row.bout_idx)]
    plot_utils.plot_bout_path(ax, utils.derive_bout_path_table(path_df, bout_row), mask,
                              plot_colorbar=False, plot_duration=True, plot_symbol=True,
                              linewidth=config.LW_EMPHASIS, marker_color=plt.cm.tab10(example_id), title="")
    plot_utils.add_panel_title(ax, f"{utils.to_traverse_number(tr_idx)}")
plot_utils.plot_illustrative_cbar(axes_f[0], aspect=20)
axes_f[1].text(-0.1, 1.01, "Traverse #", ha="left", va="bottom", fontsize=plot_utils.TICK_SIZE, transform=axes_f[1].transAxes)

# Row 5: duration and errors vs. tile and corridor in Mask A
gs30 = gs0[4].subgridspec(1, 2, width_ratios=[1.1, 1],)
scatter_colors=["tab:grey", "tab:red"]
# twin axes for each plot
ax_g = FIG.add_subplot(gs30[0])
# plot duration

d_out, d_home, _, _ = plot_utils.plot_array_data(ax_g, figure_data_dict["Wildtype A traverse duration"], stats_type="mean", markersize=config.MS_AREA_DEFAULT,
                                                 scatter_colors=[scatter_colors[0]], line_color=scatter_colors[0], plot_shade=True, connect_scatters=True, shade_alpha=1,
                                                 ylabel="Duration (s)", ylim=200, xlabel="Traverse #", ) # use the same y-lim for all cases
ax_g2 = ax_g.twinx()
plot_utils.plot_phase_lines(ax_g, [2.5, 22.5], color=scatter_colors[0])
# plot turn error rate
e_out, e_home, _, _ = plot_utils.plot_array_data(ax_g2, figure_data_dict["Wildtype A turn error rate"], stats_type="mean",
                                                 scatter_colors=[scatter_colors[1]], markersize=config.MS_AREA_DEFAULT,
                                                 line_color=scatter_colors[1], plot_shade=True, connect_scatters=True, shade_alpha=1,
                                                 xlabel="Traverse #", ylabel="Turn error rate", ylim=0.5)
# format axes color
plot_utils.format_yaxis_color(ax_g, scatter_colors[0])
plot_utils.format_yaxis_color(ax_g2, scatter_colors[1], spine_loc="right")
ax_g.legend(handles=[(d_out, e_out), (d_home, e_home)],
          labels=["Outbound", "Homebound"], handler_map={tuple: HandlerTuple(ndivide=None)},
          loc="upper right", bbox_to_anchor=(1, 1), fontsize=plot_utils.TICK_SIZE)

# panel H: corridor and tile error RATES -- the per-step fraction of moves that fail to
# progress toward the goal (same >=0 definition as the error_propagation corridor rate; chance
# ~0.5). Both are bounded [0,1], so they share ONE axis with a dashed 0.5 chance line, replacing
# the old broken/twin count axes. Corridor = grey, tile = red; outbound (^) / homebound (v).
ax_h = FIG.add_subplot(gs30[1])
c_out, c_home, _, _ = plot_utils.plot_array_data(ax_h, figure_data_dict["Wildtype A Corridor error rate array"], stats_type="mean",
                                                 scatter_colors=[scatter_colors[0]], line_color=scatter_colors[0], plot_shade=True, connect_scatters=True, markersize=config.MS_AREA_DEFAULT,
                                                 ylabel="Error rate", ylim=0.5, xlabel="Traverse #")
t_out, t_home, _, _ = plot_utils.plot_array_data(ax_h, figure_data_dict["Wildtype A tile error rate array"], stats_type="mean",
                                                 scatter_colors=[scatter_colors[1]], line_color=scatter_colors[1], plot_shade=True, connect_scatters=True, markersize=config.MS_AREA_DEFAULT,
                                                 xlabel="Traverse #", ylabel="Graph error rate")
plot_utils.plot_phase_lines(ax_h, [2.5, 22.5])
# legend distinguishes the two metrics (colour); outbound/homebound (^/v) follow panel G.
ax_h.legend(handles=[(c_out, c_home), (t_out, t_home)], labels=["Corridor", "Tile"],
            handler_map={tuple: HandlerTuple(ndivide=None)}, loc="upper right",
            fontsize=plot_utils.TICK_SIZE)
# ToDO: add dash line of coresponding color for chance level for graph error rate (0.44 for corridor, for example). Apply this convention to all graph error rate. Turn error rate null is 0.5. also mark that. The random/dashed line should not appear in legend, but should be noted in caption.
# Format labels
plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.22, 0.99), (0.45, 0.99),
                                   (0.01, 0.81), (0.01, 0.63), (0.01, 0.42), (0.01, 0.23), (0.55, 0.23)])

# Save the figure
config.save_figure(FIG, "first_mask.pdf", save_path)