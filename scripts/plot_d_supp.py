from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
import numpy as np
# add arg parser for saving figure path
import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)
save_path = config.parse_save_path()
fig_width = 5
fig_height = 4

## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()
interval_array = figure_data_dict["Wildtype D reward intervals"]
n_all_d_animals = interval_array.shape[0]
FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(3, 1, height_ratios=[0.8, 1, 1], hspace=0.05)
# Row 1: raster of rewards
gs00 = gs0[0].subgridspec(1, 1)
ax_a = FIG.add_subplot(gs00[0])
# The example sessions are exported as flat per-bout/per-tile tables (R8): "bout meta"
# carries the cumulative in-maze times and reward flags the raster needs.
example_meta = figure_data_dict["Mask D example bout meta"]
example_manifest = figure_data_dict["Mask D example manifest"]
all_colors = [plt.cm.tab10(i) for i in range(n_all_d_animals)]
plot_utils.plot_example_rasters_from_data(
    ax_a, [example_meta[example_meta.example == k] for k in example_manifest["example"]],
    cmap=all_colors[-3:], y_increment=0.13, markersize=config.MS_AREA_RASTER)

gs10 = gs0[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.05)
# plot supplementary: reward intervals and traverse speed
ax_b = FIG.add_subplot(gs10[0])

out1, home1, _, se1 = plot_utils.plot_array_data(ax_b, interval_array, stats_type="mean", scatter_colors=[plot_utils.mask_colors["D"]],
                                                 line_color= plot_utils.mask_colors["D"],
                                                  plot_shade=True, connect_scatters=True, ylim=40*60)
plot_utils.plot_phase_lines(ax_b, [2.5, 20.5])
ax_b.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=plot_utils.TICK_SIZE)
# add n sorties
ax_c = FIG.add_subplot(gs10[1])
sortie_array = figure_data_dict["Wildtype D sortie counts"]
plot_utils.plot_array_data(ax_c, sortie_array, stats_type="mean", scatter_colors=[plot_utils.mask_colors["D"]],
                           line_color=plot_utils.mask_colors["D"], plot_shade=True, connect_scatters=True,
                           ylabel="N(sorties)", ylim=60)
ax_c.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=plot_utils.TICK_SIZE)

# Add traverse speed
gs20 = gs0[2].subgridspec(1, 3, width_ratios=[2, 2, 1.5], wspace=0.05)
# add individual mouse speed
ax_d = FIG.add_subplot(gs20[0])
# example of individual
# The speed histogram is binned here, not in gen: bw/tm are panel choices. The step-time
# point process is derived from the exported tile rows.
example_tiles = figure_data_dict["Mask D example tile steps"]
speed_row = example_manifest[example_manifest.example == config.MASK_D_EXAMPLE_ID].iloc[0]
plot_utils.plot_speed_hist(
    ax_d, utils.derive_step_times(example_tiles[example_tiles.example == config.MASK_D_EXAMPLE_ID],
                                 speed_row.fps),
    all_colors[-3:][config.MASK_D_EXAMPLE_ID],
    session_span_s=speed_row.session_span_s, in_maze_end_s=speed_row.in_maze_end_s,
    # bw/tm are in SECONDS: 3-minute bins over the whole in-maze session. tm=None lets
    # binned_step_rate fall back to in_maze_end_s, which this example overruns (~249 min)
    # -- a fixed tm would silently truncate the panel and contradict the caption.
    bw=3 * 60, tm=None) # same example animal as plot_d_motif

ax_e = FIG.add_subplot(gs20[1])
speed_array = figure_data_dict["Wildtype D speed"]
plot_utils.plot_array_data(ax_e, speed_array, stats_type="mean", scatter_colors=[plot_utils.mask_colors["D"]], line_color=plot_utils.mask_colors["D"],
                           plot_shade=True, connect_scatters=True, xlabel="Traverse #",
                           ylabel="Speed (tiles/s)", ylim=3)

# First-journey bottleneck timing: time from the start of the first bout to the
# first bottleneck encounter (x) vs. time from the last bottleneck visit to the
# first reward (y), one point per animal (data from gen_wildtype_d_data).
ax_time = FIG.add_subplot(gs20[2])
timing = figure_data_dict["Wildtype D first journey timing"]
valid = ~np.isnan(timing).any(axis=1)
# color each point by its animal index (all_colors is tab10-by-index), so the
# example animals in panel A carry the same color here.
point_colors = [all_colors[i] for i in np.flatnonzero(valid)]
ax_time.scatter(timing[valid, 0], timing[valid, 1], color=point_colors, s=config.MS_AREA_LARGE, )
ax_time.set_xlabel("Time to bottleneck (s)", fontsize=plot_utils.TICK_SIZE)
ax_time.set_ylabel("Bottleneck to reward (s)", fontsize=plot_utils.TICK_SIZE)

plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.01, 0.71), (0.51, 0.71), (0.01, 0.37), (0.35, 0.37), (0.70, 0.37),],)

config.save_figure(FIG, "d_supp.pdf", save_path)