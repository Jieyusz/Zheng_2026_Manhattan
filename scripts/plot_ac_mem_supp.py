from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)

save_path = config.parse_save_path()
fig_width = 5.5
fig_height = 4

figure_data_dict = utils.load_all_figure_data()

FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(3,1, height_ratios=[1, 1, 1], wspace=0.01, hspace=0.01)

gs10 = gs0[0].subgridspec(1, 1)
ax_o = FIG.add_subplot(gs10[0])
o_intervals = [((0, 0), figure_data_dict["Acortical Mask O reward intervals"][:, :20])] + figure_data_dict["Acortical Mask O memory intervals"]
plot_utils.plot_grouped_memory(ax_o, o_intervals, upper_y=50*60, yunit="Interval (s)", xunit="Reward",
                               stats_type="mean", plot_shade=True, markersize=config.MS_AREA_SMALL,
                               scatter_colors=[plot_utils.mask_colors["O"]], connect_scatters=True)
ax_o.set_title("Mask O", fontsize=plot_utils.FONT_SIZE, color=plot_utils.mask_colors["O"])

# speed in Mask A
gs20 = gs0[1].subgridspec(1, 4, width_ratios=[5, 1, 1, 1])
ax_speed = FIG.add_subplot(gs20[0])
# plot individual memory
a_metric_dict = {"Speed (tile/s)": figure_data_dict["Acortical Mask A example memory speed"]}
a_days = figure_data_dict["Acortical Mask A example memory days"]
plot_utils.plot_individual_memory(ax_speed, a_metric_dict, utils.moving_average, a_days, linewidth=config.LW_DATA, markersize=config.MS_AREA_DEFAULT,)
ax_speed.set_ylim(0, 6)

# relative speed
speed_axes = [FIG.add_subplot(gs20[0, i+1]) for i in range(3)]
ratio_dict = utils.select_by_prefix(figure_data_dict, config.GENOTYPES, "Mask A speed relative ratio")
raw_mean_dict = utils.select_by_prefix(figure_data_dict, config.GENOTYPES, "Mask A speed gap data points")
plot_utils.plot_relative_memory(speed_axes, ratio_dict, raw_mean_dict, format_title=True,
                       ylabel="Relative speed")
for i, ax in enumerate(speed_axes):
    ax.set_ylim([0, 3])
    ax.xaxis.set_tick_params(rotation=45)

# plot example traverses. These are exported as flat per-bout tables (R8); the meta
# table's `label` column is the *day* each first traverse came from, not a traverse index.
example_meta = figure_data_dict["Acortical mem traverse example bout meta"]
example_steps = figure_data_dict["Acortical mem traverse example bout steps"]
mask_a = figure_data_dict["masks"]["A"]
gs10 = gs0[2].subgridspec(1, len(example_meta)+1, width_ratios=[1]*len(example_meta)+[0.1], wspace=0.01)
axes_traj = [FIG.add_subplot(gs10[i]) for i in range(len(example_meta))]
for ax, (day, path_df) in zip(axes_traj, utils.iter_example_bout_paths(example_steps, example_meta)):
    plot_utils.plot_bout_path(ax, path_df, mask_a, plot_colorbar=False, plot_duration=True,
                              plot_symbol=True, linewidth=config.LW_DATA, noise=0.15,
                              marker_color=plot_utils.genotype_colors["Acortical"], title="")
    ax.text(0.5, 1, f"Day{day+1}", ha="center", va="bottom", fontsize=plot_utils.TICK_SIZE, transform=ax.transAxes)
# add a colorbar
plot_utils.plot_illustrative_cbar(FIG.add_subplot(gs10[-1]), aspect=10, label_loc="right")

plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.01, 0.66), (0.56, 0.66), (0.01, 0.35)])
config.save_figure(FIG, "ac_mem_sup.pdf", save_path)