from matplotlib.figure import Figure

from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)
save_path = config.parse_save_path()
fig_width = 5.5
fig_height = 4.5

## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()
ef_color_dict = {"Acortical E":plot_utils.mask_colors["E"], "Acortical F":plot_utils.mask_colors["F"],
                 "Control E": plot_utils.genotype_colors["Control"]}
FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(3, 1, height_ratios=[0.8, 1, 1])
gs00= gs0[0].subgridspec(1, 7, width_ratios=[0.1, 1, 1, 1, 1, 1, 1])
masks = figure_data_dict["masks"]
axes_schematics = [FIG.add_subplot(gs00[i]) for i in range(3)] # the schematics of the map
masks["E"].plot_with_shortest_path(axes_schematics[1], plot_ho=True, holes_list=None, path_linewidth=config.LW_EMPHASIS) # plot Home and Out for marking
masks["F"].plot_with_shortest_path(axes_schematics[2], plot_ho=True, holes_list=None, path_linewidth=config.LW_EMPHASIS) # plot Home and Out for marking
axes_schematics[1].text(0.5, -0.05, "LRL", fontsize=plot_utils.FONT_SIZE,
          color=plot_utils.mask_colors["E"], transform=axes_schematics[1].transAxes,
               fontweight="bold", va="top", ha="center")
axes_schematics[2].text(0.5, -0.05, "RRR", fontsize=plot_utils.FONT_SIZE,
          color=plot_utils.mask_colors["F"], transform=axes_schematics[2].transAxes,
               fontweight="bold", va="top", ha="center")
plot_utils.plot_illustrative_cbar(axes_schematics[0], label_loc="left", aspect=10)
axes_traverses = [FIG.add_subplot(gs00[i+3]) for i in range(4)]
for ax, (tr_idx, path_df) in zip(axes_traverses, utils.iter_example_bout_paths(
        figure_data_dict["Acortical E traverse example bout steps"],
        figure_data_dict["Acortical E traverse example bout meta"])):
    plot_utils.plot_bout_path(ax, path_df, masks["E"], plot_colorbar=False, plot_duration=True,
                              plot_symbol=True, linewidth=config.LW_DATA, noise=0.15, title="")
    ax.text(0.5, 1, f"{utils.to_traverse_number(tr_idx)}", ha="center", va="bottom", fontsize=plot_utils.FONT_SIZE, transform=ax.transAxes)
axes_traverses[0].text(-0.15, 1.01, "Traverse #", ha="left", va="bottom", fontsize=plot_utils.TICK_SIZE, transform=axes_traverses[0].transAxes)

# plot reward interval comparison
gs10 = gs0[1].subgridspec(1, 2)
ax_int = FIG.add_subplot(gs10[0])

int_comp = {"Acortical E": figure_data_dict["Acortical Mask E reward intervals"],
                         "Acortical F": figure_data_dict["Acortical Mask F reward intervals"],}
plot_utils.plot_array_comparison(ax_int, int_comp, stats_type="mean", xlabel="Reward #",
                                 colordict=ef_color_dict,
                                 ylabel="Interval (s)", plot_shade=True, connect_scatters=True, ylim=50*60)

ax_sortie = FIG.add_subplot(gs10[1])
sortie_counts = {"Acortical E": figure_data_dict["Acortical Mask E sortie counts"],
                 "Acortical F": figure_data_dict["Acortical Mask F sortie counts"],}
plot_utils.plot_array_comparison(ax_sortie, sortie_counts, stats_type="mean", xlabel="Reward #",
                              colordict=ef_color_dict,
                              ylabel="N(sorties)", plot_shade=True, connect_scatters=True, ylim=40)

# other metrics: sortie count, tcp
gs20 = gs0[2].subgridspec(1, 2)
ax_duration = FIG.add_subplot(gs20[0])
duration_fit_results = {"Acortical E": figure_data_dict["Acortical Mask E Gen duration fit results"],
                        "Acortical F": figure_data_dict["Acortical F Gen duration fit results"],
                        "Control E": figure_data_dict["Control Mask E Gen duration fit results"],}
plot_utils.plot_curve_fit_comparison(ax_duration, result_dict=duration_fit_results,
                                     xlim=40, upper_y=100, xlabel="Traverse #", ylabel="Duration (s)", colordict=ef_color_dict,
                                     plot_scatter=False)


ax_error = FIG.add_subplot(gs20[1])
error_fit_results = {"Acortical E": figure_data_dict["Acortical Mask E Gen turn error rate fit results"],
                     "Acortical F": figure_data_dict["Acortical F Gen turn error rate fit results"],
                     "Control E": figure_data_dict["Control Mask E Gen turn error rate fit results"],}
plot_utils.plot_curve_fit_comparison(ax_error, result_dict=error_fit_results,
                                     xlim=40, upper_y=0.7, xlabel="Traverse #", ylabel="Turn error rate", colordict=ef_color_dict,
                                     plot_scatter=False)

plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.39, 0.99), (0.01, 0.73), (0.51, 0.73),
                                   (0.01, 0.35), (0.51, 0.35)])
config.save_figure(FIG, "acortical_ef_supp.pdf", save_path)

