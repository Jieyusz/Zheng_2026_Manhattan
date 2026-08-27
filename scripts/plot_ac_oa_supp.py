from manhattan_maze import plot_utils, utils
import numpy as np
import matplotlib.pyplot as plt

# add arg parser for saving figure path
import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)
save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 5

## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()

FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(3, 1, height_ratios=[1, 1, 1])

# plot mask o result in the first row.
gs00 = gs0[0].subgridspec(1, 3, width_ratios=[1.1, 2, 2], wspace=0.05)

# tile per corridor
ax_tile = FIG.add_subplot(gs00[0])
tiles_per_corridor = utils.select_by_prefix(figure_data_dict, config.GENOTYPES, "Mask O tiles per corridor")
kruskal_results = utils.kruskal_with_pairwise_mann_whitney(tiles_per_corridor, alternative="greater")
plot_utils.plot_group_scatter_box_comparison(ax_tile, tiles_per_corridor, kruskal_results,
                                             ylabel="tiles/corridor")
ax_tile.text(0.5, 1, "Mask O", fontsize=plot_utils.FONT_SIZE, ha="center", va="bottom", transform=ax_tile.transAxes,
             color=plot_utils.mask_colors["O"])

# reward interval
ax_o = FIG.add_subplot(gs00[1])
o_intervals = utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:2], "Mask O reward intervals")
plot_utils.plot_array_comparison(ax_o, o_intervals, stats_type="mean", xlabel="Reward #",
                                 ylabel="Interval (s)", plot_shade=True, connect_scatters=True, linewidth=config.LW_DATA, ylim=3100)

# sorties
ax_sortie = FIG.add_subplot(gs00[2])
sortie_counts = utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:2], "Mask O sortie counts")
plot_utils.plot_array_comparison(ax_sortie, sortie_counts, stats_type="mean", xlabel="Reward #", linewidth=config.LW_DATA,
                                 ylabel="N(sorties)", plot_shade=True, connect_scatters=True, ylim=50)

# Mask A results. Panel order follows the Results narrative: the repetition/inefficiency measures
# (speed, sorties, tile and corridor errors) come first, then the learning curves.
# row 2: D speed, E sortie counts, F tile error rate
gs10 = gs0[1].subgridspec(1, 3, wspace=0.05)
ax_speed = FIG.add_subplot(gs10[0])
at_speed = utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:2], "Mask A speed")
plot_utils.plot_array_comparison(ax_speed, at_speed, plot_shade=True, connect_scatters=True, stats_type="mean", xlabel="Traverse #", linewidth=config.LW_DATA,
                                 ylabel="Speed (tiles/s)", ylim=3.5)
speed_results = utils.time_point_kruskal_mann_whitney_u_test(at_speed, alternative="greater")
ax_speed.text(0.5, 1, "Mask A", fontsize=plot_utils.FONT_SIZE, ha="center", va="bottom", transform=ax_speed.transAxes,
             color=plot_utils.mask_colors["A"])

ax_sorties = FIG.add_subplot(gs10[1])
at_sorties = utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:2], "Mask A sortie counts")
plot_utils.plot_array_comparison(ax_sorties, at_sorties, ylim=60, plot_shade=True, connect_scatters=True, stats_type="mean", ylabel="N(sorties)", linewidth=config.LW_DATA)
ax_sorties.get_legend().remove()
# tile and corridor: per-step error RATES ([0,1], chance ~0.5 at the axis top) replacing the unbounded counts.
ax_tile_error = FIG.add_subplot(gs10[2])
at_tile_error = utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:2], "Mask A tile error rate")
plot_utils.plot_array_comparison(ax_tile_error, at_tile_error, plot_shade=True, connect_scatters=True, shade_alpha=1, linewidth=config.LW_DATA,
                                 stats_type="mean", xlabel="Traverse #", ylabel="Tile error rate", ylim=0.5)
ax_tile_error.get_legend().remove()

# row 3: G traverse duration, H turn error rate, I first-journey forward bias
gs20 = gs0[2].subgridspec(1, 3, wspace=0.05)
# duration
ax_duration = FIG.add_subplot(gs20[0])
at_durations = {**utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:2], "Mask A duration"),
                "Wildtype": figure_data_dict["Wildtype A traverse duration"][:, :20]}
plot_utils.plot_array_comparison(ax_duration, at_durations, stats_type="mean", xlabel="Traverse #", linewidth=config.LW_DATA,
                                 ylabel="Duration (s)", plot_shade=True, connect_scatters=True, ylim=500)
# turn error rate
ax_error = FIG.add_subplot(gs20[1])
at_errors = {**utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:2], "Mask A turn error rate"),
             "Wildtype": figure_data_dict["Wildtype A turn error rate"][:, :20]}
plot_utils.plot_array_comparison(ax_error, at_errors, stats_type="mean", markersize=0, linewidth=config.LW_DATA,
                                 plot_shade=True, connect_scatters=True, shade_alpha=1,
                                 xlabel="Traverse #", ylabel="Turn error rate", ylim=0.7)
ax_error.get_legend().remove()

# Panel I: latent learning before any reward. Directional persistence (beta_hat) along the
# pre-reward first journey, per genotype, against the memoryless walker's 0.5. Mirrors
# fig:oa_supp's last panel, but genotype-coded rather than mask-coded. Hand-drawn because
# plot_array_comparison expects (n_animals, n_traverses) with alternating outbound/homebound
# columns, which is meaningless for fractional journey positions. One 20%-of-journey window per
# point, truncated at the journey's ends, no post-hoc smoothing -- so all three curves span 0->1
# and the SE bands widen at both extremes, where the truncated windows hold fewer decisions.
ax_beta = FIG.add_subplot(gs20[2])
ax_beta.axhline(0.5, ls="--", color="black", lw=config.LW_HAIRLINE, zorder=config.Z_REFERENCE, label="random")
for genotype in config.GENOTYPES:
    bx, bmean, bse = figure_data_dict[f"{genotype} A first journey forward bias"]
    ok = ~np.isnan(bmean)
    color = plot_utils.genotype_colors[genotype]
    ax_beta.plot(bx[ok], bmean[ok], color=color, lw=config.LW_DATA)
    ax_beta.fill_between(bx[ok], (bmean - bse)[ok], (bmean + bse)[ok], color=color, alpha=0.25, lw=0)
ax_beta.set_xlim(0.1, 0.9)
ax_beta.set_ylim(0, 1)
ax_beta.set_xlabel("fraction of the 1st journey")
ax_beta.set_ylabel(r"Forward bias $\hat{\beta}$")
ax_beta.legend(loc="upper left", fontsize=plot_utils.TICK_SIZE)
# Panel order is deliberate and grouped by mask, not by mention order: row 1 is Mask O
# (A-C), rows 2-3 are Mask A (D-I). Against the manuscript the only residual inversion is
# I cited mid-paragraph before F/G/H; I is the full-width bottom-right slot, so a reorder
# would cost a layout rewrite to fix one panel. Leave as is and keep the captions in sync.
plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.25, 0.99), (0.63, 0.99),
                                   (0.01, 0.65), (0.33, 0.65), (0.67, 0.65),
                                   (0.01, 0.33), (0.33, 0.33), (0.67, 0.33),])
config.save_figure(FIG, "ac_oa_supp.pdf", save_path)