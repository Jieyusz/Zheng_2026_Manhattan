from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
import numpy as np
import config
from manhattan_maze import mask_d
config.set_plot_style()  # apply manuscript matplotlib style (R6)

save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 5.5

## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()


n_all_d_animals = 7
all_colors = [plt.cm.tab10(i) for i in range(n_all_d_animals)]
FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(4, 1, height_ratios=[1.2, 1, 1,1], hspace=0.05)

# move similarity measures here
# Plot similarity matrix
gs20 = gs0[0].subgridspec(1, 5, hspace=0.01, width_ratios=[1, 1, 1, 0.1, 1.2])
d_similarity_list = figure_data_dict["Wildtype D similarity matrices"]
axes_mat = [FIG.add_subplot(gs20[j]) for j in range(4)]
# axes_off_diag = [FIG.add_subplot(gs20b[j]) for j in range(3)]
# The example animal is chosen by content (the richest triplet), not by a positional index:
# the saved list is reordered by every regeneration. See utils.select_similarity_example.
j_oo, j_hh, j_oh_prime = utils.select_similarity_example(d_similarity_list)
plot_utils.plot_maskd_similarity_matrix(axes_mat, j_oo=j_oo, j_hh=j_hh, j_oh_prime=j_oh_prime,
                                        labels=list(config.SIMILARITY_LATEX.values()),
                                        axis_labels=config.TRAVERSE_LATEX,
                                        cmap=plt.cm.plasma, plot_colorbar=True)

# Plot average traverse similarity in the last axis
ax_av = FIG.add_subplot(gs20[-1])
avg_sims = figure_data_dict["Wildtype D average traverse similarity"]
# turn into array and plot scatter boxes
avg_sims = np.array(avg_sims)
avg_sims_dict = {label: avg_sims[:, i] for i, label in enumerate(config.SIMILARITY_LATEX)}
avg_sims_color_dict = {label: "black" for label in config.SIMILARITY_LATEX}
# connect the individual dots for animals
for i in range(avg_sims.shape[0]):
    ax_av.plot(range(avg_sims.shape[1]), avg_sims[i, :], color=plt.cm.tab10(i), alpha=0.8, linewidth=config.LW_HAIRLINE)
    ax_av.scatter(range(avg_sims.shape[1]), avg_sims[i, :], color=plt.cm.tab10(i), alpha=0.8, s=config.MS_AREA_SMALL)
fried_man_results = utils.friedman_with_pairwise_wilcoxon(avg_sims_dict)
plot_utils.plot_group_scatter_box_comparison(ax_av, avg_sims_dict, fried_man_results, colordict=avg_sims_color_dict,
                                             ylabel="Mean similarity", plot_ns=True, plot_scatter=False)
ax_av.set_xticklabels(list(config.SIMILARITY_LATEX.values()))
ax_av.tick_params(axis="x", rotation=45)

gs10 = gs0[1].subgridspec(1, 3)
d_similarity_list = figure_data_dict["Wildtype D similarity matrices"]
axes_off_diagonal = [FIG.add_subplot(gs10[j]) for j in range(3)]
for k, mats in enumerate(d_similarity_list):
    color = plt.cm.tab10(k)
    for j, mat in enumerate(mats):
        if mat is not None:
            axes_off_diagonal[j].plot(utils.get_mat_mean_diagonal(mat), color=color, linewidth=config.LW_DATA,)

axes_off_diagonal[1].yaxis.set_visible(False)
axes_off_diagonal[2].yaxis.set_visible(False)
for ax, mat_type in zip(axes_off_diagonal, config.SIMILARITY_LATEX.values()):
    plot_utils.add_panel_title(ax, mat_type, fontsize=plot_utils.TICK_SIZE)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.4)
    ax.set_xlabel("k-th off diagonal", )
    ax.set_ylabel("Mean similarity", )

gs00 = gs0[2].subgridspec(1, 7, width_ratios=[0.1, 3, 3, 3, 3, 3, 3], wspace=0.1)
axes_f = [FIG.add_subplot(gs00[i]) for i in range(7)]
# The example session's bouts are exported as flat tables (R8), so the motif window is a
# row selection by traverse_idx. start_time_s is already relative to the parent session's
# first frame, which is what the start-time annotation needs.
example_meta = figure_data_dict["Mask D example bout meta"]
example_meta = example_meta[example_meta.example == config.MASK_D_EXAMPLE_ID]
motif_meta = example_meta[example_meta.traverse_idx.isin(config.MASK_D_MOTIF_TRAVERSES)]
for ax, (tr_idx, path_df) in zip(axes_f[1:], utils.iter_example_bout_paths(
        figure_data_dict["Mask D example bout steps"], motif_meta, label_column="traverse_idx")):
    plot_utils.plot_bout_path(ax, path_df, figure_data_dict["masks"]["D"], plot_colorbar=False,
                              plot_symbol=True, linewidth=config.LW_EMPHASIS,
                              marker_color=all_colors[-3:][config.MASK_D_EXAMPLE_ID],
                              plot_start_time=True, plot_duration=False, title="")
    plot_utils.add_panel_title(ax, f"{utils.to_traverse_number(tr_idx)}")
plot_utils.plot_illustrative_cbar(axes_f[0], aspect=20)

# row 4 plot the different node set
# row 4: supplementary for the bottleneck preference in outbound and homebound.
# Spec-driven (raw Mask-D corridor indices); edit transition_specs to plot any combination.
goal_color, control_colors = config.bottleneck_transition_colors()
corridor_order = list(mask_d.MaskDSpec().plot_corridor_order)  # raw corridor -> display position
matrices = figure_data_dict["Wildtype D corridor transition matrices"]
adjacency = figure_data_dict.get("Wildtype D corridor adjacency")  # backdrop graph; None -> nodes+arrows only
transition_specs = [
    {"label": "Outbound", "start": 12, "goal": 1, "controls": [2, 4, 6, 8], "outbound": True},
    {"label": "Homebound",  "start": 19, "goal": 1, "controls": [5, 3, 7, 9], "outbound": False},
]

gs30 = gs0[-1].subgridspec(1, 2 * len(transition_specs),
                           width_ratios=[0.8, 2] * len(transition_specs), wspace=0)
for k, spec in enumerate(transition_specs):
    start_pos = corridor_order.index(spec["start"])
    goal_pos = corridor_order.index(spec["goal"])
    control_pos = [corridor_order.index(c) for c in spec["controls"]]
    ax_schem = FIG.add_subplot(gs30[2 * k])
    plot_utils.plot_corridor_transition_schematic(ax_schem, start_pos, goal_pos, control_pos,
                                                  goal_color, control_colors, adj=adjacency,
                                                  column_colors=("tab:blue", "tab:orange"), outbound=spec["outbound"],
                                                  plot_direction_arrows=True, grey_uninvolved_nodes=True,
                                                  red_outline_orange=True)
    ax_schem.text(0.5, 0, spec["label"], ha="center", va="top", fontsize=plot_utils.FONT_SIZE, transform=ax_schem.transAxes)
    ax_trans = FIG.add_subplot(gs30[2 * k + 1])
    node_set = (start_pos, goal_pos, control_pos)
    # conditional choice ratio among the 4 shown arms (goal + 3 controls),
    # renormalized to sum to 1 per slice so the null is 1/4 (start node has 5 edges).
    transition_to_node_dict = utils.renormalize_choice_among_arms(
        utils.select_d_transition_dict(matrices, start_pos, [goal_pos] + control_pos))

    plot_utils.plot_aggregated_choice_ratios(ax_trans, transition_to_node_dict, node_set,
                                             goal_color, control_colors, linewidth=config.LW_DATA,
                                             random_value=1 / 5, xlabel=None, ylabel=None,
                                             start_idx=0 if spec["outbound"] else 1)
    ax_trans.set_xlabel(f"{spec['label']} rewards", fontsize=plot_utils.TICK_SIZE)



plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.74, 0.99), (0.01, 0.72), (0.01, 0.45),
                                   (0.01, 0.25), (0.50, 0.25)],)
config.save_figure(FIG, "d_motif.pdf", save_path)
