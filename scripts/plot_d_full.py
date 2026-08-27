from manhattan_maze import plot_utils, utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import config
from manhattan_maze import mask_d
config.set_plot_style()  # apply manuscript matplotlib style (R6)


save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 5.5
n_roundtrips=15 # minimum number of roundtrips to be included in figure 2, default 15

## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()

FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(4, 1, height_ratios=[1, 1.2, 1.2, 1.2], hspace=0)
gs00 = gs0[0].subgridspec(1, 7, width_ratios=[0.1, 1.2, 2, 2, 2, 2, 2], wspace=0)
ax_a1 = FIG.add_subplot(gs00[0]) # p[lot the colorbar
plot_utils.plot_illustrative_cbar(ax_a1)
ax_a2 = FIG.add_subplot(gs00[1])
plot_utils.plot_schematic_d_graph(ax_a2, path_linewidth=config.LW_DATA)
# Panel A right: top view of Mask D
ax_a3 = FIG.add_subplot(gs00[2])
figure_data_dict["masks"]["D"].plot_with_shortest_path(ax_a3, maskd_bottleneck=True, plot_ho=True, path_linewidth=config.LW_DATA)


# Panel B: plot the traverse examples. The example session's bouts are exported as flat
# tables (R8), so the traverse strip is a row selection by traverse_idx.
d_example_meta = figure_data_dict["Mask D example bout meta"]
d_example_meta = d_example_meta[d_example_meta.example == config.MASK_D_EXAMPLE_ID]
example_traverses = d_example_meta[d_example_meta.traverse_idx.isin(config.EXAMPLE_TRAVERSE_INDICES)]
axes_b = [FIG.add_subplot(gs00[i+3]) for i in range(4)]
for ax, (tr_idx, path_df) in zip(axes_b, utils.iter_example_bout_paths(
        figure_data_dict["Mask D example bout steps"], example_traverses,
        label_column="traverse_idx")):
    plot_utils.plot_bout_path(ax, path_df, figure_data_dict["masks"]["D"], plot_colorbar=False,
                              plot_duration=True, plot_symbol=True, linewidth=config.LW_EMPHASIS,
                              marker_color=plt.cm.tab10(5), title="")
    ax.text(0.5, 1, f"{utils.to_traverse_number(tr_idx)}", ha="center", va="bottom", fontsize=plot_utils.FONT_SIZE, transform=ax.transAxes)

# row 2: plot traverse metrics
gs10 = gs0[1].subgridspec(1, 2, hspace=0.05, width_ratios=[1, 1])
axes_c = [FIG.add_subplot(gs10[i]) for i in range(2)]
# plot traverse durations
duration_array = figure_data_dict["Wildtype D duration"]
# per-step error RATES (fraction of moves not progressing toward goal; [0,1], chance ~0.5),
# replacing the old unbounded corridor/tile COUNT curves.
corridor_error_array = figure_data_dict["Wildtype D corridor error rate"]
tile_error_array = figure_data_dict["Wildtype D tile error rate"]
out2, home2, _, se2 = plot_utils.plot_array_data(axes_c[0], duration_array, stats_type="mean", scatter_colors=["tab:grey"], line_color="tab:grey",
                                                 plot_shade=True, connect_scatters=True, xlabel="Traverse #",
                                                 ylabel="Duration (s)", ylim=250)
# add legends
axes_c[0].legend([out2, home2, se2], ["Outbound", "Homebound", "SE"],
                 handler_map={tuple: HandlerTuple(ndivide=None)}, loc="upper right", bbox_to_anchor=(1, 1), fontsize=plot_utils.TICK_SIZE)

# Right: corridor and tile error RATES on a SINGLE axis (same format as plot_first_mask panel H).
# Both are [0,1] per-step rates on one 0..0.5 "Error rate" scale (0.5 = memoryless-walk chance at
# the axis top); corridor = grey, tile = red, told apart by colour + legend rather than a twin axis.
tile_color = "tab:red"
corridor_color = "tab:grey"
c_out, c_home, _, _ = plot_utils.plot_array_data(axes_c[1], corridor_error_array, stats_type="mean", markersize=config.MS_AREA_DEFAULT,
                                                 scatter_colors=[corridor_color], line_color=corridor_color,
                                                 plot_shade=True, connect_scatters=True, shade_alpha=1,
                                                 ylabel="Error rate", ylim=0.5, xlabel="Traverse #")
tile_out, tile_home, _, _ = plot_utils.plot_array_data(axes_c[1], tile_error_array, stats_type="mean", markersize=config.MS_AREA_DEFAULT,
                                                       scatter_colors=[tile_color], line_color=tile_color,bar_displacement=0.1,
                                                       plot_shade=True, connect_scatters=True, shade_alpha=1,
                                                       xlabel="Traverse #", ylabel="Graph error rate")
axes_c[1].legend(handles=[(c_out, c_home), (tile_out, tile_home)], labels=["Corridor", "Tile"],
                 handler_map={tuple: HandlerTuple(ndivide=None)}, loc="upper right", fontsize=plot_utils.TICK_SIZE)

# Plot the bottleneck transition. Each spec is a corridor transition to contrast
# (start corridor -> goal vs control corridors), in raw Mask-D corridor indices;
# edit this list to plot any combination. Choice ratios index the reduced 18x18
# transition matrices via the display-corridor order.
goal_color, control_colors = config.bottleneck_transition_colors()
corridor_order = list(mask_d.MaskDSpec().plot_corridor_order)  # raw corridor -> display position
matrices = figure_data_dict["Wildtype D corridor transition matrices"]
adjacency = figure_data_dict.get("Wildtype D corridor adjacency")  # backdrop graph; None -> nodes+arrows only
transition_specs = config.bottleneck_transition_specs()

gs30 = gs0[2].subgridspec(1, 2 * len(transition_specs),
                          width_ratios=[0.8, 2] * len(transition_specs), wspace=0)

for k, spec in enumerate(transition_specs):
    # map the spec's raw corridor indices to display positions (shared by schematic + data)
    start_pos = corridor_order.index(spec["start"])
    goal_pos = corridor_order.index(spec["goal"])
    control_pos = [corridor_order.index(c) for c in spec["controls"]]
    # schematic of the contrast (arrows from start to goal/controls)
    ax_schem = FIG.add_subplot(gs30[2 * k])
    plot_utils.plot_corridor_transition_schematic(ax_schem, start_pos, goal_pos, control_pos,
                                                  goal_color, control_colors, adj=adjacency,
                                                  column_colors=("tab:blue", "tab:orange"), outbound=spec["outbound"],
                                                  plot_direction_arrows=True, grey_uninvolved_nodes=True,
                                                  red_outline_orange=True)
    ax_schem.text(0.5, 0, spec["label"], ha="center", va="top", fontsize=plot_utils.TICK_SIZE, transform=ax_schem.transAxes)
    # conditional choice ratio among the 4 shown arms (goal + 3 controls),
    # renormalized to sum to 1 per slice so the null is 1/4 (start node has 5 edges).
    ax_trans = FIG.add_subplot(gs30[2 * k + 1])
    node_set = (start_pos, goal_pos, control_pos)
    transition_to_node_dict = utils.renormalize_choice_among_arms(
        utils.select_d_transition_dict(matrices, start_pos, [goal_pos] + control_pos))
    plot_utils.plot_aggregated_choice_ratios(ax_trans, transition_to_node_dict, node_set,
                                             goal_color, control_colors, linewidth=config.LW_DATA,
                                             random_value=1 / 5, xlabel=None, ylabel=None,
                                             start_idx=0 if spec["outbound"] else 1)
    ax_trans.set_xlabel(f"{spec['label']} rewards", fontsize=plot_utils.TICK_SIZE)

# Off-path biclique transitions: each off-path corridor can choose any of the 4 corridors
# in its opposite group; the 3 off-path arms are shown (the 4th, shortest-path, arm is the
# preferred "goal" and is excluded upstream). To show there is little preference *among* the
# off-path arms, the right panel plots one smoothed population-mean line per transition (18
# per group), with each start node's arms renormalized among its 3 off-path arms so the
# dashed reference is uniform = 1/3: the lines cluster near 1/3 rather than separating.
# Group 1 = Horizontal->Vertical (blue shades), Group 2 = Vertical->Horizontal (orange).
gs40 = gs0[3].subgridspec(1, 4, width_ratios=[0.8, 2] * 2, wspace=0)
spec = mask_d.MaskDSpec()
# Merge both bicliques (disjoint raw-corridor keys) into one {(start, end): (n_sessions, 50)} dict.
offpath_ratios = {**figure_data_dict["Wildtype D biclique 1 offpath choice ratios"],
                  **figure_data_dict["Wildtype D biclique 2 offpath choice ratios"]}
horizontal_corridors = set(spec.biclique_1_groups[0]) | set(spec.biclique_2_groups[1])
transition_groups = [
    ("Hor. to Ver.", [k for k in offpath_ratios if k[0] in horizontal_corridors],     plt.cm.Blues),
    ("Ver. to Hor.", [k for k in offpath_ratios if k[0] not in horizontal_corridors], plt.cm.Oranges),
]
for g, (title, keys, cmap) in enumerate(transition_groups):
    shades = cmap(np.linspace(0.4, 1.0, len(keys)))
    key_colors = {k: shades[i] for i, k in enumerate(keys)}
    # schematic: D-graph backdrop (shared with panel A) + thin color-matched transition arrows.
    # Grey out corridors not part of the off-path bicliques (only the off-path start/end
    # nodes appearing in the transitions stay blue/orange).
    transitions = [(corridor_order.index(s), corridor_order.index(e)) for (s, e) in keys]
    involved_positions = {p for t in transitions for p in t}
    ax_schem = FIG.add_subplot(gs40[2 * g])
    plot_utils.plot_schematic_d_graph(ax_schem, path_linewidth=config.LW_DATA, plot_shortest_path=False, highlight_keynodes=False,
                                      highlight_bottleneck=False,
                                      plot_direction_arrows=False, involved_positions=involved_positions)
    plot_utils.add_biclique_arrows(ax_schem, y_scale=0.5, transitions=transitions,
                                   transition_colors=[key_colors[k] for k in keys],
                                   arrow_width=0.04, head_width=0.22, head_length=0.18)
    ax_schem.text(0.5, 0, title, ha="center", va="top", fontsize=plot_utils.TICK_SIZE, transform=ax_schem.transAxes)
    # Renormalize each start node's arms among its 3 shown off-path arms (drop the
    # shortest-path arm from the denominator), so the per-transition lines measure
    # "given an off-path arm was taken, which of the 3?" with uniform reference 1/3.
    renorm_ratios = utils.renormalize_choice_among_arms(
        {k: offpath_ratios[k] for k in keys}, group_by=lambda key: key[0])
    # one smoothed population-mean line per transition, clustering near uniform = 1/3
    ax_data = FIG.add_subplot(gs40[2 * g + 1])
    plot_utils.plot_offpath_choice_ratios(ax_data, renorm_ratios, key_colors,
                                          smooth_func=utils.moving_average, smooth_window=10,
                                          start_idx=None, chance_level=1 / 3, xlabel="Rewards",
                                          ylabel="Choice ratio", ylim=0.7)


# add labels
plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.34, 0.99), (0.01, 0.79),
                                   (0.50, 0.79), (0.01, 0.54), (0.51, 0.54),
                                   (0.01, 0.27), (0.51, 0.27)])

config.save_figure(FIG, "maskd.pdf", save_path)
