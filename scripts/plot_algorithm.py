import numpy as np
from manhattan_maze import plot_utils, utils, endotaxis
import matplotlib.pyplot as plt

import config
from manhattan_maze import mask_d
config.set_plot_style()  # apply manuscript matplotlib style (R6)

save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 4
DIRECTIONS = ["H-O", "O-H"]
DIR_TITLE = {"H-O": "Outbound", "O-H": "Homebound"}
ENDO_C, Q_C = "tab:green", "tab:orange"
XCAP = 10   # traverses shown (Q ramps / Endotaxis steps to 0 well within 10)
## data loading
figure_data_dict = utils.load_all_figure_data()

FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
# Row 1 = model comparison (Panels A-D); Row 2 = the endotaxis-learning schematic (Panels E-G).
outer = FIG.add_gridspec(2, 1, height_ratios=[0.8, 1], hspace=0.05)


def plot_comparison_row(gs):
    """Row 1 (A-D): Mask-D corridor error + bottleneck choice, Q-learning vs Endotaxis.

    Q-learning (orange) is the self-play simulation loaded from figure_data
    ("Mask D model comparison", produced by gen_maskd_model_comparison.py). Endotaxis (green) is the
    parameter-free analytic one-pass step, computed inline here (its map is learned in a single
    exploratory pass, so there is no biclique-rate dependence to simulate).
    x capped at XCAP traverses. A/B = corridor error (outbound/homebound); C/D = bottleneck choice."""
    comp = figure_data_dict["Mask D model comparison"]
    meta = comp["meta"]
    L, BN = meta["L"], meta["BN_SIZE"]
    sub = gs.subgridspec(1, 4, wspace=0.05)
    a0 = FIG.add_subplot(sub[0])                 # A corridor error, outbound
    a1 = FIG.add_subplot(sub[1], sharey=a0)      # B corridor error, homebound (shares y with A)
    a2 = FIG.add_subplot(sub[2])                 # C bottleneck choice, outbound
    a3 = FIG.add_subplot(sub[3], sharey=a2)      # D bottleneck choice, homebound (shares y with C)
    axes = [a0, a1, a2, a3]

    def line(ax, arr, color, **kw):
        """Plot the per-traverse mean +/- SE shade of a (n_seeds, L) array in ``color``."""
        return plot_utils.plot_array_data(ax, arr, stats_type="mean", plot_scatter=False,
                                          plot_shade=True, line_color=color, linewidth=config.LW_DATA, **kw)

    def flat(ax, vec, color, **kw):
        """Draw a deterministic analytic curve ``vec`` (SE=0) on the same x-axis as ``line``."""
        return line(ax, np.tile(vec, (2, 1)), color, **kw)

    handles = {}
    for j, d in enumerate(DIRECTIONS):
        c, e_half = comp[d], meta["e_half"][d]
        # Endotaxis analytic step: beta=0.5 chance anchor -> perfect from traverse 2 (map learned in
        # one pass); bottleneck 1/deg(gateway)=0.2 chance -> 1.0.
        endo_err = np.concatenate([[e_half], np.zeros(L - 1)])
        endo_bn = np.concatenate([[0.2], np.ones(BN - 1)])

        ax_e = axes[j]                                   # corridor error rate: A outbound, B homebound
        _, _, hq, _ = line(ax_e, c["q_err"], Q_C, ylim=0.5)
        _, _, he, _ = flat(ax_e, endo_err, ENDO_C, ylim=0.5)
        ax_e.set(xlabel="Traverse #", ylabel="Corridor error rate" if j == 0 else "")
        ax_e.set_xlim(0.5, XCAP + 0.5)
        plot_utils.add_panel_title(ax_e, DIR_TITLE[d], fontsize=plot_utils.TICK_SIZE)
        handles = {"Q-learning": hq, "Endotaxis": he}

        ax_b = axes[j + 2]                               # bottleneck choice: C outbound, D homebound
        line(ax_b, c["q_bn"][:, :BN], Q_C, ylim=1.05)
        flat(ax_b, endo_bn, ENDO_C, ylim=1.05)
        ax_b.axhline(0.2, color="k", linestyle="--", linewidth=config.LW_HAIRLINE, zorder=config.Z_REFERENCE)
        ax_b.set(xlabel="Traverse #", ylabel="Choice of bottleneck" if j == 0 else "")
        ax_b.set_ylim(0, 1.05)
        ax_b.set_xlim(0.5, BN + 0.5)
        plot_utils.add_panel_title(ax_b, DIR_TITLE[d], fontsize=plot_utils.TICK_SIZE)

    # Shared y within each metric pair -> hide the inner panels' y-tick labels (frees panel width).
    axes[1].tick_params(labelleft=False)
    axes[3].tick_params(labelleft=False)
    axes[0].legend([handles[k] for k in ("Q-learning", "Endotaxis")],
                   ["Q-learning", "Endotaxis"], loc="upper right",
                   fontsize=plot_utils.TICK_SIZE, handlelength=1.2, borderpad=0.3, labelspacing=0.3)
    return axes

def plot_endotaxis_learning_d(gs, H_circle_x=1, V_circle_x=2,
                              n_nodes=9, radius=0.2, y_scale=0.5):
    """
    Render the Mask-D endotaxis-learning schematic panel.

    Draws the example corridor random walk, three snapshots of the learned goal-signal
    graph (with newly added edges highlighted), and the corresponding goal-signal
    profiles over the labelled corridors, illustrating how the endotaxis signal builds
    the maze graph. Operates on the outskirt-removed Mask-D corridor space (see
    ``docs/data_contracts.md`` §12, "Wildtype D corridor ..." keys).

    Parameters
    ----------
    gs : matplotlib.gridspec.SubplotSpec
        Parent gridspec cell subdivided into the panel's sub-axes.
    H_circle_x : float, optional
        X-coordinate of the horizontal-corridor node column in schematic units.
    V_circle_x : float, optional
        X-coordinate of the vertical-corridor node column in schematic units.
    n_nodes : int, optional
        Number of corridor nodes drawn per column.
    radius : float, optional
        Node-circle radius in schematic units.
    y_scale : float, optional
        Vertical spacing scale between corridor nodes.

    Returns
    -------
    None
        Draws onto figure axes created from ``gs`` in place.
    """
    gs_row = gs.subgridspec(1, 6, width_ratios=[0.8,0.15,0.4, 0.4, 0.4, 0.8], wspace=0.15)
    axes_walk = [FIG.add_subplot(gs_row[i]) for i in range(2)]
    seq = figure_data_dict["Wildtype D example corridor seq"]
    signal = figure_data_dict["Wildtype D example learned signal Out"]
    endotaxis.draw_walk(axes_walk, seq, signal, end_time=len(seq), end_corr=max(seq))
    plot_utils.add_panel_title(axes_walk[0], "Mask D", color=plot_utils.mask_colors["D"])

    adj = figure_data_dict["Wildtype D example learned adjacency Out"]
    # find the adjacency matrices for plotting
    min_indices = np.where(seq == min(seq))[0] # Home
    max_indices = np.where(seq == max(seq))[0] # Out

    step_indices = [min_indices[-2], max_indices[0], min_indices[-1]]
    axes_adj = [FIG.add_subplot(gs_row[i + 2]) for i in range(len(step_indices))]
    colors = [plot_utils.bout_type_color_dict["H-O"], plot_utils.bout_type_color_dict["O-O"], plot_utils.bout_type_color_dict["O-H"]]
    # Mark each snapshot step on the walk heatmap with a dashed line matching that snapshot's
    # color (H-O / O-O / O-H), so each learned-map panel is visually tied to its point in the
    # random walk. (Outsourced here from draw_walk so the line locations follow step_indices.)
    for step_idx, color in zip(step_indices, colors):
        axes_walk[0].axhline(step_idx, color=color, linewidth=config.LW_HAIRLINE, linestyle="--",
                             zorder=config.Z_MARKER)
    # draw_walk sets ylim(top=0, bottom=len-1) (inverted imshow); extend the bottom by half a
    # step so the final time point is not clipped on the axis line. Set both walk axes so the
    # heatmap and the bouts strip stay vertically aligned.
    for ax in axes_walk:
        ax.set_ylim(top=0, bottom=len(seq) - 1 + 0.5)
    first_mat = np.zeros_like(adj[0])
    # plot goal signals on the right
    gs_goal = gs_row[-1].subgridspec(len(step_indices), 1)
    axes_goal = [FIG.add_subplot(gs_goal[i]) for i in range(len(step_indices))]
    goal_color, control_colors = config.bottleneck_transition_colors()
    # Shared biclique/bottleneck corridor set (raw indices) -> display positions, the same
    # mapping plot_d_full uses. plot_algorithm reuses the outbound spec: the biclique fan
    # runs from the start corridor to its control arms + the bottleneck goal.
    corridor_order = list(mask_d.MaskDSpec().plot_corridor_order)
    spec = next(s for s in config.bottleneck_transition_specs() if s["outbound"])
    control_pos = [corridor_order.index(c) for c in spec["controls"]]
    start_pos = corridor_order.index(spec["start"])
    goal_pos = corridor_order.index(spec["goal"])
    signal_positions = control_pos + [start_pos, goal_pos]  # display positions [0, 2, 4, 6, 7, 8]
    for k, step_idx in enumerate(step_indices):
        sub_signal = signal[step_idx]
        plot_utils.plot_circle_with_signal_values(axes_adj[k], sub_signal, radius=radius, H_circle_x=H_circle_x,
                                                  n_nodes=n_nodes, V_circle_x=V_circle_x, y_scale=y_scale)
        added_adj = adj[step_idx] - first_mat
        plot_utils.plot_edges_based_on_adj_mat(axes_adj[k], first_mat, edge_color="tab:gray", linewidth=config.LW_DATA,
                                               H_circle_x=H_circle_x,
                                               n_nodes=n_nodes, V_circle_x=V_circle_x, y_scale=y_scale
                                               )
        plot_utils.plot_edges_based_on_adj_mat(axes_adj[k], added_adj, edge_color="red", linewidth=config.LW_EMPHASIS,
                                               H_circle_x=H_circle_x,
                                               n_nodes=n_nodes, V_circle_x=V_circle_x, y_scale=y_scale
                                               )
        first_mat = adj[step_idx]
        plot_utils.add_panel_title(axes_adj[k], f"t={step_idx}", color=colors[k])
        # hide axes
        axes_adj[k].axis("off")
        axes_adj[k].set_aspect("equal", adjustable="box")
        axes_adj[k].set_ylim([0, n_nodes * y_scale])  # hide unrelated corridors

        # plot signal
        goal_signal = sub_signal[signal_positions]
        goal_cmap = plt.cm.PiYG.reversed()  # match config PiYG palette: high signal -> goal magenta
        xs = np.arange(len(goal_signal)) + 1
        signal_colors = plot_utils.get_normalized_color_seq(goal_signal, goal_cmap)
        axes_goal[k].scatter(xs, goal_signal, color=signal_colors, zorder=config.Z_MARKER, s=config.MS_AREA_LARGE)
        axes_goal[k].text(0.5, 0.99, f"t={step_idx}", ha="center", va="top",
                            transform=axes_goal[k].transAxes,
                            color=colors[k], fontsize=plot_utils.FONT_SIZE)
        axes_goal[k].set_xlabel("")
        axes_goal[k].set_xticks([])
        axes_goal[k].set_ylabel("")
        if k<2:
            axes_goal[k].xaxis.set_visible(False)
            axes_goal[k].plot(xs, goal_signal, color="black", linewidth=config.LW_DATA)

        if k>1: # draw arrows for the connections of the nodes.

            arrow_coordinates = [(j+1, goal_signal[j]) for j in range(len(goal_signal))]
            arrow_origin = arrow_coordinates[len(control_pos)]
            arrows = [dict(x=arrow_origin[0], y=arrow_origin[1], dx=arrow_coordinates[j][0]-arrow_origin[0],
                           dy=arrow_coordinates[j][1]-arrow_origin[1], color=control_colors[j], w=0.1, zorder=5) for j in range(len(control_colors))]
            arrows.append(dict(x=arrow_origin[0], y=arrow_origin[1], dx=arrow_coordinates[-1][0]-arrow_origin[0],
                           dy=arrow_coordinates[-1][1]-arrow_origin[1], color=goal_color, w=0.1, zorder=5))
            for arrow in arrows:
                plot_utils.draw_arrow(axes_goal[k], **arrow)

    # Format Mask D specific plot elements
    endotaxis.format_d_corridor_order_ticks(axes_walk[0])
    # Add arrows for biclique structure: fan from the start corridor to its control arms and
    # the bottleneck goal, in the shared display-position space (see signal_positions above).
    # Same treatment as plot_d_full: explicit transitions with a thin shaft and a wide head.
    transitions = [(start_pos, t) for t in control_pos + [goal_pos]]
    plot_utils.add_biclique_arrows(axes_adj[-1], y_scale=y_scale, transitions=transitions,
                                   transition_colors=list(control_colors) + [goal_color],
                                   arrow_width=0.04, head_width=0.22, head_length=0.18)
    # one tick per labeled corridor, colored to match it (controls, then start + goal)
    label_colors = list(control_colors) + [goal_color, goal_color]
    axes_goal[-1].set_xticks(range(1, len(signal_positions) + 1))
    axes_goal[-1].set_xlabel("Labeled corridors")
    for tick_label, color in zip(axes_goal[-1].get_xticklabels(), label_colors):
        tick_label.set_color(color)
    axes_adj[-1].text(H_circle_x - 0.5, 4.2, 1, color=control_colors[0], fontsize=plot_utils.TICK_SIZE)
    axes_adj[-1].text(H_circle_x-0.5, 3.7, 2, color=control_colors[1], fontsize=plot_utils.TICK_SIZE)
    axes_adj[-1].text(H_circle_x - 0.5, 3.2, 3, color=control_colors[2], fontsize=plot_utils.TICK_SIZE)
    axes_adj[-1].text(H_circle_x - 0.5, 2.7, 4, color=control_colors[3], fontsize=plot_utils.TICK_SIZE)
    axes_adj[-1].text(V_circle_x + 0.4, 2.7, 5, color=goal_color, fontsize=plot_utils.TICK_SIZE)
    axes_adj[-1].text(H_circle_x - 0.5, 2.2, 6, color=goal_color, fontsize=plot_utils.TICK_SIZE)
    return axes_walk[0], axes_adj[0], axes_goal[0]


# Row 1 = model comparison (A-D); Row 2 = endotaxis-learning schematic (E-G).
ax_cmp = plot_comparison_row(outer[0])
ax_walk, ax_adj, ax_goal = plot_endotaxis_learning_d(outer[1])

# Shared goal-signal ylabel for the row-2 schematic (lower half of the figure).
FIG.text(0.78, 0.27, "Goal signal (log)", va="center", rotation=90, fontsize=plot_utils.FONT_SIZE)
plot_utils.add_letter_labels(FIG, [(0.01, 0.99),  (0.27, 0.99), (0.50, 0.99), (0.77, 0.99),
                                   (0.01, 0.55), (0.37, 0.55), (0.77, 0.55) ])

config.save_figure(FIG, "algo.pdf", save_path)


