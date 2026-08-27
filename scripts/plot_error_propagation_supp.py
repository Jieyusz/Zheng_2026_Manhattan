"""
Supplementary figure: error propagation in Mask A — animals vs. two learning models.

Mask A is the P10 linear path graph (10 corridors, 9 decision holes). This figure shows how
corridor / turn errors decline across path positions over learning, and contrasts the animals
with what a model-free (Q-learning) learner and a map-based (endotaxis) learner predict.

Layout: a top row (Panel A) illustrating the endotaxis SIMULATION on Mask A, over a 2 rows x 4
data-column grid (Panels B–G) of error rates, one colour-coded line per position sharing a single
viridis distance-to-reward colour code (one colorbar, far right).

  A  Endotaxis simulation of Mask A on the example animal's (config.MASK_A_EXAMPLE_ID; the Fig-1E
     animal) first two journeys: the corridor random walk (Steps x Corridor, Home->Out ordered
     1..10; white = trajectory, plasma = goal signal), three learned-map snapshots (red = newly
     wired edges) at a late-Home / first-Out / last-Out step, and the learned goal signal over
     corridors. Rendered in the ``plot_algorithm.py`` style from the vendored endotaxis learner
     (data: gen_endotaxis.py "Wildtype A example ..." keys).
  B  Corridor error rate by distance-to-reward.
  C  Per-hole turn error rate (first-decision, approach-conditioned; chance 0.5), holes ordered
     close->far from reward.
  (Columns of B/C: animal outbound, animal homebound, model-free RL, endotaxis.)

Model predictions (right two columns of B and C) are PARAMETER-FREE analytic functions of
``(n_positions, n_traverses)``, computed INLINE here (not loaded from figure_data): they need no
raw data, and round-tripping them through the cache would only add stale-cache surface.

  Model-free RL (``rl_model.analytic_rl_staircase``). Value exists only where reward has been
  experienced and TD propagates it back exactly one position per rewarded traverse, so the
  prediction is the closed-form backward staircase: chance on traverse 1 (no reward yet), then
  the low-error frontier steps outward one position per traverse. Direction-independent (each
  direction is learned in isolation), and that independence is itself a prediction — a model-free
  learner cannot transfer between outbound and homebound.

  Endotaxis (``endotaxis.analytic_endotaxis_step``). The map-based learner gets the SAME
  no-signal treatment: chance on traverse 1 (no goal signal yet, so a random walk — the same
  anchor as the RL column). But it learns the whole graph MAP from one exploratory traverse (Panel
  A), so a single reward then makes the gradient correct at EVERY position at once: 0 from traverse
  2, all positions together (a synchronized step, not a back-to-front staircase).

Both analytic predictions are validated by simulation: the trained per-animal RL agents in
``manhattan_maze/rl_model.py`` converge to the staircase (see gen_rl_simulation.py /
gen_rl_turn_simulation.py and tests/test_rl_error_propagation.py); the endotaxis map simulation
(``endotaxis.random_walk_complete`` + ``Learn_Mouse_tr`` + ``endo_gradient_walk``, illustrated in
Panel A) reproduces the step (see tests/test_endotaxis_error_propagation.py).

The contrast the figure makes: the animals (i) improve across the whole path roughly in parallel,
not back-to-front (the endotaxis signature, not the RL staircase); and (ii) already show
low-error holes on the very first traverse (latent learning during unrewarded exploration). Both
from-scratch models share the animals' traverse-1 chance anchor — the fair comparison — so the
panels isolate HOW competence arrives once reward is available: sequentially (RL) or all at once
(endotaxis).

Data (data/figure_data/):
    "Wildtype A corridor error by position"     -> Panel B, animal columns   (gen_error_propagation.py)
    "Wildtype A hole error by position"         -> Panel C, animal columns   (gen_error_propagation.py)
    "Wildtype A example corridor seq"           -> Panel A walk              (gen_endotaxis.py)
    "Wildtype A example learned signal Out"     -> Panel A signal/map colour (gen_endotaxis.py)
    "Wildtype A example learned adjacency Out"  -> Panel A map snapshots     (gen_endotaxis.py)
The model-prediction columns of B/C are computed inline (see above), not loaded.
"""
import numpy as np
import matplotlib.pyplot as plt

from manhattan_maze import plot_utils, utils, rl_model, endotaxis
from manhattan_maze.plot_curves import (plot_level_error_array, distance_scalar_mappable,
                                        add_distance_colorbar)
import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)

save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 4.8  # top endotaxis-sim row (A) + 2 error rows x 4 data cols + right colorbar
DIRECTIONS = [("H-O", "outbound"), ("O-H", "homebound")]
SMOOTH_WINDOW = 3  # moving-average window: small enough to keep the first-traverse
                   # (latent-learning) points, large enough to tame per-position noise
LINEWIDTH = config.LW_HAIRLINE  # per-position line width, shared across all error panels
Y_BOTTOM = -0.03   # small negative bottom margin so a learned curve at 0 clears the axis
Y_TOP = 0.6        # shared y-axis top across ALL error panels (corridor + turn, animal + model)
RL_MAX_TRAVERSE = 10  # model columns' x-axis cap (both resolve the path within ~9 traverses)


# Data loading. The two ANIMAL columns of the error grid load their empirical cohort curves from
# figure_data; the two MODEL columns are parameter-free analytic predictions computed inline
# (rl_model.analytic_rl_staircase, endotaxis.analytic_endotaxis_step). Panel A loads the
# endotaxis-simulation example keys (gen_endotaxis.py).
figure_data_dict = utils.load_all_figure_data()
corridor_data = figure_data_dict["Wildtype A corridor error by position"]
hole_data = figure_data_dict["Wildtype A hole error by position"]
n_pos = corridor_data["n_pos"]
n_h = len(hole_data["H-O"])                       # decision holes (== n_pos - 1 = 9 for Mask A)
N = corridor_data["H-O"].shape[1]                 # traverse-axis width (shared by every column)

# Model predictions (direction-independent -> one column each serves both directions).
corridor_staircase = rl_model.analytic_rl_staircase(n_pos - 1, N, dead_end_last=True)
turn_staircase = rl_model.analytic_rl_staircase(n_h, N)
endo_corridor = endotaxis.analytic_endotaxis_step(n_pos - 1, N, dead_end_last=True)
endo_turn = endotaxis.analytic_endotaxis_step(n_h, N)

# Panel A endotaxis simulation (example animal, first two journeys; Home->Out distance-ordered).
endo_seq = np.asarray(figure_data_dict["Wildtype A example corridor seq"])
endo_signal = np.asarray(figure_data_dict["Wildtype A example learned signal Out"])
endo_adj = figure_data_dict["Wildtype A example learned adjacency Out"]

# colour code for the error grid: a single discrete viridis colormap over distance-to-reward
# (dark = close to reward), shared by ALL error columns -- animal and model alike.
sm = distance_scalar_mappable(n_pos)         # viridis colorbar (error grid)
cmap_pos = plt.get_cmap("viridis", n_h)      # per-position line colours, error grid

FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
# Reserve a left strip for the rotated row-group headers ("Corridor"/"Turn"). Constrained layout
# ignores manually-placed FIG.text, so without this margin the headers fill onto the plots. The top
# is left at 1.0 (not capped) so Panel A's content reaches the same height as other figures and the
# (0.01, 0.99) panel-A letter sits snug against it rather than floating in a top margin.
FIG.get_layout_engine().set(rect=(0.03, 0, 0.955, 1.0))
# Outer split into THREE rows: Panel A (endotaxis simulation), a dedicated header STRIP that holds
# the error grid's bold group headers clear of Panel A's x-axis, then the 2 x 4 error grid. The
# spacing WITHIN the error grid (corridor vs turn rows) is the inner subgridspec's hspace, kept
# tight; hspace here is 0 so the strip's own height is the separation above the grid.
outer = FIG.add_gridspec(3, 1, height_ratios=[1.6, 0.05, 3.0], hspace=0.0)


def plot_endotaxis_learning_a(gs, H_circle_x=1, V_circle_x=3, n_nodes=5, radius=0.25, y_scale=1):
    """Panel A: the Mask-A endotaxis simulation in the plot_algorithm.py style (walk heatmap +
    learned-map snapshots + goal signal). Adapted from the legacy generalised
    ``plot_endotaxis_learning`` (mask_name='A')."""
    gs_row = gs.subgridspec(1, 7, width_ratios=[0.8, 0.15, 0.4, 0.4, 0.4, 0.6, 0.05], wspace=0.05)
    axes_walk = [FIG.add_subplot(gs_row[i]) for i in range(2)]
    endotaxis.draw_walk(axes_walk, endo_seq, endo_signal, end_time=len(endo_seq),
                        end_corr=max(endo_seq))
    axes_walk[0].text(0.5, 1.0, "Mask A", ha="center", va="bottom", transform=axes_walk[0].transAxes,
                      fontsize=plot_utils.FONT_SIZE, color=plot_utils.mask_colors["A"])
    axes_walk[0].set_xticks([0, 3, 5, 7, 9])
    axes_walk[0].set_xticklabels(["Home", 4, 6, 8, "Out"])
    axes_walk[0].set_xlabel("")   # Home->Out ticks already label the corridor axis; avoid
                                  # colliding with the error grid's group headers below

    # Three snapshots of the learned map (red = edges added since the previous snapshot): a
    # late-Home step, the first Out arrival, and the last Out step (coloured H-O / O-O / O-H).
    min_idx = np.where(endo_seq == endo_seq.min())[0]
    max_idx = np.where(endo_seq == endo_seq.max())[0]
    step_indices = [min_idx[-2], max_idx[0], max_idx[-1]]
    axes_adj = [FIG.add_subplot(gs_row[i + 2]) for i in range(len(step_indices))]
    colors = [plot_utils.bout_type_color_dict[c] for c in ("H-O", "O-O", "O-H")]
    first_mat = np.zeros_like(endo_adj[0])
    for k, si in enumerate(step_indices):
        sub_signal = endo_signal[si]
        plot_utils.plot_circle_with_signal_values(axes_adj[k], sub_signal, radius=radius,
                                                  H_circle_x=H_circle_x, n_nodes=n_nodes,
                                                  V_circle_x=V_circle_x, y_scale=y_scale)
        plot_utils.plot_edges_based_on_adj_mat(axes_adj[k], first_mat, edge_color="tab:gray",
                                               linewidth=config.LW_DATA, H_circle_x=H_circle_x, n_nodes=n_nodes,
                                               V_circle_x=V_circle_x, y_scale=y_scale)
        plot_utils.plot_edges_based_on_adj_mat(axes_adj[k], endo_adj[si] - first_mat,
                                               edge_color="red", linewidth=config.LW_EMPHASIS,
                                               H_circle_x=H_circle_x, n_nodes=n_nodes,
                                               V_circle_x=V_circle_x, y_scale=y_scale)
        first_mat = endo_adj[si]
        axes_adj[k].text(0.5, 1.0, f"t={si}", ha="center", va="bottom", transform=axes_adj[k].transAxes,
                         fontsize=plot_utils.FONT_SIZE, color=colors[k])
        axes_adj[k].axis("off")
        axes_adj[k].set_aspect("equal", adjustable="box")
        axes_adj[k].set_ylim([0, n_nodes * y_scale])

    # The learned goal signal over corridors at the final snapshot (monotone Home->Out).
    ax_goal = FIG.add_subplot(gs_row[5])
    plot_utils.plot_goal_signal(ax_goal, endo_signal[step_indices[-1]], cmap=plt.cm.plasma,
                                color="black", linewidth=config.LW_DATA)
    ax_goal.set_xticks([0, 3, 5, 7, 9])
    ax_goal.set_xticklabels(["Home", 4, 6, 8, "Out"])
    ax_goal.set_xlabel("")
    ax_goal.text(0.5, 1.0, f"t={step_indices[-1]}", ha="center", va="bottom",
                 transform=ax_goal.transAxes, color=colors[-1], fontsize=plot_utils.FONT_SIZE)
    ax_bar = FIG.add_subplot(gs_row[6])
    plot_utils.plot_illustrative_cbar(ax_bar, cmap="plasma", ticklabels=["low", "high"],
                                      label_loc="right", aspect=5)
    return axes_walk[0]


ax_panel_a = plot_endotaxis_learning_a(outer[0])

# Dedicated (invisible) strip between Panel A and the grid; the bold group headers are centred in
# it, so they sit clear of both Panel A's x-axis above and the grid's column titles below.
ax_header_strip = FIG.add_subplot(outer[1])
ax_header_strip.set_axis_off()

# Error grid (Panels B–G): 2 rows (B corridor / C turn) x 4 data columns -- animal outbound,
# animal homebound, model-free RL, endotaxis -- then a thin gap and ONE shared colorbar.
gs = outer[2].subgridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.04, 0.05], hspace=0.01, wspace=0.06)
# Animal columns (0,1) share the full-width traverse axis; the two model columns (2,3) share their
# OWN x-axis, truncated to RL_MAX_TRAVERSE.
ax0 = FIG.add_subplot(gs[0, 0])
ax_rl = FIG.add_subplot(gs[0, 2])                       # model-free RL top: independent x
ax_A = [ax0,
        FIG.add_subplot(gs[0, 1], sharex=ax0),
        ax_rl,
        FIG.add_subplot(gs[0, 3], sharex=ax_rl)]         # endotaxis shares the model x-axis
ax_B = [FIG.add_subplot(gs[1, 0], sharex=ax0),
        FIG.add_subplot(gs[1, 1], sharex=ax0),
        FIG.add_subplot(gs[1, 2], sharex=ax_rl),
        FIG.add_subplot(gs[1, 3], sharex=ax_A[3])]
cax = FIG.add_subplot(gs[:, 5])  # single shared viridis colorbar, far right


def positions_from_matrix(matrix):
    """Distance-matrix rows (reward row 0 dropped) as a plot_level_error_array dict."""
    return {k: matrix[k + 1][None, :] for k in range(matrix.shape[0] - 1)}


def staircase_dict(stair):
    """A model-prediction matrix (reward row already dropped) as a plot_level_error_array dict."""
    return {k: stair[k][None, :] for k in range(stair.shape[0])}


# Row B — corridor error rate by position (no 0.5 chance line on corridors).
for c, (dcode, dname) in enumerate(DIRECTIONS):
    plot_level_error_array(ax_A[c], positions_from_matrix(corridor_data[dcode]),
                           cmap=cmap_pos, chance_level=0.5, smooth_window=SMOOTH_WINDOW,
                           linewidth=LINEWIDTH, ylim=Y_TOP, xlabel="",
                           ylabel="Error rate" if c == 0 else "")
for ax, stair in [(ax_A[2], corridor_staircase), (ax_A[3], endo_corridor)]:
    plot_level_error_array(ax, staircase_dict(stair), cmap=cmap_pos, chance_level=0.5,
                           linewidth=LINEWIDTH, ylim=Y_TOP, xlabel="", ylabel="")

# Row C — per-hole first-decision turn error rate (0.5 chance line kept).
for c, (dcode, dname) in enumerate(DIRECTIONS):
    plot_level_error_array(ax_B[c], {k: arr for k, arr in enumerate(hole_data[dcode])},
                           cmap=cmap_pos, chance_level=0.5, smooth_window=SMOOTH_WINDOW,
                           linewidth=LINEWIDTH, ylim=Y_TOP, xlabel="",
                           ylabel="Error rate" if c == 0 else "")
for ax, stair in [(ax_B[2], turn_staircase), (ax_B[3], endo_turn)]:
    plot_level_error_array(ax, staircase_dict(stair), cmap=cmap_pos, chance_level=0.5,
                           linewidth=LINEWIDTH, ylim=Y_TOP, xlabel="", ylabel="")

for ax in ax_A + ax_B:
    ax.set_ylim(bottom=Y_BOTTOM)

# Truncate the model columns' shared x-axis; animal columns stay full width.
ax_rl.set_xlim(ax0.get_xlim()[0], RL_MAX_TRAVERSE + 0.5)

# Column direction titles: the two animal columns are outbound/homebound; each model column serves
# both directions (identical prediction), labelled compactly.
for c, (_, dname) in enumerate(DIRECTIONS):
    ax_A[c].text(0.5, 1.0, dname.capitalize(), transform=ax_A[c].transAxes,
                 ha="center", va="bottom", fontsize=plot_utils.TICK_SIZE)
for c in (2, 3):
    ax_A[c].text(0.5, 1.0, "Outbound/Homebound", transform=ax_A[c].transAxes,
                 ha="center", va="bottom", fontsize=plot_utils.TICK_SIZE)

# Shared traverse x-axis: number + "Traverse #" label only on the bottom row.
for ax in ax_A:
    ax.tick_params(labelbottom=False)
for ax in ax_B:
    ax.set_xlabel("Traverse #")
for ax in ax_A[1:] + ax_B[1:]:
    ax.tick_params(labelleft=False)

# label=None: keep the colorbar's axis label OUT of the layout (a rotated axis label reserves a
# wide right margin and squeezes the data panels); it is added back as floating text below.
add_distance_colorbar(FIG, None, sm, n_pos, cax=cax,
                      ticklocation="right", show_ticklabels=True, label=None)

# Run the constrained-layout solver now so every get_position() below returns FINAL geometry.
FIG.draw_without_rendering()

def _group_center(axes):
    """Return the figure-fraction x-center spanning a group of ``axes`` (for centered headers)."""
    boxes = [ax.get_position() for ax in axes]
    return 0.5 * (min(b.x0 for b in boxes) + max(b.x1 for b in boxes))

# Bold group headers for the error grid, centred in the dedicated strip between Panel A and grid.
strip_pos = ax_header_strip.get_position()
header_y = 0.5 * (strip_pos.y0 + strip_pos.y1)
FIG.text(_group_center(ax_A[:2]), header_y, "Day 1 Mask A",
         ha="center", va="center", fontweight="bold", fontsize=plot_utils.FONT_SIZE)
FIG.text(_group_center([ax_A[2]]), header_y, "Model-free RL",
         ha="center", va="center", fontweight="bold", fontsize=plot_utils.FONT_SIZE)
FIG.text(_group_center([ax_A[3]]), header_y, "Endotaxis",
         ha="center", va="center", fontweight="bold", fontsize=plot_utils.FONT_SIZE)
# Rotated row-group headers in the reserved far-left strip, one per error row.
row_x = 0.02
for row_axes, row_label in [(ax_A, "Corridor"), (ax_B, "Turn")]:
    pos = row_axes[0].get_position()
    FIG.text(row_x, 0.5 * (pos.y0 + pos.y1), row_label, rotation=90,
             ha="center", va="center", fontsize=plot_utils.FONT_SIZE)

# Colorbar label as floating text hugged tight to the bar (kept out of the layout via label=None
# above), so the data panels reclaim the width a rotated axis label would otherwise reserve.
cax_pos = cax.get_position()
FIG.text(cax_pos.x1 + 0.05, 0.5 * (cax_pos.y0 + cax_pos.y1), "Distance to reward",
         rotation=90, ha="center", va="center", fontsize=plot_utils.FONT_SIZE)

# Panel letters (figure-fraction coords, chosen by eye per the plot-refactor convention): A =
# endotaxis-simulation row; B-D = corridor grid groups; E-G = turn grid groups. The left-column
# letters (A, B, E) sit right of the reserved left strip so they clear the rotated "Corridor"/"Turn"
# row headers and the y-axis tick labels; the model/animal column x's and the two row y's are
# shared across the grid.
plot_utils.add_letter_labels(FIG, [(0.01, 0.99),                              # A  Panel A (endotaxis sim)
                                    (0.01, 0.65), (0.47, 0.65), (0.68, 0.65),  # B, C, D  corridor row
                                    (0.01, 0.34), (0.47, 0.34), (0.68, 0.34)]) # E, F, G  turn row

config.save_figure(FIG, "error_propagation_supp.pdf", save_path)
