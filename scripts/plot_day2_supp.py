import matplotlib.pyplot as plt

from manhattan_maze import plot_utils, utils
import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)

# Argument parsing
save_path = config.parse_save_path()

# Constants
fig_width, fig_height = 5.8, 6
n_sessions = 4
# Data loading
figure_data_dict = utils.load_all_figure_data()

# Figure setup
FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(1, 1,)
gs00 = gs0[0].subgridspec(7, n_sessions, height_ratios=[1]*7, wspace=0)
# Axes initialization
axes_list = [[FIG.add_subplot(gs00[i, j]) for j in range(n_sessions)] for i in range(7)]


# Metric specifications
metrics = [ # (unit, axes, stat, ylim, xlabel, ylabel)
    ("reward intervals", axes_list[0], "mean", 10*60, "Reward #", "Interval (s)"),
    ("sortie counts", axes_list[1], "mean", 10, "Reward #", "N(sorties)"),
    ("speed", axes_list[2], "mean", 4, "Traverse #", "Speed (tiles/s)"),
    ("duration", axes_list[3], "mean", 150, "Traverse #", "Duration (s)"),
    ("turn error rate", axes_list[4], "mean", 0.5, "Traverse #", "Turn\nerror rate"),
    # per-step error RATES ([0,1], chance ~0.5 at the axis top) replacing the unbounded counts.
    # Two-line y-labels so the long "... error rate" text does not overlap the neighbouring panel.
    ("tile error rate", axes_list[5], "mean", 0.5, "Traverse #", "Tile\nerror rate"),
    ("corridor error rate", axes_list[6], "mean", 0.5, "Traverse #", "Corridor\nerror rate"),
]

for session_idx in range(n_sessions): # each figure panel
    for i, (unit, axes, stat, ylim, xlabel, ylabel) in enumerate(metrics):
        plot_objects = []
        mask_data_dict, _ = figure_data_dict[f"Day 2 {unit}"][session_idx] # saved by each session
        shade_alpha = 1
        for mask_idx, mask_name in enumerate(["A", "B", "C"]): # each line group
            color = plot_utils.mask_colors[mask_name] # plot the data in each group
            _, _, line, shade_obj = plot_utils.plot_array_data(
                axes[session_idx], mask_data_dict[mask_name], stats_type=stat, scatter_colors=[color],
                line_color=color, plot_shade=True, plot_scatter=False,
                connect_scatters=True, linewidth=config.LW_DATA, ylim=ylim,shade_alpha=shade_alpha,
                xlabel=xlabel, ylabel=ylabel,
                bar_displacement=0.25*(mask_idx-1)
            )
            plot_objects.append((line, shade_obj))
    # format axes
    plot_utils.add_panel_title(axes_list[0][session_idx], f"Day 2.{session_idx + 1}")
    for k, ax_row in enumerate(axes_list):
        if k != 1 and k != 6:  # keep the x tick marks but drop the numbers/label
            ax_row[session_idx].tick_params(axis="x", labelbottom=False)
            ax_row[session_idx].set_xlabel("")

axes_list[0][-1].legend(handles=plot_objects, labels=[f"Mask {m}" for m in ["A", "B", "C"]], loc="upper right", fontsize=plot_utils.TICK_SIZE)

# The y-axis (tick numbers + label) lives on the leftmost column only; every row shares
# one ylim across the four sessions, so a single labelled column suffices. Left tick
# marks stay on every panel for reading.
for i, (*_, ylabel) in enumerate(metrics):
    for j in range(n_sessions):
        axes_list[i][j].set_ylabel("")  # plot_array_data set a label on every column
        left = j == 0
        axes_list[i][j].tick_params(axis="y", left=True, labelleft=left,
                                    right=False, labelright=False)
    axes_list[i][0].set_ylabel(ylabel)

# Reserve a thin strip at the far left for the panel letters, nudging every axis (and its
# y tick numbers + label) rightward so the bold letters clear them. Constrained layout
# ignores manually-placed FIG.text, so without this margin the letters overlap the labels.
FIG.get_layout_engine().set(rect=(0.01, 0.0, 1.0, 1.0))
plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.01, 0.86), (0.01, 0.69), (0.01, 0.56),
                                   (0.01, 0.43), (0.01, 0.30), (0.01, 0.18)])
config.save_figure(FIG, "day2_supp.pdf", save_path)
