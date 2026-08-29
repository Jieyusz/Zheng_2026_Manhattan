from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
import numpy as np

# add arg parser for saving figure path
import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)
save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 3.8

## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()
mask_color_dict = {"First A": "tab:grey",
                "Repeat A": plot_utils.mask_colors["A"],
                    "Mask B": plot_utils.mask_colors["B"],
                   "Mask C": plot_utils.mask_colors["C"],}

# shared kwargs for every mean-trajectory curve panel; displace_bars offsets each
# group horizontally so overlapping error bars stay legible.
CURVE_KWARGS = dict(stats_type="mean", linewidth=config.LW_DATA, plot_shade=True,
                    connect_scatters=True, plot_scatter=False, displace_bars=True)


def acortical_mask_dict(metric, first_a_metric=None):
    """Assemble the four-condition metric dict for the Acortical generalization panels.

    Builds a mapping with keys ``"First A"``, ``"Repeat A"``, ``"Mask B"`` and
    ``"Mask C"`` by looking up the corresponding entries in the module-level
    ``figure_data_dict``, so the Acortical learning trajectories across the A/B/C
    mask sequence share a common color map.

    Parameters
    ----------
    metric : str
        Metric suffix shared by the Repeat-A, Mask-B and Mask-C keys, e.g.
        ``"reward intervals"``, ``"sortie counts"`` or ``"traverse duration"``.
    first_a_metric : str, optional
        Metric suffix to use for the ``"First A"`` key when the first-exposure data is
        stored under a different name (e.g. ``"duration"`` instead of
        ``"traverse duration"``). Defaults to ``metric``.

    Returns
    -------
    dict of str to numpy.ndarray
        Condition label -> ``(n_animals, n_traverses)`` / ``(n_animals, n_rewards)``
        NaN-padded array (see docs/data_contracts.md §"figure_data files").
    """
    first_a_metric = first_a_metric or metric
    return {"First A": figure_data_dict[f"Acortical Mask A {first_a_metric}"],
            "Repeat A": figure_data_dict[f"Acortical Mask A repeat {metric}"],
            "Mask B": figure_data_dict[f"Acortical Mask B {metric}"],
            "Mask C": figure_data_dict[f"Acortical Mask C {metric}"]}


FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
# 2 rows x 3 columns, all data. All six panels show the same four mask conditions, so ONE
# legend serves the whole figure; it lives inside panel A (whose curves decay away from the
# upper right) rather than in a dedicated column, which would cost every panel ~18 % width.
gs0 = FIG.add_gridspec(2, 3)

# row 1: how quickly the Acortical mice re-learn on each new 9-turn mask
row1_panels = [(acortical_mask_dict("reward intervals"), "Reward #", "Interval (s)", 15 * 60),
               (acortical_mask_dict("sortie counts"), "Reward #", "N(sorties)", 40),
               (acortical_mask_dict("traverse duration", first_a_metric="duration"),
                "Traverse #", "Duration (s)", 200)]
axes_row1 = [FIG.add_subplot(gs0[0, i]) for i in range(3)]
for i, (ax, (data, xlabel, ylabel, ylim)) in enumerate(zip(axes_row1, row1_panels)):
    # every call draws a legend inside its own panel; only panel A's is kept (see gs0 comment)
    plot_utils.plot_array_comparison(ax, data, colordict=mask_color_dict, xlabel=xlabel,
                                     ylabel=ylabel, ylim=ylim, **CURVE_KWARGS)
    if i == 0:
        plot_utils.add_panel_title(ax, "Mask A, B, C")
    else:
        ax.get_legend().remove()

# row 2: the three graph error rates across the mask sequence. Corridor error rate (E) is
# the readout for the rule-based -- not sequence-based -- claim in the Discussion; tile
# error rate (F) is its finer-grained companion (same >=0 per-step definition, chance ~0.5).
error_panels = [("traverse turn error rate", "turn error rate", "Turn error rate"),
                ("traverse corridor error rate", "corridor error rate", "Corridor error rate"),
                ("traverse tile error rate", "tile error rate", "Tile error rate")]
axes_row2 = []
for i, (metric, first_a_metric, ylabel) in enumerate(error_panels):
    ax_error = FIG.add_subplot(gs0[1, i])
    axes_row2.append(ax_error)
    plot_utils.plot_array_comparison(ax_error, acortical_mask_dict(metric, first_a_metric=first_a_metric),
                                     stats_type="mean", markersize=0, linewidth=config.LW_DATA,
                                     colordict=mask_color_dict, plot_shade=True, connect_scatters=True,
                                     shade_alpha=1, plot_scatter=False, displace_bars=True,
                                     xlabel="Traverse #", ylabel=ylabel, ylim=0.5)
    ax_error.get_legend().remove()  # panel A carries the one legend for all six panels

# one letter per panel, in reading order; x is shared down each column of the 2x3 grid
plot_utils.add_letter_labels(FIG, [(0.01, 0.98), (0.34, 0.98), (0.67, 0.98),
                                   (0.01, 0.49), (0.34, 0.49), (0.67, 0.49)])
config.save_figure(FIG, "ac_bc_supp.pdf", save_path)
