from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
import numpy as np

# add arg parser for saving figure path
import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)
save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 3.9

## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()

# shared kwargs for every mean-trajectory curve panel; displace_bars offsets each
# group horizontally so overlapping error bars stay legible.
CURVE_KWARGS = dict(stats_type="mean", linewidth=config.LW_DATA, plot_shade=True,
                    connect_scatters=True, plot_scatter=False, displace_bars=True)


def mask_d_dict(mask_metric, wildtype_metric):
    """Assemble the Mask-D genotype-comparison metric dict for a curve panel.

    Gathers the Acortical and Control per-animal arrays under ``"Mask D {mask_metric}"``
    and appends the Wildtype array under ``"Wildtype D {wildtype_metric}"`` (truncated to
    the first 20 rewards/traverses so all genotypes span the same x-range). The Wildtype
    suffix is passed separately because it does not follow the ``"Mask D ..."`` naming
    (e.g. ``"Wildtype D duration"`` vs ``"Mask D traverse duration"``).

    Parameters
    ----------
    mask_metric : str
        Metric suffix following ``"Mask D "`` for the Acortical/Control keys.
    wildtype_metric : str
        Metric suffix following ``"Wildtype D "`` for the Wildtype key.

    Returns
    -------
    dict of str to numpy.ndarray
        Mapping ``{"Acortical", "Control", "Wildtype"}`` to per-animal arrays.
    """
    return {**utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:2], f"Mask D {mask_metric}"),
            "Wildtype": figure_data_dict[f"Wildtype D {wildtype_metric}"][:, :20]}


def mean_similarity(genotype):
    """Per-animal similarity averaged over the three groups, dropping unscored animals.

    Parameters
    ----------
    genotype : str
        One of ``config.GENOTYPES``.

    Returns
    -------
    numpy.ndarray, shape (n_scored_animals,)
        Mean adjusted-Jaccard similarity per animal.
    """
    per_animal = np.array(figure_data_dict[f"{genotype} D average traverse similarity"])
    keep = ~np.isnan(per_animal).all(axis=1)  # animals with no similarity data at all
    return np.nanmean(per_animal[keep], axis=1)


FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
# A-C: how the three genotypes learn Mask D. D-F: how similar their routes are -- D is one
# example acortical mouse's three similarity matrices, E and F the population summaries.
gs0 = FIG.add_gridspec(2, 1)

# row 1: genotype comparison under Mask D
gs00 = gs0[0].subgridspec(1, 3)
d_panels = [(mask_d_dict("reward intervals", "reward intervals"), "Reward #", "Interval (s)", 30 * 60),
            (mask_d_dict("sortie counts", "sortie counts"), "Reward #", "N(Sorties)", 30),
            (mask_d_dict("traverse duration", "duration"), "Traverse #", "Duration (s)", 300)]
axes_d = [FIG.add_subplot(gs00[i]) for i in range(3)]
for i, (ax, (data, xlabel, ylabel, ylim)) in enumerate(zip(axes_d, d_panels)):
    # rows 1 and 2 are both genotype-coded, so ONE legend serves the figure; it sits inside
    # panel A (see plot_bc_gen_supp.py, which follows the same convention)
    plot_utils.plot_array_comparison(ax, data, xlabel=xlabel, ylabel=ylabel, ylim=ylim,
                                     **CURVE_KWARGS)
    if i == 0:
        ax.text(0.5, 1, "Mask D", fontsize=plot_utils.FONT_SIZE, ha="center",
                va="bottom", transform=ax.transAxes)
    else:
        ax.get_legend().remove()

# row 2: Mask-D route similarity (adjusted Jaccard, see sec:similarity). D is the per-animal
# view -- one example acortical mouse's three traverse-similarity matrices, drawn as
# fig:d_motif A does for a wildtype mouse: same 3-matrices-plus-colorbar block, hence the
# 0.1-wide colorbar slot. E repeats the fig:d_motif B readout for Acortical mice and F
# compares the overall level across genotypes; both show routes stay dissimilar, i.e. the
# animals do not replay a fixed turn sequence.
# The matrix columns get the larger ratios even though they hold the smaller panels: the
# matrices are aspect-equal squares, so they are limited by cell *width* (0.64 in of a
# 1.37 in-tall cell) and extra width is the only thing that enlarges them.
gs10 = gs0[1].subgridspec(1, 6, width_ratios=[1.55, 1.55, 1.55, 0.1, 0.95, 0.8])
axes_mat = [FIG.add_subplot(gs10[j]) for j in range(4)]
ac_similarity_list = figure_data_dict["Acortical D similarity matrices"]
j_oo, j_hh, j_oh_prime = ac_similarity_list[config.ACORTICAL_D_SIMILARITY_EXAMPLE_ID]
assert min(j_oo.shape) >= config.SIMILARITY_EXAMPLE_MIN_SIDE, (
    f"ACORTICAL_D_SIMILARITY_EXAMPLE_ID={config.ACORTICAL_D_SIMILARITY_EXAMPLE_ID} selects a "
    f"{j_oo.shape} matrix; most entries in this list are 1x1 (a single traverse pair) and the "
    "list order reshuffles when gen_ac_generalization.py is regenerated, so re-pick the index.")
# tick-size labels: at this panel width the titles and axis labels would otherwise take a
# large share of the cell, and only spare width can enlarge the aspect-equal matrices
plot_utils.plot_maskd_similarity_matrix(axes_mat, j_oo=j_oo, j_hh=j_hh, j_oh_prime=j_oh_prime,
                                        labels=list(config.SIMILARITY_LATEX.values()),
                                        axis_labels=config.TRAVERSE_LATEX,
                                        cmap=plt.cm.plasma, plot_colorbar=True,
                                        label_fontsize=plot_utils.TICK_SIZE)

ax_sim_ac = FIG.add_subplot(gs10[4])
ac_sims = np.array(figure_data_dict["Acortical D average traverse similarity"])
# Friedman is a repeated-measures test, so restrict to animals scored in all three groups
# (an animal with too few traverses of one bout type has NaN for that group).
ac_sims = ac_sims[~np.isnan(ac_sims).any(axis=1)]
ac_sims_dict = {label: ac_sims[:, i] for i, label in enumerate(config.SIMILARITY_LATEX)}
plot_utils.plot_group_scatter_box_comparison(
    ax_sim_ac, ac_sims_dict, utils.friedman_with_pairwise_wilcoxon(ac_sims_dict),
    colordict={label: plot_utils.genotype_colors["Acortical"] for label in config.SIMILARITY_LATEX},
    ylabel="Mean similarity", plot_ns=True)
ax_sim_ac.set_xticklabels(list(config.SIMILARITY_LATEX.values()))
# rotated as in panel F: the three J labels overprint each other at this panel width
ax_sim_ac.tick_params(axis="x", rotation=45)
ax_sim_ac.text(0.5, 1, "Acortical", fontsize=plot_utils.TICK_SIZE, ha="center", va="bottom",
               transform=ax_sim_ac.transAxes, color=plot_utils.genotype_colors["Acortical"])

# genotype comparison: one value per animal, averaged over the three similarity groups
ax_sim_gt = FIG.add_subplot(gs10[5])
sim_by_genotype = {gt: mean_similarity(gt) for gt in config.GENOTYPES}
plot_utils.plot_group_scatter_box_comparison(
    ax_sim_gt, sim_by_genotype, utils.kruskal_with_pairwise_mann_whitney(sim_by_genotype),
    colordict=plot_utils.genotype_colors, ylabel="Mean similarity", plot_ns=True)
ax_sim_gt.tick_params(axis="x", rotation=45)
# E and F plot the SAME quantity, so they must share a y-scale to be comparable; autoscaling
# put them on different scales (E is inflated by its three significance brackets).
_sim_top = max(ax_sim_ac.get_ylim()[1], ax_sim_gt.get_ylim()[1])
for _ax in (ax_sim_ac, ax_sim_gt):
    _ax.set_ylim(0, _sim_top)
# F's y axis is E's, so its label and ticks are redundant; hiding them buys the row width
# back (same trick as fig:d_motif's off-diagonal panels).
ax_sim_gt.yaxis.set_visible(False)

# one letter per panel, in reading order; row 2's matrix triplet gets a single letter (D), as
# fig:d_motif does for the same block
plot_utils.add_letter_labels(FIG, [(0.01, 0.98), (0.35, 0.98), (0.67, 0.98),
                                   (0.01, 0.48), (0.72, 0.48), (0.89, 0.48)])
config.save_figure(FIG, "ac_d_supp.pdf", save_path)
