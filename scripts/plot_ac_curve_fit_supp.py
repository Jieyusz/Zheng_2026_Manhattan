import numpy as np
from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)

save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 6.2

d_params = [("D_0", r"$D_0$ (s)", (0, 800)), ("D_infty", r"$D_{\infty}$ (s)", (0, 100)), ("delta", r"$\delta$ (/traverse)", (0, 0.5))]
e_params = [("E_0", r"$E_0$", (0, 0.8)), ("E_infty", r"$E_{\infty}$", (0, 0.25)), ("epsilon", r"$\epsilon$ (/traverse)", (0, 0.2))]
## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()

# Plot the figure. Outer 1x2 split: left block holds the absolute-estimate panels
# (unchanged), right column holds the cross-genotype ratio forest panels.
FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
# 3x2 outer: left block spans all rows (outer[:, 0]); the three right-hand ratio panels come
# directly from outer[r, 1] (no nested subgridspec). Right rows favour the two 6-parameter
# generalization panels over the shorter 3-parameter Mask-D panel.
outer = FIG.add_gridspec(3, 2, width_ratios=[4, 1], wspace=0.06, height_ratios=[1, 1.6, 1.6])
# Left block runs in Results order: first-mask fits (A, B), then generalization (C, D), then Mask D
# (E). gs00 holds the 2 first-mask rows, gs10 the 3 remaining rows; height_ratios keep the five rows
# roughly equal.
# Three blocks with hspace=0 between them, so each dashed section rule lands in a clean gap
# rather than clipping an axis (the section titles sit just below each rule).
# hspace opens a gap between blocks for the bottom row's category labels and the dashed
# section rule; the rule and letter positions below are measured, not hardcoded.
gs0 = outer[:, 0].subgridspec(3, 1, hspace=0.28, height_ratios=[0.8, 0.8, 0.45])
gs00 = gs0[0].subgridspec(2, 3, )

def plot_param_comparisons_with_ci(ax, fit_results_dict, param_name="D_infty", ylim=(0, 50),
                                   colordict=plot_utils.genotype_colors, latex_str=None, width=0.3,
                                   plot_shade=True, per_animal_dict=None):
    """
    Draw grouped point-estimate-with-CI markers comparing a fit parameter across genotypes.

    For each genotype in ``fit_results_dict``, plots the parameter estimate and its
    bootstrap confidence interval (see ``docs/data_contracts.md`` §"Fitted curve tuple":
    ``summary_df`` columns ``Estimate``/``CI_lower``/``CI_upper``), shading the Control
    CI band when requested.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    fit_results_dict : dict
        Maps genotype label -> fit-results tuple ``(bs, ds, summary_df, _)``.
    param_name : str, optional
        Parameter row to plot, e.g. ``"D_infty"`` (duration, seconds) or
        ``"epsilon"`` (turn-error decay rate, per traverse).
    ylim : tuple of float, optional
        ``(low, high)`` y-axis limits in the parameter's units.
    colordict : dict, optional
        Maps genotype label -> color.
    latex_str : str, optional
        LaTeX y-axis label; defaults to ``f"${param_name}$"``.
    width : float, optional
        Half-width of the Control CI shade band, in x-axis (category) units.
    plot_shade : bool, optional
        If True, shade the Control genotype's CI band.
    per_animal_dict : dict, optional
        Maps ``param_name -> 1-D array of per-animal parameter values``. When given,
        the Control genotype is drawn as jittered per-animal points (no estimate/CI,
        no shade) instead of a bootstrap estimate — the n=3 control cohort's bootstrap
        CI is degenerate, so it is shown honestly as individual points (matching the
        raw-trace control treatment in ``plot_ac_rapid.py``). Other genotypes are
        unaffected. Only passed for the First-Mask-A panels.

    Returns
    -------
    None
        Draws onto ``ax`` in place.
    """
    if latex_str is None:
        latex_str = rf"${param_name}$"
    name_list = list(fit_results_dict.keys())
    for gt, results in fit_results_dict.items():
        bs, ds, summary_df, _ = results
        color = colordict[gt]
        label_index = name_list.index(gt)
        # n=3 control: individual per-animal parameter points, not a degenerate
        # bootstrap CI/shade (see per_animal_dict above; plot_ac_rapid.py).
        if per_animal_dict is not None and gt == "Control":
            ys = np.asarray(per_animal_dict[param_name], dtype=float)
            lo, hi = ylim
            in_range = ys[(ys >= lo) & (ys <= hi)]
            if in_range.size:
                plot_utils.plot_jittered_scatter(ax, label_index, in_range, color=color)
            # Single-animal fits can rail the learning rate to its bound, off the tight
            # rate-panel ylim; show them honestly (not dropped) as capped triangles at
            # the panel edge rather than widening ylim. See caption / recipe caveat.
            above, below = ys[ys > hi], ys[ys < lo]
            if above.size:
                plot_utils.plot_jittered_scatter(ax, label_index, np.full(above.size, hi),
                                                 color=color, marker="^", clip_on=False)
            if below.size:
                plot_utils.plot_jittered_scatter(ax, label_index, np.full(below.size, lo),
                                                 color=color, marker="v", clip_on=False)
            continue
        # shared error-bar + value annotation (see plot_stats.plot_param_estimate_with_ci)
        _, ci_lower, ci_upper = plot_utils.plot_param_estimate_with_ci(
            ax, label_index, summary_df, param_name, color, ylim, markersize=config.MS_PT_DEFAULT, capsize=config.CAPSIZE)
        # add fill_with for control
        if gt == "Control" and plot_shade:
            plot_utils.plot_ci_shade(ax, label_index, ci_lower, ci_upper, color, width=width, alpha=0.3)


    # format axes:
    ax.set_xticks(range(len(name_list)))
    # Rotated because three category names ("Acortical Control Wildtype", "Repeat A Mask B
    # Mask C") do not fit horizontally in a ~1.5 in panel: measured gaps between adjacent
    # labels were 0.01-0.04 in, i.e. touching. Same 45-degree convention as the gap-range
    # labels in plot_ac_mem_supp.py.
    ax.set_xticklabels(name_list, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim([-0.4, len(name_list)-1+0.4])
    ax.axhline(0, color="black", linestyle="--", linewidth=config.LW_HAIRLINE, zorder=config.Z_REFERENCE)
    ax.set_ylim(ylim)
    ax.set_ylabel(latex_str)

wildtype_fit_result = figure_data_dict["Wildtype two day duration fit results"][(1, "A")] # Day-1 Mask A reference
duration_fit_results = {**utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:2], "A duration fit results"),
                        "Wildtype": wildtype_fit_result}
# n=3 control per-animal parameter points (replaces the degenerate bootstrap CI, panel A).
duration_per_animal = figure_data_dict["Control A duration per-animal params"]
duration_per_animal_dict = {p: duration_per_animal[p].to_numpy() for p, _, _ in d_params}
d_axes = [FIG.add_subplot(gs00[0, i]) for i in range(3)]
for k, (param_name, latex_str, ylim) in enumerate(d_params):
    plot_param_comparisons_with_ci(d_axes[k], duration_fit_results, param_name, ylim=ylim, latex_str=latex_str,
                                   per_animal_dict=duration_per_animal_dict)
    # hide x ticks
    d_axes[k].set_xticklabels([])
# Row headings over parameter-comparison panels: these panels carry a value/CI
# annotation band above their top spine, so the heading hangs the standard TITLE_PAD
# gap above the *band* rather than above the spine.
plot_utils.add_panel_title(d_axes[0], "First (Mask A)", anchor=config.PARAM_ANNOTATION_Y)

wildtype_error = figure_data_dict["Wildtype two day turn error rate fit results"][(1, "A")] # Day-1 Mask A reference
error_fit_results = {**utils.select_by_prefix(figure_data_dict, config.GENOTYPES[:2], "A turn error rate fit results"),
                     "Wildtype": wildtype_error}
# n=3 control per-animal parameter points (replaces the degenerate bootstrap CI, panel B).
error_per_animal = figure_data_dict["Control A turn error rate per-animal params"]
error_per_animal_dict = {p: error_per_animal[p].to_numpy() for p, _, _ in e_params}
e_axes = [FIG.add_subplot(gs00[1, i]) for i in range(3)]
for k, (param_name, latex_str, ylim) in enumerate(e_params):
    plot_param_comparisons_with_ci(e_axes[k], error_fit_results, param_name, ylim=ylim, latex_str=latex_str,
                                   per_animal_dict=error_per_animal_dict)
    # B is the bottom row of the first-mask block, so it carries the block's category labels
    # (the A row above shares them and stays unlabelled).

mask_color_dict = {"Repeat A": plot_utils.mask_colors["A"],
                    "Mask B": plot_utils.mask_colors["B"],
                   "Mask C": plot_utils.mask_colors["C"],}
# add shades of control for comparison
def plot_control_ci_shade(ax, fit_results_dict, param_name="D_infty", colordict=mask_color_dict, name_list=None, width=0.3,
                          alpha=0.3):
    """
    Overlay Control-group CI shade bands behind grouped parameter comparisons.

    For each entry in ``fit_results_dict``, extracts the parameter's bootstrap CI from
    its ``summary_df`` (see ``docs/data_contracts.md`` §"Fitted curve tuple") and draws a
    translucent band at that category's x-position, used as a reference behind the
    generalization (Mask A/B/C) comparisons.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    fit_results_dict : dict
        Maps category label (e.g. ``"Repeat A"``, ``"Mask B"``) -> fit-results tuple
        ``(bs, ds, summary_df, _)``.
    param_name : str, optional
        Parameter row whose CI is shaded, e.g. ``"D_infty"`` (seconds).
    colordict : dict, optional
        Maps category label -> color.
    name_list : list of str, optional
        Ordered category labels controlling x-position; defaults to
        ``list(fit_results_dict.keys())``.
    width : float, optional
        Half-width of the shade band, in x-axis (category) units.
    alpha : float, optional
        Opacity of the shade band.

    Returns
    -------
    None
        Draws onto ``ax`` in place.
    """
    if name_list is None:
        name_list = list(fit_results_dict.keys())
    for gt, results in fit_results_dict.items():
        bs, ds, summary_df, _ = results
        color = colordict[gt]
        label_index = name_list.index(gt)
        # shared CI extraction + shade band (see plot_stats.get_param_ci / plot_ci_shade)
        _, ci_lower, ci_upper = plot_utils.get_param_ci(summary_df, param_name)
        plot_utils.plot_ci_shade(ax, label_index, ci_lower, ci_upper, color, width=width, alpha=alpha)


# add generalization results (panels C, D); Mask D (panel E) gets its own block, gs20.
gs10 = gs0[1].subgridspec(2, 3, hspace=0.05, width_ratios=[1, 1, 1])
gs20 = gs0[2].subgridspec(1, 3, width_ratios=[1, 1, 1])

def gen_fit_dict(genotype, metric):
    """
    Assemble the generalization fit-results dict for one genotype and metric.

    Looks up the saved curve-fit results (see ``docs/data_contracts.md`` §"figure_data
    files") for the three generalization conditions and returns them keyed by their
    display labels.

    Parameters
    ----------
    genotype : str
        Genotype prefix used in the figure-data keys, e.g. ``"Acortical"`` or
        ``"Control"``.
    metric : str
        Metric name embedded in the key, e.g. ``"duration"`` (seconds) or
        ``"turn error rate"`` (fraction in [0, 1]).

    Returns
    -------
    dict
        Maps ``{"Repeat A", "Mask B", "Mask C"}`` -> fit-results tuple
        ``(bs, ds, summary_df, bootstrap_curves)``.
    """
    return {"Repeat A": figure_data_dict[f"{genotype} A repeat Gen {metric} fit results"],
            "Mask B": figure_data_dict[f"{genotype} B Gen {metric} fit results"],
            "Mask C": figure_data_dict[f"{genotype} C Gen {metric} fit results"]}


gen_duration_results = gen_fit_dict("Acortical", "duration")
ct_duration_results = gen_fit_dict("Control", "duration")

gen_d_axes = [FIG.add_subplot(gs10[0, i]) for i in range(3)]
for k, (param_name, latex_str, ylim) in enumerate(d_params):
    if param_name == "D_0":
        ylim = (0, 200) # use the new one for generalization

    plot_param_comparisons_with_ci(gen_d_axes[k], gen_duration_results, param_name, ylim=ylim, latex_str=latex_str, colordict=mask_color_dict)
    plot_control_ci_shade(gen_d_axes[k], ct_duration_results, param_name, name_list= list(gen_duration_results.keys())),
    gen_d_axes[k].set_xticklabels([])
plot_utils.add_panel_title(gen_d_axes[0], "Generalization: A, B, C",  # see note at panel A
                           anchor=config.PARAM_ANNOTATION_Y)

# turn error rate
gen_error_results = gen_fit_dict("Acortical", "turn error rate")
ct_error_results = gen_fit_dict("Control", "turn error rate")

gen_e_axes = [FIG.add_subplot(gs10[1, i]) for i in range(3)]
for k, (param_name, latex_str, ylim) in enumerate(e_params):
    plot_param_comparisons_with_ci(gen_e_axes[k], gen_error_results, param_name, ylim=ylim, latex_str=latex_str, colordict=mask_color_dict)
    plot_control_ci_shade(gen_e_axes[k], ct_error_results, param_name, name_list= list(gen_error_results.keys()))
    # D is the bottom row of the generalization block and carries its category labels. These
    # differ from the genotype labels on B/E (Repeat A / Mask B / Mask C), so without them the
    # generalization panels cannot be read at all.

# Mask D duration fits (panel E), last left-column row so the block order matches the Results text.
maskd_duration_axes = [FIG.add_subplot(gs20[0, i]) for i in range(3)]
maskd_duration_results = {"Acortical": figure_data_dict["Acortical D Gen duration fit results"],
    "Control": figure_data_dict["Control D duration fit results"],
               "Wildtype": figure_data_dict["Wildtype D duration fit results"]}
for k, (param_name, latex_str, ylim) in enumerate(d_params):
    if param_name == "D_0":
        ylim = (0, 500) # use the new one for generalization
    elif param_name == "delta":
        ylim = (0, 0.5)
    else:
        ylim = (0, 80)
    plot_param_comparisons_with_ci(maskd_duration_axes[k], maskd_duration_results, param_name, ylim=ylim, latex_str=latex_str,
                                   plot_shade=False)
plot_utils.add_panel_title(maskd_duration_axes[0], "Mask D",  # see note at panel A
                           anchor=config.PARAM_ANNOTATION_Y)
# --- Right-hand cross-genotype ratio column (division; see docs/ratio_ci_method.md) ---
# Forest panels of the fitted-parameter ratios between cohorts. Colors match the left-block
# panels (genotype_colors / mask_colors), so no per-panel legend is needed. The x-axis is cut
# at 1.1: weakly-identified rate rows (delta/epsilon) whose median exceeds 1.1 run off-panel by
# design, keeping the well-identified late-value / initial rows readable.
def plot_ratio_panel(ax, ratio_df, series, title, xlabel=""):
    """Draw one ratio forest panel; ``series`` is a list of ``(Comparison, color)``."""
    for i, (comp, color) in enumerate(series):
        sub = ratio_df[ratio_df.Comparison == comp]
        plot_utils.plot_ci_ratios(ax, sub, param_latex=config.PARAM_LATEX, color=color,
                                  offset=0.22 * (i - (len(series) - 1) / 2), xlabel=xlabel)
    ax.set_xlim(0, 1.1)  # forest-plot cutoff (see note above)
    # The heading is a Text child of the axes, so it lands in the axes' tight bbox and
    # constrained layout reserves room for the 2-line title on its own -- set_title is not
    # needed for that -- letting the three panels grow to fill the right column.
    plot_utils.add_panel_title(ax, title)

maskd_ratios = figure_data_dict["Mask D genotype param ratios"]                      # Control/WT, Acortical/WT
control_gen_ratios = figure_data_dict["Acortical generalization genotype param ratios"]  # Control/Acortical
wt_gen_ratios = figure_data_dict["Wildtype generalization genotype param ratios"]        # Wildtype/Acortical
maskd_series = [("Control/Wildtype", plot_utils.genotype_colors["Control"]),
                ("Acortical/Wildtype", plot_utils.genotype_colors["Acortical"])]
gen_mask_series = [("Repeat A", plot_utils.mask_colors["A"]),
                   ("Mask B", plot_utils.mask_colors["B"]),
                   ("Mask C", plot_utils.mask_colors["C"])]
# Mask D (duration only, 3 params), then the two generalization panels, each all 6 params
# (both metrics) with masks A/B/C as offset series.
plot_ratio_panel(FIG.add_subplot(outer[0, 1]), maskd_ratios, maskd_series, "Mask D\n$\\div$ Wildtype")
plot_ratio_panel(FIG.add_subplot(outer[1, 1]), control_gen_ratios, gen_mask_series,
                 "Generalization\nControl/Acortical")
plot_ratio_panel(FIG.add_subplot(outer[2, 1]), wt_gen_ratios, gen_mask_series,
                 "Generalization\nWildtype/Acortical", xlabel="Ratio")

# Letters and section rules are placed from the LAID-OUT axes, not from hardcoded figure
# fractions: constrained layout decides the real row positions, and they shift whenever the
# figure size, hspace or a row's decoration height changes. The previous constants were an even
# 0.25/0.195 grid that no longer matched the actual rows once the block seams were opened for
# the restored category labels.
FIG.canvas.draw()  # constrained layout must settle before positions can be read
_renderer = FIG.canvas.get_renderer()


def row_extent(axes_row, include_labels=True):
    """
    Figure-fraction (bottom, top) of a row of axes.

    Parameters
    ----------
    axes_row : sequence of matplotlib.axes.Axes
        The axes making up one visual row.
    include_labels : bool, default True
        Use each axes' tight bbox (so tick labels, axis labels and any text above the axes
        count) rather than just the axes rectangle.

    Returns
    -------
    tuple of float
        ``(bottom, top)`` in figure coordinates.
    """
    to_fig = FIG.transFigure.inverted()
    boxes = [(ax.get_tightbbox(_renderer).transformed(to_fig) if include_labels
              else ax.get_position()) for ax in axes_row]
    return min(b.y0 for b in boxes), max(b.y1 for b in boxes)


# Left column rows in Results order, then the three right-hand ratio panels.
_left_rows = [d_axes, e_axes, gen_d_axes, gen_e_axes, maskd_duration_axes]
_right_rows = [[ax] for ax in (FIG.axes[-3], FIG.axes[-2], FIG.axes[-1])]
_LETTER_PAD = 0.015  # figure fraction above each row's axes for the bold letter
plot_utils.add_letter_labels(
    FIG,
    [(0.01, row_extent(row, include_labels=False)[1] + _LETTER_PAD) for row in _left_rows]
    + [(0.76, row_extent(row, include_labels=False)[1] + _LETTER_PAD) for row in _right_rows])

# Dashed section rules sit midway in the seam between blocks, measured with tick labels
# included so a rule can never cut through the labels restored on rows B and D.
_RULE_LINEWIDTH = config.LW_HAIRLINE
# Both rules are the same linewidth, but a 1 pt line is 300/72 = 4.17 device pixels at the
# dpi=300 config.save_figure uses, so one landing mid-pixel rasterizes 4 px wide and the other
# 5 px -- they LOOK like different thicknesses. Snapping each rule to a whole device pixel row
# makes them render identically.
_device_rows = fig_height * 300
for _upper, _lower in [(e_axes, gen_d_axes), (gen_e_axes, maskd_duration_axes)]:
    _rule_y = (row_extent(_upper)[0] + row_extent(_lower)[1]) / 2
    _rule_y = round(_rule_y * _device_rows) / _device_rows
    FIG.add_artist(Line2D([0.01, 0.72], [_rule_y, _rule_y], linestyle='--', color="black",
                          linewidth=_RULE_LINEWIDTH))
config.save_figure(FIG, "ac_curve_fit_supp.pdf", save_path)