import numpy as np
from manhattan_maze import plot_utils, utils
import matplotlib.pyplot as plt
import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)

save_path = config.parse_save_path()
fig_width = 5.8
fig_height = 5.5
n_sessions = 5  # number of all sessions
d_params = [("D_0", r"$D_0$ (s)", (0, 290)), ("D_infty", r"$D_{\infty}$ (s)", (0, 60)), ("delta", r"$\delta$ (/traverse)", (-0.01, 0.8))]
e_params = [("E_0", r"$E_0$", (0, 0.6)), ("E_infty", r"$E_{\infty}$", (-0.01, 0.3)), ("epsilon", r"$\epsilon$ (/traverse)", (-0.01, 0.2))]
mask_list = ["A", "B", "C"]

## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()

# Plot the figure
FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(2, 1, hspace=0, wspace=0, height_ratios=[1, 5])
gs_schem = gs0[0].subgridspec(1, 2, hspace=0.05, wspace=0.05,)
axes_schematics = [FIG.add_subplot(gs_schem[i]) for i in range(2)]
plot_utils.plot_exponential_schematic(axes_schematics[0], func=utils.exponential_func,
                            parameter_names=[r"$D_0$", r"$D_{\infty}$", r"$\delta$"],
                            equation_string=r"$D_{a,b} = D_{\infty} + \left(D_{0} - D_{\infty}\right)\exp\left[-\delta(b - 1)\right] + \xi^D_{a,b}$",
                            ylabel=r"$D_{a,b}$", color="black")
plot_utils.plot_exponential_schematic(axes_schematics[1], func=utils.exponential_func, parameter_names=[r"$E_0$", r"$E_{\infty}$", r"$\epsilon$"],
                            equation_string=r"$E_{a,b} = E_{\infty} + \left(E_{0} - E_{\infty}\right)\exp\left[-\epsilon(b - 1)\right] + \xi^E_{a,b}$",
                            ylabel=r"$E_{a,b}$", color="black")

gs1 = gs0[1].subgridspec(6, 5, hspace=0, wspace=0, width_ratios=[1, 3, 3, 3, 3])

# add an illustration of parameters for the equation


def plot_two_day_params_with_ci(axes, fit_results, param_name="D_infty", latex_str=None, ylim=(0, 50)):
    """
    Plot one fit parameter across two-day sessions, one session per axis.

    Each ``(session, mask)`` fit-results entry is drawn as a point-estimate-with-CI
    marker on the axis for that session (see ``docs/data_contracts.md`` §"Fitted curve
    tuple"). The Day-1 (session 1) CI band is shaded across every later-session axis as
    a within-subject reference.

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        One axis per session; ``axes[s-1]`` receives session ``s`` (1-based session
        index, not a traverse number).
    fit_results : dict
        Maps ``(session, mask)`` -> fit-results tuple ``(bs, ds, summary_df, _)``.
        ``session`` is 1-based; ``mask`` is one of ``"A"``, ``"B"``, ``"C"``.
    param_name : str, optional
        Parameter row to plot, e.g. ``"D_infty"`` (seconds) or ``"epsilon"``
        (per traverse).
    latex_str : str, optional
        LaTeX y-axis label; defaults to ``f"${param_name}$"``.
    ylim : tuple of float, optional
        ``(low, high)`` y-axis limits in the parameter's units.

    Returns
    -------
    None
        Draws onto ``axes`` in place.
    """
    if latex_str is None:
        latex_str = rf"${param_name}$"
    day1_est = 0
    for (s, m), (bs, ds, summary_df, _) in fit_results.items():
        ax = axes[s-1]
        color = plot_utils.mask_colors[m]
        label_index = mask_list.index(m)
        # shared error-bar + value annotation (see plot_stats.plot_param_estimate_with_ci)
        _, ci_lower, ci_upper = plot_utils.plot_param_estimate_with_ci(
            ax, label_index, summary_df, param_name, color, ylim, markersize=config.MS_PT_SMALL, capsize=config.CAPSIZE)
        if s == 1:
            # store the first session:
            day1_est = (ci_lower, ci_upper)

    # format axes:
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["A"])
    axes[0].set_ylabel(latex_str)
    for ax in axes[1:]:
        ax.set_xticks(range(len(mask_list)))
        ax.set_xticklabels(mask_list)
        ax.yaxis.set_visible(False)
        ax.set_xlim([-0.4, 2.4])

    for ax in axes:
        ax.set_ylim(ylim)
        ax.axhline(0, color="black", linestyle="--", linewidth=config.LW_HAIRLINE, zorder=config.Z_REFERENCE)
        xlim = ax.get_xlim()
        ax.fill_between(xlim, day1_est[0], day1_est[1], color="tab:gray", alpha=0.2, label="Day 1")

duration_fit_results = figure_data_dict["Wildtype two day duration fit results"]
duration_axes_list = [[FIG.add_subplot(gs1[i, j]) for j in range(n_sessions)] for i in range(len(d_params))]
for axes_list, params in zip(duration_axes_list, d_params):
    name, latex_str, ylim = params
    plot_two_day_params_with_ci(axes_list, duration_fit_results, param_name=name, ylim=ylim, latex_str=latex_str)
    for ax in axes_list:
        ax.set_xticklabels([])
# add days
# Row headings over parameter-comparison panels: these panels carry a value/CI
# annotation band above their top spine, so the heading hangs the standard TITLE_PAD
# gap above the *band* rather than above the spine.
plot_utils.add_panel_title(duration_axes_list[0][0], "Day 1", anchor=config.PARAM_ANNOTATION_Y)
for i in range(n_sessions-1):
    ax = duration_axes_list[0][i + 1]
    plot_utils.add_panel_title(ax, f"Day 2.{i + 1}", anchor=config.PARAM_ANNOTATION_Y)  # see above

te_fit_results = figure_data_dict["Wildtype two day turn error rate fit results"]
te_axes_list = [[FIG.add_subplot(gs1[i, j]) for j in range(n_sessions)] for i in np.arange(len(e_params))+len(d_params)]
for axes_list, params in zip(te_axes_list, e_params):
    name, latex_str, ylim = params
    plot_two_day_params_with_ci(axes_list, te_fit_results, param_name=name, ylim=ylim, latex_str=latex_str)

[ax.set_xticklabels([]) for axes_list in te_axes_list[:-1] for ax in axes_list]


plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.51, 0.99), (0.01, 0.83), (0.01, 0.67), (0.01, 0.55),
                                    (0.01, 0.41), (0.01, 0.27), (0.01, 0.17)],)

config.save_figure(FIG, "curve_fit_supp.pdf", save_path)

