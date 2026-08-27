from manhattan_maze import plot_utils, utils
from manhattan_maze.curve_fit import exponential_func
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

import config
config.set_plot_style()  # apply manuscript matplotlib style (R6)

save_path = config.parse_save_path()
fig_width = 5
fig_height = 5.2

## data loading (eventually will use the same for all files);
figure_data_dict = utils.load_all_figure_data()

# The turn-error curves split into outbound (H->O) and homebound (O->H)
# directions, each with its own colour and separately-fit learning curve.
directions = [("outbound", "tab:red"), ("homebound", "tab:blue")]

# Turn-error curve-fit bounds (Eq. 2); used for the direct least-squares central curve.
_err_spec = next(spec for spec in config.CURVE_FIT_SPECS if spec[0] == "turn error rate")
_err_p0, _err_lb, _err_ub = _err_spec[3], _err_spec[4], _err_spec[5]


def ls_central_curve(bs, ds, x_grid):
    """Least-squares exponential fit to the observed data, evaluated on ``x_grid``.

    Used as the plotted central learning curve in place of the stored bootstrap
    median-*of-parameters* curve. Taking each parameter's bootstrap median
    independently can yield an incoherent curve when the rate parameter rails on
    a small cohort (here post-swap homebound, n=8), leaving the central line below
    the data; the direct fit tracks the population means. The bootstrap confidence
    band is still drawn around this line.

    Parameters
    ----------
    bs : np.ndarray
        Observed 1-based traverse indices used for the fit (first element of the
        stored fit-results tuple).
    ds : np.ndarray
        Observed turn-error-rate values aligned with ``bs`` (fraction in [0, 1]).
    x_grid : np.ndarray
        Traverse-index grid on which to evaluate the fitted curve (the bootstrap
        curve's x-grid, so the central line and CI band share an x-axis).

    Returns
    -------
    np.ndarray
        Fitted turn-error rate at each point of ``x_grid``.
    """
    popt, _ = curve_fit(exponential_func, bs, ds, p0=_err_p0, bounds=(_err_lb, _err_ub), maxfev=100000)
    return exponential_func(x_grid, *popt)


# Plot figure: three rows (schematic + relative-error ratio; turn-error curve fits; raster)
FIG = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
gs0 = FIG.add_gridspec(3, 1, hspace=0.05, height_ratios=[0.9, 0.9, 1])

# === Row 1: swap-maze schematic + relative-error ratio (post-first / pre-first) ===
# The ratio (per direction) contrasts each animal's post-swap first-10 turn error
# with its own pre-swap first-10; ratio < 1 => savings/transfer across the swap.
# Same raw-data method/style as plot_ac_mem_gen Panel C (plot_relative_memory).
gs00 = gs0[0].subgridspec(1, 4, width_ratios=[1.2, 0.3, 1.2, 3.0], wspace=0.05)
axes_a = [FIG.add_subplot(gs00[i]) for i in range(3)]
plot_utils.plot_schematic_swap_maze(axes_a, add_text=False)
ax_ratio = FIG.add_subplot(gs00[3])
plot_utils.plot_relative_memory(
    [ax_ratio],
    figure_data_dict["Wildtype swap turn error rate relative ratio"],
    figure_data_dict["Wildtype swap turn error rate gap data points"],
    day_gaps=[(0, 1)], format_title=False,
    colordict={d.capitalize(): color for d, color in directions}, ylabel="Relative error")

# === Row 2: pre/post-swap turn error rate, outbound and homebound fit separately ===
gs10 = gs0[1].subgridspec(1, 2, wspace=0.05)
ax_err_pre = FIG.add_subplot(gs10[0])
ax_err_post = FIG.add_subplot(gs10[1])
for condition, ax in zip(["Pre-swap", "Post-swap"], [ax_err_pre, ax_err_post]):
    dir_handles = []
    for direction, color in directions:
        bs, ds, _, bootstrap_curves = figure_data_dict[f"Wildtype {condition} turn error rate {direction} fit results"]
        x_grid, ci_lower, _central, ci_upper = bootstrap_curves
        # Central line = direct LS fit to the data (bootstrap band kept for the CI).
        central = ls_central_curve(bs, ds, x_grid)
        plot_utils.plot_fitted_curve_and_confidence(ax, x_grid, ci_lower, central, ci_upper, color=color, linewidth=config.LW_EMPHASIS)
        dir_handles.append(plot_utils.plot_direction_mean(
            ax, figure_data_dict[f"Wildtype {condition} turn error rate {direction}"], direction, color,
            markersize=config.MS_AREA_DEFAULT, plot_errorbar=True))
    plot_utils.format_xs_ys(ax, utils.to_traverse_number(np.arange(20)), xlabel="Traverse #", ylabel="Turn error rate", ylim=0.5)
    # Chance level: the approach-conditioned turn error rate has an exact 0.5 chance.
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=config.LW_HAIRLINE, zorder=config.Z_REFERENCE)
    ax.text(0.5, 1, condition, ha="center", va="bottom", transform=ax.transAxes, fontsize=plot_utils.FONT_SIZE)
    if condition == "Post-swap":  # share the pre-swap y-axis
        ax.set_yticklabels([])
        ax.set_ylabel("")
ax_err_post.legend(handles=dir_handles, labels=[d.capitalize() for d, _ in directions],
                   loc="upper right", bbox_to_anchor=(1, 1), fontsize=plot_utils.TICK_SIZE)

# === Row 3: reward raster of all O-A-A animals, aligned to the swap ===
# Each animal's two Mask-A sessions share one row on the in-maze clock: the
# pre-swap session runs in negative time (t=0 at the swap), the post-swap session
# in positive time, so the vertical line at t=0 marks the entrance/exit swap.
ax_raster = FIG.add_subplot(gs0[2])
# The pre/post sessions are exported as flat per-bout tables (R8); the manifest's
# `pair_idx` gives the shared raster row and `segment` says which side of the swap.
swap_meta = figure_data_dict["Swap example bout meta"]
swap_manifest = figure_data_dict["Swap example manifest"]
pair_indices = sorted(swap_manifest["pair_idx"].unique())
raster_cmap = plt.cm.tab10
out_scatters, home_scatters = [], []
for i, pair_idx in enumerate(pair_indices):
    color = raster_cmap(i % 10)
    pair_rows = swap_manifest[swap_manifest.pair_idx == pair_idx].set_index("segment")
    pre_meta = swap_meta[swap_meta.example == pair_rows.loc["pre", "example"]]
    post_meta = swap_meta[swap_meta.example == pair_rows.loc["post", "example"]]
    out_pre, home_pre, _ = plot_utils.plot_reward_raster(ax_raster, pre_meta, y_loc=i + 1, color=color,
                                                        markersize=config.MS_AREA_RASTER, reverse=True, plot_end=False)
    out_post, home_post, end_line = plot_utils.plot_reward_raster(ax_raster, post_meta, y_loc=i + 1, color=color,
                                                                 markersize=config.MS_AREA_RASTER, reverse=False)
    out_scatters += [out_pre, out_post]
    home_scatters += [home_pre, home_post]
swap_line = ax_raster.axvline(0, color="black", linewidth=config.LW_HAIRLINE, zorder=config.Z_MARKER,
                              label="Swap", linestyle="--")
ax_raster.set_ylim(bottom=0.3, top=len(pair_indices) + 0.3)
ax_raster.set_yticks(np.arange(len(pair_indices)) + 1)
ax_raster.set_ylabel("Mouse")
ax_raster.set_xlabel("Time since swap (s)")
ax_raster.set_xlim(-3000, 3000)
ax_raster.legend(handles=[tuple(out_scatters), tuple(home_scatters), end_line, swap_line],
                 labels=["Out reward", "Home reward", "Session End", "Swap"], handler_map={tuple: HandlerTuple(ndivide=None)},
                 loc="lower left", bbox_to_anchor=(0, 1), ncol=4, fontsize=plot_utils.TICK_SIZE)

# Format labels and save figure
plot_utils.add_letter_labels(FIG, [(0.01, 0.99), (0.48, 0.99), (0.01, 0.71), (0.01, 0.41)])
config.save_figure(FIG, "olfaction.pdf", save_path)
