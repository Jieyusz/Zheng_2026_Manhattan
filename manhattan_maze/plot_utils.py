"""Plotting constants (font sizes, colour maps) + facade re-export of the plot_* modules.

Drawing functions were split into plot_style/plot_schematics/plot_curves/plot_behavior/
plot_stats and are re-exported below so `plot_utils.X` keeps working.

`plot_utils.X` is the intended import surface — keep calling through it rather
than importing the submodules directly. Shared plotting constants and the
generic primitives `plot_jittered_scatter` / `plot_box` / `format_value_str`
live in this module; the drawing functions live in these submodules:

    plot_style       Matplotlib style, drawing primitives, axis/legend/
                     significance/colour formatting.
    plot_curves      Learning-curve, array-data, and fitted-curve plots.
    plot_schematics  Maze / graph / model schematic figures, their
                     annotations, and experiment timelines.
    plot_behavior    Rasters, corridor-discovery, Markov-step, and memory
                     plots.
    plot_stats       Group / memory / gap comparison plots and statistical-
                     result annotations.
"""

import numpy as np

# Font sizes, line-width tiers, marker-size tiers and cap sizes are defined in
# plot_constants (import-light, no matplotlib) and re-exported here so the documented
# `plot_utils.X` surface keeps working.
from manhattan_maze.plot_constants import *  # noqa: F401,F403
from manhattan_maze.plot_constants import (  # explicit for the names used in this module
    LW_DATA, MS_AREA_SMALL,
)

mask_colors = {'O': 'tab:gray', 'A': 'tab:blue', 'B': 'tab:red', 'C': 'tab:green', 'D': 'tab:purple',
               'E': 'tab:pink', 'F':'tab:olive'}

genotype_colors = {"Control":"tab:blue", "Acortical":"tab:orange", "Wildtype":"tab:purple"}

ob_condition_color_dict={"Ablated":"tab:red", "Recovered":"tab:red", "Sham":"tab:blue", "Rest":"tab:orange"}

bout_type_color_dict = {"H-H": "tab:brown", "O-O": "tab:pink", "H-O": "tab:olive", "O-H": "tab:cyan"}


def plot_jittered_scatter(ax, x, ys, color=None, label=None, jitter=0.15, scatter_alpha=0.8,
                          markersize=MS_AREA_SMALL, marker="o", open_marker=False, **scatter_kwargs):
    """
    Plot a jittered scatter plot with the given x and ys values. jitter the x values for the scatter.

    ``marker`` sets the glyph (e.g. ``"^"``/``"v"`` for triangles); ``open_marker`` draws it
    hollow (no face, colored edge) so filled/open glyphs can distinguish two categories that
    share a single ``color``. Defaults reproduce the original filled marker.

    ``markersize`` is forwarded to ``ax.scatter(s=...)``, i.e. an AREA in pt^2 -- pass an
    ``MS_AREA_*`` tier, never an ``MS_PT_*`` one.
    """
    if color is None:
        color = "black"

    jittered_x = x + np.random.uniform(-jitter, jitter, size=len(ys))
    face_color, edge_color = ("none", color) if open_marker else (color, "none")
    scatter = ax.scatter(jittered_x, ys, label=label, zorder=5, s=markersize, alpha=scatter_alpha,
                         marker=marker, facecolors=face_color, edgecolors=edge_color, **scatter_kwargs)
    return scatter


def plot_box(ax, x, ys, color=None, label=None, box_width=0.2, face_color="white",
             linewidth=LW_DATA, plot_outlier=False, **box_kwargs):
    """
    Plot a box plot with the given x and ys values.
    """
    if color is None:
        color = "black"

    # remove nans
    ys = np.array(ys)
    ys = ys[~np.isnan(ys)]

    box = ax.boxplot(ys, positions=[x], widths=box_width, patch_artist=True, boxprops=dict(facecolor=face_color, color=color, linewidth=linewidth),
                     medianprops=dict(color=color, linewidth=linewidth), whiskerprops=dict(color=color), capprops=dict(color=color),
                        showfliers=plot_outlier, zorder=0,
                     )
    return box


# --- facade: re-export the split plot modules so existing `plot_utils.X` calls keep working ---
from manhattan_maze.plot_style import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.plot_curves import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.plot_schematics import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.plot_behavior import *  # noqa: F401,F403  (facade re-export)
from manhattan_maze.plot_stats import *  # noqa: F401,F403  (facade re-export)
