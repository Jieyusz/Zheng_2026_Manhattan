"""Shared figure style constants: font sizes, line widths, marker sizes, error-bar caps.

Deliberately import-light (no numpy, no matplotlib) so that ``scripts/config.py`` can
re-export these for the figure scripts without pulling matplotlib into a module whose
whole point is to stay cheap for path-only use (see ``config.set_plot_style``, which
imports the plotting stack lazily for exactly that reason).

Re-exported by :mod:`manhattan_maze.plot_utils`, so ``plot_utils.FONT_SIZE`` and
``config.FONT_SIZE`` both resolve to the values defined here -- one definition, two
access surfaces.

Line widths and marker sizes form deliberate emphasis tiers. Use the tier name, not the
number: a bare ``linewidth=0.9`` in one panel and ``0.5`` in another is how the manuscript
drifted out of alignment in the first place.
"""

# --- Font sizes -----------------------------------------------------------------------
LABEL_SIZE = 10   # panel letters (A, B, C...)
FONT_SIZE = 8     # axis labels, titles, in-panel text
TICK_SIZE = 6     # tick labels, legends, colorbar ticks, significance text

# --- Panel headings -------------------------------------------------------------------
# Gap in POINTS between a panel heading and the top of its axes. Points, not axes fraction:
# a fraction offset scales with the panel, so the same ``y=1.15`` is a visibly different gap
# over a tall panel than over a short one -- which is how the headings drifted apart in the
# first place. Consumed by ``plot_style.add_panel_title``, the house replacement for
# ``ax.set_title``.
TITLE_PAD = 2.0

# Axes fraction at which the two-line value/CI annotation band is top-anchored above a
# parameter-comparison panel (``plot_stats.annotate_param_estimate``). A heading over such
# a panel has to hang above the *band*, not above the spine, so it passes this as
# ``add_panel_title(..., anchor=PARAM_ANNOTATION_Y)`` and still gets the standard
# TITLE_PAD gap -- measured from the band instead of from the top spine.
PARAM_ANNOTATION_Y = 1.12

# Extra points a maze-schematic heading needs on top of TITLE_PAD. The "O" port marker is
# drawn half a tile beyond the Out port, and its glyph box measures ~3 pt clear of the top
# spine, so a heading hung TITLE_PAD above the *spine* lands on the marker. Adding this
# keeps the standard gap, measured from the marker instead.
PORT_MARKER_CLEARANCE = 4.0

# --- Line widths: 4-tier emphasis hierarchy -------------------------------------------
# Also applied to spines/ticks via set_style, so the axis frame no longer out-weighs the
# data it frames (matplotlib's default axes.linewidth is 0.8).
LW_HAIRLINE = 0.5    # reference/chance lines, box + whisker, error bars, faint per-animal traces
LW_DATA = 1.0        # standard data / group-mean line
LW_EMPHASIS = 1.5    # fitted curves, shortest paths, highlighted graph edges
LW_TRAJECTORY = 3.0  # trajectory LineCollection ribbons

# --- Opacity -------------------------------------------------------------------------
# Raw per-animal points/traces that sit UNDER a group mean or a smoothed line. The value
# matches what plot_curves already used as literals for exactly this job: 0.45 for faint
# per-animal traces under a group curve, 0.5 for scatter dots under a fitted curve.
# Pair it with a lowered z-order (Z_RAW_TRACE) -- fading alone leaves the raw markers
# stacked on top of the line they are supposed to sit behind.
ALPHA_FAINT = 0.45

# --- Marker sizes, pt^2 AREA: for ``ax.scatter(s=...)`` -------------------------------
# NOTE the units. Everything routed through plot_jittered_scatter / plot_oh_scatter_line /
# plot_direction_mean / plot_array_data / plot_reward_raster lands on ``s=``, which is an
# AREA in pt^2. Do not assign an MS_PT_* value here -- it would render ~3x off.
MS_AREA_SMALL = 3      # dense per-animal dots (many points per axis)
MS_AREA_DEFAULT = 5    # standard per-animal scatter on traverse/day curves
MS_AREA_LARGE = 10     # standalone or single-per-group scatter points
# Two named semantic tiers rather than drift: the raster tier is sized to read as an event
# tick against y_increment=0.13 row spacing, and the emphasis tier marks the direction-mean
# markers in the single-panel north figure, which are intentionally oversized.
MS_AREA_RASTER = 8
MS_AREA_EMPHASIS = 25

# Backwards-compatible alias for the pre-tier constant (same value, 10).
MARKER_SIZE = MS_AREA_LARGE

# --- Marker sizes, pt DIAMETER: for Line2D / ``ax.errorbar(markersize=...)`` -----------
# A DIAMETER in points, not an area. Kept as a separate family from MS_AREA_* so the two
# can never be cross-assigned.
MS_PT_SMALL = 2
MS_PT_DEFAULT = 3
MS_PT_LARGE = 5

# --- Error-bar caps -------------------------------------------------------------------
CAPSIZE = 2       # house value for a visible cap
CAPSIZE_NONE = 0  # caps intentionally suppressed (bare error bars drawn behind a curve)

# --- Draw order -----------------------------------------------------------------------
# The ladder the figures already followed informally. Z_REFERENCE is matplotlib's default
# Line2D zorder, which is what most chance lines were already inheriting; naming it stops
# the same dashed reference sitting *under* the SE band in some panels and over it in
# others. Schematic-internal stacking (node fill < node edge < arrow) stays local to
# plot_schematics -- those numbers are relative to each other, not house tiers.
Z_BACKGROUND = -5   # zone axvspans behind everything
Z_SHADE = 0         # error bands, box plots, fills
Z_RAW_TRACE = 1     # faint per-animal traces under the group mean
Z_REFERENCE = 2     # chance / reference lines: above the shade, below the data
Z_DATA = 5          # group-mean and data lines
Z_MARKER = 10       # scatter markers, end-point symbols
Z_ANNOTATION = 15   # significance symbols and text
Z_TOP = 20          # arrows and callouts that must never be occluded
