"""
Centralised path, DataLoader, and curve-fitting configuration for all scripts.

Usage
-----
    import config
    config.set_plot_style()                      # manuscript matplotlib style
    data = mm.DataLoader(config.DATA_DIR, **config.DATALOADER_KWARGS)
    utils.save_modular_data(key, value, output_dir=config.SAVE_DIR)
    plt.savefig(config.FIGURES_DIR / "my_figure.pdf")
    for data_type, params_name, params_latex, p0, lb, ub in config.CURVE_FIT_SPECS:
        ...
"""
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent

MAZE_SIZE = 11  # grid dimension of one floor; all tile/corridor indices scale with this

# Canonical genotype order for the acortical-vs-control(-vs-wildtype) comparison plots.
# Used with utils.select_by_prefix to gather per-genotype arrays from figure_data_dict;
# slice (e.g. GENOTYPES[:2]) for two-group panels.
GENOTYPES = ["Acortical", "Control", "Wildtype"]

# Index into figure_data_dict["Mask A example sessions"] selecting the example
# animal shown in the first-mask figure. Single source of truth shared by
# plot_first_mask.py and the supplementary-video driver so both use the same one.
MASK_A_EXAMPLE_ID = 0
# Number of bouts in the Mask A example segment shown in Panel E of
# plot_first_mask.py and rendered by Supplementary Video 2. Single source of
# truth shared by the generator (gen_wildtype_two_day_data.py) and the
# supplementary-video driver so Panel E, the raster box, and the video all
# bracket the same window.
MASK_A_SEGMENT_BOUTS = 29
# Traverse indices (0-based, into the example session's traverses) drawn as the
# example-traverse strip in plot_first_mask.py and plot_d_full.py: the first
# outbound/homebound pair and a late pair. Single source of truth shared by the
# generators (which also build the overnight-retention comparison from the same
# indices) and the plot scripts, which now select these rows out of the exported
# "<base> example bout meta" table by traverse_idx.
EXAMPLE_TRAVERSE_INDICES = [0, 1, 20, 21]
# Index into figure_data_dict["Acortical A example sessions"]; shared by
# plot_ac_rapid.py and the supplementary-video driver.
ACORTICAL_A_EXAMPLE_ID = 0
# First/last bout of the acortical example segment shown in plot_ac_rapid.py Panel E
# (and Supplementary Video 5), as a (start, end) bout-index range in Session.slice
# terms. Previously a bare slice(91, 188) duplicated between the plot script and the
# video driver.
ACORTICAL_A_SEGMENT_BOUTS = (91, 188)
# Which control example session (index into control_sessions) supplies the
# traverse reference drawn over the acortical example traverses in plot_ac_rapid.py.
# Its per-mouse colour is looked up so the reference matches that mouse everywhere
# else (raster / curves).
CONTROL_TRAVERSE_REF_ID = 1

# Index into figure_data_dict["Mask D example sessions"] (the last 3 wildtype
# Day-1 sessions, ordered [W1_a1, T3_a1, T2_a1]) selecting the example animal for
# every Mask D panel. 1 = T3_a1. Single source of truth shared by plot_d_motif.py
# (motif strip), plot_d_supp.py (speed panel), gen_endotaxis.py + gen_wildtype_d_data.py
# (example corridor/traverse data), and the supplementary-video driver, so all Mask D
# example panels show the same animal.
MASK_D_EXAMPLE_ID = 1
# The six consecutive (late, most-learned) traverse indices shown in the Mask D motif
# strip (plot_d_motif.py Panel and Supplementary Video 4) -- T3_a1's last six.
MASK_D_MOTIF_TRAVERSES = [96, 97, 98, 99, 100, 101]
# Which animal's three route-similarity matrices are drawn as the example triplet.
MASK_D_SIMILARITY_EXAMPLE_ID = 6
ACORTICAL_D_SIMILARITY_EXAMPLE_ID = 2
SIMILARITY_EXAMPLE_MIN_SIDE =10

# Acortical mask-E example: the mouse/session whose early-vs-late traverses illustrate
# the generalization concept, and which traverse indices to show. Selected in
# gen_ac_generalization.py ("Acortical E traverse examples" / "Acortical first E example
# session"); kept here so the exemplar mouse lives in one place.
ACORTICAL_E_EXAMPLE_MOUSE = "076_a1"
ACORTICAL_E_EXAMPLE_TRAVERSES = [0, 1, 10, 11]

DATA_DIR  = _REPO_ROOT / "data"
SAVE_DIR  = DATA_DIR / "figure_data"
FIGURES_DIR = _REPO_ROOT / "figures" / "pdf"
# Local, gitignored export target for the paper/Overleaf figure copies written by
# update_overleaf_plots.py. Kept in-repo under figures/ (portable, derived from __file__)
# rather than a machine-specific external path (R5); regenerated, so gitignored with figures/.
PAPER_FIGURES_DIR = _REPO_ROOT / "figures" / "paper_figures"

# DataLoader keyword arguments shared by every gen_*.py script.
#
# The published run applied manual fixes from a file that is not distributed; None disables them.
DATALOADER_KWARGS = dict(
    metadata_filename="manhattan_metadata_published.csv",
    manual_fixes_path=None,
    manual_fixes=None,
    force_reprocess=False,
)

# Curve-fitting specifications for the two learning-curve models (Eqs. 1 and 2).
# Each entry: (data_type, params_name, params_latex, p0, lower_bounds, upper_bounds).
# delta (δ): duration learning rate; epsilon (ε): turn-error learning rate
# (manuscript notation; see docs/notation_guide.md). params_latex holds
# each parameter's manuscript symbol, aligned with params_name, for figure labels.
# Previously stored in manhattan_maze.utils.curve_fit_tuples.
CURVE_FIT_SPECS = [
    ("duration",        ["D_infty", "D_0", "delta"],   [r"$D_\infty$", r"$D_0$", r"$\delta$"],   [20,  200, 0.1], [2,     5,   0.01], [60,  800, 1]),
    ("turn error rate", ["E_infty", "E_0", "epsilon"], [r"$E_\infty$", r"$E_0$", r"$\epsilon$"], [0.1, 0.5, 0.1], [0.001, 0.1, 0.01], [0.5, 1,   1]),
]

# Parameter name -> manuscript LaTeX label, derived from CURVE_FIT_SPECS so the
# symbols live in one place. Used for figure tick labels (e.g. plot_ci_ratios).
PARAM_LATEX = {
    name: latex
    for _data_type, names, latexes, *_bounds in CURVE_FIT_SPECS
    for name, latex in zip(names, latexes)
}
# Data-anchored late-performance labels: the two-day curve-derived ratio panel
# reports the fitted curve at a late in-range traverse (D_late/E_late) instead of
# the unidentifiable t→∞ asymptote (D_infty/E_infty). See compute_two_day_ratio_table.
PARAM_LATEX["D_late"] = r"$D_{\mathrm{late}}$"
PARAM_LATEX["E_late"] = r"$E_{\mathrm{late}}$"

# Jaccard traverse-similarity measures (Mask D motif analysis): plain data/stats
# label -> manuscript LaTeX label. O = outbound, H = homebound, H' = reversed
# homebound. Ordered O,O / H,H / O,H' to match the similarity-matrix columns.
SIMILARITY_LATEX = {
    "J(O, O)":  r"$J_{\mathrm{O, O}}$",
    "J(H, H)":  r"$J_{\mathrm{H, H}}$",
    "J(O, H')": r"$J_{\mathrm{O, H'}}$",
}

# Traverse-direction axis labels (similarity-matrix axes): plain bout-type string ->
# manuscript LaTeX. Same H-O / O-H / O-H' convention, just set in serif math; the
# minus is wrapped as {-} so mathtext keeps the tight hyphen (no operator spacing).
# The plain keys match the canonical bout-type strings used elsewhere as data labels.
TRAVERSE_LATEX = {
    "H-O":  r"$\mathrm{H{-}O}$",
    "O-H":  r"$\mathrm{O{-}H}$",
    "O-H'": r"$\mathrm{O{-}H'}$",
}

# Endotaxis model configuration (used by gen_endotaxis.py).
# Learning parameters passed to endotaxis.Learn_Mouse_tr, in order:
#   (gain, threshold, alpha [learning rate], decay).
ENDOTAXIS_LEARNING_PARAMETERS = (0.21, 0.2, 0.2, 0)
# Goal / port corridor indices for the endotaxis demonstrations.
ENDOTAXIS_MASK_A_GOAL_CORRIDOR = 9   # linear-track (Mask A) goal corridor index
ENDOTAXIS_MASK_D_HOME_CORRIDOR = 4   # Mask D home corridor in the reduced 9x9 frame (outskirts removed)
ENDOTAXIS_MASK_D_OUT_CORRIDOR = 13   # Mask D out/goal corridor in the reduced 9x9 frame


def __getattr__(name):
    """
    Expose the figure style constants as ``config.LW_DATA``, ``config.FONT_SIZE``, etc.

    Forwards any unknown attribute to :mod:`manhattan_maze.plot_constants`, which is the
    single definition site for font sizes and the line-width / marker-size tiers (also
    re-exported as ``plot_utils.X``). Resolved lazily via PEP 562 rather than imported at
    module level: importing anything from the ``manhattan_maze`` package executes its
    ``__init__``, which pulls in matplotlib -- and keeping ``import config`` free of
    matplotlib for path-only callers is deliberate (see :func:`set_plot_style`).
    """
    from manhattan_maze import plot_constants
    try:
        return getattr(plot_constants, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def set_plot_style():
    """
    Apply the manuscript-wide matplotlib style (serif/Computer-Modern, 300 dpi).

    Single entry point for figure scripts: call once at the top of any plot_*.py
    before plotting. Delegates to :func:`manhattan_maze.plot_utils.set_style`,
    which holds the actual rcParams (kept in the package so the style lives with
    the plotting code). Imported lazily so that using ``config`` only for paths
    or DataLoader kwargs does not pull in matplotlib.
    """
    from manhattan_maze import plot_utils
    plot_utils.set_style()


def parse_save_path(default=None):
    """
    Parse the ``-s/--save_path`` CLI argument shared by every figure script.

    Centralises the argparse boilerplate that was duplicated verbatim across all
    ``scripts/plot_*.py`` so the output directory lives in one place (R5).

    Parameters
    ----------
    default : str or pathlib.Path, optional
        Directory used when ``-s`` is omitted. Defaults to :data:`FIGURES_DIR`
        (the manuscript ``figures/pdf`` directory).

    Returns
    -------
    str
        Directory in which the figure PDF will be written.
    """
    import argparse
    if default is None:
        default = FIGURES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path", type=str, default=str(default),
                        help="Directory to save the figure PDF")
    return parser.parse_args().save_path


def save_figure(fig, filename, save_path=None):
    """
    Write a manuscript figure PDF with the standard, figure-wide settings (R5).

    Every figure is saved with ``transparent=True``, ``dpi=300`` and
    ``bbox_inches="tight"`` so the export settings match across the manuscript.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to write.
    filename : str
        PDF basename (e.g. ``"first_mask.pdf"``); joined onto ``save_path``.
    save_path : str or pathlib.Path, optional
        Output directory; defaults to :data:`FIGURES_DIR`.
    """
    if save_path is None:
        save_path = FIGURES_DIR
    fig.savefig(Path(save_path) / filename, transparent=True, dpi=300, bbox_inches="tight")


def bottleneck_transition_colors():
    """
    Return ``(goal_color, control_colors)`` for the Mask-D bottleneck-transition plots.

    The goal (bottleneck) color and the three control-node colors, sampled from the
    ``PiYG`` colormap. Shared by the transition figures (``plot_d_full``,
    ``plot_d_supp``, ``plot_algorithm``) so the palette lives in one place; the
    ``add_biclique_arrows`` helper carries the same values as its own fallback
    defaults. Matplotlib is imported lazily so that importing ``config`` only for
    paths stays matplotlib-free.
    """
    import matplotlib.pyplot as plt
    goal_color = plt.cm.PiYG(0.1)
    control_colors = [plt.cm.PiYG(0.99), plt.cm.PiYG(0.9), plt.cm.PiYG(0.8), plt.cm.PiYG(0.7)]
    return goal_color, control_colors


def bottleneck_transition_specs():
    """
    Return the Mask-D bottleneck corridor-transition specs (raw corridor indices).

    Each spec is one ``start`` corridor contrasted against the bottleneck ``goal`` and its
    ``controls`` arms, in **raw** Mask-D corridor indices. Single source of truth for the
    biclique/bottleneck transition set: ``plot_d_full`` draws both specs (schematic +
    choice-ratio panels) and ``plot_algorithm`` reuses the outbound spec for the endotaxis
    biclique-fan arrows and its goal-signal selection. Map a raw index to its display
    position with ``list(mask_d.MaskDSpec().plot_corridor_order).index(raw)`` (the
    reduced display order shared by the transition matrices and ``node_position``). Kept as
    pure raw-index data so ``config`` stays matplotlib/mask-free (as with
    :func:`bottleneck_transition_colors`).
    """
    return [
        {"label": "Outbound",  "start": 19, "goal": 1, "controls": [5, 3, 7, 9], "outbound": True},
        {"label": "Homebound", "start": 12, "goal": 1, "controls": [2, 4, 6, 8], "outbound": False},
    ]
