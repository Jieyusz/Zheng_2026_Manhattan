"""Figure-data serialization (npy/parquet/pkl) and file lookup.

Split out of utils.py; see docs.
"""
import numpy as np
import pandas as pd
from glob import glob
import warnings
import os
import pickle

__all__ = ['find_existing_file', '_live_mm_object_classes', 'save_modular_data',
           'load_all_figure_data', 'save_curve_fit_input', 'select_by_prefix', 'CURVE_FIT_INPUT_SUFFIX',
           'R8_PICKLE_ALLOWLIST']

#: Figure-data keys allowed to embed live ``manhattan_maze`` objects, under R8's
#: "fall back to .pkl only when unavoidable" clause.  Everything else that pickles a live
#: object is a bug: it will fail to unpickle once the class moves.  Single source of truth
#: for the :func:`save_modular_data` warning.
#:
#: - ``"masks"``: ``Mask`` / ``MaskDSpecial`` are *called* as objects downstream
#:   (``mask.plot``, ``mask.tiles_shortest_distances``, ``mask.holes_list``), including by
#:   the array-based renderers, which take a ``Mask`` argument.  Their geometry is
#:   reconstructible from ``data/masks/holes_*.npy`` but nothing consumes a spec form.
R8_PICKLE_ALLOWLIST = frozenset({"masks"})


def select_by_prefix(data_dict, prefixes, suffix, sep=" "):
    """
    Select a subset of a figure-data dict keyed by a shared ``"<prefix> <suffix>"`` scheme.

    Builds ``{prefix: data_dict[f"{prefix}{sep}{suffix}"]}`` for each prefix — e.g. to
    gather the per-genotype arrays that the comparison plots consume::

        select_by_prefix(figure_data_dict, ["Acortical", "Control", "Wildtype"],
                         "Mask D goal transition array")

    Parameters
    ----------
    data_dict : dict
        Figure-data dict (e.g. from :func:`load_all_figure_data`).
    prefixes : iterable of str
        Label prefixes (e.g. genotypes); become the keys of the returned dict, in order.
    suffix : str
        Shared key suffix following each prefix.
    sep : str, default " "
        Separator between prefix and suffix.

    Returns
    -------
    dict
        ``{prefix: data_dict[f"{prefix}{sep}{suffix}"]}`` preserving ``prefixes`` order.
    """
    return {prefix: data_dict[f"{prefix}{sep}{suffix}"] for prefix in prefixes}

# Suffix marking a saved curve-fit input payload. The centralised fitting step
# (scripts/gen_curve_fits.py) discovers every "<base> fit input" file, refits it,
# and writes the corresponding "<base> fit results". Keeping the convention in
# one place lets the producer scripts and the central fitter agree without a
# hand-maintained registry.
CURVE_FIT_INPUT_SUFFIX = " fit input"

def find_existing_file(filename):
    """
    Verify that a glob pattern matches an existing file and return its path.

    Parameters
    ----------
    filename : str
        Glob pattern for the target file path.

    Returns
    -------
    str
        Resolved path of the first matching file.

    Raises
    ------
    FileNotFoundError
        If no file matching ``filename`` is found.
    """
    file = glob(filename)
    if len(file) == 0:
        raise FileNotFoundError(f"File {filename} not found. Generate it first!")
    else:
        print(f"File {filename} already exists, using it...")
    return file[0]


def _live_mm_object_classes(obj, _depth=0):
    """
    Return ``"module.qualname"`` for live manhattan_maze objects nested in ``obj``.

    Detects package objects (e.g. ``Session`` / ``Bout`` / ``Mask``) embedded in
    a figure-cache payload so :func:`save_modular_data` can warn when it must fall
    back to pickle (R8).  Detection is by the ``type(obj).__module__`` string so
    that ``utils`` keeps no internal manhattan_maze imports.  Recurses into lists,
    tuples, sets, and dicts up to a bounded depth, and into the ``__dict__`` of any
    package object found, so the report names every class the pickle would reference --
    not just the outermost one.

    Parameters
    ----------
    obj : any
        Object (possibly nested) about to be pickled.
    _depth : int
        Internal recursion-depth guard.

    Returns
    -------
    set of str
        Fully-qualified class names of any embedded manhattan_maze objects;
        empty if none are found.

    Notes
    -----
    Walking the attributes of a matched object matters because the back-references are
    what make these caches fragile *and* enormous: a ``Bout`` holds ``session``, which
    holds ``trajectory``, which holds every other session and the full mask library.
    Stopping at the first match reported only ``Bout`` and hid that whole chain.
    """
    found = set()
    if _depth > 6:
        return found
    module = getattr(type(obj), "__module__", "") or ""
    if module.startswith("manhattan_maze"):
        found.add(f"{module}.{type(obj).__qualname__}")
        # keep walking: the reachable back-references are the actual hazard (see Notes)
        for value_obj in vars(obj).values() if hasattr(obj, "__dict__") else ():
            found |= _live_mm_object_classes(value_obj, _depth + 1)
        return found
    if isinstance(obj, dict):
        for key_obj, value_obj in obj.items():
            found |= _live_mm_object_classes(key_obj, _depth + 1)
            found |= _live_mm_object_classes(value_obj, _depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            found |= _live_mm_object_classes(item, _depth + 1)
    return found


def save_modular_data(key, data, output_dir="../data/figure_data/", overwrite=False):
    """
    Save data to disk, choosing format by type: .npy, .parquet, or .pkl.

    Parameters
    ----------
    key : str
        Filename stem (without extension).
    data : pd.DataFrame, np.ndarray, or any
        Data to save.  DataFrames → Parquet; arrays → .npy; everything else
        → pickle.
    output_dir : str, default '../data/figure_data/'
        Directory to write the file; created if absent.
    overwrite : bool, default False
        If False, skip writing when the file already exists.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Determine best format and serialize
    if isinstance(data, pd.DataFrame):
        file_path = os.path.join(output_dir, f"{key}.parquet")
        # Use a temporary buffer to check for changes
        current_bytes = data.to_parquet()
    elif isinstance(data, np.ndarray):
        file_path = os.path.join(output_dir, f"{key}.npy")
        from io import BytesIO
        buf = BytesIO()
        np.save(buf, data)
        current_bytes = buf.getvalue()
    else:
        # Fallback for complex nested objects (lists of sessions, dicts of stats).
        # R8: arrays/DataFrames are handled above; only payloads that cannot be
        # expressed as those reach here. Warn (naming the embedded class) if the
        # payload contains live manhattan_maze objects, because such caches break
        # when class paths change and must be regenerated after a refactor.
        file_path = os.path.join(output_dir, f"{key}.pkl")
        # Keys on R8_PICKLE_ALLOWLIST are deliberate object caches, so warning on them
        # every batch run is noise that trains readers to ignore the warning.
        live_classes = set() if key in R8_PICKLE_ALLOWLIST else _live_mm_object_classes(data)
        if live_classes:
            warnings.warn(
                f"Figure cache '{key}' falls back to pickle and embeds live "
                f"manhattan_maze objects ({', '.join(sorted(live_classes))}); it will "
                f"not load if those classes move and must be regenerated (R8).",
                stacklevel=2,
            )
        current_bytes = pickle.dumps(data)

    if os.path.exists(file_path) and not overwrite:
        print("File already exists, skipping")
        return
    else:
        with open(file_path, "wb") as f:
            f.write(current_bytes)
        print(f"Saved {key} to {file_path}")


def save_curve_fit_input(base_key, data_df, data_type, x_grid_max,
                         output_dir="../data/figure_data/", overwrite=False):
    """
    Persist the inputs needed to (re)run a bootstrap curve fit, decoupled from fitting.

    Producer scripts call this next to where they build the tidy fit frame, instead
    of fitting inline. The centralised :mod:`scripts.gen_curve_fits` step later loads
    every ``"<base_key> fit input"`` payload and produces ``"<base_key> fit results"``
    with a single shared bootstrap seed — so fits can be re-run without re-loading
    heavy Session objects, and all CIs share one reproducible RNG.

    The payload carries its own fit metadata (``data_type`` selects the parameter
    spec; ``x_grid_max`` sets the bootstrap-curve x-range), so the central fitter
    needs no per-fit configuration table.

    Parameters
    ----------
    base_key : str
        Fit identifier WITHOUT a suffix, e.g. ``"Acortical A duration"``. The input
        is saved as ``"<base_key> fit input"`` and the central step writes
        ``"<base_key> fit results"``.
    data_df : pd.DataFrame
        Tidy fit frame with columns ``Animal``, ``b``, ``Value`` (as consumed by
        :func:`manhattan_maze.bootstrap.fit_traverse_data_df_with_bootstrap`).
    data_type : str
        Metric name selecting the parameter spec (must match a ``config.CURVE_FIT_SPECS``
        entry), e.g. ``"duration"`` or ``"turn error rate"``.
    x_grid_max : float
        Upper bound of the bootstrap-curve x-grid (``np.linspace(1, x_grid_max, 100)``).
    output_dir : str, default '../data/figure_data/'
        Directory to write the file.
    overwrite : bool, default False
        Forwarded to :func:`save_modular_data`.
    """
    payload = {
        "data_df": data_df,
        "data_type": data_type,
        "x_grid_max": x_grid_max,
    }
    save_modular_data(f"{base_key}{CURVE_FIT_INPUT_SUFFIX}", payload,
                      output_dir=output_dir, overwrite=overwrite)


def load_all_figure_data(data_dir="../data/figure_data/"):
    """
    Load all figure-data files from a directory into a dict keyed by filename stem.

    Parameters
    ----------
    data_dir : str, default '../data/figure_data/'
        Directory containing .parquet, .npy, and .pkl files saved by
        :func:`save_modular_data`.

    Returns
    -------
    dict of {str: any}
        Keys are filename stems; values are the loaded objects
        (DataFrame, ndarray, or arbitrary pickle).
    """
    figure_data_dict = {}
    loaded_from = {}

    # Find all modular files
    files = glob(os.path.join(data_dir, "*"))

    for file_path in sorted(files):
        file_name = os.path.basename(file_path)
        key, ext = os.path.splitext(file_name)

        if ext not in (".parquet", ".npy", ".pkl"):
            continue

        # Keys are filename stems, so the same key written in two formats (e.g. a stale
        # ".pkl" left beside its ".npy" replacement) would resolve by glob order and
        # silently serve whichever won. Fail loudly instead: a stale cache shadowing a
        # migrated one is the quiet way for a figure to be built from the wrong data.
        if key in figure_data_dict:
            raise ValueError(
                f"figure-data key {key!r} exists in two formats: "
                f"{os.path.basename(loaded_from[key])} and {file_name}. Delete the stale "
                "one (formats changed during the R8 migration; caches are build artifacts)."
            )
        loaded_from[key] = file_path

        if ext == ".parquet":
            figure_data_dict[key] = pd.read_parquet(file_path)
        elif ext == ".npy":
            figure_data_dict[key] = np.load(file_path, allow_pickle=True)
        elif ext == ".pkl":
            with open(file_path, "rb") as f:
                figure_data_dict[key] = pickle.load(f)

    return figure_data_dict
