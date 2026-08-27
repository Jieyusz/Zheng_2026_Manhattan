"""
Generate the five-way turn-error-rate comparison arrays.

Scores every Day-1 Mask-A wildtype traverse under all five turn-error-rate definitions
(see docs/notation_guide.md) plus the two denominators those definitions disagree about.
Only method 1 -- the published, approach-conditioned pooled rate -- is cached elsewhere as
``"Wildtype A turn error rate"``; this script caches all five so they can be compared
without opening a ``DataLoader``, from ``data/figure_data/`` alone.

The five methods:

1. approach-conditioned, pooled over crossings (the published ``E_{a,b}``)
2. approach-conditioned, first decision per hole
3. approach-conditioned, hole-averaged over all visits
4. raw, pooled over crossings (deprecated)
5. raw, first decision per hole (deprecated)

Saved keys
----------
"Wildtype A turn error method {m1..m5}"      : (n_animals, n_traverses) per-traverse rate arrays, NaN-padded.
"Wildtype A turn error method n_cross"       : (n_animals, n_traverses) approach-conditioned crossing counts (method-1 denominator).
"Wildtype A turn error method n_holes"       : (n_animals, n_traverses) distinct scored decision holes (method-2/3/5 denominator).
See docs/data_contracts.md §12.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_turn_error_methods.py --overwrite
"""
import argparse
import warnings

import numpy as np

import manhattan_maze as mm
from manhattan_maze import utils
import config

N_TRAVERSES = 50  # traverses to score per animal, matching the Mask-A learning-curve panels
A_SESSION_IDX = 1  # Mask-A session index within each animal's trajectory
METHOD_KEYS = ("m1", "m2", "m3", "m4", "m5", "n_cross", "n_holes")


def rate_from_seq(correctness_seq):
    """
    Pooled per-crossing error rate: errors divided by number of crossings.

    Parameters
    ----------
    correctness_seq : array_like
        Per-crossing correctness (1 correct, 0 error).

    Returns
    -------
    float
        Error rate, or NaN if the traverse scored no crossings.
    """
    n_crossings = len(correctness_seq)
    if n_crossings == 0:
        return np.nan
    return (n_crossings - np.sum(correctness_seq)) / n_crossings


def rate_from_vec(correctness_vec):
    """
    Per-hole error rate: 1 minus the mean correctness over scored holes.

    Parameters
    ----------
    correctness_vec : array_like
        Per-hole correctness with NaN for holes that were never scored.

    Returns
    -------
    float
        Error rate, or NaN if no hole was scored.
    """
    correctness_vec = np.asarray(correctness_vec, dtype=float)
    if np.all(np.isnan(correctness_vec)):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return 1.0 - np.nanmean(correctness_vec)


def bout_rates(bout):
    """
    Score one traverse under all five definitions, plus both denominators.

    Parameters
    ----------
    bout : manhattan_maze.Bout
        A traverse to score.

    Returns
    -------
    dict
        Keys ``m1``-``m5`` (rates), ``n_cross`` (approach-conditioned crossings, the
        method-1 denominator) and ``n_holes`` (distinct scored holes, the method-2/3/5
        denominator).
    """
    seq_approach = bout.get_seq_correctness(condition="approach")  # per-crossing, approach
    seq_raw = bout.get_seq_correctness(condition="raw")            # per-crossing, raw
    vec_first = bout.get_hole_correctness_vec(include="first", condition="approach")
    return {
        "m1": rate_from_seq(seq_approach),
        "m2": rate_from_vec(vec_first),
        "m3": rate_from_vec(bout.get_hole_correctness_vec(include="all", condition="approach")),
        "m4": rate_from_seq(seq_raw),
        "m5": rate_from_vec(bout.get_hole_correctness_vec(include="first", condition="raw")),
        "n_cross": len(seq_approach),
        "n_holes": int(np.sum(~np.isnan(np.asarray(vec_first, dtype=float)))),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate figure data")
    parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    overwrite = args.overwrite

    ## shared paths and DataLoader configuration (see scripts/config.py)
    save_dir = config.SAVE_DIR
    data = mm.DataLoader(config.DATA_DIR, **config.DATALOADER_KWARGS)
    mdf = data.metadata

    # === Day-1 Mask-A wildtype cohort ===
    # The same selection gen_wildtype_two_day_data.py uses for the Mask-A learning
    # curves: BL6J animals whose configuration list is "O, A", first (a1) session.
    nicknames = mdf[(mdf["Config_label_list"] == "O, A")
                    & (mdf["Nickname"].str.contains("a1"))
                    & (mdf["Genotype"] == "BL6J")].Nickname.tolist()
    print(f"{len(nicknames)} Day-1 Mask-A BL6J animals")

    # === Score every traverse under all five definitions ===
    # Per animal, a list of per-traverse values for each method; extract_array then
    # NaN-pads them to (n_animals, N_TRAVERSES) like the other learning-metric arrays.
    per_animal = {key: [] for key in METHOD_KEYS}
    for nickname in nicknames:
        traverses = data[nickname][A_SESSION_IDX].filter("traverse")
        scored = [bout_rates(bout) for bout in traverses]
        for key in METHOD_KEYS:
            per_animal[key].append([bout_scores[key] for bout_scores in scored])

    for key in METHOD_KEYS:
        array = utils.extract_array(per_animal[key], size=N_TRAVERSES)
        utils.save_modular_data(f"Wildtype A turn error method {key}", array, save_dir,
                                overwrite=overwrite)
        print(f"  {key}: {array.shape}")


if __name__ == "__main__":
    main()
