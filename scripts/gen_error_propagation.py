"""
Generate figure data for the Mask-A error-propagation supplementary figure.

Produces the position-resolved error-rate arrays consumed by
``plot_error_propagation_supp.py`` (Panels A and B), for the Day-1 Mask-A BL6J
cohort — the same selection as ``gen_wildtype_two_day_data.py`` (``"O, A"`` &
``a1`` & ``BL6J``, session index 1).

Saved keys
----------
"Wildtype A corridor error by position"
    dict ``{"n_pos": int, "H-O": (n_pos, N), "O-H": (n_pos, N), "n_animals": {...}}``
    — cohort-mean corridor error rate by distance-to-reward (row d = distance d,
    row 0 = reward, dropped when plotting), per traverse direction. Column j is the
    j-th completed traverse.
"Wildtype A hole error by position"
    dict ``{"H-O": [ (n_animals, N) per hole ], "O-H": [...] }`` — per-hole
    first-decision turn error rate (approach-conditioned, chance 0.5), holes ordered
    close->far from reward (index 0 = reward side).
Note: the model-prediction columns (the model-free-RL backward staircase and the endotaxis
synchronized step) are PARAMETER-FREE analytic functions of ``(n_positions, n_traverses)``
(:func:`rl_model.analytic_rl_staircase`, :func:`endotaxis.analytic_endotaxis_step`) and are
computed INLINE in ``plot_error_propagation_supp.py`` — this script produces only the animal
cohort keys above, keeping the analytic curves out of the figure-data cache. The trained
per-animal simulations in ``gen_rl_simulation.py`` / ``gen_rl_turn_simulation.py`` (and the
endotaxis simulation in ``tests/test_endotaxis_error_propagation.py``) VALIDATE those analytic
predictions; see ``tests/test_rl_error_propagation.py`` for the pinned RL equivalence.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_error_propagation.py --overwrite
Also picked up automatically by ``batch_generate_figure_data.py`` (phase 2).
"""
import argparse

import manhattan_maze as mm
from manhattan_maze import analysis, utils
import config

A_SIDX = 1          # Mask-A session index within a Day-1 trajectory (session 0 = Mask O)
N_TRAVERSES = 25    # traverses/direction (matches plot window; gen_wildtype int(n_traverses/2))
DIRECTIONS = ["H-O", "O-H"]
UNIT = "corridor"


def load_mask_a_sessions():
    """Return the Day-1 Mask-A BL6J sessions (same selection as gen_wildtype_two_day_data.py)."""
    data = mm.DataLoader(config.DATA_DIR, **config.DATALOADER_KWARGS)
    mdf = data.metadata
    nicknames = mdf[(mdf["Config_label_list"] == "O, A")
                    & (mdf["Nickname"].str.contains("a1"))
                    & (mdf["Genotype"] == "BL6J")].Nickname.tolist()
    return [data[nn][A_SIDX] for nn in nicknames]


def main():
    parser = argparse.ArgumentParser(description="Generate Mask-A error-propagation figure data")
    parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    overwrite = args.overwrite
    save_dir = config.SAVE_DIR

    sessions = load_mask_a_sessions()
    if not sessions:
        raise SystemExit("No Day-1 Mask-A sessions found.")
    n_pos = analysis.observed_n_pos(sessions, UNIT)
    print(f"Mask A error propagation: n_animals={len(sessions)}, n_pos={n_pos}")

    # Panel A — corridor error rate by distance-to-reward, per direction (traverse-indexed).
    corridor_dict = {"n_pos": n_pos, "n_animals": {}}
    for d in DIRECTIONS:
        cohort, n = analysis.cohort_position_error_rate(sessions, n_pos, "traverse", UNIT, d)
        corridor_dict[d] = cohort[:, :N_TRAVERSES]
        corridor_dict["n_animals"][d] = n
    utils.save_modular_data("Wildtype A corridor error by position", corridor_dict,
                            save_dir, overwrite=overwrite)

    # Panel B — per-hole first-decision turn error rate, per direction (close->far).
    hole_dict = {d: analysis.hole_error_rate_by_direction(sessions, d, size=N_TRAVERSES,
                                                          include="first")
                 for d in DIRECTIONS}
    utils.save_modular_data("Wildtype A hole error by position", hole_dict,
                            save_dir, overwrite=overwrite)

    # The model-prediction columns (model-free-RL staircase, endotaxis step) are analytic and
    # parameter-free, so they are computed INLINE in plot_error_propagation_supp.py rather than
    # cached here (see module docstring). This script produces only the animal cohort keys.


if __name__ == "__main__":
    main()
