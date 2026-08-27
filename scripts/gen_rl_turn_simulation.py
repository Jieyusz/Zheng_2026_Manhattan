"""
VALIDATION: trained model-free-RL turn error propagation for Mask A.

Turn/decision analogue of ``gen_rl_simulation.py`` (corridors). Like it, this confirms that
the closed-form staircase drawn in the RL column of Panel B of
``plot_error_propagation_supp.py`` (produced analytically by
``rl_model.analytic_rl_staircase`` in ``gen_error_propagation.py``) is what a real trained
agent reproduces. It is NOT a figure input, needs the (uncommitted) raw trajectories, and is
excluded from ``batch_generate_figure_data.py``; the equivalence is also pinned data-free in
``tests/test_rl_error_propagation.py``.

For each Day-1 Mask-A BL6J animal (same selection as ``gen_error_propagation.py``) and
each traverse direction (H-O, O-H), one purely model-free tabular Q-learning agent is
trained on THAT animal's journeys in chronological order, then — WITHOUT learning — its
first-turn error is read out along the shortest-path holes. Learning uses every hole
decision the animal made (``Bout.get_hole_decisions``: every crossing, turn or straight
pass); transitions are data-driven (next hole crossing, reward experienced at the goal
port). The readout marches the shortest-path holes and scores the first decision at each
hole, approach-conditioned (chance 0.5) to match the empirical method-2 turn error. The
per-animal ``(n_holes, n_journeys)`` matrices are averaged into a population-mean
``(n_holes, 25)`` matrix per direction — exactly like
``analysis.hole_error_rate_by_direction``.

See ``manhattan_maze/rl_model.py`` (turn agent) for the algorithm and its justification. The
scientific point: a model-free value learner given the mice's own experience resolves
turn errors by BACKWARD PROPAGATION from the reward (reward-adjacent holes learn their
correct turn first), unlike the roughly-parallel empirical improvement.

Saved key
---------
"Mask A model-free RL turn error by position"
    dict ``{"n_holes": int, "H-O": (n_holes, 25), "O-H": (n_holes, 25),
    "n_agents": {...}}`` — the trained-agent cohort means (holes instead of distance
    bins), holes ordered close->far from reward (index 0 = reward side); column j = the
    j-th journey. The cohort means reproduce ``rl_model.analytic_rl_staircase`` (the
    plotted prediction) -- that is the validation. Produced for inspection only; no
    figure loads it.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_rl_turn_simulation.py --overwrite [--seed 0]
"""
import argparse

import manhattan_maze as mm
from manhattan_maze import rl_model, utils
import config

A_SIDX = 1          # Mask-A session index within a Day-1 trajectory (session 0 = Mask O)
WIDTH = 25          # cohort column width (journeys) — matches the empirical/corridor keys
DIRECTIONS = ["H-O", "O-H"]
HOMEBOUND = {"H-O": False, "O-H": True}
GOAL_PORT = {"H-O": "OUT", "O-H": "HOME"}

# Q-learning hyperparameters (see rl_model turn agent). gamma < 1 gives the value gradient
# along the path that lets near-reward holes be learned before far ones (backward prop).
HYPERPARAMS = dict(gamma=0.9, alpha=0.5)


def load_mask_a_sessions():
    """Day-1 Mask-A BL6J sessions (same selection as gen_error_propagation.py)."""
    data = mm.DataLoader(config.DATA_DIR, **config.DATALOADER_KWARGS)
    mdf = data.metadata
    nicknames = mdf[(mdf["Config_label_list"] == "O, A")
                    & (mdf["Nickname"].str.contains("a1"))
                    & (mdf["Genotype"] == "BL6J")].Nickname.tolist()
    return [data[nn][A_SIDX] for nn in nicknames]


def extract_animal_journeys(session):
    """Per-direction ordered journeys; each journey = list of (decisions, end_port) per bout.

    A journey = leading sorties + terminating traverse (``Session.slice_to_journeys``);
    trailing sorties-only slices are dropped. ``decisions`` = ``Bout.get_hole_decisions()``
    (every hole crossing); ``end_port`` = the port the bout ends at (``bout_type[-1]``).
    """
    out = {d: [] for d in DIRECTIONS}
    for jr in session.slice_to_journeys():
        if not (jr.bouts and jr.bouts[-1].satisfy("traverse")):
            continue
        direction = jr.bouts[-1].bout_type
        if direction in out:
            out[direction].append([(b.get_hole_decisions(),
                                     "OUT" if b.bout_type[-1] == "O" else "HOME")
                                    for b in jr.bouts])
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate Mask-A model-free RL turn-error propagation")
    parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=0,
                        help="Accepted for parity with gen_rl_simulation.py; the turn "
                             "readout is deterministic, so results are seed-independent.")
    args = parser.parse_args()

    sessions = load_mask_a_sessions()
    if not sessions:
        raise SystemExit("No Day-1 Mask-A sessions found.")
    mask = sessions[0].mask
    holes = mask.get_holes()
    n_holes = len(holes)
    print(f"Mask A turn RL: n_animals={len(sessions)}, n_holes={n_holes}, width={WIDTH}, "
          f"hyperparams={HYPERPARAMS}")

    animal_journeys = [extract_animal_journeys(s) for s in sessions]

    rl_dict = {"n_holes": n_holes, "n_agents": {}}
    for d in DIRECTIONS:
        cmap = mask.correct_approach_map(homebound=HOMEBOUND[d])
        cohort, n = rl_model.cohort_turn_error_rate(
            [aj[d] for aj in animal_journeys], cmap, holes, GOAL_PORT[d],
            width=WIDTH, **HYPERPARAMS,
        )
        rl_dict[d] = cohort
        rl_dict["n_agents"][d] = n
        print(f"  {d}: goal={GOAL_PORT[d]} matrix={cohort.shape} n_agents={n}")

    utils.save_modular_data("Mask A model-free RL turn error by position", rl_dict,
                            config.SAVE_DIR, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
