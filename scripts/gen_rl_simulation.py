"""
VALIDATION: trained model-free-RL corridor error propagation for Mask A.

This confirms that the closed-form backward staircase drawn in the RL column of Panel A of
``plot_error_propagation_supp.py`` (produced analytically by
``rl_model.analytic_rl_staircase`` in ``gen_error_propagation.py``) is what a real trained
agent reproduces. It is NOT a figure input -- the plot loads the analytic staircase, not this
key -- and it needs the (uncommitted) raw trajectories, so it is excluded from
``batch_generate_figure_data.py`` and run manually to re-validate. The same equivalence is
pinned data-free (on the committed mask files) in ``tests/test_rl_error_propagation.py``.

For each Day-1 Mask-A BL6J animal (same selection as ``gen_error_propagation.py``) and each
traverse direction (H-O, O-H), one purely model-free tabular Q-learning agent is trained on
THAT animal's journeys in chronological order (whole-journey training, reward experienced on
the terminating traverse, per-bout so only valid adjacent-corridor transitions are learned,
value accumulating across journeys). After each journey the agent's hybrid-policy simulated
walk (greedy where learned, random-walk fallback where not) is scored with the empirical
non-decreasing distance rule into a per-corridor error rate. The per-animal
``(n_pos, n_journeys)`` matrices are averaged into a population-mean ``(n_pos, 25)`` matrix
per direction — exactly like ``analysis.cohort_position_error_rate``.

See ``manhattan_maze/rl_model.py`` for the algorithm and its justification. The scientific
point: a model-free value learner given the mice's own experience resolves errors by
BACKWARD PROPAGATION from the reward, unlike the roughly-parallel empirical improvement.

Saved key
---------
"Mask A model-free RL error by position"
    dict ``{"n_pos": int, "H-O": (n_pos, 25), "O-H": (n_pos, 25), "n_agents": {...}}`` —
    the trained-agent cohort means, per direction. Row d = distance-to-reward d (row 0 =
    reward, dropped when plotting); column j = the j-th journey/traverse. The cohort means
    converge to ``rl_model.analytic_rl_staircase`` (the plotted prediction) -- that is the
    validation. This key is produced for inspection only; no figure loads it.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_rl_simulation.py --overwrite [--seed 0]
"""
import argparse

import manhattan_maze as mm
from manhattan_maze import analysis, rl_model, utils
import config

A_SIDX = 1          # Mask-A session index within a Day-1 trajectory (session 0 = Mask O)
WIDTH = 25          # cohort column width (journeys/traverses) — matches the empirical key
N_WALKS = 300       # hybrid-policy readout rollouts per journey checkpoint
UNIT = "corridor"
DIRECTIONS = ["H-O", "O-H"]

# Q-learning hyperparameters (see rl_model). gamma < 1 gives the value gradient along the
# track that lets near-reward corridors be learned before far ones (backward propagation).
HYPERPARAMS = dict(n_walks=N_WALKS, gamma=0.9, alpha=0.5)


def load_mask_a_sessions():
    """Day-1 Mask-A BL6J sessions (same selection as gen_error_propagation.py)."""
    data = mm.DataLoader(config.DATA_DIR, **config.DATALOADER_KWARGS)
    mdf = data.metadata
    nicknames = mdf[(mdf["Config_label_list"] == "O, A")
                    & (mdf["Nickname"].str.contains("a1"))
                    & (mdf["Genotype"] == "BL6J")].Nickname.tolist()
    return [data[nn][A_SIDX] for nn in nicknames]


def extract_animal_journeys(session):
    """Per-direction ordered journeys, each a list of per-bout corridor-index sequences.

    A journey = leading sorties + terminating traverse (``Session.slice_to_journeys``);
    trailing sorties-only slices are dropped. Bouts are kept separate so the caller only
    learns within-bout (valid adjacent-corridor) transitions.
    """
    out = {d: [] for d in DIRECTIONS}
    for jr in session.slice_to_journeys():
        if not (jr.bouts and jr.bouts[-1].satisfy("traverse")):
            continue
        direction = jr.bouts[-1].bout_type
        if direction in out:
            out[direction].append([b.get_corridors() for b in jr.bouts])
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate Mask-A model-free RL error propagation")
    parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed for the readout rollouts (reproducibility). Any "
                             "seed gives qualitatively identical cohort means / staircase.")
    args = parser.parse_args()

    sessions = load_mask_a_sessions()
    if not sessions:
        raise SystemExit("No Day-1 Mask-A sessions found.")
    mask = sessions[0].mask
    adj = mask.corridors_adj_mat
    n_pos = analysis.observed_n_pos(sessions, UNIT)
    start = {"H-O": mask.home_corridor, "O-H": mask.out_corridor}
    goal = {"H-O": mask.out_corridor, "O-H": mask.home_corridor}
    print(f"Mask A RL: n_animals={len(sessions)}, n_pos={n_pos}, width={WIDTH}, "
          f"n_walks={N_WALKS}, hyperparams={HYPERPARAMS}")

    animal_journeys = [extract_animal_journeys(s) for s in sessions]

    rl_dict = {"n_pos": n_pos, "n_agents": {}}
    for d in DIRECTIONS:
        distances = mask.corridors_shortest_distance[:, goal[d]]
        cohort, n = rl_model.cohort_rl_error_rate(
            [aj[d] for aj in animal_journeys], start[d], goal[d], adj, distances, n_pos,
            width=WIDTH, seed=args.seed, **HYPERPARAMS,
        )
        rl_dict[d] = cohort
        rl_dict["n_agents"][d] = n
        print(f"  {d}: start={start[d]} goal={goal[d]} matrix={cohort.shape} n_agents={n}")

    utils.save_modular_data("Mask A model-free RL error by position", rl_dict,
                            config.SAVE_DIR, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
