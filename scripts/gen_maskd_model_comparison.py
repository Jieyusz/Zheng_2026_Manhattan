"""
Generate figure data for the Mask-D model-comparison row of the main algorithm figure (fig:algo,
Panels A-D of ``plot_algorithm.py``): model-free Q-learning vs. Endotaxis, on corridor
error and bottleneck choice, per direction (outbound H-O / homebound O-H).

Only the SIMULATED model (Q-learning self-play) is produced here. The Endotaxis
prediction is a parameter-free analytic step and is computed inline in ``plot_algorithm.py`` (its map
is learned in one pass, independent of biclique routing — no simulation needed). The Q-learning ramp,
by contrast, has no closed form (TD's max is nonlinear and the biclique splits reward propagation
across parallel arms, so the propagation speed is a Monte-Carlo statistic), hence the self-play
simulation here. See ``manhattan_maze/rl_model.py`` (self-play corridor agent) for the algorithm.

Saved key
---------
"Mask D model comparison"
    dict ``{"H-O": {...}, "O-H": {...}, "meta": {...}}``. Per direction:
      ``q_err``     (n_seeds, L)      self-play Q corridor error per traverse (mean over rollouts)
      ``q_bn``      (n_seeds, L)      self-play Q P(gateway->bottleneck) per traverse
    ``meta``: L, BN_SIZE, n_seeds, gamma, alpha, gateway, bottleneck, and e_half (the beta=0.5 null
    per direction — the shared traverse-1 anchor the plot uses to build the Endotaxis step).

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_maskd_model_comparison.py --overwrite [--seed 0]
"""
import argparse

import numpy as np

import manhattan_maze as mm
from manhattan_maze import rl_model, random_walk, utils
import config

DIRECTIONS = ["H-O", "O-H"]
L = 25                       # traverses/direction (matches the empirical Mask-D keys)
BN_SIZE = 10                 # rewards/direction shown for bottleneck choice (main figure's 20, split)
N_SEEDS = 20                 # independent self-play agents; SE band is across seeds #TODO: Change this to be the same as the number of animal =6
N_WALKS = 300                # readout rollouts per traverse checkpoint
MAX_STEPS = 5000
GAMMA, ALPHA = 0.9, 0.5      # same as Mask A; gamma<1 gives the backward value gradient
BOTTLENECK = 1               # Mask-D bottleneck corridor (manhattan_maze/mask_d.py)
GATEWAY = {"H-O": 19, "O-H": 12}   # gateway->bottleneck start node (out_/home_node_set)


def main():
    parser = argparse.ArgumentParser(description="Generate Mask-D model-comparison figure data")
    parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for the self-play agents.")
    args = parser.parse_args()

    data = mm.DataLoader(config.DATA_DIR, **config.DATALOADER_KWARGS)
    sessions = utils.get_wildtype_d_sessions(data, data.metadata)
    if not sessions:
        raise SystemExit("No Day-1 Mask-D sessions found.")
    mask = sessions[0].mask
    adj = mask.corridors_adj_mat
    start = {"H-O": mask.home_corridor, "O-H": mask.out_corridor}
    goal = {"H-O": mask.out_corridor, "O-H": mask.home_corridor}
    print(f"Mask D comparison: n_sessions={len(sessions)} home={mask.home_corridor} out={mask.out_corridor}")

    comparison = {"meta": {"L": L, "BN_SIZE": BN_SIZE, "n_seeds": N_SEEDS, "gamma": GAMMA,
                           "alpha": ALPHA, "gateway": GATEWAY, "bottleneck": BOTTLENECK, "e_half": {}}}
    for d in DIRECTIONS:
        dist = mask.corridors_shortest_distance[:, goal[d]]
        # Memoryless (beta=0.5) null as a per-step RATE: on the bipartite corridor graph the mean
        # walk length is 2*E + L (L = shortest-path length = dist[start]), so the per-step error
        # fraction is E / (2*E + L). This is the traverse-1 anchor for the Q/endotaxis rates.
        e_half_count = random_walk.expected_corridor_errors(adj, dist, start[d], goal[d], beta=0.5)
        L_path = float(dist[start[d]])
        e_half = e_half_count / (2 * e_half_count + L_path)
        q_err, q_bn = rl_model.cohort_selfplay(
            start[d], goal[d], GATEWAY[d], BOTTLENECK, adj, dist, L, N_WALKS,
            n_seeds=N_SEEDS, seed=args.seed, gamma=GAMMA, alpha=ALPHA, max_steps=MAX_STEPS,
            error_type="rate")
        comparison[d] = {"q_err": q_err, "q_bn": q_bn}
        comparison["meta"]["e_half"][d] = float(e_half)
        print(f"  [{d}] gateway={GATEWAY[d]} e_half_rate={e_half:.3f}  q_err0={q_err[:,0].mean():.3f} "
              f"q_bn[1,3,5,10]={np.round(q_bn.mean(0)[[0,2,4,9]],2)}")

    utils.save_modular_data("Mask D model comparison", comparison, config.SAVE_DIR,
                            overwrite=args.overwrite)


if __name__ == "__main__":
    main()
