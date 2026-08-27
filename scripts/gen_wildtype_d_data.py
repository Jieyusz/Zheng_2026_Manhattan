"""
Generate the wildtype Mask-D figure data (fig:maskd, fig:maskd_supp, fig:d_motif, fig:north).

Computes Day-1 Mask-D learning/behavioral metrics, the bottleneck goal-choice sequence, traverse
path-similarity, first-journey timing, and the single-north housing-condition summaries; also
stores the shared Mask-D example sessions/traverses used across the Mask-D panels.

Saved keys
----------
"Wildtype D {unit}"                          : per-animal Mask-D metric arrays (duration / turn error rate / corridor error / speed).
"Wildtype D average traverse similarity" / "Wildtype D similarity matrices" : adjusted-Jaccard path similarity.
"Wildtype D first journey timing"            : (n, 2) in-maze seconds [start->first bottleneck, last bottleneck->first reward].
"Wildtype D first traverse forward bias"     : per-animal beta_hat on the first traverse alone (tab:walker row); NaN where the session has no traverse.
"Wildtype D sortie count by direction"       : per-animal sorties/journey split by starting port.
"Wildtype Mask D goal transition array|tiles per corridor|first hour reward" : bottleneck-choice series + behavioral summaries.
"Single north {condition} {unit}|traverse {unit}" / "Single north pooled sortie count by direction" : single-north housing condition.
"Mask D example {bout steps|tile steps|bout meta|manifest}" : flat per-bout/per-tile tables for the
                                             three shared example sessions (.parquet).  The Mask-D
                                             motif panels and the example-traverse panels are row
                                             selections on these (by ``example`` / ``traverse_idx`` /
                                             ``bout_idx``), so no separate traverse cache is needed.
See docs/data_contracts.md §12.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_wildtype_d_data.py --overwrite [--seed 0]
"""
import manhattan_maze as mm
from manhattan_maze import utils
import argparse
import numpy as np
import config


def main():
    parser = argparse.ArgumentParser(description="Generate figure data")
    parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for random sampling / bootstrap; any seed gives qualitatively identical figures (R11).")
    args = parser.parse_args()
    overwrite = args.overwrite
    np.random.seed(args.seed)  # controls random sampling / bootstrap only (R11)

    ## shared paths and DataLoader configuration (see scripts/config.py)
    save_dir = config.SAVE_DIR
    data = mm.DataLoader(config.DATA_DIR, **config.DATALOADER_KWARGS)
    mdf = data.metadata
    bl6j_mdf = mdf[mdf.Genotype=="BL6J"]

    print("Processing Wildtype Mask D data...")
    # === Wildtype Mask D cohort, example sessions, and learning report ===
    # Collect Day-1 Mask-D sessions for all BL6J animals, pick the last 3 as example
    # sessions, and print how many obtained >10 rewards.
    wildtype_d_sessions = utils.get_wildtype_d_sessions(data, mdf)
    # select example sessions
    example_sessions = wildtype_d_sessions[-3:]
    # R8 replacement: flat per-bout/per-tile tables for the same three sessions. Every
    # bout is exported, so the motif window (config.MASK_D_MOTIF_TRAVERSES) and the
    # example-traverse selection stay row filters applied at plot time.
    for suffix, table in utils.get_example_session_tables(example_sessions,
                                                         cache="Mask D example").items():
        utils.save_modular_data(f"Mask D example {suffix}", table, save_dir, overwrite=overwrite)

    # Print the ones that learned Mask O
    total_rewards = [s._n_rewards for s in wildtype_d_sessions]
    print(f"Number of mice that obtained more than 10 rewards: {np.sum(np.array(total_rewards)>10)} out of {len(total_rewards)}")

    n_traverses = 50 # number of traverses to consider for Mask A plotting

    d_example_id = config.MASK_D_EXAMPLE_ID

    # === Mask D per-traverse and per-reward learning curves (all Day-1 animals) ===
    # Per-traverse metric arrays (speed/duration/corridor error/tile error), then
    # per-reward reward-interval and sortie-count arrays, plus tiles-per-corridor.
    # traverse data for all mice on Day 1
    # "corridor/tile error rate" = per-step non-progress fraction ([0,1], chance ~0.5); the plain
    # "corridor/tile error" counts are kept alongside during the rate migration (removed in cleanup).
    for unit in ["speed", "duration", "corridor error", "tile error",
                 "corridor error rate", "tile error rate"]:
        data_array = utils.extract_array([s.filter("traverse").get_bout_stats(unit=unit) for s in wildtype_d_sessions], size=n_traverses)
        utils.save_modular_data(f"Wildtype D {unit}", data_array, save_dir, overwrite=overwrite)

    # Forward bias on the SAME first traverse that gives column 0 of "Wildtype D corridor
    # error", so tab:walker's E, rho and beta_hat describe one traverse. The 7th session has
    # no traverse and drops to NaN, matching the n=6 reported in the table.
    first_traverse_beta = np.array([utils.first_traverse_forward_bias(s) for s in wildtype_d_sessions])
    utils.save_modular_data("Wildtype D first traverse forward bias", first_traverse_beta,
                            save_dir, overwrite=overwrite)
    _ftb = first_traverse_beta[~np.isnan(first_traverse_beta)]
    print(f"Mask D first-traverse beta_hat: {_ftb.mean():.3f} +/- "
          f"{_ftb.std(ddof=1)/np.sqrt(_ftb.size):.3f} (n={_ftb.size}; tab:walker)")

    # reward intervals and sortie counts for all mice
    for unit in ["reward intervals", "sortie counts"]:
        array = utils.extract_array([s.get_slice_stats(unit=unit) for s in wildtype_d_sessions], size=n_traverses)
        utils.save_modular_data(f"Wildtype D {unit}", array, save_dir, overwrite=overwrite)

    # Per-animal mean sorties per journey, split by starting port (H-H = home, O-O = out),
    # for the north_supp row-3 direction box plots. (n_animals, 2) array, columns [H-H, O-O].
    sortie_dir_array = np.array([utils.sorties_per_journey_by_direction(s) for s in wildtype_d_sessions])
    utils.save_modular_data("Wildtype D sortie count by direction", sortie_dir_array, save_dir, overwrite=overwrite)

    # === Mask D first-journey bottleneck timing (session start -> first reward) ===
    # Characterise each animal's first journey (all bouts up to the first reward,
    # sorties included) around the bottleneck (corridor 1), measured on the in-maze
    # clock (cumulative sleep-thresholded bout duration = cumsum(get_bout_stats
    # ("duration"))), so out-of-maze gaps between bouts do not count. x = in-maze
    # time from the start of the first bout to the first bottleneck encounter; y =
    # in-maze time from the last bottleneck visit (its exit) to the first reward
    # (first-traverse completion). NaN if the animal got no reward or never visited
    # the bottleneck before the first reward.
    bottleneck_corridor = 1

    def inmaze_time_at(session, target_frame, sleep_threshold=5):
        """Cumulative in-maze time (s) from session start to target_frame, summing
        per-tile min(dwell, sleep_threshold) and skipping out-of-maze gaps between
        bouts. Matches cumsum(get_bout_stats("duration")) at bout boundaries."""
        total = 0.0
        for bout in session:
            frames = bout.get_frames()  # (n_tiles, 2) absolute in/out frames
            if frames.size == 0:
                continue
            in_t, out_t = frames[:, 0].astype(float), frames[:, 1].astype(float)
            if in_t[0] >= target_frame:
                break  # chronological: this and later bouts start after the target
            mask = in_t < target_frame
            tout = np.minimum(out_t[mask], target_frame)  # clip the straddling tile
            total += np.minimum((tout - in_t[mask]) / session.FPS, sleep_threshold).sum()
        return total

    first_journey_timing = []
    for s in wildtype_d_sessions:
        ho_idx, oh_idx = s.get_traverse_indices()
        reward_bouts = sorted(ho_idx + oh_idx)  # traverse bouts; first one is the first reward
        cdf = s.concat_corridors_df()
        if not reward_bouts or cdf.empty:
            first_journey_timing.append((np.nan, np.nan))
            continue
        journey = cdf[cdf.bout_idx <= reward_bouts[0]].sort_values("in_frame")
        bn = journey[journey.corridor == bottleneck_corridor]
        if bn.empty:
            first_journey_timing.append((np.nan, np.nan))
            continue
        first_reward_time = s.get_slice_stats(unit="time to first reward")  # in-maze seconds
        x = inmaze_time_at(s, bn.in_frame.iloc[0])                          # start -> first bottleneck
        y = first_reward_time - inmaze_time_at(s, bn.out_frame.iloc[-1])    # last bottleneck exit -> reward
        first_journey_timing.append((x, y))
    utils.save_modular_data("Wildtype D first journey timing", np.array(first_journey_timing, dtype=float),
                            save_dir, overwrite=overwrite)

    #tcp
    tpc_list = [s.slice_by_traverse_idx(None, 1).get_tiles_per_corridor() for s in wildtype_d_sessions]
    utils.save_modular_data(f"Wildtype Mask D tiles per corridor", np.array(tpc_list), save_dir, overwrite=overwrite)

    # === Mask D duration exponential curve fit (bootstrap CIs) ===
    # fit traverse duration
    td_df = utils.get_traverse_data_df(wildtype_d_sessions, "duration")
    # Save the fit input; the centralised gen_curve_fits.py step produces the
    # "Wildtype D duration fit results" cache.
    utils.save_curve_fit_input("Wildtype D duration", td_df, "duration", n_traverses, save_dir, overwrite=overwrite)


    # === Mask D first-hour reward count (rewards within 3600 s of session start) ===
    # use reward df for reward count
    reward_counts = []
    for s in wildtype_d_sessions:
        rwd_array = np.cumsum(s.reward_interval_seconds)
        reward_counts.append(len(rwd_array[rwd_array<3600]))  # 3600 s = 1 hour (reward intervals in seconds) (R3)
    utils.save_modular_data(f"Wildtype Mask D first hour reward", np.array(reward_counts), save_dir, overwrite=overwrite)

    # === Mask D traverse-similarity matrices and per-animal average similarity ===
    # For each session compute the (H-O self, O-H self, cross) adjusted-Jaccard
    # similarity matrices, then the per-animal average traverse similarity used to
    # quantify path stereotypy on Mask D.
    # Mask D similarity matrix data:
    d_similarity_list = [ ]
    for k, s in enumerate(wildtype_d_sessions):
        # find the matrix and get the off diagonal values for each situation
        d_similarity_list.append(s.get_three_traverse_similarity_matrix())

    utils.save_modular_data("Wildtype D similarity matrices", d_similarity_list, save_dir, overwrite=overwrite)

    # average similarity
    avg_sims = []
    for sim_tuple in d_similarity_list:
        avg_sims.append(utils.get_average_traverse_similarity(*sim_tuple))



    utils.save_modular_data("Wildtype D average traverse similarity", np.array(avg_sims), save_dir, overwrite=overwrite)
    # === Mask D bottleneck choice: goal-transition preference by reward ===
    # (The per-corridor transition matrices for the bottleneck-transition figure are
    # generated in gen_endotaxis.py, in the outskirt-removed corridor space.)
    bottleneck_choice = utils.extract_array([s.get_slice_stats(unit="bottleneck choice") for s in wildtype_d_sessions], size=20)
    utils.save_modular_data(f"Wildtype Mask D goal transition array", bottleneck_choice, save_dir, overwrite=overwrite)

    # === Single-north supplementary: reward intervals, sorties, duration/turn error across days/masks ===
    # For the single-north housing condition, collect per-reward and per-traverse
    # metrics for Day-1 D and Day-2 D/A/C sessions for the supplementary figure.
    # Single north supplementary
    print("Plot single north experiment results")
    nicknames_north = bl6j_mdf[bl6j_mdf["Condition"] == "Single_north"].Nickname.tolist()
    day1_d_sessions = [data[nn][1] for nn in nicknames_north if "d1" in nn]
    day2_d_sessions = [data[nn][0] for nn in nicknames_north if "d2" in nn]
    day2_a_sessions = [data[nn][1] for nn in nicknames_north if "d2" in nn]
    day2_c_sessions = [data[nn][2] for nn in nicknames_north if "d2" in nn]

    # traverse duration and turn error rates for all
    n_rewards = 30
    n_traverses = 40  # per-traverse metrics run longer than the per-reward-slice metrics (north_supp turn-error panel shows up to 40 traverses; the longest Mask A North sessions reach ~46)
    for condition, sessions in zip(["Day 1 D", "Day 2 D", "Day 2 A", "Day 2 C"], [day1_d_sessions, day2_d_sessions, day2_a_sessions, day2_c_sessions]):
        for unit in ["reward intervals", "sortie counts"]:
            array = utils.extract_array([s.get_slice_stats(unit=unit) for s in sessions], size=n_rewards)
            utils.save_modular_data(f"Single north {condition} {unit}", array, save_dir, overwrite=overwrite)

        for unit in ["duration", "turn error rate"]:
            if " D" in condition and unit == "turn error rate":
                continue
            array = utils.extract_array([s.filter("traverse").get_bout_stats(unit) for s in sessions], size=n_traverses)
            utils.save_modular_data(f"Single north {condition} traverse {unit}", array, save_dir, overwrite=overwrite)

    # Mean sorties per journey split by starting port (H-H = home, O-O = cage), pooled across ALL
    # single-north sessions (Day-1 D and Day-2 D/A/C) to maximise the number of paired data points
    # for the north_supp row-3 box plots. One row per session -> (n_sessions, 2) array [H-H, O-O].
    # Note: a mouse contributes several sessions, so the pooled paired test is pseudoreplicated.
    north_all_sessions = day1_d_sessions + day2_d_sessions + day2_a_sessions + day2_c_sessions
    north_dir_array = np.array([utils.sorties_per_journey_by_direction(s) for s in north_all_sessions])
    utils.save_modular_data("Single north pooled sortie count by direction",
                            north_dir_array, save_dir, overwrite=overwrite)


if __name__ == "__main__":
    main()
