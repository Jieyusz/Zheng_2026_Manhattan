"""
Generate the wildtype Mask-A and Day-2 figure data (fig:first_mask, fig:day2, fig:error_prop).

Computes the Day-1 Mask-A learning curves and error metrics, the Day-2 cross-mask (A/B/C)
comparison, the first-journey forward bias and per-hole error diagnostics, and stores the shared
Mask-A example sessions/segment/traverses. Bootstrap curve fitting is deferred to
``gen_curve_fits.py`` (this writes the tidy ``"<base> tidy"`` / fit-input payloads).

Saved keys
----------
"Wildtype A traverse duration|turn error rate|traverse speed"    : per-animal Mask-A learning-metric arrays (n_animals, n_traverses).
"Wildtype A Corridor error array|Corridor error rate array|tile error array|tile error rate array" : per-traverse error metrics (count + per-step rate).
"Wildtype A first journey forward bias"                          : per-animal pre-reward first-journey directional-persistence readout.
"Wildtype A first traverse forward bias"                         : per-animal beta_hat on the first traverse alone (tab:walker row; NOT the journey readout above).
"Wildtype A hole by hole error rate"                             : per-hole (close->far) turn-error-rate arrays.
"Wildtype A reward intervals|sortie counts|sortie count by direction" : Mask-A behavioral summaries.
"Day 2 {metric}" / "Overnight traverse comparison"              : Day-2 cross-mask (A/B/C) metrics + Day1-late vs Day2-early overnight comparison.
"Wildtype two day {data_type} tidy|fit results|bootstrap params" : tidy per-traverse frames + curve-fit outputs (fit results/params from gen_curve_fits).
"Wildtype two day first journey forward bias"                     : per-(session, mask) first-journey beta_hat curves (.parquet, tidy; Session 1 = Day 1, 2-5 = Day2-1..4).
"Mask A example {bout steps|tile steps|bout meta|manifest}"       : flat per-bout/per-tile tables for the
                                                                   three shared Mask-A example sessions
                                                                   (.parquet); the example segment and
                                                                   example traverses are row selections
                                                                   on these.
"Overnight traverse example {bout steps|tile steps|bout meta|manifest}" : same for the overnight-retention
                                                                   traverses, keyed by
                                                                   direction/set_idx/animal_idx.
"masks"                                                          : mask geometry (.pkl; allowlisted under R8 --
                                                                   Mask objects must be called as objects).
"Wildtype Mask A|O ... first hour reward|tiles per corridor|reward intervals" : first-hour + baseline summaries.
See docs/data_contracts.md §12.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_wildtype_two_day_data.py --overwrite [--seed 0] [--bootstrap-seed 0]
"""
import manhattan_maze as mm
import numpy as np
import pandas as pd
from manhattan_maze import utils
from scipy.optimize import brentq
import argparse
import config

# parameters for Mask A data selection
a_roundtrips= 15 # minimum number of roundtrips to be included in figure 2, default 15
n_traverses = 50 # number of traverses to consider for Mask A plotting
n_examples = 3 # number of examples to plot for visualization
a_sidx = 1  # all Mask A session


def main():
    parser = argparse.ArgumentParser(description="Generate figure data for wildtype two day comparison")
    parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=2,
                        help="RNG seed for example-session selection; any seed gives qualitatively identical figures (R11).")
    parser.add_argument("--bootstrap-seed", type=int, default=0,
                        help="RNG seed for the shared-resample curve-fit bootstrap (paired param draws); "
                             "fixed so the Day-2/Day-1 and cross-mask ratio CIs are reproducible run-to-run.")
    args = parser.parse_args()
    overwrite = args.overwrite
    np.random.seed(args.seed)  # controls random example-session selection only (R11)

    ## shared paths and DataLoader configuration (see scripts/config.py)
    save_dir = config.SAVE_DIR
    data = mm.DataLoader(config.DATA_DIR, **config.DATALOADER_KWARGS)
    mdf = data.metadata
    bl6j_mdf = mdf[mdf.Genotype=="BL6J"]
    # === Cohort demographics report (age range/mean, sex counts) — printed, not saved ===
    # Print all mice age at the first day of experiments
    first_days = mdf[mdf.Nickname.str.contains("a1")]["Age"].tolist()
    print("range of age at first day of experiments: ", (min(first_days), max(first_days)))
    print("mean age at first day of experiments: ", f"{np.mean(first_days):.2f} days")
    # sex of the animals
    print("Number of male mice:" f"{len(mdf[mdf.Sex=="M"].Animal.unique())}")
    print("Number of female mice:" f"{len(mdf[mdf.Sex=="F"].Animal.unique())}")

    # === Day-1 Mask A cohort, example sessions, and example traverses ===
    # Select the "O, A" Day-1 BL6J animals, randomly pick example sessions (RNG fixed
    # by --seed), and save the masks, an example session, a config.MASK_A_SEGMENT_BOUTS
    # (29)-bout segment, and selected example traverses for the Mask-A trajectory panels.
    # load mask A data
    print("Loading Mask A data...")
    nicknames_maskA = mdf[(mdf["Config_label_list"]=="O, A")&(mdf["Nickname"].str.contains("a1"))&(mdf["Genotype"]=="BL6J")].Nickname.tolist()
    # Keep only animals that also completed the two-day (t1) protocol, so the Day-1
    # Mask A learning-curve cohort (Fig 1) matches the two-day cohort (Fig 2). One BL6J
    # mouse ran Day-1 Mask A but has no t1 recording (injured after Day 1 and euthanised
    # before Day 2); excluding it keeps n consistent across the two figures (26 -> 25).
    t1_animals = set(mdf[mdf["Nickname"].str.contains("t1")].Animal)
    nicknames_maskA = [nn for nn in nicknames_maskA if nn.split("_")[0] in t1_animals]
    # select three examples
    f2_example_indices = np.random.choice(np.arange(len(nicknames_maskA)), n_examples, replace=False)
    nicknames_maskA_examples = [nicknames_maskA[i] for i in f2_example_indices]
    animal_examples = [name.split("_")[0] for name in nicknames_maskA_examples]
    print(f"Selected mice: {animal_examples}")

    example_id = config.MASK_A_EXAMPLE_ID
    # R8: allowlisted pickle — Mask/MaskDSpecial must be *called* as objects by the
    # renderers (mask.plot, mask.tiles_shortest_distances, ...), so this one legitimately
    # stays a pickle under R8's fallback clause. See io.R8_PICKLE_ALLOWLIST.
    utils.save_modular_data("masks", data.masks, save_dir, overwrite=overwrite)
    mask_a_example_sessions = [data[nn][a_sidx] for nn in nicknames_maskA_examples]
    # R8 replacement: flat per-bout/per-tile tables for the same three sessions. All bouts
    # are exported, so the segment (bouts < config.MASK_A_SEGMENT_BOUTS) and the example
    # traverses stay row selections made at plot time rather than separate caches.
    for suffix, table in utils.get_example_session_tables(mask_a_example_sessions,
                                                         cache="Mask A example").items():
        utils.save_modular_data(f"Mask A example {suffix}", table, save_dir, overwrite=overwrite)

    # === Day-1 Mask A per-traverse learning curves (duration, turn error, speed, errors) ===
    # Build (n_animals x n_traverses) NaN-padded arrays of per-traverse metrics over
    # all Day-1 Mask-A animals; these drive the main Mask-A learning-curve panels.
    # traverses in Mask A
    traverse_indices = config.EXAMPLE_TRAVERSE_INDICES # indices of traverses to plot on panel A
    day1_traverses = [data[nickname][a_sidx].filter("traverse") for nickname in nicknames_maskA]
    duration_array = utils.extract_array([session.get_bout_stats(unit="duration", sleep_threshold=5) for session in day1_traverses],
                                         size=n_traverses) # duration of traverses (in seconds) with sleep threshold of 5 seconds
    utils.save_modular_data("Wildtype A traverse duration", duration_array, save_dir, overwrite=overwrite)
    error_array = utils.extract_array([session.get_bout_stats(unit="turn error rate") for session in day1_traverses],
                                      size=n_traverses) # turn error rate (first-decision per hole, approach-conditioned; chance level 0.5)
    utils.save_modular_data("Wildtype A turn error rate", error_array, save_dir, overwrite=overwrite)
    corridor_error_array = utils.extract_array([session.get_bout_stats(unit="corridor error") for session in day1_traverses],
                                               size=n_traverses) # corridor error (number of errors per traverse)
    utils.save_modular_data("Wildtype A Corridor error array", corridor_error_array, save_dir, overwrite=overwrite)
    # Forward bias on the SAME first traverse that gives column 0 above, so tab:walker's
    # E, rho and beta_hat describe one traverse. Not the first-*journey* readout below,
    # which spans the pre-reward sorties and carries the latent-learning claim instead.
    first_traverse_beta = np.array([utils.first_traverse_forward_bias(data[nn][a_sidx])
                                    for nn in nicknames_maskA])
    utils.save_modular_data("Wildtype A first traverse forward bias", first_traverse_beta,
                            save_dir, overwrite=overwrite)
    _ftb = first_traverse_beta[~np.isnan(first_traverse_beta)]
    print(f"Mask A first-traverse beta_hat: {_ftb.mean():.3f} +/- "
          f"{_ftb.std(ddof=1)/np.sqrt(_ftb.size):.3f} (n={_ftb.size}; tab:walker)")
    tile_error_array = utils.extract_array([session.get_bout_stats(unit="tile error") for session in day1_traverses],
                                           size=n_traverses) # tile error (number of erroneous tiles per traverse)
    utils.save_modular_data("Wildtype A tile error array", tile_error_array, save_dir, overwrite=overwrite)
    # Per-step error *rates* (fraction of steps that fail to progress toward the goal; chance ~0.5,
    # same >=0 definition as the error_propagation corridor rate). These bounded [0,1] curves let
    # corridor and tile error share one axis with a 0.5 chance line (panel H), replacing the counts.
    corridor_error_rate_array = utils.extract_array([session.get_bout_stats(unit="corridor error rate") for session in day1_traverses],
                                                    size=n_traverses)
    utils.save_modular_data("Wildtype A Corridor error rate array", corridor_error_rate_array, save_dir, overwrite=overwrite)
    tile_error_rate_array = utils.extract_array([session.get_bout_stats(unit="tile error rate") for session in day1_traverses],
                                                size=n_traverses)
    utils.save_modular_data("Wildtype A tile error rate array", tile_error_rate_array, save_dir, overwrite=overwrite)
    traverse_speed = utils.extract_array([session.get_bout_stats(unit="speed") for session in day1_traverses], size=n_traverses) # speed (in tiles/s)
    utils.save_modular_data("Wildtype A traverse speed", traverse_speed, save_dir, overwrite=overwrite)

    # === Day-1 Mask A per-reward metrics: sorties, reward intervals, first-hour count, tiles/corridor ===
    # Per-reward sortie counts and reward intervals, the count of rewards within the
    # first hour (3600 s), and tiles-per-corridor on the first journey.
    # slicing the reward intervals
    day1_sessions = [data[nickname][a_sidx] for nickname in nicknames_maskA]
    sortie_array = utils.extract_array([s.get_slice_stats(unit="sortie counts") for s in day1_sessions], size=n_traverses)
    utils.save_modular_data("Wildtype A sortie counts", sortie_array, save_dir, overwrite=overwrite)

    # Per-animal mean sorties per journey, split by starting port (H-H = home, O-O = out),
    # for the north_supp row-3 direction box plots. (n_animals, 2) array, columns [H-H, O-O].
    sortie_dir_array = np.array([utils.sorties_per_journey_by_direction(s) for s in day1_sessions])
    utils.save_modular_data("Wildtype A sortie count by direction", sortie_dir_array, save_dir, overwrite=overwrite)

    reward_array = utils.extract_array([s.get_slice_stats(unit="reward intervals") for s in day1_sessions], size=n_traverses)
    utils.save_modular_data("Wildtype A reward intervals", reward_array, save_dir, overwrite=overwrite)
    # count rewards in the first hour and save
    reward_times = [np.cumsum(s.reward_interval_seconds) for s in day1_sessions] # cumulative reward times in seconds
    first_hour_rewards = [len(rt[rt<3600]) for rt in reward_times]  # 3600 s = 1 hour (R3)
    utils.save_modular_data("Wildtype Mask A first hour reward", np.array(first_hour_rewards), save_dir, overwrite=overwrite)

    # tpc for the first 4 rewards:
    tpc_list = [s.slice_by_traverse_idx(None, 1).get_tiles_per_corridor() for s in day1_sessions]
    utils.save_modular_data(f"Wildtype Mask A tiles per corridor", np.array(tpc_list), save_dir, overwrite=overwrite)

    # === Day-1 Mask A first-journey empirical forward bias (beta_hat) curve ===
    # Reversal-based, degree-corrected forward bias along each animal's merged first journey
    # (pre-reward outbound sorties + first traverse), evaluated on an 18-point grid over [0, 1] of
    # the journey. Each point pools the decisions within +/-10% of it and fits beta_hat ONCE: the
    # 20% window is the only smoothing, with no moving average on top of it. Only fully-supported
    # positions are kept (mode="valid", matching utils.moving_average and the smoothed lines of
    # fig:ac_mem_gen A), so 14 of 18 points are reported, x=0.118..0.882; the dropped ends rested
    # on half-width windows. beta_hat inverts the model reversal probability
    # p_rev(beta,d)=(1-b)/((1-b)+(d-1)b) (eq:betahat); on the P10 linear track (all interior deg=2)
    # this reduces to the exactly unbiased 1 - reversal_rate at any decision count.
    beta_curve = utils.first_journey_forward_bias_curve(day1_sessions)
    utils.save_modular_data("Wildtype A first journey forward bias", beta_curve, save_dir, overwrite=overwrite)
    _beta_ok = beta_curve[1][~np.isnan(beta_curve[1])]
    print(f"Mask A first-journey beta_hat: {_beta_ok[0]:.3f} -> {_beta_ok[-1]:.3f} "
          f"({_beta_ok.size}/{beta_curve.shape[1]} points populated, cohort n={len(day1_sessions)})")

    # === Day-1 Mask A hole-by-hole error rate (per hole, per traverse, by direction) ===
    # For each bout direction (H-O, O-H), build one (n_animals x n_traverses) array
    # per hole of the per-hole error rate, for the hole-resolved learning panel.
    # hole by hole learning
    hole_data_dict = {"H-O": [], "O-H": []} # save data for each traverse
    for bout_type in hole_data_dict.keys():
        correctness_array_list = [t.filter(bout_type).get_bout_stats("error rate by hole") for t in day1_traverses]
        # get the hole data dict (organize the data list based on the hole idex)
        # remove empty list
        correctness_array_list = [arr for arr in correctness_array_list if len(arr) > 0]
        n_holes = correctness_array_list[0].shape[0] # number of holes
        for k in range(n_holes):
            hole_data_list = [arr[k, :] for arr in correctness_array_list]
            hole_data_array = utils.extract_array(hole_data_list, size=int(n_traverses/2))
            hole_data_dict[bout_type].append(hole_data_array)
    utils.save_modular_data("Wildtype A hole by hole error rate", hole_data_dict, save_dir, overwrite=overwrite)

    # === Mask O supplementary: reward intervals, first-hour count, tiles/corridor ===
    # Day-1 Mask-O sessions (session 0 of the same animals): per-session reward
    # intervals, first-hour reward count (3600 s), and tiles-per-corridor.
    # Supplementary Figure O data
    print("Processing reward intervals for Mask O...")
    # oneliner
    o_sessions = [data[nn][0] for nn in nicknames_maskA]
    o_intervals = [session.reward_interval_seconds for session in o_sessions]
    int_array = utils.extract_array(o_intervals, size=n_traverses)
    utils.save_modular_data("Wildtype O reward intervals", int_array, save_dir, overwrite=overwrite)
    # also count rewards in the first hour and save
    o_reward_times = np.cumsum(int_array, axis=1) # cumulative reward times in seconds
    o_first_hour_rewards = np.sum(o_reward_times <= 3600, axis=1) # count rewards within the first hour (3600 s; intervals in seconds) (R3)
    utils.save_modular_data("Wildtype O first hour reward", np.array(o_first_hour_rewards), save_dir, overwrite=overwrite)
    # get the tiles per corridors in the first 4 rewards
    tpc_list = [s.slice_by_traverse_idx(None, 1).get_tiles_per_corridor() for s in o_sessions]
    utils.save_modular_data(f"Wildtype Mask O tiles per corridor", np.array(tpc_list), save_dir, overwrite=overwrite)

    # === Overnight (Day 1 vs Day 2-1) traverse comparison example trajectories ===
    # Pick example Day-2-1 Mask-A sessions and their matching Day-1 sessions, and
    # save paired outbound/homebound example traverses (first, 10th, and Day-2-1
    # first) for the overnight-retention comparison panel.
    # Day 2 figure data
    t1_df = utils.create_t1_df(bl6j_mdf)
    day21_a_sessions = utils.select_t1_sessions(data, t1_df, session_idx=0, mask_name="A")
    # only select the number of examples
    # randomly select 3 sessions of day 21
    day21_example_indices = np.random.choice(np.arange(len(day21_a_sessions)), n_examples, replace=False)
    day21_example_sessions = [day21_a_sessions[i] for i in day21_example_indices]
    day1_a_sessions = [data[f"{s.name.split('_')[0]}_a1"][a_sidx] for s in day21_example_sessions] # find the day 1 sessions

    outbound_traverses = [(traverse_indices[0], [s.filter("traverse")[traverse_indices[0]] for s in day1_a_sessions]), # first traverse of Day 1
                          (traverse_indices[2], [s.filter("traverse")[traverse_indices[2]] for s in day1_a_sessions]), # 10th traverse
                          (traverse_indices[0], [s.filter("traverse")[traverse_indices[0]] for s in day21_example_sessions])] # first traverse of Day 2-1
    homebound_traverses = [(traverse_indices[1], [s.filter("traverse")[traverse_indices[1]] for s in day1_a_sessions]), # second traverse of Day 1
                            (traverse_indices[3], [s.filter("traverse")[traverse_indices[3]] for s in day1_a_sessions]), # 21st traverse
                            (traverse_indices[1], [s.filter("traverse")[traverse_indices[1]] for s in day21_example_sessions])] # second traverse of Day 2-1
    # R8 replacement: the same bouts as flat tables. The nested
    # [direction][set][animal] structure is flattened, with `example` the position in
    # that flattened order and the three index columns recorded alongside so plot_day2
    # can pick the same element without live objects.
    overnight_index, overnight_bouts = [], []
    for direction, traverse_sets in (("outbound", outbound_traverses),
                                     ("homebound", homebound_traverses)):
        for set_idx, (traverse_label, bouts) in enumerate(traverse_sets):
            for animal_idx, bout in enumerate(bouts):
                overnight_index.append({"example": len(overnight_bouts), "direction": direction,
                                        "set_idx": set_idx, "animal_idx": animal_idx})
                overnight_bouts.append((traverse_label, bout))
    overnight_index = pd.DataFrame(overnight_index)
    for suffix, table in utils.get_example_bout_tables(
            overnight_bouts, cache="Overnight traverse example").items():
        # only the one-row-per-bout tables carry the index columns; the per-step tables
        # stay narrow and are joined on `example` when needed.
        if suffix in ("bout meta", "manifest"):
            table = table.merge(overnight_index, on="example", how="left")
        utils.save_modular_data(f"Overnight traverse example {suffix}", table, save_dir,
                                overwrite=overwrite)

    print("Processing Day 2 results...")
    # === Day 2 (sessions 2-1..2-4) per-mask metrics ===
    # For each Day-2 session and metric, build per-mask (A/B/C) arrays, saved per
    # metric for the Day-2 multi-mask comparison panels (helper run only when
    # --overwrite). The per-time-point Kruskal-Mann-Whitney star layer was removed:
    # the Day-2 differences are reported via curve-fit parameter ratios/CIs (see
    # gen_curve_fits.py), so the uncorrected pairwise tests are no longer computed.
    # data for Day 2-1, 2, 3, 4 traverse duration and turn error rate
    n_roundtrips_day2 = 6 # number of roundtrips to consider for each session
    metric_specs = [("duration", "traverse"), ("turn error rate", "traverse"),
                    ("reward intervals", "reward"), ("sortie counts", "reward"), ("speed", "traverse"), ("tile error", "traverse"), ("corridor error", "traverse"),
                    # per-step non-progress rates ([0,1], chance ~0.5); counts kept alongside during migration
                    ("tile error rate", "traverse"), ("corridor error rate", "traverse"),
                    ("tiles per corridor", "")]

    def get_day2_metrics():
        for metric, slice_by in metric_specs:
            metric_list = []
            for session_idx in range(t1_df.shape[1]-1):
                mask_data_dict = {}
                for mask_idx, mask_name in enumerate(["A", "B", "C"]):
                    sessions = utils.select_t1_sessions(data, t1_df, session_idx=session_idx, mask_name=mask_name)
                    if metric == "tiles per corridor":
                        tcp_list = [s.slice_by_traverse_idx(None, 1).get_tiles_per_corridor() for s in sessions]
                        array = np.array(tcp_list)
                    elif metric == "reward intervals": # special treatment for reward intervals to convert to minutes
                        reward_time = [s.get_slice_stats(unit=metric) for s in sessions]
                        array = utils.extract_array(reward_time, size=n_roundtrips_day2 * 2)
                    elif slice_by == "reward":
                        array = utils.extract_array([s.get_slice_stats(unit=metric) for s in sessions],
                                                    size=n_roundtrips_day2 * 2)
                    else: # traverse
                        array = utils.extract_array([s.filter(slice_by).get_bout_stats(unit=metric) for s in sessions],
                                                    size=n_roundtrips_day2 * 2)

                    mask_data_dict[mask_name] = array
                # stats_res kept as None to preserve the stored (data, stats) tuple contract;
                # the pairwise Mann-Whitney star layer for these panels was removed.
                metric_list.append((mask_data_dict, None)) # store the data for each session
            utils.save_modular_data(f"Day 2 {metric}", metric_list, save_dir, overwrite=overwrite)

    if overwrite:
        get_day2_metrics()

    # === Per-(session, mask) first-journey forward bias (beta_hat) curves ===
    # The same estimator and window as the Day-1 Mask-A curve above
    # (utils.first_journey_forward_bias_curve, 20% window, mode="valid" -> 14 of 18 points),
    # applied per two-day cell: Day 1 is Mask A only, while each Day-2 session splits the
    # cohort across Masks A/B/C. Session numbering matches get_two_day_data_df and the
    # fit-result keys: 1 = Day 1, 2..5 = Day2-1..Day2-4, so the (1, "A") rows reproduce
    # "Wildtype A first journey forward bias" exactly (same cohort, same defaults).
    # NOTE on Day 2: the "first journey" is only *mechanically* the same window here
    # (pre-first-reward-of-this-session plus traverse 0). These animals already earned
    # rewards on Day 1, so this does NOT carry the latent-learning reading that the Day-1
    # curve does; it measures how forward-biased an experienced animal already is when it
    # re-enters a maze. Draws no random numbers, so it leaves the global RNG stream (and
    # hence the unseeded bootstrap fits below) untouched.
    fb_cells = {(1, "A"): day1_sessions}
    for session_idx in range(t1_df.shape[1] - 1):
        for mask_name in ["A", "B", "C"]:
            fb_cells[(session_idx + 2, mask_name)] = utils.select_t1_sessions(
                data, t1_df, session_idx=session_idx, mask_name=mask_name)
    fb_rows = []
    for (fb_session, fb_mask), fb_sessions in fb_cells.items():
        fb_grid, fb_mean, fb_se = utils.first_journey_forward_bias_curve(fb_sessions)
        fb_rows.append(pd.DataFrame({"Session": fb_session, "Mask": fb_mask,
                                     "n_animals": len(fb_sessions), "x": fb_grid,
                                     "beta": fb_mean, "se": fb_se}))
        _fb_ok = fb_mean[~np.isnan(fb_mean)]
        _fb_range = f"{_fb_ok[0]:.3f} -> {_fb_ok[-1]:.3f}" if _fb_ok.size else "all NaN"
        print(f"Session {fb_session} Mask {fb_mask} first-journey beta_hat: {_fb_range} "
              f"({_fb_ok.size}/{fb_grid.size} points populated, cohort n={len(fb_sessions)})")
    utils.save_modular_data("Wildtype two day first journey forward bias",
                            pd.concat(fb_rows, ignore_index=True), save_dir, overwrite=overwrite)

    # curve fit results:
    def _two_day_x_grid(group_key):
        """Per-(session, mask) bootstrap-curve grid: Day 1 spans n_traverses, Day 2 spans
        n_roundtrips_day2*2 (group_key is the (Session, Mask) tuple from groupby)."""
        session = group_key[0]
        x_max = n_traverses if session < 2 else n_roundtrips_day2 * 2
        return np.linspace(1, x_max, 100)

    # === Two-day exponential curve-fit results per (session, mask) for duration & turn error ===
    # Fit the learning-curve model with animal-level bootstrap separately for each
    # session/mask combination across Day 1 and Day 2, for the two-day fit panels.
    # Uses the shared grouped fitter so each group's result has the same format as
    # every other fit; the (session, mask) identity is the dict key, and the raw
    # bootstrap parameter draws are saved alongside for cross-group ratio CIs.
    for sub_tuple in config.CURVE_FIT_SPECS:
        data_type, params_name, _params_latex, p0, lower_bounds, upper_bounds = sub_tuple
        two_day_df = utils.get_two_day_data_df(t1_df, data, bl6j_mdf, data_type=data_type, traverse=True) # traverse duration
        # Persist the tidy per-traverse frame (Animal/Session/Mask/b/Value) so
        # gen_curve_fits.py can compute the model-free late-window ratio that
        # cross-checks the curve-derived X_late ratio (see docs/ratio_ci_method.md).
        utils.save_modular_data(f"Wildtype two day {data_type} tidy",
                                two_day_df[["Animal", "Session", "Mask", "b", "Value"]],
                                save_dir, overwrite=True)
        # Absolute marginal fit per (session, mask) — independent animal-level bootstrap.
        results = utils.fit_grouped_data_df_with_bootstrap(
            two_day_df, ["Session", "Mask"], _two_day_x_grid,
            p0=p0, lower_bounds=lower_bounds, upper_bounds=upper_bounds, params_name=params_name)
        utils.save_modular_data(f"Wildtype two day {data_type} fit results", results, save_dir, overwrite=True)
        # Within-subject paired draws: ONE shared animal resample per iteration across the
        # Day-1 reference and all Day-2 groups, aligned by iteration so per-iteration
        # Day-2/Day-1 ratios are within-subject (see gen_two_day_param_ratios.py).
        _, paired_params = utils.fit_grouped_data_df_with_shared_bootstrap(
            two_day_df, ["Session", "Mask"], _two_day_x_grid,
            p0=p0, lower_bounds=lower_bounds, upper_bounds=upper_bounds, params_name=params_name,
            seed=args.bootstrap_seed)
        utils.save_modular_data(f"Wildtype two day {data_type} bootstrap params", paired_params, save_dir, overwrite=True)

if __name__ == "__main__":
    main()
