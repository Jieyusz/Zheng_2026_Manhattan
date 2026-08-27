"""
Generate the acortical over-days memory figure data (fig:ac_mem_gen, fig:ac_mem_supp).

Computes the across-day retention metrics for acortical mice on Masks A/D/O and the
example-animal memory traces shown in the memory panels.

Saved keys
----------
"{gt} Mask A {metric} relative ratio" / "{gt} Mask A {metric} gap data points" : across-day retention ratios + per-animal points.
"Acortical Mask A example memory duration|speed|turn error rate|days" : example-animal Mask-A memory traces.
"Acortical mem traverse example {bout steps|tile steps|bout meta|manifest}" : flat per-bout/per-tile tables
                                                                       for the first-traverse examples
                                                                       (.parquet); `label` is the *day*.
"Acortical Mask D example memory bottleneck choice|intervals|days"    : example-animal Mask-D memory traces.
"Acortical Mask O memory intervals|speeds"                            : Mask-O baseline memory summaries.
See docs/data_contracts.md §12.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_ac_mem.py --overwrite [--seed 0]
"""
import manhattan_maze as mm
from manhattan_maze import utils
from manhattan_maze import mask_d
import argparse
import pandas as pd
import numpy as np
import config


def main():
    parser = argparse.ArgumentParser(description="Generate figure data")
    parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the relative-performance bootstrap CIs; any seed gives "
                             "qualitatively identical figures (R11). Without it the CIs move every run.")
    args = parser.parse_args()
    overwrite = args.overwrite

    ## shared paths and DataLoader configuration (see scripts/config.py)
    save_dir = config.SAVE_DIR
    data = mm.DataLoader(config.DATA_DIR, **config.DATALOADER_KWARGS)
    mdf = data.metadata
    acortical_mdf = mdf[mdf.Genotype=="HO"]
    control_mdf = mdf[mdf.Genotype=="WT"]
    maskd_special_params = mask_d.MaskDSpec()
    n_rewards = 20
    n_first_traverses = 10
    bl6j_mdf = mdf[mdf.Genotype=="BL6J"]

    # === Example acortical animal: Mask A memory across repeated days (animal 683) ===
    # Per-session traverse duration and turn-error arrays across the repeated Mask-A
    # days for the example memory-retention panel, with the day index of each session.
    # Example animal for memory of Mask A:
    a_memory_sessions, a_days = utils.get_animal_repeated_mask_sessions(data, acortical_mdf, animal="683", mask_name="A")
    # get duration array:
    a_memory_durations = [s.filter("traverse").get_bout_stats("duration") for s in a_memory_sessions]
    a_memory_turns = [s.filter("traverse").get_bout_stats("turn error rate") for s in a_memory_sessions]
    a_memory_speed = [s.filter("traverse").get_bout_stats("speed") for s in a_memory_sessions]
    a_memory_duration_array = utils.extract_array(a_memory_durations, size=n_rewards)
    a_memory_turn_array = utils.extract_array(a_memory_turns, size=n_rewards)
    a_memory_speed_array = utils.extract_array(a_memory_speed, size=n_rewards)
    utils.save_modular_data("Acortical Mask A example memory duration", a_memory_duration_array,  save_dir, overwrite=overwrite)
    utils.save_modular_data("Acortical Mask A example memory turn error rate", a_memory_turn_array, save_dir, overwrite=overwrite)
    utils.save_modular_data("Acortical Mask A example memory days", a_days, save_dir, overwrite=overwrite)
    utils.save_modular_data("Acortical Mask A example memory speed", a_memory_speed_array, save_dir, overwrite=overwrite)

    # Get trends for first hour reward over days.
    control_animals = control_mdf.Animal.unique()
    acortical_animals = acortical_mdf.Animal.unique()

    # === Example first-traverse trajectories across day ranges (memory illustration) ===
    # Pick the first traverse of a couple of sessions in each day-gap range to show
    # how the retained path looks immediately, after ~weeks, and after ~months.
    # generate traverse examples for visualization
    a_first_traverses = []
    a_day_ranges = [(1, 2), (7, 28), (28, 100)]
    for day_range in a_day_ranges:
        # find the first index of a_days in this range
        indices = np.where((a_days >= day_range[0]) & (a_days < day_range[1]))[0]
        # get the corresponding session and the first traverse
        # pick the first two:
        for idx in indices[:2]:
            session = a_memory_sessions[idx]
            first_traverse = session.filter("traverse")[0]
            a_first_traverses.append((a_days[idx], first_traverse))

    # add day 1:
    memory_example_traverses = [(a_days[0], a_memory_sessions[0].filter("traverse")[0])]+a_first_traverses
    # R8 replacement: the same traverses as flat per-bout/per-tile tables. Here the meta
    # table's `label` column holds the *day* each traverse came from, not a traverse index
    # -- which is why the column is generically named.
    for suffix, table in utils.get_example_bout_tables(
            memory_example_traverses, cache="Acortical mem traverse example").items():
        utils.save_modular_data(f"Acortical mem traverse example {suffix}", table, save_dir,
                                overwrite=overwrite)

    # === Cohorts for the Mask-A memory-vs-gap analysis (acortical / control / wildtype) ===
    # Read the learning-count table (written by gen_acortical_learning.py) and build
    # the animal lists whose repeated Mask-A sessions are compared by inter-session gap.
    # Group A duration gaps for plotting groups
    count_df = pd.read_csv(f"{save_dir}/Acortical_learning_count_df.csv")
    acortical_a_repeated = count_df[count_df.Mask=="A"].Long_term_animal_list.values[0].split(", ")
    t1_df = utils.create_t1_df(bl6j_mdf)
    wildtype_a_nicknames = t1_df[t1_df[0]=="A"].Nickname.tolist()
    wildtype_a_animals = [string.split("_")[0] for string in wildtype_a_nicknames]

    # get the duration array stored:
    # Normalize the traverse duration based on the average of the first five

    # Helper: build a tidy per-bout dataframe (Animal, Day, Gap, Bout, Value) for one
    # animal's repeated Mask-A sessions, used as input to the gap-ratio bootstrap.
    def get_metric_df(animal_df, animal, metric, n_rewards=n_first_traverses):
        animal_sessions = utils.get_session_list_from_df(data, animal_df)
        # use the average of the first five as baseline:
        metric_tuple_list = []
        animal_df["Day"] = animal_df.Day - animal_df.Day.iloc[0] # reset day count to start from day 1
        for k, row in animal_df.iterrows():
            s = animal_sessions[k]
            day = row.Day
            gap = row.Gap
            sub_data_list = s.filter("traverse").get_bout_stats(metric)[:n_rewards]
            for bout, value in enumerate(sub_data_list):
                metric_tuple_list.append((animal, int(day), int(gap), bout+1, value))
        metric_df = pd.DataFrame(metric_tuple_list, columns=["Animal", "Day", "Gap", "Bout", "Value"])
        return metric_df


    day_gaps = [(1, 2), (2, 8), (8, 61)]

    # === Build per-animal metric dataframe cache (duration, turn error) ===
    # Precompute the tidy per-bout dataframes once per animal (with day gaps) so the
    # bootstrap below can reuse them across genotype/gap combinations.
    # Build a cache of animal metric dataframes to avoid recomputation
    print("Building metric dataframe cache...")
    metric_df_cache = {}
    for animal in list(acortical_a_repeated) + list(control_animals) + list(wildtype_a_animals):
        animal_df = utils.get_animal_df(mdf, animal)
        animal_df = animal_df[animal_df.Mask=="A"].reset_index(drop=True)
        # special treatment for 154_a1 because 154 didn't learn Mask A on the first day
        # remove first row if animal is 154
        if animal == "154":
            animal_df.drop(index=0, inplace=True)
            animal_df = animal_df.reset_index(drop=True)
        if animal_df.empty:
            continue
        animal_df["Gap"] = animal_df["Day"].diff().fillna(0)
        metric_df_cache[animal] = {}
        for metric in ["duration", "turn error rate", "speed"]:
            metric_df_cache[animal][metric] = get_metric_df(animal_df, animal, metric)

    # === Memory-vs-gap relative ratio (bootstrap CIs) per metric and genotype ===
    # For each genotype and inter-session gap range, bootstrap the ratio of the
    # post-gap session mean to the same animal's baseline (day-0) mean, and also
    # save per-session ratios; these drive the memory-retention-by-gap panels.
    # Process bootstrap for each metric and genotype combination
    for metric in ["duration", "turn error rate", "speed"]:
        for gt, animal_list in zip(["Acortical", "Control", "Wildtype"],
                                   [acortical_a_repeated, control_animals, wildtype_a_animals]):
            ratio_list = []
            metric_list = []

            # Build combined metric dataframe for this genotype
            for animal in animal_list:
                if animal in metric_df_cache and metric in metric_df_cache[animal]:
                    animal_metric_df = metric_df_cache[animal][metric].copy()
                    animal_metric_df["Genotype"] = gt
                    metric_list.append(animal_metric_df)

            if not metric_list:
                continue

            metric_full_df = pd.concat(metric_list, ignore_index=True)
            baseline_df = metric_full_df[metric_full_df.Day==0]

            # Bootstrap for each gap range
            gap_data_list = []
            for gap in day_gaps:
                sub_df = metric_full_df[(metric_full_df.Gap>=gap[0]) & (metric_full_df.Gap<gap[1])]
                if sub_df.empty:
                    continue
                ratio_data_df = pd.concat([baseline_df, sub_df], ignore_index=True)

                # Perform bootstrap (estimator lives in manhattan_maze.bootstrap, re-exported via utils)
                observed_ratio, (low, high) = utils.relative_performance(ratio_data_df, n_iterations=1000,
                                                                         seed=args.seed)
                ratio_list.append((gap, observed_ratio, low, high))

                # Calculate raw per-session data: for each session, divide its mean by the animal's baseline mean
                session_means = ratio_data_df.groupby(['Animal', 'Day', 'Gap']).agg({
                    'Value': 'mean',
                    'Genotype': 'first'
                }).reset_index().rename(columns={'Value': 'SessionMean'})

                # Get baseline mean for each animal
                baseline_by_animal = ratio_data_df[ratio_data_df.Day == 0].groupby('Animal').agg({
                    'Value': 'mean'
                }).reset_index().rename(columns={'Value': 'BaselineMean'})

                # Merge and calculate per-session ratios
                session_means = session_means.merge(baseline_by_animal, on='Animal', how='left')
                session_means['SessionRatio'] = session_means['SessionMean'] / session_means['BaselineMean']

                # Save aggregated session data for this gap
                gap_data_list.append((gap, session_means[['Animal', 'Day', 'SessionMean', 'BaselineMean', 'SessionRatio', 'Genotype']]))
            # Save both the ratio results and the individual data points
            utils.save_modular_data(f"{gt} Mask A {metric} relative ratio", ratio_list, save_dir, overwrite=overwrite)
            utils.save_modular_data(f"{gt} Mask A {metric} gap data points", gap_data_list, save_dir, overwrite=overwrite)



    # === Supplementary: Mask O overnight memory (reward intervals by gap) ===
    # For each gap range, take the post-gap Mask-O sessions and save their reward-
    # interval arrays for the Mask-O memory supplementary panel.
    # Supplementary: Overnight memory of Mask O
    day_gaps = [(1, 2), (2, 8), (8, 60)]
    o_memory_intervals = []
    o_memory_speeds = []
    for gap in day_gaps:
        _, post_session_df = utils.get_session_df_by_gap_size(acortical_mdf, "O", gap_size=gap)
        # print(f"Acortical sessions: {len(acortical_df)}")
        session_list = utils.get_session_list_from_df(data, post_session_df)
        int_list = []
        speed_list = []
        for session in session_list:
            int_list.append(session.reward_interval_seconds)
            speed_list.append(session.filter("traverse").get_bout_stats("speed")[:n_rewards])
            # count first hour rewards
        array = utils.extract_array(int_list, size=n_rewards) # only take the first 10 rewards
        speed_array = utils.extract_array(speed_list, size=n_rewards)
        o_memory_intervals.append((gap, array))
        o_memory_speeds.append((gap, speed_array))
    utils.save_modular_data("Acortical Mask O memory intervals", o_memory_intervals, save_dir, overwrite=overwrite)
    utils.save_modular_data("Acortical Mask O memory speeds", o_memory_speeds, save_dir, overwrite=overwrite)

    # === Example acortical animal: Mask D memory (intervals & bottleneck choice, animal 073) ===
    # Across the repeated Mask-D days (dropping a glitched Day-8 session), save per-
    # reward interval and bottleneck-choice arrays plus the day indices for the
    # Mask-D example memory panel.
    # example for mask D
    d_memory_sessions, d_days = utils.get_animal_repeated_mask_sessions(data, acortical_mdf, animal="073", mask_name="D",
                                                                        self_day_reference=True)
    # remove the Day 8 session, list index 2 (which is a glitched session with video recording)
    d_memory_sessions.pop(2)
    d_days = np.delete(d_days, 2)
    # save first traverses
    d_memory_intervals = [s.get_slice_stats("reward intervals") for s in d_memory_sessions]
    d_memory_interval_array = utils.extract_array(d_memory_intervals, size=n_rewards)
    utils.save_modular_data("Acortical Mask D example memory intervals", d_memory_interval_array, save_dir, overwrite=overwrite)
    d_memory_bottleneck = [s.get_slice_stats("bottleneck choice") for s in d_memory_sessions]
    d_memory_bottleneck_array = utils.extract_array(d_memory_bottleneck, size=n_rewards)
    utils.save_modular_data("Acortical Mask D example memory bottleneck choice", d_memory_bottleneck_array, save_dir, overwrite=overwrite)
    utils.save_modular_data("Acortical Mask D example memory days", d_days, save_dir, overwrite=overwrite)

if __name__ == "__main__":
    main()
