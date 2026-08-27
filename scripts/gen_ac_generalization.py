"""
Generate the acortical/control generalization figure data (fig:ac_mem_gen, fig:ac_bc_supp, fig:ac_d_supp, fig:ac_ef_supp).

Covers generalization to new masks: the repeat-A sessions, the intermediate masks (B/C/D/E/F),
and the first-E vs after-E traverses that illustrate transfer. Reads the learning-count table
written by ``gen_count_df.py`` to select animals that actually contributed data per mask.

Saved keys
----------
"{gt} Mask {mask} traverse {metric}|reward intervals|sortie counts|tiles per corridor" : per-genotype per-mask summaries (gt in {Acortical, Control}; mask in B/C/D/E/F).
"{gt} Mask A repeat {metric}|repeat traverse {metric}|repeat tiles per corridor"       : repeat-A within-animal comparison.
"{gt} E first traverse {data_type}" / "{gt} A after E traverse {data_type}"            : first-E vs post-E transfer traverses.
"{gt} D average traverse similarity" / "{gt} D similarity matrices"                    : Mask-D path-similarity (adjusted Jaccard).
"{gt} Mask D goal transition array" / "{gt} Mask D tiles per corridor" : Mask-D per-reward bottleneck-choice series + tiles-per-corridor.
"Acortical E traverse example {bout steps|tile steps|bout meta|manifest}" : flat per-bout/per-tile tables
                                             for the Mask-E example traverses (.parquet); `label` holds the
                                             config.ACORTICAL_E_EXAMPLE_TRAVERSES index.
"Acortical Mask F traverse ..."                                 : Mask-F summaries.
See docs/data_contracts.md §12.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_ac_generalization.py --overwrite
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
    count_df = pd.read_csv(f"{save_dir}/Acortical_learning_count_df.csv")


    def drop_empty_rows(arr):
        """Drop all-NaN rows so a saved array's row count reports the EFFECTIVE n -- the animals
        that actually contributed data -- not every animal that merely entered the mask. Animals
        that completed no reward/traverse become all-NaN rows in extract_array output and are
        invisible under nanmean anyway; removing them keeps the reported n honest (e.g. the Fig S13
        Mask-F reward-interval / sortie panels: 6 sessions -> 4 with data)."""
        return arr[~np.isnan(arr).all(axis=1)]

    # === Generalization cohorts: mask-learning order for later masks (A..E) ===
    # Read the learning-count table (written by gen_acortical_learning.py) and build
    # the per-animal mask-order dataframes (Mask_order>0 = a later, non-first mask)
    # for acortical and control groups, used by the generalization analyses.
    # Learning curves of later masks
    control_names = control_mdf.Animal.unique().tolist()
    ac_gen_all_names = []
    for mask in ["A", "B", "C", "D", "E"]:
        ac_gen_all_names.extend(count_df[count_df.Mask==mask].Highly_rewarded_animal_list.values[0].split(", "))
    ac_gen_names = list(set(ac_gen_all_names))# unique
    ac_gen_df = utils.create_mask_learn_seq_df(acortical_mdf, ac_gen_names, mask_list=["A", "B", "C", "D", "E"])
    ac_gen_df = ac_gen_df[ac_gen_df.Mask_order>0]
    ct_gen_df = utils.create_mask_learn_seq_df(control_mdf, control_names, mask_list=["A", "B", "C", "D", "E"])
    ct_gen_df = ct_gen_df[ct_gen_df.Mask_order>0]

    # get the ones where these animals repeated Mask A the second time
    def get_a_repeat_sessions(data, m_df, animal_names):
        repeat_sessions = []
        for animal in animal_names:
            repeat_df = utils.get_animal_repeated_mask_df(m_df, animal, "A").reset_index(drop=True)
            if len(repeat_df) <=1:
                continue
            # only use the ones with more sessions
            sub_sessions = utils.get_session_list_from_df(data, repeat_df)
            repeat_sessions.append(sub_sessions[1]) # get the second session where A was repeated
        return repeat_sessions

    ac_repeat_sessions = get_a_repeat_sessions(data, acortical_mdf, ac_gen_names)
    ct_repeat_sessions = get_a_repeat_sessions(data, control_mdf, control_names)

        # get the metric
    for gt, sessions in zip(["Acortical", "Control"], [ac_repeat_sessions, ct_repeat_sessions]):
        for metric in ["duration", "turn error rate", "speed", "tile error", "corridor error",
                       "tile error rate", "corridor error rate"]:  # rate = per-step non-progress fraction
            metric_array = utils.extract_array([s.filter("traverse").get_bout_stats(unit=metric) for s in sessions], size=n_rewards)
            utils.save_modular_data(f"{gt} Mask A repeat traverse {metric}", metric_array, save_dir, overwrite=overwrite)
        repeat_tpc = [s.slice_by_traverse_idx(None, 1).get_tiles_per_corridor() for s in sessions]
        utils.save_modular_data(f"{gt} Mask A repeat tiles per corridor", np.array(repeat_tpc), save_dir,
                                    overwrite=overwrite)
        for metric in ["reward intervals", "sortie counts"]:
            metric_array = utils.extract_array([s.get_slice_stats(unit=metric) for s in sessions], size=n_rewards)
            utils.save_modular_data(f"{gt} Mask A repeat {metric}", metric_array, save_dir, overwrite=overwrite)

    # extract list of traverse duration
    ac_bcd_session_list = []
    ct_bcd_session_list = []
    for mask in ["B", "C", "D"]:
        session_df = ac_gen_df[ac_gen_df.Mask == mask]
        print(session_df)
        sessions = utils.get_session_list_from_df(data, session_df)
        ac_bcd_session_list.append(sessions)
        ct_sessions = utils.get_session_list_from_df(data, ct_gen_df[ct_gen_df.Mask==mask])
        ct_bcd_session_list.append(ct_sessions)

    # === Acortical Mask-D duration-fit cohort: the successful learners only ===
    # The Mask-D duration curve (Fig 5G, key "Acortical D Gen duration") should describe the
    # animals that actually LEARNED Mask D -- the highly-rewarded list from gen_count_df -- not
    # every animal that happened to complete a stray traverse. Two subtleties motivate the
    # handling below:
    #   (1) create_mask_learn_seq_df selects each animal's FIRST Mask-D session. One learner's
    #       first Mask-D session contains zero traverses (it only crossed the reward criterion on
    #       a LATER Mask-D session), so it would silently drop out of the traverse-based fit. For
    #       that animal we advance to its first Mask-D session that actually contains traverses, so
    #       the learner is represented by a real run everywhere it appears (its tiles-per-corridor
    #       "+" point in Fig 5I included).
    #   (2) A couple of NON-learners complete a stray traverse or two on their first Mask-D session;
    #       they belong in the tiles-per-corridor unsuccessful ("Acor. -") comparison but NOT in the
    #       "successful learner" duration curve.
    # We therefore (a) swap each learner's shared-cohort session to its first traverse-bearing
    # Mask-D session, and (b) build a learner-only cohort used SOLELY for the duration fit, leaving
    # every other Mask-D panel (tiles/corridor, bottleneck choice, similarity) on the full cohort.
    maskd_learners = count_df[count_df.Mask == "D"].Highly_rewarded_animal_list.values[0].split(", ")

    def first_maskd_session_with_traverses(animal):
        """First Mask-D session for `animal` containing >=1 traverse (skips an empty first
        exposure). Falls back to the first session if none has traverses -- should not happen for
        a highly-rewarded learner."""
        sessions, _days = utils.get_animal_repeated_mask_sessions(data, acortical_mdf, animal=animal, mask_name="D")
        for s in sessions:
            if len(s.filter("traverse")) > 0:
                return s
        return sessions[0]

    # (a) Represent each learner by its first traverse-bearing Mask-D session in the shared cohort.
    _learner_session = {a: first_maskd_session_with_traverses(a) for a in maskd_learners}
    ac_bcd_session_list[-1] = [_learner_session.get(s.name.split("_")[0], s) for s in ac_bcd_session_list[-1]]
    # (b) Learner-only cohort for the duration fit (all now have traverses -> n == the 5 learners).
    ac_maskd_fit_sessions = [s for s in ac_bcd_session_list[-1] if s.name.split("_")[0] in maskd_learners]

    for gt, session_list in zip(["Acortical", "Control"], [ac_bcd_session_list, ct_bcd_session_list]):
        for mask, sessions in zip(["B", "C", "D"], session_list):
            # Fig S11's Acortical Mask-D supplementary arrays (traverse duration, sorties, reward
            # intervals) must describe the SAME 5 successful learners as the Fig 5G duration fit,
            # not the 2 non-learners with a stray traverse. Use the learner-only cohort here. The
            # Mask-D tiles-per-corridor tuple and bottleneck-choice array are re-saved from the FULL
            # cohort in the separate loop below, so Fig 5I/H keep their learned-vs-unlearned split.
            if gt == "Acortical" and mask == "D":
                sessions = ac_maskd_fit_sessions
            for metric in ["duration", "turn error rate", "speed", "tile error", "corridor error",
                           "tile error rate", "corridor error rate"]:  # rate = per-step non-progress fraction
                if metric == "turn error rate" and mask == "D":
                    continue
                metric_array = utils.extract_array([s.filter("traverse").get_bout_stats(unit=metric) for s in sessions], size=n_rewards)
                utils.save_modular_data(f"{gt} Mask {mask} traverse {metric}", metric_array, save_dir, overwrite=overwrite)
            # rewward dependent array
            sortie_array = utils.extract_array([s.get_slice_stats(unit="sortie counts") for s in sessions], size=n_rewards)
            utils.save_modular_data(f"{gt} Mask {mask} sortie counts", sortie_array, save_dir, overwrite=overwrite)
            interval_array = utils.extract_array([s.get_slice_stats(unit="reward intervals") for s in sessions], size=n_rewards)
            utils.save_modular_data(f"{gt} Mask {mask} reward intervals", interval_array, save_dir, overwrite=overwrite)
            # tiles per corridor
            tpc_list = [s.slice_by_traverse_idx(None, 1).get_tiles_per_corridor() for s in sessions]
            utils.save_modular_data(f"{gt} Mask {mask} tiles per corridor", np.array(tpc_list), save_dir,
                                    overwrite=overwrite)
    # curve fit for all
    # For Acortical Mask D, fit the learner-only cohort (ac_maskd_fit_sessions) so the fitted
    # duration curve in Fig 5G / S9 describes the 5 successful learners; "A repeat"/B/C and every
    # Control fit are unchanged (ac_bcd_session_list[:-1] == [B, C]).
    ac_fit_session_list = [ac_repeat_sessions] + ac_bcd_session_list[:-1] + [ac_maskd_fit_sessions]
    for gt, session_list in zip(["Acortical", "Control"], [ac_fit_session_list, [ct_repeat_sessions]+ct_bcd_session_list]):
        for mask, sessions in zip(["A repeat", "B", "C", "D"], session_list):
            for sub_tuple in config.CURVE_FIT_SPECS:
                data_type, params_name, _params_latex, p0, lower_bounds, upper_bounds = sub_tuple
                if data_type == "turn error rate" and mask=="D":
                    continue
                td_df = utils.get_traverse_data_df(sessions, data_type)
                utils.save_curve_fit_input(f"{gt} {mask} Gen {data_type}", td_df, data_type, n_rewards, save_dir, overwrite=overwrite)

    # also fit

    print("Processing Mask D learning data...")
    # bias towards the bottleneck
    for gt, sessions in zip(["Acortical", "Control"], [ac_bcd_session_list[-1], ct_bcd_session_list[-1]]):
        bottleneck_choice = utils.extract_array([s.get_slice_stats(unit="bottleneck choice") for s in sessions], size=n_rewards)
        utils.save_modular_data(f"{gt} Mask D goal transition array", bottleneck_choice, save_dir, overwrite=overwrite)
        # tpc in the first two rewards
        tpc_list = [s.slice_by_traverse_idx(None, 1).get_tiles_per_corridor() for s in sessions]
        # Differentiate success and unsuccessful mouse:
        if gt == "Acortical" and mask == "D":
            learned_animals = count_df[count_df.Mask==mask].Highly_rewarded_animal_list.values[0].split(", ")
            tpc_list =[(1, tpc) if s.name.split("_")[0] in learned_animals else (0, tpc) for tpc, s in zip(tpc_list, sessions)]
        utils.save_modular_data(f"{gt} Mask D tiles per corridor", np.array(tpc_list), save_dir, overwrite=overwrite)
        # traverse fit:
        data_type = config.CURVE_FIT_SPECS[0][0]  # only fit duration
        td_df = utils.get_traverse_data_df(sessions, data_type)
        utils.save_curve_fit_input(f"{gt} {mask} {data_type}", td_df, data_type, n_rewards, save_dir, overwrite=overwrite)

        # similarity list
        similarity_list = []
        for session in sessions:
            if len(session.filter("traverse")) == 0:
                continue
            similarity_list.append(session.get_three_traverse_similarity_matrix())
        utils.save_modular_data(f"{gt} D similarity matrices", similarity_list, save_dir, overwrite=overwrite)
        # average traverse similarity:
        avg_sims = []
        for sim_tuple in similarity_list:
            avg_sims.append(utils.get_average_traverse_similarity(*sim_tuple))
        utils.save_modular_data(f"{gt} D average traverse similarity", np.array(avg_sims), save_dir, overwrite=overwrite)

    # Supplementary: Mask E and F
    # count first hour reward for mask O:
    print("Processing Mask E Supplementary data...")
    # acortical e
    acortical_e_names = count_df[count_df.Mask=="E"].Highly_rewarded_animal_list.values[0].split(", ")
    e_sub_df = acortical_mdf[acortical_mdf.Animal.isin(acortical_e_names)]
    acortical_first_e = utils.get_first_learning_session(data, e_sub_df, mask_name="E", strict_first=True,)
    control_first_e = utils.get_first_learning_session(data, control_mdf, mask_name="E", strict_first=True,)
    for gt, sessions in zip(["Acortical", "Control"], [acortical_first_e, control_first_e]):
        reward_ints = [s.reward_interval_seconds for s in sessions]
        # drop_empty_rows: report effective n (animals with reward data), not sessions that entered E
        utils.save_modular_data(f"{gt} Mask E reward intervals", drop_empty_rows(utils.extract_array(reward_ints, size=n_rewards)), save_dir, overwrite=overwrite)
        sortie_array = drop_empty_rows(utils.extract_array([s.get_slice_stats(unit="sortie counts") for s in sessions], size=n_rewards))
        utils.save_modular_data(f"{gt} Mask E sortie counts", sortie_array, save_dir, overwrite=overwrite)
        tpc_list = [s.slice_by_traverse_idx(None, 1).get_tiles_per_corridor() for s in sessions]
        utils.save_modular_data(f"{gt} Mask E tiles per corridor", np.array(tpc_list), save_dir, overwrite=overwrite)
        # traverse data
        for data_type in ["turn error rate", "duration", "speed"]:
            data_array = utils.extract_array([s.filter("traverse").get_bout_stats(unit=data_type) for s in sessions], size=n_rewards)
            utils.save_modular_data(f"{gt} Mask E traverse {data_type}", data_array, save_dir, overwrite=overwrite)
        # curve fits
        for sub_tuple in config.CURVE_FIT_SPECS:
            data_type, params_name, _params_latex, p0, lower_bounds, upper_bounds = sub_tuple
            td_df = utils.get_traverse_data_df(sessions, data_type)
            utils.save_curve_fit_input(f"{gt} Mask E Gen {data_type}", td_df, data_type, n_rewards*2, save_dir, overwrite=overwrite)

    acortical_first_f = utils.get_first_learning_session(data, e_sub_df, mask_name="F", strict_first=False, e_trained="Yes")

    # save the same thing for acortical first f
    reward_ints = [s.reward_interval_seconds for s in acortical_first_f]
    # drop_empty_rows: 2 of the 6 Mask-F animals completed 0 rewards/traverses -> all-NaN rows;
    # drop them so the Fig S13 C/D reward-interval & sortie panels report the effective n=4
    # (matching the traverse-based E/F curve fits), instead of the 6 sessions that entered Mask F.
    utils.save_modular_data(f"Acortical Mask F reward intervals", drop_empty_rows(utils.extract_array(reward_ints, size=n_rewards)), save_dir,
                            overwrite=overwrite)
    sortie_array = drop_empty_rows(utils.extract_array([s.get_slice_stats(unit="sortie counts") for s in acortical_first_f], size=n_rewards))
    utils.save_modular_data(f"Acortical Mask F sortie counts", sortie_array, save_dir, overwrite=overwrite)
    tpc_list = [s.slice_by_traverse_idx(None, 1).get_tiles_per_corridor() for s in acortical_first_f]
    utils.save_modular_data(f"Acortical Mask F tiles per corridor", np.array(tpc_list), save_dir, overwrite=overwrite)
    # traverse data
    for data_type in ["turn error rate", "duration", "speed"]:
        data_array = utils.extract_array([s.filter("traverse").get_bout_stats(unit=data_type) for s in acortical_first_f],
                                         size=n_rewards)
        utils.save_modular_data(f"Acortical Mask F traverse {data_type}", data_array, save_dir, overwrite=overwrite)

    for sub_tuple in config.CURVE_FIT_SPECS:
        data_type, params_name, _params_latex, p0, lower_bounds, upper_bounds = sub_tuple
        td_df = utils.get_traverse_data_df(acortical_first_f, data_type)
        utils.save_curve_fit_input(f"Acortical F Gen {data_type}", td_df, data_type, n_rewards*2, save_dir, overwrite=overwrite)

    example_e_traverses = [(k, data[config.ACORTICAL_E_EXAMPLE_MOUSE][1].filter("traverse")[k]) for k in config.ACORTICAL_E_EXAMPLE_TRAVERSES]
    # R8 replacement: the same traverses as flat per-bout/per-tile tables, with the
    # config.ACORTICAL_E_EXAMPLE_TRAVERSES index kept in the meta table's `label` column.
    for suffix, table in utils.get_example_bout_tables(
            example_e_traverses, cache="Acortical E traverse example").items():
        utils.save_modular_data(f"Acortical E traverse example {suffix}", table, save_dir,
                                overwrite=overwrite)
    # 
    # === Learn A after E: first-E vs A-after-E learning curves and fits (both groups) ===
    # For animals that saw mask E before A, fit and array the per-traverse metrics
    # (duration, turn error) for the first-E session and the subsequent A session,
    # to test transfer/generalization from E to A.
    # Learn A after E
    acortical_names = acortical_mdf.Animal.unique().tolist()
    ac_ea_df = utils.create_mask_learn_seq_df(acortical_mdf, ac_gen_names, mask_list=["E", "A"])
    ct_ea_df = utils.create_mask_learn_seq_df(control_mdf, control_names, mask_list=["E", "A"])

    for gt, df in zip(["Acortical", "Control"], [ac_ea_df, ct_ea_df]):
        # filter out the first e
        e_df = df[(df.Mask=="E")&(df.Mask_order==0)]
        e_sessions = utils.get_session_list_from_df(data, e_df)
        a_df = df[(df.Mask=="A")&(df.Mask_order==1)]
        a_sessions = utils.get_session_list_from_df(data, a_df)
        for sub_tuple in config.CURVE_FIT_SPECS:
            data_type, params_name, _params_latex, p0, lower_bounds, upper_bounds = sub_tuple
            e_results = utils.get_traverse_data_df(e_sessions, data_type)
            utils.save_curve_fit_input(f"{gt} E first {data_type}", e_results, data_type, n_rewards*2, save_dir, overwrite=overwrite)
            a_results = utils.get_traverse_data_df(a_sessions, data_type)
            utils.save_curve_fit_input(f"{gt} A after E {data_type}", a_results, data_type, n_rewards*2, save_dir, overwrite=overwrite)

            e_array = utils.extract_array([s.filter("traverse").get_bout_stats(unit=data_type) for s in e_sessions], size=n_rewards)
            a_array = utils.extract_array([s.filter("traverse").get_bout_stats(unit=data_type) for s in a_sessions], size=n_rewards)
            utils.save_modular_data(f"{gt} E first traverse {data_type}", e_array, save_dir, overwrite=overwrite)
            utils.save_modular_data(f"{gt} A after E traverse {data_type}", a_array, save_dir, overwrite=overwrite)


if __name__ == "__main__":
    main()
