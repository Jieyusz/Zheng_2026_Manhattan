"""
Generate the olfactory landmark-swap figure data (fig:swap).

Compares turn-error and duration across the pre-swap vs post-swap sessions to test whether a
massive landmark change resets turn learning, and stores the paired swap sessions.

Saved keys
----------
"Wildtype {condition} duration"                          : per-animal traverse-duration arrays per swap condition.
"Wildtype {condition} turn error rate {direction}"       : per-animal turn-error-rate arrays per condition and travel direction.
"Wildtype swap turn error rate relative ratio" / "... gap data points" : pre/post swap ratio + per-animal points.
"Swap example {bout steps|tile steps|bout meta|manifest}" : flat per-bout/per-tile tables for the
                                                           pre/post-swap sessions (.parquet); `pair_idx`
                                                           gives the shared raster row and `segment` is
                                                           "pre"/"post".
See docs/data_contracts.md §12.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_olfaction.py --overwrite [--seed 0]
"""
import manhattan_maze as mm
from manhattan_maze import utils
import pandas as pd
import argparse
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
    bl6j_mdf = mdf[mdf.Genotype=="BL6J"]
    n_traverses = 20

    # === Mask-A swap experiment: pre- vs post-swap learning curves (BL6J "O, A, A" animals) ===
    # The swap cohort runs Mask O (session 0) then two Mask A sessions with the maze
    # entrance/exit swapped between them: session 1 = pre-swap, session 2 = post-swap.
    # Duration is fit as a single combined curve over interleaved traverses; turn
    # error rate is split into outbound (H->O) and homebound (O->H) traverses and fit
    # separately (each direction re-indexed 1..N). The live pre/post Session objects
    # are also saved for the swap-aligned reward raster.
    n_dir_traverses = 20  # per-direction traverses shown for the turn-error panels
    # Traverse direction -> Bout.bout_type criterion (see Session.filter / Bout.satisfy).
    directions = [("outbound", "H-O"), ("homebound", "O-H")]

    nicknames_swap = bl6j_mdf[bl6j_mdf["Config_label_list"].str.contains("A, A")].Nickname.tolist()
    preswap_sessions = [data[nickname][1] for nickname in nicknames_swap]
    postswap_sessions = [data[nickname][2] for nickname in nicknames_swap]

    for condition, sessions in zip(["Pre-swap", "Post-swap"], [preswap_sessions, postswap_sessions]):
        # Duration: one combined curve over the interleaved outbound/homebound traverses.
        # The centralised gen_curve_fits.py step turns each saved "... fit input" into a
        # "... fit results" cache; x_grid_max=n_traverses matches the plotted window.
        dur_array = utils.extract_array([s.filter("traverse").get_bout_stats(unit="duration") for s in sessions],
                                        size=n_traverses)
        utils.save_modular_data(f"Wildtype {condition} duration", dur_array, save_dir, overwrite=overwrite)
        dur_df = utils.get_traverse_data_df(sessions, "duration")
        utils.save_curve_fit_input(f"Wildtype {condition} duration", dur_df, "duration", n_traverses, save_dir, overwrite=overwrite)

        # Turn error rate: outbound and homebound fit separately. Direction-filtering
        # before get_traverse_data_df re-indexes each direction as its own 1..N
        # traverse sequence; the fit still uses every available traverse per direction.
        for direction, criterion in directions:
            dir_sessions = [s.filter(criterion) for s in sessions]
            err_array = utils.extract_array([s.get_bout_stats(unit="turn error rate") for s in dir_sessions],
                                            size=n_dir_traverses)
            utils.save_modular_data(f"Wildtype {condition} turn error rate {direction}", err_array, save_dir, overwrite=overwrite)
            err_df = utils.get_traverse_data_df(dir_sessions, "turn error rate")
            utils.save_curve_fit_input(f"Wildtype {condition} turn error rate {direction}", err_df,
                                       "turn error rate", n_dir_traverses, save_dir, overwrite=overwrite)

    # R8 replacement: flat tables for the same sessions. The pre/post pair sharing one
    # raster row is flattened to consecutive `example` rows, with `pair_idx` (the raster
    # row) and `segment` ("pre"/"post") recorded so plot_olfaction can apply the right
    # reverse/plot_end per segment without live objects.
    swap_index, swap_sessions = [], []
    for pair_idx, (pre_session, post_session) in enumerate(zip(preswap_sessions, postswap_sessions)):
        for segment, session in (("pre", pre_session), ("post", post_session)):
            swap_index.append({"example": len(swap_sessions), "pair_idx": pair_idx,
                               "segment": segment})
            swap_sessions.append(session)
    swap_index = pd.DataFrame(swap_index)
    for suffix, table in utils.get_example_session_tables(swap_sessions,
                                                         cache="Swap example").items():
        if suffix in ("bout meta", "manifest"):
            table = table.merge(swap_index, on="example", how="left")
        utils.save_modular_data(f"Swap example {suffix}", table, save_dir, overwrite=overwrite)

    # === Swap turn-error relative ratio (post-first / pre-first), split by direction ===
    # For each direction, compare each animal's post-swap first-N turn error to its own
    # pre-swap first-N (baseline), via the same hierarchical bootstrap as the memory
    # panels (utils.relative_performance). Ratio < 1 => savings/transfer across the swap.
    # Stored keyed by direction ("Outbound"/"Homebound") for plot_utils.plot_relative_memory,
    # with a single "gap" (Day 0 = pre-swap baseline, Day 1 = post-swap comparison).
    n_first_traverses = 10
    swap_gap = (0, 1)
    ratio_dict = {}
    gap_points_dict = {}
    for direction, criterion in directions:
        rows = []
        for pre_session, post_session in zip(preswap_sessions, postswap_sessions):
            animal = pre_session.name.split("_")[0]
            for day, session in [(0, pre_session), (1, post_session)]:
                values = session.filter(criterion).get_bout_stats("turn error rate")[:n_first_traverses]
                for bout, value in enumerate(values):
                    rows.append((animal, day, bout + 1, value, "Wildtype"))
        ratio_df = pd.DataFrame(rows, columns=["Animal", "Day", "Bout", "Value", "Genotype"])
        observed_ratio, (low, high) = utils.relative_performance(ratio_df, n_iterations=1000, seed=args.seed)
        ratio_dict[direction.capitalize()] = [(swap_gap, observed_ratio, low, high)]
        # Per-animal SessionRatio for the raw scatter (mirrors gen_ac_mem.py:257-273).
        session_means = ratio_df.groupby(["Animal", "Day"]).agg(
            {"Value": "mean", "Genotype": "first"}).reset_index().rename(columns={"Value": "SessionMean"})
        baseline_by_animal = ratio_df[ratio_df.Day == 0].groupby("Animal").agg(
            {"Value": "mean"}).reset_index().rename(columns={"Value": "BaselineMean"})
        session_means = session_means.merge(baseline_by_animal, on="Animal", how="left")
        session_means["SessionRatio"] = session_means["SessionMean"] / session_means["BaselineMean"]
        gap_points_dict[direction.capitalize()] = [
            (swap_gap, session_means[["Animal", "Day", "SessionMean", "BaselineMean", "SessionRatio", "Genotype"]])]

    utils.save_modular_data("Wildtype swap turn error rate relative ratio", ratio_dict, save_dir, overwrite=overwrite)
    utils.save_modular_data("Wildtype swap turn error rate gap data points", gap_points_dict, save_dir, overwrite=overwrite)

if __name__ == "__main__":
    main()
