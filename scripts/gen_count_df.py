"""
Generate the acortical learning-count table — the pipeline PREREQUISITE.

Scores, per acortical/control animal, how many masks each learned and the per-animal
learning counts, and writes them (plus a plain-CSV mirror) as the shared source that the
acortical producers (``gen_acortical_learning.py``, ``gen_ac_generalization.py``,
``gen_ac_mem.py``) read back. Runs FIRST/serially in ``batch_generate_figure_data.py`` (R7).

Saved keys
----------
"Acortical learning count df"          : DataFrame of per-animal/per-mask learning counts (also mirrored to Acortical_learning_count_df.csv).
"Acortical animal learned masks df"    : DataFrame of which masks each animal learned (mirrored to Acortical_animal_learned_masks_df.csv).
See docs/data_contracts.md §12.

Run (m_maze env, from scripts/, repo on PYTHONPATH):
    python gen_count_df.py --overwrite
"""
import manhattan_maze as mm
from manhattan_maze import utils
import argparse
import config


def main():
    parser = argparse.ArgumentParser(description="Generate the acortical mask-learning count table")
    parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    overwrite = args.overwrite

    ## shared paths and DataLoader configuration (see scripts/config.py)
    save_dir = config.SAVE_DIR
    data = mm.DataLoader(config.DATA_DIR, **config.DATALOADER_KWARGS)
    mdf = data.metadata
    acortical_mdf = mdf[mdf.Genotype=="HO"]

    print("Processing Acortical learning count table...")
    # === Acortical learning summary table (which masks each animal learned) ===
    # Build/cache the per-(animal, mask) reward-count table that classifies animals
    # as highly-rewarded / long-term, then derive the learned-masks summary. This is
    # a standalone prerequisite (R7): the count CSV it writes is read by the parallel
    # producer scripts (gen_acortical_learning.py, gen_ac_generalization.py,
    # gen_ac_mem.py), so it must run before them and only once.
    if overwrite:
        count_df = utils.get_mask_learning_count_df(data, acortical_mdf, ) # takes some time to generate
        utils.save_modular_data("Acortical learning count df", count_df, save_dir, overwrite=overwrite)
        count_df.to_csv(f"{save_dir}/Acortical_learning_count_df.csv", index=False)
        # also save as csv for easier inspection
    else:
        import pandas as pd
        count_df = pd.read_csv(f"{save_dir}/Acortical_learning_count_df.csv")

    animal_mask_df = utils.get_animal_learning_masks_df(count_df)
    utils.save_modular_data("Acortical animal learned masks df", animal_mask_df, save_dir, overwrite=overwrite)
    animal_mask_df.to_csv(f"{save_dir}/Acortical_animal_learned_masks_df.csv", index=False)


if __name__ == "__main__":
    main()
