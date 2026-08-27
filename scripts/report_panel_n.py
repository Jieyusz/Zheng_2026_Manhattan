"""
Standalone, READ-ONLY reviewer-support report: number of animals (n) per panel
of the six MAIN manuscript figures, plus a Methods-level "Subjects" summary.

Why this exists
---------------
A reviewer asked for sample sizes. In this pipeline n is *data-driven*: every
panel's cohort is produced by a ``scripts/gen_*.py`` that filters
``data/manhattan_metadata_published.csv`` and calls ``manhattan_maze.analysis`` selectors.
The most defensible per-panel n is therefore read straight from the produced
artifacts in ``data/figure_data/`` (regenerate them first with the producer
scripts), not hand-counted from the CSV:

* per-animal metric arrays are ``(n_animals, n_traverses)`` -> ``n = shape[0]``
  (``docs/data_contracts.md`` §12);
* each curve-fit panel reads a ``"<base> fit input"`` payload whose tidy
  ``data_df`` carries an ``Animal`` column -> ``n = data_df.Animal.nunique()``;
* the two-day panels read ``"Wildtype two day <metric> tidy"`` -> n per
  ``(Session, Mask)`` group;
* example / schematic panels have n = 1 example animal or no n.

This script does not modify anything. It writes ``docs/panel_n_report.md``.

PII: the report emits only aggregate counts. It never reads or prints Nickname,
sex, or age (the producers print some of those to stdout; we do not).

Run (from scripts/, in the m_maze env, with the repo on PYTHONPATH):
    python report_panel_n.py
"""
import os
import numpy as np
import pandas as pd

from manhattan_maze import utils
import config

# ---------------------------------------------------------------------------
# Load the freshly regenerated figure-data cache and the metadata table.
# ---------------------------------------------------------------------------
dd = utils.load_all_figure_data(str(config.SAVE_DIR))

_GENOTYPE_LABEL = {"BL6J": "Wildtype", "HO": "Acortical", "WT": "Control"}
metadata = pd.read_csv(config.DATA_DIR / config.DATALOADER_KWARGS["metadata_filename"])
metadata["Animal"] = metadata["Nickname"].str.split("_").str[0]  # derived like DataLoader._get_metadata

# Collect cross-check notes (array shape[0] vs fit-input Animal count, etc.).
crosschecks = []


# ---------------------------------------------------------------------------
# n-extraction helpers (each returns an int or None if the key is absent).
# ---------------------------------------------------------------------------
def n_array(key):
    """n = first-axis length of a per-animal metric array."""
    arr = dd.get(key)
    if arr is None:
        return None
    return int(np.asarray(arr).shape[0])


def n_with_data(key):
    """Count array rows that are not entirely NaN (animals that actually contributed data)."""
    arr = dd.get(key)
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    return int((~np.isnan(a).all(axis=1)).sum())


def n_fit_input(base):
    """n = distinct animals in the tidy fit-input frame for ``<base>``."""
    payload = dd.get(f"{base} fit input")
    if payload is None:
        return None
    return int(payload["data_df"]["Animal"].nunique())


def n_list(key):
    """n = number of items (example sessions / traverses / points)."""
    obj = dd.get(key)
    if obj is None:
        return None
    return len(obj)


def n_unique(key, column):
    """
    n = distinct values of ``column`` in the table at ``key``.

    Used where one exported row is not one animal: the swap-example manifest holds a
    pre- and a post-swap row per animal, so the animal count is the number of distinct
    ``pair_idx`` values rather than the row count.
    """
    table = dd.get(key)
    if table is None:
        return None
    return int(table[column].nunique())


def tidy_group_counts(metric):
    """{(str(Session), Mask): n_distinct_animals} for the two-day tidy frame.

    Session is normalised to ``str`` so lookups are robust to int/str dtype.
    """
    tidy = dd.get(f"Wildtype two day {metric} tidy")
    if tidy is None:
        return {}
    g = tidy.groupby(["Session", "Mask"])["Animal"].nunique()
    return {(str(s), str(m)): int(v) for (s, m), v in g.items()}


def paired_n(metric, group_a, group_b):
    """Within-subject paired n: animals present in BOTH (Session, Mask) groups."""
    tidy = dd.get(f"Wildtype two day {metric} tidy")
    if tidy is None:
        return None

    def animals(sess, mask):
        sub = tidy[(tidy["Session"].astype(str) == str(sess)) & (tidy["Mask"] == mask)]
        return set(sub["Animal"].unique())

    return len(animals(*group_a) & animals(*group_b))


def crosscheck(panel, base_array_key, fit_base):
    """Record array-shape n vs fit-input n for a panel that has both.

    The array's first axis counts *session rows* (one per animal in these cohorts),
    while the fit-input tidy counts *animals with >=1 scored traverse*. When an animal
    completes zero traverses it is an all-NaN array row but contributes no fit-input
    rows, so ``array > fit_input`` is expected, not an error.
    """
    a = n_array(base_array_key)
    f = n_fit_input(fit_base)
    if a is not None and f is not None:
        if a == f:
            status = "OK"
        else:
            status = f"array={a} animals attempted; fit-input={f} with traverse data ({a - f} completed 0 traverses)"
        crosschecks.append((panel, base_array_key, a, fit_base, f, status))
    return a if a is not None else f


def fmt(n):
    return "n/a" if n is None else str(n)


# ---------------------------------------------------------------------------
# Methods-level Subjects summary (distinct animals; counts only, no IDs).
# ---------------------------------------------------------------------------
def subjects_section():
    lines = ["## Subjects (Methods totals)", ""]
    total = metadata["Animal"].nunique()
    n_rec = len(metadata)
    lines.append(f"Total distinct animals: **{total}** across {n_rec} recordings "
                 "(animal ID = the `Nickname` prefix before the first `_`).")
    lines.append("")
    lines.append("**By genotype (manuscript label):**")
    lines.append("")
    lines.append("| Genotype | n animals |")
    lines.append("|---|---|")
    for gt_code, label in _GENOTYPE_LABEL.items():
        n = metadata.loc[metadata["Genotype"] == gt_code, "Animal"].nunique()
        lines.append(f"| {label} (`{gt_code}`) | {n} |")
    lines.append("")
    lines.append("**By housing / cage-position condition:**")
    lines.append("")
    lines.append("| Condition | n animals |")
    lines.append("|---|---|")
    cond = metadata["Condition"].astype(str)
    west = metadata.loc[cond.str.contains("west", case=False), "Animal"].nunique()
    north = metadata.loc[cond == "Single_north", "Animal"].nunique()
    lines.append(f"| West cohort (`*_west`) | {west} |")
    lines.append(f"| North cohort (`Single_north`) | {north} |")
    lines.append("")
    lines.append("**Inclusion criteria (applied by the producers):** learning-based "
                 "inclusion keeps highly-rewarded animals (`Reward_count >= ~20` in >=1 "
                 "session; `analysis.get_mask_learning_count_df`); first-exposure analyses "
                 "use `strict_first`/`e_trained` filters (`analysis.get_first_learning_session`); "
                 "`A_flipped` sessions are dropped. Per-panel n below is therefore <= these totals.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Per-figure panel tables. Each row: letter, description, n, unit, notes.
# ---------------------------------------------------------------------------
def table(rows):
    out = ["| Panel | Shows | n (animals) | Statistical unit | Notes |",
           "|---|---|---|---|---|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    out.append("")
    return out


def figure1():
    n_dur = crosscheck("Fig1 G", "Wildtype A traverse duration", "Wildtype A duration")
    rows = [
        ["A", "3D maze schematic", "n/a", "—", "Schematic"],
        ["B", "Mask A top view + shortest path", "n/a", "—", "Schematic (geometry)"],
        ["C", "P10 path-graph + tile-graph schematic", "n/a", "—", "Schematic (geometry)"],
        ["D", "Reward-interval raster (example sessions)",
         fmt(n_list("Mask A example manifest")), "example animals", "Illustrative examples"],
        ["E", "Tile-distance vs time (example segment)", "1", "1 example animal", "Single example segment"],
        ["F", "Example traverses (one animal)", "1", "1 example animal", "Traverses of one example animal"],
        ["G", "Traverse duration & turn-error rate vs traverse #",
         fmt(n_dur), "animals (mean +/- shade)",
         "Wildtype Day-1 Mask A cohort; duration & turn-error arrays share this n"],
        ["H", "Corridor & tile error vs traverse #",
         fmt(n_array("Wildtype A Corridor error array")), "animals (mean +/- shade)",
         "Same Day-1 Mask A cohort"],
    ]
    return ["### Figure 1 — `first_mask.pdf` (Wildtype Mask A learning)", ""] + table(rows)


def figure2():
    dur_groups = tidy_group_counts("duration")
    te_groups = tidy_group_counts("turn error rate")

    def group_line(groups, mask):
        items = sorted((s, n) for (s, m), n in groups.items() if m == mask)
        return ", ".join(f"S{s}={n}" for s, n in items) if items else "n/a"

    overnight = paired_n("duration", ("1", "A"), ("2", "A"))
    bc_b = dur_groups.get(("2", "B"))
    bc_c = dur_groups.get(("2", "C"))
    rows = [
        ["A", "Mask B & C top views", "n/a", "—", "Schematic (geometry)"],
        ["B", "Two-day experiment timeline", "n/a", "—", "Schematic"],
        ["C", "Overnight early-vs-late traverse comparison (Mask A)",
         "1", "1 example animal", "`bouts[:1]` -> one animal shown; drawn from 3 example sessions"],
        ["D", "Duration curve fits, Day1 + 4x Day2 (masks A/B/C)",
         f"per (Session,Mask): A[{group_line(dur_groups, 'A')}]",
         "animals (animal-level bootstrap CI)",
         f"B[{group_line(dur_groups, 'B')}], C[{group_line(dur_groups, 'C')}]; S1=Day1, S2-5=Day2-1..4"],
        ["E", "Turn-error-rate curve fits, Day1 + 4x Day2",
         f"per (Session,Mask): A[{group_line(te_groups, 'A')}]",
         "animals (animal-level bootstrap CI)",
         f"B[{group_line(te_groups, 'B')}], C[{group_line(te_groups, 'C')}]"],
        ["F", "Overnight CI ratios (Mask A Day2-1/Day1)",
         fmt(overnight), "animals (within-subject bootstrap)",
         "Paired n = animals present in both Day1 A and Day2-1 A"],
        ["G", "Generalization CI ratios (Day2-2+/Day1, per mask)",
         f"A[{group_line(dur_groups, 'A')}]", "animals (within-subject bootstrap)",
         "Per mask; within-subject over animals present in both sessions"],
        ["H", "Turn-sequence CI ratios (Day2-1 B/C)",
         f"Day2-1 B={fmt(bc_b)}, C={fmt(bc_c)}",
         "animals (within-subject bootstrap)",
         "Day2-1 Mask B vs Mask C"],
    ]
    note = ("_Fig 1 (Day-1 Mask A learning curve) and Fig 2 (Session 1, Mask A) use the same "
            "cohort: both draw Day-1 from each animal's `_a1` 'O, A' session, restricted to "
            "animals that also completed the two-day (`t1`) protocol. One BL6J mouse ran "
            "Day-1 Mask A but was injured and euthanised before Day 2 (no `t1` recording); it "
            "is excluded from both figures for consistency._")
    return (["### Figure 2 — `day2.pdf` (Wildtype two-day retention & generalization)", ""]
            + table(rows) + [note, ""])


def figure3():
    crosscheck("Fig3 C", "Wildtype D duration", "Wildtype D duration")
    n_att = n_array("Wildtype D duration")            # animals attempted (session rows)
    n_dat = n_fit_input("Wildtype D duration")         # animals with >=1 traverse
    n_d = f"{fmt(n_att)} ({fmt(n_dat)} with traverse data)" if n_att != n_dat else fmt(n_att)
    note = f"Wildtype Day-1 Mask D cohort; {n_att - n_dat if (n_att and n_dat) else 0} animal(s) completed 0 traverses (all-NaN row)"
    rows = [
        ["A", "Mask D graph + top view (bottleneck)", "n/a", "—", "Schematic (geometry)"],
        ["B", "Example Mask D traverses", "1", "1 example animal", "Traverses of one example session"],
        ["C", "Traverse duration vs traverse #", n_d, "animals (mean +/- SE)", note],
        ["D", "Corridor & tile error vs traverse #", n_d, "animals (mean +/- shade)", "Same Mask D cohort"],
        ["E", "Corridor-transition choice ratios (Outbound)", n_d, "animals (pooled transitions)",
         "Aggregated over the Mask D cohort; matrices pool sessions"],
        ["F", "Corridor-transition choice ratios (Homebound)", n_d, "animals (pooled transitions)", "Same cohort"],
        ["G", "Biclique off-path choice ratios (Hor->Ver)", n_d, "animals (pooled transitions)", "Same cohort"],
        ["H", "Biclique off-path choice ratios (Ver->Hor)", n_d, "animals (pooled transitions)", "Same cohort"],
    ]
    return ["### Figure 3 — `maskd.pdf` (Wildtype Mask D)", ""] + table(rows)


def figure4():
    n_ac = crosscheck("Fig4 E (Acortical)", "Acortical Mask A duration", "Acortical A duration")
    n_ct = crosscheck("Fig4 E (Control)", "Control Mask A duration", "Control A duration")
    # Wildtype Day-1 Mask A reference = two-day (Session 1, Mask A)
    n_wt = tidy_group_counts("duration").get(("1", "A"))
    rows = [
        ["A", "Reward raster (acortical + control examples)",
         f"Ac {fmt(n_list('Acortical A example manifest'))} + Ct {fmt(n_list('Control A example manifest'))}",
         "example animals", "Illustrative rasters"],
        ["B", "Tile-distance vs time (acortical example)", "1", "1 example animal", "One acortical example session"],
        ["C", "Tiles per corridor (by genotype)",
         f"Ac {fmt(n_array('Acortical Mask A tiles per corridor'))}, "
         f"Ct {fmt(n_array('Control Mask A tiles per corridor'))}, "
         f"WT {fmt(n_array('Wildtype Mask A tiles per corridor'))}",
         "animals (scatter + box)", "Kruskal-Wallis across genotypes"],
        ["D", "Example outbound/homebound traverses", "1", "1 example animal (+1 control ref)", "Illustrative"],
        ["E", "Duration curve fits (Acortical / Control raw / WT ref)",
         f"Ac {fmt(n_ac)}, Ct {fmt(n_ct)}, WT {fmt(n_wt)}",
         "animals (bootstrap CI; Control = raw traces)",
         "Control shown as raw per-animal traces (n too small for CI)"],
        ["F", "Turn-error-rate curve fits (same groups)",
         f"Ac {fmt(n_fit_input('Acortical A turn error rate'))}, "
         f"Ct {fmt(n_array('Control Mask A turn error rate'))}, WT {fmt(n_wt)}",
         "animals (bootstrap CI; Control = raw traces)", "Same cohorts as E"],
        ["G", "Genotype CI parameter ratios (Wildtype/Acortical)",
         f"Ac {fmt(n_ac)} vs WT {fmt(n_wt)}", "animals (bootstrap ratio CI)",
         "Only Wildtype/Acortical shown (control n too small)"],
    ]
    return ["### Figure 4 — `acortical_rapid.pdf` (Acortical rapid learning, Mask A)", ""] + table(rows)


def figure5():
    def mem_n(genotype):
        # distinct animals contributing to the Mask-A memory-vs-gap points
        obj = dd.get(f"{genotype} Mask A duration gap data points")
        if obj is None:
            return None
        animals = set()
        for _gap, df in obj:
            animals |= set(df["Animal"].unique())
        return len(animals)

    # Fig 5I is a learned-vs-unlearned split of the full Mask-D cohort: "+" = the successful
    # learners (== the Fig 5G fit cohort), "-" = the rest.
    _ac_tpc = n_list("Acortical Mask D tiles per corridor")
    _ac_learners = n_fit_input("Acortical D Gen duration")
    _ac_unlearned = (_ac_tpc - _ac_learners) if (_ac_tpc is not None and _ac_learners is not None) else None
    rows = [
        ["A", "Example acortical Mask-A memory across days", "1", "1 example animal", "One animal across repeated days"],
        ["B", "Relative-duration memory vs gap (by genotype)",
         f"Ac {fmt(mem_n('Acortical'))}, Ct {fmt(mem_n('Control'))}, WT {fmt(mem_n('Wildtype'))}",
         "animals (bootstrap ratio CI)", "Ratio vs each animal's baseline day"],
        ["C", "Relative-error memory vs gap (by genotype)",
         f"Ac {fmt(mem_n('Acortical'))}, Ct {fmt(mem_n('Control'))}, WT {fmt(mem_n('Wildtype'))}",
         "animals (bootstrap ratio CI)", "Same cohorts as B"],
        ["D", "Generalization duration fits (First A / Repeat A / B / C)",
         f"First A {fmt(n_fit_input('Acortical A duration'))}, "
         f"Repeat A {fmt(n_fit_input('Acortical A repeat Gen duration'))}, "
         f"B {fmt(n_fit_input('Acortical B Gen duration'))}, "
         f"C {fmt(n_fit_input('Acortical C Gen duration'))}",
         "animals (bootstrap CI)", "Acortical only"],
        ["E", "Generalization turn-error fits (same cohorts)",
         f"First A {fmt(n_fit_input('Acortical A turn error rate'))}, "
         f"Repeat A {fmt(n_fit_input('Acortical A repeat Gen turn error rate'))}, "
         f"B {fmt(n_fit_input('Acortical B Gen turn error rate'))}, "
         f"C {fmt(n_fit_input('Acortical C Gen turn error rate'))}",
         "animals (bootstrap CI)", "Acortical only"],
        ["F", "Generalization CI parameter ratios",
         f"see D (First A vs Repeat A/B/C)", "animals (bootstrap ratio CI)", "Within acortical cohort"],
        ["G", "Mask D duration fits (cross-genotype)",
         f"Ac {fmt(n_fit_input('Acortical D Gen duration'))}, "
         f"Ct {fmt(n_fit_input('Control D duration'))}, "
         f"WT {fmt(n_fit_input('Wildtype D duration'))}",
         "animals (bootstrap CI)", "Mask D across genotypes"],
        ["H", "Mask D bottleneck choice by reward (cross-genotype)",
         f"Ac {fmt(n_array('Acortical Mask D goal transition array'))} "
         f"({fmt(n_with_data('Acortical Mask D goal transition array'))} w/ reward data), "
         f"Ct {fmt(n_array('Control Mask D goal transition array'))}, "
         f"WT {fmt(n_array('Wildtype Mask D goal transition array'))} "
         f"({fmt(n_with_data('Wildtype Mask D goal transition array'))} w/ reward data)",
         "animals (mean +/- shade)", "Chance = 0.2; counts = animals attempted (full cohort)"],
        ["I", "Mask D tiles per corridor (Acortical +/- , Control)",
         f"Ac {fmt(_ac_tpc)} (+ {fmt(_ac_learners)} learners / - {fmt(_ac_unlearned)} unlearned), "
         f"Ct {fmt(n_list('Control Mask D tiles per corridor'))}",
         "animals (scatter + box)", "Acortical split into learned(+)/unlearned(-)"],
    ]
    return ["### Figure 5 — `ac_mem_gen.pdf` (Acortical memory & generalization)", ""] + table(rows)


def figure6():
    rows = [
        ["A", "Example corridor random walk", "n/a", "—", "Model input (one example corridor sequence)"],
        ["B-D", "Learned goal-signal graph snapshots", "n/a", "—", "Endotaxis-model schematic"],
        ["E-G", "Goal-signal (log) profiles over corridors", "n/a", "—", "Endotaxis-model schematic"],
    ]
    return ["### Figure 6 — `algo.pdf` (Endotaxis algorithm illustration)", "",
            "_Model/algorithm schematic on a single example corridor sequence — no per-animal n._", ""] + table(rows)


def n_bypos(key):
    """n animals for a '<...> error by position' dict: arrays are (n_pos, n_animals)."""
    v = dd.get(key)
    if v is None:
        return None
    return int(np.asarray(v["H-O"]).shape[1])


def n_day2(metric, sidx, mask):
    """n animals for a Day-2 (session slot sidx, mask) group in the 'Day 2 <metric>' cache."""
    obj = dd.get(f"Day 2 {metric}")
    if obj is None:
        return None
    return int(np.asarray(obj[sidx][0][mask]).shape[0])


def trip(prefix_suffix):
    """Compact 'Ac x, Ct y, WT z' string for a per-genotype key suffix (array shape[0])."""
    ac, ct, wt = (f"{g} {prefix_suffix}" for g in ("Acortical", "Control", "Wildtype"))
    return f"Ac {fmt(n_array(ac))}, Ct {fmt(n_array(ct))}, WT {fmt(n_array(wt))}"


def duo(prefix_suffix):
    """Compact 'Ac x, Ct y' string for a two-genotype key suffix (array shape[0])."""
    ac, ct = (f"{g} {prefix_suffix}" for g in ("Acortical", "Control"))
    return f"Ac {fmt(n_array(ac))}, Ct {fmt(n_array(ct))}"


def build_supplementary():
    lines = ["## Supplementary figures", "",
             "_Supplementary figure numbers follow their order of appearance in the manuscript "
             "(after the Supplementary Materials marker). n derived exactly as for the main "
             "figures. Mask-D wildtype cohort = 7 attempted / 6 with traverse data (see Fig 3)._",
             ""]

    # S1 — oa_supp.pdf
    n_wa = n_array("Wildtype A traverse duration")  # 25
    lines += ["### Fig S1 — `oa_supp.pdf` (Day-1 supplementary)", ""] + table([
        ["A", "Labeled top-view photo of the maze", "n/a", "—", "Photograph"],
        ["B", "Mask O top view + path/tile graph", "n/a", "—", "Schematic (geometry)"],
        ["C", "Hole-decision schematic", "n/a", "—", "Schematic"],
        ["D", "Mask O reward intervals", fmt(n_array("Wildtype O reward intervals")),
         "animals (mean +/- shade)", "Wildtype Day-1 Mask O cohort (same animals as Mask A)"],
        ["E", "Example Mask A speed profile", "1", "1 example animal", "One example session"],
        ["F", "Mask A traverse speed", fmt(n_wa), "animals (mean +/- shade)", "Wildtype Day-1 Mask A cohort"],
        ["G", "Mask A sortie counts", fmt(n_array("Wildtype A sortie counts")), "animals (mean +/- shade)", "Same cohort"],
        ["H", "Mask A reward intervals", fmt(n_array("Wildtype A reward intervals")), "animals (mean +/- shade)", "Same cohort"],
    ])

    # S2 — day2_supp.pdf (7 metric rows; masks A/B/C across Day2-1..4)
    def day2_line(mask):
        return "S2-" + ", ".join(f"{j+1}={fmt(n_day2('duration', j, mask))}" for j in range(4))
    lines += ["### Fig S2 — `day2_supp.pdf` (additional Day-2 metrics)", "",
              f"_All 7 rows (A reward intervals, B sorties, C speed, D duration, E turn error, "
              f"F tile error, G corridor error) use the same Day-2 (session, mask) cohorts as "
              f"Fig 2 D/E. n animals per Day-2 session — Mask A: [{day2_line('A')}]; "
              f"Mask B: [{day2_line('B')}]; Mask C: [{day2_line('C')}]._", ""]

    # S3 — curve_fit_supp.pdf (two-day CI params; same cohort as Fig 2 D-H)
    dg = tidy_group_counts("duration")
    def gl(m):
        return ", ".join(f"S{s}={n}" for (s, mm), n in sorted(dg.items()) if mm == m)
    lines += ["### Fig S3 — `curve_fit_supp.pdf` (two-day curve-fit parameter CIs)", "",
              f"_8 panels = CI forests of the duration & turn-error curve parameters "
              f"(D_inf/D_0/delta, E_inf/E_0/epsilon) for the two-day fits. Same cohort as Fig 2 "
              f"D-H; n per (Session, Mask) — A[{gl('A')}], B[{gl('B')}], C[{gl('C')}] "
              f"(S1=Day1, S2-5=Day2-1..4)._", ""]

    # S4 — d_supp.pdf
    n_d_att = n_array("Wildtype D reward intervals")
    n_d_dat = n_fit_input("Wildtype D duration")
    nd = f"{fmt(n_d_att)} ({fmt(n_d_dat)} with traverse data)"
    lines += ["### Fig S4 — `d_supp.pdf` (additional Mask-D first-mask metrics)", ""] + table([
        ["A", "Reward raster (example Mask D sessions)", fmt(n_list("Mask D example manifest")), "example animals", "Illustrative"],
        ["B", "Mask D reward intervals", nd, "animals (mean +/- shade)", "Wildtype Day-1 Mask D cohort"],
        ["C", "Mask D sortie counts", nd, "animals (mean +/- shade)", "Same cohort"],
        ["D", "Example Mask D speed profile", "1", "1 example animal", "One example session"],
        ["E", "Mask D traverse speed", nd, "animals (mean +/- shade)", "Same cohort"],
        ["F", "First-journey bottleneck timing", f"{fmt(n_d_att)} ({fmt(n_d_dat)} valid points)",
         "animals (scatter)", "One point/animal; NaN (no bottleneck/reward) dropped"],
    ])

    # S5 — d_motif.pdf (Wildtype Mask D route similarity)
    lines += ["### Fig S5 — `d_motif.pdf` (varied routes in Mask D)", "",
              f"_Wildtype Mask D cohort (7 attempted / 6 with traverse data). Similarity-matrix "
              f"and transition panels aggregate this cohort "
              f"(`Wildtype D average traverse similarity` n={fmt(n_array('Wildtype D average traverse similarity'))}, "
              f"`Wildtype D similarity matrices` n={fmt(n_list('Wildtype D similarity matrices'))}); "
              f"the example-route panel shows 1 example session._", ""]

    # S6 / S7 — MRI images
    lines += ["### Fig S6 / S7 — `mri3.png`, `MRI_HSV.png` (mutant-mouse anatomy)", "",
              "_Anatomical MRI/atlas overlays of the acortical (mutant) mouse — imaging figures, "
              "not a behavioural cohort statistic (single imaged specimen)._", ""]

    # S8 — ac_oa_supp.pdf
    lines += ["### Fig S8 — `ac_oa_supp.pdf` (acortical vs control, Mask O & Mask A)", ""] + table([
        ["A", "Mask O tiles per corridor", trip("Mask O tiles per corridor"), "animals (scatter + box)", "Kruskal-Wallis"],
        ["B", "Mask O reward intervals", duo("Mask O reward intervals"), "animals (mean +/- shade)", "Acortical vs Control"],
        ["C", "Mask O sortie counts", duo("Mask O sortie counts"), "animals (mean +/- shade)", "Acortical vs Control"],
        ["D", "Mask A speed", duo("Mask A speed"), "animals (mean +/- shade)", "Acortical vs Control"],
        ["E", "Mask A duration", f"{duo('Mask A duration')}, WT {fmt(n_wa)}", "animals (mean +/- shade)", "+ Wildtype ref"],
        ["F", "Mask A turn error rate", f"{duo('Mask A turn error rate')}, WT {fmt(n_wa)}", "animals (mean +/- shade)", "+ Wildtype ref"],
        ["G", "Mask A sortie counts", duo("Mask A sortie counts"), "animals (mean +/- shade)", "Acortical vs Control"],
        ["H", "Mask A tile error", duo("Mask A tile error"), "animals (mean +/- shade)", "Acortical vs Control"],
        ["I", "Mask A corridor error", duo("Mask A corridor error"), "animals (mean +/- shade)", "Acortical vs Control"],
    ])

    # S9 — ac_curve_fit_supp.pdf
    n_wt_d1a = tidy_group_counts("duration").get(("1", "A"))
    lines += ["### Fig S9 — `ac_curve_fit_supp.pdf` (curve-fit params: acortical/control/wildtype)", ""] + table([
        ["A", "Mask A duration params (First A)", f"Ac {fmt(n_fit_input('Acortical A duration'))}, "
         f"Ct {fmt(n_fit_input('Control A duration'))}, WT {fmt(n_wt_d1a)}",
         "animals (bootstrap CI; Control = per-animal points)", "First Mask A"],
        ["B", "Mask A turn-error params (First A)", f"Ac {fmt(n_fit_input('Acortical A turn error rate'))}, "
         f"Ct {fmt(n_fit_input('Control A turn error rate'))}, WT {fmt(n_wt_d1a)}",
         "animals (bootstrap CI; Control = per-animal points)", "First Mask A"],
        ["C", "Mask D duration params", f"Ac {fmt(n_fit_input('Acortical D Gen duration'))}, "
         f"Ct {fmt(n_fit_input('Control D duration'))}, WT {fmt(n_fit_input('Wildtype D duration'))}",
         "animals (bootstrap CI)", "Mask D across genotypes"],
        ["D", "Generalization duration params (Repeat A / B / C)",
         f"Ac: RA {fmt(n_fit_input('Acortical A repeat Gen duration'))}, B {fmt(n_fit_input('Acortical B Gen duration'))}, "
         f"C {fmt(n_fit_input('Acortical C Gen duration'))}", "animals (bootstrap CI)", "Control shown as CI shade"],
        ["E", "Generalization turn-error params (Repeat A / B / C)",
         f"Ac: RA {fmt(n_fit_input('Acortical A repeat Gen turn error rate'))}, B {fmt(n_fit_input('Acortical B Gen turn error rate'))}, "
         f"C {fmt(n_fit_input('Acortical C Gen turn error rate'))}", "animals (bootstrap CI)", "Control shown as CI shade"],
        ["F", "Mask D ratio forests (Control/WT, Acortical/WT)",
         f"Ac D {fmt(n_fit_input('Acortical D Gen duration'))}, Ct D {fmt(n_fit_input('Control D duration'))}, "
         f"WT D {fmt(n_fit_input('Wildtype D duration'))}", "animals (bootstrap ratio CI)", "Division"],
        ["G", "Generalization ratios (Control/Acortical)",
         "see D/E cohorts", "animals (bootstrap ratio CI)", "Repeat A/B/C"],
        ["H", "Generalization ratios (Wildtype/Acortical)",
         "see D/E cohorts", "animals (bootstrap ratio CI)", "Repeat A/B/C"],
    ])

    # S10 — ac_mem_sup.pdf
    def mem_n(g):
        obj = dd.get(f"{g} Mask A speed gap data points")
        if obj is None:
            return None
        s = set()
        for _gap, df in obj:
            s |= set(df["Animal"].unique())
        return len(s)
    lines += ["### Fig S10 — `ac_mem_sup.pdf` (long-term memory: Mask O & speed)", ""] + table([
        ["A", "Mask O memory intervals across gaps (acortical)",
         fmt(n_array("Acortical Mask O reward intervals")), "animals (mean +/- shade)", "Acortical Mask O cohort"],
        ["B", "Example acortical Mask-A memory speed", "1", "1 example animal", "One animal across days"],
        ["C", "Relative-speed memory vs gap (by genotype)",
         f"Ac {fmt(mem_n('Acortical'))}, Ct {fmt(mem_n('Control'))}, WT {fmt(mem_n('Wildtype'))}",
         "animals (bootstrap ratio CI)", "Same cohorts as Fig 5 B/C"],
        ["D", "First-traverse memory example trajectories", "1", "1 example animal", "Illustrative"],
    ])

    # S11 — ac_bc_supp.pdf and S12 — ac_d_supp.pdf (split from the former ac_bcd_supp.pdf)
    # These panels plot raw per-animal arrays (mean over animals); rows = animals attempted,
    # but animals with zero traverses are all-NaN and drop out of the mean. Show both.
    def att_dat(array_key, fit_base):
        a, d = n_array(array_key), n_fit_input(fit_base)
        return fmt(a) if (a is None or a == d) else f"{a} ({fmt(d)} w/ traverse data)"
    lines += ["### Fig S11 — `ac_bc_supp.pdf` (acortical generalization in B, C)", "",
              f"_6 panels comparing First A / Repeat A / Mask B / Mask C (acortical raw arrays): "
              f"reward intervals, sorties, duration, and the turn / corridor / tile error rates. "
              f"n animals (attempted; with-traverse-data where it differs): "
              f"First A {att_dat('Acortical Mask A duration', 'Acortical A duration')}, "
              f"Repeat A {att_dat('Acortical Mask A repeat traverse duration', 'Acortical A repeat Gen duration')}, "
              f"Mask B {att_dat('Acortical Mask B traverse duration', 'Acortical B Gen duration')}, "
              f"Mask C {att_dat('Acortical Mask C traverse duration', 'Acortical C Gen duration')}._", ""]
    lines += ["### Fig S12 — `ac_d_supp.pdf` (acortical generalization in D + route similarity)", "",
              f"_6 panels. A-C: Mask-D learning per genotype (matches Fig 5G's learner cohort): Ac "
              f"{att_dat('Acortical Mask D traverse duration', 'Acortical D Gen duration')} (successful learners), "
              f"Ct {att_dat('Control Mask D traverse duration', 'Control D duration')}, "
              f"WT {att_dat('Wildtype D duration', 'Wildtype D duration')}. "
              f"D: acortical route similarity, n = animals scored in all three similarity groups "
              f"(listwise-complete rows of `Acortical D average traverse similarity`). "
              f"E: per-genotype mean similarity, n = animals with any similarity data. "
              f"F: a single example acortical mouse (`config.ACORTICAL_D_SIMILARITY_EXAMPLE_ID`), "
              f"n = 1, illustrative._", ""]

    # S13 — error_propagation_supp.pdf
    n_ep = n_bypos("Wildtype A corridor error by position")
    lines += ["### Fig S13 — `error_propagation_supp.pdf` (error propagation vs model-free RL)", ""] + table([
        ["A", "Corridor error by position (animals, out/home)", fmt(n_ep), "animals (per-position mean)",
         "Wildtype Mask A cohort (from `gen_error_propagation`)"],
        ["B", "Corridor error staircase (model-free RL)", "n/a", "—", "RL model (no animals)"],
        ["C", "Turn/hole error by position (animals)", fmt(n_ep), "animals (per-position mean)", "Same cohort"],
        ["D", "Turn error staircase (model-free RL)", "n/a", "—", "RL model (no animals)"],
    ])
    if n_ep is not None and n_wa is not None and n_ep != n_wa:
        lines += [f"_Note: S12 wildtype Mask A cohort n={n_ep} vs main-figure n={n_wa}; "
                  f"`gen_error_propagation.py` selects the cohort independently and was not "
                  f"updated by the two-day mouse-exclusion filter. Consider aligning it if the "
                  f"euthanised mouse should be excluded here too._", ""]

    # S14 — acortical_ef_supp.pdf
    lines += ["### Fig S14 — `acortical_ef_supp.pdf` (acortical E->F generalization)", ""] + table([
        ["A", "Mask E & F top views", "n/a", "—", "Schematic (geometry)"],
        ["B", "Example Mask E traverses", "1", "1 example animal", "Illustrative"],
        ["C", "Reward intervals (E vs F)", f"E {fmt(n_array('Acortical Mask E reward intervals'))}, "
         f"F {fmt(n_array('Acortical Mask F reward intervals'))}", "animals (mean +/- shade)", "Acortical"],
        ["D", "Sortie counts (E vs F)", f"E {fmt(n_array('Acortical Mask E sortie counts'))}, "
         f"F {fmt(n_array('Acortical Mask F sortie counts'))}", "animals (mean +/- shade)", "Acortical"],
        ["E", "Duration curve fits (Ac E / Ac F / Ct E)",
         f"AcE {fmt(n_fit_input('Acortical Mask E Gen duration'))}, AcF {fmt(n_fit_input('Acortical F Gen duration'))}, "
         f"CtE {fmt(n_fit_input('Control Mask E Gen duration'))}", "animals (bootstrap CI)", ""],
        ["F", "Turn-error curve fits (Ac E / Ac F / Ct E)",
         f"AcE {fmt(n_fit_input('Acortical Mask E Gen turn error rate'))}, AcF {fmt(n_fit_input('Acortical F Gen turn error rate'))}, "
         f"CtE {fmt(n_fit_input('Control Mask E Gen turn error rate'))}", "animals (bootstrap CI)", ""],
    ])

    # S15 — olfaction.pdf
    lines += ["### Fig S15 — `olfaction.pdf` (tray-swap / olfaction)", ""] + table([
        ["A", "Swap-maze schematic", "n/a", "—", "Schematic"],
        ["B", "Relative turn-error (post/pre first-10, by direction)",
         fmt(n_fit_input("Wildtype Pre-swap turn error rate outbound")), "animals (bootstrap ratio CI)",
         "O-A-A swap cohort"],
        ["C", "Pre/post-swap turn-error curve fits (out/home)",
         fmt(n_fit_input("Wildtype Pre-swap turn error rate outbound")), "animals (bootstrap CI)",
         "Fit per direction/condition"],
        ["D", "Reward raster aligned to swap", fmt(n_unique("Swap example manifest", "pair_idx")), "animals", "All swap-cohort animals"],
    ])

    # S16 — north_supp.pdf
    lines += ["### Fig S16 — `north_supp.pdf` (navigation starting from O; cage relocation)", ""] + table([
        ["A", "Cage-relocation schematic", "n/a", "—", "Schematic"],
        ["B", "Mask A (West) sorties per journey", fmt(n_array("Wildtype A sortie count by direction")),
         "animals (paired Wilcoxon)", "West Mask A cohort; listwise-deleted for pairing"],
        ["C", "North (pooled) sorties per journey", fmt(n_array("Single north pooled sortie count by direction")),
         "sessions (paired Wilcoxon)", "North cohort = 6 animals; rows pooled across sessions (pseudoreplicated)"],
        ["D", "Mask A (West) turn error by direction", fmt(n_wa), "animals (direction means)", "West cohort"],
        ["E", "Mask A (North) turn error by direction", fmt(n_array("Single north Day 2 A traverse turn error rate")),
         "animals (direction means)", "North Day-2 A cohort"],
    ])

    return lines


def crosscheck_section():
    lines = ["## Cross-checks (array shape[0] vs fit-input Animal count)", ""]
    if not crosschecks:
        lines.append("_No panels had both an array and a fit-input to compare._")
        lines.append("")
        return lines
    lines.append("| Panel | Array key | shape[0] | Fit-input base | Animal.nunique | Status |")
    lines.append("|---|---|---|---|---|---|")
    for panel, akey, a, fb, f, status in crosschecks:
        lines.append(f"| {panel} | {akey} | {a} | {fb} | {f} | {status} |")
    lines.append("")
    lines.append("Additional independent anchors: `plot_ac_rapid.py:122` states "
                 "Acortical A n=4, Wildtype n=25, Control n=3; producer stdout prints the "
                 "acortical/control Mask-A counts at selection time.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Assemble and write the report.
# ---------------------------------------------------------------------------
def main():
    md = ["# Per-panel animal counts (n) — main & supplementary figures", "",
          "Generated by `scripts/report_panel_n.py` (read-only) from the regenerated "
          "`data/figure_data/` artifacts. n = number of distinct animals; the "
          "*Statistical unit* column states what the error bars / CIs are computed over. "
          "Counts only — no animal identifiers.", ""]
    md += subjects_section()
    md += ["## Per-figure panels", ""]
    md += figure1()
    md += figure2()
    md += figure3()
    md += figure4()
    md += figure5()
    md += figure6()
    md += build_supplementary()
    md += crosscheck_section()

    out_dir = config.DATA_DIR.parent / "docs"  # repo-root/docs
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "panel_n_report.md"
    with open(out_path, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote {out_path}")
    # Console summary of any divergences for the operator (array vs fit-input;
    # array>fit-input is expected when some animals complete 0 traverses).
    div = [c for c in crosschecks if c[-1] != "OK"]
    print(f"Cross-checks: {len(crosschecks)} compared, {len(div)} explained divergence(s).")
    for c in div:
        print("  DIVERGENCE (explained):", c[0], "->", c[-1])


if __name__ == "__main__":
    main()
