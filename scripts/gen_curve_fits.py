"""
Centralised bootstrap curve fitting (and derived ratios), decoupled from data extraction.

Each producer ``gen_*.py`` script saves a ``"<base> fit input"`` payload (via
``utils.save_curve_fit_input``) holding the tidy fit frame plus the metadata
needed to fit it (``data_type`` selects the parameter spec from
``config.CURVE_FIT_SPECS``; ``x_grid_max`` sets the bootstrap-curve x-range).
This script discovers every such payload and produces the matching
``"<base> fit results"`` (+ raw ``"<base> bootstrap params"``) with a single
shared RNG seed — so fits can be re-run without re-loading heavy Session objects,
and all CIs are reproducible.

It also computes related derived ratio tables, kept here so all bootstrap-fit
outputs come from one place:
  * the **two-day Day-2 / Day-1 within-subject parameter ratios** from the paired
    (shared-resample) bootstrap params written by ``gen_wildtype_two_day_data.py``;
  * the **cross-genotype Mask-A parameter ratios** — Acortical / Control and
    Acortical / Wildtype (independent cohorts) — from the Acortical/Control fits
    produced in this run and the Wildtype (BL6J) Day-1 Mask-A two-day fit.

Saved keys
----------
"<base> fit results"                : (bs, ds, summary_df, bootstrap_curves) fitted-curve tuple (.pkl), per data base.
"<base> bootstrap params"           : raw per-iteration bootstrap parameter draws (.parquet).
"<base> per-animal params"          : per-animal fit params (.parquet; Control A only).
"Wildtype two day {metric} param ratios" / "Wildtype two day param ratios"       : Day2/Day1 within-subject curve-derived ratio CIs.
"Wildtype two day {metric} mask param ratios" / "Wildtype day21 {metric} mask BC param ratios" : per-mask / Day2.1-BC ratio tables.
"Wildtype two day {metric} ratio robustness" / "Wildtype two day ratio robustness"           : model-free late-window robustness cross-check.
"Acortical A {metric} genotype param ratios" / "Acortical generalization ... " / "Mask D {metric} genotype param ratios" : cross-genotype ratio tables.
All ratio tables share the tidy schema in docs/ratio_ci_method.md; see docs/data_contracts.md §12.

Run AFTER the producer scripts (they write the inputs) and BEFORE plotting.

Usage
-----
    python gen_curve_fits.py                      # write fit-results, bootstrap params,
                                                  #   and two-day ratio tables
    python gen_curve_fits.py --seed 0             # bootstrap seed (default 0)
    python gen_curve_fits.py --no-overwrite       # keep existing caches
"""
import argparse
import warnings
import numpy as np
import pandas as pd
import config
from manhattan_maze import utils
from manhattan_maze.io import CURVE_FIT_INPUT_SUFFIX

# Day-1 reference key for the two-day within-subject ratios (see get_two_day_data_df).
TWO_DAY_REFERENCE_KEY = (1, "A")
# Day-2 sessions (get_two_day_data_df numbers them 2..5 = Day2-1..Day2-4) used for the
# per-mask "settled" summary — focus on Day2-2/2-3/2-4 (drop the Day2-1 re-exposure).
TWO_DAY_LATE_SESSIONS = (3, 4, 5)
TWO_DAY_MASKS = ("A", "B", "C")
# Day2-1 session index (= Session 2) for the within-session cross-mask Mask C / Mask B ratio.
TWO_DAY_FIRST_SESSION = 2


# --- Shared parameter-ratio estimator policy -----------------------------------
# Every ratio forest panel (two-day, cross-mask, cross-genotype) applies one
# consistent policy, so the four builders below all route through these helpers:
#   * asymptote (param_names[0], D_infty/E_infty): the t→∞ value is unidentifiable
#     — the exponential's slow decay never converges within the observed traverses,
#     so the fit rails it to a bound and the raw ratio CI is severely skewed. It is
#     replaced by a data-anchored *late-performance* value: the fitted curve read
#     at an in-range traverse t_late (see late_performance_samples), reported *_late.
#   * learning rate (param_names[2], delta/epsilon): a fraction of draws rail to a
#     bound and inflate the upper CI, so those draws are NaN-masked before the ratio.
#   * initial value (param_names[1], D_0/E_0): well-identified, used as-is.
# See docs/ratio_ci_method.md.

def _nan_saturated(arr, lb, ub, eps_frac=1e-3):
    """
    Return a float copy of ``arr`` with bound-saturated draws set to NaN.

    Bound-saturated bootstrap draws pile up at a fit boundary and inflate/skew the
    ratio CI. NaN-masking (rather than dropping rows) preserves iteration
    alignment, so the same helper serves both pairwise ratios and cross-group
    summaries — ``bootstrap_ratio_ci`` / ``bootstrap_summary_ratio_ci`` already
    drop the non-finite ratios these NaNs produce.
    """
    arr = np.array(arr, dtype=float)  # copy; do not mutate caller's array
    eps = eps_frac * (ub - lb)
    arr[(np.abs(arr - lb) <= eps) | (np.abs(arr - ub) <= eps)] = np.nan
    return arr


def _saturation_fraction(arr, lb, ub, eps_frac=1e-3):
    """
    Fraction of finite draws sitting within ``eps_frac*(ub-lb)`` of either bound.

    Mirrors the mask :func:`_nan_saturated` applies, but *counts* rather than masks,
    so every ratio row can report how boundary-railed its underlying fit parameter
    was — the motivation for the late-performance/NaN-mask policy (see
    docs/ratio_ci_method.md). Pre-existing NaNs (failed-fit iterations) are excluded
    from the denominator so this measures saturation among converged draws.
    """
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    eps = eps_frac * (ub - lb)
    at_bound = (np.abs(finite - lb) <= eps) | (np.abs(finite - ub) <= eps)
    return float(at_bound.mean())


def _ratio_dropped_fraction(a, b):
    """
    Fraction of paired iterations dropped as non-finite when forming ratio ``a/b``.

    Replicates the truncate-to-shorter + ``np.isfinite`` drop of
    ``bootstrap_ratio_ci`` so the reported ``mask_frac`` matches the iterations that
    CI actually used (failed-fit NaNs, saturation NaNs, and zero-denominator infs).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    return float((~np.isfinite(a[:n] / b[:n])).mean())


def _summary_dropped_fraction(numerators, ref, reduce=np.nanmedian):
    """
    Fraction of summary iterations dropped as non-finite, mirroring
    ``bootstrap_summary_ratio_ci`` (per-group ratios → reduce across groups →
    ``np.isfinite`` drop).
    """
    ref = np.asarray(ref, dtype=float)
    rows = []
    for num in numerators:
        num = np.asarray(num, dtype=float)
        n = min(len(num), len(ref))
        rows.append(num[:n] / ref[:n])
    if not rows:
        return np.nan
    n = min(len(r) for r in rows)
    if n == 0:
        return np.nan
    with warnings.catch_warnings():  # all-NaN iterations reduce to NaN (counted as dropped)
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = reduce(np.vstack([r[:n] for r in rows]), axis=0)
    return float((~np.isfinite(summary)).mean())


def _ratio_samples(params_df, param, param_names, param_bounds, t_late):
    """
    Per-iteration sample array for one parameter under the shared estimator policy:
    asymptote → in-range late-performance value; rate → NaN-masked at its bounds;
    initial value → raw draws.
    """
    asymptote, _initial, rate = param_names
    if param == asymptote:
        return utils.late_performance_samples(params_df, param_names, t_late)
    samples = params_df[param].to_numpy()
    if param == rate:
        samples = _nan_saturated(samples, *param_bounds[param])
    return samples


def _out_param(param, param_names):
    """Display name: the asymptote row is reported as late performance (``*_late``)."""
    return param.replace("_infty", "_late") if param == param_names[0] else param


def _saturation_cols(param, param_bounds, num_df, den_dfs):
    """
    ``(sat_frac_num, sat_frac_den)`` for one ratio row: the boundary-saturation
    fraction of the *raw* fit-parameter draws (``param`` at ``param_bounds[param]``)
    for the numerator and denominator. For the asymptote this is the ``X_infty``
    saturation that motivates the ``X_late`` substitution; for the rate it is the
    delta/epsilon at-bound fraction that gets NaN-masked; for the initial value ≈0.
    ``num_df`` is one frame; ``den_dfs`` a list (averaged for summary rows).
    """
    lb, ub = param_bounds[param]
    sat_num = _saturation_fraction(num_df[param].to_numpy(), lb, ub)
    sat_den = float(np.nanmean([_saturation_fraction(d[param].to_numpy(), lb, ub) for d in den_dfs]))
    return sat_num, sat_den


def _ratio_row(num_df, den_df, param, param_names, param_bounds, t_late, require_aligned):
    """
    ``(out_param, est, lo, hi, sat_frac_num, sat_frac_den, mask_frac)`` for one
    pairwise parameter ratio num/den. The last three are diagnostics: the raw-param
    boundary-saturation fraction of each side and the fraction of paired ratio
    iterations dropped as non-finite (see docs/ratio_ci_method.md).
    """
    a = _ratio_samples(num_df, param, param_names, param_bounds, t_late)
    b = _ratio_samples(den_df, param, param_names, param_bounds, t_late)
    est, lo, hi = utils.bootstrap_ratio_ci(a, b, require_aligned=require_aligned)
    sat_num, sat_den = _saturation_cols(param, param_bounds, num_df, [den_df])
    return _out_param(param, param_names), est, lo, hi, sat_num, sat_den, _ratio_dropped_fraction(a, b)


def _summary_ratio_row(num_dfs, den_df, param, param_names, param_bounds, t_late, require_aligned=True):
    """
    ``(out_param, est, lo, hi, sat_frac_num, sat_frac_den, mask_frac)`` for a summary
    ratio reduced across several numerator groups against one denominator.
    ``sat_frac_num`` is averaged over the numerator groups. ``require_aligned`` is
    ``True`` for within-subject (shared-resample) denominators and ``False`` for an
    independent-cohort denominator (cross-genotype), which truncates to the shorter.
    """
    numerators = [_ratio_samples(df, param, param_names, param_bounds, t_late) for df in num_dfs]
    ref = _ratio_samples(den_df, param, param_names, param_bounds, t_late)
    est, lo, hi = utils.bootstrap_summary_ratio_ci(numerators, ref, require_aligned=require_aligned)
    lb, ub = param_bounds[param]
    sat_num = float(np.nanmean([_saturation_fraction(df[param].to_numpy(), lb, ub) for df in num_dfs]))
    sat_den = _saturation_fraction(den_df[param].to_numpy(), lb, ub)
    return (_out_param(param, param_names), est, lo, hi,
            sat_num, sat_den, _summary_dropped_fraction(numerators, ref))


def _ratio_cols(est, lo, hi, sat_num, sat_den, mask_frac):
    """Common ratio + diagnostic columns shared by every builder's row dict."""
    return {"Ratio": est, "CI_lower": lo, "CI_upper": hi,
            "sat_frac_num": sat_num, "sat_frac_den": sat_den, "mask_frac": mask_frac}


def _late_window_mean(tidy_df, t_late, w=4):
    """
    Model-free late performance: mean ``Value`` over the trailing window of the last
    ``w`` traverses ending at ``t_late`` (``t_late-w < b <= t_late``), aggregated
    within ``Animal`` first then across animals (matching the animal-level bootstrap).

    A fully data-driven counterpart to the curve-derived ``X_late`` — it never touches
    the fitted parameters, so juxtaposing the two ratios shows whether ``X_late``
    reflects real late performance rather than a fit artifact.
    """
    win = tidy_df[(tidy_df["b"] > t_late - w) & (tidy_df["b"] <= t_late)]
    if win.empty:
        return np.nan
    return float(win.groupby("Animal")["Value"].mean().mean())


def _robustness_row(num_params, den_params, num_tidy, den_tidy,
                    param_names, param_bounds, t_late, w=4):
    """
    One robustness-check row comparing the shipped estimator against simpler baselines:
    the curve-derived ``X_late`` ratio vs a model-free late-window ratio, and the rate
    ratio (median + 97.5th pct) computed with vs without boundary NaN-masking.
    """
    _asymptote, _initial, rate = param_names
    a_late = utils.late_performance_samples(num_params, param_names, t_late)
    b_late = utils.late_performance_samples(den_params, param_names, t_late)
    xlate = utils.bootstrap_ratio_ci(a_late, b_late)[0]
    mf_num, mf_den = _late_window_mean(num_tidy, t_late, w), _late_window_mean(den_tidy, t_late, w)
    modelfree = mf_num / mf_den if mf_den else np.nan
    lb, ub = param_bounds[rate]
    rm = utils.bootstrap_ratio_ci(_nan_saturated(num_params[rate].to_numpy(), lb, ub),
                                  _nan_saturated(den_params[rate].to_numpy(), lb, ub))
    ru = utils.bootstrap_ratio_ci(num_params[rate].to_numpy(), den_params[rate].to_numpy())
    return {"Xlate_ratio": xlate, "modelfree_ratio": modelfree,
            "rate_masked": rm[0], "rate_masked_hi": rm[2],
            "rate_unmasked": ru[0], "rate_unmasked_hi": ru[2]}


def _log_robustness(table, name):
    """Print the robustness comparison (run-log diagnostic)."""
    if table is None or table.empty:
        return
    print(f"[robustness] {name}")
    for _, r in table.iterrows():
        print(f"    {str(r['Comparison']):<16} "
              f"Xlate={r['Xlate_ratio']:.3f} model_free={r['modelfree_ratio']:.3f} | "
              f"rate masked={r['rate_masked']:.2f}(hi {r['rate_masked_hi']:.2f}) "
              f"unmasked={r['rate_unmasked']:.2f}(hi {r['rate_unmasked_hi']:.2f})")


def _pct(x):
    return "  n/a" if x is None or not np.isfinite(x) else f"{x:4.0%}"


def _log_saturation(table, name):
    """Print each row's boundary-saturation and mask fractions (run-log diagnostic)."""
    if table is None or table.empty:
        return
    grp_cols = [c for c in ("Session", "Mask", "Comparison") if c in table.columns]
    print(f"[saturation] {name}")
    for _, r in table.iterrows():
        grp = " ".join(str(r[c]) for c in grp_cols)
        print(f"    {str(r['Parameter']):>9} {grp:<14} "
              f"sat_num={_pct(r['sat_frac_num'])} sat_den={_pct(r['sat_frac_den'])} "
              f"mask={_pct(r['mask_frac'])}")


def compute_two_day_ratio_table(params_dict, metric, param_names, param_bounds, t_late,
                                reference_key=TWO_DAY_REFERENCE_KEY):
    """
    Tidy Day-2/Day-1 within-subject ratio-CI table for every fitted parameter.

    ``params_dict`` is ``{(session, mask): aligned param_samples_df}`` (paired
    shared-resample draws). For each parameter, computes the bootstrap median and
    95% CI of ``R_{s,m} = param_Day2(s,m) / param_Day1`` per Day-2 group, plus a
    ``Session=Mask="all"`` summary row (median ratio across the Day-2 groups per
    iteration), under the shared estimator policy (asymptote → ``*_late``, rate →
    NaN-saturated; see the module helpers above). Returns columns
    ``[Metric, Parameter, Session, Mask, Ratio, CI_lower, CI_upper]``.

    Parameters
    ----------
    param_names : sequence of str
        Fit-parameter names in model-argument order ``[asymptote, initial, rate]``.
    param_bounds : dict
        ``{name: (lower, upper)}`` fit bounds, used for the rate saturation mask.
    t_late : float
        In-range traverse index at which to read late performance (choose inside
        every compared group's observed range, e.g. the min observed max-traverse).
    """
    ref_df = params_dict[reference_key]
    day2_keys = [k for k in params_dict if k != reference_key]
    rows = []
    for param in param_names:
        for (s, m) in day2_keys:
            out_param, *stats = _ratio_row(params_dict[(s, m)], ref_df, param,
                                           param_names, param_bounds, t_late, True)
            rows.append({"Metric": metric, "Parameter": out_param, "Session": str(s), "Mask": m,
                         **_ratio_cols(*stats)})
        # across-combinations summary: median over the Day-2 groups per iteration
        out_param, *stats = _summary_ratio_row(
            [params_dict[k] for k in day2_keys], ref_df, param, param_names, param_bounds, t_late)
        rows.append({"Metric": metric, "Parameter": out_param, "Session": "all", "Mask": "all",
                     **_ratio_cols(*stats)})
    return pd.DataFrame(rows)


def compute_two_day_mask_ratio_table(params_dict, metric, param_names, param_bounds, t_late,
                                     sessions=TWO_DAY_LATE_SESSIONS, masks=TWO_DAY_MASKS,
                                     reference_key=TWO_DAY_REFERENCE_KEY):
    """
    Per-mask Day-2/Day-1 within-subject summary ratio over the "settled" Day-2 sessions.

    Like the ``Session=Mask="all"`` row of :func:`compute_two_day_ratio_table`, but
    instead of medianing over *all* Day-2 groups it produces one summary row per mask:
    for each parameter and each mask in ``masks``, the bootstrap median ratio is taken
    across only the late Day-2 ``sessions`` (default Day2-2/2-3/2-4) of that mask,
    still relative to the Day-1 ``reference_key`` so the comparison stays within-subject.
    Returns the same columns as :func:`compute_two_day_ratio_table`, with
    ``Session="late"`` and ``Mask`` set to the mask. Applies the shared estimator
    policy (asymptote → ``*_late``, rate → NaN-saturated); see the module helpers.
    """
    ref_df = params_dict[reference_key]
    # Which masks actually have late-session draws — warn once for any that don't.
    masks_with_data = {m: [s for s in sessions if (s, m) in params_dict] for m in masks}
    for mask, avail in masks_with_data.items():
        if not avail:
            print(f"[ratios warn] {metric}: mask {mask!r} has no late sessions "
                  f"{sessions} in params — no summary row written for it.")
    rows = []
    for param in param_names:
        for mask, avail in masks_with_data.items():
            if not avail:
                continue
            out_param, *stats = _summary_ratio_row(
                [params_dict[(s, mask)] for s in avail], ref_df, param,
                param_names, param_bounds, t_late)
            rows.append({"Metric": metric, "Parameter": out_param, "Session": "late", "Mask": mask,
                         **_ratio_cols(*stats)})
    return pd.DataFrame(rows)


def compute_day21_mask_cb_ratio_table(params_dict, metric, param_names, param_bounds, t_late,
                                      session=TWO_DAY_FIRST_SESSION,
                                      numerator_mask="C", denominator_mask="B"):
    """
    Within-session cross-mask parameter-ratio CI table for Day2-1.

    For each fitted parameter, computes the bootstrap median and 95% CI of
    ``param_{numerator_mask} / param_{denominator_mask}`` at the Day2-1 ``session``
    (default Session 2), using the paired shared-resample draws so the cross-mask
    ratio is within-subject. Applies the shared estimator policy (asymptote →
    ``*_late``, rate → NaN-saturated; see the module helpers).
    Returns columns ``[Metric, Parameter, Session, Mask, Ratio, CI_lower, CI_upper]``
    with ``Mask`` set to ``"{numerator_mask}/{denominator_mask}"``.
    """
    num_key, den_key = (session, numerator_mask), (session, denominator_mask)
    if num_key not in params_dict or den_key not in params_dict:
        missing = [k for k in (num_key, den_key) if k not in params_dict]
        print(f"[ratios warn] {metric}: Day2-1 cross-mask ratio skipped — "
              f"missing group(s) {missing} in params.")
        return pd.DataFrame()
    num_df, den_df = params_dict[num_key], params_dict[den_key]
    rows = []
    for param in param_names:
        out_param, *stats = _ratio_row(num_df, den_df, param, param_names,
                                       param_bounds, t_late, True)
        rows.append({"Metric": metric, "Parameter": out_param, "Session": str(session),
                     "Mask": f"{numerator_mask}/{denominator_mask}",
                     **_ratio_cols(*stats)})
    return pd.DataFrame(rows)


def compute_independent_ratio_table(numerator_df, denominator_df, metric, comparison,
                                    param_names, param_bounds, t_late):
    """
    Parameter-ratio CI table between two independently-bootstrapped fits.

    For every fitted parameter present in both frames, computes the bootstrap median
    and 95% CI of ``param_numerator / param_denominator`` — e.g. cross-genotype
    (Acortical / Control) or within-cohort across-mask (Repeat A / First A). The two
    sets of draws come from **separate** bootstrap procedures (not a shared resample),
    so they are paired iteration-wise only as a Monte-Carlo ratio distribution — this
    is NOT a within-subject ratio, hence ``bootstrap_ratio_ci`` is called with
    ``require_aligned=False`` (lengths may differ between the two procedures;
    it truncates to the shorter, which is fine for independent draws).
    Applies the shared estimator policy (asymptote → ``*_late``, rate →
    NaN-saturated; see the module helpers). Returns columns
    ``[Metric, Parameter, Comparison, Ratio, CI_lower, CI_upper]``; ``Comparison``
    (e.g. ``"Acortical/Control"``) distinguishes the rows.
    """
    rows = []
    for param in param_names:
        if param not in numerator_df.columns or param not in denominator_df.columns:
            continue
        out_param, *stats = _ratio_row(numerator_df, denominator_df, param,
                                       param_names, param_bounds, t_late, False)
        rows.append({"Metric": metric, "Parameter": out_param, "Comparison": comparison,
                     **_ratio_cols(*stats)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Centralised curve fitting from saved fit inputs")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (default 0).")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("-ow", "--overwrite", action=argparse.BooleanOptionalAction, default=True,
                        help="Overwrite existing fit-results caches (default True).")
    args = parser.parse_args()

    save_dir = config.SAVE_DIR

    # data_type -> (params_name, p0, lower_bounds, upper_bounds) from the shared specs.
    SPEC_BY_DATA_TYPE = {
        data_type: (params_name, p0, lb, ub)
        for data_type, params_name, _params_latex, p0, lb, ub in config.CURVE_FIT_SPECS
    }


    def names_and_bounds(metric):
        """``(param_names, {name: (lower, upper)})`` for a metric — the arguments the
        ratio-table builders need for the shared late-performance / saturation policy."""
        params_name, _p0, lb, ub = SPEC_BY_DATA_TYPE[metric]
        return params_name, {n: (l, u) for n, l, u in zip(params_name, lb, ub)}

    all_data = utils.load_all_figure_data(save_dir)
    input_keys = sorted(k for k in all_data if k.endswith(CURVE_FIT_INPUT_SUFFIX))
    if not input_keys:
        raise SystemExit(
            f"No '*{CURVE_FIT_INPUT_SUFFIX}' payloads found in {save_dir}. "
            "Run the producer gen_*.py scripts first (with --overwrite)."
        )

    print(f"Found {len(input_keys)} curve-fit inputs in {save_dir}")
    print(f"seed={args.seed}, bootstrap_iterations={args.bootstrap_iterations}, overwrite={args.overwrite}\n")

    # Bootstrap params produced in THIS run, keyed by base (for cross-genotype ratios below;
    # they are not in all_data, which was loaded once before this loop wrote them).
    produced_params = {}
    # Observed max-traverse per base, for the in-range late-performance t_late of the
    # independent (cross-genotype / generalization) ratios below.
    produced_tmax = {}
    # Raw tidy fit frames (columns b/Value/Animal), retained for the model-free
    # late-window robustness check on the independent-cohort ratios below.
    produced_data = {}
    for key in input_keys:
        base = key[: -len(CURVE_FIT_INPUT_SUFFIX)]
        payload = all_data[key]
        data_df = payload["data_df"]
        data_type = payload["data_type"]
        x_grid_max = payload["x_grid_max"]

        if data_type not in SPEC_BY_DATA_TYPE:
            print(f"[skip] {base}: unknown data_type {data_type!r} (not in CURVE_FIT_SPECS)")
            continue
        params_name, p0, lower_bounds, upper_bounds = SPEC_BY_DATA_TYPE[data_type]

        bs, ds, summary_df, bootstrap_curves, param_samples = utils.fit_traverse_data_df_with_bootstrap(
            data_df, params_name=params_name, p0=p0,
            lower_bounds=lower_bounds, upper_bounds=upper_bounds,
            x_grid=np.linspace(1, x_grid_max, 100),
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
            return_param_samples=True,
        )

        utils.save_modular_data(f"{base} fit results", (bs, ds, summary_df, bootstrap_curves), save_dir, overwrite=args.overwrite)
        # Raw per-iteration parameter draws, for cross-group ratio CIs (bootstrap_ratio_ci).
        utils.save_modular_data(f"{base} bootstrap params", param_samples, save_dir, overwrite=args.overwrite)
        produced_params[base] = param_samples
        produced_tmax[base] = int(np.nanmax(bs))
        produced_data[base] = data_df

        # n=3 control cohort: also fit each animal independently (no bootstrap). Its
        # animal-level bootstrap CI is degenerate (~6 distinct resamples), so the
        # First-Mask-A supplement shows per-animal parameter points instead — matching
        # the raw-trace control treatment in plot_ac_rapid.py.
        if base in ("Control A duration", "Control A turn error rate"):
            per_animal_df = utils.fit_per_animal(
                data_df, p0=p0, lower_bounds=lower_bounds,
                upper_bounds=upper_bounds, params_name=params_name)
            utils.save_modular_data(f"{base} per-animal params", per_animal_df, save_dir, overwrite=args.overwrite)

    # --- Two-day Day-2 / Day-1 within-subject parameter ratios -------------------
    # Derived from the paired (shared-resample) bootstrap params that
    # gen_wildtype_two_day_data.py writes; turned into per-(session, mask) and
    # across-combinations ratio CIs (one table per metric, plus a combined table).
    df_list = []
    mask_df_list = []
    cb_df_list = []
    rob_df_list = []
    for metric in SPEC_BY_DATA_TYPE:
        params_key = f"Wildtype two day {metric} bootstrap params"
        if params_key not in all_data:
            print(f"[ratios skip] {params_key} not found — run gen_wildtype_two_day_data.py first.")
            continue
        params_dict = all_data[params_key]
        if TWO_DAY_REFERENCE_KEY not in params_dict:
            print(f"[ratios skip] {metric}: reference {TWO_DAY_REFERENCE_KEY} missing from {params_key}.")
            continue
        # Late-performance eval point: the min observed max-traverse across all groups,
        # so t_late is in-range (interpolated, not extrapolated) for every group.
        fit_key = f"Wildtype two day {metric} fit results"
        if fit_key not in all_data:
            print(f"[ratios skip] {metric}: {fit_key} not found — needed for late-performance t_late.")
            continue
        t_late = min(int(np.nanmax(bs)) for (bs, *_rest) in all_data[fit_key].values())
        params_name, param_bounds = names_and_bounds(metric)
        table = compute_two_day_ratio_table(params_dict, metric, params_name, param_bounds, t_late)
        utils.save_modular_data(f"Wildtype two day {metric} param ratios", table, save_dir, overwrite=args.overwrite)
        _log_saturation(table, f"Wildtype two day {metric} param ratios")
        df_list.append(table)

        # Per-mask summary over the settled Day-2 sessions (Day2-2/2-3/2-4), vs Day-1 A.
        mask_table = compute_two_day_mask_ratio_table(params_dict, metric, params_name, param_bounds, t_late)
        utils.save_modular_data(f"Wildtype two day {metric} mask param ratios", mask_table, save_dir, overwrite=args.overwrite)
        _log_saturation(mask_table, f"Wildtype two day {metric} mask param ratios")
        mask_df_list.append(mask_table)

        # Day2-1 within-session cross-mask ratio: Mask B parameter / Mask C parameter.
        cb_table = compute_day21_mask_cb_ratio_table(params_dict, metric, params_name, param_bounds, t_late,
                                                     numerator_mask="B", denominator_mask="C")
        if not cb_table.empty:
            utils.save_modular_data(f"Wildtype day21 {metric} mask BC param ratios", cb_table, save_dir, overwrite=args.overwrite)
            _log_saturation(cb_table, f"Wildtype day21 {metric} mask BC param ratios")
            cb_df_list.append(cb_table)

        # Robustness: curve-derived X_late vs model-free late-window ratio, and rate ratio
        # with/without boundary masking, for every Day-2 group vs the Day-1 A reference.
        tidy = all_data.get(f"Wildtype two day {metric} tidy")
        if tidy is None:
            print(f"[robustness skip] {metric}: 'Wildtype two day {metric} tidy' not found — "
                  "re-run gen_wildtype_two_day_data.py to enable the model-free late-window check.")
        else:
            ref_df = params_dict[TWO_DAY_REFERENCE_KEY]
            day2_keys = [k for k in params_dict if k != TWO_DAY_REFERENCE_KEY]
            ref_tidy = tidy[(tidy["Session"] == TWO_DAY_REFERENCE_KEY[0]) & (tidy["Mask"] == TWO_DAY_REFERENCE_KEY[1])]
            rob_rows = [{"Metric": metric, "Comparison": f"Day2-{s - 1} {m}/Day1 A",
                         **_robustness_row(params_dict[(s, m)], ref_df,
                                           tidy[(tidy["Session"] == s) & (tidy["Mask"] == m)], ref_tidy,
                                           params_name, param_bounds, t_late)}
                        for (s, m) in day2_keys]
            rob_table = pd.DataFrame(rob_rows)
            utils.save_modular_data(f"Wildtype two day {metric} ratio robustness", rob_table, save_dir, overwrite=args.overwrite)
            _log_robustness(rob_table, f"Wildtype two day {metric} ratio robustness")
            rob_df_list.append(rob_table)

    if df_list:
        utils.save_modular_data("Wildtype two day param ratios", pd.concat(df_list), save_dir, overwrite=args.overwrite)
    if mask_df_list:
        utils.save_modular_data("Wildtype two day mask param ratios", pd.concat(mask_df_list), save_dir, overwrite=args.overwrite)
    if cb_df_list:
        # Combined cross-mask B/C table consumed by plot_day2.py (panel G).
        utils.save_modular_data("Wildtype day21 mask BC param ratios", pd.concat(cb_df_list, ignore_index=True), save_dir, overwrite=args.overwrite)
    if rob_df_list:
        utils.save_modular_data("Wildtype two day ratio robustness", pd.concat(rob_df_list, ignore_index=True), save_dir, overwrite=args.overwrite)

    # --- Cross-genotype Mask-A parameter ratios: Acortical / Control and Acortical / Wildtype ---
    # Independent-cohort ratios of the fitted Mask-A learning-curve parameters. Acortical and
    # Control come from this run's own fit loop (produced_params); Wildtype is the BL6J Day-1
    # Mask-A fit from gen_wildtype_two_day_data.py (the (1, "A") entry of its two-day params).
    cg_df_list = []
    for metric in SPEC_BY_DATA_TYPE:
        acortical = produced_params.get(f"Acortical A {metric}")
        if acortical is None:
            print(f"[ratios skip] 'Acortical A {metric}' params not produced — run gen_acortical_learning.py first.")
            continue
        params_name, param_bounds = names_and_bounds(metric)
        ac_tmax = produced_tmax[f"Acortical A {metric}"]
        numerators = []  # (label, numerator_df, numerator_tmax)
        control = produced_params.get(f"Control A {metric}")
        if control is not None:
            numerators.append(("Control/Acortical", control, produced_tmax[f"Control A {metric}"]))
        else:
            print(f"[ratios warn] {metric}: 'Control A {metric}' params missing — skipping Control/Acortical.")
        wt_dict = all_data.get(f"Wildtype two day {metric} bootstrap params")
        wt_fit = all_data.get(f"Wildtype two day {metric} fit results")
        if wt_dict is not None and TWO_DAY_REFERENCE_KEY in wt_dict and wt_fit is not None:
            wt_tmax = int(np.nanmax(wt_fit[TWO_DAY_REFERENCE_KEY][0]))
            numerators.append(("Wildtype/Acortical", wt_dict[TWO_DAY_REFERENCE_KEY], wt_tmax))
        else:
            print(f"[ratios warn] {metric}: Wildtype Day-1 Mask-A params missing — skipping Wildtype/Acortical.")

        # t_late per comparison: in-range for both cohorts (min of their max-traverse).
        metric_tables = [compute_independent_ratio_table(num, acortical, metric, label,
                                                         params_name, param_bounds, min(num_tmax, ac_tmax))
                         for label, num, num_tmax in numerators]
        if metric_tables:
            table = pd.concat(metric_tables, ignore_index=True)
            utils.save_modular_data(f"Acortical A {metric} genotype param ratios", table, save_dir, overwrite=args.overwrite)
            _log_saturation(table, f"Acortical A {metric} genotype param ratios")
            cg_df_list.append(table)

    if cg_df_list:
        utils.save_modular_data("Acortical A genotype param ratios", pd.concat(cg_df_list, ignore_index=True), save_dir, overwrite=args.overwrite)


    # --- Acortical generalization Mask ratios: Repeat A / First A, Mask B / First A, Mask C / First A ---
    # Independent-bootstrap ratios of the acortical generalization fits (produced this run) to the
    # un-generalized First-A reference, for the generalization parameter panels. Acortical only.
    ACORTICAL_GEN_MASKS = [("Repeat A", "Acortical A repeat Gen"),
                           ("Mask B", "Acortical B Gen"),
                           ("Mask C", "Acortical C Gen")]
    gen_df_list = []
    for metric in SPEC_BY_DATA_TYPE:
        first_a = produced_params.get(f"Acortical A {metric}")
        if first_a is None:
            print(f"[ratios skip] 'Acortical A {metric}' params not produced — run gen_acortical_learning.py first.")
            continue
        params_name, param_bounds = names_and_bounds(metric)
        first_a_tmax = produced_tmax[f"Acortical A {metric}"]
        metric_tables = []
        for label, base in ACORTICAL_GEN_MASKS:
            num = produced_params.get(f"{base} {metric}")
            if num is None:
                print(f"[ratios warn] {metric}: '{base} {metric}' params missing — skipping {label}/First A "
                      "(run gen_ac_generalization.py first).")
                continue
            t_late = min(produced_tmax[f"{base} {metric}"], first_a_tmax)
            metric_tables.append(compute_independent_ratio_table(num, first_a, metric, f"{label}/First A",
                                                                 params_name, param_bounds, t_late))
        if metric_tables:
            table = pd.concat(metric_tables, ignore_index=True)
            utils.save_modular_data(f"Acortical generalization {metric} param ratios", table, save_dir, overwrite=args.overwrite)
            _log_saturation(table, f"Acortical generalization {metric} param ratios")
            gen_df_list.append(table)

    if gen_df_list:
        utils.save_modular_data("Acortical generalization param ratios", pd.concat(gen_df_list, ignore_index=True), save_dir, overwrite=args.overwrite)


    # --- Mask-D cross-genotype parameter ratios: Control/Wildtype and Acortical/Wildtype ---
    # Independent-cohort duration ratios of the Mask-D fits, both against the Wildtype cohort (the
    # slow reference), so the forest ratios sit at/below 1. Mask D is duration-only. Denominator is
    # Wildtype (from gen_wildtype_d_data.py); numerators are the Control and Acortical Mask-D fits.
    maskd_df_list = []
    for metric in ["duration"]:  # Mask D has no turn-error-rate fit
        wildtype = produced_params.get(f"Wildtype D {metric}")
        if wildtype is None:
            print(f"[ratios skip] 'Wildtype D {metric}' params not produced — run gen_wildtype_d_data.py first.")
            continue
        params_name, param_bounds = names_and_bounds(metric)
        wt_tmax = produced_tmax[f"Wildtype D {metric}"]
        numerators = []  # (label, numerator_df, numerator_tmax)
        control = produced_params.get(f"Control D {metric}")
        if control is not None:
            numerators.append(("Control/Wildtype", control, produced_tmax[f"Control D {metric}"]))
        else:
            print(f"[ratios warn] {metric}: 'Control D {metric}' params missing — skipping Control/Wildtype.")
        acortical = produced_params.get(f"Acortical D Gen {metric}")
        if acortical is not None:
            numerators.append(("Acortical/Wildtype", acortical, produced_tmax[f"Acortical D Gen {metric}"]))
        else:
            print(f"[ratios warn] {metric}: 'Acortical D Gen {metric}' params missing — skipping Acortical/Wildtype.")

        # t_late per comparison: in-range for both cohorts (min of their max-traverse).
        metric_tables = [compute_independent_ratio_table(num, wildtype, metric, label,
                                                         params_name, param_bounds, min(num_tmax, wt_tmax))
                         for label, num, num_tmax in numerators]
        if metric_tables:
            table = pd.concat(metric_tables, ignore_index=True)
            utils.save_modular_data(f"Mask D {metric} genotype param ratios", table, save_dir, overwrite=args.overwrite)
            _log_saturation(table, f"Mask D {metric} genotype param ratios")
            maskd_df_list.append(table)

    if maskd_df_list:
        utils.save_modular_data("Mask D genotype param ratios", pd.concat(maskd_df_list, ignore_index=True), save_dir, overwrite=args.overwrite)


    # --- Generalization cross-genotype parameter ratios: Control / Acortical, per mask ---
    # Independent-cohort ratios of the Control generalization fits to the Acortical generalization
    # fits, for each generalization mask and both metrics (Comparison = mask label). Distinct from
    # "Acortical generalization param ratios" (which is Acortical-internal Mask/First A).
    GEN_GENOTYPE_MASKS = [("Repeat A", "A repeat Gen"), ("Mask B", "B Gen"), ("Mask C", "C Gen")]
    gen_gt_df_list = []
    for metric in SPEC_BY_DATA_TYPE:
        params_name, param_bounds = names_and_bounds(metric)
        metric_tables = []
        for label, mask_base in GEN_GENOTYPE_MASKS:
            num = produced_params.get(f"Control {mask_base} {metric}")
            den = produced_params.get(f"Acortical {mask_base} {metric}")
            if num is None or den is None:
                print(f"[ratios warn] {metric}: '{mask_base} {metric}' Control or Acortical params "
                      f"missing — skipping {label} Control/Acortical (run gen_ac_generalization.py first).")
                continue
            t_late = min(produced_tmax[f"Control {mask_base} {metric}"],
                         produced_tmax[f"Acortical {mask_base} {metric}"])
            metric_tables.append(compute_independent_ratio_table(num, den, metric, label,
                                                                 params_name, param_bounds, t_late))
        if metric_tables:
            table = pd.concat(metric_tables, ignore_index=True)
            utils.save_modular_data(f"Acortical generalization {metric} genotype param ratios", table, save_dir, overwrite=args.overwrite)
            _log_saturation(table, f"Acortical generalization {metric} genotype param ratios")
            gen_gt_df_list.append(table)

    if gen_gt_df_list:
        utils.save_modular_data("Acortical generalization genotype param ratios", pd.concat(gen_gt_df_list, ignore_index=True), save_dir, overwrite=args.overwrite)


    # --- Generalization Wildtype/Acortical ratios: pooled settled Day-2 Wildtype vs Acortical gen ---
    # Cross-genotype counterpart of the Acortical-denominator generalization table, but the numerator
    # is the Wildtype settled Day-2 re-exposure (Day2-2/2-3/2-4 = TWO_DAY_LATE_SESSIONS), pooled per
    # bootstrap iteration (median across sessions) then ratioed against the Acortical generalization
    # fit for the matching mask. Independent cohorts, so require_aligned=False. Wildtype mask A maps to
    # the Acortical "Repeat A" re-exposure condition; B/C map to Mask B / Mask C.
    WT_GEN_MASKS = [("Repeat A", "A repeat Gen", "A"), ("Mask B", "B Gen", "B"), ("Mask C", "C Gen", "C")]
    wt_gen_df_list = []
    for metric in SPEC_BY_DATA_TYPE:
        wt_bp = all_data.get(f"Wildtype two day {metric} bootstrap params")
        wt_fr = all_data.get(f"Wildtype two day {metric} fit results")
        if wt_bp is None or wt_fr is None:
            print(f"[ratios skip] Wildtype two day {metric} params/results missing — run gen_wildtype_two_day_data.py first.")
            continue
        params_name, param_bounds = names_and_bounds(metric)
        rows = []
        for label, ac_base, wt_mask in WT_GEN_MASKS:
            den = produced_params.get(f"Acortical {ac_base} {metric}")
            if den is None:
                print(f"[ratios warn] {metric}: 'Acortical {ac_base} {metric}' missing — skipping {label} Wildtype/Acortical.")
                continue
            sessions = [s for s in TWO_DAY_LATE_SESSIONS if (s, wt_mask) in wt_bp]
            if not sessions:
                print(f"[ratios warn] {metric}: no settled Day-2 sessions for Wildtype mask {wt_mask!r} — skipping {label}.")
                continue
            num_dfs = [wt_bp[(s, wt_mask)] for s in sessions]
            # t_late in-range for every group being compared (Wildtype sessions + Acortical).
            t_late = min([int(np.nanmax(wt_fr[(s, wt_mask)][0])) for s in sessions]
                         + [produced_tmax[f"Acortical {ac_base} {metric}"]])
            for param in params_name:
                out_param, *stats = _summary_ratio_row(num_dfs, den, param, params_name,
                                                       param_bounds, t_late, require_aligned=False)
                rows.append({"Metric": metric, "Parameter": out_param, "Comparison": label,
                             **_ratio_cols(*stats)})
        if rows:
            table = pd.DataFrame(rows)
            utils.save_modular_data(f"Wildtype generalization {metric} genotype param ratios", table, save_dir, overwrite=args.overwrite)
            _log_saturation(table, f"Wildtype generalization {metric} genotype param ratios")
            wt_gen_df_list.append(table)

    if wt_gen_df_list:
        utils.save_modular_data("Wildtype generalization genotype param ratios", pd.concat(wt_gen_df_list, ignore_index=True), save_dir, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
