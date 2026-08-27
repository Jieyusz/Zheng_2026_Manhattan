"""Animal-level bootstrap fitting, bootstrap curve CIs, and permutation testing.

Split out of utils.py; see docs.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
import warnings
from manhattan_maze.utils import exponential_func

__all__ = ['permutation_test', 'get_bootstrap_curve_confidence_intervals', 'fit_traverse_data_df_with_bootstrap', 'fit_grouped_data_df_with_bootstrap', 'fit_grouped_data_df_with_shared_bootstrap', 'fit_per_animal', 'bootstrap_ratio_ci', 'bootstrap_summary_ratio_ci', 'relative_performance']

def permutation_test(data_dict, n_permutations=1000):
    """
    Two-sample permutation test for difference in means.

    Parameters
    ----------
    data_dict : dict of {any: array-like}
        Must contain exactly two entries.  Keys are group labels; values are
        1-D data arrays.
    n_permutations : int, default 1000
        Number of random permutations.

    Returns
    -------
    float
        Two-tailed p-value: fraction of permutations whose absolute mean
        difference equals or exceeds the observed difference.
    """
    group_names = list(data_dict.keys())
    group_data = [data_dict[name] for name in group_names]

    observed_diff = np.mean(group_data[0]) - np.mean(group_data[1])

    combined_data = np.concatenate(group_data)
    count_greater = 0

    for _ in range(n_permutations):
        np.random.shuffle(combined_data)
        permuted_group_1 = combined_data[:len(group_data[0])]
        permuted_group_2 = combined_data[len(group_data[0]):]
        permuted_diff = np.mean(permuted_group_1) - np.mean(permuted_group_2)

        if abs(permuted_diff) >= abs(observed_diff):
            count_greater += 1

    p_value = count_greater / n_permutations
    return p_value


def get_bootstrap_curve_confidence_intervals(
    bootstrap_results,
    func,
    x_grid,
    alpha=0.05,
):
    """
    Compute bootstrap confidence intervals for the fitted curve.

    Parameters
    ----------
    bootstrap_results : array-like, shape (n_bootstraps, n_params)
        Bootstrap parameter estimates.

    func : callable
        Curve function.

    x_grid : ndarray
        X-values at which to evaluate the curve.

    alpha : float
        Significance level.

    Returns
    -------
    x_grid : ndarray
    curve_ci_lower : ndarray
    curve_ci_upper : ndarray
    """

    bootstrap_predictions = np.array([
        func(x_grid, *params)
        for params in bootstrap_results
    ])

    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)

    curve_ci_lower = np.percentile(
        bootstrap_predictions,
        lower_pct,
        axis=0,
    )

    curve_ci_upper = np.percentile(
        bootstrap_predictions,
        upper_pct,
        axis=0,
    )

    return x_grid, curve_ci_lower, curve_ci_upper


def _drop_nan_xy(b_values, y_values):
    """Return ``(b, y)`` with rows where ``y`` is NaN removed (shared NaN filter)."""
    b_values = np.asarray(b_values)
    y_values = np.asarray(y_values)
    valid = ~np.isnan(y_values)
    return b_values[valid], y_values[valid]


def _build_animal_arrays(data_df):
    """
    Group a tidy fit frame into per-animal ``(b, value)`` arrays, dropping NaN values.

    Parameters
    ----------
    data_df : pd.DataFrame
        Columns ``Animal``, ``b``, ``Value``.

    Returns
    -------
    dict of {str: tuple of ndarray}
        ``{animal: (b, value)}`` — the unit resampled during the animal-level bootstrap.
    """
    animal_arrays = {}
    for animal, grp in data_df.groupby("Animal"):
        animal_arrays[animal] = _drop_nan_xy(grp.b.to_numpy(), grp.Value.to_numpy())
    return animal_arrays


def _run_bootstrap_curve_fits(animal_arrays, func, p0, lower_bounds, upper_bounds,
                              bootstrap_iterations, rng):
    """
    Resample animals with replacement and fit ``func`` once per bootstrap iteration.

    Resampling is at the animal level (whole animals drawn with replacement),
    preserving within-animal structure. Iterations whose fit fails to converge are
    skipped.

    Parameters
    ----------
    animal_arrays : dict of {str: tuple of ndarray}
        Per-animal ``(b, value)`` arrays from :func:`_build_animal_arrays`.
    func : callable
        Model function ``f(x, *params)``.
    p0, lower_bounds, upper_bounds : array-like
        Initial guess and parameter bounds passed to ``scipy.optimize.curve_fit``.
    bootstrap_iterations : int
        Number of resamples to attempt.
    rng : numpy RNG
        ``np.random`` (legacy global) or a ``np.random.Generator``; must provide ``choice``.

    Returns
    -------
    np.ndarray, shape (n_successful, n_params)
        Stacked parameter estimates from converged fits.

    Raises
    ------
    RuntimeError
        If every bootstrap fit failed to converge.

    Warns
    -----
    UserWarning
        If fewer than 90% of iterations converged.
    """
    animals = list(animal_arrays.keys())
    bootstrap_results = []

    for _ in tqdm(range(bootstrap_iterations), desc="Bootstrapping curve fit"):
        sampled_animals = rng.choice(animals, size=len(animals), replace=True)
        bs_boot = np.concatenate([animal_arrays[a][0] for a in sampled_animals])
        ds_boot = np.concatenate([animal_arrays[a][1] for a in sampled_animals])
        try:
            popt_boot, _ = curve_fit(
                func, bs_boot, ds_boot,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000,
            )
            bootstrap_results.append(popt_boot)
        except RuntimeError:
            continue

    n_successful = len(bootstrap_results)
    if n_successful < 0.9 * bootstrap_iterations:
        warnings.warn(
            f"Only {n_successful}/{bootstrap_iterations} bootstrap fits converged."
        )
    bootstrap_array = np.asarray(bootstrap_results)
    if bootstrap_array.shape[0] == 0:
        raise RuntimeError("All bootstrap fits failed.")
    return bootstrap_array


def _bootstrap_param_cis(bootstrap_array, alpha):
    """
    Reduce bootstrap parameter estimates to a median and percentile CI per parameter.

    Returns
    -------
    tuple of ndarray
        ``(median_params, ci_lower, ci_upper)``, each of shape ``(n_params,)``.
    """
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)
    median_params = np.median(bootstrap_array, axis=0)
    ci_lower = np.percentile(bootstrap_array, lower_pct, axis=0)
    ci_upper = np.percentile(bootstrap_array, upper_pct, axis=0)
    return median_params, ci_lower, ci_upper


def _build_summary_df(params_name, median_params, ci_lower, ci_upper, alpha, n_successful):
    """
    Assemble the parameter summary DataFrame (with bootstrap metadata in ``.attrs``).

    Returns
    -------
    pd.DataFrame
        Columns ``Parameter``, ``Estimate`` (bootstrap median), ``CI_lower``, ``CI_upper``.
    """
    if params_name is None:
        params_name = [f"param_{i}" for i in range(len(median_params))]
    summary_df = pd.DataFrame({
        "Parameter": params_name,
        "Estimate": median_params,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
    })
    summary_df.attrs = {
        "n_bootstrap_iterations": n_successful,
        "alpha": alpha,
        "confidence_level": (1 - alpha) * 100,
        "method": "Animal-level bootstrap percentile CI",
        "resampling_level": "Animal",
    }
    return summary_df


def fit_traverse_data_df_with_bootstrap(
        data_df,
        p0,
        lower_bounds,
        upper_bounds,
        params_name=None,
        func=None,
        x_grid=np.linspace(1, 20, 100),
        bootstrap_iterations=1000,
        alpha=0.05,
        seed=None,
        return_param_samples=False,
):
    """
    Fit a nonlinear curve and estimate parameter uncertainty via animal-level bootstrap.

    Bootstrap resamples at the animal level (not per-observation), preserving
    within-animal temporal structure — required for valid CIs given pseudoreplication.

    Parameters
    ----------
    data_df : pd.DataFrame
        Tidy dataframe with columns "Animal" (str), "b" (int, 1-based traverse index),
        and "Value" (float, metric value).  NaN values in "Value" are dropped per animal.
    p0 : array-like
        Initial parameter guesses for ``func``.
    lower_bounds : array-like
        Lower bounds for each parameter in ``func``.
    upper_bounds : array-like
        Upper bounds for each parameter in ``func``.
    params_name : list of str or None
        Parameter names for the summary DataFrame.  Defaults to ["param_0", ...].
    func : callable or None
        Model function with signature ``f(x, *params) → ndarray``.
        Defaults to :func:`exponential_func`.
    x_grid : ndarray, default linspace(1, 20, 100)
        X-values at which to evaluate bootstrap curves.
    bootstrap_iterations : int, default 1000
        Number of bootstrap resamples.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    seed : int or None, default None
        Seed for the animal-level resampling RNG.  ``None`` preserves the legacy
        behaviour (draws from the global NumPy RNG); pass an int for reproducible
        bootstrap CIs (used by the centralised ``gen_curve_fits.py`` step).
    return_param_samples : bool, default False
        If True, also return ``param_samples`` (the raw per-iteration bootstrap
        parameter draws), e.g. to form ratio CIs across groups with
        :func:`bootstrap_ratio_ci`.

    Returns
    -------
    bs : ndarray
        Original x-values (traverse indices, 1-based) after NaN removal.
    ds : ndarray
        Original y-values after NaN removal.
    summary_df : pd.DataFrame
        Columns: Parameter, Estimate (bootstrap median), CI_lower, CI_upper.
    bootstrap_curves : tuple of ndarray
        ``(x_grid, curve_lower, central_curve, curve_upper)``.
        Central curve uses bootstrap median parameters (not the full-data fit).
    param_samples : pd.DataFrame
        Only returned when ``return_param_samples=True``.  One row per converged
        bootstrap iteration; columns are the parameter names.

    Notes
    -----
    Bootstrap CI uses the percentile method on 1000 bootstrap parameter estimates.
    Fewer than 90% convergence triggers a warning but does not raise an error.
    """
    if func is None:
        func = exponential_func
    rng = np.random if seed is None else np.random.default_rng(seed)

    # Per-animal arrays (the bootstrap resampling unit) and the full-data originals.
    animal_arrays = _build_animal_arrays(data_df)
    bs, ds = _drop_nan_xy(data_df.b.to_numpy(), data_df.Value.to_numpy())

    # Animal-level bootstrap: one converged fit per iteration.
    bootstrap_array = _run_bootstrap_curve_fits(
        animal_arrays, func, p0, lower_bounds, upper_bounds, bootstrap_iterations, rng)

    # Parameter CIs, curve CIs (central curve = bootstrap-median params), and summary.
    median_params, ci_lower, ci_upper = _bootstrap_param_cis(bootstrap_array, alpha)
    x_grid, curve_lower, curve_upper = get_bootstrap_curve_confidence_intervals(
        bootstrap_array, func=func, x_grid=x_grid, alpha=alpha)
    central_curve = func(x_grid, *median_params)
    bootstrap_curves = (x_grid, curve_lower, central_curve, curve_upper)

    summary_df = _build_summary_df(
        params_name, median_params, ci_lower, ci_upper, alpha, bootstrap_array.shape[0])

    if return_param_samples:
        param_samples = pd.DataFrame(bootstrap_array, columns=summary_df["Parameter"].tolist())
        return bs, ds, summary_df, bootstrap_curves, param_samples
    return bs, ds, summary_df, bootstrap_curves


def fit_per_animal(data_df, p0, lower_bounds, upper_bounds, params_name, func=exponential_func):
    """
    Fit the learning curve once per animal (no bootstrap), returning per-animal params.

    Companion to :func:`fit_traverse_data_df_with_bootstrap` for cohorts too small
    for a meaningful animal-level bootstrap CI — the n=3 control cohort yields only
    ~6 distinct resamples, so its bootstrap CI is degenerate (see
    ``scripts/plot_ac_rapid.py``). Rather than draw that CI, each animal is fit
    independently so the cohort can be shown as individual per-animal parameter
    points. Reuses :func:`_build_animal_arrays` (the same per-animal ``(b, value)``
    split the bootstrap resamples) and the same ``scipy.optimize.curve_fit`` call.

    Parameters
    ----------
    data_df : pd.DataFrame
        Tidy frame with columns "Animal" (str), "b" (int, 1-based traverse index),
        and "Value" (float).  NaN "Value" rows are dropped per animal.
    p0, lower_bounds, upper_bounds : array-like
        Initial guess and per-parameter bounds passed to ``scipy.optimize.curve_fit``.
    params_name : list of str
        Parameter names, in model-argument order; become the output columns.
    func : callable, default :func:`exponential_func`
        Model function with signature ``f(x, *params) -> ndarray``.

    Returns
    -------
    pd.DataFrame
        One row per ``Animal`` (an ``"Animal"`` column) plus one column per name in
        ``params_name``.  An animal whose fit fails to converge gets NaN parameters.
    """
    animal_arrays = _build_animal_arrays(data_df)
    rows = []
    for animal, (b, value) in animal_arrays.items():
        try:
            popt, _ = curve_fit(
                func, b, value,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000,
            )
        except RuntimeError:
            popt = [np.nan] * len(params_name)
        rows.append({"Animal": animal, **dict(zip(params_name, popt))})
    return pd.DataFrame(rows, columns=["Animal", *params_name])


def fit_grouped_data_df_with_bootstrap(
        data_df,
        group_cols,
        x_grid,
        p0,
        lower_bounds,
        upper_bounds,
        params_name=None,
        func=None,
        bootstrap_iterations=1000,
        alpha=0.05,
        seed=None,
        return_param_samples=False,
):
    """
    Fit the bootstrap learning curve separately for each group of a tidy frame.

    Thin wrapper over :func:`fit_traverse_data_df_with_bootstrap`: it splits
    ``data_df`` by ``group_cols`` and fits each subset with the standard pipeline,
    so every group's result has the same format as a single-group fit. Group
    identity is carried by the returned dict key (not baked into the result),
    which keeps the fit payload uniform across the whole codebase.

    Parameters
    ----------
    data_df : pd.DataFrame
        Tidy frame with ``Animal``, ``b``, ``Value`` plus the ``group_cols``.
    group_cols : str or list of str
        Column(s) to group by (e.g. ``["Session", "Mask"]``).  The group key
        becomes the dict key (a tuple when ``group_cols`` has more than one entry).
    x_grid : ndarray or callable
        Bootstrap-curve x-grid.  Either a shared array, or a callable
        ``group_key -> ndarray`` when the grid depends on the group (e.g. a
        per-session traverse count).
    p0, lower_bounds, upper_bounds, params_name, func, bootstrap_iterations, alpha, seed
        Forwarded unchanged to :func:`fit_traverse_data_df_with_bootstrap`.  The
        same ``seed`` is used for every group; groups are independent by virtue of
        distinct animal pools.
    return_param_samples : bool, default False
        If True, also return the per-group raw bootstrap parameter draws.

    Returns
    -------
    results : dict
        ``{group_key: (bs, ds, summary_df, bootstrap_curves)}``.
    param_samples : dict
        Only when ``return_param_samples=True``: ``{group_key: param_samples_df}``.
    """
    results = {}
    param_samples = {}
    for group_key, sub_df in data_df.groupby(group_cols):
        grid = x_grid(group_key) if callable(x_grid) else x_grid
        out = fit_traverse_data_df_with_bootstrap(
            sub_df, p0=p0, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
            params_name=params_name, func=func, x_grid=grid,
            bootstrap_iterations=bootstrap_iterations, alpha=alpha, seed=seed,
            return_param_samples=return_param_samples,
        )
        if return_param_samples:
            bs, ds, summary_df, bootstrap_curves, samples = out
            results[group_key] = (bs, ds, summary_df, bootstrap_curves)
            param_samples[group_key] = samples
        else:
            results[group_key] = out

    if return_param_samples:
        return results, param_samples
    return results


def fit_grouped_data_df_with_shared_bootstrap(
        data_df,
        group_cols,
        x_grid,
        p0,
        lower_bounds,
        upper_bounds,
        params_name=None,
        func=None,
        bootstrap_iterations=1000,
        alpha=0.05,
        seed=None,
):
    """
    Joint animal-level bootstrap across groups, for within-subject comparisons.

    Unlike :func:`fit_grouped_data_df_with_bootstrap` (which resamples each group's
    animals independently), this resamples the **whole animal set once per
    iteration** and refits every group on that same draw — each group using the
    subset of resampled animals present in it.  The returned ``param_samples`` are
    therefore **aligned by iteration across groups**: row ``b`` of every group is
    the same animal resample, so a per-iteration ratio
    ``group_b / reference_b`` (via :func:`bootstrap_ratio_ci`) is within-subject.

    Each group's marginal ``summary_df`` / curves are computed from that group's
    converged iterations (NaN rows — failed or empty fits — are excluded from the
    marginal stats but **retained** in ``param_samples`` to preserve cross-group
    alignment).

    Parameters
    ----------
    data_df : pd.DataFrame
        Tidy frame with ``Animal``, ``b``, ``Value`` plus the ``group_cols``.
    group_cols : str or list of str
        Column(s) to group by; the group key becomes the dict key.
    x_grid : ndarray or callable
        Shared array, or ``group_key -> ndarray`` when the grid is group-specific.
    p0, lower_bounds, upper_bounds, params_name, func, bootstrap_iterations, alpha, seed
        As in :func:`fit_traverse_data_df_with_bootstrap`.

    Returns
    -------
    results : dict
        ``{group_key: (bs, ds, summary_df, bootstrap_curves)}`` (marginal per group).
    param_samples : dict
        ``{group_key: param_samples_df}`` with exactly ``bootstrap_iterations`` rows
        each, aligned across groups (NaN row where that group's fit failed).
    """
    if func is None:
        func = exponential_func
    rng = np.random if seed is None else np.random.default_rng(seed)
    n_params = len(p0)

    group_animal_arrays = {}
    grids = {}
    originals = {}
    for group_key, sub_df in data_df.groupby(group_cols):
        group_animal_arrays[group_key] = _build_animal_arrays(sub_df)
        grids[group_key] = x_grid(group_key) if callable(x_grid) else x_grid
        originals[group_key] = _drop_nan_xy(sub_df.b.to_numpy(), sub_df.Value.to_numpy())

    all_animals = list(data_df["Animal"].unique())
    group_keys = list(group_animal_arrays.keys())
    # Per-group (iteration x params) draws; NaN rows mark failed/empty fits.
    draws = {g: np.full((bootstrap_iterations, n_params), np.nan) for g in group_keys}

    for it in tqdm(range(bootstrap_iterations), desc="Shared bootstrapping curve fits"):
        sampled_animals = rng.choice(all_animals, size=len(all_animals), replace=True)
        for g in group_keys:
            arrays = group_animal_arrays[g]
            present = [a for a in sampled_animals if a in arrays]
            if not present:
                continue
            bs_boot = np.concatenate([arrays[a][0] for a in present])
            ds_boot = np.concatenate([arrays[a][1] for a in present])
            if len(ds_boot) < n_params:
                continue
            try:
                popt_boot, _ = curve_fit(
                    func, bs_boot, ds_boot, p0=p0,
                    bounds=(lower_bounds, upper_bounds), maxfev=10000)
                draws[g][it] = popt_boot
            except RuntimeError:
                continue

    results = {}
    param_samples = {}
    for g in group_keys:
        full = draws[g]
        valid = full[~np.isnan(full).any(axis=1)]
        n_successful = valid.shape[0]
        if n_successful == 0:
            warnings.warn(f"All shared-bootstrap fits failed for group {g!r}.")
            nan_params = np.full(n_params, np.nan)
            summary_df = _build_summary_df(params_name, nan_params, nan_params, nan_params, alpha, 0)
            grid = grids[g]
            nan_curve = np.full_like(np.asarray(grid, dtype=float), np.nan)
            bootstrap_curves = (grid, nan_curve, nan_curve, nan_curve)
        else:
            if n_successful < 0.9 * bootstrap_iterations:
                warnings.warn(
                    f"Group {g!r}: only {n_successful}/{bootstrap_iterations} "
                    f"shared-bootstrap fits converged.")
            median_params, ci_lower, ci_upper = _bootstrap_param_cis(valid, alpha)
            grid, curve_lower, curve_upper = get_bootstrap_curve_confidence_intervals(
                valid, func=func, x_grid=grids[g], alpha=alpha)
            central_curve = func(grid, *median_params)
            bootstrap_curves = (grid, curve_lower, central_curve, curve_upper)
            summary_df = _build_summary_df(
                params_name, median_params, ci_lower, ci_upper, alpha, n_successful)
        bs, ds = originals[g]
        results[g] = (bs, ds, summary_df, bootstrap_curves)
        # Keep ALL iteration rows (NaN where failed) so rows align across groups.
        param_samples[g] = pd.DataFrame(full, columns=summary_df["Parameter"].tolist())

    return results, param_samples


def bootstrap_ratio_ci(numerator_samples, denominator_samples, alpha=0.05, require_aligned=False):
    """
    Bootstrap confidence interval for the ratio of two fitted parameters.

    Forms the ratio distribution by pairing the two groups' bootstrap parameter
    draws iteration-wise.  For a *within-subject* ratio the two arrays must come
    from a shared resample (see :func:`fit_grouped_data_df_with_shared_bootstrap`),
    so row ``b`` of each is the same animal draw; for independently-fit groups the
    order is arbitrary but pairing still yields a valid Monte-Carlo ratio
    distribution.  If lengths differ, both are truncated to the shorter, and
    non-finite paired ratios (a failed-fit NaN in either array, or a zero
    denominator) are dropped.

    Parameters
    ----------
    numerator_samples, denominator_samples : array-like
        1-D bootstrap parameter draws for the numerator and denominator (e.g. the
        ``D_0`` column of a Day-2 group's and the Day-1 reference's ``param_samples``).
    alpha : float, default 0.05
        Significance level; the CI spans the ``alpha/2`` and ``1 - alpha/2`` percentiles.
    require_aligned : bool, default False
        If True, raise unless the two arrays have equal length.  Use for
        within-subject ratios, where iteration-wise pairing is only meaningful when
        both come from the same shared resample (equal length); a length mismatch
        then signals a misaligned input rather than something to silently truncate.

    Returns
    -------
    tuple of float
        ``(estimate, ci_lower, ci_upper)`` where ``estimate`` is the median of the
        paired ratios.
    """
    a = np.asarray(numerator_samples, dtype=float)
    b = np.asarray(denominator_samples, dtype=float)
    if require_aligned and len(a) != len(b):
        raise ValueError(
            f"require_aligned: numerator ({len(a)}) and denominator ({len(b)}) "
            "lengths differ; within-subject pairing needs a shared resample.")
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError("Empty bootstrap sample array(s); cannot form a ratio CI.")
    ratios = a[:n] / b[:n]
    ratios = ratios[np.isfinite(ratios)]  # drop failed-fit NaNs and zero-denominator infs
    if ratios.size == 0:
        raise ValueError("No finite paired ratios; all paired iterations failed.")
    return (
        float(np.median(ratios)),
        float(np.percentile(ratios, 100 * alpha / 2)),
        float(np.percentile(ratios, 100 * (1 - alpha / 2))),
    )


def bootstrap_summary_ratio_ci(numerator_samples_list, denominator_samples, alpha=0.05,
                               reduce=np.nanmedian, require_aligned=False):
    """
    Bootstrap CI for a summary ratio that reduces across several groups per iteration.

    Given a list of numerator draws (e.g. all 12 Day-2 ``(session, mask)`` groups)
    sharing one denominator (the Day-1 reference), all aligned by iteration via a
    shared-resample bootstrap, this computes the per-iteration ratios, reduces them
    across the groups (default: median over groups), then returns the bootstrap CI
    of that per-iteration summary.  Implements
    ``R_summary^(b) = reduce_g( numerator_g^(b) / denominator^(b) )``.

    Parameters
    ----------
    numerator_samples_list : list of array-like
        One 1-D bootstrap draw array per group (all the same iteration ordering).
    denominator_samples : array-like
        Shared denominator draws (the reference group).
    alpha : float, default 0.05
        Significance level for the percentile CI.
    reduce : callable, default ``np.nanmedian``
        Per-iteration reduction across groups (axis 0); NaN-aware so groups whose
        fit failed at an iteration are ignored for that iteration.
    require_aligned : bool, default False
        If True, raise unless every numerator array has the same length as the
        denominator.  Use for within-subject summaries, where the per-iteration
        reduction is only meaningful when all groups share the same resample
        (equal length); a mismatch then signals misaligned input.

    Returns
    -------
    tuple of float
        ``(estimate, ci_lower, ci_upper)`` for the summary ratio.
    """
    den = np.asarray(denominator_samples, dtype=float)
    if len(numerator_samples_list) == 0:
        raise ValueError("numerator_samples_list is empty.")
    if require_aligned:
        bad = [len(np.asarray(num)) for num in numerator_samples_list
               if len(np.asarray(num)) != len(den)]
        if bad:
            raise ValueError(
                f"require_aligned: {len(bad)} numerator array(s) differ in length "
                f"from the denominator ({len(den)}); need a shared resample.")
    ratio_rows = []
    for num in numerator_samples_list:
        num = np.asarray(num, dtype=float)
        n = min(len(num), len(den))
        ratio_rows.append(num[:n] / den[:n])
    n = min(len(r) for r in ratio_rows)
    stack = np.vstack([r[:n] for r in ratio_rows])  # (n_groups, n_iter)
    summary_per_iter = reduce(stack, axis=0)         # (n_iter,)
    summary_per_iter = summary_per_iter[np.isfinite(summary_per_iter)]
    if summary_per_iter.size == 0:
        raise ValueError("No finite summary ratios across iterations.")
    return (
        float(np.median(summary_per_iter)),
        float(np.percentile(summary_per_iter, 100 * alpha / 2)),
        float(np.percentile(summary_per_iter, 100 * (1 - alpha / 2))),
    )


def relative_performance(raw_with_baseline_df, n_iterations=1000, n_rewards=10, seed=None):
    """
    Baseline-relative metric ratio with a two-level (session, bout) bootstrap CI.

    Computes how a comparison condition's metric compares to a per-cohort baseline,
    as ``mean(comparison values) / mean(baseline values)`` over the pooled first-N
    per-bout values, with a hierarchical bootstrap confidence interval. Used for the
    memory-vs-gap ratios (``gen_ac_mem.py``: baseline = Day-0 session, comparison =
    post-gap session) and the pre-/post-swap ratio (``gen_olfaction.py``: baseline =
    pre-swap session, comparison = post-swap session).

    Baseline rows are those with ``Day == 0``; every other ``Day`` value is treated
    as a comparison row, so the caller encodes the contrast through the ``Day``
    column (0 = baseline, non-zero = comparison).

    Parameters
    ----------
    raw_with_baseline_df : pd.DataFrame
        Tidy per-bout frame with columns ``Animal`` (str), ``Day`` (int; 0 =
        baseline), and ``Value`` (float, the per-bout metric). Additional columns
        are ignored.
    n_iterations : int, default 1000
        Number of bootstrap iterations.
    n_rewards : int, default 10
        Per-``(animal, day)`` bout-resample cap; each resampled session contributes
        ``min(n_bouts, n_rewards)`` bouts per iteration.
    seed : int or None, default None
        Seed for the hierarchical resampling RNG.  ``None`` preserves the legacy
        behaviour (draws from the global NumPy RNG), which makes the returned CI
        differ on every run; pass an int for reproducible CIs (the producers
        ``gen_ac_mem.py`` and ``gen_olfaction.py`` pass their ``--seed``).
        ``observed_ratio`` is unaffected either way -- it is computed from the
        un-resampled data.

    Returns
    -------
    observed_ratio : float
        ``nanmean(comparison values) / nanmean(baseline values)`` from the original
        (un-resampled) data; ``nan`` if the baseline mean is 0.
    ci : tuple of float
        ``(low, high)`` = 2.5th / 97.5th percentiles of the bootstrap ratio
        distribution (95% CI).

    Notes
    -----
    Hierarchical bootstrap: each iteration resamples ``(animal, day)`` pairs with
    replacement *within* the baseline and comparison groups separately, then
    resamples bouts with replacement within each chosen pair. The resampling unit
    is therefore the session, with an inner bout-level resample.
    """
    rng = np.random if seed is None else np.random.default_rng(seed)

    # Build lookup: (animal, day) -> array of per-bout values.
    sample_data, animal_day_pairs = {}, []
    for _, row in raw_with_baseline_df.iterrows():
        pair = (row['Animal'], row['Day'])
        if pair not in sample_data:
            sample_data[pair] = []
            animal_day_pairs.append(pair)
        sample_data[pair].append(row['Value'])
    for pair in sample_data:
        sample_data[pair] = np.array(sample_data[pair])

    baseline_pairs = [(a, d) for a, d in animal_day_pairs if d == 0]
    comparison_pairs = [(a, d) for a, d in animal_day_pairs if d != 0]

    # Observed ratio from the original data (pooled mean of bouts per group).
    baseline_values = np.concatenate([sample_data[p] for p in baseline_pairs])
    comparison_values = np.concatenate([sample_data[p] for p in comparison_pairs])
    observed_ratio = (np.nanmean(comparison_values) / np.nanmean(baseline_values)
                      if np.nanmean(baseline_values) != 0 else np.nan)

    # Hierarchical bootstrap: resample (animal, day) pairs, then bouts within each.
    n_baseline, n_comparison = len(baseline_pairs), len(comparison_pairs)
    baseline_pairs_array = np.array(baseline_pairs, dtype=object)
    comparison_pairs_array = np.array(comparison_pairs, dtype=object)
    boot_ratios = []
    for _ in tqdm(range(n_iterations), desc="Bootstrapping Ratios"):
        resampled_baseline = [tuple(p) for p in
                              baseline_pairs_array[rng.choice(n_baseline, size=n_baseline, replace=True)]]
        resampled_comparison = [tuple(p) for p in
                                comparison_pairs_array[rng.choice(n_comparison, size=n_comparison, replace=True)]]
        boot_baseline_values, boot_comparison_values = [], []
        for pair in resampled_baseline:
            bouts = sample_data[pair]
            boot_baseline_values.extend(rng.choice(bouts, size=min(len(bouts), n_rewards), replace=True))
        for pair in resampled_comparison:
            bouts = sample_data[pair]
            boot_comparison_values.extend(rng.choice(bouts, size=min(len(bouts), n_rewards), replace=True))
        if boot_baseline_values and boot_comparison_values and np.mean(boot_baseline_values) != 0:
            boot_ratios.append(np.mean(boot_comparison_values) / np.mean(boot_baseline_values))

    low = np.nanpercentile(boot_ratios, 2.5)
    high = np.nanpercentile(boot_ratios, 97.5)
    return observed_ratio, (low, high)
