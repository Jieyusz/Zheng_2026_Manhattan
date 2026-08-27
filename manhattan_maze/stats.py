"""Nonparametric group statistics (Friedman/Kruskal/Wilcoxon/Mann-Whitney/Levene) and helpers.

Split out of utils.py; see docs.
"""
import numpy as np
import pandas as pd
from scipy import stats
from manhattan_maze.utils import extract_timepoint_from_array_dict

__all__ = ['get_ecdf', 'friedman_with_pairwise_wilcoxon', 'kruskal_with_pairwise_mann_whitney', 'create_within_subject_data_dict', 'levene_test', 'pairwise_wilcoxon_signed_rank_test', 'pairwise_mann_whitney_u_test', 'time_point_kruskal_mann_whitney_u_test']

def get_ecdf(x):
    """
    Compute the empirical cumulative distribution function of ``x``.

    Parameters
    ----------
    x : array-like
        1-D data sample.

    Returns
    -------
    x_sorted : np.ndarray
        Data sorted in ascending order.
    y : np.ndarray
        ECDF values in [0, 1], linearly spaced over the number of observations.
    """
    x = np.sort(x)
    # create the ECDF
    y = np.linspace(0, 1, len(x))
    return x, y


def friedman_with_pairwise_wilcoxon(data_dict, **kwargs):
    """
    Friedman test followed by pairwise Wilcoxon signed-rank tests.

    For exactly two groups, skips Friedman and runs pairwise Wilcoxon directly.
    For three or more groups, runs Friedman first and performs pairwise tests
    only if the omnibus p-value < 0.05.

    Parameters
    ----------
    data_dict : dict of {str: array-like}
        Keys are group labels; values are matched (within-subject) 1-D arrays.
        Arrays are truncated to the length of the shortest group before testing.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`pairwise_wilcoxon_signed_rank_test`.

    Returns
    -------
    friedman_stat : float or None
        Friedman test statistic; None when only two groups.
    friedman_p : float or None
        Friedman p-value; None when only two groups.
    pairwise_results : pd.DataFrame or None
        Output of :func:`pairwise_wilcoxon_signed_rank_test`, or None if
        Friedman was not significant.
    """
    vals = list(data_dict.values())
    if len(vals) == 2:
        pairwise_results = pairwise_wilcoxon_signed_rank_test(data_dict, **kwargs)
        return None, None, pairwise_results
    else:
        # make sure the array size is the same
        array_size = min([len(v) for v in vals])
        neat_dict = {key:val[:array_size] for key, val in data_dict.items()}
        neat_vals = list(neat_dict.values())
        friedman_stat, friedman_p = stats.friedmanchisquare(*neat_vals)
        if friedman_p < 0.05:
            pairwise_results = pairwise_wilcoxon_signed_rank_test(neat_dict, **kwargs)
            return friedman_stat, friedman_p, pairwise_results
        else:
            return friedman_stat, friedman_p, None


def kruskal_with_pairwise_mann_whitney(data_dict, **kwargs):
    """
    Kruskal-Wallis test followed by pairwise Mann-Whitney U tests.

    For exactly two groups, skips Kruskal-Wallis and runs pairwise Mann-Whitney
    directly.  For three or more groups, runs Kruskal-Wallis first and performs
    pairwise tests only if the omnibus p-value < 0.05.

    Parameters
    ----------
    data_dict : dict of {str: array-like}
        Keys are group labels; values are independent 1-D data arrays.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`pairwise_mann_whitney_u_test`.

    Returns
    -------
    kruskal_stat : float or None
        Kruskal-Wallis H statistic; None when only two groups.
    kruskal_p : float or None
        Kruskal-Wallis p-value; None when only two groups.
    pairwise_results : pd.DataFrame or None
        Output of :func:`pairwise_mann_whitney_u_test`, or None if Kruskal was
        not significant.
    """
    # Perform Kruskal-Wallis test
    vals = list(data_dict.values())
    if len(vals) == 2:
        # only one pair, return mann_whitney_u_test results:
        pairwise_results = pairwise_mann_whitney_u_test(data_dict, **kwargs)
        return None, None, pairwise_results
    else:
        kruskal_stat, kruskal_p = stats.kruskal(*vals)
        # If the Kruskal-Wallis test is significant, perform pairwise comparisons
        if kruskal_p < 0.05:
            pairwise_results = pairwise_mann_whitney_u_test(data_dict, **kwargs)
            return kruskal_stat, kruskal_p, pairwise_results
        else:
            return kruskal_stat, kruskal_p, None


def create_within_subject_data_dict(data_list):
    """
    Filter to complete cases across a list of paired 1-D arrays.

    Builds a dictionary of arrays sharing the same set of non-NaN positions
    across all input arrays (listwise deletion of missing values).

    Parameters
    ----------
    data_list : list of np.ndarray
        List of 1-D float arrays of the same length.  Each array is one
        measurement condition; NaN indicates a missing observation.

    Returns
    -------
    dict of {int: np.ndarray}
        Integer-keyed dictionary (0, 1, …) where each value is the filtered
        array retaining only positions that are non-NaN in all input arrays.
    """
    mask = ~np.isnan(data_list[0])  # Initialize mask with the first array
    for d_array in data_list[1:]:
        mask = mask & ~np.isnan(d_array)  # Update mask to include only non-NaN values across all arrays

    data_dict = {}
    for i, d_array in enumerate(data_list):
        # Filter the data array based on the mask
        filtered_data = d_array[mask]
        data_dict[i] = filtered_data

    return data_dict


def levene_test(sample1, sample2, center='median'):
    """
    Levene's test for homogeneity of variances between two samples.

    Parameters
    ----------
    sample1 : array-like
        First sample data.
    sample2 : array-like
        Second sample data.
    center : {'median', 'mean', 'trimmed'}, default 'median'
        Centering method.  ``'median'`` gives the Brown-Forsythe variant.

    Returns
    -------
    stat : float
        Levene test statistic.
    p_value : float
        Two-tailed p-value.
    """
    vals = [sample1, sample2]
    stat, p_value = stats.levene(*vals, center=center)
    return stat, p_value


def pairwise_wilcoxon_signed_rank_test(data_dict, alternative="two-sided"):
    """
    Perform all pairwise Wilcoxon signed-rank tests between groups.

    Parameters
    ----------
    data_dict : dict of {any: array-like}
        Keys are group labels; values are matched (within-subject) 1-D arrays
        of equal length.
    alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
        Alternative hypothesis passed to :func:`scipy.stats.wilcoxon`.

    Returns
    -------
    pd.DataFrame
        Columns: ``group1``, ``group2``, ``u_stat``, ``p_value``.  One row per
        unique pair (i < j).
    """
    pairwise_results = []
    group_names = list(data_dict.keys())
    for i, key1 in enumerate(group_names):
        for j, key2 in enumerate(group_names):
            if i >= j:
                continue
            u_stat, p_value = stats.wilcoxon(data_dict[key1], data_dict[key2],
                                                 alternative=alternative,)
            pairwise_results.append({
                'group1': key1,
                'group2': key2,
                'u_stat': u_stat,
                'p_value': p_value
            })
    return pd.DataFrame(pairwise_results)


def pairwise_mann_whitney_u_test(data_dict, alternative='two-sided', nan_policy="omit"):
    """
    Perform all pairwise Mann-Whitney U tests between groups.

    Parameters
    ----------
    data_dict : dict of {any: array-like}
        Keys are group labels; values are independent 1-D data arrays.
    alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
        Alternative hypothesis passed to :func:`scipy.stats.mannwhitneyu`.
    nan_policy : {'omit', 'propagate', 'raise'}, default 'omit'
        How to handle NaN values.

    Returns
    -------
    pd.DataFrame
        Columns: ``group1``, ``group2``, ``u_stat``, ``p_value``.  One row per
        unique pair (i < j).
    """
    pairwise_results = []
    group_names = list(data_dict.keys())
    for i, key1 in enumerate(group_names):
        for j, key2 in enumerate(group_names):
            if i >= j:
                continue
            u_stat, p_value = stats.mannwhitneyu(data_dict[key1], data_dict[key2],
                                                 alternative=alternative, nan_policy=nan_policy)
            pairwise_results.append({
                'group1': key1,
                'group2': key2,
                'u_stat': u_stat,
                'p_value': p_value
            })
    return pd.DataFrame(pairwise_results)


def time_point_kruskal_mann_whitney_u_test(data_dict, time_range=None, alternative='two-sided',  nan_policy="omit", ):
    """
    Run per-time-point Kruskal-Wallis and Mann-Whitney U tests.

    Parameters
    ----------
    data_dict : dict of {any: np.ndarray}
        Values are 2-D arrays of shape (n_animals, n_time).
    time_range : int or None, default None
        Number of time points to test.  Defaults to the minimum n_time across
        all arrays.
    alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
        Passed to :func:`kruskal_with_pairwise_mann_whitney`.
    nan_policy : str, default 'omit'
        Passed to :func:`kruskal_with_pairwise_mann_whitney`.

    Returns
    -------
    list of tuple
        One element per time point; each element is the return value of
        :func:`kruskal_with_pairwise_mann_whitney` (kruskal_stat, kruskal_p,
        pairwise_results).
    """
    stats_results = []
    if time_range is None:
        # get the array shape for this
        array_sizes = [val.shape[1]for val in data_dict.values()]
        time_range = np.min(array_sizes)
    for t in range(time_range):
        sub_dict = extract_timepoint_from_array_dict(data_dict, t)
        u_tests = kruskal_with_pairwise_mann_whitney(sub_dict, alternative=alternative, nan_policy=nan_policy)
        stats_results.append(u_tests)
    return stats_results
