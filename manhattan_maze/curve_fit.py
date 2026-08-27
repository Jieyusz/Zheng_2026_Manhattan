"""Learning-curve models, nonlinear fitting, analytic confidence intervals, and fit quality.

Split out of utils.py; see docs.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

__all__ = ['linear_func', 'exponential_func', 'late_performance_samples', 'exponential_tau_with_ci', '_piecewise_func_legacy', 'fit_linear_curve_with_ci', 'linear_curve_confidence_interval', 'get_generalized_ci', 'day_on_learning_rate_model', 'day_on_base_model', 'stats_summary_of_curve_fit', 'calculate_rss', 'calculate_aic']

def linear_func(x, a, b):
    """
    Linear model for curve fitting: f(x) = -a*x + b.

    Parameters
    ----------
    x : float or array-like
        Independent variable (e.g., traverse index).
    a : float
        Slope magnitude; constrained ≥ 0 in fits.
    b : float
        Intercept (value at x=0); constrained ≥ 0 in fits.

    Returns
    -------
    float or np.ndarray
        Predicted value: -a*x + b.
    """
    return -a*x + b


def exponential_func(b, D_infty, D_0, k):
    """
    Exponential learning curve used for both traverse duration (Eq. 1) and turn
    error rate (Eq. 2): value(b) = D_infty + (D_0 - D_infty) * exp(-k * (b - 1)).

    Parameters
    ----------
    b : int or array-like
        Traverse index, 1-based. At b=1 the function returns exactly D_0.
    D_infty : float
        Asymptotic value as b → ∞ [same units as D_0].
    D_0 : float
        Value at b=1 (first traverse).
    k : float
        Learning rate [1/traverse]. Characteristic time scale tau = 1/k.
        Stored as ``delta`` (δ) for duration fits and ``epsilon`` (ε) for
        turn-error fits (manuscript notation; see docs/notation_guide.md).

    Returns
    -------
    float or ndarray
        Predicted value at traverse b.

    Notes
    -----
    b is indexed from 1, not 0 — this is baked into the relationship tau = 1/k.
    """
    return D_infty + ((D_0 - D_infty) * np.exp(- k * (b - 1)))


def late_performance_samples(params_df, param_names, t_late, func=exponential_func):
    """
    Per-iteration fitted value at a late, in-range traverse — a data-anchored
    stand-in for the t→∞ asymptote.

    The exponential's asymptote (``D_infty`` / ``E_infty``) is unidentifiable on
    Day-2 re-exposure — the fit rails it to the physical floor because the slow
    decay never converges within the observed traverses — so its Day-2/Day-1
    ratio is meaningless and its bootstrap CI is severely right-skewed. Reading
    the fitted curve *inside* the observed range instead (at ``t_late``) gives a
    well-identified "late/settled performance" whose ratio has a near-symmetric
    CI. Evaluating from the saved bootstrap param draws needs no refit.

    Parameters
    ----------
    params_df : pd.DataFrame
        Bootstrap parameter draws with one column per name in ``param_names``
        (rows = iterations), e.g. a group's ``param_samples`` frame.
    param_names : sequence of str
        The three fit-parameter names in ``func``'s argument order
        ``[asymptote, initial, rate]`` (e.g. ``["D_infty", "D_0", "delta"]``).
    t_late : float
        Traverse index at which to evaluate the curve; choose a value inside the
        observed range of every group being compared (e.g. the minimum observed
        max-traverse) so the value is interpolated, not extrapolated.
    func : callable, default :func:`exponential_func`
        Curve model ``func(b, asymptote, initial, rate)``.

    Returns
    -------
    np.ndarray
        1-D array of per-iteration curve values, aligned like the input columns
        (so it drops straight into :func:`bootstrap_ratio_ci`).
    """
    asymptote, initial, rate = param_names
    return func(t_late, params_df[asymptote].to_numpy(),
                params_df[initial].to_numpy(), params_df[rate].to_numpy())


def exponential_tau_with_ci(popt, pcov, n_observations):
    """
    Compute learning time constant tau and its 95% confidence interval.

    Applies the delta method to propagate uncertainty from the fitted rate
    constant k (``popt[2]``) to tau = 1/k.

    Parameters
    ----------
    popt : array-like, shape (3,)
        Fitted parameters [D_infty, D_0, k] from :func:`exponential_func`.
    pcov : np.ndarray, shape (3, 3)
        Parameter covariance matrix from :func:`scipy.optimize.curve_fit`.
    n_observations : int
        Number of data points used for fitting; determines degrees of freedom.

    Returns
    -------
    tau : float
        Learning time constant [traverses].
    tau_ci_low : float
        Lower 95% confidence bound on tau [traverses].
    tau_ci_high : float
        Upper 95% confidence bound on tau [traverses].
    """
    # 1. Get k and its standard error
    k_fit = popt[2]
    k_var = pcov[2, 2]  # Variance of k from the diagonal of pcov
    k_se = np.sqrt(k_var)

    # 2. Calculate Tau
    tau = 1 / k_fit

    # 3. Propagate error to Tau (Delta Method)
    tau_se = k_se / (k_fit ** 2)

    # 4. Calculate 95% Confidence Interval for Tau
    dof = n_observations - len(popt)
    t_val = stats.t.ppf(0.975, dof)

    tau_ci_low = tau - (t_val * tau_se)
    tau_ci_high = tau + (t_val * tau_se)
    return tau, tau_ci_low, tau_ci_high


def _piecewise_func_legacy(b, e_infty, e0, alpha):
    """
    Piecewise-linear turn-error model.  NOT used in manuscript analysis.

    Retained for legacy comparison only. The publication pipeline uses
    exponential_func for both duration and turn-error rate (R14).

    Parameters
    ----------
    b : float
        Traverse index (0-based in this formulation).
    e_infty : float
        Floor: minimum turn error rate.
    e0 : float
        Ceiling: initial turn error rate.
    alpha : float
        Linear decline rate [error_rate / traverse].

    Returns
    -------
    float
        clip(e0 - alpha * b, e_infty, e0)
    """
    return np.clip(e0 - alpha * b, e_infty, e0)


def fit_linear_curve_with_ci(xs, ys):
    """
    Fit a linear model to data and return parameters with confidence intervals.

    Uses :func:`linear_func` (f(x) = -a*x + b) with non-negative parameter
    constraints.

    Parameters
    ----------
    xs : array-like
        Independent variable values.
    ys : array-like
        Observed dependent variable values (same length as ``xs``).

    Returns
    -------
    popt : np.ndarray, shape (2,)
        Fitted parameters [a, b].
    pcov : np.ndarray, shape (2, 2)
        Parameter covariance matrix.
    t : float
        Two-tailed t-critical value at alpha=0.05 for the residual degrees of freedom.
    ci : np.ndarray, shape (2,)
        95% confidence interval half-widths for each parameter.
    std : float
        Standard deviation of residuals.
    """
    popt, pcov = curve_fit(f=linear_func, xdata=xs, ydata=ys,
                           p0=(1, 20),
                           bounds = ([0, 0], [np.inf, np.inf]))
    n = len(xs)
    p = len(popt)
    sse = np.sum((ys - linear_func(xs, *popt))**2) # sum of squared errors
    dof = max(0, n - p) # degrees of freedom
    std = np.sqrt(sse / dof) # standard deviation of the residuals
    t = stats.t.ppf(1 - 0.05 / 2., dof) # t-score
    p_err = np.sqrt(np.diag(pcov)) # standard deviation of the parameters
    ci = t * p_err # confidence interval
    return popt, pcov, t, ci, std


def linear_curve_confidence_interval(xs, popt, pcov, t):
    """
    Evaluate the pointwise 95% confidence band for a fitted linear curve.

    Parameters
    ----------
    xs : array-like
        X values at which to evaluate the confidence interval.
    popt : array-like, shape (2,)
        Fitted parameters [a, b] from :func:`fit_linear_curve_with_ci`.
    pcov : np.ndarray, shape (2, 2)
        Parameter covariance matrix.
    t : float
        Two-tailed t-critical value (from :func:`fit_linear_curve_with_ci`).

    Returns
    -------
    y_fit : np.ndarray
        Fitted curve values at ``xs``.
    ci : tuple of (np.ndarray, np.ndarray)
        (upper_bound, lower_bound) of the confidence band at each x in ``xs``.
    """
    y_fit = linear_func(xs, *popt)
    # calculate the errors
    jacobian = np.array([-xs, np.ones_like(xs)])
    y_err = np.sqrt(np.diag(np.dot(jacobian.T, np.dot(pcov, jacobian))))  # standard deviation of the fitted curve
    ci = (y_fit+y_err*t, y_fit-y_err*t)
    return y_fit, ci


def get_generalized_ci(model_func, x_grid, popt, pcov, n_observations, alpha=0.05):
    """
    Compute pointwise confidence intervals for any model via numerical Jacobian.

    Uses central-difference finite differences to approximate the Jacobian of
    ``model_func`` with respect to its parameters, then propagates covariance
    via the delta method.

    Parameters
    ----------
    model_func : callable
        Model function with signature ``f(x, *params) -> np.ndarray``.
    x_grid : np.ndarray
        X values at which to evaluate the confidence band.
    popt : np.ndarray, shape (n_params,)
        Fitted parameter vector.
    pcov : np.ndarray, shape (n_params, n_params)
        Parameter covariance matrix.
    n_observations : int
        Number of observations used for the fit; sets degrees of freedom.
    alpha : float, default 0.05
        Significance level for the two-tailed confidence interval.

    Returns
    -------
    x_grid : np.ndarray
        Echo of the input x grid.
    lower : np.ndarray
        Lower confidence bound at each x.
    y_mu : np.ndarray
        Predicted mean curve at each x.
    upper : np.ndarray
        Upper confidence bound at each x.
    """
    y_mu = model_func(x_grid, *popt)

    eps = 1e-6

    n_params = len(popt)
    jacobian = np.zeros((len(x_grid), n_params))

    for i in range(n_params):
        p_step = np.zeros(n_params)
        p_step[i] = eps * (abs(popt[i]) if popt[i] != 0 else 1)

        # Central difference
        y_plus = model_func(x_grid, *(popt + p_step))
        y_minus = model_func(x_grid, *(popt - p_step))
        jacobian[:, i] = (y_plus - y_minus) / (2 * p_step[i])

    # Error propagation
    sigma_y = np.sqrt(np.einsum('ij,jk,ik->i', jacobian, pcov, jacobian))

    dof = n_observations - n_params
    t_val = stats.t.ppf(1-alpha/2, dof)

    return x_grid, y_mu - t_val * sigma_y, y_mu,  y_mu + t_val * sigma_y


def day_on_learning_rate_model(x, D_infty, D_0, b_mask, k_base, b_day_k):
    """
    Exponential learning model where day modulates the learning rate k.

    f(bout, day, mask) = D_infty + (D_0 + b_mask*mask - D_infty) *
                         exp(-(k_base + b_day_k*(day-1)) * (bout-1))

    Parameters
    ----------
    x : np.ndarray, shape (3, n)
        Row 0 = bout index (1-based), row 1 = day index (1-based),
        row 2 = mask index (int).
    D_infty : float
        Asymptotic performance level.
    D_0 : float
        Baseline at bout 1, day 1.
    b_mask : float
        Additive shift to the starting value per mask unit.
    k_base : float
        Base learning rate at day 1 [1/traverse].
    b_day_k : float
        Day-to-day change in learning rate.

    Returns
    -------
    np.ndarray
        Predicted metric value for each observation in ``x``.
    """
    assert x.shape[0] == 3, "x should have three rows: bout, day, and mask"
    bout = x[0]
    day = x[1]
    mask = x[2]

    # We still keep b_mask on the start if you think it affects baseline,
    # but let's see how Day affects the Rate k.
    current_start = D_0 + (b_mask * mask)

    # The effective learning rate changes by day
    effective_k = k_base + (b_day_k * (day-1))

    return D_infty + (current_start - D_infty) * np.exp(-effective_k * (bout-1))


def day_on_base_model(x, D_infty, D_0, b_mask, b_day, k):
    """
    Exponential learning model where day modulates the initial performance baseline.

    f(bout, day, mask) = D_infty + (D_0 + b_mask*mask + b_day*(day-1) - D_infty) *
                         exp(-k * (bout-1))

    Parameters
    ----------
    x : np.ndarray, shape (3, n)
        Row 0 = bout index (1-based), row 1 = day index (1-based),
        row 2 = mask index (int).
    D_infty : float
        Asymptotic performance level.
    D_0 : float
        Baseline at bout 1, day 1.
    b_mask : float
        Additive shift to the starting value per mask unit.
    b_day : float
        Day-to-day additive shift to the starting value.
    k : float
        Learning rate [1/traverse].

    Returns
    -------
    np.ndarray
        Predicted metric value for each observation in ``x``.
    """
    assert x.shape[0] == 3, "x should have three rows: bout, day, and mask"
    bout = x[0]
    day = x[1]
    mask = x[2]

    # The baseline changes by day
    current_start = D_0 + (b_mask * mask) + (b_day * (day-1))

    return D_infty + (current_start - D_infty) * np.exp(-k * (bout-1))


def stats_summary_of_curve_fit(y, popt, pcov, params_names=None, alpha=0.05, lower_bounds=None, upper_bounds=None, p0=None):
    """
    Summarise a curve fit with parameter estimates, SEs, CIs, and t-statistics.

    Parameters
    ----------
    y : array-like
        Observed data values used for the fit.
    popt : np.ndarray, shape (n_params,)
        Fitted parameter vector from :func:`scipy.optimize.curve_fit`.
    pcov : np.ndarray, shape (n_params, n_params)
        Parameter covariance matrix.
    params_names : list of str or None, default None
        Names for each parameter.  Defaults to ``['param_0', 'param_1', …]``.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    lower_bounds : array-like or None, default None
        Lower parameter bounds; appended to the summary when provided.
    upper_bounds : array-like or None, default None
        Upper parameter bounds; appended to the summary when provided.
    p0 : array-like or None, default None
        Initial parameter guesses; appended to the summary when provided.

    Returns
    -------
    pd.DataFrame
        Columns: ``Parameter``, ``Estimate``, ``Std`` (SE), ``ci_lower``,
        ``ci_upper``, ``t_statistic``, ``p_value``.  Optional columns
        ``lower_bound``, ``upper_bound``, ``initialization`` when provided.
    """
    if params_names is None:
        params_names = [f"param_{i}" for i in range(len(popt))]

    n = len(y)  # number of data points
    df = max(0, n - len(popt))  # degrees of freedom
    perr = np.sqrt(np.diag(pcov))  # standard errors of the parameters
    t_stat = popt / perr  # t-statistics for each parameter compared to zero
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))  # two-tailed p-values
    t_crit = stats.t.ppf(1-alpha/2, df) # two tailed critical t value for confidence intervals

    conf_low = popt - t_crit * perr
    conf_high = popt + t_crit * perr

    summary = pd.DataFrame({
        "Parameter": params_names,
        "Estimate": popt,
        "Std": perr,
        "ci_lower": conf_low,
        "ci_upper": conf_high,
        "t_statistic": t_stat,
        "p_value": p_values
    })
    for name, initial_params in zip(["lower_bound", "upper_bound", "initialization"],[lower_bounds, upper_bounds, p0]):
        if initial_params is not None:
            assert len(initial_params) == len(params_names), "Only save the initial parameters if the length matches the number of parameters to be estimated."
            summary[name] = initial_params

    return summary


def calculate_rss(func, x, y, popt):
    """
    Compute the residual sum of squares for a model.

    Parameters
    ----------
    func : callable
        Model function with signature ``f(x, *popt) -> array-like``.
    x : array-like
        Independent variable values.
    y : array-like
        Observed dependent variable values.
    popt : array-like
        Fitted parameter vector.

    Returns
    -------
    float
        Sum of squared residuals: Σ (y_i − f(x_i, *popt))².
    """
    residuals = y - func(x, *popt)
    rss = np.sum(residuals**2)
    return rss


def calculate_aic(func, x, y, popt):
    """
    Compute the Akaike Information Criterion (AIC) for a model.

    Parameters
    ----------
    func : callable
        Model function with signature ``f(x, *popt) -> array-like``.
    x : array-like
        Independent variable values.
    y : array-like
        Observed dependent variable values.
    popt : array-like
        Fitted parameter vector.

    Returns
    -------
    float
        AIC = n*log(RSS/n) + 2k, where n is the number of observations and
        k is the number of parameters.
    """
    rss = calculate_rss(func, x, y, popt)
    n = len(y)  # number of data points
    k = len(popt)  # number of parameters
    aic = n * np.log(rss / n) + 2 * k
    return aic
