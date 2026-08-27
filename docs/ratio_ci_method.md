# Curve-derived ratio confidence intervals: late-performance & saturation handling

How the learning-curve **curve-derived ratio forest panels** (formerly called
"parameter-ratio" panels) compute their point estimates and 95% bootstrap CIs. All
such panels share one estimator policy, implemented once in
`scripts/gen_curve_fits.py` and reused by every ratio-table builder. The panels are
called *curve-derived* rather than *parameter* ratios because the asymptote row is no
longer a raw fit parameter but a quantity read off the fitted curve (`X_late`, below);
the on-disk cache keys still contain `param ratios` for backward compatibility.

## The learning-curve model

Duration and turn-error learning curves are fit with the exponential
(`manhattan_maze/curve_fit.py::exponential_func`):

    value(b) = X_infty + (X_0 - X_infty) * exp(-k * (b - 1))          (b = traverse #, 1-based)

giving three parameters per metric (`CURVE_FIT_SPECS` in `scripts/config.py`):

| role      | duration            | turn error            | fit bounds (duration / turn error) |
|-----------|---------------------|-----------------------|------------------------------------|
| asymptote | `D_infty`           | `E_infty`             | [2, 60] / [0.001, 0.5]             |
| initial   | `D_0`               | `E_0`                 | [5, 800] / [0.1, 1]                |
| rate      | `delta` (δ)         | `epsilon` (ε)         | [0.01, 1] / [0.01, 1]              |

Curve-derived ratio panels report the within-subject or cross-cohort ratio of each
curve-derived quantity (e.g. Day2/Day1), as a forest plot with a bootstrap median and
percentile CI.

The turn error rate fed to these fits is **approach-conditioned** (a hole crossing is
scored only when the mouse enters on the shortest-path corridor; chance level 0.5 —
see `docs/notation_guide.md`). All turn-error numbers below reflect that default; the
duration numbers are unaffected.

## The problem

Two parameters produce meaningless, severely right-skewed ratio CIs (median
pinned near the lower bound, huge upper arm):

1. **Asymptote (`X_infty`) is unidentifiable.** The `t→∞` asymptote is an
   *extrapolation*: when the decay is slow relative to the observed traverses the
   exponential never converges within the data, so the fit trades off `X_infty`
   against the rate and rails the asymptote to a fit bound. Measured saturation
   (now saved per row as `sat_frac_num`/`sat_frac_den`; see "Reported diagnostics"):
   Day2-1 Mask A `D_infty` **65%** at floor, `E_infty` **43%**; acortical `E_infty`
   floored **50%**, control **53%**; E_infty saturation runs 0–43% across the other
   sessions/masks. The resulting ratio arm-asymmetry (upper/lower) reached **32×**
   (`D_infty`), **287×** (`E_infty` two-day) and **~2.9×10⁴×** (`E_infty`
   cross-genotype — the ratio of two near-floor asymptotes is pathological).
   Biologically the asymptote is also the
   wrong construct on Day-2 re-exposure — the mice already know the route, so
   late traverse-time variance reflects explore/exploit state switching, not a
   learning plateau.

2. **Rate (`delta`/`epsilon`) partially rails to its ceiling.** A fraction of
   draws (≈0–11% for the two-day turn-error masks, up to ~80% for some generalization
   masks) hit the rate ceiling of 1, inflating the upper CI (e.g. `epsilon` Day2-1 B
   upper ≈31 before masking; see "Reported diagnostics"). Under the approach-conditioned
   turn-error metric the Day-2 curves are flatter (near the 0.5-chance floor), so
   `epsilon` is only weakly identified and its ratio upper arm stays wide even after
   masking — see the robustness note below.

Neither is a bug in the CI math (percentile of paired bootstrap ratios); both are
identifiability/boundary artifacts of the fit. Lowering the bounds is not an
option — e.g. the `D_infty` floor of 2 s is the physical minimum traverse time.

## The method

Implemented as shared helpers in `scripts/gen_curve_fits.py`
(`_nan_saturated`, `_ratio_samples`, `_ratio_row`, `_summary_ratio_row`), plus
`late_performance_samples` in `manhattan_maze/curve_fit.py`:

- **Asymptote → data-anchored late performance (`X_late`).** Instead of the
  extrapolated `X_infty`, evaluate the fitted curve *inside the observed range*
  at a late traverse `t_late`, per bootstrap iteration, then take the ratio.
  `value(t_late)` is well-constrained by the data there, so the ratio CI is
  near-symmetric. The row is renamed `D_late` / `E_late`
  (`config.PARAM_LATEX` carries the labels). No refit — computed from the saved
  per-iteration param draws.
  - **`t_late` = the minimum observed max-traverse across the groups being
    compared** (= 44 for the two-day tables), so it is interpolated, not
    extrapolated, for every group. For independent cross-cohort ratios it is
    `min(tmax_numerator, tmax_denominator)`.
  - **Validation:** at `t_late` the fitted curve tracks the pooled data, and a
    fully model-free matched-late-window mean (the trailing `w=4` traverses ending
    at `t_late`, averaged within animal then across animals) gives a consistent
    ratio — Day2-1 Mask A duration `X_late` 0.40 vs model-free 0.35, turn error
    0.58 vs 0.57 — so `X_late` reflects real late performance, not a fit artifact.
    This comparison is now computed for every Day-2 group and saved (see "Robustness
    check").

- **Rate → NaN-saturated.** Bootstrap draws within `eps_frac=1e-3` of a bound are
  set to NaN before forming the ratio (`_nan_saturated`); `bootstrap_ratio_ci` /
  `bootstrap_summary_ratio_ci` already drop non-finite ratios. NaN-masking
  (rather than dropping rows) preserves iteration alignment, so it works for both
  pairwise ratios and cross-group summaries. Effect (from the saved robustness
  table): masking trims the saturated tail while the median moves little — e.g.
  Day2-1 A turn-error `epsilon` upper 29.9→23.0 (median 0.86→0.79), Day2-1 B upper
  30.7→23.2. The upper arm stays large after masking: with the flatter
  approach-conditioned Day-2 turn-error curves `epsilon` is weakly identified, so most
  of that residual width is genuine identifiability uncertainty (rate `mask_frac`
  ≤ ~11%), not saturation.

- **Initial value (`D_0`/`E_0`) is used as-is** — it is well-identified (never at
  a bound, arm ≈1).

## Table → panel map

| ratio table (`gen_curve_fits.py` builder) | figure panel |
|---|---|
| `compute_two_day_ratio_table` — `Wildtype two day param ratios` | `plot_day2.py` Overnight Mask A |
| `compute_day21_mask_cb_ratio_table` — `Wildtype day21 mask BC param ratios` | `plot_day2.py` Turn-sequence B/C |
| `compute_two_day_mask_ratio_table` — `Wildtype two day mask param ratios` | `plot_day2.py` Generalization |
| `compute_independent_ratio_table` — `Acortical A genotype param ratios` | `plot_ac_rapid.py` /Acortical |
| `compute_independent_ratio_table` — `Acortical generalization param ratios` | `plot_ac_mem_gen.py` generalization |

All show `D_late`/`E_late` (not `D_infty`/`E_infty`), NaN-saturated rates, and
raw initial values. (The on-disk cache keys retain the `param ratios` name for
backward compatibility even though the panels are now "curve-derived".)

## Reported diagnostics

Every ratio row now carries three diagnostic columns (added in `gen_curve_fits.py`
via `_saturation_fraction` / `_ratio_dropped_fraction`; also printed to the run log):

- **`sat_frac_num`, `sat_frac_den`** — fraction of the numerator's / denominator's
  *raw* fit-parameter draws (that row's parameter at its own bounds) sitting on a
  bound. For an asymptote row this is the `X_infty` saturation that motivates using
  `X_late`; for a rate row it is the `delta`/`epsilon` at-bound fraction that gets
  NaN-masked; for the initial value it is ~0.
- **`mask_frac`** — fraction of paired bootstrap iterations dropped as non-finite
  when forming that ratio (failed-fit NaNs + saturation NaNs + zero-denominator
  infs), mirroring what `bootstrap_ratio_ci` / `bootstrap_summary_ratio_ci` discard.

### Findings (seed 0, 1000 iterations)

- **Asymptote saturation is severe and is exactly what `X_late` sidesteps.** Two-day
  Day2-1 Mask A: `D_infty` **65%** floored, `E_infty` **43%**; the `X_late`/`X_0`
  rows and the model-free window (below) are unaffected. `D_0`/`E_0` never saturate
  (0%). Generalization masks show the largest rate saturation (e.g. duration `delta`
  Mask B ~79%), which is why those rows have the widest residual rate CIs.
- **Rate masking trims the saturated tail, not the signal.** `mask_frac` on the rate
  rows equals the rate saturation; masking leaves the median essentially unchanged
  while cutting the upper arm (Day2-1 A turn-error `epsilon` upper 29.9→23.0; Day2-1 B
  30.7→23.2). Under the approach-conditioned metric these turn-error `epsilon` ratios
  are intrinsically wide — the flat Day-2 curves barely identify a decay rate — so a
  large upper arm remains after masking; that is genuine uncertainty, not boundary
  inflation.

## Robustness check

`gen_curve_fits.py` writes a diagnostic table `"Wildtype two day {metric} ratio
robustness"` (combined: `"Wildtype two day ratio robustness"`) — it does **not**
change any shipped ratio. For every Day-2 group vs the Day-1 A reference it reports:

- **`Xlate_ratio` vs `modelfree_ratio`** — the curve-derived `X_late` ratio against a
  fully model-free late-window ratio (`_late_window_mean`: mean `Value` over the last
  `w=4` traverses ending at `t_late`, animal-first then across animals). For the
  headline Day2-1 Mask A the two agree — duration 0.40 vs 0.35, turn error 0.58 vs
  0.57 — and they track in direction across the other groups, confirming `X_late` is
  data-driven, not a fit artifact.
- **`rate_masked`/`rate_masked_hi` vs `rate_unmasked`/`rate_unmasked_hi`** — the rate
  ratio median and 97.5th percentile with vs without boundary NaN-masking. Medians
  move negligibly; the unmasked upper percentile is systematically larger (the
  saturation tail masking removes).
