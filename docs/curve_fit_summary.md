# Curve-fit summary: choices and sample sizes

A single scannable reference for the exponential learning-curve fits used throughout the
manuscript figures: the model and its bounds, how the fits are produced, the
late-performance substitution, an **authoritative per-curve sample-size table**, the small-n
policy, and a digest of the fit-quality diagnostics. This doc consolidates and cross-links;
it does not restate the derivations. For the full treatment see
[`ratio_ci_method.md`](ratio_ci_method.md), [`notation_guide.md`](notation_guide.md),
and [`data_contracts.md`](data_contracts.md#12-figure_data-files).

---

## 1. The model

Each learning curve is a saturating exponential in traverse index `b` (1-based; at `b=1` the
curve returns `X_0` exactly):

```
value(b) = X_∞ + (X_0 − X_∞) · exp[−k (b − 1)]
```

fit by nonlinear least squares (`manhattan_maze/curve_fit.py`, `exponential_func`). Parameters,
bounds, and initial values come from `config.CURVE_FIT_SPECS`:

| Metric | Params | Initial `p0` | Lower | Upper |
|--------|--------|--------------|-------|-------|
| duration (s) | `D_∞`, `D_0`, `δ` | 20, 200, 0.1 | 2, 5, 0.01 | 60, 800, 1 |
| turn error rate | `E_∞`, `E_0`, `ε` | 0.1, 0.5, 0.1 | 0.001, 0.1, 0.01 | 0.5, 1, 1 |

`X_∞` = asymptote, `X_0` = initial value, `k` (`δ`/`ε`) = learning rate (1/traverse). See
[`notation_guide.md` § Learning-curve models](notation_guide.md#learning-curve-models)
(Eqs. 1–2, 16) and
[`ratio_ci_method.md` § The learning-curve model](ratio_ci_method.md#the-learning-curve-model).
Turn error rate is first-decision-per-hole, approach-conditioned (chance 0.5).

## 2. Fitting procedure

Fits are produced by an **animal-level bootstrap**
(`manhattan_maze/bootstrap.py`, `fit_traverse_data_df_with_bootstrap`):

- **1000 iterations**, whole animals resampled with replacement (repeated-measures dependence
  is handled here, not by a formal independence test).
- Reproducible with a fixed seed (`gen_curve_fits.py`, seed 0).
- Central curve = **bootstrap median**; 95% CIs = 2.5th/97.5th percentiles. Non-converged
  iterations are skipped.

Pipeline: each producer (`gen_*.py`) saves a `"<base> fit input"` payload (tidy
`data_df` with columns `Animal, b, Value`); `scripts/gen_curve_fits.py` refits every input and
writes `"<base> fit results"` = `(bs, ds, summary_df, bootstrap_curves)` plus
`"<base> bootstrap params"` (per-iteration draws for ratio CIs). See
[`data_contracts.md` § 12](data_contracts.md#12-figure_data-files), "Fitted curve tuple".

## 3. Asymptote → `X_late`, and rate masking

On short/re-exposure ranges the **asymptote `X_∞` is unidentifiable** — the slow-decay fit
rails it to its bound (e.g. Day2-1 Mask A: `D_∞` floored in 65% of draws, `E_∞` in 43%),
producing wildly skewed ratio CIs. Relative-magnitude comparisons therefore use a
**data-anchored late value `X_late`**: the fitted curve evaluated at an in-range traverse
`t_late` (the minimum observed max-traverse across the compared groups), computed from the
saved bootstrap draws without refitting. Learning-rate draws lying within `eps_frac` of a
bound are NaN-masked before ratios are formed. Each ratio row carries three diagnostic
columns — `sat_frac_num`, `sat_frac_den` (fraction of raw draws at a bound) and `mask_frac`
(fraction of paired iterations dropped as non-finite). Full method and numbers:
[`ratio_ci_method.md` § The method](ratio_ci_method.md#the-method) and
[§ Reported diagnostics](ratio_ci_method.md#reported-diagnostics).

## 4. Curves and sample sizes

`n` is the number of **unique animals in the fit input** (`data_df.Animal.nunique()`) — the
animals actually driving each bootstrap (eligible/successful learners), **not** the full
genotype cohort. Duration and turn-error fits share the same cohort per condition unless
noted. **Bold = n ≤ 4** (see § 5).

### First-mask learning and two-day (wildtype)

| Curve | n | Consuming figure (script) |
|-------|---|---------------------------|
| Acortical A (dur, turn err) | **4** | fig:ac_rapid (`plot_ac_rapid`), fig:ac_curve (`plot_ac_curve_fit_supp`) |
| Control A (dur, turn err) | **3** | fig:ac_curve (`plot_ac_curve_fit_supp`); raw traces in fig:ac_rapid |
| Wildtype two-day — Day-1 Mask A | 25 | fig:day2 (`plot_day2`), fig:2day_curve (`plot_curve_fit_supp`) |

Two-day per `(session, mask)` — n animals (same for duration and turn error):

| Session | A | B | C |
|---------|---|---|---|
| 1 (Day-1) | 25 | — | — |
| 2 (Day2-1) | 8 | 6 | 11 |
| 3 (Day2-2) | 8 | 10 | 6 |
| 4 (Day2-3) | 8 | 6 | 11 |
| 5 (Day2-4) | 8 | 8 | 8 |

### Generalization (acortical / control)

| Curve | n | Consuming figure |
|-------|---|------------------|
| Acortical A repeat Gen (dur, turn err) | 6 | fig:ac_mem_gen (`plot_ac_mem_gen`) |
| Acortical B Gen (dur, turn err) | **4** | fig:ac_mem_gen |
| Acortical C Gen (dur, turn err) | 6 | fig:ac_mem_gen |
| Acortical D Gen = Acortical D (dur) | 6 | fig:ac_mem_gen, fig:ac_curve |
| Acortical E first (dur, turn err) | 6 | fig:ac_ef_supp (`plot_ef_gen_supp`) |
| Acortical Mask E Gen (dur, turn err) | 8 | fig:ac_ef_supp |
| Acortical F Gen (dur, turn err) | **4** | fig:ac_ef_supp |
| Acortical A after E (dur, turn err) | **4** | Mask-E transfer reference |
| Control A repeat Gen (dur, turn err) | 6 | fig:ac_curve |
| Control B Gen (dur, turn err) | **4** | fig:ac_curve |
| Control C Gen (dur, turn err) | 6 | fig:ac_curve |
| Control D = Control D Gen (dur) | 8 | fig:ac_mem_gen, fig:ac_curve |
| Control E first (dur, turn err) | 5 | fig:ac_ef_supp |
| Control Mask E Gen (dur, turn err) | 5 | fig:ac_ef_supp |
| Control A after E (dur, turn err) | **3** | Mask-E transfer reference |

### Mask D and olfaction swap (wildtype)

| Curve | n | Consuming figure |
|-------|---|------------------|
| Wildtype D (dur) | 6 | fig:ac_mem_gen, fig:ac_curve |
| Wildtype Pre-swap (dur, turn err, + outbound/homebound turn err) | 8 | fig:olfaction (`plot_olfaction`) |
| Wildtype Post-swap (dur, turn err, + outbound/homebound turn err) | 7 | fig:olfaction |

## 5. Small-n handling and caveats

> A bootstrap over `n` animals has only `C(2n−1, n)` distinct resamples — 10 for n=3, 35 for
> n=4 — so CIs for these curves are wide and quantized, and the asymptote/rate parameters are
> especially unreliable. Read them as point estimates with weak uncertainty, not as
> comparable to the n≥8 curves.

Curves with **n ≤ 4**: Control A (3), Control A-after-E (3), Acortical A (4),
Acortical A-after-E (4), Acortical B Gen (4), Acortical F Gen (4), Control B Gen (4).

Established policy for the n=3 control cohort (Mask A):

- **fig:ac_rapid** does not fit control — it shows raw per-animal traces and drops the
  Control/Acortical ratio (the ratio would inherit the degenerate n=3 CI).
- **fig:ac_curve** panels A/B use per-animal parameter points rather than a bootstrap CI.

The **Mask B generalization** comparison rests on **n=4 per genotype** (Acortical B Gen and
Control B Gen) — the smallest generalization contrast; interpret its `D_0`/`E_0` shifts
accordingly.

## 6. Fit quality (diagnostics digest)

Headline results from the curve-fit diagnostics pass (per-fit tables and residual panels
were produced in a diagnostics notebook that is not distributed).

- **Discovery (§A):** 39 fit inputs, each with `data_type`, `n_points`, `n_animals` (the
  authoritative source for § 4 above).
- **Goodness of fit (§B):** report `R²_trend` (curve vs per-traverse means), median ≈ 0.52,
  range [0.19, 0.84]; raw pooled `R²` is lower by construction and not the headline metric.
- **Functional form (§C):** by AIC, exponential is best for 21/39 fits; where it loses,
  ΔAIC < 6 (indistinguishable from linear/piecewise). A turn-error-only re-check under the
  method-2 metric confirms the exponential should stay — see [§ 7](#7-turn-error-form-re-check-under-method-2-2026-07-27).
- **Structure (§C2):** across the two-day data, Day and Mask act on **initial performance**
  with a **single shared learning rate** — richer rate-modulating models do not improve AIC.
  This justifies the one-rate-per-group exponential.
- **Residuals (§D):** no systematic tail structure; the mean fit leaves no trend behind.
- **Nonparametric agreement (§E):** the curve-derived `X_late` ratio tracks a model-free
  late-window ratio (e.g. Day2-1 Mask A: duration 0.40 vs 0.35, turn error 0.58 vs 0.57);
  across well-supported fits the median absolute difference is ≈ 24%.
- **Not reported (§F):** formal independence/normality tests — standard for learning curves
  but misleading under repeated measures, which the animal-level bootstrap already handles.

Conclusion: the saturating exponential is adequate and well-identified for the initial value
and rate; the asymptote is not identifiable, which is why ratios use `X_late`.

## 7. Turn-error form re-check under method 2 (2026-07-27)

**Motivation.** The turn-error metric is now first-decision-per-hole, approach-conditioned
(`include="first"`, chance 0.5). Because that measure starts **below chance** (initial rate
≈ 0.16–0.55, median ≈ 0.32, not the ~0.5 the fit's `p0` assumes) and has a small dynamic
range, we re-verified two things on freshly regenerated method-2 data: (a) whether adjusting
the exponential's bounds/`p0` improves the fit, and (b) whether a different functional form
(linear, piecewise) should replace the exponential for the manuscript. Comparison is
turn-error-only, 21 curves; the primary check is the well-powered Day-1 Mask-A curve.
(The exponential/piecewise/linear fitters mirror those in
`manhattan_maze/curve_fit.py`.)

**(a) Bounds/`p0` do not help.** Refitting every curve with a below-chance-anchored spec
(`E_0 p0 0.5→0.35`, `E_0` floor `0.1→0.05`, looser `ε`/`E_∞`) changed AIC by a mean of
**0.01** and improved **0/21** curves (ΔR²_trend = 0.000). The optimizer already reaches the
same optimum — the current bounds were never clipping it. The persistent railing of `E_∞`
to its floor (~8/21) and `ε` to its floor is **intrinsic to the shallow, below-chance data
shape**, not a bounds artifact, and is already handled by reporting `X_late` (§ 3), not `E_∞`.
→ **No `config.CURVE_FIT_SPECS` change is warranted.**

**(b) Exponential stays.** Three-way AIC across all 21 turn-error curves nominally splits
**linear 12 / exp 5 / piecewise 4**, but the margins are tiny (mostly |ΔAIC| < 4 on
n≈100–300 *pooled single-traverse* points) and driven by linear's one-fewer-parameter
parsimony credit — exp still has the highest **mean R²_trend (0.461 vs piecewise 0.443 vs
lower for linear)**. The decision is settled by the primary curve and by physical validity:

| Day-1 Mask-A curve | n (pooled obs) | exp AIC / R²_trend | piecewise | linear | winner |
|--------------------|----------------|--------------------|-----------|--------|--------|
| Wildtype two-day Day-1 Mask A | 1354 | **−5553 / 0.754** | −5539 / 0.693 | −5487 / 0.648 | **exp** (ΔAIC 14 vs pw, 66 vs lin) |
| Acortical A (n=4 animals) | 130 | −485 / 0.616 | −488 / 0.660 | −480 / 0.521 | pw (ΔAIC 3.5) |
| Control A (n=3 animals) | 132 | −572 / 0.383 | −573 / 0.392 | −574 / 0.371 | tie (all within 2 AIC) |

- On the **well-powered Day-1 Mask-A wildtype curve the exponential wins outright** — highest
  R²_trend and ΔAIC ≥ 14 over both alternatives.
- Where linear/piecewise nominally win, it is only on the small **n≤4 genotype-control**
  curves where all three models are visually indistinguishable *within* the observed range.
- **Linear is physically invalid for a rate:** with no floor it drives the predicted turn
  error **negative** just past the observed range on every curve, and has **no asymptote**,
  which would break the `X_late`/τ machinery (§ 3) the learning claims depend on. Piecewise is
  a worse-conditioned twin of the exponential (legacy, `_piecewise_func_legacy`) and never
  decisively beats it on the primary curve.

**(c) Outbound/homebound split — helps the well-powered curves, not the small-n ones.**
Each traverse has an outbound (`H-O`, home→reward) and homebound (`O-H`, reward→home) leg;
`Session.filter("H-O"/"O-H")` splits them (as `gen_olfaction.py` already does for the swap
curves). Fitting the legs separately halves the points per fit but removes a **misspecification**
— the two directions learn at different rates, so one pooled exponential leaves structured
residual. Fit quality (exp R²_trend) on freshly recomputed method-2 data:

| Curve | animals | pooled | outbound (n, ε) | homebound (n, ε) |
|-------|---------|--------|-----------------|------------------|
| Wildtype Day-1 Mask-A | 25 | 0.754 (1354) | **0.901** (679, ε=0.039) | **0.832** (675, ε=0.122) |
| Acortical A | 4 | 0.616 (130) | 0.712 (65) | 0.516 (65) |
| Control A | 3 | 0.383 (132) | 0.273 (66) | 0.591 (66) |
| Wildtype Post-swap (olfaction) | 7 | 0.532 (235) | 0.709 (118) | 0.373 (117, ε rails →1) |

- **Well-powered curves gain a lot and reveal real structure:** on Day-1 Mask-A both legs fit
  markedly better than pooled (0.90 / 0.83 vs 0.75) because **homebound learns faster** (ε≈0.12)
  than outbound (ε≈0.04) — a direction dissociation that pooling blurs. Exp still beats piecewise
  on every leg.
- **Small-n curves do not benefit:** with n≤4 animals, halving the data destabilizes the fit
  (Control A outbound R²_trend *drops* to 0.27; homebound legs rail ε→1 in the olfaction swap),
  and the § 5 bootstrap-CI caveat (10–35 distinct resamples) only worsens.

So splitting is **not** a blanket improvement — it is worth doing **selectively** on the
well-powered curves (Day-1 Mask-A; olfaction already does), where it both improves the fit and
surfaces the outbound/homebound rate dissociation, while the many n≤8 generalization/control
curves should stay pooled.

**(d) Presentation decision for Fig 2E (main figure): keep pooled, consistent.**
Fig 2E (`plot_day2.py`) is a **grid of 13 turn-error exponential panels** (Session × Mask); the
first panel (Day-1 Mask-A, n=25) is reused as the purple curve in Fig 4F. Two quantitative facts
decide whether to show a direction-split there:

- *The dissociation is robust* on Day-1 Mask-A (animal-level bootstrap, n=25, 2000 iters, all
  95% CIs exclude 0): ε_home − ε_out = **+0.080 [+0.038, +0.133]** (homebound learns ~3× faster,
  τ ≈ 8 vs 26 traverses; Δτ 95% CI [+6, +22]); E_late_home − E_late_out = −0.094 [−0.123, −0.069];
  E0_home − E0_out = −0.042 [−0.073, −0.008].
- *But it is unrepeatable across the figure:* **0/21** turn-error fit-input curves reach n ≥ 15,
  and **only 1 of the 13 Fig 2E panels** (Day-1 A) is split-ready — the other 12 are n = 6–11,
  where halving the data destabilizes the fit (per (c): Control A n=3 outbound R²_trend → 0.27,
  olfaction homebound ε rails → 1).

Splitting only the one well-powered panel while pooling the other twelve would make Fig 2E
**internally inconsistent** and invite the reviewer question "why not the rest?" — whose honest
answer is "they are underpowered", i.e. exactly the caveat. The pooled fit is not wrong (level
and E_late are well-identified; only the rate is a direction blend, and it tracks the slow
outbound leg), and the direction effect is orthogonal to Fig 2's overnight-memory claim.

**Recommendation (MVP):** keep Fig 2E (and Fig 4F) **pooled and consistent**; do **not** split in
the main figures. Because the dissociation is a real, significant result, preserve it in a
**single supplementary panel** (Day-1 Mask-A outbound vs homebound, with the bootstrap CIs above)
plus one methods sentence: *"Direction-split (outbound/homebound) exponential fits were examined;
they are only well-powered for the n=25 Day-1 dataset, so a consistent pooled treatment is used
throughout."* This is the minimal, most defensible change — no main-figure or pipeline edits.

**Conclusion.** The method-2 metric switch does **not** break the model. Keep the saturating
exponential and its current bounds/`p0` — no model swap (exp→piecewise gives ties at best and
loses the asymptote) and no bounds change (0/21 improvement) improves the paper overall. The one
change worth considering is a **targeted** outbound/homebound split of the well-powered Day-1
Mask-A curve; the earlier §6 verdict otherwise holds for turn error.

## 8. Reproducibility

- Regenerate all fits: run `python gen_curve_fits.py` in the `m_maze` env from `scripts/`
  (seed 0, 1000 iterations) — this rewrites every `"<base> fit results"` and
  `"<base> bootstrap params"`.
- Regenerate the § 4 n-table: `d["<base> fit input"]["data_df"]["Animal"].nunique()` for each
  `"* fit results"` key, and the two-day `"Wildtype two day <metric> tidy"` frame grouped by
  `(Session, Mask)`.

See also: [`ratio_ci_method.md`](ratio_ci_method.md) ·
[`notation_guide.md`](notation_guide.md) ·
[`data_contracts.md` § 12](data_contracts.md#12-figure_data-files).
