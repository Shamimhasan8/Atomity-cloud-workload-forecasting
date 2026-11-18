# Quality Assurance Record — Review Findings, Corrections and Verification

**Project:** Cloud Workload Forecasting — Atomity AI/ML Engineering take-home (candidate: Adil Shamim)
**Scope of this document:** an independent technical review of the initial submission identified the issues below. Every issue was reproduced in the sandbox before fixing, corrected, and verified with the evidence quoted. The frozen prediction model and `predictions.csv` were **not** modified at any point — verified bit-identical before and after all corrections.

---

## Issue 1 — Notebook execution errors (2× `ValueError: Unknown format code 'f' for object of type 'str'`)

- **Where:** `notebooks/03_evaluation.ipynb`, headline-results cell and forecast-horizon cell.
- **Root cause:** pandas `Styler.format` was given numeric format strings for (a) a mixed-type column ("MAE vs baseline" held both floats and the string "undefined (baseline MAE = 0)") and (b) the whole frame including the string-valued "horizon" column. The executed notebook had silently recorded two error outputs (nbconvert completed without failing the conversion), which the original verification — which inspected only stream outputs — missed.
- **Correction:** numeric formatting is applied only to numeric columns (a format dict keyed by numeric columns; the 0/0 case is now `NaN` with `na_rep="undefined (0/0)"`). All three notebooks were re-executed from clean kernels.
- **Verification:** (i) `pytest -q` includes `tests/test_notebooks.py`, which executes each notebook via nbconvert in a fresh kernel and **fails on any `output_type == "error"` cell output**; (ii) a direct scan of the executed notebooks reports zero error outputs; (iii) executed-output counts confirm all code cells produced outputs.

## Issue 2 — Prediction intervals computed with the wrong sign convention

- **Where:** `src/cwfc/evaluate.py::empirical_interval`.
- **Root cause:** residuals are defined as `r = prediction − actual`, so the interval for the actual value is `[forecast − Q_hi(r), forecast − Q_lo(r)]` (actual = forecast − residual). The old implementation returned the raw quantiles, and callers *added* them to the forecast — placing the band on the wrong side of the forecast whenever residuals were biased. The error was masked in the original submission because the CV residuals happened to be nearly symmetric.
- **Reproduction (asymmetric residuals):** residuals all in [0.8, 2.0] (persistent over-forecast) produced a band `[forecast+0.86, forecast+1.70]` — entirely above a forecast whose actuals all lie below it. Correct band: `[forecast−1.70, forecast−0.86]`.
- **Correction:** `empirical_interval` now returns offsets `lower = −Q_{1−α/2}(r)`, `upper = −Q_{α/2}(r)` with the convention documented in the docstring; all callers add the offsets to the forecast.
- **Verification:** new tests `tests/test_models.py::TestEmpiricalIntervalSignConvention` — over-forecasting residuals put the band below the forecast, under-forecasting above, and the defining property `r ∈ [q_lo, q_hi] ⟺ actual ∈ [pred+lower, pred+upper]` is asserted on asymmetric residuals, with extreme misses on the correct side. **Regenerated results:** holdout band coverage 62% → **67% (16/24)**; 7 of 8 misses on the upper side (late-horizon recovery), 1 on the lower side; band half-width unchanged (0.224 pp) but now correctly positioned (mean offsets −0.213 / +0.235 pp). Figures 14/15 and notebook 02/03 bands regenerated.

## Issue 3 — Stale segment features in `features_at_origin`

- **Where:** `src/cwfc/features.py::features_at_origin`.
- **Root cause:** during recursive forecasting, the extended history contains timestamps beyond the segment frame's coverage. `build_feature_frame` reindexed the segment frame (NaN at future timestamps), the lagged segment features at the origin row became NaN, and the warm-up `dropna` **silently removed the origin row** — so `frame.iloc[-1]` returned an *earlier, stale* row. From recursion step 2 onward the same stale row was returned every time: a segment-based ridge produced **24 identical predictions** (verified: `n_unique = 1`), and its CV MAE of 0.31–0.33 was an artefact of that degenerate flat forecast.
- **Correction:** future segment means are unknown at forecast time and are now held at their **last observed values** (forward-fill of the segment frame onto the extended index — the standard leakage-safe last-known-value treatment for exogenous covariates in recursive forecasting). In addition, the function now **raises `RuntimeError` if the returned row's timestamp is not the next timestamp**, so a stale row can never be returned silently. A non-overlapping segment frame raises a clear `ValueError`.
- **Verification:** new tests `tests/test_features.py::TestSegmentFeatureOrigin`: origin row equals the next timestamp (both specs); returned timestamps strictly advance across simulated recursion steps (24300 → 24600 → 24900 …); segment lag features resolve to last observed values; non-degenerate segment forecasts (> 3 unique values; was 1). **Recomputed comparison (training-window validation only):** the segment-augmented recursive ridge now shows its true behaviour — recursive divergence, CV MAE ≈ 5.4×10³ (α = 10) and ≈ 2.7×10⁶ (α = 1). Every non-segment candidate's CV numbers are bit-identical to the pre-fix run; the frozen selection (`ridge_lag_a100`) remains the CV winner; the frozen predictions are unchanged. Figure 2 now clips the diverging bars with annotation.

## Issue 4 — Capacity policy calibrated on holdout-contaminated pooled errors

- **Where:** notebook 03 §8 (and report §10 / README).
- **Root cause:** the safety margin was sized from the pooled absolute errors of CV **plus the holdout**, i.e. the holdout influenced the policy — post-hoc contamination. Under-provisioning risk was also measured as absolute error rather than signed demand shortfall.
- **Correction:** the policy is now derived in two strictly separated steps. **Calibration:** training-window CV residuals only (168 signed steps), with `e = actual − forecast`, shortfall `max(e − s, 0)`, idle `max(s − e, 0)`; margin s\* = 0.5 pp chosen from the CV curves. **Independent evaluation:** the frozen policy is evaluated once on the sealed holdout and reported as measurement, never fed back. Post-hoc references (e.g. the historical-mean row) are labelled as such.
- **Verification (measured):** CV calibration — s = 0.5 pp gives 97.6% no-shortfall probability, E[shortfall] 0.0009 pp, E[idle] 0.5152 pp (s = 0.35 pp: 92.9% / 0.0089 / 0.3732). Holdout evaluation — s = 0.5 pp: **0 of 24 shortfall steps** (worst under-forecast +0.4976 pp at step 16, covered with 0.0024 pp to spare), E[idle] 0.3793 pp; s = 0.35 pp: 2/24 shortfall steps, max 0.1476 pp. Figure 4 regenerated as a two-panel design (calibration curve | holdout evaluation).

## Issue 5 — Report described the holdout bias with the wrong direction

- **Where:** report §9 (and notebook 03 §7 narrative).
- **Root cause:** the report said the ridge "over-forecast by 0.12 pp on average". The measured facts: forecast mean 9.148% < holdout mean 9.269%, signed bias (forecast − actual) = **−0.121 pp** — both the frozen model and persistence **under-forecast** the shifted regime (persistence bias −0.104 pp). The holdout mean sat *between* persistence's anchor (9.165%) and the training mean (9.414%); the ridge's additional downward drift across the horizon (9.21% → 9.09%) made its under-forecast slightly larger.
- **Correction:** §9 rewritten with the correct direction and mechanism; notebook 03 §7 rewritten to match; a full numerical audit of README, notebooks, report (md + pdf) and figures was performed against executed outputs.
- **Verification:** automated scans of the final PDF/README/notebooks confirm the corrected claims ("UNDER-forecast", 67%, 97.6%, 0/24) and the absence of stale ones (62%, 97.9%, "over-forecast").

## Issue 6 — Model-development EDA exposed the holdout

- **Where:** notebook 01 (and report §5).
- **Root cause:** the EDA displayed the full 134-timestamp series with the holdout shaded in descriptive plots (fig01, fig05, fig07), and the distribution/heavy-hitter statistics used all 1.34 M readings.
- **Correction:** **all EDA is now computed on the 110-timestamp training window exclusively** — holdout values are never loaded, plotted or summarized during model development; the constant-count fact is justified structurally (per-VM completeness) rather than by reading holdout values; holdout visualizations live only in the final evaluation notebook (which unseals the holdout after freezing). Figures 1/5/7 regenerated on the training window.
- **Verification:** the EDA notebook prints its scope ("training window = 110 timestamps … 1,100,000 readings"); all figures verified regenerated and non-blank; the frozen model, features and `predictions.csv` were untouched (no reselection or retuning occurred at any point — confirmed by the byte-identical predictions check below).

## Additional defects found and fixed during this revision (not in the review)

7. **`adaptive_split` crashed for datasets with more than 134 timestamps** (holdout index length ≠ horizon → DataFrame length mismatch). Now the holdout is always the final 24 timestamps (train = n − 24), which reproduces the official 110/24 exactly at n = 134. Tests added for n = 140 and n = 200.
8. **Always-true assertion removed** (`assert "duplicate" in ... or True` in `test_smoke_dataset_end_to_end`) — replaced with a real assertion on the loader's preprocessing notes (this had masked the fact that the smoke fixture's invalid reading rides on the duplicated row and is dropped with it).

## New tests added (51 → 67)

| Area | Tests |
|---|---|
| Prediction-interval correctness | `TestEmpiricalIntervalSignConvention` — 3 tests (over-forecast, under-forecast, asymmetric defining property) |
| Segment feature availability | `TestSegmentFeatureOrigin` — 5 tests (next-timestamp, advancing under recursion, last-known-value semantics, loud failure, non-degenerate forecast) |
| Recursive forecasting | strengthened `test_forecast_does_not_read_future_values` (anti-vacuous: input changes must change the forecast; dead code removed) |
| Holdout perturbation invariance | `test_holdout_perturbation_invariance` — scrambling holdout readings leaves predictions byte-identical (140-timestamp synthetic dataset) |
| Missing timestamp handling | 2 tests — all-invalid timestamp → gap row + interpolation warning; absent timestamp → grid skips, predictions align to observed grid |
| Prediction timestamp alignment | `test_larger_than_official_dataset` (n = 140) asserts exact final-24 timestamps from the source grid |
| Notebook execution | `tests/test_notebooks.py` — 3 tests, one per notebook, executed from clean kernels; fails on any cell error; self-skips without dataset/Jupyter |
| Preprocessing notes assertion | smoke end-to-end now asserts the duplicate-handling note (replacing the always-true assertion) |

## Final verification (fresh environment, official `data-v1` dataset)

| Check | Result |
|---|---|
| Fresh virtualenv install (`pip install -e ".[dev]"` + `.[notebooks]`) | succeeds |
| Full test suite (`python -m pytest -q`) | **67 passed** (incl. 3 notebook-execution tests, ~1 min) |
| All notebooks executed from clean kernels | zero error outputs |
| `python -m cwfc.predict --data challenge_data --output predictions.csv` | validates and writes 24 rows; **byte-identical** to the committed `predictions.csv` and to the pre-correction output |
| Frozen model untouched | `FROZEN_CONFIG` unchanged (persistence; ridge α = 100, lag spec); `--select` reproduces the same selection |
| PDF report QA (metadata, page size, overflow, fill ratio, fonts, punctuation, blank pages) | all checks pass; 12 sections, 4 figures; corrected numbers present; no placeholders |
| Repository hygiene | no credentials, no `.venv`/caches/egg-info/pytest-cache committed, `challenge_data/` contents excluded (placeholder README only), intermediates (`report_body.pdf`, `cover.pdf`) ignored |

**Honest deviations noted:** the PDF report runs 6 content pages plus a references page (target was 4–6; the corrections documentation in §11 was deliberately kept rather than cut); the empirical interval remains coarse (7 folds) and under-covers (67% vs nominal 80%) — reported as measured.
