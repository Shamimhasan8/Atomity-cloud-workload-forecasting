"""Unit tests for cwfc.models and cwfc.evaluate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.evaluate import (
    CVResult,
    cv_summary_table,
    empirical_interval,
    horizon_errors,
    rolling_origin_folds,
    run_cv,
    select_model,
)
from cwfc.features import FeatureSpec
from cwfc.models import (
    DirectRidge,
    Drift,
    HistoricalMean,
    HGBAR,
    HoltDamped,
    MovingAverage,
    Persistence,
    RidgeAR,
    SES,
    SeasonalNaive,
    build_candidate_pool,
)

LAG_SPEC = FeatureSpec(name="lag")


def _series(n: int = 90, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        9.0 + rng.normal(0, 0.2, size=n).cumsum() * 0.02,
        index=pd.RangeIndex(n) * 300,
        name="aggregate_cpu_utilization",
    )


class TestBaselines:
    def test_persistence(self):
        y = _series()
        out = Persistence().fit(y).forecast(24)
        np.testing.assert_allclose(out, y.iloc[-1])

    def test_historical_mean(self):
        y = _series()
        out = HistoricalMean().fit(y).forecast(5)
        np.testing.assert_allclose(out, y.mean())

    def test_moving_average_window(self):
        y = _series()
        out = MovingAverage(12).fit(y).forecast(3)
        np.testing.assert_allclose(out, y.iloc[-12:].mean())

    def test_drift(self):
        y = pd.Series([0.0, 1.0, 2.0, 3.0], index=pd.RangeIndex(4) * 300)
        out = Drift().fit(y).forecast(2)
        np.testing.assert_allclose(out, [4.0, 5.0])

    def test_seasonal_naive(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.RangeIndex(4) * 300)
        out = SeasonalNaive(2).fit(y).forecast(3)
        np.testing.assert_allclose(out, [3.0, 4.0, 3.0])

    def test_ses_flat_forecast(self):
        y = pd.Series([5.0] * 40, index=pd.RangeIndex(40) * 300)
        out = SES(0.3).fit(y).forecast(4)
        np.testing.assert_allclose(out, 5.0, atol=1e-6)

    def test_holt_damped_converges(self):
        y = pd.Series(np.linspace(1, 5, 60), index=pd.RangeIndex(60) * 300)
        out = HoltDamped().fit(y).forecast(24)
        assert len(out) == 24
        assert np.isfinite(out).all()
        assert out[-1] <= out[0] + 1e-9 or out[-1] >= out[0]  # damped: no explosion
        assert np.all(np.abs(out) < 1000)


class TestSupervised:
    def test_ridge_ar_forecast_reasonable(self):
        y = _series(90)
        fc = RidgeAR(1.0, LAG_SPEC).fit(y).forecast(24)
        assert len(fc) == 24 and np.isfinite(fc).all()
        assert np.all(fc > 5) and np.all(fc < 15)

    def test_ridge_ar_deterministic(self):
        y = _series(90, seed=11)
        a = RidgeAR(10.0, LAG_SPEC).fit(y).forecast(24)
        b = RidgeAR(10.0, LAG_SPEC).fit(y).forecast(24)
        np.testing.assert_array_equal(a, b)

    def test_hgb_deterministic(self):
        y = _series(90, seed=11)
        a = HGBAR(LAG_SPEC).fit(y).forecast(24)
        b = HGBAR(LAG_SPEC).fit(y).forecast(24)
        np.testing.assert_array_equal(a, b)

    def test_direct_ridge(self):
        y = _series(90)
        fc = DirectRidge(10.0, LAG_SPEC).fit(y).forecast(24)
        assert len(fc) == 24 and np.isfinite(fc).all()

    def test_constant_series(self):
        """All models must handle the constant active_vm_count series."""
        y = pd.Series([100.0] * 60, index=pd.RangeIndex(60) * 300)
        seg = pd.DataFrame(
            {"A": np.full(60, 100.0)}, index=y.index
        )
        for model in build_candidate_pool():
            fitted = model.fit(y, segments=seg)
            out = fitted.forecast(24)
            assert len(out) == 24
            assert np.isfinite(out).all()
            if model.name in ("persistence", "historical_mean", "ma_6", "ses_0.3"):
                np.testing.assert_allclose(out, 100.0, atol=1e-6)

    def test_pool_without_segments_excludes_segment_models(self):
        pool = build_candidate_pool(segments_available=False)
        names = [m.name for m in pool]
        assert not any("seg" in n for n in names)
        assert "ridge_lag_a1" in names and "hgb_lag" in names


class TestRollingOriginCV:
    def test_folds_count_and_bounds(self):
        folds = rolling_origin_folds(110, horizon=24, min_train=50, step=6)
        assert [f.val_start for f in folds] == [50, 56, 62, 68, 74, 80, 86]
        assert folds[-1].val_end == 110

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            rolling_origin_folds(60, horizon=24, min_train=50)

    def test_agrees_with_starter_implementation(self):
        """Our positional folds must match the starter's timestamp-based helper."""
        from cwfc.validation import rolling_origin_splits

        n, horizon, min_train, step = 60, 12, 20, 6
        frame = pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")}
        )
        starter = rolling_origin_splits(frame, horizon=horizon, min_train_size=min_train, step=step)
        ours = rolling_origin_folds(n, horizon=horizon, min_train=min_train, step=step)
        assert len(starter) == len(ours)
        for s, o in zip(starter, ours):
            assert s.validation_start == frame["timestamp"].iloc[o.val_start]
            assert s.validation_end == frame["timestamp"].iloc[o.val_end - 1]

    def test_run_cv_shapes_and_selection(self):
        y = _series(110, seed=5)
        folds = rolling_origin_folds(110, horizon=24, min_train=50, step=6)
        candidates = [Persistence(), MovingAverage(12), SES(0.3), RidgeAR(1.0, LAG_SPEC)]
        results = run_cv(y, candidates, folds)
        assert set(results) == {"persistence", "ma_12", "ses_0.3", "ridge_lag_a1"}
        for res in results.values():
            assert len(res.preds) == 7
            for k in res.preds:
                assert res.preds[k].shape == (24,)
                assert res.actuals[k].shape == (24,)

        table = cv_summary_table(results)
        selected, _ = select_model(results)
        assert selected == table.iloc[0]["model"]
        assert set(table.columns) >= {"MAE", "RMSE", "peak_MAE", "MAE_vs_baseline"}

    def test_summary_metrics_values(self):
        """CVResult.summary computes the documented metrics correctly."""
        result = CVResult(
            name="m", complexity=1,
            preds={0: np.array([1.0, 2.0])},
            actuals={0: np.array([2.0, 4.0])},
            is_peak={0: np.array([False, True])},
        )
        s = result.summary()
        assert s["MAE"] == pytest.approx(1.5)
        assert s["RMSE"] == pytest.approx(np.sqrt((1 + 4) / 2))
        assert s["peak_MAE"] == pytest.approx(2.0)
        assert s["n_peak"] == 1

    def test_zero_denominator_handled(self):
        """normalized_mae must not divide by zero (starter contract)."""
        from cwfc.metrics import normalized_mae

        y = [1.0, 2.0]
        perfect = [1.0, 2.0]
        assert normalized_mae(y, perfect, perfect) == 0.0
        assert normalized_mae(y, [3.0, 4.0], perfect) == float("inf")

    def test_empirical_interval_and_horizon_errors(self):
        y = _series(110, seed=9)
        folds = rolling_origin_folds(110, horizon=24, min_train=50, step=6)
        results = run_cv(y, [MovingAverage(12)], folds)
        res = results["ma_12"]
        lower, upper = empirical_interval(res, alpha=0.10)
        assert len(lower) == len(upper) == 24
        assert np.all(lower <= upper)
        hz = horizon_errors(res)
        assert len(hz) == 24
        assert (hz["horizon"].to_numpy() == np.arange(1, 25)).all()


class TestEmpiricalIntervalSignConvention:
    """Regression tests for the interval sign bug found in review.

    Residuals are ``r = prediction - actual``.  The interval for the actual
    value around a forecast ``f`` must therefore be
    ``[f - Q_hi(r), f - Q_lo(r)]`` (actual = forecast - residual).  The old
    implementation returned ``[f + Q_lo(r), f + Q_hi(r)]`` — the wrong side
    of the forecast whenever residuals are biased.
    """

    def _result_from_residuals(self, resid_by_fold: dict[int, np.ndarray]) -> CVResult:
        preds, actuals, peaks = {}, {}, {}
        for k, r in resid_by_fold.items():
            base = np.arange(len(r), dtype=float)
            preds[k] = base + r
            actuals[k] = base
            peaks[k] = np.zeros(len(r), dtype=bool)
        return CVResult("m", 1, preds, actuals, peaks)

    def test_overforecasting_residuals_put_band_below_forecast(self):
        # model consistently over-forecasts: r = pred - actual > 0
        resid = {0: np.array([1.0, 0.8]), 1: np.array([1.2, 0.9]),
                 2: np.array([0.9, 1.1]), 3: np.array([1.0, 1.05])}
        res = self._result_from_residuals(resid)
        lo, hi = empirical_interval(res, alpha=0.20)
        for h in range(2):
            r_h = np.array([resid[k][h] for k in sorted(resid)])
            q_lo = np.quantile(r_h, 0.10)
            q_hi = np.quantile(r_h, 0.90)
            # the documented convention: lower = -Q_hi, upper = -Q_lo
            assert lo[h] == pytest.approx(-q_hi)
            assert hi[h] == pytest.approx(-q_lo)
            # band for the actual value sits BELOW the forecast
            assert lo[h] < 0 and hi[h] < 0

    def test_underforecasting_residuals_put_band_above_forecast(self):
        resid = {0: np.array([-1.0, -0.8]), 1: np.array([-1.2, -0.9]),
                 2: np.array([-0.9, -1.1]), 3: np.array([-1.0, -1.05])}
        res = self._result_from_residuals(resid)
        lo, hi = empirical_interval(res, alpha=0.20)
        assert np.all(lo > 0) and np.all(hi > 0)

    def test_asymmetric_residuals_cover_the_right_mass(self):
        # skewed residuals: mostly small positive, one large positive per step
        # (4 folds x 2 horizons; the h=0 pool across folds is the first value
        # of each fold, the h=1 pool the second)
        resid = {
            0: np.array([0.1, 0.1]),
            1: np.array([0.2, 0.1]),
            2: np.array([0.15, 0.12]),
            3: np.array([2.0, 1.8]),
        }
        res = self._result_from_residuals(resid)
        lo, hi = empirical_interval(res, alpha=0.20)
        # actual at (k, h) equals the base value h; prediction = h + r_kh.
        # The defining property: r inside [q_lo, q_hi] <=> actual inside the
        # band applied to that fold's forecast; extreme over-forecasts must
        # leave the actual BELOW the band (never above it).
        for k in sorted(resid):
            for h in range(2):
                r = resid[k][h]
                pred, actual = h + r, float(h)
                lower, upper = pred + lo[h], pred + hi[h]
                r_pool = [resid[j][h] for j in sorted(resid)]
                q_lo, q_hi = np.quantile(r_pool, 0.1), np.quantile(r_pool, 0.9)
                if q_lo <= r <= q_hi:
                    assert lower <= actual <= upper
                elif r > q_hi:
                    assert actual < lower  # extreme over-forecast: actual below band
                else:
                    assert actual > upper  # extreme under-forecast: actual above band
        assert lo[0] == pytest.approx(-np.quantile([0.1, 0.2, 0.15, 2.0], 0.9))
        assert hi[0] == pytest.approx(-np.quantile([0.1, 0.2, 0.15, 2.0], 0.1))
