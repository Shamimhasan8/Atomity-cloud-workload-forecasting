"""Leakage-prevention tests — the guard rails for Phase 5.

These are property tests: they *perturb the future* and assert that nothing
computed for the past changes.  If a future regression introduces leakage
(e.g. a rolling window that includes the current value, or a recursive
forecaster that reads actuals), these tests fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.data import FUTURE_METADATA_FIELDS
from cwfc.features import FeatureSpec, build_feature_frame, features_at_origin, segment_means

LAG_SPEC = FeatureSpec(name="lag")
SEG_SPEC = FeatureSpec(name="seg", with_segments=True)


def _series(n: int = 60, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(10, 0.3, size=n).cumsum() * 0.05 + 10,
                     index=pd.RangeIndex(n) * 300, name="y")


class TestFeatureConstruction:
    def test_features_use_only_the_past(self):
        """Perturbing y[t:] must not change feature rows at positions <= t."""
        y = _series()
        frame = build_feature_frame(y, LAG_SPEC)
        t = 40
        row_before = frame.loc[y.index[t]].drop("target")

        y_future_changed = y.copy()
        y_future_changed.iloc[t:] += 100.0
        frame_after = build_feature_frame(y_future_changed, LAG_SPEC)
        row_after = frame_after.loc[y.index[t]].drop("target")

        pd.testing.assert_series_equal(row_before, row_after)

    def test_warmup_rows_are_dropped(self):
        """Rows without full lookback history must not appear."""
        y = _series()
        frame = build_feature_frame(y, LAG_SPEC)
        # max lookback = max(lag 24, window 12 + 1, slope 6 + 1) = 24
        assert frame.index.min() == y.index[24]
        assert frame["target"].notna().all()
        assert not frame.drop(columns="target").isna().any().any()

    def test_lag_semantics(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] + list(range(6, 40)),
                      index=pd.RangeIndex(39) * 300)
        frame = build_feature_frame(y, FeatureSpec(lags=(1, 2), rolling_windows=(3,), name="t"))
        # max lookback for this spec = slope window 6 + 1 = 7 -> first row at t=7
        assert frame.index.min() == y.index[7]
        assert frame["lag_1"].iloc[0] == y.iloc[6]
        # roll_mean_3 at position t = mean(y[t-3:t]) (strictly past)
        pos = 30
        assert frame["lag_1"].loc[y.index[pos]] == y.iloc[pos - 1]
        assert frame["lag_2"].loc[y.index[pos]] == y.iloc[pos - 2]
        assert frame["roll_mean_3"].loc[y.index[pos]] == pytest.approx(
            y.iloc[pos - 3 : pos].mean()
        )

    def test_features_at_origin_uses_only_history(self):
        """The origin feature row must be unaffected by any 'future' data."""
        y = _series(n=60)
        row = features_at_origin(y, LAG_SPEC)
        # lag_1 of the next step is the last observed value
        assert row["lag_1"] == pytest.approx(y.iloc[-1])
        assert row["slope_6"] == pytest.approx(
            np.polyfit(np.arange(6), y.iloc[-6:].to_numpy(), 1)[0]
        )

    def test_segment_features_are_lagged(self):
        y = _series(60)
        seg = pd.DataFrame(
            {"A": np.linspace(1, 2, 60), "B": np.linspace(5, 4, 60)},
            index=y.index,
        )
        frame = build_feature_frame(y, SEG_SPEC, segments=seg)
        pos = 40
        assert frame["seg_A_lag_1"].loc[y.index[pos]] == pytest.approx(seg["A"].iloc[pos - 1])
        assert frame["seg_B_lag_6"].loc[y.index[pos]] == pytest.approx(seg["B"].iloc[pos - 6])

    def test_no_future_metadata_fields_in_features(self):
        """Feature frames must never carry full-lifetime metadata statistics."""
        y = _series()
        seg = pd.DataFrame({"A": np.linspace(1, 2, len(y))}, index=y.index)
        frame = build_feature_frame(y, SEG_SPEC, segments=seg)
        for field in FUTURE_METADATA_FIELDS:
            assert field not in frame.columns
            assert field not in [c for c in frame.columns if c.startswith("seg_")]


class TestRecursiveForecasting:
    def test_forecast_does_not_read_future_values(self):
        """A recursive model's forecast must be a pure function of its input."""
        from cwfc.models import RidgeAR

        y = _series(n=80, seed=3)
        model = RidgeAR(alpha=1.0, spec=LAG_SPEC).fit(y)
        fc = model.forecast(24)

        # Anti-vacuous: repeated forecasts of the same fitted model are
        # deterministic, and a fresh refit on the SAME data reproduces them.
        np.testing.assert_array_equal(fc, model.forecast(24))
        model_b = RidgeAR(alpha=1.0, spec=LAG_SPEC).fit(y)
        np.testing.assert_allclose(fc, model_b.forecast(24))

        # Anti-vacuous: changing the input series must change the forecast
        # (guards against a degenerate model that ignores its input).
        y_changed = y.copy()
        y_changed.iloc[-5:] += 2.0
        fc_changed = RidgeAR(alpha=1.0, spec=LAG_SPEC).fit(y_changed).forecast(24)
        assert not np.allclose(fc, fc_changed)

    def test_forecast_length_and_finiteness(self):
        from cwfc.models import RidgeAR, MovingAverage

        y = _series(n=80, seed=1)
        for model in (RidgeAR(1.0, LAG_SPEC), MovingAverage(12)):
            fitted = model.fit(y)
            out = fitted.forecast(24)
            assert len(out) == 24
            assert np.isfinite(out).all()


class TestSegmentFeatureOrigin:
    """Regression tests for the stale-segment-feature bug found in review.

    Before the fix, ``features_at_origin`` dropped the origin row whenever
    its *lagged segment* features were NaN (future segment observations do
    not exist), silently returning an earlier, stale row — so segment-based
    recursive models emitted one identical prediction for every horizon
    step.  The fix forward-fills the segment frame to its last observed
    values (leakage-safe) and asserts the returned row is at the next
    timestamp.
    """

    def _series_with_segments(self, n: int = 80, seed: int = 4):
        rng = np.random.default_rng(seed)
        idx = pd.RangeIndex(n) * 300
        y = pd.Series(10 + rng.normal(0, 0.2, n).cumsum() * 0.05, index=idx)
        seg = pd.DataFrame(
            {"A": np.linspace(8, 12, n), "B": np.linspace(5, 4, n)}, index=idx
        )
        return y, seg

    def test_origin_row_timestamp_is_next_timestamp(self):
        y, seg = self._series_with_segments()
        for spec in (LAG_SPEC, SEG_SPEC):
            row = features_at_origin(y, spec, segments=seg)
            assert row.name == y.index[-1] + 300

    def test_origin_row_advances_under_recursion(self):
        """The returned feature timestamp must advance at every recursion step."""
        y, seg = self._series_with_segments()
        history = y.copy()
        seen: list = []
        for _ in range(4):
            row = features_at_origin(history, SEG_SPEC, segments=seg)
            seen.append(row.name)
            # append a dummy prediction, exactly like a recursive forecaster
            history = pd.concat(
                [history, pd.Series([float(y.iloc[-1])], index=[row.name])]
            )
        assert seen == [24000, 24300, 24600, 24900]
        assert seen == sorted(set(seen))  # strictly advancing, never stale

    def test_segment_features_use_last_observed_values(self):
        """Future segment means are unknown; features must hold the last
        observed values (leakage-safe last-known-value treatment)."""
        y, seg = self._series_with_segments()
        history = pd.concat([y, pd.Series([10.0], index=[24000])])
        row = features_at_origin(history, SEG_SPEC, segments=seg)
        # recursion step 2: origin 24300; seg_lag_1 resolves through the
        # forward-filled frame to the last OBSERVED segment value (t=23700)
        assert row.name == 24300
        assert row["seg_A_lag_1"] == pytest.approx(seg["A"].iloc[-1])
        # seg_lag_6 at origin 24300 reads t = 24300 - 6*300 = 22500 (observed)
        assert row["seg_B_lag_6"] == pytest.approx(seg["B"].loc[22500])

    def test_non_overlapping_segments_raise(self):
        """A segment frame that cannot cover the origin must fail loudly."""
        y, _ = self._series_with_segments()
        bad = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.Index(["x", "y"]))
        with pytest.raises(ValueError, match="no observations at or before"):
            features_at_origin(y, SEG_SPEC, segments=bad)

    def test_segment_model_forecast_is_not_degenerate(self):
        """The bug made segment-ridge repeat one prediction 24 times."""
        from cwfc.models import RidgeAR

        y, seg = self._series_with_segments()
        fc = RidgeAR(1.0, SEG_SPEC).fit(y, segments=seg).forecast(24)
        assert len(np.unique(np.round(fc, 9))) > 3, "forecast is degenerate (stale features)"


class TestValidationDesign:
    def test_folds_stay_inside_training_window(self):
        from cwfc.evaluate import rolling_origin_folds

        folds = rolling_origin_folds(n_total=110, horizon=24, min_train=50, step=6)
        assert len(folds) == 7
        for fold in folds:
            assert fold.val_end <= 110  # never touches positions 110..133 (holdout)
            assert fold.val_start >= 50
            assert fold.train_end == fold.val_start

    def test_folds_are_strictly_ordered(self):
        from cwfc.evaluate import rolling_origin_folds

        folds = rolling_origin_folds(n_total=110, horizon=24, min_train=50, step=6)
        starts = [f.val_start for f in folds]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)

    def test_holdout_split_is_disjoint(self):
        from cwfc.data import chronological_split

        idx = pd.RangeIndex(134) * 300
        targets = pd.DataFrame(
            {"active_vm_count": 10, "aggregate_cpu_utilization": 9.0}, index=idx
        )
        train, holdout = chronological_split(targets)
        assert set(train.index) & set(holdout.index) == set()
        assert train.index.max() < holdout.index.min()
