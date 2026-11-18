"""Forecasting models: simple baselines, exponential smoothing, and
regularized/gradient-boosted autoregressions.

Every model follows one protocol::

    model = SomeModel(name=..., **params)
    model.fit(series, segments=None)      # segments: optional exogenous frame
    point_forecast = model.forecast(h)    # np.ndarray of length h

``fit`` receives **only the past** (the training slice of the target series,
plus optionally per-segment mean series derived from the readings).  Nothing
in this module ever reads values beyond the end of the series it was fitted
on — the recursive supervised models roll their *own predictions* forward,
which is verified by the leakage tests in ``tests/test_features.py``.

Multi-step strategy: smoothing models extend their level/trend natively;
supervised models are iterated one step at a time (recursive strategy).  A
direct per-horizon ridge variant is included for strategy comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

from .baselines import persistence_forecast, seasonal_naive_forecast
from .features import FeatureSpec, build_feature_frame, features_at_origin

SEED = 42  # deterministic random seeds everywhere


class BaseModel:
    """Common interface + bookkeeping for all forecasting models."""

    name: str = "model"
    complexity: int = 1  # lower = simpler; used as a pre-declared tie-break rule

    def fit(self, series: pd.Series, segments: pd.DataFrame | None = None) -> "BaseModel":
        self._y = series.astype(float).copy()
        self._segments = segments
        return self

    def _check_horizon(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")

    @property
    def min_history_required(self) -> int:
        """Minimum training length this model needs to fit at all."""
        return 1

    def forecast(self, horizon: int) -> np.ndarray:
        """Produce a point forecast for ``horizon`` future steps (abstract)."""
        self._check_horizon(horizon)
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.name!r}>"


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
class Persistence(BaseModel):
    """Last observed value, held constant over the horizon."""

    def __init__(self, name: str = "persistence"):
        self.name = name
        self.complexity = 1

    def forecast(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        template = pd.DataFrame(index=pd.RangeIndex(horizon))
        out = persistence_forecast(
            self._y.reset_index(drop=True).to_frame("y"), template, "y"
        )
        return out.to_numpy()


class HistoricalMean(BaseModel):
    """Mean of all observed training values, held constant."""

    def __init__(self, name: str = "historical_mean"):
        self.name = name
        self.complexity = 1

    def forecast(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        return np.full(horizon, float(self._y.mean()))


class MovingAverage(BaseModel):
    """Mean of the last ``window`` observations, held constant."""

    def __init__(self, window: int, name: str | None = None):
        self.window = window
        self.name = name or f"ma_{window}"
        self.complexity = 2

    @property
    def min_history_required(self) -> int:
        return self.window

    def forecast(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        tail = self._y.iloc[-self.window :]
        return np.full(horizon, float(tail.mean()))


class Drift(BaseModel):
    """Random-walk-with-drift: last value + average slope of the history."""

    def __init__(self, name: str = "drift"):
        self.name = name
        self.complexity = 2

    def forecast(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        y = self._y.to_numpy()
        slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0.0
        return y[-1] + slope * np.arange(1, horizon + 1)


class SeasonalNaive(BaseModel):
    """Value from ``period`` steps earlier, cycled (kept for comparison).

    With ≈11 h of data no seasonality is identifiable, so this baseline is
    expected to lose; it is included to demonstrate that empirically.
    """

    def __init__(self, period: int, name: str | None = None):
        self.period = period
        self.name = name or f"seasonal_naive_{period}"
        self.complexity = 2

    @property
    def min_history_required(self) -> int:
        return self.period

    def forecast(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        template = pd.DataFrame(index=pd.RangeIndex(horizon))
        out = seasonal_naive_forecast(
            self._y.reset_index(drop=True).to_frame("y"), template, "y", self.period
        )
        return out.to_numpy()


# --------------------------------------------------------------------------
# Exponential smoothing (statsmodels)
# --------------------------------------------------------------------------
class SES(BaseModel):
    """Simple exponential smoothing: flat forecast at the smoothed level."""

    def __init__(self, alpha: float, name: str | None = None):
        self.alpha = alpha
        self.name = name or f"ses_{alpha:g}"
        self.complexity = 3

    def forecast(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        fit = SimpleExpSmoothing(self._y.to_numpy(), initialization_method="estimated").fit(
            smoothing_level=self.alpha, optimized=False
        )
        return np.asarray(fit.forecast(horizon), dtype=float)


class HoltDamped(BaseModel):
    """Holt's method with damped trend (level + decaying trend)."""

    def __init__(self, name: str = "holt_damped"):
        self.name = name
        self.complexity = 3

    def forecast(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        fit = ExponentialSmoothing(
            self._y.to_numpy(),
            trend="add",
            damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.asarray(fit.forecast(horizon), dtype=float)


# --------------------------------------------------------------------------
# Supervised autoregressive models (recursive multi-step)
# --------------------------------------------------------------------------
class _RecursiveSupervised(BaseModel):
    """Shared machinery: train on lag features, roll predictions forward."""

    def __init__(self, spec: FeatureSpec, name: str):
        self.spec = spec
        self.name = name

    @property
    def min_history_required(self) -> int:
        return self.spec.max_lookback + 5  # + supervised rows needed to fit

    def _fit_regressor(self, X: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def _predict_one(self, row: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError

    def fit(self, series: pd.Series, segments: pd.DataFrame | None = None) -> "_RecursiveSupervised":
        super().fit(series, segments)
        frame = build_feature_frame(self._y, self.spec, segments=self._segments)
        if len(frame) < 5:
            raise ValueError(f"not enough supervised rows to fit {self.name}")
        X = frame.drop(columns="target").to_numpy(dtype=float)
        y = frame["target"].to_numpy(dtype=float)
        self._feature_names = list(frame.columns[:-1])
        self._fit_regressor(X, y)
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        history = self._y.copy()
        out: list[float] = []
        for _ in range(horizon):
            row = features_at_origin(history, self.spec, segments=self._segments)
            x = row.reindex(self._feature_names).to_numpy(dtype=float)
            if np.isnan(x).any():  # defensive: fall back to last value
                out.append(float(history.iloc[-1]))
            else:
                out.append(float(self._predict_one(x)))
            history = pd.concat([history, pd.Series([out[-1]], index=[history.index[-1] + (history.index[-1] - history.index[-2])])])
        return np.asarray(out, dtype=float)


class RidgeAR(_RecursiveSupervised):
    """Ridge regression on lag/rolling features, iterated recursively."""

    def __init__(self, alpha: float, spec: FeatureSpec, name: str | None = None):
        super().__init__(spec, name or f"ridge_{spec.name}_a{alpha:g}")
        self.alpha = alpha
        self.complexity = 4
        self._model = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha, random_state=SEED))]
        )

    def _fit_regressor(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def _predict_one(self, row: np.ndarray) -> float:
        return float(self._model.predict(row.reshape(1, -1))[0])


class HGBAR(_RecursiveSupervised):
    """Gradient boosting on lag/rolling features, iterated recursively.

    Included to test whether a flexible learner beats simpler models on only
    ~100 training points; heavily regularized, fixed seed.
    """

    def __init__(self, spec: FeatureSpec, name: str = "hgb_lag"):
        super().__init__(spec, name)
        self.complexity = 6
        self._model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=SEED,
        )

    def _fit_regressor(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def _predict_one(self, row: np.ndarray) -> float:
        return float(self._model.predict(row.reshape(1, -1))[0])


class DirectRidge(BaseModel):
    """Direct multi-step: one ridge model per forecast horizon h = 1..H."""

    def __init__(self, alpha: float, spec: FeatureSpec, name: str = "ridge_direct"):
        self.alpha = alpha
        self.spec = spec
        self.name = name
        self.complexity = 5

    @property
    def min_history_required(self) -> int:
        return self.spec.max_lookback + 26  # + rows for the h=24 direct model

    def fit(self, series: pd.Series, segments: pd.DataFrame | None = None) -> "DirectRidge":
        super().fit(series, segments)
        frame = build_feature_frame(self._y, self.spec, segments=self._segments)
        X_all = frame.drop(columns="target")
        self._feature_names = list(X_all.columns)
        values = self._y.to_numpy(dtype=float)
        pos = {ts: i for i, ts in enumerate(self._y.index)}
        self._models: dict[int, Pipeline] = {}
        # Rows with features at position p predict target at position p+h.
        for h in range(1, 25):
            rows, targets = [], []
            for p, ts in enumerate(frame.index):
                i = pos[ts]
                if i + h < len(values):
                    rows.append(X_all.loc[ts].to_numpy(dtype=float))
                    targets.append(values[i + h])
            if len(rows) < 5:
                raise ValueError(f"not enough rows for direct horizon {h}")
            model = Pipeline(
                [("scale", StandardScaler()), ("ridge", Ridge(alpha=self.alpha, random_state=SEED))]
            )
            model.fit(np.asarray(rows), np.asarray(targets))
            self._models[h] = model
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        self._check_horizon(horizon)
        row = features_at_origin(self._y, self.spec, segments=self._segments)
        x = row.reindex(self._feature_names).to_numpy(dtype=float).reshape(1, -1)
        return np.asarray(
            [float(self._models[h].predict(x)[0]) for h in range(1, horizon + 1)],
            dtype=float,
        )


# --------------------------------------------------------------------------
# Candidate pool + frozen configuration
# --------------------------------------------------------------------------
LAG_SPEC = FeatureSpec(name="lag")
SEG_SPEC = FeatureSpec(name="seg", with_segments=True)


def build_candidate_pool(segments_available: bool = True) -> list[BaseModel]:
    """All candidate models compared in the rolling-origin validation.

    Hyperparameter grids (moving-average windows, SES levels, ridge alphas)
    are treated as separate candidates and compared on the same validation
    folds — no fold's own validation data is used to pick its
    hyperparameters, and the final holdout is never involved.

    ``segments_available=False`` excludes the segment-feature ridge variants
    (used when no ``vm_category`` metadata exists, e.g. minimal smoke data).
    """
    pool: list[BaseModel] = [
        Persistence(),
        HistoricalMean(),
        MovingAverage(6),
        MovingAverage(12),
        MovingAverage(24),
        Drift(),
        SeasonalNaive(24),
        SES(0.3),
        SES(0.6),
        SES(0.9),
        HoltDamped(),
    ]
    for alpha in (0.1, 1.0, 10.0, 100.0):
        pool.append(RidgeAR(alpha, LAG_SPEC))
    if segments_available:
        for alpha in (1.0, 10.0):
            pool.append(RidgeAR(alpha, SEG_SPEC))
    # NOTE: DirectRidge (direct multi-step strategy) is implemented and unit
    # tested but excluded from the pool: with the earliest folds training on
    # 50 points, the h=24 direct model would be fit on as few as 2 supervised
    # rows.  The direct strategy is data-starved at this window size — a
    # documented negative design decision, not an omission.
    pool.append(HGBAR(LAG_SPEC))
    return pool


@dataclass(frozen=True)
class ModelSpec:
    """Frozen description of a selected model (chosen on validation data only)."""

    kind: str  # persistence | moving_average | ses | ridge | ...
    params: dict

    def build(self) -> BaseModel:
        builders = {
            "persistence": lambda p: Persistence(),
            "historical_mean": lambda p: HistoricalMean(),
            "moving_average": lambda p: MovingAverage(int(p["window"])),
            "drift": lambda p: Drift(),
            "seasonal_naive": lambda p: SeasonalNaive(int(p["period"])),
            "ses": lambda p: SES(float(p["alpha"])),
            "holt_damped": lambda p: HoltDamped(),
            "ridge_ar": lambda p: RidgeAR(
                float(p["alpha"]),
                SEG_SPEC if p.get("spec") == "seg" else LAG_SPEC,
            ),
            "direct_ridge": lambda p: DirectRidge(float(p["alpha"]), LAG_SPEC),
            "hgb_ar": lambda p: HGBAR(LAG_SPEC),
        }
        return builders[self.kind](self.params)


# Configuration frozen BEFORE the final holdout was evaluated.  Selected by
# the pre-declared rule (lowest pooled rolling-origin CV MAE inside the
# 110-timestamp training window; ties -> lower complexity), see
# notebooks/02_forecasting.ipynb for the full comparison table.
#   active_vm_count:        all candidates tie at CV MAE = 0 (constant
#                           series) -> simplest model wins (persistence).
#   aggregate_cpu_utilization: ridge (alpha=100) on lag/rolling features,
#                           recursive multi-step; CV MAE 0.183 vs persistence
#                           0.215 (-15.1%), stable at 4- and 7-fold designs.
FROZEN_CONFIG: dict[str, ModelSpec] = {
    "active_vm_count": ModelSpec("persistence", {}),
    "aggregate_cpu_utilization": ModelSpec("ridge_ar", {"alpha": 100.0, "spec": "lag"}),
}
