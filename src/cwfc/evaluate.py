"""Time-aware validation, evaluation and uncertainty estimation.

Validation design (fixed before looking at the holdout):

* **Rolling-origin (expanding-window) evaluation** inside the 110-timestamp
  public training window — never on the final 24 holdout timestamps.
* Folds: with ``n = 110`` training points, ``horizon = 24`` (matches the
  2-hour target scenario), first origin at ``min_train = 50``, step ``6`` →
  origins {50, 56, 62, 68, 74, 80, 86} = 7 folds.  Consecutive validation
  windows overlap; this is expected and documented (only 110 points exist).
* Every candidate model is fitted **only on that fold's past** and produces a
  full 24-step forecast, mimicking the production forecasting task exactly.
* Peak-period error uses a leakage-safe threshold: peaks are validation
  points whose *actual* value is at or above ``μ + σ`` of that fold's
  training slice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import mae, normalized_mae, rmse
from .models import BaseModel


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold (positional indices into the series)."""

    train_end: int  # exclusive
    val_start: int
    val_end: int  # exclusive

    @property
    def train_idx(self) -> np.ndarray:
        return np.arange(self.val_start)

    @property
    def val_idx(self) -> np.ndarray:
        return np.arange(self.val_start, self.val_end)


def rolling_origin_folds(
    n_total: int,
    horizon: int = 24,
    min_train: int = 50,
    step: int = 6,
) -> list[Fold]:
    """Expanding-window folds for a series of length ``n_total``.

    The first fold trains on ``[0, min_train)`` and validates on the next
    ``horizon`` points; each subsequent fold moves the origin forward by
    ``step``.  Raises if ``n_total < min_train + horizon``.
    """
    if horizon <= 0 or min_train <= 0 or step <= 0:
        raise ValueError("horizon, min_train and step must be positive")
    if n_total < min_train + horizon:
        raise ValueError(
            f"need n_total >= min_train + horizon = {min_train + horizon}, got {n_total}"
        )
    folds: list[Fold] = []
    origin = min_train
    while origin + horizon <= n_total:
        folds.append(Fold(train_end=origin, val_start=origin, val_end=origin + horizon))
        origin += step
    return folds


@dataclass
class CVResult:
    """Per-model rolling-origin results."""

    name: str
    complexity: int
    preds: dict[int, np.ndarray]  # fold id -> predictions
    actuals: dict[int, np.ndarray]  # fold id -> actuals
    is_peak: dict[int, np.ndarray]  # fold id -> boolean peak mask

    # -- convenience -----------------------------------------------------
    def errors(self) -> dict[int, np.ndarray]:
        return {k: self.preds[k] - self.actuals[k] for k in self.preds}

    def pooled(self, what: str) -> np.ndarray:
        attr = self.preds if what == "pred" else self.actuals
        return np.concatenate([attr[k] for k in sorted(attr)])

    def summary(self, baseline: "CVResult | None" = None) -> dict[str, float]:
        err = self.errors()
        all_pred = np.concatenate([self.preds[k] for k in sorted(self.preds)])
        all_true = np.concatenate([self.actuals[k] for k in sorted(self.actuals)])
        peak_true = np.concatenate(
            [self.actuals[k][self.is_peak[k]] for k in sorted(self.actuals)]
        )
        peak_pred = np.concatenate(
            [self.preds[k][self.is_peak[k]] for k in sorted(self.preds)]
        )
        out = {
            "model": self.name,
            "complexity": self.complexity,
            "MAE": mae(all_true, all_pred),
            "RMSE": rmse(all_true, all_pred),
            "peak_MAE": mae(peak_true, peak_pred) if len(peak_true) else float("nan"),
            "n_peak": int(len(peak_true)),
            "n_folds": len(self.preds),
        }
        if baseline is not None:
            out["MAE_vs_baseline"] = normalized_mae(all_true, all_pred, baseline.pooled("pred"))
            out["skill_vs_baseline"] = 1.0 - out["MAE_vs_baseline"]
        return out


def run_cv(
    series: pd.Series,
    candidates: list[BaseModel],
    folds: list[Fold],
    segments: pd.DataFrame | None = None,
    skip_too_hungry: bool = True,
) -> dict[str, CVResult]:
    """Fit every candidate on each fold's past; forecast the fold's horizon.

    ``series`` is the full training-window series (the holdout must already
    be excluded by the caller).  ``segments`` is sliced positionally to each
    fold so supervised models only ever see past segment aggregates.

    Candidates that cannot fit on the *smallest* fold (per
    ``model.min_history_required``) are skipped up front — otherwise small
    datasets would crash instead of comparing the models that are feasible.
    Set ``skip_too_hungry=False`` to disable the filter.
    """
    values = series.reset_index(drop=True)
    seg_pos = (
        segments.reset_index(drop=True) if segments is not None else None
    )
    smallest_train = min(fold.val_start for fold in folds)
    skipped: list[str] = []
    if skip_too_hungry:
        usable, skipped = [], []
        for model in candidates:
            (usable if model.min_history_required <= smallest_train else skipped).append(model)
        candidates = usable
    if skipped:
        print(
            f"skipped (need more history than the smallest fold of "
            f"{smallest_train} points): {', '.join(m.name for m in skipped)}"
        )
    if not candidates:
        raise ValueError("no candidate models can fit on the smallest fold")

    results: dict[str, CVResult] = {}

    for model in candidates:
        preds: dict[int, np.ndarray] = {}
        actuals: dict[int, np.ndarray] = {}
        is_peak: dict[int, np.ndarray] = {}
        for fold_id, fold in enumerate(folds):
            train = values.iloc[: fold.val_start]
            seg_slice = seg_pos.iloc[: fold.val_start] if seg_pos is not None else None
            model.fit(train, segments=seg_slice)
            preds[fold_id] = np.asarray(model.forecast(fold.val_end - fold.val_start))
            actuals[fold_id] = values.iloc[fold.val_start : fold.val_end].to_numpy()
            mu, sd = float(train.mean()), float(train.std(ddof=1))
            is_peak[fold_id] = actuals[fold_id] >= (mu + sd)
        results[model.name] = CVResult(
            name=model.name, complexity=model.complexity,
            preds=preds, actuals=actuals, is_peak=is_peak,
        )
    return results


def cv_summary_table(
    results: dict[str, CVResult], baseline_name: str = "persistence"
) -> pd.DataFrame:
    """Rank candidates by pooled MAE; ties broken by pre-declared complexity."""
    baseline = results.get(baseline_name)
    rows = [res.summary(baseline) for res in results.values()]
    table = pd.DataFrame(rows)
    table = table.sort_values(["MAE", "complexity"], kind="stable").reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


def horizon_errors(result: CVResult) -> pd.DataFrame:
    """Absolute error by forecast horizon (1..H), pooled over folds."""
    err = result.errors()
    h_max = max(len(v) for v in err.values())
    data = {
        f"fold_{k}": np.pad(np.abs(v), (0, h_max - len(v)), constant_values=np.nan)
        for k, v in err.items()
    }
    frame = pd.DataFrame(data)
    frame.insert(0, "horizon", np.arange(1, h_max + 1))
    return frame


def empirical_interval(
    result: CVResult, alpha: float = 0.10
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical per-horizon prediction interval from CV residuals.

    Residuals are defined as ``r = prediction - actual`` (see
    :meth:`CVResult.errors`).  A residual quantile pair therefore maps to an
    interval for the *actual* value via ``actual = forecast - residual``:

    * lower bound = forecast − upper residual quantile  ``Q_{1-α/2}(r)``
    * upper bound = forecast − lower residual quantile  ``Q_{α/2}(r)``

    The returned arrays ``(lower, upper)`` are per-horizon *offsets* to be
    added to a point forecast ``f``: the interval is
    ``[f + lower[h], f + upper[h]]``.  Equivalently
    ``lower[h] = -Q_{1-α/2}(r_h)`` and ``upper[h] = -Q_{α/2}(r_h)``, where
    the quantiles are taken across folds at horizon step ``h``.

    With few folds these quantiles are coarse — they are used as a relative
    uncertainty signal for capacity planning, not as a calibrated
    probabilistic guarantee (documented limitation).
    """
    err = result.errors()
    h_max = max(len(v) for v in err.values())
    lower, upper = np.zeros(h_max), np.zeros(h_max)
    for h in range(h_max):
        residuals = np.concatenate([v[h : h + 1] for v in err.values() if len(v) > h])
        lower[h] = -np.quantile(residuals, 1 - alpha / 2)
        upper[h] = -np.quantile(residuals, alpha / 2)
    return lower, upper


def select_model(
    results: dict[str, CVResult], baseline_name: str = "persistence"
) -> tuple[str, pd.DataFrame]:
    """Pre-declared selection rule: lowest pooled CV MAE, ties -> simpler.

    Returns the selected model name and the ranking table.
    """
    table = cv_summary_table(results, baseline_name)
    best = table.iloc[0]
    return str(best["model"]), table
