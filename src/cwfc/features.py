"""Leakage-safe feature construction for short-horizon workload forecasting.

Design rules (enforced and unit-tested):

* Every feature for a target timestamp ``t`` is computed **only from observations
  strictly before ``t``** (lags, trailing rolling statistics, trailing slopes,
  lagged segment aggregates).  This is what "could have been computed at the
  forecast origin" means in ASSIGNMENT.md §5.
* No metadata that encodes future information is ever used.  The full-lifetime
  statistics (``avg_cpu``, ``max_cpu``, ``p95_max_cpu``) and deletion timestamps
  in the metadata file summarize the *entire* VM lifetime, i.e. information from
  after the observation window; they are excluded by design (see
  :data:`cwfc.data.FUTURE_METADATA_FIELDS`).
* With ≈11 hours of history, daily/weekly seasonality is unidentifiable, so no
  calendar or seasonal features are constructed (documented decision, not an
  omission).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (3, 6, 12)
SLOPE_WINDOW = 6
SEGMENT_LAGS: tuple[int, ...] = (1, 6)
DEFAULT_CATEGORIES: tuple[str, ...] = ("Delay-insensitive", "Interactive", "Unknown")


def _slope(values: np.ndarray) -> float:
    """Least-squares slope of a 1-D array against 0..n-1."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), values.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (values - ym)).sum() / denom)


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative description of a feature set.

    ``lags`` / ``rolling_windows``
        Computed on the target series itself.
    ``with_segments``
        Additionally use lagged segment-mean aggregates (e.g. per
        ``vm_category``) as exogenous features.  These are derived from the
        CPU readings alone, so they are available at forecast time.
    """

    lags: tuple[int, ...] = DEFAULT_LAGS
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS
    with_segments: bool = False
    segment_lags: tuple[int, ...] = SEGMENT_LAGS
    name: str = "lag"

    @property
    def max_lookback(self) -> int:
        lookbacks: list[int] = [max(self.lags), max(self.rolling_windows) + 1, SLOPE_WINDOW + 1]
        if self.with_segments:
            lookbacks.append(max(self.segment_lags))
        return max(lookbacks)


def segment_means(cpu: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Per-timestamp mean ``avg_cpu`` for each workload category.

    Built from the readings (past-only information at each timestamp) and the
    *static* ``vm_category`` label.  Lifetime CPU statistics and deletion
    timestamps are deliberately not used.
    """
    cats = (
        metadata[["vm_id", "vm_category"]]
        if not metadata.empty and "vm_category" in metadata
        else None
    )
    if cats is None:
        return pd.DataFrame(index=sorted(cpu["timestamp"].unique()))
    joined = cpu[cpu["_valid"]].merge(cats, on="vm_id", how="left")
    return joined.groupby(["timestamp", "vm_category"])["avg_cpu"].mean().unstack(
        fill_value=np.nan
    ).sort_index()


def build_feature_frame(
    y: pd.Series,
    spec: FeatureSpec,
    segments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the supervised feature frame for a target series.

    The returned frame is indexed like ``y`` (after dropping warm-up rows whose
    features are undefined) and contains feature columns plus a ``target``
    column.  For row at position ``p`` every feature is a function of
    ``y[:p]`` only — see the property tests in ``tests/test_features.py``.

    Parameters
    ----------
    y:
        Target series with a monotonic index (positional order = time order).
    spec:
        Which features to build.
    segments:
        Optional per-timestamp segment means (columns = segment names) aligned
        to ``y``'s index; required when ``spec.with_segments``.
    """
    if not y.index.is_monotonic_increasing:
        raise ValueError("target series index must be monotonic increasing")

    past = y.shift(1)  # y[t-1] at position t: the most recent *past* value
    feats: dict[str, pd.Series] = {}

    for lag in spec.lags:
        feats[f"lag_{lag}"] = y.shift(lag)

    for w in spec.rolling_windows:
        feats[f"roll_mean_{w}"] = past.rolling(w).mean()
        feats[f"roll_std_{w}"] = past.rolling(w).std(ddof=1)

    # Recent change: y[t-1] - y[t-1-k] for the shortest and mid lags.
    for k in (1, 6):
        feats[f"diff_{k}"] = past - y.shift(1 + k)

    # Local trend: slope of the last SLOPE_WINDOW observations ending at t-1.
    feats["slope_6"] = past.rolling(SLOPE_WINDOW).apply(_slope, raw=True)

    if spec.with_segments:
        if segments is None or segments.empty:
            raise ValueError("with_segments=True requires a segment frame")
        seg = segments.reindex(y.index)
        for col in seg.columns:
            for lag in spec.segment_lags:
                feats[f"seg_{col}_lag_{lag}"] = seg[col].shift(lag)

    frame = pd.DataFrame(feats, index=y.index)
    frame["target"] = y
    return frame.dropna(subset=[c for c in frame.columns if c != "target"])


def features_at_origin(
    y: pd.Series,
    spec: FeatureSpec,
    segments: pd.DataFrame | None = None,
) -> pd.Series:
    """Feature row for the *next* timestamp after the end of ``y``.

    This is the row a recursive forecaster feeds to its model when it has
    observed ``y[0..T]`` and must predict ``y[T+1]``: every feature is derived
    from observations up to and including the last element of ``y``.

    Segment features (``spec.with_segments``): the per-segment means at
    *future* timestamps are unknown at forecast time, so they cannot be read
    from the segment frame.  Instead, the segment frame is extended with its
    **last observed values** (forward-fill) — the standard leakage-safe
    "last-known-value" treatment for exogenous covariates in recursive
    forecasting.  Without this, reindexing would leave NaN segment features
    at the origin row, ``build_feature_frame``'s warm-up ``dropna`` would
    silently drop it, and a *stale, earlier* feature row would be returned
    (a real bug found in review: segment-based recursive models then emitted
    identical predictions for every horizon step).  Two guards prevent any
    regression of that bug: the forward-fill above, and an explicit check
    below that the returned row's timestamp equals the next timestamp.
    """
    if len(y) < 2:
        raise ValueError("features_at_origin needs at least two observations to infer cadence")
    step = y.index[-1] - y.index[-2]
    next_index = y.index[-1] + step
    extended = pd.Series([np.nan], index=[next_index], dtype=float)
    history = pd.concat([y, extended])

    if spec.with_segments:
        if segments is None or segments.empty:
            raise ValueError("with_segments=True requires a segment frame")
        seg = segments.reindex(history.index).ffill()
        if seg.iloc[-1].isna().any():
            raise ValueError(
                "segment frame has no observations at or before the forecast "
                "origin; cannot construct leakage-safe segment features"
            )
    else:
        seg = segments
    frame = build_feature_frame(history, spec, segments=seg)
    row = frame.iloc[-1].drop("target")
    if row.name != next_index:
        raise RuntimeError(
            f"features_at_origin returned a stale row (index {row.name!r} != "
            f"expected {next_index!r}); refusing to proceed"
        )
    return row
