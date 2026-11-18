"""Generate the required ``predictions.csv`` from the challenge data.

Usage::

    python -m cwfc.predict --data challenge_data --output predictions.csv

Behaviour
---------
* Loads and validates the dataset through :mod:`cwfc.data`.
* Applies the official 110/24 chronological split.  If the supplied dataset
  has fewer than 134 timestamps (e.g. a reviewer smoke dataset), the holdout
  size is scaled to preserve the same 110:24 ratio and a warning is printed —
  nothing is hard-coded to the official file.
* Uses the configuration frozen in :data:`cwfc.models.FROZEN_CONFIG`
  (selected by rolling-origin CV on the training window only).  Pass
  ``--select`` to re-run the selection procedure instead.
* The holdout *timestamps* are taken from the data; the holdout *values* are
  never read, so this command cannot leak them into the forecast.
* Validates the output against the SUBMISSION.md contract before writing.

The generated forecasts are clipped to the valid ranges
(``active_vm_count >= 0`` and integer-valued when the observed counts are
integers; ``aggregate_cpu_utilization`` in ``[0, 100]``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    TARGET_AGGREGATE_CPU,
    TARGET_ACTIVE_VM_COUNT,
    DataValidationError,
    load_dataset,
)
from .evaluate import rolling_origin_folds, run_cv, select_model
from .features import segment_means
from .models import FROZEN_CONFIG, BaseModel, build_candidate_pool

OFFICIAL_TRAIN_SIZE = 110
OFFICIAL_HORIZON = 24
PREDICTION_COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]


def adaptive_split(n_total: int) -> tuple[int, int]:
    """Return (train_size, horizon): the final 24 timestamps are the holdout.

    Following SUBMISSION.md, "the holdout timestamps are the final 24
    timestamps in the supplied dataset".  For the official release
    (n = 134) this yields exactly the documented 110/24 split.  Larger
    datasets keep a 24-timestamp holdout and train on everything before it
    (train_size = n - 24).  Smaller datasets (reviewer smoke data) get a
    proportionally scaled holdout so the command still works.
    """
    if n_total >= OFFICIAL_TRAIN_SIZE + OFFICIAL_HORIZON:
        return n_total - OFFICIAL_HORIZON, OFFICIAL_HORIZON
    horizon = max(
        1, round(n_total * OFFICIAL_HORIZON / (OFFICIAL_TRAIN_SIZE + OFFICIAL_HORIZON))
    )
    train = n_total - horizon
    if train < 2:
        raise DataValidationError(
            f"need at least a few timestamps to forecast, got {n_total}"
        )
    print(
        f"WARNING: dataset has {n_total} timestamps "
        f"(< {OFFICIAL_TRAIN_SIZE + OFFICIAL_HORIZON}); using proportional "
        f"split train={train} / holdout={horizon} (official protocol is 110/24)."
    )
    return train, horizon


def _fit_forecast(
    model: BaseModel,
    train_series: pd.Series,
    segments: pd.DataFrame | None,
    horizon: int,
) -> np.ndarray:
    """Fit on the training slice only.

    Falls back to persistence if a supervised model cannot be fitted (e.g. a
    very small smoke dataset), with a warning — never silently.
    """
    try:
        model.fit(train_series, segments=segments)
        return np.asarray(model.forecast(horizon), dtype=float)
    except ValueError as exc:
        print(
            f"WARNING: {model.name} could not be fitted on {len(train_series)} "
            f"points ({exc}); falling back to persistence for this target."
        )
        return np.full(horizon, float(train_series.iloc[-1]))


def _resolve_models(
    train: pd.DataFrame,
    seg_train: pd.DataFrame,
    reselect: bool,
    horizon: int,
) -> dict[str, BaseModel]:
    """Return one fitted-ready model per target (frozen config by default)."""
    if not reselect:
        return {name: spec.build() for name, spec in FROZEN_CONFIG.items()}

    print("Re-running rolling-origin model selection on the training window ...")
    n_train = len(train)
    # Reproduce the fold design used for the frozen selection (min_train=50,
    # step=6) whenever the training window is large enough; scale down for
    # smaller (smoke) datasets.
    if n_train >= 50 + horizon:
        min_train, step = 50, 6
    else:
        min_train, step = max(10, n_train // 2), max(1, horizon // 4)
    folds = rolling_origin_folds(n_train, horizon=horizon, min_train=min_train, step=step)
    models: dict[str, BaseModel] = {}
    for target_name in (TARGET_ACTIVE_VM_COUNT, TARGET_AGGREGATE_CPU):
        pool = build_candidate_pool(segments_available=not seg_train.empty)
        results = run_cv(train[target_name], pool, folds, segments=seg_train)
        selected, _ = select_model(results)
        print(f"  selected for {target_name}: {selected}")
        models[target_name] = next(m for m in pool if m.name == selected)
    return models


def generate_predictions(
    data_dir: str | Path,
    output_path: str | Path,
    reselect: bool = False,
    quiet: bool = False,
) -> pd.DataFrame:
    """Run the forecasting pipeline and write ``predictions.csv``.

    Returns the prediction frame (also written to ``output_path``).
    """
    ds = load_dataset(data_dir)
    targets = ds.targets
    if not quiet:
        print("Preprocessing notes:")
        for note in ds.notes:
            print(f"  - {note}")
    train_size, horizon = adaptive_split(len(targets))

    train = targets.iloc[:train_size].copy()
    holdout_index = targets.index[train_size:]

    # Internal gaps (no valid readings at some timestamp) would produce NaN
    # utilization values; interpolate them linearly on the time axis and
    # report, so downstream models see a complete series.
    agg = train[TARGET_AGGREGATE_CPU]
    if agg.isna().any():
        n_gap = int(agg.isna().sum())
        print(
            f"WARNING: {n_gap} training timestamp(s) had no valid readings; "
            "linearly interpolated on the timestamp axis."
        )
        train[TARGET_AGGREGATE_CPU] = agg.interpolate(
            method="index", limit_direction="both"
        )

    seg_all = segment_means(ds.cpu, ds.metadata)
    seg_train = (
        seg_all.loc[seg_all.index.isin(train.index)]
        if not seg_all.empty
        else seg_all
    )

    models = _resolve_models(train, seg_train, reselect, horizon)

    counts = _fit_forecast(
        models[TARGET_ACTIVE_VM_COUNT], train[TARGET_ACTIVE_VM_COUNT], seg_train, horizon
    )
    cpu = _fit_forecast(
        models[TARGET_AGGREGATE_CPU], train[TARGET_AGGREGATE_CPU], seg_train, horizon
    )

    # Assemble and sanitize the prediction frame. Predictions are rounded to
    # 6 decimals so the file is byte-identical across platforms despite
    # floating-point summation-order noise (~1e-13) between library builds.
    if (train[TARGET_ACTIVE_VM_COUNT] % 1 == 0).all():
        counts = np.rint(counts)
    out = pd.DataFrame(
        {
            "timestamp": holdout_index,
            "active_vm_count": np.maximum(counts, 0.0),
            "aggregate_cpu_utilization": np.clip(np.round(cpu, 6), 0.0, 100.0),
        }
    )

    validate_predictions(out, n_expected=horizon)
    out.to_csv(output_path, index=False)
    if not quiet:
        print(f"\nwrote {output_path} ({len(out)} rows)")
        print(out.head(2).to_string(index=False))
        print("  ...")
        print(out.tail(2).to_string(index=False))
    return out


def validate_predictions(frame: pd.DataFrame, n_expected: int = 24) -> None:
    """Assert the SUBMISSION.md contract; raise on any violation."""
    errors: list[str] = []
    if list(frame.columns) != PREDICTION_COLUMNS:
        errors.append(f"columns must be {PREDICTION_COLUMNS}, got {list(frame.columns)}")
    if len(frame) != n_expected:
        errors.append(f"expected exactly {n_expected} rows, got {len(frame)}")
    ts = frame["timestamp"]
    if ts.duplicated().any():
        errors.append("duplicate timestamps present")
    if not ts.is_monotonic_increasing:
        errors.append("timestamps must be in chronological (ascending) order")
    values = frame[["active_vm_count", "aggregate_cpu_utilization"]]
    if values.isna().any().any():
        errors.append("missing (NaN) predictions present")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        errors.append("non-finite predictions present")
    if (frame["active_vm_count"] < 0).any():
        errors.append("negative active_vm_count present")
    if ((frame["aggregate_cpu_utilization"] < 0) | (frame["aggregate_cpu_utilization"] > 100)).any():
        errors.append("aggregate_cpu_utilization outside [0, 100]")
    if errors:
        raise DataValidationError(
            "predictions.csv validation failed:\n- " + "\n- ".join(errors)
        )
    print(
        f"predictions.csv validated: {len(frame)} rows, chronological order, "
        "no duplicates/NaN, finite values, ranges OK."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cwfc.predict",
        description="Generate the required predictions.csv from the challenge data.",
    )
    parser.add_argument(
        "--data", default="challenge_data", help="path to the challenge_data directory"
    )
    parser.add_argument(
        "--output", default="predictions.csv", help="output CSV path"
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="re-run rolling-origin model selection on the training window "
        "(default: use the frozen configuration)",
    )
    args = parser.parse_args(argv)

    output = Path(args.output)
    if str(output.parent) not in (".", ""):
        output.parent.mkdir(parents=True, exist_ok=True)
    generate_predictions(args.data, output, reselect=args.select)
    return 0


if __name__ == "__main__":
    sys.exit(main())
