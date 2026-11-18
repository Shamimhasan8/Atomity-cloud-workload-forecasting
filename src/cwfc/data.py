"""Data loading, validation and target construction.

This module implements the preprocessing contract described in
``DATASET.md``:

* load the two release files and validate their schemas;
* detect duplicate ``(vm_id, timestamp)`` records and invalid CPU readings;
* preserve chronological ordering;
* construct the two required aggregate targets on the five-minute grid:

  - ``active_vm_count``: number of distinct VM IDs with a *valid* CPU
    observation at each timestamp (a VM counts at most once);
  - ``aggregate_cpu_utilization``: mean of the VM-level ``avg_cpu`` values
    present at each timestamp, expressed as a percentage in ``[0, 100]``.

A reading is *valid* when ``avg_cpu`` is finite and lies within ``[0, 100]``.
Invalid readings are excluded from both targets and reported by
:meth:`Dataset.report` so the treatment is transparent.

The target index is the sorted set of timestamps observed in the CPU file
(we never invent timestamps that the release does not contain).  Internal
gaps, if any, remain in the frame with ``active_vm_count == 0`` and a NaN
utilization so downstream code can detect them explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CPU_COLUMNS = ("timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu")
META_COLUMNS = (
    "vm_id",
    "subscription_id",
    "deployment_id",
    "timestamp_vm_created",
    "timestamp_vm_deleted",
    "max_cpu",
    "avg_cpu",
    "p95_max_cpu",
    "vm_category",
    "virtual_core_bucket",
    "memory_gb_bucket",
)

# Metadata fields that encode information from *after* the observation
# window (full-lifetime CPU statistics, deletion timestamps).  They must
# never be used as forecasting features - see ASSIGNMENT.md section 9.
FUTURE_METADATA_FIELDS = (
    "max_cpu",
    "avg_cpu",
    "p95_max_cpu",
    "timestamp_vm_deleted",
)

TARGET_ACTIVE_VM_COUNT = "active_vm_count"
TARGET_AGGREGATE_CPU = "aggregate_cpu_utilization"
TARGETS = (TARGET_ACTIVE_VM_COUNT, TARGET_AGGREGATE_CPU)


class DataValidationError(ValueError):
    """Raised when input data violates the challenge dataset contract."""


@dataclass
class Dataset:
    """Validated challenge dataset plus the derived aggregate targets.

    Attributes
    ----------
    cpu:
        VM-level CPU readings (validated, sorted chronologically).
    metadata:
        VM metadata (one row per VM).
    targets:
        Aggregate target frame indexed by timestamp with columns
        ``active_vm_count`` and ``aggregate_cpu_utilization``.
    notes:
        Human-readable record of every preprocessing decision taken
        (duplicates dropped, invalid readings, gaps, ...).
    """

    cpu: pd.DataFrame
    metadata: pd.DataFrame
    targets: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        """Return a human-readable summary of the preprocessing decisions."""
        return "\n".join(f"- {note}" for note in self.notes)

    @property
    def cadence_seconds(self) -> int:
        """Modal spacing (seconds) between consecutive target timestamps."""
        if len(self.targets) < 2:
            return 300
        diffs = self.targets.index.to_series().diff().dropna()
        return int(diffs.mode().iloc[0])


def _check_columns(frame: pd.DataFrame, expected: tuple[str, ...], name: str) -> None:
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise DataValidationError(f"{name}: missing required columns {missing}")
    extra = [c for c in frame.columns if c not in expected]
    if extra:
        raise DataValidationError(f"{name}: unexpected columns {extra}")


def load_cpu_readings(path: str | Path) -> pd.DataFrame:
    """Load ``vm_cpu_readings.csv.gz`` and validate its schema."""
    frame = pd.read_csv(path)
    _check_columns(frame, CPU_COLUMNS, "vm_cpu_readings")
    if frame.empty:
        raise DataValidationError("vm_cpu_readings: file contains no rows")
    for col in ("min_cpu", "max_cpu", "avg_cpu"):
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise DataValidationError(f"vm_cpu_readings: column {col!r} must be numeric")
    if not pd.api.types.is_numeric_dtype(frame["timestamp"]):
        raise DataValidationError("vm_cpu_readings: 'timestamp' must be numeric")
    return frame


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load ``vm_metadata.csv.gz`` and validate its schema."""
    frame = pd.read_csv(path)
    _check_columns(frame, META_COLUMNS, "vm_metadata")
    if frame.empty:
        raise DataValidationError("vm_metadata: file contains no rows")
    return frame


def _clean_cpu(cpu: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Sort, de-duplicate and flag invalid readings in the CPU frame."""
    cpu = cpu.copy()

    n_dup = int(cpu.duplicated(subset=["vm_id", "timestamp"]).sum())
    if n_dup:
        cpu = cpu.drop_duplicates(subset=["vm_id", "timestamp"], keep="first")
        notes.append(
            f"Dropped {n_dup} duplicate (vm_id, timestamp) rows, keeping the first occurrence."
        )
    else:
        notes.append("No duplicate (vm_id, timestamp) rows found.")

    cpu = cpu.sort_values(["timestamp", "vm_id"], kind="stable").reset_index(drop=True)

    avg = cpu["avg_cpu"]
    invalid = ~np.isfinite(avg) | (avg < 0) | (avg > 100)
    n_invalid = int(invalid.sum())
    if n_invalid:
        notes.append(
            f"Excluded {n_invalid} rows with invalid avg_cpu "
            "(non-finite or outside [0, 100]) from both targets."
        )
    else:
        notes.append("All avg_cpu readings are finite and within [0, 100].")

    cpu["_valid"] = ~invalid
    return cpu


def build_targets(cpu: pd.DataFrame) -> pd.DataFrame:
    """Construct the aggregate target series from VM-level readings.

    Parameters
    ----------
    cpu:
        CPU frame already carrying the boolean ``_valid`` flag produced by
        :func:`_clean_cpu`.

    Returns
    -------
    pd.DataFrame
        Indexed by the observed timestamps (ascending) with columns
        ``active_vm_count`` and ``aggregate_cpu_utilization``.
    """
    valid = cpu[cpu["_valid"]]
    grouped = valid.groupby("timestamp", sort=True)

    targets = pd.DataFrame(
        {
            TARGET_ACTIVE_VM_COUNT: grouped["vm_id"].nunique().astype(int),
            TARGET_AGGREGATE_CPU: grouped["avg_cpu"].mean().astype(float),
        }
    )
    # Timestamps present in the raw file but with zero valid readings keep a
    # row (count 0, NaN utilization) instead of being silently dropped.
    targets = targets.reindex(sorted(cpu["timestamp"].unique()))
    targets[TARGET_ACTIVE_VM_COUNT] = targets[TARGET_ACTIVE_VM_COUNT].fillna(0).astype(int)
    return targets


def load_dataset(data_dir: str | Path, validate_meta: bool = True) -> Dataset:
    """Load, validate and aggregate the challenge dataset.

    Parameters
    ----------
    data_dir:
        Directory containing ``vm_cpu_readings.csv.gz`` and
        ``vm_metadata.csv.gz`` (the ``challenge_data/`` directory).
    validate_meta:
        Cross-check that the CPU and metadata VM populations match.

    Returns
    -------
    Dataset
        Validated dataset with derived targets and preprocessing notes.

    Raises
    ------
    DataValidationError
        If any schema or cross-file contract is violated, or if the target
        construction produces no usable timestamps.
    """
    data_dir = Path(data_dir)
    notes: list[str] = []

    cpu_path = data_dir / "vm_cpu_readings.csv.gz"
    meta_path = data_dir / "vm_metadata.csv.gz"
    if not cpu_path.exists():
        raise DataValidationError(f"missing CPU readings file: {cpu_path}")
    if not meta_path.exists() and validate_meta:
        raise DataValidationError(f"missing metadata file: {meta_path}")

    cpu_raw = load_cpu_readings(cpu_path)
    notes.append(
        f"Loaded {len(cpu_raw):,} CPU rows for {cpu_raw['vm_id'].nunique():,} VMs."
    )

    cpu = _clean_cpu(cpu_raw, notes)

    metadata = load_metadata(meta_path) if meta_path.exists() else pd.DataFrame(
        columns=list(META_COLUMNS)
    )

    if validate_meta and not metadata.empty:
        cpu_vms = set(cpu["vm_id"])
        meta_vms = set(metadata["vm_id"])
        if cpu_vms != meta_vms:
            only_cpu = len(cpu_vms - meta_vms)
            only_meta = len(meta_vms - cpu_vms)
            raise DataValidationError(
                "VM populations differ between files "
                f"({only_cpu} VMs only in CPU readings, {only_meta} only in metadata)."
            )
        if metadata["vm_id"].duplicated().any():
            raise DataValidationError("vm_metadata: duplicate vm_id rows.")
        notes.append("CPU and metadata VM populations match exactly.")

    targets = build_targets(cpu)
    if targets.empty:
        raise DataValidationError("no valid observations: target series is empty.")

    n_gaps = int(targets[TARGET_AGGREGATE_CPU].isna().sum())
    if n_gaps:
        notes.append(
            f"{n_gaps} timestamp(s) have no valid readings "
            "(active_vm_count=0, utilization=NaN)."
        )
    else:
        notes.append("Every observed timestamp has at least one valid reading.")

    cadence = Dataset(cpu, metadata, targets, notes).cadence_seconds
    notes.append(f"Modal timestamp cadence: {cadence} seconds.")

    return Dataset(cpu=cpu, metadata=metadata, targets=targets, notes=notes)


def chronological_split(
    targets: pd.DataFrame,
    train_size: int = 110,
    horizon: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the target frame into the official train/holdout windows.

    The first ``train_size`` timestamps form the public training window and
    the final ``horizon`` timestamps form the untouched holdout, following
    ASSIGNMENT.md.  The two windows never overlap.

    Raises
    ------
    DataValidationError
        If the series is shorter than ``train_size + horizon``.
    """
    n = len(targets)
    if n < train_size + horizon:
        raise DataValidationError(
            f"need at least {train_size + horizon} timestamps for the "
            f"{train_size}/{horizon} split, found {n}."
        )
    train = targets.iloc[:train_size]
    holdout = targets.iloc[train_size:]
    return train, holdout
