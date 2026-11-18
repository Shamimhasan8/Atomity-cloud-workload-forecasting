"""Unit tests for cwfc.data: loading, validation and target construction."""
from __future__ import annotations

import pandas as pd
import pytest

from cwfc.data import (
    CPU_COLUMNS,
    DataValidationError,
    build_targets,
    chronological_split,
    load_dataset,
    _clean_cpu,
)


def _cpu_frame(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]
    )


class TestCleanAndTargets:
    def test_basic_targets(self):
        cpu = _cpu_frame(
            [
                (0, "vm1", 1.0, 3.0, 2.0),
                (0, "vm2", 0.0, 1.0, 0.5),
                (300, "vm1", 1.0, 3.0, 2.5),
            ]
        )
        cleaned = _clean_cpu(cpu, notes := [])
        targets = build_targets(cleaned)
        assert list(targets.index) == [0, 300]
        assert targets["active_vm_count"].tolist() == [2, 1]
        assert targets["aggregate_cpu_utilization"].tolist() == [1.25, 2.5]

    def test_vm_counted_once_per_timestamp(self):
        """Duplicate (vm_id, timestamp) rows must not inflate the count."""
        cpu = _cpu_frame(
            [
                (0, "vm1", 1.0, 3.0, 2.0),
                (0, "vm1", 1.0, 3.0, 9.0),  # duplicate row
            ]
        )
        cleaned = _clean_cpu(cpu, notes := [])
        assert len(cleaned) == 1
        targets = build_targets(cleaned)
        assert targets["active_vm_count"].iloc[0] == 1
        assert targets["aggregate_cpu_utilization"].iloc[0] == pytest.approx(2.0)

    def test_invalid_readings_excluded(self):
        cpu = _cpu_frame(
            [
                (0, "vm1", 0.0, 1.0, 150.0),  # out of range -> invalid
                (0, "vm2", 0.0, 1.0, 50.0),
                (300, "vm1", 0.0, 1.0, float("nan")),  # NaN -> invalid
            ]
        )
        cleaned = _clean_cpu(cpu, notes := [])
        targets = build_targets(cleaned)
        assert targets["active_vm_count"].tolist() == [1, 0]
        assert targets["aggregate_cpu_utilization"].iloc[0] == pytest.approx(50.0)
        assert pd.isna(targets["aggregate_cpu_utilization"].iloc[1])

    def test_chronological_ordering_preserved(self):
        cpu = _cpu_frame(
            [
                (600, "vm1", 0.0, 1.0, 1.0),
                (0, "vm1", 0.0, 1.0, 2.0),
                (300, "vm1", 0.0, 1.0, 3.0),
            ]
        )
        cleaned = _clean_cpu(cpu, notes := [])
        assert cleaned["timestamp"].is_monotonic_increasing
        targets = build_targets(cleaned)
        assert list(targets.index) == [0, 300, 600]

    def test_timestamp_with_no_valid_readings_kept(self):
        cpu = _cpu_frame(
            [
                (0, "vm1", 0.0, 1.0, 10.0),
                (300, "vm1", 0.0, 1.0, float("nan")),
                (600, "vm1", 0.0, 1.0, 20.0),
            ]
        )
        cleaned = _clean_cpu(cpu, notes := [])
        targets = build_targets(cleaned)
        assert list(targets.index) == [0, 300, 600]
        assert targets["active_vm_count"].tolist() == [1, 0, 1]


class TestChronologicalSplit:
    def test_official_split_sizes(self):
        idx = pd.RangeIndex(134)
        targets = pd.DataFrame(
            {"active_vm_count": 10, "aggregate_cpu_utilization": 9.0}, index=idx
        )
        train, holdout = chronological_split(targets, train_size=110, horizon=24)
        assert len(train) == 110
        assert len(holdout) == 24
        assert train.index[-1] == 109
        assert holdout.index[0] == 110
        assert holdout.index[-1] == 133
        # No overlap between the windows.
        assert not set(train.index) & set(holdout.index)

    def test_short_series_raises(self):
        targets = pd.DataFrame({"active_vm_count": [1] * 100})
        with pytest.raises(DataValidationError):
            chronological_split(targets, train_size=110, horizon=24)


class TestSchemaValidation:
    def test_missing_column_raises(self, tmp_path):
        bad = pd.DataFrame({"timestamp": [0], "vm_id": ["a"], "avg_cpu": [1.0]})
        bad.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
        pd.DataFrame(columns=list(CPU_COLUMNS)).to_csv(
            tmp_path / "vm_metadata.csv.gz", index=False, compression="gzip"
        )
        with pytest.raises(DataValidationError, match="missing required columns"):
            load_dataset(tmp_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DataValidationError, match="missing"):
            load_dataset(tmp_path)

    def test_population_mismatch_raises(self, tmp_path):
        cpu = _cpu_frame([(0, "vm1", 0.0, 1.0, 5.0)])
        meta = pd.DataFrame(
            [
                {
                    "vm_id": "vm_other",
                    "subscription_id": "s",
                    "deployment_id": "d",
                    "timestamp_vm_created": 0,
                    "timestamp_vm_deleted": 1,
                    "max_cpu": 1.0,
                    "avg_cpu": 1.0,
                    "p95_max_cpu": 1.0,
                    "vm_category": "Unknown",
                    "virtual_core_bucket": "2",
                    "memory_gb_bucket": "2",
                }
            ]
        )
        cpu.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
        meta.to_csv(tmp_path / "vm_metadata.csv.gz", index=False, compression="gzip")
        with pytest.raises(DataValidationError, match="populations differ"):
            load_dataset(tmp_path)


class TestSyntheticSmokeDataset:
    """A tiny synthetic dataset shaped like the challenge release."""

    @pytest.fixture()
    def smoke_dir(self, tmp_path):
        rows = []
        for t in range(4):  # 4 timestamps
            for vm in range(3):  # 3 VMs
                rows.append((t * 300, f"vm{vm}", 0.0, 10.0, 10.0 + vm + t))
        cpu = _cpu_frame(rows)
        meta = pd.DataFrame(
            {
                "vm_id": ["vm0", "vm1", "vm2"],
                "subscription_id": ["s0", "s1", "s2"],
                "deployment_id": ["d0", "d1", "d2"],
                "timestamp_vm_created": [0, 0, 0],
                "timestamp_vm_deleted": [9e6, 9e6, 9e6],
                "max_cpu": [50.0, 50.0, 50.0],
                "avg_cpu": [11.0, 12.0, 13.0],
                "p95_max_cpu": [40.0, 40.0, 40.0],
                "vm_category": ["Interactive", "Delay-insensitive", "Unknown"],
                "virtual_core_bucket": ["2", "4", "8"],
                "memory_gb_bucket": ["2", "4", "8"],
            }
        )
        cpu.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
        meta.to_csv(tmp_path / "vm_metadata.csv.gz", index=False, compression="gzip")
        return tmp_path

    def test_load_smoke_dataset(self, smoke_dir):
        ds = load_dataset(smoke_dir)
        assert len(ds.targets) == 4
        assert ds.targets["active_vm_count"].eq(3).all()
        # mean of 10+t+vm over vm in {0,1,2} == 11+t
        assert ds.targets["aggregate_cpu_utilization"].tolist() == [11.0, 12.0, 13.0, 14.0]
        assert ds.cadence_seconds == 300
