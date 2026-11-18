"""Integration tests: the prediction CLI on synthetic smoke data.

These mirror the reviewer scenario in SUBMISSION.md: replace the public
challenge data with a tiny internal smoke dataset and check the pipeline is
not hard-coded to the supplied files (different VM IDs, timestamps, cadence,
churn, dirty rows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.predict import (
    PREDICTION_COLUMNS,
    adaptive_split,
    generate_predictions,
    validate_predictions,
)


@pytest.fixture()
def smoke_dir(tmp_path):
    rng = np.random.default_rng(0)
    n_ts, n_vms = 60, 6
    rows = []
    for vm in range(n_vms):
        start = int(rng.integers(0, 5))
        end = n_ts - int(rng.integers(0, 8))  # churn: VMs disappear late
        for t in range(start, end):
            rows.append(
                (t * 60, f"smoke-vm-{vm}", 0.0, 10.0,
                 float(np.clip(10 + vm + 3 * np.sin(t / 7), 0, 100)))
            )
    cpu = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
    # dirty rows: a duplicate and an out-of-range reading
    cpu = pd.concat([cpu, cpu.iloc[[5]]], ignore_index=True)
    cpu.loc[cpu.index[-1], "avg_cpu"] = 200.0
    meta = pd.DataFrame(
        {
            "vm_id": [f"smoke-vm-{i}" for i in range(n_vms)],
            "subscription_id": ["s"] * n_vms,
            "deployment_id": ["d"] * n_vms,
            "timestamp_vm_created": [0] * n_vms,
            "timestamp_vm_deleted": [999999] * n_vms,
            "max_cpu": [50.0] * n_vms,
            "avg_cpu": [11.0] * n_vms,
            "p95_max_cpu": [40.0] * n_vms,
            "vm_category": ["Interactive", "Delay-insensitive", "Unknown"] * 2,
            "virtual_core_bucket": ["2"] * n_vms,
            "memory_gb_bucket": ["2"] * n_vms,
        }
    )
    cpu.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    meta.to_csv(tmp_path / "vm_metadata.csv.gz", index=False, compression="gzip")
    return tmp_path


class TestAdaptiveSplit:
    def test_official_size_uses_protocol(self):
        # n == 134 reproduces the documented 110/24 split exactly
        assert adaptive_split(134) == (110, 24)

    def test_larger_data_keeps_final_24_holdout(self):
        # the holdout is always the final 24 timestamps; everything before
        # them is training (latent crash fixed: holdout length == horizon)
        assert adaptive_split(140) == (116, 24)
        assert adaptive_split(200) == (176, 24)

    def test_smaller_data_scales_proportionally(self):
        train, horizon = adaptive_split(60)
        assert train + horizon == 60
        assert horizon == round(60 * 24 / 134)


class TestGeneratePredictions:
    def test_smoke_dataset_end_to_end(self, smoke_dir):
        out = generate_predictions(
            smoke_dir, smoke_dir / "predictions.csv", quiet=True
        )
        # The dirty rows were handled. Note the deliberate interaction in
        # this fixture: the out-of-range 200.0 reading rides ON the appended
        # duplicate row, so it disappears together with the duplicate
        # (keep='first') and no separate invalid-reading note is expected —
        # invalid-reading exclusion is covered by the unit tests in
        # test_data.py.
        from cwfc.data import load_dataset

        notes = "\n".join(load_dataset(smoke_dir).notes)
        assert "duplicate (vm_id, timestamp) rows" in notes
        assert "All avg_cpu readings are finite and within [0, 100]" in notes
        assert list(out.columns) == PREDICTION_COLUMNS
        # Proportional split: 60 timestamps -> holdout 11
        assert len(out) == 11
        # Exact holdout timestamps, in order, from the source grid
        expected_ts = [t * 60 for t in range(49, 60)]
        assert out["timestamp"].tolist() == expected_ts
        assert out["timestamp"].is_monotonic_increasing
        assert (out["active_vm_count"] >= 0).all()
        assert out["aggregate_cpu_utilization"].between(0, 100).all()
        assert np.isfinite(out[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy()).all()
        # The file was written and matches the frame
        on_disk = pd.read_csv(smoke_dir / "predictions.csv")
        pd.testing.assert_frame_equal(on_disk, out)

    def test_holdout_perturbation_invariance(self, tmp_path):
        """Perturbing the HOLDOUT observations must not change predictions.

        This is the end-to-end leakage guard: the pipeline may read holdout
        timestamps, never holdout values.
        """
        import copy

        def build(base_dir, corrupt_holdout: bool):
            rng = np.random.default_rng(7)
            n_ts, n_vms = 140, 5  # > 134: official-style final-24 holdout
            rows = []
            for vm in range(n_vms):
                start = int(rng.integers(0, 4))
                for t in range(start, n_ts):
                    val = 20 + vm + 4 * np.sin(t / 9) + 0.5 * t / n_ts
                    if corrupt_holdout and t >= n_ts - 24:
                        val = 95.0 + vm  # wildly different future
                    rows.append((t * 300, f"vm-{vm}", 0.0, 100.0, float(val)))
            cpu = pd.DataFrame(
                rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]
            )
            meta = pd.DataFrame(
                {
                    "vm_id": [f"vm-{i}" for i in range(n_vms)],
                    "subscription_id": ["s"] * n_vms,
                    "deployment_id": ["d"] * n_vms,
                    "timestamp_vm_created": [0] * n_vms,
                    "timestamp_vm_deleted": [999999] * n_vms,
                    "max_cpu": [50.0] * n_vms,
                    "avg_cpu": [21.0] * n_vms,
                    "p95_max_cpu": [40.0] * n_vms,
                    "vm_category": ["Interactive", "Delay-insensitive", "Unknown", "Interactive", "Delay-insensitive"],
                    "virtual_core_bucket": ["2"] * n_vms,
                    "memory_gb_bucket": ["2"] * n_vms,
                }
            )
            cpu.to_csv(base_dir / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
            meta.to_csv(base_dir / "vm_metadata.csv.gz", index=False, compression="gzip")

        clean_dir, corrupt_dir = tmp_path / "clean", tmp_path / "corrupt"
        clean_dir.mkdir(), corrupt_dir.mkdir()
        build(clean_dir, corrupt_holdout=False)
        build(corrupt_dir, corrupt_holdout=True)

        out_clean = generate_predictions(clean_dir, clean_dir / "p.csv", quiet=True)
        out_corrupt = generate_predictions(corrupt_dir, corrupt_dir / "p.csv", quiet=True)
        pd.testing.assert_frame_equal(
            out_clean, out_corrupt, check_exact=True,
        ), "predictions depend on holdout values (leakage)"

    def test_larger_than_official_dataset(self, tmp_path):
        """A dataset with MORE than 134 timestamps must still yield 24 rows."""
        rng = np.random.default_rng(3)
        n_ts, n_vms = 140, 4
        rows = [
            (t * 300, f"vm-{vm}", 0.0, 100.0,
             float(np.clip(15 + vm + 3 * np.sin(t / 8), 0, 100)))
            for vm in range(n_vms)
            for t in range(n_ts)
        ]
        cpu = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
        meta = pd.DataFrame(
            {
                "vm_id": [f"vm-{i}" for i in range(n_vms)],
                "subscription_id": ["s"] * n_vms, "deployment_id": ["d"] * n_vms,
                "timestamp_vm_created": [0] * n_vms,
                "timestamp_vm_deleted": [999999] * n_vms,
                "max_cpu": [50.0] * n_vms, "avg_cpu": [16.0] * n_vms,
                "p95_max_cpu": [40.0] * n_vms,
                "vm_category": ["Interactive"] * n_vms,
                "virtual_core_bucket": ["2"] * n_vms,
                "memory_gb_bucket": ["2"] * n_vms,
            }
        )
        cpu.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
        meta.to_csv(tmp_path / "vm_metadata.csv.gz", index=False, compression="gzip")

        out = generate_predictions(tmp_path, tmp_path / "p.csv", quiet=True)
        assert len(out) == 24
        # holdout = the FINAL 24 observed timestamps, exactly, in order
        expected_ts = [t * 300 for t in range(116, 140)]
        assert out["timestamp"].tolist() == expected_ts
        assert out["timestamp"].is_monotonic_increasing

    def test_missing_timestamp_all_invalid_rows_interpolated(self, tmp_path, capsys):
        """A timestamp whose readings are all invalid becomes a gap row
        (count 0, NaN utilization) and is interpolated in the train window."""
        rng = np.random.default_rng(5)
        n_ts, n_vms = 70, 4
        rows = []
        for vm in range(n_vms):
            for t in range(n_ts):
                val = float(np.clip(12 + vm + 3 * np.sin(t / 6), 0, 100))
                if t == 30:  # every reading at this timestamp is invalid
                    val = 999.0
                rows.append((t * 60, f"vm-{vm}", 0.0, 100.0, val))
        cpu = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
        meta = pd.DataFrame(
            {
                "vm_id": [f"vm-{i}" for i in range(n_vms)],
                "subscription_id": ["s"] * n_vms, "deployment_id": ["d"] * n_vms,
                "timestamp_vm_created": [0] * n_vms,
                "timestamp_vm_deleted": [999999] * n_vms,
                "max_cpu": [50.0] * n_vms, "avg_cpu": [13.0] * n_vms,
                "p95_max_cpu": [40.0] * n_vms, "vm_category": ["Interactive"] * n_vms,
                "virtual_core_bucket": ["2"] * n_vms, "memory_gb_bucket": ["2"] * n_vms,
            }
        )
        cpu.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
        meta.to_csv(tmp_path / "vm_metadata.csv.gz", index=False, compression="gzip")

        out = generate_predictions(tmp_path, tmp_path / "p.csv", quiet=False)
        assert "interpolated" in capsys.readouterr().out  # explicit warning emitted
        assert len(out) > 0
        assert out["aggregate_cpu_utilization"].between(0, 100).all()

        # The gap is visible in the target frame itself (count 0, NaN mean).
        from cwfc.data import load_dataset

        targets = load_dataset(tmp_path).targets
        assert targets.loc[30 * 60, "active_vm_count"] == 0
        assert pd.isna(targets.loc[30 * 60, "aggregate_cpu_utilization"])

    def test_missing_timestamp_absent_rows_grid_skips(self, tmp_path):
        """A timestamp with NO rows at all is absent from the observed grid;
        the pipeline still produces predictions on the final observed window."""
        rng = np.random.default_rng(6)
        n_ts, n_vms = 70, 4
        rows = []
        for vm in range(n_vms):
            for t in range(n_ts):
                if t == 31:  # entire timestamp absent
                    continue
                rows.append((t * 60, f"vm-{vm}", 0.0, 100.0,
                             float(np.clip(12 + vm + 3 * np.sin(t / 6), 0, 100))))
        cpu = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
        meta = pd.DataFrame(
            {
                "vm_id": [f"vm-{i}" for i in range(n_vms)],
                "subscription_id": ["s"] * n_vms, "deployment_id": ["d"] * n_vms,
                "timestamp_vm_created": [0] * n_vms,
                "timestamp_vm_deleted": [999999] * n_vms,
                "max_cpu": [50.0] * n_vms, "avg_cpu": [13.0] * n_vms,
                "p95_max_cpu": [40.0] * n_vms, "vm_category": ["Interactive"] * n_vms,
                "virtual_core_bucket": ["2"] * n_vms, "memory_gb_bucket": ["2"] * n_vms,
            }
        )
        cpu.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
        meta.to_csv(tmp_path / "vm_metadata.csv.gz", index=False, compression="gzip")

        out = generate_predictions(tmp_path, tmp_path / "p.csv", quiet=True)
        observed = sorted(cpu["timestamp"].unique())
        horizon = len(out)
        assert out["timestamp"].tolist() == observed[-horizon:]  # aligned to observed grid
        assert out["timestamp"].is_monotonic_increasing
        assert not out["timestamp"].duplicated().any()

    def test_reselect_path_works(self, smoke_dir):
        out = generate_predictions(
            smoke_dir, smoke_dir / "predictions.csv", reselect=True, quiet=True
        )
        assert len(out) == 11
        assert out["aggregate_cpu_utilization"].between(0, 100).all()

    def test_predictions_are_deterministic(self, smoke_dir):
        a = generate_predictions(smoke_dir, smoke_dir / "p1.csv", quiet=True)
        b = generate_predictions(smoke_dir, smoke_dir / "p2.csv", quiet=True)
        pd.testing.assert_frame_equal(a, b)


class TestValidatePredictions:
    def test_valid_frame_passes(self):
        frame = pd.DataFrame(
            {
                "timestamp": [33000, 33300],
                "active_vm_count": [10000.0, 10000.0],
                "aggregate_cpu_utilization": [9.2, 9.1],
            }
        )
        validate_predictions(frame, n_expected=2)  # no exception

    def test_rejects_bad_frames(self):
        good = {
            "timestamp": [1, 2],
            "active_vm_count": [1.0, 1.0],
            "aggregate_cpu_utilization": [9.0, 9.0],
        }
        from cwfc.data import DataValidationError

        with pytest.raises(DataValidationError, match="exactly 1 rows"):
            validate_predictions(pd.DataFrame(good), n_expected=1)
        bad = dict(good, active_vm_count=[-1.0, 1.0])
        with pytest.raises(DataValidationError, match="negative"):
            validate_predictions(pd.DataFrame(bad), n_expected=2)
        bad = dict(good, aggregate_cpu_utilization=[101.0, 9.0])
        with pytest.raises(DataValidationError, match="outside"):
            validate_predictions(pd.DataFrame(bad), n_expected=2)
        bad = dict(good, aggregate_cpu_utilization=[float("nan"), 9.0])
        with pytest.raises(DataValidationError, match="NaN"):
            validate_predictions(pd.DataFrame(bad), n_expected=2)
        bad = dict(good, timestamp=[2, 1])
        with pytest.raises(DataValidationError, match="chronological"):
            validate_predictions(pd.DataFrame(bad), n_expected=2)
