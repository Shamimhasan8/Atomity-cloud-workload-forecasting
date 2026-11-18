"""Notebook execution tests: every notebook must run from a clean kernel
with zero errors, on the official challenge dataset.

These tests self-skip when the dataset or a Jupyter installation is absent
(e.g. in a minimal CI environment), so the standard suite stays usable; run
them explicitly with ``python -m pytest -q`` on a full checkout.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["01_eda", "02_forecasting", "03_evaluation"]


def _prerequisites() -> None:
    data = REPO / "challenge_data" / "vm_cpu_readings.csv.gz"
    if not data.exists():
        pytest.skip("challenge dataset not present under challenge_data/")
    if shutil.which("jupyter") is None:
        pytest.skip("jupyter/nbconvert not installed (pip install -e '.[notebooks]')")


@pytest.mark.notebooks
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_executes_without_errors(name: str, tmp_path: Path) -> None:
    """Execute the notebook in a fresh kernel; fail on any cell error.

    This is a regression test for the review finding of silently recorded
    ``ValueError`` outputs in ``03_evaluation.ipynb``: nbconvert's executed
    output is parsed and *any* error output fails the test, even when the
    conversion itself exits 0.
    """
    _prerequisites()
    src = REPO / "notebooks" / f"{name}.ipynb"
    proc = subprocess.run(
        [
            "jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=1200",
            f"--output-dir={tmp_path}",
            str(src),
        ],
        capture_output=True,
        text=True,
        timeout=1500,
        cwd=REPO,
    )
    assert proc.returncode == 0, (
        f"{name}: nbconvert failed\nstdout:\n{proc.stdout[-1500:]}\nstderr:\n{proc.stderr[-1500:]}"
    )

    executed = tmp_path / f"{name}.ipynb"
    assert executed.exists(), f"{name}: executed notebook not written"
    nb = json.loads(executed.read_text())
    errors = [
        (i, out.get("ename"), out.get("evalue"))
        for i, cell in enumerate(nb["cells"])
        if cell.get("cell_type") == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not errors, f"{name}: {len(errors)} cell error(s): {errors}"

    executed_cells = sum(1 for c in nb["cells"] if c.get("cell_type") == "code")
    assert executed_cells >= 5, f"{name}: suspiciously few code cells executed"
