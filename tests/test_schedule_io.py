"""Tests for schedule CSV → ScheduleRunner DataFrame shape (no hardware)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bayesian_optimization.microfluidics.schedule_io import load_schedule_dataframe


def test_load_legacy_schedule(tmp_path: Path):
    rows = [
        list(range(8)) + [""],
        [10.0, 10.0, 10.0, 20.0, 0, 0, 0, 0, 30.0],
        [0, 0, 0, 0, 0, 0, 0, 0, 20.0],
    ]
    path = tmp_path / "recipes_iter_000.csv"
    pd.DataFrame(rows).to_csv(path, index=False, header=False)

    sch = load_schedule_dataframe(path, n_syringes=8)
    assert list(sch.columns[:8]) == [str(i) for i in range(8)]
    assert "8" in sch.columns
    assert len(sch) == 2
    assert float(sch.iloc[0]["8"]) == 30.0
