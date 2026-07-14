"""Tests for ProcessOptimizer ScalarEI and MixerSession (generic configs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ProcessOptimizer")

from bayesian_optimization.microfluidics import MixerConfig, MixerSession
from bayesian_optimization.scalar_ei import ScalarEISuggester
from bayesian_optimization.space import DesignSpace


def test_scalar_ei_suggest_shape():
    space = DesignSpace(
        names=["a", "b", "c"],
        bounds={"a": (0.5, 10.0), "b": (0.5, 10.0), "c": (0.5, 10.0)},
        sum_equals=12.0,
    )
    opt = ScalarEISuggester(space, random_state=0)
    X = opt.sample_initial(6, seed=0)
    y = X[:, 0] / 10.0
    nxt = opt.suggest(X, y, n_points=2)
    assert nxt.shape == (2, 3)
    for row in nxt:
        assert abs(row.sum() - 12.0) < 1e-3
        space.check_feasible(row, tol=1e-3)


def test_mixer_session_csv_roundtrip(tmp_path: Path):
    cfg = MixerConfig(
        data_dir=str(tmp_path),
        var_to_syringe={"A": 0, "B": 1, "C": 2, "D": 3},
        bounds={k: (1.0, 40.0) for k in "ABCD"},
        n_syringes=8,
        total_speed=50.0,
        time_synth=30.0,
        default_n_points=3,
        random_state=0,
        separator_syringe=7,
        separator_speed=0.0,
        time_separator=20.0,
        legacy_csv=True,
    )
    mixer = MixerSession(cfg)
    path0 = mixer.generate_lhs_iter0(n_points=3)
    assert Path(path0).is_file()

    X = mixer.sample_initial(3, seed=1)
    rows = [list(range(8)) + [""]]
    for i, x in enumerate(X):
        speeds = [0.0] * 8
        for val, name in zip(x, mixer.var_names):
            speeds[mixer.var_to_syringe[name]] = float(val)
        rows.append(speeds + [30.0, 0.5 + 0.1 * i])
        rows.append([0.0] * 8 + [20.0])
    res = tmp_path / "recipes_iter_000_results.csv"
    pd.DataFrame(rows).to_csv(res, index=False, header=False)

    out = mixer.suggest_next(iter_idx=0, n_points=2)
    assert Path(out).is_file()
    assert "recipes_iter_001.csv" in out
