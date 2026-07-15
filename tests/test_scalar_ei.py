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


def test_scalar_ei_with_fixed_and_linear_deps():
    """Au-style chemistry space: free box → project onto fixed / deps / sum."""
    space = DesignSpace(
        names=["au", "ag", "peg", "ctab", "cit", "pvp", "ascorb", "teos"],
        bounds={
            "au": (3.3, 8.5),
            "peg": (1e-3, 19.999),
            "ctab": (1e-3, 19.999),
            "cit": (1e-3, 19.999),
            "pvp": (1e-3, 19.999),
            "ag": (0.0, 1.0),
            "ascorb": (0.0, 20.0),
            "teos": (0.0, 1.0),
        },
        sum_equals=20.0,
        fixed={"ag": 0.0, "teos": 0.0},
        linear_deps={"ascorb": {"au": 1.2, "ag": 1.2}},
    )
    opt = ScalarEISuggester(space, random_state=0, maximize=True)
    X = opt.sample_initial(5, seed=0)
    y = -((X[:, 0] - 5.0) ** 2)
    nxt = opt.suggest(X, y, n_points=1)
    assert nxt.shape == (1, 8)
    space.check_feasible(nxt[0], tol=1e-4)


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
    space = DesignSpace(
        names=["A", "B", "C", "D"],
        bounds={k: (1.0, 40.0) for k in "ABCD"},
        sum_equals=50.0,
    )
    mixer = MixerSession(cfg, ScalarEISuggester(space, random_state=0, maximize=True))
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
