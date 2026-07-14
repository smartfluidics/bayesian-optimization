"""Smoke test for UncertaintySuggester (skipped without torch/gpytorch)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gpytorch")

from bayesian_optimization.space import make_sum_space
from bayesian_optimization.uncertainty import UncertaintySuggester


def test_uncertainty_suggest_smoke():
    space = make_sum_space(
        names=["a", "b", "c", "d", "e", "f", "g", "h"],
        bounds={
            "a": (3.3, 8.5),
            "c": (1e-3, 19.0),
            "d": (1e-3, 19.0),
            "e": (1e-3, 19.0),
            "f": (1e-3, 19.0),
        },
        sum_equals=20.0,
        fixed={"b": 0.0, "h": 0.0},
        linear_deps={"g": {"a": 1.2, "b": 1.2}},
    )
    opt = UncertaintySuggester(
        space,
        n_candidates=200,
        training_iter=15,
        seed=0,
        output_names=["y0", "y1", "y2", "y3"],
    )
    X = opt.sample_initial(12, seed=0)
    rng = np.random.default_rng(0)
    y = np.column_stack(
        [
            X[:, 0] / 10.0 + 0.05 * rng.normal(size=len(X)),
            X[:, 2] / 10.0 + 0.05 * rng.normal(size=len(X)),
            X[:, 3] / 10.0 + 0.05 * rng.normal(size=len(X)),
            X[:, 5] / 10.0 + 0.05 * rng.normal(size=len(X)),
        ]
    )
    nxt = opt.suggest(X, y, n_points=1)
    assert nxt.shape == (1, 8)
    space.check_feasible(nxt[0], tol=1e-4)
