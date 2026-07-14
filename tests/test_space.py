"""Tests for DesignSpace projection and sampling (experiment-agnostic)."""

from __future__ import annotations

import numpy as np

from bayesian_optimization.space import DesignSpace, make_sum_space


def test_sum_projection():
    space = make_sum_space(
        names=[f"x{i}" for i in range(8)],
        bounds={f"x{i}": (0.3, 40.0) for i in range(8)},
        sum_equals=40.0,
    )
    x = np.ones(8) * 10.0
    y = space.project_to_feasible(x)
    assert y.shape == (8,)
    assert abs(y.sum() - 40.0) < 1e-5
    space.check_feasible(y)


def test_lhs_sum():
    space = make_sum_space(
        names=[f"x{i}" for i in range(8)],
        bounds={f"x{i}": (0.3, 40.0) for i in range(8)},
        sum_equals=40.0,
    )
    pts = space.sample_feasible(5, seed=0, method="lhs")
    assert pts.shape == (5, 8)
    for row in pts:
        assert abs(row.sum() - 40.0) < 1e-4
        space.check_feasible(row, tol=1e-4)


def test_fixed_and_linear_deps():
    space = DesignSpace(
        names=["a", "b", "c", "d"],
        bounds={"a": (1.0, 5.0), "d": (0.1, 10.0)},
        sum_equals=20.0,
        fixed={"b": 0.0},
        linear_deps={"c": {"a": 1.2}},
    )
    pts = space.sample_feasible(15, seed=42, method="rejection")
    for row in pts:
        d = space.to_dict(row)
        assert abs(d["b"]) < 1e-8
        assert abs(d["c"] - 1.2 * d["a"]) < 1e-5
        assert abs(sum(d.values()) - 20.0) < 1e-4
        space.check_feasible(row, tol=1e-4)


def test_fixed_only():
    space = DesignSpace(
        names=["a", "b", "c"],
        bounds={"a": (1.0, 3.0), "b": (0.1, 5.0)},
        sum_equals=10.0,
        fixed={"c": 2.0},
    )
    pts = space.sample_feasible(5, seed=1, method="rejection")
    for row in pts:
        assert abs(row[2] - 2.0) < 1e-8
        assert abs(row.sum() - 10.0) < 1e-4
