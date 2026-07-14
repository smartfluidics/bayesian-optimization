"""Design space, feasibility projection, and candidate sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class DesignSpace:
    """Box bounds with optional sum, fixed values, and affine dependencies.

    ``linear_deps[name] = {other: coef, ...}`` means
    ``name = sum_j coef_j * other_j``.
    """

    names: list[str]
    bounds: dict[str, tuple[float, float]]
    sum_equals: float | None = None
    fixed: dict[str, float] = field(default_factory=dict)
    linear_deps: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("names must be non-empty")
        if len(set(self.names)) != len(self.names):
            raise ValueError("names must be unique")
        for name in self.names:
            if name not in self.bounds and name not in self.fixed and name not in self.linear_deps:
                raise ValueError(f"{name}: missing bounds (or fixed / linear_deps)")
        for name, (lo, hi) in self.bounds.items():
            if lo >= hi:
                raise ValueError(f"{name}: bound must satisfy low < high")
        for name in self.fixed:
            if name not in self.names:
                raise ValueError(f"fixed key {name} not in names")
        for name, coefs in self.linear_deps.items():
            if name not in self.names:
                raise ValueError(f"linear_deps key {name} not in names")
            for other in coefs:
                if other not in self.names:
                    raise ValueError(f"linear_deps[{name}] references unknown {other}")

    @property
    def n_dims(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def free_names(self) -> list[str]:
        return [n for n in self.names if n not in self.fixed and n not in self.linear_deps]

    def primary_free_names(self) -> list[str]:
        """Free vars that appear on the RHS of linear_deps (sampled before remainder)."""
        deps_inputs: set[str] = set()
        for coefs in self.linear_deps.values():
            deps_inputs.update(coefs)
        free = set(self.free_names())
        primary = [n for n in self.names if n in free and n in deps_inputs]
        if primary:
            return primary
        return list(self.free_names())

    def remainder_free_names(self) -> list[str]:
        primary = set(self.primary_free_names())
        if self.linear_deps or self.fixed:
            return [n for n in self.free_names() if n not in primary]
        return []

    def to_array(self, row: dict[str, float] | Sequence[float]) -> np.ndarray:
        if isinstance(row, dict):
            return np.array([float(row[n]) for n in self.names], dtype=float)
        arr = np.asarray(row, dtype=float).reshape(-1)
        if arr.shape[0] != self.n_dims:
            raise ValueError(f"expected length {self.n_dims}, got {arr.shape[0]}")
        return arr

    def to_dict(self, x: Sequence[float]) -> dict[str, float]:
        arr = self.to_array(x)
        return {n: float(arr[i]) for i, n in enumerate(self.names)}

    def _apply_fixed_and_deps(self, values: dict[str, float]) -> dict[str, float]:
        out = dict(values)
        for name, val in self.fixed.items():
            out[name] = float(val)
        for name, coefs in self.linear_deps.items():
            out[name] = float(sum(coefs[o] * out[o] for o in coefs))
        return out

    def project_to_feasible(self, x: Sequence[float], max_iter: int = 16) -> np.ndarray:
        """Project a point onto bounds / fixed / linear deps / sum constraint."""
        values = self.to_dict(x)
        free = self.free_names()
        rem_names = self.remainder_free_names()
        primary = self.primary_free_names()

        for name in free:
            lo, hi = self.bounds[name]
            values[name] = float(np.clip(values[name], lo, hi))

        values = self._apply_fixed_and_deps(values)

        if self.sum_equals is None:
            return self.to_array(values)

        # Rescale free pool so total matches sum_equals.
        pool = rem_names if rem_names else free
        for _ in range(max_iter):
            values = self._apply_fixed_and_deps(values)
            total = float(sum(values[n] for n in self.names))
            if abs(total - self.sum_equals) < 1e-6:
                break
            fixed_part = total - float(sum(values[n] for n in pool))
            target_pool = self.sum_equals - fixed_part
            cur_pool = float(sum(values[n] for n in pool))
            if cur_pool <= 0:
                for name in pool:
                    lo, hi = self.bounds[name]
                    values[name] = 0.5 * (lo + hi)
                continue
            scale = target_pool / cur_pool
            for name in pool:
                lo, hi = self.bounds[name]
                values[name] = float(np.clip(values[name] * scale, lo, hi))

            # Keep primary free within bounds (already clipped); recompute deps.
            for name in primary:
                lo, hi = self.bounds[name]
                values[name] = float(np.clip(values[name], lo, hi))
            values = self._apply_fixed_and_deps(values)

        return self.to_array(values)

    def check_feasible(self, x: Sequence[float], tol: float = 1e-6) -> None:
        values = self.to_dict(x)
        for name, val in self.fixed.items():
            if abs(values[name] - val) > tol:
                raise ValueError(f"Constraint violation: {name} != {val}")
        for name, coefs in self.linear_deps.items():
            expected = sum(coefs[o] * values[o] for o in coefs)
            if abs(values[name] - expected) > tol:
                raise ValueError(f"Constraint violation: {name} != linear dep")
        for name in self.free_names():
            lo, hi = self.bounds[name]
            if not (lo - tol <= values[name] <= hi + tol):
                raise ValueError(f"Constraint violation: {name} out of [{lo}, {hi}]")
        if self.sum_equals is not None:
            s = float(sum(values[n] for n in self.names))
            if abs(s - self.sum_equals) > tol:
                raise ValueError(f"Constraint violation: sum={s} != {self.sum_equals}")

    def sample_feasible(
        self,
        n: int,
        *,
        seed: int = 0,
        method: str = "auto",
        eps: float = 1e-3,
    ) -> np.ndarray:
        """Sample ``n`` feasible points. method: auto | lhs | rejection."""
        if n <= 0:
            return np.zeros((0, self.n_dims), dtype=float)
        if method == "auto":
            method = "lhs" if not self.fixed and not self.linear_deps else "rejection"
        if method == "lhs":
            return self._sample_lhs(n, seed=seed)
        if method == "rejection":
            return self._sample_rejection(n, seed=seed, eps=eps)
        raise ValueError(f"unknown method: {method}")

    def _sample_lhs(self, n: int, *, seed: int) -> np.ndarray:
        free = self.free_names()
        if not free:
            raise ValueError("LHS sampling requires free variables")
        sampler = qmc.LatinHypercube(d=len(free), seed=seed)
        u = sampler.random(n=n)
        lows = np.array([self.bounds[name][0] for name in free], dtype=float)
        highs = np.array([self.bounds[name][1] for name in free], dtype=float)
        raw = lows + u * (highs - lows)
        if self.sum_equals is not None:
            raw = raw / raw.sum(axis=1, keepdims=True) * self.sum_equals
        out = np.zeros((n, self.n_dims), dtype=float)
        for i in range(n):
            values = {name: 0.0 for name in self.names}
            for j, name in enumerate(free):
                values[name] = float(raw[i, j])
            values = self._apply_fixed_and_deps(values)
            out[i] = self.project_to_feasible(self.to_array(values))
        return out

    def _sample_rejection(self, n: int, *, seed: int, eps: float) -> np.ndarray:
        rng = np.random.default_rng(seed)
        primary = self.primary_free_names()
        remainder = self.remainder_free_names()
        pts: list[np.ndarray] = []
        max_attempts = max(10_000, n * 400)

        for _ in range(max_attempts):
            values = {name: 0.0 for name in self.names}
            for name, val in self.fixed.items():
                values[name] = float(val)

            for name in primary:
                lo, hi = self.bounds[name]
                values[name] = float(rng.uniform(lo, hi))

            values = self._apply_fixed_and_deps(values)

            if remainder and self.sum_equals is not None:
                used = float(sum(values[n] for n in self.names if n not in remainder))
                rem = self.sum_equals - used
                if rem <= len(remainder) * eps:
                    continue
                w = rng.dirichlet(np.ones(len(remainder)))
                alloc = rem * w
                ok = True
                for name, val in zip(remainder, alloc):
                    lo, hi = self.bounds[name]
                    if val <= eps or val >= hi - eps or val < lo:
                        ok = False
                        break
                    values[name] = float(val)
                if not ok:
                    continue
            elif self.sum_equals is not None and not remainder:
                # All free already sampled; project.
                pass

            try:
                x = self.project_to_feasible(self.to_array(values))
                self.check_feasible(x, tol=1e-5)
            except ValueError:
                continue
            pts.append(x)
            if len(pts) >= n:
                break

        if len(pts) < n:
            raise RuntimeError(f"Could not sample enough feasible points ({len(pts)}/{n}).")
        return np.asarray(pts, dtype=float)


def make_sum_space(
    names: list[str],
    bounds: dict[str, tuple[float, float]],
    sum_equals: float,
    *,
    fixed: dict[str, float] | None = None,
    linear_deps: dict[str, dict[str, float]] | None = None,
) -> DesignSpace:
    """Convenience constructor for a constrained composition / flow space."""
    return DesignSpace(
        names=list(names),
        bounds=dict(bounds),
        sum_equals=float(sum_equals),
        fixed=dict(fixed or {}),
        linear_deps=dict(linear_deps or {}),
    )
