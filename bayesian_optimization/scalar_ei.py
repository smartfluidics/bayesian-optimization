"""Scalar Bayesian optimization via ProcessOptimizer (GP + EI + SumEquals)."""

from __future__ import annotations

import numpy as np

from bayesian_optimization.space import DesignSpace

try:
    import ProcessOptimizer as po
    from ProcessOptimizer.space.constraints import Constraints, SumEquals

    _HAS_PO = True
except ImportError:  # pragma: no cover
    po = None  # type: ignore
    Constraints = None  # type: ignore
    SumEquals = None  # type: ignore
    _HAS_PO = False


def _require_process_optimizer() -> None:
    if not _HAS_PO:
        raise ImportError(
            "ScalarEISuggester requires ProcessOptimizer. "
            "Install with: pip install ProcessOptimizer"
        )


class ScalarEISuggester:
    """Optimize a scalar objective with ProcessOptimizer.

    Typical settings:

    - ``base_estimator="GP"``
    - ``acq_func="EI"``
    - ``n_initial_points=0``
    - batch ask with ``strategy="cl_min"``
    - ``SumEquals`` when ``DesignSpace.sum_equals`` is set and there are no
      ``fixed`` / ``linear_deps``

    Spaces with ``fixed`` / ``linear_deps`` are supported by optimizing the
    free variables in a box and projecting each suggestion onto the full
    :class:`DesignSpace` (same station constraints as ``UncertaintySuggester``).
    """

    def __init__(
        self,
        space: DesignSpace,
        *,
        random_state: int = 0,
        maximize: bool = True,
        batch_strategy: str = "cl_min",
        base_estimator: str = "GP",
        acq_func: str = "EI",
        acq_optimizer: str = "auto",
    ) -> None:
        _require_process_optimizer()
        free = space.free_names()
        if not free:
            raise ValueError("ScalarEISuggester needs at least one free variable")
        self.space = space
        self._free_names = free
        self._constrained = bool(space.fixed or space.linear_deps)
        self.random_state = int(random_state)
        self.maximize = bool(maximize)
        self.batch_strategy = batch_strategy
        self.base_estimator = base_estimator
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer

    def sample_initial(self, n_points: int, *, seed: int | None = None) -> np.ndarray:
        return self.space.sample_feasible(
            int(n_points),
            seed=self.random_state if seed is None else int(seed),
            method="auto",
        )

    def suggest(self, X: np.ndarray, y: np.ndarray, n_points: int = 1) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must have shape (n, n_dims)")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y length mismatch")
        if X_arr.shape[1] != self.space.n_dims:
            raise ValueError(f"X must have {self.space.n_dims} columns")
        if n_points <= 0:
            return np.zeros((0, self.space.n_dims), dtype=float)

        X_arr, y_arr = self._deduplicate(X_arr, y_arr)
        X_free = self._to_free(X_arr)
        opt = self._make_optimizer()
        sign = -1.0 if self.maximize else 1.0
        for x_i, y_i in zip(X_free, y_arr):
            opt.tell(list(map(float, x_i)), float(sign * y_i))

        asked = opt.ask(n_points=int(n_points), strategy=self.batch_strategy)
        if int(n_points) == 1 and not isinstance(asked[0], (list, tuple, np.ndarray)):
            asked = [asked]
        out = np.asarray([self._from_free(x) for x in asked], dtype=float)
        return out

    def _free_index(self, name: str) -> int:
        return self._free_names.index(name)

    def _to_free(self, X: np.ndarray) -> np.ndarray:
        idx = [self.space.index(n) for n in self._free_names]
        return np.asarray(X[:, idx], dtype=float)

    def _from_free(self, x_free) -> np.ndarray:
        values = {n: 0.0 for n in self.space.names}
        for name, val in zip(self._free_names, np.asarray(x_free, dtype=float).reshape(-1)):
            values[name] = float(val)
        values = self.space._apply_fixed_and_deps(values)
        return self.space.project_to_feasible(self.space.to_array(values))

    def _make_optimizer(self):
        bounds = [
            (float(self.space.bounds[n][0]), float(self.space.bounds[n][1]))
            for n in self._free_names
        ]
        po_space = po.Space(bounds)
        opt = po.Optimizer(
            po_space,
            base_estimator=self.base_estimator,
            n_initial_points=0,
            acq_func=self.acq_func,
            acq_optimizer=self.acq_optimizer,
            lhs=False,
            random_state=self.random_state,
        )
        # SumEquals on free dims only when there is no chemistry fixed/deps rewrite.
        if self.space.sum_equals is not None and not self._constrained:
            constraints = Constraints(
                [SumEquals(list(range(len(self._free_names))), float(self.space.sum_equals))],
                po_space,
            )
            opt.set_constraints(constraints)
        return opt

    @staticmethod
    def _deduplicate(X: np.ndarray, y: np.ndarray, decimals: int = 8) -> tuple[np.ndarray, np.ndarray]:
        keys = np.round(X, decimals=decimals)
        _, keep_idx = np.unique(keys, axis=0, return_index=True)
        keep_idx = np.sort(keep_idx)
        return X[keep_idx], y[keep_idx]


ProcessOptimizerSuggester = ScalarEISuggester
