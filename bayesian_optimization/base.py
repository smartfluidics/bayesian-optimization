"""Shared protocol for Bayesian suggesters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from bayesian_optimization.space import DesignSpace


@runtime_checkable
class BayesianSuggester(Protocol):
    """Common API implemented by ScalarEISuggester and UncertaintySuggester."""

    space: DesignSpace

    def sample_initial(self, n_points: int, *, seed: int | None = None) -> np.ndarray:
        """Draw an initial feasible design (e.g. LHS / rejection)."""

    def suggest(self, X: np.ndarray, y: np.ndarray, n_points: int = 1) -> np.ndarray:
        """Propose next experiment(s).

        Parameters
        ----------
        X:
            Observed inputs, shape ``(n, n_dims)``.
        y:
            Observed targets. Scalar ProcessOptimizer expects ``(n,)``;
            uncertainty sampling expects ``(n, n_tasks)``.
        n_points:
            Batch size to suggest.
        """
