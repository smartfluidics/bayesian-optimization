"""Unified Bayesian suggestion library for smartfluidics experiments."""

from bayesian_optimization.base import BayesianSuggester
from bayesian_optimization.scalar_ei import ProcessOptimizerSuggester, ScalarEISuggester
from bayesian_optimization.space import DesignSpace, make_sum_space

__all__ = [
    "BayesianSuggester",
    "DesignSpace",
    "ProcessOptimizerSuggester",
    "ScalarEISuggester",
    "make_sum_space",
    "UncertaintySuggester",
]


def __getattr__(name: str):
    if name == "UncertaintySuggester":
        from bayesian_optimization.uncertainty import UncertaintySuggester

        return UncertaintySuggester
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
