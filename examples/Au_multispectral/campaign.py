"""CSV-driven multitask GP active learning (epistemic uncertainty sampling)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from bayesian_optimization.space import DesignSpace
from bayesian_optimization.uncertainty import UncertaintySuggester, predict_long_table

from .eval_gp import loocv_report, train_fit_report


@dataclass
class GPHyperParams:
    """Hyperparameters for the multitask ICM GP + acquisition."""

    rank: int = 3
    training_iter: int = 200
    lr: float = 0.06
    n_candidates: int = 9000
    seed: int = 42
    acquisition: str = "max_variance"
    ucb_beta: float = 2.0
    batch_method: str = "farthest_point"


@dataclass
class SpaceSpec:
    """Optional constraints. If bounds is None, a padded box is taken from the CSV."""

    sum_equals: float | None = None
    fixed: dict[str, float] = field(default_factory=dict)
    linear_deps: dict[str, dict[str, float]] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] | None = None


class UncertaintyCSVCampaign:
    """Load CSV → feature/target columns → GP settings → suggest / LOOCV / plots."""

    def __init__(
        self,
        csv_path: str | Path,
        feature_cols: Sequence[str],
        target_cols: Sequence[str],
        *,
        gp: GPHyperParams | None = None,
        space_spec: SpaceSpec | None = None,
        id_col: str | None = "tag",
        dropna: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.gp = gp or GPHyperParams()
        self.space_spec = space_spec or SpaceSpec()
        self.id_col = id_col

        df = pd.read_csv(self.csv_path)
        missing = [c for c in self.feature_cols + self.target_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {self.csv_path}: {missing}")
        if dropna:
            df = df.dropna(subset=self.feature_cols + self.target_cols).reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No rows left after dropna")
        self.df = df
        self.space = self._build_space()
        self.last_suggestions: pd.DataFrame | None = None
        self.last_loocv: tuple[pd.DataFrame, pd.DataFrame] | None = None
        self.last_train_fit: tuple[pd.DataFrame, pd.DataFrame] | None = None

    def _build_space(self) -> DesignSpace:
        spec = self.space_spec
        if spec.bounds is None:
            bounds: dict[str, tuple[float, float]] = {}
            for c in self.feature_cols:
                if c in spec.fixed or c in spec.linear_deps:
                    continue
                lo = float(self.df[c].min())
                hi = float(self.df[c].max())
                if abs(hi - lo) < 1e-12:
                    hi = lo + 1.0
                pad = 0.05 * (hi - lo)
                bounds[c] = (lo - pad, hi + pad)
        else:
            bounds = dict(spec.bounds)
        return DesignSpace(
            names=list(self.feature_cols),
            bounds=bounds,
            sum_equals=spec.sum_equals,
            fixed=dict(spec.fixed),
            linear_deps=dict(spec.linear_deps),
        )

    def _suggester(self, *, n_candidates: int | None = None) -> UncertaintySuggester:
        g = self.gp
        return UncertaintySuggester(
            self.space,
            acquisition=g.acquisition,
            ucb_beta=g.ucb_beta,
            batch_method=g.batch_method,
            n_candidates=int(n_candidates if n_candidates is not None else g.n_candidates),
            rank=g.rank,
            training_iter=g.training_iter,
            lr=g.lr,
            seed=g.seed,
            output_names=list(self.target_cols),
        )

    @property
    def X(self) -> np.ndarray:
        return self.df[self.feature_cols].to_numpy(dtype=float)

    @property
    def Y(self) -> np.ndarray:
        return self.df[self.target_cols].to_numpy(dtype=float)

    def suggest(self, n_points: int = 1, *, n_candidates: int | None = None) -> pd.DataFrame:
        """Propose ``n_points`` that maximize total epistemic uncertainty."""
        sug = self._suggester(n_candidates=n_candidates)
        pts = sug.suggest(self.X, self.Y, n_points=int(n_points))
        out = pd.DataFrame(pts, columns=self.feature_cols)
        out.insert(0, "suggestion_no", range(1, len(out) + 1))
        if sug.last_fit is not None:
            long_pred = predict_long_table(sug.last_fit, pts)
            for col in self.target_cols:
                sub = long_pred.loc[long_pred["output"] == col].set_index("point").sort_index()
                out[f"{col}_pred_mean"] = sub["y_mean_from_f"].to_numpy()
                out[f"{col}_pred_std"] = sub["y_std_epistemic"].to_numpy()
            out["total_epistemic_uncertainty"] = (
                long_pred.groupby("point")["y_std_epistemic"].sum().to_numpy()
            )
        self.last_suggestions = out
        return out

    def train_fit(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        wide, summary = train_fit_report(
            self.df,
            self.feature_cols,
            self.target_cols,
            space=self.space,
            id_col=self.id_col,
            rank=self.gp.rank,
            training_iter=self.gp.training_iter,
            lr=self.gp.lr,
            seed=self.gp.seed,
        )
        self.last_train_fit = (wide, summary)
        return wide, summary

    def loocv(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        wide, summary = loocv_report(
            self.df,
            self.feature_cols,
            self.target_cols,
            space=self.space,
            id_col=self.id_col,
            rank=self.gp.rank,
            training_iter=self.gp.training_iter,
            lr=self.gp.lr,
            seed=self.gp.seed,
        )
        self.last_loocv = (wide, summary)
        return wide, summary

    def hyperparams_dict(self) -> dict:
        return {
            "csv": str(self.csv_path),
            "feature_cols": self.feature_cols,
            "target_cols": self.target_cols,
            "n_rows": len(self.df),
            "gp": asdict(self.gp),
            "space": {
                "sum_equals": self.space.sum_equals,
                "fixed": dict(self.space.fixed),
                "linear_deps": {k: dict(v) for k, v in self.space.linear_deps.items()},
            },
        }
