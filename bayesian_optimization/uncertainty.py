"""Multitask GP uncertainty sampling (epistemic variance maximization)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from bayesian_optimization.space import DesignSpace

try:
    import gpytorch
    import torch

    torch.set_default_dtype(torch.double)
    _HAS_UNCERTAINTY = True
except ImportError:  # pragma: no cover
    gpytorch = None  # type: ignore
    torch = None  # type: ignore
    _HAS_UNCERTAINTY = False


def _require_uncertainty() -> None:
    if not _HAS_UNCERTAINTY:
        raise ImportError(
            "UncertaintySuggester requires torch and gpytorch. "
            "Install with: pip install 'bayesian-optimization[uncertainty]'"
        )


if _HAS_UNCERTAINTY:

    class MultitaskICMGP(gpytorch.models.ExactGP):
        """ICM multitask GP (default rank=3)."""

        def __init__(self, train_x, train_y, likelihood, n_tasks: int, rank: int = 3):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=n_tasks
            )
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                base_kernel, num_tasks=n_tasks, rank=rank
            )

        def forward(self, x):
            return gpytorch.distributions.MultitaskMultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

else:  # pragma: no cover

    class MultitaskICMGP:  # type: ignore[no-redef]
        pass


@dataclass
class GPFitState:
    model: object
    likelihood: object
    x_scaler: Dict[str, np.ndarray]
    y_mean: np.ndarray
    y_std: np.ndarray
    output_cols: List[str]


def _fit_x_scaler(x: np.ndarray) -> Dict[str, np.ndarray]:
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    x_range = np.where(x_max - x_min < 1e-12, 1.0, x_max - x_min)
    return {"min": x_min, "range": x_range}


def _normalize_x(x: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return (x - scaler["min"]) / scaler["range"]


def _epistemic_std_sum(model: MultitaskICMGP, x_cand_n: np.ndarray, n_tasks: int) -> np.ndarray:
    n = x_cand_n.shape[0]
    out = np.zeros(n, dtype=float)
    xt = torch.tensor(x_cand_n, dtype=torch.double)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(n):
            f_post = model(xt[i : i + 1])
            cov = f_post.covariance_matrix.squeeze().cpu().numpy()
            var = np.maximum(np.diag(cov), 0.0)
            out[i] = float(np.sum(np.sqrt(var[:n_tasks])))
    return out


def _pick_farthest_point(
    x_norm: np.ndarray,
    scores: np.ndarray,
    k: int,
    pool_top: int = 150,
) -> np.ndarray:
    order = np.argsort(-scores)
    pool = order[: min(pool_top, len(order))]
    if k >= len(pool):
        return pool[:k]
    chosen = [int(pool[0])]
    while len(chosen) < k:
        rest = np.array([i for i in pool if i not in chosen], dtype=int)
        if rest.size == 0:
            break
        d2 = ((x_norm[rest][:, None, :] - x_norm[chosen][None, :, :]) ** 2).sum(axis=-1)
        min_d = np.sqrt(d2.min(axis=1))
        chosen.append(int(rest[int(np.argmax(min_d))]))
    return np.array(chosen, dtype=int)


class UncertaintySuggester:
    """Suggest compositions that maximize total epistemic uncertainty.

    Multitask ICM GP with ``max_variance`` (sum of epistemic std) or ``ucb``
    acquisition on a configurable :class:`DesignSpace`.
    """

    def __init__(
        self,
        space: DesignSpace,
        *,
        acquisition: str = "max_variance",
        ucb_beta: float = 2.0,
        batch_method: str = "farthest_point",
        n_candidates: int = 9000,
        rank: int = 3,
        training_iter: int = 200,
        lr: float = 0.06,
        seed: int = 42,
        output_names: List[str] | None = None,
    ) -> None:
        _require_uncertainty()
        if acquisition not in ("max_variance", "ucb"):
            raise ValueError(f"unknown acquisition: {acquisition}")
        if batch_method not in ("top", "farthest_point"):
            raise ValueError(f"unknown batch_method: {batch_method}")
        self.space = space
        self.acquisition = acquisition
        self.ucb_beta = float(ucb_beta)
        self.batch_method = batch_method
        self.n_candidates = int(n_candidates)
        self.rank = int(rank)
        self.training_iter = int(training_iter)
        self.lr = float(lr)
        self.seed = int(seed)
        self.output_names = list(output_names) if output_names is not None else []
        self.last_fit: Optional[GPFitState] = None
        self.last_candidate_table: Optional[object] = None

    def sample_initial(self, n_points: int, *, seed: int | None = None) -> np.ndarray:
        return self.space.sample_feasible(
            int(n_points),
            seed=self.seed if seed is None else int(seed),
            method="auto",
        )

    def suggest(self, X: np.ndarray, y: np.ndarray, n_points: int = 1) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must have shape (n, n_dims)")
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.ndim != 2:
            raise ValueError("y must have shape (n,) or (n, n_tasks)")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y length mismatch")
        if X_arr.shape[1] != self.space.n_dims:
            raise ValueError(f"X must have {self.space.n_dims} columns")
        if np.isnan(X_arr).any() or np.isnan(y_arr).any():
            raise ValueError("training arrays contain NaN")

        n_tasks = y_arr.shape[1]
        output_cols = self.output_names or [f"y{i}" for i in range(n_tasks)]
        if len(output_cols) != n_tasks:
            raise ValueError("output_names length must match y columns")

        state = self._train_gp(X_arr, y_arr, output_cols)
        self.last_fit = state

        x_train_n = _normalize_x(X_arr, state.x_scaler)
        x_cand = self.space.sample_feasible(self.n_candidates, seed=self.seed, method="auto")
        for i in range(len(x_cand)):
            self.space.check_feasible(x_cand[i], tol=1e-5)
        x_cand_n = _normalize_x(x_cand, state.x_scaler)

        if self.acquisition == "max_variance":
            scores = _epistemic_std_sum(state.model, x_cand_n, n_tasks)
        else:
            scores = self.ucb_beta * _epistemic_std_sum(state.model, x_cand_n, n_tasks)

        if n_points <= 1 or self.batch_method == "top":
            pick_idx = np.argsort(-scores)[:n_points]
        else:
            pick_idx = _pick_farthest_point(x_cand_n, scores, n_points)

        next_points = x_cand[pick_idx]
        for row in next_points:
            self.space.check_feasible(row, tol=1e-5)

        import pandas as pd

        cand = pd.DataFrame(x_cand, columns=self.space.names)
        cand["uncertainty_score"] = scores
        cand["min_dist_to_train"] = np.sqrt(
            ((x_cand_n[:, None, :] - x_train_n[None, :, :]) ** 2).sum(axis=-1)
        ).min(axis=1)
        picked = np.zeros(len(cand), dtype=bool)
        picked[pick_idx] = True
        cand["picked"] = picked
        self.last_candidate_table = cand.sort_values("uncertainty_score", ascending=False).reset_index(
            drop=True
        )
        return np.asarray(next_points, dtype=float)

    def _train_gp(self, x: np.ndarray, y: np.ndarray, output_cols: List[str]) -> GPFitState:
        x_scaler = _fit_x_scaler(x)
        xn = _normalize_x(x, x_scaler)
        y_mean = y.mean(axis=0)
        y_std = np.where(y.std(axis=0, ddof=0) < 1e-12, 1.0, y.std(axis=0, ddof=0))
        yz = (y - y_mean) / y_std

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        train_x = torch.tensor(xn, dtype=torch.double)
        train_y = torch.tensor(yz, dtype=torch.double)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(output_cols))
        model = MultitaskICMGP(
            train_x, train_y, likelihood, n_tasks=len(output_cols), rank=self.rank
        )

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(self.training_iter):
            optimizer.zero_grad()
            out = model(train_x)
            loss = -mll(out, train_y)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        return GPFitState(
            model=model,
            likelihood=likelihood,
            x_scaler=x_scaler,
            y_mean=y_mean,
            y_std=y_std,
            output_cols=list(output_cols),
        )


def predict_long_table(state: GPFitState, x: np.ndarray) -> "object":
    """Per-(point, output) epistemic prediction table."""
    _require_uncertainty()
    import pandas as pd

    xn = _normalize_x(np.asarray(x, dtype=float), state.x_scaler)
    xt = torch.tensor(xn, dtype=torch.double)
    state.model.eval()
    state.likelihood.eval()
    rows: List[dict] = []
    for i in range(len(x)):
        sl = xt[i : i + 1]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_post = state.model(sl)
            y_post = state.likelihood(state.model(sl))
        fz_mean = f_post.mean.squeeze(0).cpu().numpy()
        f_cov = f_post.covariance_matrix.squeeze().cpu().numpy()
        fz_std = np.sqrt(np.maximum(np.diag(f_cov), 0.0))
        yz_mean = y_post.mean.squeeze(0).cpu().numpy()
        y_cov = y_post.covariance_matrix.squeeze().cpu().numpy()
        yz_std = np.sqrt(np.maximum(np.diag(y_cov), 0.0))
        sign, ld = np.linalg.slogdet(f_cov + 1e-8 * np.eye(len(state.output_cols)))
        logdet = float(ld) if sign > 0 else float("-inf")
        for j, cname in enumerate(state.output_cols):
            rows.append(
                {
                    "point": i,
                    "output": cname,
                    "f_mean_z": fz_mean[j],
                    "f_std_z_epistemic": fz_std[j],
                    "y_mean_from_f": state.y_mean[j] + state.y_std[j] * fz_mean[j],
                    "y_std_epistemic": state.y_std[j] * fz_std[j],
                    "y_pred_obs_mean": state.y_mean[j] + state.y_std[j] * yz_mean[j],
                    "y_pred_obs_std_total": state.y_std[j] * yz_std[j],
                    "logdet_cov_f_z": logdet if j == 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)
