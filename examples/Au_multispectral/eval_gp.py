"""GP train-fit and LOOCV utilities (prediction vs measured)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from bayesian_optimization.space import DesignSpace
from bayesian_optimization.uncertainty import UncertaintySuggester, predict_long_table


def _fit_state(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    space: DesignSpace,
    *,
    rank: int,
    training_iter: int,
    lr: float,
    seed: int,
):
    sug = UncertaintySuggester(
        space,
        n_candidates=8,
        rank=rank,
        training_iter=training_iter,
        lr=lr,
        seed=seed,
        output_names=list(target_cols),
    )
    X = df[list(feature_cols)].to_numpy(dtype=float)
    Y = df[list(target_cols)].to_numpy(dtype=float)
    return sug._train_gp(X, Y, list(target_cols))


def _predict_means(state, x: np.ndarray) -> np.ndarray:
    long_pred = predict_long_table(state, x)
    cols = state.output_cols
    n = x.shape[0]
    out = np.zeros((n, len(cols)), dtype=float)
    for j, c in enumerate(cols):
        out[:, j] = (
            long_pred.loc[long_pred["output"] == c]
            .set_index("point")
            .sort_index()["y_mean_from_f"]
            .to_numpy()
        )
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, target_cols: Sequence[str], kind: str) -> pd.DataFrame:
    rows = []
    for j, oc in enumerate(target_cols):
        y = y_true[:, j]
        p = y_pred[:, j]
        err = p - y
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
        if kind == "train":
            rows.append({"target": oc, "n": len(y), "mae": float(np.mean(np.abs(err))),
                         "rmse": float(np.sqrt(np.mean(err**2))), "r2_train": r2})
        else:
            rows.append({"target": oc, "n": len(y), "mae_loocv": float(np.mean(np.abs(err))),
                         "rmse_loocv": float(np.sqrt(np.mean(err**2))), "r2_loocv": r2})
    return pd.DataFrame(rows)


def train_fit_report(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    *,
    space: DesignSpace,
    id_col: str | None = "tag",
    rank: int = 3,
    training_iter: int = 200,
    lr: float = 0.06,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = _fit_state(
        df, feature_cols, target_cols, space, rank=rank, training_iter=training_iter, lr=lr, seed=seed
    )
    X = df[list(feature_cols)].to_numpy(dtype=float)
    Y = df[list(target_cols)].to_numpy(dtype=float)
    pred = _predict_means(state, X)

    out = pd.DataFrame()
    if id_col and id_col in df.columns:
        out[id_col] = df[id_col].to_numpy()
    for j, oc in enumerate(target_cols):
        out[oc] = Y[:, j]
        out[f"{oc}_pred"] = pred[:, j]
        out[f"{oc}_err"] = out[f"{oc}_pred"] - out[oc]
    return out, _metrics(Y, pred, target_cols, "train")


def loocv_report(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    *,
    space: DesignSpace,
    id_col: str | None = "tag",
    rank: int = 3,
    training_iter: int = 200,
    lr: float = 0.06,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    records: list[dict] = []
    Y_all = df[list(target_cols)].to_numpy(dtype=float)
    P_all = np.zeros_like(Y_all)
    index = list(df.index)

    for i, idx in enumerate(index):
        fold = df.drop(index=idx).reset_index(drop=True)
        hold = df.loc[[idx]].reset_index(drop=True)
        state = _fit_state(
            fold, feature_cols, target_cols, space, rank=rank, training_iter=training_iter, lr=lr, seed=seed
        )
        y_pred = _predict_means(state, hold[list(feature_cols)].to_numpy(dtype=float))[0]
        P_all[i] = y_pred
        rec: dict = {}
        if id_col and id_col in hold.columns:
            rec[id_col] = hold[id_col].iloc[0]
        for j, oc in enumerate(target_cols):
            rec[oc] = float(hold[oc].iloc[0])
            rec[f"{oc}_pred"] = float(y_pred[j])
            rec[f"{oc}_err"] = rec[f"{oc}_pred"] - rec[oc]
        records.append(rec)
        print(f"LOOCV {i + 1}/{n}: {rec.get(id_col, i)}")

    return pd.DataFrame(records), _metrics(Y_all, P_all, target_cols, "loocv")
