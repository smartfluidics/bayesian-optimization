"""Plots: LOOCV pred vs experiment, epistemic uncertainty replay curve."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesian_optimization.uncertainty import UncertaintySuggester, predict_long_table


def plot_loocv_pred_vs_meas(
    pred_df: pd.DataFrame,
    target_cols: list[str],
    out_path: str | Path,
    *,
    title: str = "GP LOOCV: prediction vs experiment",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(target_cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, tgt in zip(axes, target_cols):
        y = pred_df[tgt].to_numpy(dtype=float)
        p = pred_df[f"{tgt}_pred"].to_numpy(dtype=float)
        err = p - y
        mae = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
        lo = float(min(y.min(), p.min()))
        hi = float(max(y.max(), p.max()))
        pad = 0.05 * (hi - lo + 1e-12)
        ax.scatter(y, p, s=50, alpha=0.85, edgecolors="k", linewidths=0.3)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
        ax.set_xlabel(f"Measured {tgt}")
        ax.set_ylabel(f"LOOCV pred {tgt}")
        ax.set_title(f"{tgt}: R2={r2:.3f}, MAE={mae:.3f}")
        ax.grid(alpha=0.25)
    for ax in axes[len(target_cols) :]:
        ax.axis("off")
    fig.suptitle(f"{title} (n={len(pred_df)})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def replay_uncertainty_curve(
    campaign,
    *,
    min_train: int = 8,
    step: int = 2,
    n_candidates: int = 800,
    training_iter: int | None = None,
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Incremental train size → total epistemic uncertainty of the next suggestion."""
    df = campaign.df
    feature_cols = campaign.feature_cols
    target_cols = campaign.target_cols
    space = campaign.space
    training_iter = int(training_iter if training_iter is not None else max(40, campaign.gp.training_iter // 3))
    rows = []
    n = len(df)
    for k in range(min_train, n + 1, step):
        sub = df.iloc[:k]
        sug = UncertaintySuggester(
            space,
            n_candidates=n_candidates,
            training_iter=training_iter,
            seed=campaign.gp.seed,
            rank=campaign.gp.rank,
            lr=campaign.gp.lr,
            acquisition=campaign.gp.acquisition,
            output_names=list(target_cols),
        )
        pts = sug.suggest(
            sub[feature_cols].to_numpy(float),
            sub[target_cols].to_numpy(float),
            n_points=1,
        )
        lp = predict_long_table(sug.last_fit, pts)
        total = float(lp.groupby("point")["y_std_epistemic"].sum().iloc[0])
        last = sub.iloc[-1]
        tag = last[campaign.id_col] if campaign.id_col and campaign.id_col in sub.columns else k
        rows.append({"n_train": k, "last_id": tag, "total_epistemic_std": total})
        print(f"replay {k}/{n}: U={total:.3f}")
    summary = pd.DataFrame(rows)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_csv, index=False)
    return summary


def plot_uncertainty_curve(summary: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(summary["n_train"], summary["total_epistemic_std"], "-o", lw=1.8, ms=4, mfc="white")
    ax.set_xlabel("Training points (row order)")
    ax.set_ylabel("Epistemic uncertainty (sum std over targets)")
    ax.set_title("Uncertainty of next suggested point vs training size")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
