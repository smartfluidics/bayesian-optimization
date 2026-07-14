"""
Demo for Au_multispectral:

  1) prepare training CSV from station history
  2) UncertaintyCSVCampaign: suggest n points
  3) LOOCV + plot prediction vs experiment
  4) uncertainty decay curve plot

Run from repo root:
  python -m examples.Au_multispectral.demo_run
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.Au_multispectral.campaign import GPHyperParams, SpaceSpec, UncertaintyCSVCampaign
from examples.Au_multispectral.plots import (
    plot_loocv_pred_vs_meas,
    plot_uncertainty_curve,
    replay_uncertainty_curve,
)
from examples.Au_multispectral.prepare_data import FLOW, TARGETS, prepare_station_training_csv


def main() -> None:
    data = HERE / "data"
    plots = HERE / "plots"
    plots.mkdir(exist_ok=True)

    ready = prepare_station_training_csv(
        data / "experiment_history.csv",
        data / "best_features_with_flows.csv",
        data / "training_ready.csv",
    )
    print("training CSV:", ready)

    # Station chemistry constraints (example-specific; the campaign API is generic)
    space = SpaceSpec(
        sum_equals=20.0,
        fixed={"ag": 0.0, "teos": 0.0},
        linear_deps={"ascorb": {"au": 1.2, "ag": 1.2}},
        bounds={
            "au": (3.3, 8.5),
            "peg": (1e-3, 19.999),
            "ctab": (1e-3, 19.999),
            "cit": (1e-3, 19.999),
            "pvp": (1e-3, 19.999),
        },
    )
    gp = GPHyperParams(
        rank=3,
        training_iter=80,  # lighter for demo; raise for production
        n_candidates=1500,
        seed=42,
        acquisition="max_variance",
    )

    camp = UncertaintyCSVCampaign(
        ready,
        feature_cols=FLOW,
        target_cols=TARGETS,
        gp=gp,
        space_spec=space,
        id_col="tag",
    )
    print("hyperparams:", camp.hyperparams_dict())

    # --- suggest next points (uncertainty reduction / exploration) ---
    nxt = camp.suggest(n_points=1)
    nxt.to_csv(data / "next_suggestion.csv", index=False)
    print("\n=== Next suggestion ===")
    print(nxt.to_string(index=False))

    # --- train fit ---
    _, fit_sum = camp.train_fit()
    fit_sum.to_csv(data / "gp_train_fit_summary.csv", index=False)
    print("\n=== Train fit ===")
    print(fit_sum.to_string(index=False))

    # --- LOOCV + pred vs experiment plot ---
    loocv_df, loocv_sum = camp.loocv()
    loocv_df.to_csv(data / "gp_loocv_predictions.csv", index=False)
    loocv_sum.to_csv(data / "gp_loocv_summary.csv", index=False)
    print("\n=== LOOCV ===")
    print(loocv_sum.to_string(index=False))
    p1 = plot_loocv_pred_vs_meas(
        loocv_df,
        TARGETS,
        plots / "loocv_gp_pred_vs_meas.png",
        title="GP LOOCV: prediction vs experiment",
    )
    print("saved", p1)

    # --- uncertainty decay curve ---
    summary = replay_uncertainty_curve(
        camp,
        min_train=8,
        step=3,
        n_candidates=600,
        training_iter=50,
        out_csv=data / "epistemic_uncertainty_replay.csv",
    )
    p2 = plot_uncertainty_curve(summary, plots / "epistemic_uncertainty_curve.png")
    print("saved", p2)


if __name__ == "__main__":
    main()
