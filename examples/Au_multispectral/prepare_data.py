"""Prepare a ready-to-train CSV from station history (IHS UV fill + filters)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

FLOW = ["au", "ag", "peg", "ctab", "cit", "pvp", "ascorb", "teos"]
TARGETS = ["uv1", "uv2", "saxs1", "saxs2"]
OUTLIERS = {"day5_bayes1_1_cell", "day5_bayes2_10_cell", "day5_bayes4_4_cell"}


def prepare_station_training_csv(
    history_csv: str | Path,
    best_features_csv: str | Path,
    out_csv: str | Path,
) -> Path:
    df = pd.read_csv(history_csv)
    if "exclude_from_training" in df.columns:
        df = df[~df["exclude_from_training"].fillna(False).astype(bool)]
    df = df[~df["tag"].isin(OUTLIERS)]
    df = df[~df["tag"].astype(str).str.startswith("station_test_")]

    best = pd.read_csv(best_features_csv)
    uv_map = best[
        ["cell_id", "uv_ratio_best1_p1_460.0_p2_540.0", "uv_ratio_best2_p1_660.0_p2_760.0"]
    ].rename(
        columns={
            "uv_ratio_best1_p1_460.0_p2_540.0": "_uv1",
            "uv_ratio_best2_p1_660.0_p2_760.0": "_uv2",
        }
    )
    df = df.copy()
    df["_cell"] = df["tag"].map(
        lambda t: int(m.group(1)) if (m := re.match(r"^IHS_(\d+)_cell$", str(t))) else None
    )
    df = df.merge(uv_map, left_on="_cell", right_on="cell_id", how="left")
    ihs = df["dataset"].eq("IHS")
    df.loc[ihs & df["uv1"].isna(), "uv1"] = df.loc[ihs & df["uv1"].isna(), "_uv1"]
    df.loc[ihs & df["uv2"].isna(), "uv2"] = df.loc[ihs & df["uv2"].isna(), "_uv2"]
    df = df.drop(columns=["_cell", "cell_id", "_uv1", "_uv2"], errors="ignore")

    ok = df[FLOW + TARGETS].notna().all(axis=1)
    df = df.loc[ok].copy()

    def _rank(tag: str):
        tag = str(tag)
        if tag.startswith("IHS_"):
            m = re.match(r"IHS_(\d+)_cell", tag)
            return (0, 0, int(m.group(1)) if m else 0)
        m = re.match(r"day5_bayes(\d+)_(\d+)_cell", tag)
        if m:
            return (1, int(m.group(1)), int(m.group(2)))
        m = re.match(r"day6_bayes(\d+)_(\d+)_cell", tag)
        if m:
            return (2, int(m.group(1)), int(m.group(2)))
        return (3, 999, 0)

    df["_s"] = df["tag"].map(_rank)
    df = df.sort_values("_s").drop(columns="_s").reset_index(drop=True)

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["tag", "dataset", "iteration"] + FLOW + TARGETS
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(out, index=False)
    return out


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    path = prepare_station_training_csv(
        here / "data" / "experiment_history.csv",
        here / "data" / "best_features_with_flows.csv",
        here / "data" / "training_ready.csv",
    )
    print("wrote", path)
