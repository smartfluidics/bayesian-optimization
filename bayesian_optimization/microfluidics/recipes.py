"""Recipe CSV helpers for microfluidic schedules."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda s: pd.to_numeric(s.astype(str).str.strip(), errors="coerce"))


def resolve_syringe_columns(cols: set[str], n_syringes: int) -> list[str] | None:
    """Prefer 0-based headers ``0..n-1``, else 1-based ``1..n``."""
    zero_based = [str(i) for i in range(n_syringes)]
    one_based = [str(i + 1) for i in range(n_syringes)]
    if all(c in cols for c in zero_based):
        return zero_based
    if all(c in cols for c in one_based):
        return one_based
    return None


def extract_xy_from_dataset(
    file_path: str | Path,
    *,
    var_names: Sequence[str],
    var_to_syringe: dict[str, int],
    n_syringes: int,
    result_col: str = "result",
) -> tuple[np.ndarray, np.ndarray]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    raw = pd.read_csv(file_path, skipinitialspace=True)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    raw_num = clean_numeric(raw)

    candidate_cols = [result_col.lower(), "result", "results", "score", "target", "y"]
    y_col = next((c for c in candidate_cols if c in raw_num.columns), None)
    syringe_cols = resolve_syringe_columns(set(raw_num.columns), n_syringes)

    if y_col is None or syringe_cols is None:
        return extract_xy_legacy_alt_rows(
            file_path,
            var_names=var_names,
            var_to_syringe=var_to_syringe,
            n_syringes=n_syringes,
        )

    y_series = raw_num[y_col]
    speeds = raw_num[syringe_cols]
    valid_mask = y_series.notna() & speeds.notna().all(axis=1)
    speeds_valid = speeds.loc[valid_mask].reset_index(drop=True)
    y_valid = y_series.loc[valid_mask].astype(float).to_numpy()
    if len(y_valid) == 0:
        raise ValueError(f"No valid (X, y) rows were found in {file_path}.")

    X = []
    for _, row in speeds_valid.iterrows():
        point = [float(row.iloc[var_to_syringe[name]]) for name in var_names]
        X.append(point)
    return np.asarray(X, dtype=float), y_valid


def extract_xy_legacy_alt_rows(
    file_path: str | Path,
    *,
    var_names: Sequence[str],
    var_to_syringe: dict[str, int],
    n_syringes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load results CSV: header + alternating recipe/separator rows; score in last column."""
    raw = pd.read_csv(file_path, header=None)
    rows_recipes = raw.iloc[1::2].reset_index(drop=True)
    speeds = rows_recipes.iloc[:, :n_syringes]
    y = pd.to_numeric(rows_recipes.iloc[:, -1], errors="coerce")
    valid = y.notna() & speeds.notna().all(axis=1)
    speeds = speeds.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(float).to_numpy()
    if len(y) == 0:
        raise ValueError(f"No valid legacy recipe rows in {file_path}")

    X = []
    for _, row in speeds.iterrows():
        X.append([float(row.iloc[var_to_syringe[name]]) for name in var_names])
    return np.asarray(X, dtype=float), y


def write_schedule_legacy(
    X_new: np.ndarray,
    out_path: str | Path,
    *,
    var_names: Sequence[str],
    var_to_syringe: dict[str, int],
    n_syringes: int,
    total_speed: float,
    time_synth: float,
    separator_syringe: int,
    separator_speed: float,
    time_separator: float,
) -> str:
    """Write schedule CSV: 0-based syringe header, alternating recipe/separator rows."""
    rows: list[list[float | int | str]] = []
    rows.append(list(range(n_syringes)) + [""])
    for x in X_new:
        if not np.isclose(float(np.sum(x)), total_speed, atol=1e-4):
            raise ValueError(f"Flow sum must be {total_speed}, got {float(np.sum(x))}")
        row1 = to_full_syringe_row(
            x,
            var_names=var_names,
            var_to_syringe=var_to_syringe,
            n_syringes=n_syringes,
        ) + [time_synth]
        sep = [0.0] * n_syringes
        sep[separator_syringe] = float(separator_speed)
        row2 = sep + [float(time_separator)]
        rows.append(row1)
        rows.append(row2)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, header=False)
    return str(out)

def to_full_syringe_row(
    x_point: Sequence[float],
    *,
    var_names: Sequence[str],
    var_to_syringe: dict[str, int],
    n_syringes: int,
) -> list[float]:
    row = [0.0] * n_syringes
    for val, name in zip(x_point, var_names):
        row[var_to_syringe[name]] = float(val)
    return row


def write_schedule(
    X_new: np.ndarray,
    out_path: str | Path,
    *,
    var_names: Sequence[str],
    var_to_syringe: dict[str, int],
    n_syringes: int,
    total_speed: float,
    time_synth: float,
    start_index: int = 4,
    include_separator_rows: bool = True,
    drain_idx: int = 1,
    separator_syringe: int | None = None,
    separator_speed: float = 0.0,
    time_separator: float | None = None,
) -> str:
    rows: list[list[float | int | str]] = []
    rows.append(list(range(1, n_syringes + 1)) + ["time", "index"])

    idx_counter = int(start_index)
    for x in X_new:
        if not np.isclose(float(np.sum(x)), total_speed, atol=1e-4):
            raise ValueError(f"Flow sum must be {total_speed}, got {float(np.sum(x))}")
        rows.append(
            to_full_syringe_row(
                x,
                var_names=var_names,
                var_to_syringe=var_to_syringe,
                n_syringes=n_syringes,
            )
            + [time_synth, idx_counter]
        )
        if include_separator_rows:
            if separator_syringe is None or time_separator is None:
                raise ValueError("Separator settings are required when include_separator_rows=True.")
            sep = [0.0] * n_syringes
            sep[separator_syringe] = float(separator_speed)
            rows.append(sep + [float(time_separator), int(drain_idx)])
        idx_counter += 1

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, header=False)
    return str(out)
