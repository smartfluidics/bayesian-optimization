"""Load recipe CSVs into the DataFrame format expected by ``ScheduleRunner``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_schedule_dataframe(path: str | Path, *, n_syringes: int = 8) -> pd.DataFrame:
    """Normalize a recipe CSV to columns ``"0"..`` , ``"8"`` (time), optional ``index``.

    This matches what ``ScheduleRunner.load`` / ``run`` expect (unchanged runner API).
    """
    path = Path(path)
    raw = pd.read_csv(path, header=None, skipinitialspace=True)
    if raw.empty:
        raise ValueError(f"Empty schedule: {path}")

    body = raw
    # Drop header row like: 0,1,...,7,  or 1,..,8,time,index
    row0 = [str(v).strip().lower() for v in raw.iloc[0].tolist()]
    if _looks_like_header_row(row0, n_syringes):
        body = raw.iloc[1:].reset_index(drop=True)

    if body.shape[1] < n_syringes + 1:
        raise ValueError(
            f"Schedule needs >= {n_syringes + 1} columns (syringes + time), got {body.shape[1]}"
        )

    out = pd.DataFrame()
    for i in range(n_syringes):
        out[str(i)] = pd.to_numeric(body.iloc[:, i], errors="coerce")

    out["8"] = pd.to_numeric(body.iloc[:, n_syringes], errors="coerce")

    if body.shape[1] > n_syringes + 1:
        out["index"] = pd.to_numeric(body.iloc[:, n_syringes + 1], errors="coerce")
    else:
        out["index"] = pd.NA

    out = out.dropna(subset=[str(i) for i in range(n_syringes)] + ["8"]).reset_index(drop=True)
    return out


def _looks_like_header_row(row0: list[str], n_syringes: int) -> bool:
    if any(v in {"time", "index"} for v in row0):
        return True
    syringe_like = 0
    for v in row0[:n_syringes]:
        if v in {str(i) for i in range(n_syringes + 1)}:
            syringe_like += 1
    return syringe_like >= max(2, n_syringes // 2)


async def run_schedule_csv(
    runner,
    csv_path: str | Path,
    *,
    n_syringes: int = 8,
    start_idx: int = 0,
) -> None:
    """Load ``csv_path`` into ``runner`` and ``await runner.run()``."""
    sch = load_schedule_dataframe(csv_path, n_syringes=n_syringes)
    runner.load(sch)
    await runner.run(start_idx=start_idx)
