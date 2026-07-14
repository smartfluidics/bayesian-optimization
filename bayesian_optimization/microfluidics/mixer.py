"""Generic microfluidic CSV session over ProcessOptimizer ScalarEISuggester."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from bayesian_optimization.microfluidics import recipes
from bayesian_optimization.scalar_ei import ScalarEISuggester
from bayesian_optimization.space import DesignSpace


@dataclass(frozen=True)
class MixerConfig:
    """Experiment-agnostic mixer configuration.

    All reagent names, bounds, syringe map, and schedule timings are provided
    by the caller (see ``examples/`` for concrete experiment setups).
    """

    data_dir: str
    var_to_syringe: dict[str, int]
    bounds: dict[str, tuple[float, float]]
    n_syringes: int
    total_speed: float
    time_synth: float
    default_n_points: int = 10
    random_state: int = 0
    separator_syringe: int | None = None
    separator_speed: float = 0.0
    time_separator: float | None = None
    results_col: str = "Results"
    # ProcessOptimizer minimizes; True => scores are maximized via -y.
    maximize: bool = True
    # Alternating recipe/separator rows with 0-based syringe header.
    legacy_csv: bool = True


class MixerSession:
    """CSV in / CSV out Bayesian mixer session (generic schedule + optimization)."""

    def __init__(self, config: MixerConfig) -> None:
        self.cfg = config
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.var_to_syringe = dict(config.var_to_syringe)
        self.var_names = list(self.var_to_syringe.keys())
        self.n_syringes = int(config.n_syringes)
        self.total_speed = float(config.total_speed)
        self.time_synth = float(config.time_synth)
        self.results_col = config.results_col
        self._validate_config()

        self.space = DesignSpace(
            names=list(self.var_names),
            bounds={n: config.bounds[n] for n in self.var_names},
            sum_equals=self.total_speed,
        )
        self.suggester = ScalarEISuggester(
            self.space,
            random_state=config.random_state,
            maximize=config.maximize,
        )

    def _validate_config(self) -> None:
        if not self.var_names:
            raise ValueError("var_to_syringe is empty.")
        if self.n_syringes <= 0:
            raise ValueError("n_syringes must be > 0.")
        if self.total_speed <= 0:
            raise ValueError("total_speed must be > 0.")
        if len(set(self.var_to_syringe.values())) != len(self.var_to_syringe):
            raise ValueError("var_to_syringe has duplicate syringe indices.")
        for name, idx in self.var_to_syringe.items():
            if idx < 0 or idx >= self.n_syringes:
                raise ValueError(f"{name}: syringe index {idx} is out of range.")
            if name not in self.cfg.bounds:
                raise ValueError(f"{name}: missing bounds")
            lo, hi = self.cfg.bounds[name]
            if lo >= hi:
                raise ValueError("Each bound must satisfy low < high.")
        if self.cfg.separator_syringe is None or self.cfg.time_separator is None:
            raise ValueError("separator_syringe and time_separator are required.")
        if self.cfg.separator_syringe < 0 or self.cfg.separator_syringe >= self.n_syringes:
            raise ValueError("separator_syringe is out of range.")

    def sample_initial(self, n_points: int, *, seed: int | None = None) -> np.ndarray:
        return self.suggester.sample_initial(n_points, seed=seed)

    def suggest(self, X: np.ndarray, y: np.ndarray, n_points: int = 1) -> np.ndarray:
        return self.suggester.suggest(X, y, n_points=n_points)

    def _write_schedule(self, X_new: np.ndarray, out_file: str | Path) -> str:
        out_path = Path(out_file)
        if not out_path.is_absolute():
            out_path = self.data_dir / out_path
        if self.cfg.legacy_csv:
            return recipes.write_schedule_legacy(
                X_new,
                out_path,
                var_names=self.var_names,
                var_to_syringe=self.var_to_syringe,
                n_syringes=self.n_syringes,
                total_speed=self.total_speed,
                time_synth=self.time_synth,
                separator_syringe=int(self.cfg.separator_syringe),
                separator_speed=float(self.cfg.separator_speed),
                time_separator=float(self.cfg.time_separator),
            )
        return recipes.write_schedule(
            X_new,
            out_path,
            var_names=self.var_names,
            var_to_syringe=self.var_to_syringe,
            n_syringes=self.n_syringes,
            total_speed=self.total_speed,
            time_synth=self.time_synth,
            include_separator_rows=True,
            separator_syringe=self.cfg.separator_syringe,
            separator_speed=self.cfg.separator_speed,
            time_separator=self.cfg.time_separator,
        )

    def generate_lhs_iter0(
        self,
        n_points: int | None = None,
        out_file: str = "recipes_iter_000.csv",
    ) -> str:
        n_points = int(n_points or self.cfg.default_n_points)
        X_new = self.sample_initial(n_points, seed=self.cfg.random_state)
        return self._write_schedule(X_new, out_file=out_file)

    def suggest_from_datasets(
        self,
        dataset_files: Iterable[str | Path],
        n_points: int | None = None,
        out_file: str = "recipes_next.csv",
        result_col: str = "result",
    ) -> str:
        n_points = int(n_points or self.cfg.default_n_points)
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        for fp in dataset_files:
            X, y = recipes.extract_xy_from_dataset(
                fp,
                var_names=self.var_names,
                var_to_syringe=self.var_to_syringe,
                n_syringes=self.n_syringes,
                result_col=result_col,
            )
            all_x.append(X)
            all_y.append(y)
        X_arr = np.vstack(all_x)
        y_arr = np.concatenate(all_y)
        X_new = self.suggest(X_arr, y_arr, n_points=n_points)
        return self._write_schedule(X_new, out_file=out_file)

    def suggest_next(self, iter_idx: int, n_points: int | None = None) -> str:
        """Ingest ``recipes_iter_000..iter_idx`` results CSVs and write the next schedule."""
        n_points = int(n_points or self.cfg.default_n_points)
        all_x: list[list[float]] = []
        all_y: list[float] = []

        for k in range(int(iter_idx)):
            fname_res = self.data_dir / f"recipes_iter_{k:03d}_results.csv"
            if not fname_res.exists():
                raise RuntimeError(f"Missing {fname_res} for history iteration {k}")
            X_prev, y_prev = recipes.extract_xy_legacy_alt_rows(
                fname_res,
                var_names=self.var_names,
                var_to_syringe=self.var_to_syringe,
                n_syringes=self.n_syringes,
            )
            all_x.extend(X_prev.tolist())
            all_y.extend(y_prev.tolist())

        fname_res_cur = self.data_dir / f"recipes_iter_{int(iter_idx):03d}_results.csv"
        target_iter = int(iter_idx)
        if fname_res_cur.exists():
            X_cur, y_cur = recipes.extract_xy_legacy_alt_rows(
                fname_res_cur,
                var_names=self.var_names,
                var_to_syringe=self.var_to_syringe,
                n_syringes=self.n_syringes,
            )
            all_x.extend(X_cur.tolist())
            all_y.extend(y_cur.tolist())
            target_iter = int(iter_idx) + 1

        if not all_x:
            raise RuntimeError("No results loaded; cannot suggest_next.")

        X_new = self.suggest(
            np.asarray(all_x, dtype=float),
            np.asarray(all_y, dtype=float),
            n_points=n_points,
        )
        out_file = f"recipes_iter_{target_iter:03d}.csv"
        return self._write_schedule(X_new, out_file=out_file)

    def collect_all_results(self, max_iter: int = 20, results_col: str | None = None):
        results_col = results_col or self.results_col
        all_rows = []
        for k in range(max_iter):
            fname_res = self.data_dir / f"recipes_iter_{k:03d}_results.csv"
            if not fname_res.exists():
                continue
            try:
                X, y = recipes.extract_xy_legacy_alt_rows(
                    fname_res,
                    var_names=self.var_names,
                    var_to_syringe=self.var_to_syringe,
                    n_syringes=self.n_syringes,
                )
            except Exception:
                continue
            vars_df = pd.DataFrame(X, columns=self.var_names)
            vars_df[results_col] = y
            vars_df["iter"] = k
            all_rows.append(vars_df)

        if not all_rows:
            return None
        big_df = pd.concat(all_rows, ignore_index=True)
        big_fname = self.data_dir / "all_results.csv"
        big_df.to_csv(big_fname, index=False)
        return str(big_fname)


def space_from_mixer_config(config: MixerConfig) -> DesignSpace:
    return DesignSpace(
        names=list(config.var_to_syringe.keys()),
        bounds={n: config.bounds[n] for n in config.var_to_syringe},
        sum_equals=float(config.total_speed),
    )
