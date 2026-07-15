# bayesian-optimization

Common Bayesian optimization modes for **microfluidic synthesis**: compose recipes under flow constraints, suggest the next experiments, and drive real pumps from a notebook.

Package: [smartfluidics/bayesian-optimization](https://github.com/smartfluidics/bayesian-optimization).

## Quick API

**Scalar EI** (Expected Improvement on one scalar score, e.g. yield) — `MixerSession` + CSV recipes:

```python
from bayesian_optimization import DesignSpace, ScalarEISuggester
from bayesian_optimization.microfluidics import MixerConfig, MixerSession

cfg = MixerConfig(
    data_dir="data_bayes",
    var_to_syringe=dict(TWEEN=0, PEG=1, LIPID=2, LEY=3),
    bounds={k: (1.0, 40.0) for k in ["TWEEN", "PEG", "LIPID", "LEY"]},
    n_syringes=8, total_speed=50.0, time_synth=30.0,
    separator_syringe=7, time_separator=20.0, legacy_csv=True,
)
space = DesignSpace(
    names=list(cfg.var_to_syringe),
    bounds=cfg.bounds,
    sum_equals=cfg.total_speed,
)
mixer = MixerSession(cfg, ScalarEISuggester(space, maximize=True))

recipe = mixer.generate_lhs_iter0(n_points=10)
next_csv = mixer.suggest_next(iter_idx=0, n_points=10)  # needs *_results.csv
```

**Uncertainty** (several outputs, e.g. UV/SAXS) — maximize epistemic variance:

```python
from bayesian_optimization import DesignSpace, UncertaintySuggester

FLOW = ["au", "ag", "peg", "ctab", "cit", "pvp", "ascorb", "teos"]
TARGETS = ["uv1", "uv2", "saxs1", "saxs2"]

space = DesignSpace(
    names=FLOW,
    bounds={
        "au": (3.3, 8.5),
        "peg": (1e-3, 19.999), "ctab": (1e-3, 19.999),
        "cit": (1e-3, 19.999), "pvp": (1e-3, 19.999),
        "ag": (0.0, 1.0), "ascorb": (0.0, 20.0), "teos": (0.0, 1.0),
    },
    sum_equals=20.0,
    fixed={"ag": 0.0, "teos": 0.0},
    linear_deps={"ascorb": {"au": 1.2, "ag": 1.2}},
)
sug = UncertaintySuggester(
    space,
    acquisition="max_variance",
    output_names=TARGETS,
)
# X: (n, n_dims), Y: (n, n_tasks)
next_x = sug.suggest(X, Y, n_points=1)
```

| Class | Role |
|-------|------|
| `MixerConfig` | Microfluidic CSV / schedule settings: syringe map, bounds, total flow, synthesis time, separator rows. No optimizer knobs. |
| `DesignSpace` | Feasible composition space: box bounds, optional `sum_equals`, fixed variables, linear deps; sampling and projection. |
| `ScalarEISuggester` | Bayesian optimization with a Gaussian process and **Expected Improvement (EI)** on one scalar objective (e.g. yield). Via ProcessOptimizer; `maximize=True` to maximize the score. |
| `UncertaintySuggester` | Multitask ICM GP; suggest points that maximize epistemic uncertainty (or UCB) over several outputs. |
| `MixerSession` | Lab CSV loop: LHS → recipe CSV, then `suggest_next` after you write `*_results.csv`. Takes config + any suggester above. |

## Examples

| Example | What |
|---------|------|
| [`examples/SLNP_bayes/bayes_mf_loop.ipynb`](examples/SLNP_bayes/bayes_mf_loop.ipynb) | Live MF loop: connect pumps → valves → LHS → `ScheduleRunner` → score → next suggestion |
| [`examples/synthetic_test_slnp/synthetic_bayes_loop.ipynb`](examples/synthetic_test_slnp/synthetic_bayes_loop.ipynb) | Same SLNP API without hardware (synthetic yield + EI) |
| [`examples/Au_multispectral_synthetic/au_multispectral_synthetic.ipynb`](examples/Au_multispectral_synthetic/au_multispectral_synthetic.ipynb) | Synthetic Au: uncertainty sampling + EI toward a UV/SAXS target |
| [`examples/printer_calibration/printer_calibration.ipynb`](examples/printer_calibration/printer_calibration.ipynb) | Dispenser (printer): home / jog / local origin / `tube_coordinates.json` |

## Hardware control

Pump client, `ScheduleRunner`, and dispenser (printer) live in [`bayesian_optimization/microfluidics/`](bayesian_optimization/microfluidics/). Usage examples (schedule without / with printer): [`bayesian_optimization/microfluidics/README.md`](bayesian_optimization/microfluidics/README.md).

## Install

Python 3.10–3.12.

```bash
pip install -e .
pip install -e ".[uncertainty]"   # UncertaintySuggester (torch + gpytorch)
```

### Hardware (`smart_pump` + dispenser / printer)

```bash
pip install --extra-index-url https://lab:iUULBNNmgslKVjaX35MA@pypi.stud.mmcs.sfedu.ru smart_pump
pip install pyserial   # Printer3DClient (dispenser / printer)
```

## Notes
- Pump YAML: [`examples/SLNP_bayes/sample_config.yml`](examples/SLNP_bayes/sample_config.yml).
- After install, **manually** overwrite the installed modules with lab copies from this repo:

```text
smart_pump_overrides/pumps.py        →  <site-packages>/smart_pump/pumps.py
smart_pump_overrides/communicator.py →  <site-packages>/smart_pump/communicator.py
```
