# bayesian-optimization

Bayesian experiment suggestion for [smartfluidics](https://github.com/smartfluidics): microfluidic recipe loops (scalar EI) and multispectral uncertainty sampling (multitask GP).

Concrete setups live in `examples/`. Docs and messages are in English.

## Install

Python 3.10–3.12.

```bash
pip install -e .
pip install -e ".[uncertainty]"
```

### Hardware (`smart_pump`)

```bash
pip install --extra-index-url https://lab:iUULBNNmgslKVjaX35MA@pypi.stud.mmcs.sfedu.ru smart_pump
```

After install, **manually** overwrite the installed modules with lab copies from this repo:

```text
smart_pump_overrides/pumps.py        →  <site-packages>/smart_pump/pumps.py
smart_pump_overrides/communicator.py →  <site-packages>/smart_pump/communicator.py
```

```bash
python -c "import smart_pump, pathlib; print(pathlib.Path(smart_pump.__file__).parent)"
# then copy the two files into that folder (re-do after every smart_pump reinstall)
```

Config for MF notebooks: `examples/SLNP_bayes/sample_config.yml`.

## Code map

### Library — `bayesian_optimization/`

| File | What it does |
|------|----------------|
| `__init__.py` | Public exports (`DesignSpace`, `ScalarEISuggester`, lazy `UncertaintySuggester`) |
| `base.py` | `BayesianSuggester` Protocol (typing only; unused at runtime) |
| `space.py` | `DesignSpace`: bounds, `sum_equals`, fixed / linear deps, LHS & rejection sampling |
| `scalar_ei.py` | `ScalarEISuggester`: ProcessOptimizer GP + EI + SumEquals for one scalar score |
| `uncertainty.py` | `UncertaintySuggester`: GPyTorch multitask ICM GP + epistemic / UCB suggest |

### Microfluidics — `bayesian_optimization/microfluidics/`

| File | What it does |
|------|----------------|
| `__init__.py` | Exports mixer / runner / client / schedule_io |
| `mixer.py` | `MixerConfig` + `MixerSession`: LHS → recipe CSV, `suggest_next` from `*_results.csv` |
| `recipes.py` | Read/write lab recipe & results CSVs (legacy alternating recipe/separator rows) |
| `schedule_io.py` | Load recipe CSV as DataFrame; `run_schedule_csv` → `ScheduleRunner.run` |
| `schedule_runner.py` | Lab schedule executor (pumps ± printer, auto-fill); ported as-is |
| `client.py` | `create_pump_client` / `connect_mf` wrappers around `smart_pump` |

### Lab pump overrides — `smart_pump_overrides/`

| File | What it does |
|------|----------------|
| `pumps.py` | Lab-patched `smart_pump.pumps` (copy into site-packages after install) |
| `communicator.py` | Lab-patched `smart_pump.communicator` (same) |

### Examples — `examples/`

| Path | What it does |
|------|----------------|
| `SLNP_bayes/bayes_mf_loop.ipynb` | Hardware SLNP loop: connect → valves → LHS → ScheduleRunner → suggest |
| `SLNP_bayes/sample_config.yml` | Syringe / pump YAML for `smart_pump` |
| `SLNP_bayes/data_bayes/*.csv` | Sample recipe / results CSVs |
| `synthetic_test_slnp/synthetic_bayes_loop.ipynb` | No-hardware Bayes demo: synthetic yield + EI + plots |
| `Au_multispectral/campaign.py` | `UncertaintyCSVCampaign`: CSV → columns → GP → suggest / LOOCV |
| `Au_multispectral/eval_gp.py` | Train-fit and LOOCV prediction tables |
| `Au_multispectral/plots.py` | LOOCV scatter + uncertainty replay curve plots |
| `Au_multispectral/prepare_data.py` | Station history CSV → training-ready table |
| `Au_multispectral/demo_run.py` | End-to-end Au demo CLI |
| `Au_multispectral/__init__.py` | Exports campaign API |
| `Au_multispectral/data/` | Station CSVs (+ generated outputs from demos) |

### Tests — `tests/`

| File | What it does |
|------|----------------|
| `test_space.py` | DesignSpace feasibility / sampling |
| `test_scalar_ei.py` | ScalarEI + MixerSession CSV roundtrip |
| `test_schedule_io.py` | Recipe CSV → runner DataFrame shape |
| `test_uncertainty_smoke.py` | Multitask GP suggest smoke (needs torch/gpytorch) |
| `fixtures/` | Small CSV fixtures for tests |

### Root packaging

| File | What it does |
|------|----------------|
| `pyproject.toml` | Package metadata, deps, optional `[uncertainty]` / `[hardware]` / `[dev]` |
| `requirements.txt` | Editable core install |
| `requirements-uncertainty.txt` | Core + uncertainty extras |
| `requirements-hardware.txt` | `smart_pump` from private index |

## Examples (how to run)

**Synthetic (no COM):** open `examples/synthetic_test_slnp/synthetic_bayes_loop.ipynb`.

**SLNP hardware:** `examples/SLNP_bayes/bayes_mf_loop.ipynb`  
(connect → FILL → calibrate → WORK → LHS → `ScheduleRunner` → score → `suggest_next`).

**Au uncertainty:**

```bash
python -m examples.Au_multispectral.demo_run
```

## Quick API

One Bayes stack: `DesignSpace` + `ScalarEISuggester` (LHS + EI).  
`MixerSession` is only a CSV / syringe wrapper around the same suggester — not a second optimiser.

```python
from bayesian_optimization import DesignSpace, ScalarEISuggester

space = DesignSpace(
    names=["TWEEN", "PEG", "LIPID", "LEY"],
    bounds={n: (1.0, 40.0) for n in ["TWEEN", "PEG", "LIPID", "LEY"]},
    sum_equals=50.0,
)
opt = ScalarEISuggester(space, maximize=True)
X0 = opt.sample_initial(10, seed=0)          # LHS
X_next = opt.suggest(X0, y, n_points=10)     # EI (+ SumEquals)
# ProcessOptimizer minimizes; maximize=True passes -y
```

Lab CSV loop (same LHS/EI inside):

```python
from bayesian_optimization.microfluidics import MixerConfig, MixerSession

mixer = MixerSession(MixerConfig(
    data_dir="data_bayes",
    var_to_syringe=dict(TWEEN=0, PEG=1, LIPID=2, LEY=3),
    bounds={k: (1.0, 40.0) for k in ["TWEEN", "PEG", "LIPID", "LEY"]},
    n_syringes=8, total_speed=50.0, time_synth=30.0,
    separator_syringe=7, time_separator=20.0, legacy_csv=True, maximize=True,
))
recipe = mixer.generate_lhs_iter0(n_points=10)           # → ScalarEISuggester.sample_initial
next_csv = mixer.suggest_next(iter_idx=0, n_points=10)   # needs *_results.csv
```

## Notes

- Do not casually refactor `ScheduleRunner` control logic.
- Re-copy `smart_pump_overrides/` after every `smart_pump` reinstall.
