# `bayesian_optimization.microfluidics` — hardware

Pump client, schedule runner, and dispenser (printer) for lab notebooks.

Install: `smart_pump` (lab index) + optional `pyserial` for the dispenser (printer) — see [root README](../../README.md).

---

## Example 1 — schedule without printer

Pumps only; dispenser (printer) steps are skipped.

```python
from bayesian_optimization.microfluidics import (
    ScheduleRunner,
    connect_mf,
    create_pump_client,
    load_schedule_dataframe,
)

MF = create_pump_client("sample_config.yml")
await connect_mf(MF, "COM9")

# valves FILL → calibrate → fill → WORK (see lab notebook)

runner = ScheduleRunner(
    MF,
    pumps=(0, 1, 2, 3, 4, 5, 6),
    printer=None,
    tick_sleep=0.01,
)
sch = load_schedule_dataframe("recipes_iter_000.csv")
runner.load(sch)
await runner.run()
```

Full loop: [`examples/SLNP_bayes/bayes_mf_loop.ipynb`](../../examples/SLNP_bayes/bayes_mf_loop.ipynb).

---

## Example 2 — schedule with printer

Dispenser (printer) moves to tube XY from `tube_coordinates.json` before each pump step (schedule column `index` = tube id).

Calibrate the dispenser (printer) and build the JSON first: [`examples/printer_calibration/printer_calibration.ipynb`](../../examples/printer_calibration/printer_calibration.ipynb).

```python
from bayesian_optimization.microfluidics import (
    ScheduleRunner,
    Printer3DClient,
    connect_mf,
    create_pump_client,
    load_schedule_dataframe,
)

MF = create_pump_client("sample_config.yml")
await connect_mf(MF, "COM9")

printer = Printer3DClient(port="COM11", baudrate=115200)
printer.connect()
printer.home()
printer.move_safe(35)

runner = ScheduleRunner(
    MF,
    pumps=(0, 1, 2, 3, 4, 5, 6),
    printer=printer,
    coordinates_file="tube_coordinates.json",
    tick_sleep=0.01,
)
runner.PRE_PUMP_DELAY = 7
runner.XY_SETTLE_DELAY = 0.1
runner.Z_SETTLE_DELAY = 0.1

sch = load_schedule_dataframe("recipes_iter_000.csv")
runner.load(sch)
await runner.run()
```

---

## Modules

| Module | Role |
|--------|------|
| `client.py` | `create_pump_client`, `connect_mf` |
| `schedule_runner.py` | `ScheduleRunner` |
| `schedule_io.py` | `load_schedule_dataframe`, `run_schedule_csv` |
| `printer3d.py` | `Printer3DClient` |
| `tube_coordinates.py` | `TubeCoordinatesGenerator`, `create_interactive_generator` |
