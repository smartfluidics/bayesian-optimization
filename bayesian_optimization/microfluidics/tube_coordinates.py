"""Tube XY map for dispenser (printer) schedule (``tube_coordinates.json``).

JSON format used by ``ScheduleRunner``::

    {"1": [-90.0, -500.0], "2": [-40.0, -500.0], ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class TubeCoordinatesGenerator:
    """Load / edit / save tube index -> (x, y) local coordinates."""

    def __init__(self, filepath: str | Path = "tube_coordinates.json") -> None:
        self.filepath = Path(filepath)
        self.coords: dict[int, tuple[float, float]] = {}
        if self.filepath.is_file():
            self.load()

    def load(self, filepath: str | Path | None = None) -> None:
        path = Path(filepath) if filepath is not None else self.filepath
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self.coords = {
            int(k): (float(v[0]), float(v[1])) for k, v in dict(raw).items()
        }
        self.filepath = path
        print(f"[COORDINATES] Loaded {len(self.coords)} tubes from '{path}'")

    def save(self, filepath: str | Path | None = None) -> Path:
        path = Path(filepath) if filepath is not None else self.filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {str(k): [float(v[0]), float(v[1])] for k, v in sorted(self.coords.items())}
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.filepath = path
        print(f"[COORDINATES] Saved {len(self.coords)} tubes to '{path}'")
        return path

    def add_tube(self, tube_id: int, x: float, y: float) -> None:
        self.coords[int(tube_id)] = (float(x), float(y))

    def remove_tube(self, tube_id: int) -> None:
        self.coords.pop(int(tube_id), None)

    def get_tube(self, tube_id: int) -> tuple[float, float] | None:
        return self.coords.get(int(tube_id))

    def list_all(self) -> None:
        if not self.coords:
            print("[COORDINATES] empty")
            return
        for tid in sorted(self.coords):
            x, y = self.coords[tid]
            print(f"  tube {tid:3d}: X={x:.3f}  Y={y:.3f}")
        print(f"[COORDINATES] {len(self.coords)} tubes")

    def add_many(self, items: Iterable[tuple[int, float, float]]) -> None:
        for tube_id, x, y in items:
            self.add_tube(tube_id, x, y)

    def as_dict(self) -> dict[int, tuple[float, float]]:
        return dict(self.coords)


def create_interactive_generator(
    filepath: str | Path = "tube_coordinates.json",
) -> TubeCoordinatesGenerator:
    """Return a generator; if ipywidgets is available, show a small editor UI."""
    gen = TubeCoordinatesGenerator(filepath)
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print(
            "[COORDINATES] ipywidgets not installed; use gen.add_tube / gen.save programmatically"
        )
        return gen

    tid = widgets.IntText(value=1, description="tube id")
    xw = widgets.FloatText(value=0.0, description="X")
    yw = widgets.FloatText(value=0.0, description="Y")
    out = widgets.Output()

    def _add(_=None) -> None:
        with out:
            out.clear_output()
            gen.add_tube(int(tid.value), float(xw.value), float(yw.value))
            print(f"added tube {int(tid.value)} -> ({float(xw.value)}, {float(yw.value)})")

    def _save(_=None) -> None:
        with out:
            out.clear_output()
            gen.save()

    def _list(_=None) -> None:
        with out:
            out.clear_output()
            gen.list_all()

    def _reload(_=None) -> None:
        with out:
            out.clear_output()
            gen.load()

    btn_add = widgets.Button(description="Add")
    btn_save = widgets.Button(description="Save")
    btn_list = widgets.Button(description="List")
    btn_load = widgets.Button(description="Reload")
    btn_add.on_click(_add)
    btn_save.on_click(_save)
    btn_list.on_click(_list)
    btn_load.on_click(_reload)
    display(widgets.VBox([tid, xw, yw, widgets.HBox([btn_add, btn_save, btn_list, btn_load]), out]))
    return gen
