"""3D printer / dispenser client for Ender-style G-code over serial.

Ported from lab ``printer3d.Printer3DClient`` (no interactive UI here).
Requires ``pyserial``.
"""

from __future__ import annotations

import time
from typing import Any


class Printer3DClient:
    """Ender-style helper: connect, home, jog, local origin, rack moves."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.5):
        self.port_name = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Any = None

        self.small_pitch_x = 10.0
        self.small_pitch_y = 10.0
        self.small_cols = 10
        self.small_rows = 5
        self.big_tubes: dict[int, tuple[float, float]] = {
            1: (20.0, 95.0),
            2: (70.0, 95.0),
            3: (115.0, 95.0),
        }

    def connect(self) -> None:
        if self.ser and self.ser.is_open:
            return
        try:
            import serial
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Printer3DClient requires pyserial. Install with: pip install pyserial"
            ) from e

        self.ser = serial.Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            timeout=self.timeout,
            write_timeout=5,
            dsrdtr=False,
            rtscts=False,
        )
        time.sleep(5.0)
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass
        self.ser.write(b"\n")
        self.ser.flush()
        time.sleep(0.5)
        print(f"[PRINTER] connected on {self.port_name}")

    def disconnect(self) -> None:
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        finally:
            self.ser = None
            print("[PRINTER] disconnected")

    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def _clean(self, line: str) -> str:
        if ";" in line:
            line = line.split(";", 1)[0]
        return line.strip()

    def _timeout_for_cmd(self, cmd: str) -> float:
        up = cmd.upper().strip()
        if up.startswith("G28"):
            return 120.0
        if up.startswith("G29"):
            return 180.0
        if up.startswith("G4"):
            dwell_s = 0.0
            for part in up.split()[1:]:
                if part.startswith("S"):
                    try:
                        dwell_s = float(part[1:])
                    except Exception:
                        pass
                elif part.startswith("P"):
                    try:
                        dwell_s = float(part[1:]) / 1000.0
                    except Exception:
                        pass
            return max(15.0, dwell_s + 15.0)
        return 20.0

    def send(self, cmd: str, verbose: bool = True) -> None:
        if not self.is_connected():
            raise RuntimeError("Printer is not connected.")
        clean = self._clean(cmd)
        if not clean:
            return
        if verbose:
            print(f"TX: {clean}")
        self.ser.write((clean + "\n").encode("utf-8"))
        self.ser.flush()
        timeout = self._timeout_for_cmd(clean)
        end_t = time.time() + timeout
        got_any = False
        while time.time() < end_t:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if not line:
                continue
            got_any = True
            if verbose:
                print(f"RX: {line}")
            low = line.lower()
            if low == "ok" or low.startswith("ok "):
                return
            if "error" in low:
                raise RuntimeError(f"Printer error: {line}")
            if "busy" in low or low == "wait" or "processing" in low or low.startswith("echo:"):
                continue
        if got_any:
            raise TimeoutError(f"No OK from printer for line: {clean}")
        raise TimeoutError(f"No response from printer for line: {clean}")

    def setup_mm_absolute(self) -> None:
        self.send("G21")
        self.send("G90")

    def use_relative(self) -> None:
        self.send("G91")

    def use_absolute(self) -> None:
        self.send("G90")

    def home(self) -> None:
        self.setup_mm_absolute()
        self.send("G28")

    def move_xy(self, x: float, y: float, xy_feed: float = 3000.0) -> None:
        self.use_absolute()
        self.send(f"G1 X{x:.3f} Y{y:.3f} F{xy_feed:.0f}")

    def move_z(self, z: float, z_feed: float = 1200.0) -> None:
        self.use_absolute()
        self.send(f"G1 Z{z:.3f} F{z_feed:.0f}")

    def move_xyz(
        self,
        x: float,
        y: float,
        z: float,
        xy_feed: float = 3000.0,
        z_feed: float = 1200.0,
    ) -> None:
        self.move_xy(x, y, xy_feed=xy_feed)
        self.move_z(z, z_feed=z_feed)

    def move_safe(self, safe_z: float, z_feed: float = 1200.0) -> None:
        self.move_z(safe_z, z_feed=z_feed)

    def jog_x(self, dx: float, feed: float = 1500.0) -> None:
        self.use_relative()
        self.send(f"G1 X{dx:.3f} F{feed:.0f}")
        self.use_absolute()

    def jog_y(self, dy: float, feed: float = 1500.0) -> None:
        self.use_relative()
        self.send(f"G1 Y{dy:.3f} F{feed:.0f}")
        self.use_absolute()

    def jog_z(self, dz: float, feed: float = 600.0) -> None:
        self.use_relative()
        self.send(f"G1 Z{dz:.3f} F{feed:.0f}")
        self.use_absolute()

    def jog_xy(self, dx: float, dy: float, feed: float = 1500.0) -> None:
        self.use_relative()
        self.send(f"G1 X{dx:.3f} Y{dy:.3f} F{feed:.0f}")
        self.use_absolute()

    def set_local_origin_here(self, z_value: float = 0.0) -> None:
        """Set current machine point as local X0 Y0 Z=z_value (G92)."""
        self.send(f"G92 X0 Y0 Z{float(z_value):.3f}")
        self.send("G4 S1")
        print(f"[PRINTER] current point set as local X0 Y0 Z{float(z_value):.3f}")

    def goto_local_zero(self, xy_feed: float = 3000.0, z_feed: float = 1200.0) -> None:
        self.use_absolute()
        self.send(f"G1 X0 Y0 F{xy_feed:.0f}")
        self.send(f"G1 Z0 F{z_feed:.0f}")

    def goto_local_point(
        self,
        x: float,
        y: float,
        z: float,
        xy_feed: float = 3000.0,
        z_feed: float = 1200.0,
    ) -> None:
        self.use_absolute()
        self.send(f"G1 X{x:.3f} Y{y:.3f} F{xy_feed:.0f}")
        self.send(f"G1 Z{z:.3f} F{z_feed:.0f}")

    def set_small_rack_geometry(
        self,
        pitch_x: float = 10.0,
        pitch_y: float = 10.0,
        cols: int = 10,
        rows: int = 5,
    ) -> None:
        self.small_pitch_x = float(pitch_x)
        self.small_pitch_y = float(pitch_y)
        self.small_cols = int(cols)
        self.small_rows = int(rows)

    def set_big_tube_positions(self, big_tubes: dict) -> None:
        self.big_tubes = {int(k): (float(v[0]), float(v[1])) for k, v in big_tubes.items()}

    def small_tube_xy(self, tube_index: int) -> tuple[float, float]:
        idx = int(tube_index) - 1
        if idx < 0:
            raise ValueError("tube_index must start from 1")
        row = idx // self.small_cols
        col = idx % self.small_cols
        if row >= self.small_rows:
            raise ValueError(f"tube_index {tube_index} outside rack size")
        return col * self.small_pitch_x, row * self.small_pitch_y

    def goto_small_tube(
        self,
        tube_index: int,
        safe_z: float = 35.0,
        work_z: float = 35.0,
        xy_feed: float = 3000.0,
        z_feed: float = 1200.0,
    ) -> None:
        x, y = self.small_tube_xy(tube_index)
        print(f"[PRINTER] goto small tube {tube_index} -> X={x:.3f}, Y={y:.3f}")
        self.move_safe(safe_z, z_feed=z_feed)
        self.move_xy(x, y, xy_feed=xy_feed)
        self.move_z(work_z, z_feed=z_feed)

    def goto_big_tube(
        self,
        tube_index: int,
        safe_z: float = 35.0,
        work_z: float = 35.0,
        xy_feed: float = 3000.0,
        z_feed: float = 1200.0,
    ) -> None:
        if tube_index not in self.big_tubes:
            raise ValueError(f"big tube {tube_index} not found")
        x, y = self.big_tubes[tube_index]
        print(f"[PRINTER] goto big tube {tube_index} -> X={x:.3f}, Y={y:.3f}")
        self.move_safe(safe_z, z_feed=z_feed)
        self.move_xy(x, y, xy_feed=xy_feed)
        self.move_z(work_z, z_feed=z_feed)

    def park(
        self,
        x: float = 0.0,
        y: float = 0.0,
        safe_z: float = 35.0,
        xy_feed: float = 3000.0,
        z_feed: float = 1200.0,
    ) -> None:
        self.move_safe(safe_z, z_feed=z_feed)
        self.move_xy(x, y, xy_feed=xy_feed)
        print(f"[PRINTER] parked at X={x:.3f}, Y={y:.3f}, safe Z={safe_z:.3f}")
