"""ScheduleRunner — ported as-is from the lab notebook (do not refactor)."""

from __future__ import annotations

import asyncio
import json
import os
import time

import numpy as np
import pandas as pd


class ScheduleRunner:
    """
    ============================================================================
    CLASS: ScheduleRunner
    ============================================================================
    Purpose:
        Automatic execution of a syringe-pump (MF) schedule
        SYNCHRONOUSLY with 3D printer moves (Printer3DClient).

    Step logic:
        0. Volume check and auto-fill if needed (before printer motion).
        1. Raise the printer to SAFE_Z.
        2. Move to the tube XY coordinates (by index from JSON).
        3. Lower to WORK_Z.
        4. Wait the settle delay (PRE_PUMP_DELAY).
        5. Run the pump step for the exact scheduled time.
        6. Raise the printer to SAFE_Z.
    ============================================================================
    """

    def __init__(
        self,
        MF,
        printer=None,
        pumps=(0, 1, 2, 3),
        tick_sleep=0.01,
        coordinates_file="tube_coordinates.json",
    ):
        self.MF = MF
        self.printer = printer
        self.pumps = list(pumps)
        self.tick_sleep = float(tick_sleep)

        # VOLUME / POSITIONS
        self.MAX_TICKS = 76800
        self.SAFETY_MARGIN = 500  # can be increased if desired

        # PRINTER
        self.SAFE_Z = 35.0
        self.WORK_Z = 35.0
        self.XY_FEED = 3000.0
        self.Z_FEED = 1200.0

        # TIMINGS
        self.XY_SETTLE_DELAY = 0.3
        self.Z_SETTLE_DELAY = 0.5
        self.PRE_PUMP_DELAY = 0.5

        # SCHEDULE STATE
        self.sch = None
        self.sch_len = 0
        self.idx = 0
        self.time_left = 0.0

        # EVENTS
        self._pause_evt = asyncio.Event()
        self._pause_evt.set()
        self._stop_evt = asyncio.Event()
        self._paused = False

        # CURRENT PUMP STEP STATE
        self._row_active = False
        self._row_idx = None
        self._row_speed = None
        self._row_t_total = 0.0
        self._row_t_left = 0.0
        self._row_t_start = 0.0

        # TUBE COORDINATES
        self.tube_coordinates = self._load_coordinates(coordinates_file)

        # VolumeTick for pumps
        self.volume_tick = self._resolve_volume_ticks()

        self.pumps_to_fill = pumps

        if not hasattr(MF, "SetSpeedAndVolumeToMove"):
            raise AttributeError("MF has no SetSpeedAndVolumeToMove() method.")

        if self.printer is not None:
            print("[SCHEDULE] Printer integration ENABLED")
        else:
            print("[SCHEDULE] Printer integration DISABLED (MF only)")

        print(f"[SCHEDULE] Coordinates loaded: {len(self.tube_coordinates)} tubes")
        print("[SCHEDULE] real pause enabled: pause -> MF.Stop(), resume -> reissue")

    # =========================================================================
    # COORDINATES
    # =========================================================================

    def _load_coordinates(self, filepath):
        coordinates = {}
        if not os.path.exists(filepath):
            print(f"[WARNING] Coordinates file '{filepath}' not found. Using empty map.")
            return coordinates

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                tube_id = int(key)
                x, y = float(value[0]), float(value[1])
                coordinates[tube_id] = (x, y)
            print(f"[COORDINATES] Loaded {len(coordinates)} tube positions from '{filepath}'")
        except Exception as e:
            print(f"[ERROR] Failed to load coordinates: {e}")
        return coordinates

    def reload_coordinates(self, filepath=None):
        if filepath is None:
            filepath = "tube_coordinates.json"
        self.tube_coordinates = self._load_coordinates(filepath)

    # =========================================================================
    # CONTROL
    # =========================================================================

    def pause(self):
        self._pause_evt.clear()
        print("[SCHEDULE] pause requested")

    def resume(self):
        self._pause_evt.set()
        print("[SCHEDULE] resume requested")

    def stop(self):
        self._stop_evt.set()
        self._pause_evt.set()
        print("[SCHEDULE] stop requested")

    # =========================================================================
    # SCHEDULE LOADING
    # =========================================================================

    def load(self, sch):
        df = sch
        self.sch = df
        self.sch_len = len(df)
        self.idx = 0
        self.time_left = float(df["8"].sum())
        print(f"[SCHEDULE] loaded rows={self.sch_len}, total_time={self.time_left:.2f}s")

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _resolve_volume_ticks(self):
        dev = getattr(self.MF, "device", None)
        if dev is None:
            c = getattr(self.MF, "client", None)
            dev = getattr(c, "device", None) if c is not None else None
        if dev is None or not hasattr(dev, "syringes"):
            raise AttributeError("Could not find device.syringes for VolumeTick.")

        vt = {}
        for i in range(max(self.pumps) + 1):
            syr = dev.syringes[i]
            vt[i] = float(syr.parameters.VolumeTick)
        return vt

    async def _stop_motors_safe(self, pumps=None):
        try:
            if pumps is None:
                await self.MF.Stop()
            else:
                await self.MF.Stop(pumps)
        except Exception as e:
            print("[SCHEDULE] Stop() error:", repr(e))

    async def _start_motors_safe(self, pumps=None):
        try:
            if pumps is None:
                await self.MF.Start()
            else:
                await self.MF.Start(pumps)
        except Exception as e:
            print("[SCHEDULE] Start() error:", repr(e))

    # =========================================================================
    # PUMP MOTION (NO AUTO-FILL, CLEAN STEP ONLY)
    # =========================================================================

    async def _issue_motion_for_time(self, speed, t_seconds):
        """
        Computes target volume from speed and t_seconds,
        and sends a single SetSpeedAndVolumeToMove command.
        Assumes volume was already checked and auto-fill done ABOVE.
        """
        cur_all = await self.MF.GetPositionsMF()
        cur_ticks = np.array([float(cur_all[p]) for p in self.pumps], dtype=float)

        active_speeds = np.array([float(speed[p]) for p in self.pumps], dtype=float)

        # ΔV = v * t
        delta_vol = active_speeds * float(t_seconds)
        delta_ticks = np.zeros_like(delta_vol, dtype=float)
        for j, p in enumerate(self.pumps):
            delta_ticks[j] = delta_vol[j] / self.volume_tick[p]

        tgt_ticks = cur_ticks - delta_ticks
        tgt_vol = np.zeros_like(tgt_ticks, dtype=float)
        for j, p in enumerate(self.pumps):
            tgt_vol[j] = tgt_ticks[j] * self.volume_tick[p]

        print(f"[MOTION] About to send command...")
        print(f"[MOTION] pumps={self.pumps}")
        print(f"[MOTION] speeds={np.abs(active_speeds).tolist()}")
        print(f"[MOTION] volumes={tgt_vol.tolist()}")
        print(f"active_speeds {active_speeds}")
        print(f"delta_ticks {delta_ticks}")
        print(f"tgt_vol {tgt_vol}")

        await self.MF.SetSpeedAndVolumeToMove(
            self.pumps,
            np.abs(active_speeds).tolist(),
            tgt_vol.tolist(),
        )
        print(f"[MOTION] Command sent! pumps={self.pumps}")
        await asyncio.sleep(0.1)

    # =========================================================================
    # EXECUTE ONE SCHEDULE ROW (PUMPS ONLY)
    # =========================================================================

    async def _run_row(self, row_idx, speed, t_total):
        """
        Runs pump motion for time t_total at the given speeds.
        Pause/resume recalculate remaining time correctly.
        """
        self._row_active = True
        self._row_idx = row_idx
        self._row_speed = speed.copy()
        self._row_t_total = float(t_total)
        self._row_t_left = float(t_total)
        self._row_t_start = time.monotonic()
        self._paused = False

        # First command issuance
        await self._issue_motion_for_time(self._row_speed, self._row_t_left)
        await self._start_motors_safe()
        print(f"[MOTOR] Motors started for row {row_idx+1}")

        while self._row_t_left > 0:
            if self._stop_evt.is_set():
                return

            # Pause
            if not self._pause_evt.is_set():
                if not self._paused:
                    elapsed = time.monotonic() - self._row_t_start
                    self._row_t_left = max(0.0, self._row_t_total - elapsed)
                    self._paused = True
                    print(f"[SCHEDULE] PAUSE now | row {row_idx+1} | left_in_row={self._row_t_left:.2f}s")
                    await self._stop_motors_safe()

                await self._pause_evt.wait()

                if self._stop_evt.is_set():
                    return

                if self._paused and self._row_t_left > 0:
                    print(f"[SCHEDULE] RESUME | row {row_idx+1} | reissue left={self._row_t_left:.2f}s")
                    self._row_t_start = time.monotonic()
                    self._paused = False
                    await self._issue_motion_for_time(self._row_speed, self._row_t_left)
                    await self._start_motors_safe()

            await asyncio.sleep(min(self.tick_sleep, self._row_t_left))

            if not self._paused:
                elapsed = time.monotonic() - self._row_t_start
                self._row_t_left = max(0.0, self._row_t_total - elapsed)

        self._row_active = False

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def run(self, start_idx=0):
        """
        Main loop: schedule step = (auto-fill -> printer -> pumps -> printer).
        Auto-fill is called BEFORE printer and syringe motion for each row.
        """
        if self.sch is None:
            raise RuntimeError("Schedule not loaded. Call runner.load(sch)")

        self._stop_evt.clear()
        self._pause_evt.set()
        self._paused = False

        self.idx = max(0, int(start_idx))
        self.time_left = float(self.sch.loc[self.idx :, "8"].sum())

        # Printer initialization
        if self.printer is not None:
            print("[SCHEDULE] Initializing printer coordinates...")
            self.printer.setup_mm_absolute()
            await asyncio.sleep(0.5)
            print(f"[PRINTER] Move to SAFE_Z = {self.SAFE_Z}")
            self.printer.move_safe(self.SAFE_Z, z_feed=self.Z_FEED)
            await asyncio.sleep(1.0)

        try:
            for k in range(self.idx, self.sch_len):
                if self._stop_evt.is_set():
                    print("[SCHEDULE] stopped by user")
                    break

                await self._pause_evt.wait()

                row = self.sch.iloc[k]
                t = float(row["8"])
                speed = row[[str(i) for i in range(8)]].values.astype(float)

                # Tube index
                try:
                    big_index = int(row["index"])
                except (KeyError, ValueError, TypeError):
                    big_index = None

                self.idx = k
                self.time_left -= t

                timestamp = time.strftime("%H:%M:%S")
                print(f"\n[{timestamp}] [SCHEDULE] row {k+1}/{self.sch_len} | t={t}s")
                print(f"[{timestamp}] [SCHEDULE] speed={speed.tolist()}")

                # 0. VOLUME CHECK / AUTO-FILL
                await self._check_pumps_not_empty(speed, t, auto_fill=True)

                # 1. PRINTER MOTION
                if self.printer is not None and big_index is not None:
                    if big_index in self.tube_coordinates:
                        x, y = self.tube_coordinates[big_index]
                        print(f"[{timestamp}] [PRINTER] Tube Index {big_index} -> XY({x}, {y})")

                        self.printer.move_safe(self.SAFE_Z, z_feed=self.Z_FEED)
                        await asyncio.sleep(self.Z_SETTLE_DELAY)

                        self.printer.move_xy(x, y, xy_feed=self.XY_FEED)
                        await asyncio.sleep(self.XY_SETTLE_DELAY)

                        print(f"[{timestamp}] [PRINTER] Move WORK_Z ({self.WORK_Z})")
                        self.printer.move_z(self.WORK_Z, z_feed=self.Z_FEED)
                        await asyncio.sleep(self.Z_SETTLE_DELAY)

                        if self.PRE_PUMP_DELAY > 0:
                            print(f"[{timestamp}] [PRINTER] Waiting {self.PRE_PUMP_DELAY}s before pumps...")
                            await asyncio.sleep(self.PRE_PUMP_DELAY)
                    else:
                        print(f"[WARNING] Tube index {big_index} not in coordinates map.")
                else:
                    if self.printer is None:
                        print("[PRINTER] Disabled")

                # 2. PUMPS
                pump_start_time = time.strftime("%H:%M:%S")
                print(f"[{pump_start_time}] [PUMPS] Starting dispensing...")
                await self._run_row(k, speed, t)

                # 3. PRINTER RETURN
                if self.printer is not None:
                    print(f"[{time.strftime('%H:%M:%S')}] [PRINTER] Return to SAFE_Z")
                    self.printer.move_safe(self.SAFE_Z, z_feed=self.Z_FEED)
                    await asyncio.sleep(self.Z_SETTLE_DELAY)

                if self._stop_evt.is_set():
                    break

        finally:
            print("[SCHEDULE] Stop motors (global)")
            await self._stop_motors_safe()

            if self.printer is not None:
                print("[PRINTER] Return to Home/Safe")
                self.printer.move_safe(self.SAFE_Z, z_feed=self.Z_FEED)
                self.printer.move_xy(0.0, 0.0, xy_feed=self.XY_FEED)

            print("[SCHEDULE] done")
            print(
                """ 
__| |__________________________________________________________| |__ 
__   __________________________________________________________   __ 
  | |                                                          | |  
  | |                                                          | |  
  | |  ░█▀▀░█▄█░█▀█░█▀▄░▀█▀░░░█▀▀░█░░░█░█░▀█▀░█▀▄░▀█▀░█▀▀░█▀▀  | |  
  | |  ░▀▀█░█░█░█▀█░█▀▄░░█░░░░█▀▀░█░░░█░█░░█░░█░█░░█░░█░░░▀▀█  | |  
  | |  ░▀▀▀░▀░▀░▀░▀░▀░▀░░▀░░░░▀░░░▀▀▀░▀▀▀░▀▀▀░▀▀░░▀▀▀░▀▀▀░▀▀▀  | |  
  | |                                                          | |  
__| |__________________________________________________________| |__
__   __________________________________________________________   __
  | |                                                          | |  """
            )
            print(
                """
The Smart Materials Research Institute at the Southern Federal University:
[https://nano.sfedu.ru/](https://nano.sfedu.ru/)
Contacts:
344090, Rostov-on-Don,
Andrei Sladkov St., 178/24
+7 (863) 305-1996
[nano@sfedu.ru](mailto:nano@sfedu.ru)
                  """
            )

    # =========================================================================
    # VOLUME CHECK + AUTO-FILL
    # =========================================================================

    async def _check_pumps_not_empty(self, speed, t_seconds, auto_fill=True):
        """
        Volume check and auto-fill before a step.
        IMPORTANT: no longer called inside _issue_motion_for_time,
        to avoid creating "extra" virtual pump steps.
        """
        pumps_to_check = self.pumps
        pumps_to_fill = self.pumps_to_fill

        cur_all = await self.MF.GetPositionsMF()
        vt_map = self.volume_tick

        errors = []
        details = []
        need_fill = []

        for ch in pumps_to_check:
            if ch not in vt_map:
                continue

            cur_ticks = float(cur_all[ch])
            vt = float(vt_map[ch])

            speed_ch = float(speed[ch]) if ch < len(speed) else 0.0

            delta_vol = speed_ch * t_seconds
            delta_ticks = delta_vol / vt if vt > 0 else 0.0
            target_ticks = cur_ticks - delta_ticks

            is_fill_candidate = ch in pumps_to_fill

            detail = {
                "pump": ch,
                "cur_ticks": cur_ticks,
                "target_ticks": target_ticks,
                "delta_ticks": delta_ticks,
                "direction": "OUT" if speed_ch > 0 else ("IN" if speed_ch < 0 else "HOLD"),
                "fill_candidate": is_fill_candidate,
            }
            details.append(detail)

            # Approaching 0 (empty)
            if speed_ch > 0 and target_ticks < self.SAFETY_MARGIN:
                errors.append(
                    f"PUMP {ch}: insufficient volume! "
                    f"cur={cur_ticks:.0f}, need={delta_ticks:.0f}, target={target_ticks:.0f}"
                )
                if is_fill_candidate:
                    need_fill.append(ch)
                    print(f"[VOLUME CHECK] PUMP {ch}: low volume AND in fill list → will auto-fill")
                else:
                    print(f"[VOLUME CHECK] PUMP {ch}: low volume BUT NOT in fill list → error only")

            # Overfill
            elif speed_ch < 0 and target_ticks > (self.MAX_TICKS - self.SAFETY_MARGIN):
                errors.append(
                    f"PUMP {ch}: syringe overfill! "
                    f"cur={cur_ticks:.0f}, delta={delta_ticks:.0f}, "
                    f"target={target_ticks:.0f}, max={self.MAX_TICKS}"
                )
                raise RuntimeError(
                    f"PUMP {ch}: syringe overfill! target={target_ticks:.0f} > max={self.MAX_TICKS}"
                )

        print("\n[VOLUME CHECK] Summary:")
        for d in details:
            fill_pct = (d["cur_ticks"] / self.MAX_TICKS) * 100
            candidate_mark = " [FILL CANDIDATE]" if d["fill_candidate"] else ""
            print(
                f"  PUMP {d['pump']} | {d['direction']} | "
                f"cur={d['cur_ticks']:.0f}/{self.MAX_TICKS} ({fill_pct:.1f}%){candidate_mark} | "
                f"target={d['target_ticks']:.0f} (delta={d['delta_ticks']:.0f})"
            )
        print(f"  → Pumps to auto-fill: {need_fill}\n")

        if errors and auto_fill and need_fill:
            print(f"[VOLUME CHECK] Low volume detected on pumps {need_fill}! Starting auto-fill...")
            await self._fill_pumps_to_max(need_fill)
            print("[VOLUME CHECK] Re-checking volume after fill...")
            return await self._check_pumps_not_empty(speed, t_seconds, auto_fill=False)

        if errors:
            print("[VOLUME CHECK] FAILED")
            for err in errors:
                print(f" {err}")
            raise RuntimeError(f"Volume check failed: {errors}")

        print("[VOLUME CHECK] PASSED ")
        return True

    # =========================================================================
    # AUTO-FILL
    # =========================================================================

    async def _fill_pumps_to_max(self, pumps_to_fill, fill_speeds=[150, 150, 150]):
        if not pumps_to_fill:
            return

        print(f"[FILL] Starting fill procedure for pumps: {pumps_to_fill}")

        # Valves to FILL mode
        for i in pumps_to_fill:
            await self.MF.ValveChangeOne(i, 1)
            print(f"[FILL] Pump {i}: Valve -> FILL (state=1)")

        await asyncio.sleep(0.5)

        vt_map = self.volume_tick
        fill_speeds_list = []
        target_volumes = []

        for i, p in enumerate(pumps_to_fill):
            spd = fill_speeds[i] if i < len(fill_speeds) else fill_speeds[-1]
            fill_speeds_list.append(float(spd))
            target_volumes.append(float(self.MAX_TICKS * vt_map[p]))

        print(f"[FILL] SetSpeedAndVolumeToMove: pumps={pumps_to_fill}, speeds={fill_speeds_list}")

        await self.MF.Stop(pumps_to_fill)
        await self.MF.SetSpeedAndVolumeToMove(pumps_to_fill, fill_speeds_list, target_volumes)
        await self.MF.Start(pumps_to_fill)

        max_fill_time = 0.0
        cur_all = await self.MF.GetPositionsMF()

        for p in pumps_to_fill:
            cur_pos = float(cur_all[p])
            vt = float(vt_map[p])
            spd_idx = pumps_to_fill.index(p)
            spd_ml_s = fill_speeds_list[spd_idx]

            remaining_ticks = self.MAX_TICKS - cur_pos
            ticks_per_sec = spd_ml_s / vt if vt > 0 else 100
            est_time = remaining_ticks / ticks_per_sec if ticks_per_sec > 0 else 60.0
            total_time = est_time + 2.0

            print(
                f"  [FILL] Pump {p}: {cur_pos:.0f} -> {self.MAX_TICKS} ticks | "
                f"{remaining_ticks:.0f} ticks left | {spd_ml_s} uL/s | "
                f"~{total_time:.1f}s"
            )
            max_fill_time = max(max_fill_time, total_time)

        fill_wait_time = max(max_fill_time, 1.0)
        print(f"[FILL] Waiting {fill_wait_time:.1f}s for fill to complete...")
        await asyncio.sleep(fill_wait_time)

        await self.MF.Stop(pumps_to_fill)

        for i in pumps_to_fill:
            await self.MF.ValveChangeOne(i, 0)
            print(f"[FILL] Pump {i}: Valve -> WORK (state=0)")

        await asyncio.sleep(0.5)

        after_fill = await self.MF.GetPositionsMF()
        for p in pumps_to_fill:
            fill_pct = (float(after_fill[p]) / self.MAX_TICKS) * 100
            print(f"[FILL] Pump {p}: position={after_fill[p]:.0f}/{self.MAX_TICKS} ({fill_pct:.1f}%)")

        print("[FILL] Fill procedure completed ")
