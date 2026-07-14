import time
import logging
from typing import Any
import asyncio

from smart_pump.syringe import Syringe
from smart_pump.logs import format_modbus_log, print_timestamp
from smart_pump.communicator import *
from smart_pump.exceptions import *
from smart_pump.syringe_config import *
from smart_pump.syringe_parameters import *
from smart_pump.syringe_parameters import SyringeParameters


# ============================================================
#  Firmware register map (from  Definitions.h)
#  INT32 are stored as 2xU16: HI then LO  (getAnalogOutput(REG)<<16 | getAnalogOutput(REG+1))
# ============================================================
AO_SPEED_BACK_BASE = 0      # STP_SPEED_BACK_0 = 0, 1 motor = 2 regs
AO_SPEED_FORW_BASE = 16     # STP_SPEED_0      = 16
AO_POS_FOLLOW_BASE = 32     # STP_POS_FOLOW_0  = 32
AO_POS_MAX_BASE = 48        # STP_POS_MAX_0    = 48
AO_POS_CUR_BASE = 64        # STP_POS_CUR_0    = 64  (READ)

COIL_DIR_BASE = 1           # STP_DIR_0 = 1
COIL_MOTOR_ON_BASE = 9      # STP_ON_0  = 9
COIL_CALIB_BASE = 35        # STP_Calibration_0 = 35

COIL_CALIBRATION_GLOBAL = 17
COIL_CALIBRATION_RETR = 18

COIL_VALVE_STATE_BASE = 19  # VALVE_STATE_0 = 19
COIL_VALVE_MODE_BASE = 27   # VALVE_MODE_0  = 27


# ---------------------------
# Helpers: pack/unpack INT32 <-> U16 registers (HI, LO)
# ---------------------------
def _u16(x: int) -> int:
    return int(x) & 0xFFFF


def pack_int32_hi_lo(values: list[int]) -> list[int]:
    """Pack list of int32/uint32 into Modbus U16 registers [HI,LO]..."""
    regs: list[int] = []
    for v in values:
        v32 = int(v) & 0xFFFFFFFF
        regs.append((v32 >> 16) & 0xFFFF)
        regs.append(v32 & 0xFFFF)
    return regs


def unpack_int32_hi_lo(regs: list[int]) -> list[int]:
    """Unpack Modbus U16 registers [HI,LO]... into list of uint32."""
    if len(regs) % 2 != 0:
        regs = regs[:-1]
    out: list[int] = []
    for i in range(0, len(regs), 2):
        hi = int(regs[i]) & 0xFFFF
        lo = int(regs[i + 1]) & 0xFFFF
        out.append((hi << 16) | lo)
    return out


class ModbusLogger(object):
    @classmethod
    def log_packet(cls, sender: str, action: str, function: str, address: int, values: list[Any]):
        logging.debug(
            "%s | {sender}}: %s",
            print_timestamp(),
            format_modbus_log("{action}}", "{function}", address, values),
        )


class PumpDevice(object):
    syringes: list[Syringe]

    def __init__(self, config_dict):
        self.config = config_dict
        self.pumps_count = len(config_dict["channels"])

        self.syringes = []
        self.active_syringes = []

        i = 0
        for pump_config in config_dict["channels"]:
            if not pump_config["disabled"]:
                s = Syringe(str(i), pump_config["syringe"])
                self.syringes.append(s)
                self.active_syringes.append(1)
            else:
                self.syringes.append(None)
                self.active_syringes.append(0)
            i += 1

    def ValidatePosition(self, idx: int, position: int) -> int:
        validated_position = position
        if self.syringes[idx] is None:
            return 0
        if position > self.syringes[idx].parameters.MaxTick:
            validated_position = self.syringes[idx].parameters.MaxTick
        if position < 0:
            validated_position = 0
        return validated_position

    def SetSyringeNames(self, name_dict: dict[int, str]):
        for i, s in enumerate(self.syringes):
            if s is None:
                continue
            if i in name_dict:
                s.Name = name_dict[i]

    def About(self):
        print("count: ", self.pumps_count)
        for i, s in enumerate(self.syringes):
            print(i, s)

    def convertVolumeToTiks(self, pos, volume):
        return round(volume / self.syringes[pos].parameters.VolumeTick)

    def convertTiksToVolume(self, pos, tiks):
        return tiks * self.syringes[pos].parameters.VolumeTick

    # ---------------------------
    # Build full-array values for all pumps
    # ---------------------------
    def _get_speed_intervals_all(self) -> list[int]:
        """Return speed interval (delay) for each pump (INT32 value)."""
        out = [0] * self.pumps_count
        for i in range(self.pumps_count):
            if self.active_syringes[i] == 1 and self.syringes[i] is not None:
                # Syringe.SpeedToInterval returns int (delay). Keep current Speed if set.
                sp = self.syringes[i].Speed
                if sp is None or sp <= 0:
                    # default interval from parameters
                    out[i] = int(self.syringes[i].parameters.IntervalForward)
                else:
                    out[i] = int(self.syringes[i].SpeedToInterval(sp))
        return out

    def _get_valve_mode_all(self, manual: bool) -> list[bool]:
        return [bool(manual)] * self.pumps_count

    def _get_motor_on_all(self) -> list[bool]:
        out = [False] * self.pumps_count
        for i in range(self.pumps_count):
            if self.active_syringes[i] == 1 and self.syringes[i] is not None:
                out[i] = bool(self.syringes[i].MotorWork)
        return out

    # ============================================================
    # Prepare methods: return packets list [(is_coil, addr, values)]
    # is_coil: True -> write_coils, False -> write_registers(U16)
    # ============================================================
    def PrepareValveSetManual(self, a: int) -> list[tuple[bool, int, list[Any]]]:
        manual = True if a == 1 else False
        # In firmware: valve mode coils 27..34
        return [(True, COIL_VALVE_MODE_BASE, self._get_valve_mode_all(manual))]

    def PrepareValveChange(self, pos, action) -> list[tuple[bool, int, list[Any]]]:
        if isinstance(pos, int):
            pos, action = [pos], [action]
        states = [False] * self.pumps_count
        for i in range(self.pumps_count):
            if self.active_syringes[i] == 1 and self.syringes[i] is not None:
                states[i] = bool(self.syringes[i].ValveMode)
        for p in pos:
            a = action[pos.index(p)]
            states[p] = True if a == 1 else False
            if self.syringes[p] is not None:
                if states[p]:
                    self.syringes[p].valveON()
                else:
                    self.syringes[p].valveOFF()
        return [(True, COIL_VALVE_STATE_BASE, states)]

    def PrepareValveChangeOne(self, pos: int, action: int) -> list[tuple[bool, int, list[Any]]]:
        state = True if action == 1 else False
        if self.syringes[pos] is not None:
            if state:
                self.syringes[pos].valveON()
            else:
                self.syringes[pos].valveOFF()
        return [(True, COIL_VALVE_STATE_BASE + pos, [state])]

    def PrepareStart(self, pos="all") -> list[tuple[bool, int, list[Any]]]:
        if pos == "all":
            pos = list(range(self.pumps_count))
        if isinstance(pos, int):
            pos = [pos]
        states = self._get_motor_on_all()
        for p in pos:
            if self.active_syringes[p] == 1 and self.syringes[p] is not None:
                self.syringes[p].start()
                states[p] = True
        # Motor ON coils 9..16
        return [(True, COIL_MOTOR_ON_BASE, states)]

    def PrepareStop(self, pos="all") -> list[tuple[bool, int, list[Any]]]:
        if pos == "all":
            pos = list(range(self.pumps_count))
        if isinstance(pos, int):
            pos = [pos]
        states = self._get_motor_on_all()
        for p in pos:
            if self.active_syringes[p] == 1 and self.syringes[p] is not None:
                self.syringes[p].stop()
                states[p] = False
        return [(True, COIL_MOTOR_ON_BASE, states)]

    def PrepareStartOne(self, pos: int) -> list[tuple[bool, int, list[Any]]]:
        return self.PrepareStart(pos)

    def PrepareStopOne(self, pos: int) -> list[tuple[bool, int, list[Any]]]:
        return self.PrepareStop(pos)

    def PrepareSetSpeed(self, pos, speed) -> list[tuple[bool, int, list[Any]]]:
        """Set speed (forward+back blocks). speed is in Syringe units; we convert to interval/delay INT32."""
        if isinstance(pos, int):
            pos, speed = [pos], [speed]

        # update syringe objects speed
        for p in pos:
            if self.active_syringes[p] == 1 and self.syringes[p] is not None:
                self.syringes[p].Speed = float(speed[pos.index(p)])

        delays = self._get_speed_intervals_all()  # INT32 delays
        regs = pack_int32_hi_lo([int(x) for x in delays])  # U16

        # Firmware has two blocks: BACK at 0, FORW at 16
        # We write same delay to both unless you want different back speed.
        return [
            (False, AO_SPEED_BACK_BASE, regs),
            (False, AO_SPEED_FORW_BASE, regs),
        ]

    def PrepareSetSpeedOne(self, pos: int, speed: float) -> list[tuple[bool, int, list[Any]]]:
        return self.PrepareSetSpeed(pos, speed)

    def PrepareSetSpeedAndVolumeToMove(self, current_positions: list[int], pos, speed, volumes) -> list[tuple[bool, int, list[Any]]]:
        """Set speed + target position (FOLLOW)."""
        if isinstance(pos, int):
            pos, speed, volumes = [pos], [speed], [volumes]

        # set speeds
        for p in pos:
            if self.active_syringes[p] == 1 and self.syringes[p] is not None:
                self.syringes[p].Speed = float(speed[pos.index(p)])

        delays = self._get_speed_intervals_all()
        regs_speed = pack_int32_hi_lo([int(x) for x in delays])

        # build target positions array in TICKS
        tgt_ticks = [0] * self.pumps_count
        for i in range(self.pumps_count):
            if self.active_syringes[i] == 1 and self.syringes[i] is not None:
                if i in pos:
                    v = float(volumes[pos.index(i)])
                    t = self.convertVolumeToTiks(i, v)
                    t = self.ValidatePosition(i, t)
                    self.syringes[i].VolumeWork = self.convertTiksToVolume(i, t)
                    tgt_ticks[i] = int(t)
                else:
                    # keep current
                    tgt_ticks[i] = int(current_positions[i])
            else:
                tgt_ticks[i] = 0

        regs_pos = pack_int32_hi_lo([int(x) for x in tgt_ticks])

        return [
            (False, AO_SPEED_BACK_BASE, regs_speed),
            (False, AO_SPEED_FORW_BASE, regs_speed),
            (False, AO_POS_FOLLOW_BASE, regs_pos),
        ]

    def PrepareSetSpeedAndVolumeToMoveOne(self, current_position: int, pos: int, speed: float, volume: float) -> list[tuple[bool, int, list[Any]]]:
        return self.PrepareSetSpeedAndVolumeToMove([current_position] * self.pumps_count, [pos], [speed], [volume])

    def PrepareSetVolumeToMove(self, current_positions, pos, volumes) -> list[tuple[bool, int, list[Any]]]:
        if isinstance(pos, int):
            pos, volumes = [pos], [volumes]

        tgt_ticks = [0] * self.pumps_count
        for i in range(self.pumps_count):
            if self.active_syringes[i] == 1 and self.syringes[i] is not None:
                if i in pos:
                    v = float(volumes[pos.index(i)])
                    t = self.convertVolumeToTiks(i, v)
                    t = self.ValidatePosition(i, t)
                    self.syringes[i].VolumeWork = self.convertTiksToVolume(i, t)
                    tgt_ticks[i] = int(t)
                else:
                    tgt_ticks[i] = int(current_positions[i])
            else:
                tgt_ticks[i] = 0

        regs_pos = pack_int32_hi_lo([int(x) for x in tgt_ticks])
        return [(False, AO_POS_FOLLOW_BASE, regs_pos)]

    def PrepareSetVolumeToMoveOne(self, current_position: int, pos: int, volume: float) -> list[tuple[bool, int, list[Any]]]:
        return self.PrepareSetVolumeToMove([current_position] * self.pumps_count, [pos], [volume])

    def PrepareCalibrate(self, pos="all active") -> tuple[list[int], list[int], list[tuple[bool, int, list[Any]]]]:
        """Return (pos_list, speeds_list, packets). Speeds are converted and written by PrepareSetSpeed."""
        if isinstance(pos, int):
            pos = [pos]
        if pos == "all active":
            pos = [i for i in range(self.pumps_count) if self.active_syringes[i] == 1 and self.syringes[i] is not None]

        # speeds: your old heuristic оставим, но движение калибровки в прошивке зависит от enable + flag
        speeds: list[int] = []
        for p in pos:
            s = self.syringes[p]
            speeds.append(round(s.parameters.MaxTick * s.parameters.VolumeTick / 20))

        # maxpos block must be written to AO_POS_MAX_BASE (48..)
        maxpos = [0] * self.pumps_count
        for i in range(self.pumps_count):
            if self.active_syringes[i] == 1 and self.syringes[i] is not None:
                maxpos[i] = int(self.syringes[i].parameters.MaxTick)
            else:
                maxpos[i] = 0
        regs_max = pack_int32_hi_lo(maxpos)

        # per-motor calibration coils 35..42
        cal_flags = [False] * self.pumps_count
        for p in pos:
            cal_flags[p] = True

        packets: list[tuple[bool, int, list[Any]]] = [
            # ensure motors ON for selected pumps (often required)
            (True, COIL_MOTOR_ON_BASE, self._motor_on_with_pos(pos, True)),
            # write max positions
            (False, AO_POS_MAX_BASE, regs_max),
            # start calibration flags
            (True, COIL_CALIB_BASE, cal_flags),
        ]
        return pos, speeds, packets

    def _motor_on_with_pos(self, pos_list: list[int], state: bool) -> list[bool]:
        states = self._get_motor_on_all()
        for p in pos_list:
            if self.active_syringes[p] == 1 and self.syringes[p] is not None:
                states[p] = bool(state)
                if state:
                    self.syringes[p].start()
                else:
                    self.syringes[p].stop()
        return states

    def PrepareCalibrateOne(self, pos: int) -> tuple[int, list[tuple[bool, int, list[Any]]]]:
        speed = round(self.syringes[pos].parameters.MaxTick * self.syringes[pos].parameters.VolumeTick / 20)

        maxpos = [0] * self.pumps_count
        for i in range(self.pumps_count):
            if self.active_syringes[i] == 1 and self.syringes[i] is not None:
                maxpos[i] = int(self.syringes[i].parameters.MaxTick)
        regs_max = pack_int32_hi_lo(maxpos)

        cal_one = [False] * self.pumps_count
        cal_one[pos] = True

        packets = [
            (True, COIL_MOTOR_ON_BASE, self._motor_on_with_pos([pos], True)),
            (False, AO_POS_MAX_BASE, regs_max),
            (True, COIL_CALIB_BASE, cal_one),
        ]
        return speed, packets

    def DumpStatus(self, positions: list[int]) -> str:
        info = []
        for i in range(self.pumps_count):
            if self.active_syringes[i] == 1 and self.syringes[i] is not None:
                syr = self.syringes[i].GetStatus()
                cur_vol = self.convertTiksToVolume(i, positions[i])
                sec = abs(syr[4] - cur_vol) / (syr[3] + 0.0001)
                minutes = int(sec // 60)
                remaining_seconds = sec % 60
                t = f"{minutes:02d}:{remaining_seconds:05.2f}"
                info.append([i, syr[0], syr[1], syr[2], cur_vol, syr[3], syr[4], cur_vol - syr[4], t])
            else:
                info.append([0] * 9)

        st = [
            "насос       :|",
            "реагент     :|",
            "статус      :|",
            "клапан      :|",
            "V текущий   :|",
            "скорость    :|",
            "V назначения:|",
            "V оставшийся:|",
            "время       :|",
        ]
        for i in range(self.pumps_count):
            st[0] += "%10s|" % info[i][0]
            st[1] += "%10s|" % info[i][1]
            st[2] += "%10s|" % ("ON" if info[i][2] else "OFF")
            st[3] += "%10s|" % ("ON" if info[i][3] else "OFF")
            st[4] += "%10.2f|" % info[i][4]
            st[5] += "%10.2f|" % info[i][5]
            st[6] += "%10.2f|" % info[i][6]
            st[7] += "%10.2f|" % info[i][7]
            st[8] += "%10s|" % info[i][8]
        return "\n".join(st) + "\n"


class PumpClient:
    device: PumpDevice

    def __init__(self, config_dict):
        self.device = PumpDevice(config_dict)
        self.com = "none"
        self.messenger = PortCommunicator()

    def SetSyringeNames(self, name_dict: dict[int, str]):
        self.device.SetSyringeNames(name_dict)

    def About(self):
        return self.device.About()

    async def FindPortAndConnect(self, port_glob="*") -> bool:
        await self.messenger.find_port(port_glob)
        if self.messenger.port_name != "":
            self.com = self.messenger.port_name
            await self.ValveSetManual(1)
            return True
        return False

    async def Connect(self, com):
        await self.messenger.connect(com)
        self.com = self.messenger.port_name
        await self.ValveSetManual(1)

    def Disconnect(self):
        self.messenger.disconnect()
        self.com = self.messenger.port_name

    # ---------------------------
    # Low-level I/O
    # ---------------------------
    async def write_coils(self, address: int, values: list[Any]):
        ModbusLogger.log_packet("CLIENT", "write", "coils", address, values)
        return await self.messenger.write_coils(address, values, device_id=1)

    async def write_registers_u16(self, address: int, values_u16: list[int]):
        # ensure 0..65535
        vals = [_u16(v) for v in values_u16]
        ModbusLogger.log_packet("CLIENT", "write", "holding_u16", address, vals)
        return await self.messenger.write_registers(address, vals, device_id=1)

    async def read_holding_u16(self, address: int, count_u16: int) -> list[int]:
        rr = await self.messenger.read_holding_registers(address=address, count=count_u16, device_id=1)
        regs = getattr(rr, "registers", None)
        if regs is None:
            return []
        return [int(x) & 0xFFFF for x in regs]

    async def read_coils(self, address: int, count: int) -> list[bool]:
        rr = await self.messenger.read_coils(address=address, count=count, device_id=1)
        bits = getattr(rr, "bits", None)
        if bits is None:
            return []
        return [bool(x) for x in bits[:count]]

    async def read_discrete_inputs(self, address: int, count: int) -> list[bool]:
        rr = await self.messenger.read_discrete_inputs(address=address, count=count, device_id=1)
        bits = getattr(rr, "bits", None)
        if bits is None:
            return []
        return [bool(x) for x in bits[:count]]

    # ---------------------------
    # High-level API
    # ---------------------------
    async def ValveSetManual(self, a: int):
        packets = self.device.PrepareValveSetManual(a)
        for is_coil, addr, vals in packets:
            if is_coil:
                await self.write_coils(addr, vals)

    async def ValveChange(self, pos, action):
        packets = self.device.PrepareValveChange(pos, action)
        for is_coil, addr, vals in packets:
            await self.write_coils(addr, vals)

    async def ValveChangeOne(self, pos: int, action: int):
        packets = self.device.PrepareValveChangeOne(pos, action)
        for is_coil, addr, vals in packets:
            await self.write_coils(addr, vals)

    async def Start(self, pos="all"):
        packets = self.device.PrepareStart(pos)
        for is_coil, addr, vals in packets:
            await self.write_coils(addr, vals)

    async def StartOne(self, pos: int):
        await self.Start(pos)

    async def Stop(self, pos="all"):
        packets = self.device.PrepareStop(pos)
        for is_coil, addr, vals in packets:
            await self.write_coils(addr, vals)

    async def StopOne(self, pos: int):
        await self.Stop(pos)

    async def SetSpeed(self, pos, speed):
        packets = self.device.PrepareSetSpeed(pos, speed)
        for is_coil, addr, vals in packets:
            await self.write_registers_u16(addr, vals)

    async def SetSpeedOne(self, pos: int, speed: float):
        await self.SetSpeed(pos, speed)

    async def GetPositionsMF(self) -> list[int]:
        regs = await self.read_holding_u16(AO_POS_CUR_BASE, self.device.pumps_count * 2)
        return unpack_int32_hi_lo(regs)

    async def GetPositionMF(self, pos: int) -> int:
        regs = await self.read_holding_u16(AO_POS_CUR_BASE + pos * 2, 2)
        v = unpack_int32_hi_lo(regs)
        return v[0] if v else 0

    async def SetSpeedAndVolumeToMove(self, pos, speed, volumes):
        current_positions = await self.GetPositionsMF()
        packets = self.device.PrepareSetSpeedAndVolumeToMove(current_positions, pos, speed, volumes)
        for is_coil, addr, vals in packets:
            await self.write_registers_u16(addr, vals)

    async def SetVolumeToMove(self, pos, volumes):
        current_positions = await self.GetPositionsMF()
        packets = self.device.PrepareSetVolumeToMove(current_positions, pos, volumes)
        for is_coil, addr, vals in packets:
            await self.write_registers_u16(addr, vals)

    async def Calibrate(self, pos="all active"):
        pos_list, speeds, packets = self.device.PrepareCalibrate(pos)
        # 1) set speeds (writes BACK and FORW blocks)
        await self.SetSpeed(pos_list, speeds)
        # 2) write maxpos, enable motors, set calibration coils
        for is_coil, addr, vals in packets:
            if is_coil:
                await self.write_coils(addr, vals)
            else:
                await self.write_registers_u16(addr, vals)

    async def CalibrateOne(self, pos: int):
        speed, packets = self.device.PrepareCalibrateOne(pos)
        await self.SetSpeedOne(pos, speed)
        for is_coil, addr, vals in packets:
            if is_coil:
                await self.write_coils(addr, vals)
            else:
                await self.write_registers_u16(addr, vals)

    async def StatusThread(self) -> str:
        positions = await self.GetPositionsMF()
        return self.device.DumpStatus(positions)


class SyncPumpClient:
    device: PumpDevice

    def __init__(self, config_dict):
        self.device = PumpDevice(config_dict)
        self.com = "none"
        self.messenger = SyncPortCommunicator()

    def SetSyringeNames(self, name_dict: dict[int, str]):
        self.device.SetSyringeNames(name_dict)

    def About(self):
        return self.device.About()

    def FindPortAndConnect(self, port_glob="*") -> bool:
        self.messenger.find_port(port_glob)
        if self.messenger.port_name != "":
            self.com = self.messenger.port_name
            self.ValveSetManual(1)
            return True
        return False

    def Connect(self, com):
        self.messenger.connect(com)
        self.com = self.messenger.port_name
        self.ValveSetManual(1)

    def Disconnect(self):
        self.messenger.disconnect()
        self.com = self.messenger.port_name

    def write_coils(self, address: int, values: list[Any]):
        return self.messenger.write_coils(address, values, device_id=1)

    def write_registers_u16(self, address: int, values_u16: list[int]):
        vals = [_u16(v) for v in values_u16]
        return self.messenger.write_registers(address, vals, device_id=1)

    def read_holding_u16(self, address: int, count_u16: int) -> list[int]:
        rr = self.messenger.read_holding_registers(address=address, count=count_u16, device_id=1)
        regs = getattr(rr, "registers", None)
        if regs is None:
            return []
        return [int(x) & 0xFFFF for x in regs]

    def ValveSetManual(self, a: int):
        packets = self.device.PrepareValveSetManual(a)
        for is_coil, addr, vals in packets:
            self.write_coils(addr, vals)

    def Start(self, pos="all"):
        packets = self.device.PrepareStart(pos)
        for is_coil, addr, vals in packets:
            self.write_coils(addr, vals)

    def Stop(self, pos="all"):
        packets = self.device.PrepareStop(pos)
        for is_coil, addr, vals in packets:
            self.write_coils(addr, vals)

    def SetSpeed(self, pos, speed):
        packets = self.device.PrepareSetSpeed(pos, speed)
        for is_coil, addr, vals in packets:
            self.write_registers_u16(addr, vals)

    def GetPositionsMF(self) -> list[int]:
        regs = self.read_holding_u16(AO_POS_CUR_BASE, self.device.pumps_count * 2)
        return unpack_int32_hi_lo(regs)

    def Calibrate(self, pos="all active"):
        pos_list, speeds, packets = self.device.PrepareCalibrate(pos)
        self.SetSpeed(pos_list, speeds)
        for is_coil, addr, vals in packets:
            if is_coil:
                self.write_coils(addr, vals)
            else:
                self.write_registers_u16(addr, vals)
