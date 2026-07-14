# ======================================================================
# smart_pump/communicator.py  ( pymodbus 3.x compatible)
# ======================================================================
# Works with:
#   pymodbus.client.serial.AsyncModbusSerialClient  (async)
#   pymodbus.client.serial.ModbusSerialClient       (sync)
#
# Key fixes:
# 1) Use device_id=... (NOT slave=..., NOT unit=...)     here Anton
# 2) Await async client methods
# 3) Add small gap between requests (USB-CDC friendly)
# 4) Print debug info on connect and each request  (OPTIONAL: self.debug)
# ======================================================================

import asyncio
import time
import serial
import serial.tools.list_ports

from pymodbus.client.serial import AsyncModbusSerialClient, ModbusSerialClient


def _now_ms() -> int:
    return int(time.time() * 1000)


class PortCommunicator:
    """
    Async Modbus RTU over Serial/USB-CDC, pymodbus 3.x.
    Used by PumpClient (async).
    """

    def __init__(self):
        self.port = None  # underlying pyserial port (if accessible)
        self.port_name = ""
        self.modbus_client: AsyncModbusSerialClient | None = None

        # defaults (you can override after connect)
        self.baudrate = 115200
        self.bytesize = 8
        self.parity = "N"
        self.stopbits = 1
        self.timeout = 0.7
        self.write_timeout = 0.7

        # request pacing (important for USB-CDC)
        self.min_gap_s = 0.01
        self._last_io_ms = 0

        # IMPORTANT: disable prints by default
        self.debug = False

    async def _sleep_gap(self):
        dt = (_now_ms() - self._last_io_ms) / 1000.0
        if dt < self.min_gap_s:
            await asyncio.sleep(self.min_gap_s - dt)

    def _touch(self):
        self._last_io_ms = _now_ms()

    async def find_port(self, port_glob="*"):
        """
        Find a port by probing (optional). Here: just pick the first available
        if only one exists; otherwise keep empty. You can extend probing logic.
        """
        ports = list(serial.tools.list_ports.comports())
        if self.debug:
            print("[MF] available ports:", [p.device for p in ports])

        if len(ports) == 1:
            await self.connect(ports[0].device)
            return

        # If multiple ports, do not guess silently.
        # Keep empty and let user specify.
        self.port_name = ""
        self.modbus_client = None

    async def connect(self, com: str):
        self.port_name = com
        if self.debug:
            print(f"[MF] CONNECT to {com}")

        # Create async client
        self.modbus_client = AsyncModbusSerialClient(
            port=com,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
            timeout=self.timeout,
        )

        ok = await self.modbus_client.connect()
        if self.debug:
            print(f"[MF] modbus_client.connect() -> {ok}")
            print(
                f"[MF] SERIAL SETTINGS: baud={self.baudrate}, bytesize={self.bytesize}, "
                f"parity={self.parity}, stopbits={self.stopbits}, timeout={self.timeout}, "
                f"write_timeout={self.write_timeout}"
            )

        # Try to access pyserial port to set write_timeout & flush buffers
        try:
            self.port = getattr(self.modbus_client, "transport", None)
            # In pymodbus 3.x, internal port is usually at:
            # self.modbus_client.comm_params or self.modbus_client.transport
            # We try a few known places:
            if self.port is None:
                self.port = getattr(self.modbus_client, "serial", None)
            if self.port is None:
                self.port = getattr(self.modbus_client, "socket", None)

            # If it's a real Serial instance, set write_timeout and flush
            if hasattr(self.modbus_client, "protocol") and hasattr(self.modbus_client.protocol, "transport"):
                tr = self.modbus_client.protocol.transport
                if tr is not None and hasattr(tr, "serial"):
                    self.port = tr.serial

            if self.port is not None and hasattr(self.port, "write_timeout"):
                self.port.write_timeout = self.write_timeout
            if self.port is not None and hasattr(self.port, "reset_input_buffer"):
                self.port.reset_input_buffer()
            if self.port is not None and hasattr(self.port, "reset_output_buffer"):
                self.port.reset_output_buffer()
            if self.debug:
                print("[MF] buffers flushed")
        except Exception as e:
            if self.debug:
                print("[MF] flush/set write_timeout skipped:", e)

        self._touch()
        return ok

    def disconnect(self):
        try:
            if self.modbus_client is not None:
                self.modbus_client.close()
        finally:
            self.modbus_client = None
            self.port = None
            self.port_name = ""

    # -------------------------
    # Async Modbus wrappers
    # -------------------------
    async def read_holding_registers(self, address: int, count: int, device_id: int = 1):
        await self._sleep_gap()
        rr = await self.modbus_client.read_holding_registers(
            address=address, count=count, device_id=device_id
        )
        self._touch()
        return rr

    async def read_input_registers(self, address: int, count: int, device_id: int = 1):
        await self._sleep_gap()
        rr = await self.modbus_client.read_input_registers(
            address=address, count=count, device_id=device_id
        )
        self._touch()
        return rr

    async def write_registers(self, address: int, values, device_id: int = 1):
        await self._sleep_gap()
        rr = await self.modbus_client.write_registers(
            address=address, values=values, device_id=device_id
        )
        self._touch()
        return rr

    async def read_coils(self, address: int, count: int, device_id: int = 1):
        await self._sleep_gap()
        rr = await self.modbus_client.read_coils(
            address=address, count=count, device_id=device_id
        )
        self._touch()
        return rr

    async def write_coils(self, address: int, values, device_id: int = 1):
        await self._sleep_gap()
        rr = await self.modbus_client.write_coils(
            address=address, values=values, device_id=device_id
        )
        self._touch()
        return rr

    async def read_discrete_inputs(self, address: int, count: int, device_id: int = 1):
        await self._sleep_gap()
        rr = await self.modbus_client.read_discrete_inputs(
            address=address, count=count, device_id=device_id
        )
        self._touch()
        return rr


class SyncPortCommunicator:
    """
    Sync Modbus RTU over Serial/USB-CDC, pymodbus 3.x.
    Used by SyncPumpClient (sync).
    """

    def __init__(self):
        self.port = None
        self.port_name = ""
        self.modbus_client: ModbusSerialClient | None = None

        self.baudrate = 115200
        self.bytesize = 8
        self.parity = "N"
        self.stopbits = 1
        self.timeout = 0.7
        self.write_timeout = 0.7

        self.min_gap_s = 0.01
        self._last_io_ms = 0

        # IMPORTANT: disable prints by default
        self.debug = False

    def _sleep_gap(self):
        dt = (_now_ms() - self._last_io_ms) / 1000.0
        if dt < self.min_gap_s:
            time.sleep(self.min_gap_s - dt)

    def _touch(self):
        self._last_io_ms = _now_ms()

    def find_port(self, port_glob="*"):
        ports = list(serial.tools.list_ports.comports())
        if self.debug:
            print("[MF] available ports:", [p.device for p in ports])

        if len(ports) == 1:
            self.connect(ports[0].device)
            return

        self.port_name = ""
        self.modbus_client = None

    def connect(self, com: str):
        self.port_name = com
        if self.debug:
            print(f"[MF] CONNECT (sync) to {com}")

        self.modbus_client = ModbusSerialClient(
            port=com,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
            timeout=self.timeout,
        )

        ok = self.modbus_client.connect()
        if self.debug:
            print(f"[MF] modbus_client.connect() -> {ok}")
            print(
                f"[MF] SERIAL SETTINGS: baud={self.baudrate}, bytesize={self.bytesize}, "
                f"parity={self.parity}, stopbits={self.stopbits}, timeout={self.timeout}, "
                f"write_timeout={self.write_timeout}"
            )

        # Access pyserial port to set write_timeout & flush
        try:
            self.port = getattr(self.modbus_client, "socket", None)  # sometimes Serial instance
            if self.port is None:
                self.port = getattr(self.modbus_client, "serial", None)

            if self.port is not None and hasattr(self.port, "write_timeout"):
                self.port.write_timeout = self.write_timeout
            if self.port is not None and hasattr(self.port, "reset_input_buffer"):
                self.port.reset_input_buffer()
            if self.port is not None and hasattr(self.port, "reset_output_buffer"):
                self.port.reset_output_buffer()
            if self.debug:
                print("[MF] buffers flushed")
        except Exception as e:
            if self.debug:
                print("[MF] flush/set write_timeout skipped:", e)

        self._touch()
        return ok

    def disconnect(self):
        try:
            if self.modbus_client is not None:
                self.modbus_client.close()
        finally:
            self.modbus_client = None
            self.port = None
            self.port_name = ""

    # -------------------------
    # Sync Modbus wrappers
    # -------------------------
    def read_holding_registers(self, address: int, count: int, device_id: int = 1):
        self._sleep_gap()
        rr = self.modbus_client.read_holding_registers(
            address=address, count=count, device_id=device_id
        )
        self._touch()
        return rr

    def read_input_registers(self, address: int, count: int, device_id: int = 1):
        self._sleep_gap()
        rr = self.modbus_client.read_input_registers(
            address=address, count=count, device_id=device_id
        )
        self._touch()
        return rr

    def write_registers(self, address: int, values, device_id: int = 1):
        self._sleep_gap()
        rr = self.modbus_client.write_registers(
            address=address, values=values, device_id=device_id
        )
        self._touch()
        return rr

    def read_coils(self, address: int, count: int, device_id: int = 1):
        self._sleep_gap()
        rr = self.modbus_client.read_coils(
            address=address, count=count, device_id=device_id
        )
        self._touch()
        return rr

    def write_coils(self, address: int, values, device_id: int = 1):
        self._sleep_gap()
        rr = self.modbus_client.write_coils(
            address=address, values=values, device_id=device_id
        )
        self._touch()
        return rr

    def read_discrete_inputs(self, address: int, count: int, device_id: int = 1):
        self._sleep_gap()
        rr = self.modbus_client.read_discrete_inputs(
            address=address, count=count, device_id=device_id
        )
        self._touch()
        return rr
