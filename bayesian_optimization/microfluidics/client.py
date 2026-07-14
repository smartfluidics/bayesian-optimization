"""Thin helpers around ``smart_pump.PumpClient`` (optional hardware extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def require_smart_pump():
    try:
        from smart_pump.pumps import PumpClient
        from smart_pump.syringe_config import ConfigReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Hardware helpers require smart_pump. Install with:\n"
            "  pip install --extra-index-url "
            "https://lab:iUULBNNmgslKVjaX35MA@pypi.stud.mmcs.sfedu.ru smart_pump\n"
            "or: pip install -r requirements-hardware.txt\n"
            "Then replace pumps.py / communicator.py — see README."
        ) from exc
    return PumpClient, ConfigReader


def create_pump_client(config_path: str | Path) -> Any:
    """Build a ``PumpClient`` from a YAML syringe config (same as lab notebooks)."""
    PumpClient, ConfigReader = require_smart_pump()
    config_obj = ConfigReader.read_config(str(config_path))
    return PumpClient(config_obj)


async def connect_mf(mf: Any, port: str | None = None) -> Any:
    """Connect MF on ``port`` or auto-find a port."""
    if port:
        await mf.Connect(port)
    else:
        await mf.FindPortAndConnect()
    return mf
