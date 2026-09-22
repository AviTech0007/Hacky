"""Loads config.yaml once and hands typed sections to the rest of the pipeline.

Everything downstream reads its parameters from here, so there is exactly one
place to tune the demo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader so we do not need python-dotenv as a hard dep."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass
class SiteConfig:
    name: str
    latitude: float
    longitude: float
    timezone: str
    base_load_kw: float


@dataclass
class PVConfig:
    peak_power_kw: float
    derate: float
    temp_coeff_per_c: float
    ref_temp_c: float


@dataclass
class BatteryConfig:
    capacity_kwh: float
    soc_init_frac: float
    soc_min_frac: float
    soc_max_frac: float
    soc_comfort_frac: float
    soc_terminal_frac: float
    max_charge_kw: float
    max_discharge_kw: float
    soft_rate_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    cycle_cost_per_kwh: float

    # Convenience conversions from fractions to kWh, used by the optimizer.
    @property
    def soc_init_kwh(self) -> float:
        return self.soc_init_frac * self.capacity_kwh

    @property
    def soc_min_kwh(self) -> float:
        return self.soc_min_frac * self.capacity_kwh

    @property
    def soc_max_kwh(self) -> float:
        return self.soc_max_frac * self.capacity_kwh

    @property
    def soc_comfort_kwh(self) -> float:
        return self.soc_comfort_frac * self.capacity_kwh

    @property
    def soc_terminal_kwh(self) -> float:
        return self.soc_terminal_frac * self.capacity_kwh


@dataclass
class OptimizerConfig:
    horizon_hours: int
    time_limit_s: int
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class AppConfig:
    site: SiteConfig
    pv: PVConfig
    battery: BatteryConfig
    optimizer: OptimizerConfig
    workload: Dict[str, Any]
    explainer: Dict[str, Any]
    raw: Dict[str, Any]


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    _load_dotenv()
    raw = yaml.safe_load(Path(path).read_text())
    return AppConfig(
        site=SiteConfig(**raw["site"]),
        pv=PVConfig(**raw["pv"]),
        battery=BatteryConfig(**raw["battery"]),
        optimizer=OptimizerConfig(**raw["optimizer"]),
        workload=raw.get("workload", {}),
        explainer=raw.get("explainer", {}),
        raw=raw,
    )
