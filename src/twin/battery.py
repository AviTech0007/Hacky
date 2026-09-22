"""Battery digital twin.

Deliberately kept *linear* so that the physics the simulator enforces are the
same physics the MILP reasons about. If you make this model fancier
(temperature, SoC-dependent efficiency), mirror the change in optimizer.py or
the plan and the reality will drift apart.

Degradation is modelled as two additive linear terms:
  1. throughput wear  - every kWh in or out costs cycle_cost_per_kwh
  2. depth stress     - every kWh-hour spent below the comfort SoC is penalised
Plus a rate-stress term for charging/discharging harder than the soft C-rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.config import BatteryConfig


@dataclass
class BatteryTelemetry:
    soc_kwh: List[float] = field(default_factory=list)
    charged_kwh: List[float] = field(default_factory=list)
    discharged_kwh: List[float] = field(default_factory=list)


class Battery:
    """Stateful battery you can step hour by hour."""

    def __init__(self, cfg: BatteryConfig, soc_kwh: float | None = None):
        self.cfg = cfg
        self.soc_kwh = cfg.soc_init_kwh if soc_kwh is None else soc_kwh
        self.throughput_kwh = 0.0
        self.depth_stress_kwh_h = 0.0
        self.rate_stress_kwh = 0.0
        self.min_soc_kwh = self.soc_kwh
        self.telemetry = BatteryTelemetry()

    # --- physical limits -------------------------------------------------
    def headroom_kwh(self, dt_h: float = 1.0) -> float:
        """How much more energy can go in this step."""
        by_rate = self.cfg.max_charge_kw * dt_h
        by_space = (self.cfg.soc_max_kwh - self.soc_kwh) / self.cfg.charge_efficiency
        return max(0.0, min(by_rate, by_space))

    def available_kwh(self, dt_h: float = 1.0) -> float:
        """How much energy can be delivered to the load this step."""
        by_rate = self.cfg.max_discharge_kw * dt_h
        by_charge = (self.soc_kwh - self.cfg.soc_min_kwh) * self.cfg.discharge_efficiency
        return max(0.0, min(by_rate, by_charge))

    # --- stepping --------------------------------------------------------
    def step(self, charge_kwh: float, discharge_kwh: float, dt_h: float = 1.0) -> tuple[float, float]:
        """Apply one hour of charge/discharge, clipped to what is physical.

        Returns the amounts actually accepted/delivered.
        """
        charge_kwh = min(max(0.0, charge_kwh), self.headroom_kwh(dt_h))
        discharge_kwh = min(max(0.0, discharge_kwh), self.available_kwh(dt_h))

        self.soc_kwh += charge_kwh * self.cfg.charge_efficiency
        self.soc_kwh -= discharge_kwh / self.cfg.discharge_efficiency

        # accounting
        self.throughput_kwh += charge_kwh + discharge_kwh
        self.min_soc_kwh = min(self.min_soc_kwh, self.soc_kwh)
        self.depth_stress_kwh_h += max(0.0, self.cfg.soc_comfort_kwh - self.soc_kwh)
        soft = self.cfg.soft_rate_kw * dt_h
        self.rate_stress_kwh += max(0.0, charge_kwh - soft) + max(0.0, discharge_kwh - soft)

        self.telemetry.soc_kwh.append(self.soc_kwh)
        self.telemetry.charged_kwh.append(charge_kwh)
        self.telemetry.discharged_kwh.append(discharge_kwh)
        return charge_kwh, discharge_kwh

    # --- reporting -------------------------------------------------------
    @property
    def soc_frac(self) -> float:
        return self.soc_kwh / self.cfg.capacity_kwh

    @property
    def equivalent_full_cycles(self) -> float:
        return self.throughput_kwh / (2 * self.cfg.capacity_kwh)

    def degradation_cost(self, weights: dict) -> float:
        """Same formula the optimizer minimises, evaluated on what happened."""
        return (
            weights.get("throughput", 1.0) * self.cfg.cycle_cost_per_kwh * self.throughput_kwh
            + weights.get("deep_discharge", 0.0) * self.depth_stress_kwh_h
            + weights.get("rate_stress", 0.0) * self.rate_stress_kwh
        )
