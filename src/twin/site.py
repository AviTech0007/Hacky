"""Digital twin / state manager.

The optimizer produces a *plan*. This module is what happens when the plan
meets reality: actual PV differs from the forecast, so the twin dispatches the
scheduled tasks, serves them from sun first, tops up or drains the pack, and
records what really occurred.

Keeping planning and execution separate is what lets you say, on stage, "the
coordinator planned X, the site delivered Y, and here is the gap."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import AppConfig
from src.coordinator.optimizer import Plan
from src.twin.battery import Battery
from src.twin.tasks import Task


@dataclass
class ExecutionResult:
    hours: int
    pv_actual_kwh: List[float]
    pv_to_load_kwh: List[float]
    pv_to_battery_kwh: List[float]
    battery_to_load_kwh: List[float]
    load_kwh: List[float]
    unserved_kwh: List[float]
    curtailed_kwh: List[float]
    soc_kwh: List[float]
    completed: Dict[str, bool] = field(default_factory=dict)
    battery: Optional[Battery] = None


class SiteTwin:
    """Holds live state for the edge node and steps it forward."""

    def __init__(self, cfg: AppConfig, soc_kwh: Optional[float] = None):
        self.cfg = cfg
        self.battery = Battery(cfg.battery, soc_kwh=soc_kwh)

    def execute(
        self,
        plan: Plan,
        tasks: List[Task],
        pv_actual_kwh: Optional[List[float]] = None,
        dt_h: float = 1.0,
    ) -> ExecutionResult:
        pv_actual = list(pv_actual_kwh or plan.pv_available_kwh)
        H = plan.horizon
        pv_actual = pv_actual[:H] + [0.0] * max(0, H - len(pv_actual))
        by_id = {t.id: t for t in tasks}

        pv_to_load, pv_to_batt, batt_to_load = [], [], []
        load_trace, unserved, curtailed = [], [], []
        soc_trace = [self.battery.soc_kwh]
        executed_slots: Dict[str, int] = {t.id: 0 for t in tasks}

        for t in range(H):
            running = plan.tasks_in_slot(t)
            load = self.cfg.site.base_load_kw * dt_h + sum(
                by_id[tid].power_kw * dt_h for tid in running if tid in by_id
            )

            direct = min(pv_actual[t], load)
            deficit = load - direct
            surplus = pv_actual[t] - direct

            # Follow the plan's charge/discharge intent, but never violate physics
            # and never let a scheduled task go dark if the pack can cover it.
            want_charge = max(plan.charge_kwh[t], 0.0)
            want_discharge = max(deficit, plan.discharge_kwh[t] if deficit > 0 else 0.0)
            charge = min(want_charge, surplus)
            charge_done, discharge_done = self.battery.step(charge, want_discharge, dt_h)

            served_by_batt = min(discharge_done, deficit)
            short = max(0.0, deficit - served_by_batt)

            for tid in running:
                executed_slots[tid] = executed_slots.get(tid, 0) + (0 if short > 1e-6 else 1)

            pv_to_load.append(direct)
            pv_to_batt.append(charge_done)
            batt_to_load.append(served_by_batt)
            load_trace.append(load)
            unserved.append(short)
            curtailed.append(max(0.0, surplus - charge_done))
            soc_trace.append(self.battery.soc_kwh)

        completed = {
            t.id: executed_slots.get(t.id, 0) >= t.duration_h for t in tasks
        }
        return ExecutionResult(
            hours=H,
            pv_actual_kwh=pv_actual,
            pv_to_load_kwh=pv_to_load,
            pv_to_battery_kwh=pv_to_batt,
            battery_to_load_kwh=batt_to_load,
            load_kwh=load_trace,
            unserved_kwh=unserved,
            curtailed_kwh=curtailed,
            soc_kwh=soc_trace,
            completed=completed,
            battery=self.battery,
        )
