"""The strawman: what every other team will build.

Earliest-deadline-first, run-as-soon-as-possible, charge whenever there is
surplus sun, discharge whenever there is a deficit, and pause work only once
the pack is nearly flat. No foresight, no trade-offs.

It returns the same `Plan` object as the MILP so metrics.py and the dashboard
can compare them side by side without special-casing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.config import BatteryConfig, OptimizerConfig
from src.coordinator.optimizer import Plan
from src.twin.battery import Battery
from src.twin.tasks import Task


def greedy_plan(
    tasks: List[Task],
    pv_kwh: List[float],
    battery: BatteryConfig,
    opt: OptimizerConfig,
    base_load_kw: float,
    soc_init_kwh: Optional[float] = None,
    horizon: Optional[int] = None,
    dt_h: float = 1.0,
) -> Plan:
    H = horizon or min(opt.horizon_hours, len(pv_kwh))
    pv_kwh = list(pv_kwh[:H]) + [0.0] * max(0, H - len(pv_kwh))
    bat = Battery(battery, soc_kwh=soc_init_kwh)

    remaining: Dict[str, int] = {t.id: t.duration_h for t in tasks}
    schedule: Dict[str, List[int]] = {t.id: [] for t in tasks}
    by_deadline = sorted(tasks, key=lambda t: (t.deadline_h, -t.value))

    pv_used, charge, discharge, load, unserved = [], [], [], [], []
    soc_trace = [bat.soc_kwh]

    for t in range(H):
        slot_load = base_load_kw * dt_h
        supply = pv_kwh[t] + bat.available_kwh(dt_h)
        busy: set[str] = set()

        for tk in by_deadline:
            if remaining[tk.id] <= 0 or not (tk.release_h <= t <= tk.deadline_h):
                continue
            if tk.id in busy:
                continue
            # Non-preemptible jobs must start late enough to finish in one run.
            if not tk.preemptible and schedule[tk.id] and schedule[tk.id][-1] != t - 1:
                continue
            need = tk.power_kw * dt_h
            if slot_load + need <= supply:
                slot_load += need
                remaining[tk.id] -= 1
                schedule[tk.id].append(t)
                busy.add(tk.id)

        # Serve the load from PV first, then the battery.
        from_pv = min(pv_kwh[t], slot_load)
        deficit = slot_load - from_pv
        surplus = pv_kwh[t] - from_pv
        c, d = bat.step(charge_kwh=surplus, discharge_kwh=deficit, dt_h=dt_h)

        pv_used.append(from_pv + c)
        charge.append(c)
        discharge.append(d)
        load.append(slot_load)
        unserved.append(max(0.0, deficit - d))
        soc_trace.append(bat.soc_kwh)

    completed = {t.id: remaining[t.id] <= 0 for t in tasks}
    w = opt.weights
    return Plan(
        status="Greedy",
        objective=float("nan"),
        horizon=H,
        schedule=schedule,
        completed=completed,
        pv_available_kwh=pv_kwh,
        pv_used_kwh=pv_used,
        charge_kwh=charge,
        discharge_kwh=discharge,
        load_kwh=load,
        unserved_kwh=unserved,
        soc_kwh=soc_trace,
        cost_breakdown={
            "unserved": w["unserved"] * sum(unserved),
            "missed_tasks": w["missed_task"] * sum(t.value for t in tasks if not completed[t.id]),
            "battery_throughput": w["throughput"] * battery.cycle_cost_per_kwh * bat.throughput_kwh,
            "deep_discharge": w["deep_discharge"] * bat.depth_stress_kwh_h,
            "rate_stress": w["rate_stress"] * bat.rate_stress_kwh,
            "curtailment": w["curtailment"] * sum(
                max(0.0, pv_kwh[t] - pv_used[t]) for t in range(H)
            ),
        },
    )
