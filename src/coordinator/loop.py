"""Rolling-horizon (MPC) loop - the feature that makes this look like a system
rather than a one-shot solve.

Each hour: re-forecast, re-optimize over the remaining horizon with the *real*
current SoC and the tasks that are still outstanding, execute only the first
hour of the new plan, then advance. This is how real energy management systems
work, and it is a two-sentence talking point that separates you from a batch
solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.config import AppConfig
from src.coordinator.forecast import apply_forecast_error
from src.coordinator.optimizer import Plan, optimize
from src.twin.battery import Battery
from src.twin.tasks import Task, shift_tasks


@dataclass
class RollingResult:
    hours: int
    soc_kwh: List[float]
    pv_actual_kwh: List[float] = field(default_factory=list)
    pv_to_load_kwh: List[float] = field(default_factory=list)
    charge_kwh: List[float] = field(default_factory=list)
    discharge_kwh: List[float] = field(default_factory=list)
    load_kwh: List[float] = field(default_factory=list)
    unserved_kwh: List[float] = field(default_factory=list)
    running: List[List[str]] = field(default_factory=list)
    replans: List[Plan] = field(default_factory=list)
    completed: Dict[str, bool] = field(default_factory=dict)


def run_rolling_horizon(
    cfg: AppConfig,
    tasks: List[Task],
    pv_forecast_kwh: List[float],
    steps: int | None = None,
    forecast_sigma: float = 0.15,
    seed: int = 3,
) -> RollingResult:
    H = min(cfg.optimizer.horizon_hours, len(pv_forecast_kwh))
    steps = steps or H
    actual_pv = apply_forecast_error(pv_forecast_kwh[:H], seed=seed, sigma=forecast_sigma)

    battery = Battery(cfg.battery)
    result = RollingResult(hours=steps, soc_kwh=[battery.soc_kwh])
    remaining_duration: Dict[str, int] = {t.id: t.duration_h for t in tasks}
    by_id = {t.id: t for t in tasks}

    for t in range(steps):
        # Rebuild the task list from what is still outstanding, re-based to now.
        live: List[Task] = []
        for tk in tasks:
            left = remaining_duration[tk.id]
            if left <= 0:
                continue
            live.append(
                Task(tk.id, tk.name, tk.kind, tk.priority, tk.power_kw, left,
                     tk.release_h, tk.deadline_h, tk.preemptible)
            )
        live = shift_tasks(live, by_hours=t, horizon_h=H - t)

        plan = optimize(
            tasks=live,
            pv_kwh=pv_forecast_kwh[t:H],
            battery=cfg.battery,
            opt=cfg.optimizer,
            base_load_kw=cfg.site.base_load_kw,
            soc_init_kwh=battery.soc_kwh,
            horizon=max(1, H - t),
        )
        result.replans.append(plan)

        # Commit only the first slot of the fresh plan.
        running = plan.tasks_in_slot(0)
        load = cfg.site.base_load_kw + sum(by_id[tid].power_kw for tid in running)
        pv = actual_pv[t]
        direct = min(pv, load)
        deficit = load - direct
        surplus = pv - direct
        charge, discharge = battery.step(
            min(plan.charge_kwh[0], surplus), max(deficit, 0.0)
        )
        short = max(0.0, deficit - discharge)
        if short <= 1e-6:
            for tid in running:
                remaining_duration[tid] -= 1

        result.pv_actual_kwh.append(pv)
        result.pv_to_load_kwh.append(direct)
        result.charge_kwh.append(charge)
        result.discharge_kwh.append(discharge)
        result.load_kwh.append(load)
        result.unserved_kwh.append(short)
        result.running.append(running)
        result.soc_kwh.append(battery.soc_kwh)

    result.completed = {tid: rem <= 0 for tid, rem in remaining_duration.items()}
    return result
