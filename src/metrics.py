"""The five numbers that win the judging round.

Every metric here maps directly onto one of the three goals in the brief:
deadlines met, degradation minimised, renewables preferred.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from src.config import BatteryConfig
from src.coordinator.optimizer import Plan
from src.twin.tasks import Task


@dataclass
class Scorecard:
    critical_deadline_misses: int
    tasks_completed: int
    tasks_total: int
    load_served_kwh: float
    solar_share_pct: float          # fraction of load met directly by PV
    battery_throughput_kwh: float
    equivalent_full_cycles: float
    min_soc_pct: float
    deep_discharge_kwh_h: float
    curtailed_kwh: float
    unserved_kwh: float
    degradation_cost: float
    total_cost: float

    def to_dict(self) -> Dict:
        return asdict(self)


def score_plan(
    plan: Plan,
    tasks: List[Task],
    battery: BatteryConfig,
    weights: Dict[str, float],
) -> Scorecard:
    H = plan.horizon
    load = sum(plan.load_kwh)
    unserved = sum(plan.unserved_kwh)
    served = max(1e-9, load - unserved)

    # PV that went straight to the load (the rest of pv_used went into storage).
    pv_direct = sum(
        max(0.0, min(plan.pv_used_kwh[t] - plan.charge_kwh[t], plan.load_kwh[t]))
        for t in range(H)
    )
    throughput = sum(plan.charge_kwh) + sum(plan.discharge_kwh)
    depth_stress = sum(
        max(0.0, battery.soc_comfort_kwh - s) for s in plan.soc_kwh
    )
    rate_stress = sum(
        max(0.0, plan.charge_kwh[t] - battery.soft_rate_kw)
        + max(0.0, plan.discharge_kwh[t] - battery.soft_rate_kw)
        for t in range(H)
    )
    misses = sum(
        1 for t in tasks if t.priority == "critical" and not plan.completed.get(t.id, False)
    )
    degradation = (
        weights.get("throughput", 1.0) * battery.cycle_cost_per_kwh * throughput
        + weights.get("deep_discharge", 0.0) * depth_stress
        + weights.get("rate_stress", 0.0) * rate_stress
    )

    return Scorecard(
        critical_deadline_misses=misses,
        tasks_completed=sum(1 for t in tasks if plan.completed.get(t.id, False)),
        tasks_total=len(tasks),
        load_served_kwh=round(served, 3),
        solar_share_pct=round(100.0 * pv_direct / served, 1),
        battery_throughput_kwh=round(throughput, 3),
        equivalent_full_cycles=round(throughput / (2 * battery.capacity_kwh), 3),
        min_soc_pct=round(100.0 * min(plan.soc_kwh) / battery.capacity_kwh, 1),
        deep_discharge_kwh_h=round(depth_stress, 3),
        curtailed_kwh=round(
            sum(max(0.0, plan.pv_available_kwh[t] - plan.pv_used_kwh[t]) for t in range(H)), 3
        ),
        unserved_kwh=round(unserved, 4),
        degradation_cost=round(degradation, 2),
        total_cost=round(sum(plan.cost_breakdown.values()), 2),
    )


def compare(a: Scorecard, b: Scorecard, label_a: str = "MILP", label_b: str = "Greedy") -> str:
    """One-line-per-metric text diff, handy for the terminal and for the LLM."""
    lines = [f"{'metric':<28}{label_a:>14}{label_b:>14}"]
    for key in a.to_dict():
        lines.append(f"{key:<28}{getattr(a, key)!s:>14}{getattr(b, key)!s:>14}")
    return "\n".join(lines)
