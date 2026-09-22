"""Run every named scenario back to back and print one comparison table.

    python -m scripts.compare_scenarios

Good for a slide: shows the coordinator holding zero critical misses across
every condition while flexible-task completion and battery health scale
sensibly with how much sun is actually available.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.coordinator.baseline import greedy_plan
from src.coordinator.forecast import forecast_pv
from src.coordinator.optimizer import optimize
from src.data.scenarios import list_scenarios, get_scenario
from src.metrics import score_plan
from src.twin.tasks import default_workload


def main() -> None:
    cfg = load_config()
    H = cfg.optimizer.horizon_hours
    tasks = default_workload(horizon_h=H, seed=cfg.workload.get("seed", 7),
                             n_flexible=cfg.workload.get("n_flexible", 6))

    header = (
        f"{'scenario':<20}{'pv_kWh':>8}{'coord_miss':>11}{'greedy_miss':>12}"
        f"{'coord_done':>11}{'greedy_done':>12}{'coord_minSoC':>13}{'greedy_minSoC':>14}"
        f"{'coord_deg$':>11}{'greedy_deg$':>12}"
    )
    print(header)
    print("-" * len(header))

    for name in list_scenarios():
        weather = get_scenario(name, hours=H)
        pv = forecast_pv(weather, cfg.pv)
        plan = optimize(tasks, pv.pv_kwh, cfg.battery, cfg.optimizer, cfg.site.base_load_kw)
        base = greedy_plan(tasks, pv.pv_kwh, cfg.battery, cfg.optimizer, cfg.site.base_load_kw)
        a = score_plan(plan, tasks, cfg.battery, cfg.optimizer.weights)
        b = score_plan(base, tasks, cfg.battery, cfg.optimizer.weights)
        print(
            f"{name:<20}{pv.total_kwh:>8.2f}{a.critical_deadline_misses:>11}"
            f"{b.critical_deadline_misses:>12}{a.tasks_completed:>11}{b.tasks_completed:>12}"
            f"{a.min_soc_pct:>13.1f}{b.min_soc_pct:>14.1f}"
            f"{a.degradation_cost:>11.2f}{b.degradation_cost:>12.2f}"
        )


if __name__ == "__main__":
    main()
