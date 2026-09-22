"""End-to-end pipeline in one command. Run this before you touch the UI.

    python -m scripts.run_sim
    python -m scripts.run_sim --synthetic --rolling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.coordinator.baseline import greedy_plan
from src.coordinator.explainer import build_decision_trace, template_explanation
from src.coordinator.forecast import forecast_pv
from src.coordinator.loop import run_rolling_horizon
from src.coordinator.optimizer import optimize
from src.data.scenarios import SCENARIO_BLURBS, get_scenario, list_scenarios
from src.data.weather import fetch_weather, synthetic_weather
from src.metrics import compare, score_plan
from src.twin.site import SiteTwin
from src.twin.tasks import default_workload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the edge energy coordinator")
    parser.add_argument("--synthetic", action="store_true", help="skip the weather API")
    parser.add_argument(
        "--scenario", choices=list_scenarios(), default=None,
        help="use a named synthetic weather scenario instead of live/plain synthetic data",
    )
    parser.add_argument("--list-scenarios", action="store_true",
                        help="print available scenarios and exit")
    parser.add_argument("--rolling", action="store_true", help="also run the MPC loop")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.list_scenarios:
        for name in list_scenarios():
            print(f"{name:20s} {SCENARIO_BLURBS[name]}")
        return

    cfg = load_config()
    H = cfg.optimizer.horizon_hours

    # 1. DATA ---------------------------------------------------------------
    if args.scenario:
        weather = get_scenario(args.scenario, hours=H)
    elif args.synthetic:
        weather = synthetic_weather(hours=H)
    else:
        weather = fetch_weather(cfg.site.latitude, cfg.site.longitude, hours=H,
                                timezone=cfg.site.timezone)
    print(f"weather source      : {weather.source}  ({len(weather)} hours)")

    # 2. FORECAST -----------------------------------------------------------
    pv = forecast_pv(weather, cfg.pv)
    print(f"forecast PV energy  : {pv.total_kwh:.2f} kWh  peak slots {pv.peak_hours()}")

    # 3. WORKLOAD -----------------------------------------------------------
    tasks = default_workload(
        horizon_h=H,
        seed=args.seed if args.seed is not None else cfg.workload.get("seed", 7),
        n_flexible=cfg.workload.get("n_flexible", 6),
    )
    print(f"tasks generated     : {len(tasks)} "
          f"({sum(1 for t in tasks if t.priority == 'critical')} critical)")

    # 4. OPTIMIZE -----------------------------------------------------------
    plan = optimize(tasks, pv.pv_kwh, cfg.battery, cfg.optimizer, cfg.site.base_load_kw)
    print(f"MILP status         : {plan.status} in {plan.solve_seconds}s")

    base = greedy_plan(tasks, pv.pv_kwh, cfg.battery, cfg.optimizer, cfg.site.base_load_kw)

    # 5. SCORE --------------------------------------------------------------
    s_milp = score_plan(plan, tasks, cfg.battery, cfg.optimizer.weights)
    s_greedy = score_plan(base, tasks, cfg.battery, cfg.optimizer.weights)
    print()
    print(compare(s_milp, s_greedy))

    # 6. EXECUTE IN THE TWIN ------------------------------------------------
    twin = SiteTwin(cfg)
    execution = twin.execute(plan, tasks)
    print(f"\ntwin end SoC        : {100 * twin.battery.soc_frac:.1f}%  "
          f"cycles {twin.battery.equivalent_full_cycles:.3f}  "
          f"unserved {sum(execution.unserved_kwh):.3f} kWh")

    # 7. EXPLAIN ------------------------------------------------------------
    trace = build_decision_trace(plan, tasks, cfg, timestamps=pv.timestamps)
    print("\n--- explanation ---")
    print(template_explanation(trace))

    # 8. OPTIONAL ROLLING HORIZON ------------------------------------------
    if args.rolling:
        roll = run_rolling_horizon(cfg, tasks, pv.pv_kwh)
        done = sum(1 for v in roll.completed.values() if v)
        print(f"\nrolling horizon     : {len(roll.replans)} re-plans, "
              f"{done}/{len(tasks)} tasks finished, "
              f"end SoC {100 * roll.soc_kwh[-1] / cfg.battery.capacity_kwh:.1f}%")


if __name__ == "__main__":
    main()
