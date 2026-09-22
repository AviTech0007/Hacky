"""The MILP coordinator. This is the heart of the project.

Decision variables per hourly slot t in 0..H-1
--------------------------------------------
  run[i, t]    binary   task i is executing in slot t
  start[i, t]  binary   task i begins in slot t   (non-preemptible tasks only)
  done[i]      binary   task i completes within its window
  pv_use[t]    >= 0     PV energy actually consumed (rest is curtailed)
  chg[t]       >= 0     energy drawn off the bus into the battery
  dis[t]       >= 0     energy delivered from the battery to the bus
  chg_on/dis_on binary  mutual exclusion, no simultaneous charge+discharge
  soc[t]       >= 0     state of charge in kWh at the *start* of slot t (t=0..H)
  unserved[t]  >= 0     slack: load we failed to power (huge penalty)
  dod_gap[t]   >= 0     how far below the comfort SoC we sank
  rate_over[t] >= 0     how far above the soft C-rate we pushed the cells

Constraints
-----------
  completion   sum_t run[i,t] == duration_i * done[i]
  contiguity   run[i,t] == sum of start[i,s] covering t   (non-preemptible)
  windows      run[i,t] == 0 outside [release_i, deadline_i]
  power balance pv_use[t] + dis[t] + unserved[t] == load[t] + chg[t]
  SoC dynamics soc[t+1] == soc[t] + eta_c*chg[t] - dis[t]/eta_d
  SoC bounds   soc_min <= soc[t] <= soc_max, soc[H] >= soc_terminal

Objective (minimise)
--------------------
  w_unserved * unserved
+ w_missed   * sum value_i * (1 - done_i)
+ w_through  * cycle_cost * (chg + dis)
+ w_depth    * dod_gap
+ w_rate     * rate_over
+ w_curtail  * (pv - pv_use)

The last term is what makes the plan *prefer sunlight over stored energy*:
leaving free PV on the table is never rewarded, so the solver pulls flexible
work into sunny hours instead of draining the pack at night.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pulp

from src.config import BatteryConfig, OptimizerConfig
from src.twin.tasks import Task


def _pick_solver(time_limit_s: int):
    """CBC ships with PuLP. Newer PuLP prefers COIN_CMD; older only has
    PULP_CBC_CMD. Try both so the project runs on whatever your teammates
    happen to have installed."""
    for name in ("PULP_CBC_CMD", "COIN_CMD"):
        factory = getattr(pulp, name, None)
        if factory is None:
            continue
        try:
            solver = factory(msg=0, timeLimit=time_limit_s)
            if solver.available():
                return solver
        except Exception:
            continue
    return None  # PuLP falls back to its default solver


@dataclass
class Plan:
    """Everything the dashboard, the simulator and the explainer need."""

    status: str
    objective: float
    horizon: int
    schedule: Dict[str, List[int]]        # task id -> slots it runs in
    completed: Dict[str, bool]
    pv_available_kwh: List[float]
    pv_used_kwh: List[float]
    charge_kwh: List[float]
    discharge_kwh: List[float]
    load_kwh: List[float]
    unserved_kwh: List[float]
    soc_kwh: List[float]                  # length H+1
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    solve_seconds: float = 0.0

    def tasks_in_slot(self, t: int) -> List[str]:
        return [tid for tid, slots in self.schedule.items() if t in slots]

    def soc_frac(self, capacity_kwh: float) -> List[float]:
        return [s / capacity_kwh for s in self.soc_kwh]


def optimize(
    tasks: List[Task],
    pv_kwh: List[float],
    battery: BatteryConfig,
    opt: OptimizerConfig,
    base_load_kw: float,
    soc_init_kwh: Optional[float] = None,
    horizon: Optional[int] = None,
    dt_h: float = 1.0,
) -> Plan:
    import time

    start_time = time.time()
    H = horizon or min(opt.horizon_hours, len(pv_kwh))
    pv_kwh = list(pv_kwh[:H])
    if len(pv_kwh) < H:                      # pad if the forecast is short
        pv_kwh += [0.0] * (H - len(pv_kwh))
    w = opt.weights
    soc0 = battery.soc_init_kwh if soc_init_kwh is None else soc_init_kwh
    base_load_kwh = base_load_kw * dt_h

    prob = pulp.LpProblem("edge_energy_coordinator", pulp.LpMinimize)

    # ---------------- task variables ------------------------------------
    run: Dict[tuple, pulp.LpVariable] = {}
    done: Dict[str, pulp.LpVariable] = {}

    for tk in tasks:
        done[tk.id] = pulp.LpVariable(f"done_{tk.id}", cat="Binary")
        lo, hi = max(0, tk.release_h), min(H - 1, tk.deadline_h)
        feasible_window = (hi - lo + 1) >= tk.duration_h

        for t in range(H):
            run[tk.id, t] = pulp.LpVariable(f"run_{tk.id}_{t}", cat="Binary")
            if not feasible_window or t < lo or t > hi:
                prob += run[tk.id, t] == 0, f"window_{tk.id}_{t}"

        if not feasible_window:
            prob += done[tk.id] == 0, f"infeasible_{tk.id}"
            continue

        if tk.preemptible:
            prob += (
                pulp.lpSum(run[tk.id, t] for t in range(lo, hi + 1))
                == tk.duration_h * done[tk.id],
                f"complete_{tk.id}",
            )
        else:
            # Contiguous execution: pick one start slot, it covers duration_h slots.
            valid_starts = range(lo, hi - tk.duration_h + 2)
            start_vars = {
                s: pulp.LpVariable(f"start_{tk.id}_{s}", cat="Binary") for s in valid_starts
            }
            prob += pulp.lpSum(start_vars.values()) == done[tk.id], f"onestart_{tk.id}"
            for t in range(lo, hi + 1):
                covering = [
                    start_vars[s]
                    for s in valid_starts
                    if s <= t <= s + tk.duration_h - 1
                ]
                prob += run[tk.id, t] == pulp.lpSum(covering), f"cover_{tk.id}_{t}"

    # ---------------- energy variables -----------------------------------
    pv_use = pulp.LpVariable.dicts("pv_use", range(H), lowBound=0)
    chg = pulp.LpVariable.dicts("chg", range(H), lowBound=0)
    dis = pulp.LpVariable.dicts("dis", range(H), lowBound=0)
    chg_on = pulp.LpVariable.dicts("chg_on", range(H), cat="Binary")
    dis_on = pulp.LpVariable.dicts("dis_on", range(H), cat="Binary")
    soc = pulp.LpVariable.dicts("soc", range(H + 1), lowBound=battery.soc_min_kwh,
                                upBound=battery.soc_max_kwh)
    unserved = pulp.LpVariable.dicts("unserved", range(H), lowBound=0)
    dod_gap = pulp.LpVariable.dicts("dod_gap", range(H + 1), lowBound=0)
    rate_over = pulp.LpVariable.dicts("rate_over", range(H), lowBound=0)

    prob += soc[0] == soc0, "soc_init"
    prob += soc[H] >= battery.soc_terminal_kwh, "soc_terminal"

    for t in range(H):
        load_t = base_load_kwh + pulp.lpSum(
            tk.power_kw * dt_h * run[tk.id, t] for tk in tasks
        )

        prob += pv_use[t] <= pv_kwh[t], f"pv_cap_{t}"
        prob += pv_use[t] + dis[t] + unserved[t] == load_t + chg[t], f"balance_{t}"

        prob += chg[t] <= battery.max_charge_kw * dt_h * chg_on[t], f"chg_cap_{t}"
        prob += dis[t] <= battery.max_discharge_kw * dt_h * dis_on[t], f"dis_cap_{t}"
        prob += chg_on[t] + dis_on[t] <= 1, f"exclusive_{t}"

        prob += (
            soc[t + 1]
            == soc[t]
            + battery.charge_efficiency * chg[t]
            - dis[t] / battery.discharge_efficiency,
            f"soc_dyn_{t}",
        )

        prob += dod_gap[t + 1] >= battery.soc_comfort_kwh - soc[t + 1], f"dod_{t}"
        soft = battery.soft_rate_kw * dt_h
        prob += rate_over[t] >= chg[t] - soft, f"rate_c_{t}"
        prob += rate_over[t] >= dis[t] - soft, f"rate_d_{t}"

    # ---------------- objective -------------------------------------------
    cost_unserved = w["unserved"] * pulp.lpSum(unserved[t] for t in range(H))
    cost_missed = w["missed_task"] * pulp.lpSum(
        tk.value * (1 - done[tk.id]) for tk in tasks
    )
    cost_throughput = (
        w["throughput"]
        * battery.cycle_cost_per_kwh
        * pulp.lpSum(chg[t] + dis[t] for t in range(H))
    )
    cost_depth = w["deep_discharge"] * pulp.lpSum(dod_gap[t] for t in range(H + 1))
    cost_rate = w["rate_stress"] * pulp.lpSum(rate_over[t] for t in range(H))
    cost_curtail = w["curtailment"] * pulp.lpSum(pv_kwh[t] - pv_use[t] for t in range(H))

    prob += cost_unserved + cost_missed + cost_throughput + cost_depth + cost_rate + cost_curtail

    prob.solve(_pick_solver(opt.time_limit_s))
    status = pulp.LpStatus[prob.status]

    def val(v) -> float:
        raw = pulp.value(v)
        return 0.0 if raw is None else float(raw)

    schedule = {
        tk.id: [t for t in range(H) if val(run[tk.id, t]) > 0.5] for tk in tasks
    }
    load_kwh = [
        base_load_kwh + sum(tk.power_kw * dt_h for tk in tasks if t in schedule[tk.id])
        for t in range(H)
    ]

    return Plan(
        status=status,
        objective=val(prob.objective),
        horizon=H,
        schedule=schedule,
        completed={tk.id: val(done[tk.id]) > 0.5 for tk in tasks},
        pv_available_kwh=pv_kwh,
        pv_used_kwh=[val(pv_use[t]) for t in range(H)],
        charge_kwh=[val(chg[t]) for t in range(H)],
        discharge_kwh=[val(dis[t]) for t in range(H)],
        load_kwh=load_kwh,
        unserved_kwh=[val(unserved[t]) for t in range(H)],
        soc_kwh=[val(soc[t]) for t in range(H + 1)],
        cost_breakdown={
            "unserved": val(cost_unserved),
            "missed_tasks": val(cost_missed),
            "battery_throughput": val(cost_throughput),
            "deep_discharge": val(cost_depth),
            "rate_stress": val(cost_rate),
            "curtailment": val(cost_curtail),
        },
        solve_seconds=round(time.time() - start_time, 3),
    )
