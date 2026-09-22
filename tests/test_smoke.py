"""Run with:  python -m pytest -q

These are the invariants a judge might poke at. If one of these breaks, your
plan is physically impossible and the demo is a liability.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.config import load_config
from src.coordinator.baseline import greedy_plan
from src.coordinator.explainer import build_decision_trace, template_explanation
from src.coordinator.forecast import forecast_pv
from src.coordinator.optimizer import optimize
from src.data.weather import synthetic_weather
from src.metrics import score_plan
from src.twin.site import SiteTwin
from src.twin.tasks import default_workload

TOL = 1e-6


@pytest.fixture(scope="module")
def solved():
    cfg = load_config()
    weather = synthetic_weather(hours=cfg.optimizer.horizon_hours)
    pv = forecast_pv(weather, cfg.pv)
    tasks = default_workload(horizon_h=cfg.optimizer.horizon_hours, seed=7)
    plan = optimize(tasks, pv.pv_kwh, cfg.battery, cfg.optimizer, cfg.site.base_load_kw)
    return cfg, pv, tasks, plan


def test_solver_reaches_optimality(solved):
    _, _, _, plan = solved
    assert plan.status == "Optimal"


def test_power_balance_holds_every_hour(solved):
    cfg, _, _, plan = solved
    for t in range(plan.horizon):
        lhs = plan.pv_used_kwh[t] + plan.discharge_kwh[t] + plan.unserved_kwh[t]
        rhs = plan.load_kwh[t] + plan.charge_kwh[t]
        assert lhs == pytest.approx(rhs, abs=1e-4), f"imbalance at hour {t}"


def test_soc_never_violates_limits(solved):
    cfg, _, _, plan = solved
    for s in plan.soc_kwh:
        assert cfg.battery.soc_min_kwh - TOL <= s <= cfg.battery.soc_max_kwh + TOL
    assert plan.soc_kwh[-1] >= cfg.battery.soc_terminal_kwh - 1e-4


def test_no_simultaneous_charge_and_discharge(solved):
    _, _, _, plan = solved
    for t in range(plan.horizon):
        assert min(plan.charge_kwh[t], plan.discharge_kwh[t]) < 1e-4


def test_tasks_respect_their_windows_and_durations(solved):
    _, _, tasks, plan = solved
    for tk in tasks:
        slots = plan.schedule[tk.id]
        for s in slots:
            assert tk.release_h <= s <= tk.deadline_h
        if plan.completed[tk.id]:
            assert len(slots) == tk.duration_h
            if not tk.preemptible and slots:
                assert slots == list(range(min(slots), min(slots) + tk.duration_h))


def test_critical_tasks_all_complete(solved):
    _, _, tasks, plan = solved
    for tk in tasks:
        if tk.priority == "critical":
            assert plan.completed[tk.id], f"{tk.id} missed its deadline"


def test_milp_beats_greedy_on_degradation(solved):
    cfg, pv, tasks, plan = solved
    base = greedy_plan(tasks, pv.pv_kwh, cfg.battery, cfg.optimizer, cfg.site.base_load_kw)
    a = score_plan(plan, tasks, cfg.battery, cfg.optimizer.weights)
    b = score_plan(base, tasks, cfg.battery, cfg.optimizer.weights)
    assert a.degradation_cost < b.degradation_cost
    assert a.min_soc_pct >= b.min_soc_pct


def test_twin_executes_plan_without_blackout(solved):
    cfg, _, tasks, plan = solved
    execution = SiteTwin(cfg).execute(plan, tasks)
    assert sum(execution.unserved_kwh) < 1e-3


def test_explainer_works_without_an_api_key(solved):
    cfg, pv, tasks, plan = solved
    trace = build_decision_trace(plan, tasks, cfg, timestamps=pv.timestamps)
    text = template_explanation(trace)
    assert "Solar forecast" in text and len(trace["task_decisions"]) == len(tasks)
