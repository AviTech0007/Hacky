"""Explainer agent: turns the solver's numbers into sentences a human trusts.

Two layers, and the split matters:

  1. `build_decision_trace` is pure Python. It extracts the *facts* - which
     slot each task landed in, what the PV was doing then, whether the SoC
     floor was binding, what each cost term contributed. No LLM involved, so
     it can never hallucinate.

  2. The LLM is only allowed to narrate that trace. It receives the JSON and a
     system prompt that forbids inventing numbers.

If there is no API key, `explain_plan` returns a deterministic template
explanation instead of failing. Demo it offline with confidence.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from src.config import AppConfig
from src.coordinator.optimizer import Plan
from src.twin.tasks import Task

SYSTEM_PROMPT = """You are the explainability layer of an energy coordinator for an
off-grid edge computing node. You are given a JSON decision trace produced by a
mixed-integer linear program.

Rules:
- Only state facts present in the trace. Never invent numbers, hours or tasks.
- Refer to hours by their slot index and clock time as given.
- Be concrete: name the task, the hour, and the binding reason (sunlight
  available, state-of-charge floor, deadline window, charge-rate limit).
- Write for an operator, not a mathematician. No jargon like "dual variable".
- Keep it under 200 words unless asked for more.
"""


# ---------------------------------------------------------------- trace ----
def build_decision_trace(
    plan: Plan,
    tasks: List[Task],
    cfg: AppConfig,
    timestamps: Optional[List] = None,
) -> Dict:
    """Ground truth about the plan, in a shape an LLM can narrate safely."""
    H = plan.horizon
    cap = cfg.battery.capacity_kwh

    def clock(t: int) -> str:
        if timestamps and t < len(timestamps):
            return timestamps[t].strftime("%H:%M")
        return f"h{t}"

    hours = [
        {
            "slot": t,
            "time": clock(t),
            "pv_available_kwh": round(plan.pv_available_kwh[t], 3),
            "pv_used_kwh": round(plan.pv_used_kwh[t], 3),
            "load_kwh": round(plan.load_kwh[t], 3),
            "battery_charge_kwh": round(plan.charge_kwh[t], 3),
            "battery_discharge_kwh": round(plan.discharge_kwh[t], 3),
            "soc_pct_end": round(100 * plan.soc_kwh[t + 1] / cap, 1),
            "running": plan.tasks_in_slot(t),
        }
        for t in range(H)
    ]

    decisions = []
    for tk in tasks:
        slots = plan.schedule.get(tk.id, [])
        earliest = tk.release_h
        pv_at_slots = [round(plan.pv_available_kwh[s], 3) for s in slots if s < H]
        pv_at_earliest = (
            round(plan.pv_available_kwh[earliest], 3) if earliest < H else None
        )
        decisions.append(
            {
                "task": tk.id,
                "name": tk.name,
                "priority": tk.priority,
                "power_kw": tk.power_kw,
                "duration_h": tk.duration_h,
                "window": [tk.release_h, tk.deadline_h],
                "slack_h": tk.slack_h,
                "completed": plan.completed.get(tk.id, False),
                "scheduled_slots": slots,
                "scheduled_times": [clock(s) for s in slots],
                "deferred_by_h": (min(slots) - earliest) if slots else None,
                "pv_at_scheduled_slots_kwh": pv_at_slots,
                "pv_at_earliest_slot_kwh": pv_at_earliest,
                "ran_on_sunlight": all(
                    plan.pv_available_kwh[s] >= tk.power_kw for s in slots if s < H
                )
                if slots
                else False,
            }
        )

    soc_pct = [round(100 * s / cap, 1) for s in plan.soc_kwh]
    binding = []
    if min(soc_pct) <= cfg.battery.soc_min_frac * 100 + 1.0:
        binding.append("state-of-charge floor was reached")
    if min(soc_pct) < cfg.battery.soc_comfort_frac * 100:
        binding.append("pack dipped below the comfort band, incurring depth-of-discharge cost")
    if sum(plan.unserved_kwh) > 1e-6:
        binding.append("some load could not be served")
    if any(
        plan.pv_available_kwh[t] - plan.pv_used_kwh[t] > 0.05 for t in range(H)
    ):
        binding.append("surplus solar was curtailed because the pack was full or rate-limited")

    return {
        "site": cfg.site.name,
        "horizon_hours": H,
        "solver_status": plan.status,
        "solve_seconds": plan.solve_seconds,
        "objective": round(plan.objective, 2) if plan.objective == plan.objective else None,
        "cost_breakdown": {k: round(v, 2) for k, v in plan.cost_breakdown.items()},
        "battery": {
            "capacity_kwh": cap,
            "soc_start_pct": soc_pct[0],
            "soc_end_pct": soc_pct[-1],
            "soc_min_pct": min(soc_pct),
            "hard_floor_pct": cfg.battery.soc_min_frac * 100,
            "comfort_floor_pct": cfg.battery.soc_comfort_frac * 100,
            "throughput_kwh": round(sum(plan.charge_kwh) + sum(plan.discharge_kwh), 3),
        },
        "pv_total_kwh": round(sum(plan.pv_available_kwh), 3),
        "binding_constraints": binding,
        "task_decisions": decisions,
        "hourly": hours,
    }


# ------------------------------------------------------------------ LLM ----
def _groq_client(cfg: AppConfig):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    try:
        from groq import Groq
    except ImportError:
        return None
    return Groq(api_key=key)


def _chat(cfg: AppConfig, messages: List[Dict]) -> Optional[str]:
    client = _groq_client(cfg)
    model = os.environ.get("GROQ_MODEL", cfg.explainer.get("model", "openai/gpt-oss-120b"))
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model= model,
            temperature=cfg.explainer.get("temperature", 0.2),
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as exc:  # network down, bad key, rate limit
        return f"(LLM unavailable: {exc})"


def explain_plan(trace: Dict, cfg: AppConfig) -> str:
    """Narrate the whole schedule."""
    llm = _chat(
        cfg,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Summarise today's energy plan in four short paragraphs: "
                "the shape of the solar day, what you scheduled and when, how you "
                "protected the battery, and any risk to watch.\n\n"
                + json.dumps(trace)[:12000],
            },
        ],
    )
    return llm or template_explanation(trace)


def explain_task(task_id: str, trace: Dict, cfg: AppConfig) -> str:
    """Answer 'why did you delay Task X?' for one specific task."""
    decision = next((d for d in trace["task_decisions"] if d["task"] == task_id), None)
    if decision is None:
        return f"No decision recorded for {task_id}."
    llm = _chat(
        cfg,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Explain in three sentences why task {task_id} was scheduled "
                f"the way it was.\n\nTask decision: {json.dumps(decision)}\n\n"
                f"Context: {json.dumps({k: trace[k] for k in ('battery', 'binding_constraints', 'cost_breakdown')})}",
            },
        ],
    )
    return llm or template_task_explanation(decision, trace)


def answer_question(question: str, trace: Dict, cfg: AppConfig) -> str:
    """Free-form chat panel on the dashboard."""
    llm = _chat(
        cfg,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}\n\nDecision trace:\n{json.dumps(trace)[:12000]}"},
        ],
    )
    return llm or (
        "No LLM key configured. Set GROQ_API_KEY in .env to enable the chat panel. "
        "Here is the deterministic summary instead:\n\n" + template_explanation(trace)
    )


# ------------------------------------------------------------- fallback ----
def template_explanation(trace: Dict) -> str:
    bat = trace["battery"]
    deferred = [
        d for d in trace["task_decisions"] if d["deferred_by_h"] and d["deferred_by_h"] > 0
    ]
    dropped = [d for d in trace["task_decisions"] if not d["completed"]]
    parts = [
        f"Solar forecast for the next {trace['horizon_hours']}h totals "
        f"{trace['pv_total_kwh']} kWh. Solver status: {trace['solver_status']} "
        f"({trace['solve_seconds']}s).",
        f"The pack starts at {bat['soc_start_pct']}%, bottoms out at "
        f"{bat['soc_min_pct']}% against a hard floor of {bat['hard_floor_pct']}%, "
        f"and finishes at {bat['soc_end_pct']}% with {bat['throughput_kwh']} kWh "
        f"of total throughput.",
    ]
    if deferred:
        names = ", ".join(
            f"{d['task']} by {d['deferred_by_h']}h to {d['scheduled_times'][0]}"
            for d in deferred[:4]
        )
        parts.append(f"Deferred to reach sunlight or protect the pack: {names}.")
    if dropped:
        parts.append(
            "Not scheduled: " + ", ".join(f"{d['task']} ({d['priority']})" for d in dropped) + "."
        )
    if trace["binding_constraints"]:
        parts.append("Binding limits: " + "; ".join(trace["binding_constraints"]) + ".")
    return "\n\n".join(parts)


def template_task_explanation(decision: Dict, trace: Dict) -> str:
    if not decision["completed"]:
        return (
            f"{decision['task']} ({decision['name']}, {decision['priority']}) was not "
            f"scheduled. Its window was hours {decision['window'][0]}-{decision['window'][1]} "
            f"and completing it would have cost more in battery wear than its priority "
            f"value justified."
        )
    slots = ", ".join(decision["scheduled_times"])
    line = (
        f"{decision['task']} ({decision['name']}) runs at {slots}, drawing "
        f"{decision['power_kw']} kW for {decision['duration_h']}h."
    )
    if decision["deferred_by_h"]:
        line += (
            f" It was held back {decision['deferred_by_h']}h from its release time: "
            f"solar at the earliest slot was {decision['pv_at_earliest_slot_kwh']} kWh "
            f"versus {decision['pv_at_scheduled_slots_kwh']} kWh at the chosen slots, "
            f"so running later avoids drawing on the battery."
        )
    else:
        line += " It ran at the first opportunity; nothing was gained by waiting."
    return line
