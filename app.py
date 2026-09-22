"""Presentation layer.

Run with:  streamlit run app.py

Four tabs, in the order you should demo them:
  1. Plan      - the schedule the coordinator chose, as a timeline
  2. Energy    - where every kWh came from and what the pack did
  3. Scorecard - MILP vs the greedy rule-based baseline
  4. Ask       - natural-language questions about the decisions
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import load_config
from src.coordinator.baseline import greedy_plan
from src.coordinator.explainer import (
    answer_question,
    build_decision_trace,
    explain_task,
    template_explanation,
)
from src.coordinator.forecast import forecast_pv
from src.coordinator.optimizer import optimize
from src.data.scenarios import SCENARIO_BLURBS, get_scenario, list_scenarios
from src.data.weather import CONDITIONS, fetch_weather, weather_from_conditions
from src.metrics import score_plan
from src.twin.tasks import default_workload

SOLAR = "#F2A33C"
STORED = "#3FA7A0"
DRAW = "#8C93A8"
RISK = "#E2574C"
INK = "#0E1116"

st.set_page_config(page_title="Edge Energy Coordinator", layout="wide", page_icon="🔆")


# ------------------------------------------------------------------ data ---
@st.cache_data(show_spinner=False)
def build_everything(
    weather_mode: str,
    horizon: int,
    seed: int,
    n_flexible: int,
    soc_min: float,
    soc_comfort: float,
    w_depth: float,
    w_through: float,
    w_curtail: float,
    custom_weather_data: tuple | None = None,
):
    cfg = load_config()
    cfg.optimizer.horizon_hours = horizon
    cfg.battery.soc_min_frac = soc_min
    cfg.battery.soc_comfort_frac = soc_comfort
    cfg.optimizer.weights.update(
        {"deep_discharge": w_depth, "throughput": w_through, "curtailment": w_curtail}
    )

    if weather_mode == "live":
        weather = fetch_weather(cfg.site.latitude, cfg.site.longitude, hours=horizon,
                                timezone=cfg.site.timezone)
    elif weather_mode == "custom":
        # custom_weather_data = (conditions, temps, peak_ghi, sunrise_h, sunset_h)
        # conditions/temps are tuples (not lists) so this stays hashable and
        # st.cache_data can key on it - built in the sidebar editor below.
        conditions, temps, peak_ghi, sunrise_h, sunset_h = custom_weather_data
        weather = weather_from_conditions(
            list(conditions), list(temps),
            peak_ghi=peak_ghi, sunrise_h=sunrise_h, sunset_h=sunset_h,
        )
    else:
        weather = get_scenario(weather_mode, hours=horizon)

    pv = forecast_pv(weather, cfg.pv)
    tasks = default_workload(horizon_h=horizon, seed=seed, n_flexible=n_flexible)
    plan = optimize(tasks, pv.pv_kwh, cfg.battery, cfg.optimizer, cfg.site.base_load_kw)
    base = greedy_plan(tasks, pv.pv_kwh, cfg.battery, cfg.optimizer, cfg.site.base_load_kw)
    trace = build_decision_trace(plan, tasks, cfg, timestamps=pv.timestamps)
    return cfg, pv, tasks, plan, base, trace


# --------------------------------------------------------------- sidebar ---
st.sidebar.title("Site controls")
horizon = st.sidebar.slider("Planning horizon (hours)", 12, 24, 24)

weather_options = ["live", "custom"] + list_scenarios()
weather_labels = {
    "live": "Live (Open-Meteo)",
    "custom": "Custom (enter your own)",
    **{k: k.replace("_", " ").title() for k in list_scenarios()},
}
weather_mode = st.sidebar.selectbox(
    "Weather",
    weather_options,
    index=weather_options.index("afternoon_clouds"),
    format_func=lambda k: weather_labels[k],
    help="Pick a named synthetic scenario, pull today's real forecast, or type in your own hourly values.",
)
if weather_mode in SCENARIO_BLURBS:
    st.sidebar.caption(SCENARIO_BLURBS[weather_mode])

custom_weather_data = None
if weather_mode == "custom":
    st.sidebar.caption(
        "Describe each hour the way you'd actually know it - a condition "
        "and a temperature. No irradiance numbers needed; the coordinator "
        "works those out on its own."
    )

    with st.sidebar.expander("Advanced: daylight shape", expanded=False):
        peak_ghi = st.slider(
            "Peak sun intensity (clear-sky reference, W/m²)", 400.0, 1100.0, 900.0, 50.0,
            help="Only reached on a fully 'Clear' hour at solar noon - every other condition scales down from this.",
        )
        sunrise_h, sunset_h = st.slider("Daylight window (hour)", 0, 23, (6, 18))

    condition_options = list(CONDITIONS.keys())
    if "custom_weather_base" not in st.session_state or len(st.session_state["custom_weather_base"]) != horizon:
        st.session_state["custom_weather_base"] = pd.DataFrame(
            {
                "Hour": list(range(horizon)),
                "Condition": ["Sunny"] * horizon,
                "Temp (°C)": [24.0] * horizon,
            }
        )

    # Pass the SAME base frame in every rerun and never write the widget's
    # return value back into it. st.data_editor already tracks per-cell
    # edits internally against its own `key` and re-applies them onto
    # whatever `data` you pass - looping edited_df back in as `data` here
    # double-tracks state, which is what caused an edited row to jump to
    # the top of the table instead of staying put.
    edited_df = st.sidebar.data_editor(
        st.session_state["custom_weather_base"],
        disabled=["Hour"],
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Condition": st.column_config.SelectboxColumn(options=condition_options),
            "Temp (°C)": st.column_config.NumberColumn(step=0.5),  # no artificial bounds - any value is accepted
        },
        key="custom_weather_editor",
    )

    # Tuples, not lists, so this is hashable and st.cache_data can key on it.
    custom_weather_data = (
        tuple(edited_df["Condition"].tolist()),
        tuple(edited_df["Temp (°C)"].tolist()),
        peak_ghi,
        sunrise_h,
        sunset_h,
    )

seed = st.sidebar.number_input("Workload seed", value=7, step=1)
n_flex = st.sidebar.slider("Deferrable jobs", 0, 12, 6)

st.sidebar.subheader("Battery protection")
soc_min = st.sidebar.slider("Hard SoC floor", 0.05, 0.4, 0.20, 0.01)
soc_comfort = st.sidebar.slider("Comfort SoC floor", 0.2, 0.7, 0.40, 0.05)

st.sidebar.subheader("Objective weights")
w_depth = st.sidebar.slider("Deep-discharge penalty", 0.0, 100.0, 25.0, 5.0)
w_through = st.sidebar.slider("Throughput (wear) weight", 0.0, 10.0, 1.0, 0.5)
w_curtail = st.sidebar.slider("Curtailment penalty", 0.0, 1.0, 0.05, 0.05)

cfg, pv, tasks, plan, base, trace = build_everything(
    weather_mode, horizon, int(seed), n_flex, soc_min, soc_comfort, w_depth, w_through, w_curtail,
    custom_weather_data,
)
score = score_plan(plan, tasks, cfg.battery, cfg.optimizer.weights)
score_base = score_plan(base, tasks, cfg.battery, cfg.optimizer.weights)
times = pv.timestamps
by_id = {t.id: t for t in tasks}

# ---------------------------------------------------------------- header ---
st.title(cfg.site.name)

if pv.source == "open-meteo":
    badge_color, badge_bg, badge_text = "#1E7A46", "#12291D", "🟢 LIVE — Open-Meteo real-time forecast"
elif pv.source == "custom":
    badge_color, badge_bg = "#8C4FB0", "#241533"
    badge_text = "🟣 CUSTOM — user-entered weather"
elif pv.source.startswith("synthetic:"):
    scenario_id = pv.source.split(":", 1)[1]
    badge_color, badge_bg = "#3B6FB0", "#12203A"
    badge_text = f"🔵 SYNTHETIC SCENARIO — {scenario_id.replace('_', ' ').title()}"
else:  # bare "synthetic" -> live was requested but the API call failed
    badge_color, badge_bg = "#B0803B", "#332510"
    badge_text = "🟠 FALLBACK — live weather unreachable, showing a synthetic day instead"

st.markdown(
    f"""<div style="display:inline-block;padding:6px 14px;border-radius:6px;
    border:1px solid {badge_color};background:{badge_bg};color:{badge_color};
    font-weight:600;font-size:0.95rem;margin-bottom:8px;">{badge_text}</div>""",
    unsafe_allow_html=True,
)
st.caption(
    f"{pv.total_kwh:.2f} kWh of sun expected over {plan.horizon}h · "
    f"solver {plan.status} in {plan.solve_seconds}s"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Critical deadlines missed", score.critical_deadline_misses,
          delta=score.critical_deadline_misses - score_base.critical_deadline_misses,
          delta_color="inverse")
c2.metric("Load met directly by sun", f"{score.solar_share_pct}%",
          delta=f"{score.solar_share_pct - score_base.solar_share_pct:+.1f} pts")
c3.metric("Battery throughput", f"{score.battery_throughput_kwh:.2f} kWh",
          delta=f"{score.battery_throughput_kwh - score_base.battery_throughput_kwh:+.2f}",
          delta_color="inverse")
c4.metric("Lowest state of charge", f"{score.min_soc_pct}%",
          delta=f"{score.min_soc_pct - score_base.min_soc_pct:+.1f} pts")

tab_plan, tab_energy, tab_score, tab_ask = st.tabs(
    ["Plan", "Energy", "Scorecard", "Ask the coordinator"]
)

# ------------------------------------------------------------------ plan ---
with tab_plan:
    rows = []
    for tk in tasks:
        for slot in plan.schedule.get(tk.id, []):
            rows.append(
                dict(
                    Task=f"{tk.id} · {tk.name}",
                    Start=times[slot],
                    Finish=times[slot] + timedelta(hours=1),
                    Priority=tk.priority,
                    OnSun=pv.pv_kwh[slot] >= tk.power_kw,
                )
            )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pv.pv_kwh,
            y=[t.strftime("%H:%M") for t in times],
            orientation="h",
            marker_color=SOLAR,
            opacity=0.18,
            name="solar available",
            hovertemplate="%{y} · %{x:.2f} kWh<extra></extra>",
        )
    )
    if rows:
        df = pd.DataFrame(rows)
        order = {"critical": RISK, "high": STORED, "flexible": DRAW}
        for priority, group in df.groupby("Priority"):
            fig.add_trace(
                go.Bar(
                    x=[1] * len(group),
                    y=group["Start"].dt.strftime("%H:%M"),
                    orientation="h",
                    name=priority,
                    marker_color=order[priority],
                    text=group["Task"],
                    textposition="inside",
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    fig.update_layout(
        barmode="stack", height=720, template="plotly_dark",
        paper_bgcolor=INK, plot_bgcolor=INK,
        yaxis=dict(autorange="reversed", title=None),
        xaxis_title="tasks running (stacked) over solar availability",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Task decisions")
    table = []
    for tk in tasks:
        slots = plan.schedule.get(tk.id, [])
        table.append(
            {
                "id": tk.id,
                "task": tk.name,
                "priority": tk.priority,
                "kW": tk.power_kw,
                "hours": tk.duration_h,
                "window": f"{tk.release_h}-{tk.deadline_h}",
                "slack": tk.slack_h,
                "scheduled": ", ".join(times[s].strftime("%H:%M") for s in slots) or "—",
                "done": "yes" if plan.completed.get(tk.id) else "no",
            }
        )
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

    pick = st.selectbox("Explain one decision", [t.id for t in tasks])
    if st.button("Why this schedule?"):
        st.info(explain_task(pick, trace, cfg))

# ---------------------------------------------------------------- energy ---
with tab_energy:
    H = plan.horizon
    labels = [t.strftime("%H:%M") for t in times[:H]]
    pv_direct = [
        max(0.0, min(plan.pv_used_kwh[t] - plan.charge_kwh[t], plan.load_kwh[t]))
        for t in range(H)
    ]
    energy = go.Figure()
    energy.add_bar(x=labels, y=pv_direct, name="sun → load", marker_color=SOLAR)
    energy.add_bar(x=labels, y=plan.discharge_kwh[:H], name="battery → load",
                   marker_color=STORED)
    energy.add_bar(x=labels, y=[-c for c in plan.charge_kwh[:H]], name="→ battery",
                   marker_color=DRAW)
    energy.add_scatter(x=labels, y=plan.pv_available_kwh[:H], name="solar available",
                       mode="lines", line=dict(color=SOLAR, dash="dot"))
    energy.add_scatter(x=labels, y=plan.load_kwh[:H], name="total load",
                       mode="lines+markers", line=dict(color="#E9ECF1"))
    energy.update_layout(
        barmode="relative", template="plotly_dark", height=420,
        paper_bgcolor=INK, plot_bgcolor=INK, yaxis_title="kWh per hour",
        legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(energy, width="stretch")

    cap = cfg.battery.capacity_kwh
    soc = go.Figure()
    soc.add_scatter(x=labels + [labels[-1]], y=[100 * s / cap for s in plan.soc_kwh],
                    name="planned SoC", line=dict(color=STORED, width=3))
    soc.add_scatter(x=labels + [labels[-1]],
                    y=[100 * s / cap for s in base.soc_kwh],
                    name="greedy SoC", line=dict(color=DRAW, dash="dash"))
    soc.add_hline(y=100 * cfg.battery.soc_comfort_frac, line_dash="dot",
                  line_color=SOLAR, annotation_text="comfort floor")
    soc.add_hline(y=100 * cfg.battery.soc_min_frac, line_color=RISK,
                  annotation_text="hard floor")
    soc.update_layout(template="plotly_dark", height=360, paper_bgcolor=INK,
                      plot_bgcolor=INK, yaxis_title="state of charge (%)",
                      legend=dict(orientation="h", y=1.15),
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(soc, width="stretch")

# ------------------------------------------------------------- scorecard ---
with tab_score:
    comparison = pd.DataFrame(
        {"Coordinator (MILP)": score.to_dict(), "Greedy baseline": score_base.to_dict()}
    )
    st.dataframe(comparison, width="stretch")

    st.subheader("What the objective paid for")
    costs = pd.DataFrame(
        {
            "MILP": plan.cost_breakdown,
            "Greedy": base.cost_breakdown,
        }
    ).round(2)
    st.bar_chart(costs)

# ------------------------------------------------------------------- ask ---
with tab_ask:
    st.write(template_explanation(trace))
    question = st.text_input(
        "Ask about the plan",
        placeholder="Why is the vibration sweep running at 07:00 instead of at sunrise?",
    )
    if question:
        with st.spinner("Reading the decision trace..."):
            st.markdown(answer_question(question, trace, cfg))
    with st.expander("Raw decision trace (what the LLM is allowed to see)"):
        st.json(trace)
