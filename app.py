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

from src.database import init_database, load_settings, save_settings
from src.auth import send_email_otp, verify_otp, logout

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


st.set_page_config(
    page_title="Edge Energy Coordinator",
    layout="wide",
    page_icon="🔆",
)

init_database()


# ======================================================================
# AUTHENTICATION
# ======================================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "otp_sent" not in st.session_state:
    st.session_state.otp_sent = False


if not st.session_state.authenticated:

    st.title("Edge Energy Coordinator")
    st.subheader("Login")

    if not st.session_state.otp_sent:

        email = st.text_input(
            "Email",
            placeholder="you@example.com",
        )

        if st.button("Send OTP", type="primary"):

            if not email or "@" not in email:
                st.error("Enter a valid email address.")

            else:
                try:
                    send_email_otp(email)

                    st.session_state.otp_sent = True
                    st.session_state.login_email = email

                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to send OTP: {e}")

    else:

        st.success(
            f"OTP sent to {st.session_state.login_email}"
        )

        otp = st.text_input(
            "Enter OTP",
            max_chars=6,
            type="password",
            placeholder="123456",
        )

        if st.button("Verify OTP", type="primary"):

            if verify_otp(
                st.session_state.login_email,
                otp,
            ):
                st.session_state.otp_sent = False

                # Force settings to reload for the newly authenticated user.
                st.session_state.pop("user_settings", None)
                st.session_state.pop("custom_weather_base", None)

                st.rerun()

            else:
                st.error("Invalid or expired OTP.")

        if st.button("Use different email"):
            st.session_state.otp_sent = False
            st.session_state.pop("login_email", None)
            st.rerun()

    st.stop()


# ======================================================================
# USER SETTINGS
# ======================================================================

if "user_settings" not in st.session_state:
    st.session_state.user_settings = load_settings(
        st.session_state.user_id
    )

settings = st.session_state.user_settings


# ======================================================================
# SIDEBAR ACCOUNT
# ======================================================================

st.sidebar.divider()

st.sidebar.write(
    f"Logged in as: {st.session_state.user_email}"
)

if st.sidebar.button("Logout"):

    # Clear account-specific UI state as well.
    st.session_state.pop("user_settings", None)
    st.session_state.pop("custom_weather_base", None)
    st.session_state.pop("custom_weather_editor", None)

    logout()
    st.rerun()


# ======================================================================
# DATA / OPTIMIZATION
# ======================================================================

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
        {
            "deep_discharge": w_depth,
            "throughput": w_through,
            "curtailment": w_curtail,
        }
    )

    if weather_mode == "live":

        weather = fetch_weather(
            cfg.site.latitude,
            cfg.site.longitude,
            hours=horizon,
            timezone=cfg.site.timezone,
        )

    elif weather_mode == "custom":

        # custom_weather_data =
        # (
        #     conditions,
        #     temps,
        #     peak_ghi,
        #     sunrise_h,
        #     sunset_h
        # )

        conditions, temps, peak_ghi, sunrise_h, sunset_h = custom_weather_data

        weather = weather_from_conditions(
            list(conditions),
            list(temps),
            peak_ghi=peak_ghi,
            sunrise_h=sunrise_h,
            sunset_h=sunset_h,
        )

    else:

        weather = get_scenario(
            weather_mode,
            hours=horizon,
        )

    pv = forecast_pv(
        weather,
        cfg.pv,
    )

    tasks = default_workload(
        horizon_h=horizon,
        seed=seed,
        n_flexible=n_flexible,
    )

    plan = optimize(
        tasks,
        pv.pv_kwh,
        cfg.battery,
        cfg.optimizer,
        cfg.site.base_load_kw,
    )

    base = greedy_plan(
        tasks,
        pv.pv_kwh,
        cfg.battery,
        cfg.optimizer,
        cfg.site.base_load_kw,
    )

    trace = build_decision_trace(
        plan,
        tasks,
        cfg,
        timestamps=pv.timestamps,
    )

    return cfg, pv, tasks, plan, base, trace


# ======================================================================
# SIDEBAR
# ======================================================================

st.sidebar.title("Site controls")


# ----------------------------------------------------------------------
# Planning horizon
# ----------------------------------------------------------------------

horizon = st.sidebar.slider(
    "Planning horizon (hours)",
    12,
    24,
    int(settings.get("horizon", 24)),
)


# ----------------------------------------------------------------------
# Weather mode
# ----------------------------------------------------------------------

weather_options = ["live", "custom"] + list_scenarios()

weather_labels = {
    "live": "Live (Open-Meteo)",
    "custom": "Custom (enter your own)",
    **{
        k: k.replace("_", " ").title()
        for k in list_scenarios()
    },
}

saved_weather_mode = settings.get(
    "weather_mode",
    "custom",
)

if saved_weather_mode not in weather_options:
    saved_weather_mode = "custom"

weather_mode = st.sidebar.selectbox(
    "Weather",
    weather_options,
    index=weather_options.index(saved_weather_mode),
    format_func=lambda x: weather_labels.get(x, x),
)


if weather_mode in SCENARIO_BLURBS:
    st.sidebar.caption(
        SCENARIO_BLURBS[weather_mode]
    )


# ======================================================================
# CUSTOM WEATHER
# ======================================================================

custom_weather_data = None

if weather_mode == "custom":

    st.sidebar.caption(
        "Describe each hour the way you'd actually know it - a condition "
        "and a temperature. No irradiance numbers needed; the coordinator "
        "works those out on its own."
    )

    saved_custom = settings.get(
        "custom_weather_data",
        {},
    )

    if not isinstance(saved_custom, dict):
        saved_custom = {}

    # --------------------------------------------------------------
    # Advanced daylight settings
    # --------------------------------------------------------------

    with st.sidebar.expander(
        "Advanced: daylight shape",
        expanded=False,
    ):

        peak_ghi = st.slider(
            "Peak sun intensity (clear-sky reference, W/m²)",
            400.0,
            1100.0,
            float(
                saved_custom.get(
                    "peak_ghi",
                    900.0,
                )
            ),
            50.0,
            help=(
                "Only reached on a fully 'Clear' hour at solar noon - "
                "every other condition scales down from this."
            ),
        )

        saved_sunrise = int(
            saved_custom.get(
                "sunrise_h",
                6,
            )
        )

        saved_sunset = int(
            saved_custom.get(
                "sunset_h",
                18,
            )
        )

        # Make sure old/invalid database values cannot break the slider.
        saved_sunrise = max(
            0,
            min(23, saved_sunrise),
        )

        saved_sunset = max(
            saved_sunrise,
            min(23, saved_sunset),
        )

        sunrise_h, sunset_h = st.slider(
            "Daylight window (hour)",
            0,
            23,
            (saved_sunrise, saved_sunset),
        )


    # --------------------------------------------------------------
    # Custom weather table
    # --------------------------------------------------------------

    condition_options = list(
        CONDITIONS.keys()
    )

    saved_conditions = saved_custom.get(
        "conditions",
        [],
    )

    saved_temps = saved_custom.get(
        "temps",
        [],
    )


    # Rebuild the editor when:
    # 1. there is no existing editor data
    # 2. the planning horizon changed
    #
    # If saved settings exist for this exact horizon, load them.
    if (
        "custom_weather_base" not in st.session_state
        or len(
            st.session_state["custom_weather_base"]
        ) != horizon
    ):

        if (
            len(saved_conditions) == horizon
            and len(saved_temps) == horizon
        ):

            st.session_state["custom_weather_base"] = pd.DataFrame(
                {
                    "Hour": list(range(horizon)),
                    "Condition": list(saved_conditions),
                    "Temp (°C)": list(saved_temps),
                }
            )

        else:

            st.session_state["custom_weather_base"] = pd.DataFrame(
                {
                    "Hour": list(range(horizon)),
                    "Condition": ["Sunny"] * horizon,
                    "Temp (°C)": [24.0] * horizon,
                }
            )


    # --------------------------------------------------------------
    # Data editor
    # --------------------------------------------------------------

    edited_df = st.sidebar.data_editor(
        st.session_state["custom_weather_base"],
        disabled=["Hour"],
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Condition": st.column_config.SelectboxColumn(
                options=condition_options,
            ),
            "Temp (°C)": st.column_config.NumberColumn(
                step=0.5,
            ),
        },
        key=f"custom_weather_editor_{st.session_state.user_id}",
    )


    # Tuples keep this object hashable for st.cache_data.
    custom_weather_data = (
        tuple(
            edited_df["Condition"].tolist()
        ),
        tuple(
            edited_df["Temp (°C)"].tolist()
        ),
        float(peak_ghi),
        int(sunrise_h),
        int(sunset_h),
    )


# ======================================================================
# WORKLOAD SETTINGS
# ======================================================================

seed = st.sidebar.number_input(
    "Workload seed",
    value=int(
        settings.get(
            "seed",
            7,
        )
    ),
    step=1,
)


n_flex = st.sidebar.slider(
    "Deferrable jobs",
    0,
    12,
    int(
        settings.get(
            "n_flex",
            6,
        )
    ),
)


# ======================================================================
# BATTERY PROTECTION
# ======================================================================

st.sidebar.subheader(
    "Battery protection"
)


soc_min = st.sidebar.slider(
    "Hard SoC floor",
    0.05,
    0.4,
    float(
        settings.get(
            "soc_min",
            0.20,
        )
    ),
    0.01,
)


soc_comfort = st.sidebar.slider(
    "Comfort SoC floor",
    0.2,
    0.7,
    float(
        settings.get(
            "soc_comfort",
            0.40,
        )
    ),
    0.05,
)


# ======================================================================
# OBJECTIVE WEIGHTS
# ======================================================================

st.sidebar.subheader(
    "Objective weights"
)


w_depth = st.sidebar.slider(
    "Deep-discharge penalty",
    0.0,
    100.0,
    float(
        settings.get(
            "w_depth",
            25.0,
        )
    ),
    5.0,
)


w_through = st.sidebar.slider(
    "Throughput (wear) weight",
    0.0,
    10.0,
    float(
        settings.get(
            "w_through",
            1.0,
        )
    ),
    0.5,
)


w_curtail = st.sidebar.slider(
    "Curtailment penalty",
    0.0,
    1.0,
    float(
        settings.get(
            "w_curtail",
            0.05,
        )
    ),
    0.05,
)


# ======================================================================
# SAVE ALL SETTINGS
# ======================================================================

# Preserve previously saved custom weather when the user is currently
# using Live/Synthetic weather.
#
# This is important because otherwise switching away from Custom would
# overwrite the user's saved custom weather with empty values.

if weather_mode == "custom" and custom_weather_data is not None:

    saved_custom_weather = {
        "conditions": list(
            custom_weather_data[0]
        ),
        "temps": list(
            custom_weather_data[1]
        ),
        "peak_ghi": float(
            custom_weather_data[2]
        ),
        "sunrise_h": int(
            custom_weather_data[3]
        ),
        "sunset_h": int(
            custom_weather_data[4]
        ),
    }

else:

    saved_custom_weather = settings.get(
        "custom_weather_data",
        {
            "conditions": [],
            "temps": [],
            "peak_ghi": 900.0,
            "sunrise_h": 6,
            "sunset_h": 18,
        },
    )


new_settings = {
    # Weather
    "weather_mode": weather_mode,

    # Custom weather
    "custom_weather_data": saved_custom_weather,

    # Planning
    "horizon": int(horizon),
    "seed": int(seed),
    "n_flex": int(n_flex),

    # Battery
    "soc_min": float(soc_min),
    "soc_comfort": float(soc_comfort),

    # Objective
    "w_depth": float(w_depth),
    "w_through": float(w_through),
    "w_curtail": float(w_curtail),
}


# Only write to SQLite if something actually changed.
if new_settings != st.session_state.user_settings:

    save_settings(
        st.session_state.user_id,
        new_settings,
    )

    st.session_state.user_settings = new_settings

    # Keep the local reference synchronized too.
    settings = new_settings


# ======================================================================
# BUILD MODEL
# ======================================================================

cfg, pv, tasks, plan, base, trace = build_everything(
    weather_mode,
    horizon,
    int(seed),
    n_flex,
    soc_min,
    soc_comfort,
    w_depth,
    w_through,
    w_curtail,
    custom_weather_data,
)


score = score_plan(
    plan,
    tasks,
    cfg.battery,
    cfg.optimizer.weights,
)

score_base = score_plan(
    base,
    tasks,
    cfg.battery,
    cfg.optimizer.weights,
)

times = pv.timestamps

by_id = {
    t.id: t
    for t in tasks
}


# ======================================================================
# HEADER
# ======================================================================

st.title(
    cfg.site.name
)


if pv.source == "open-meteo":

    badge_color = "#1E7A46"
    badge_bg = "#12291D"
    badge_text = (
        "🟢 LIVE — Open-Meteo real-time forecast"
    )

elif pv.source == "custom":

    badge_color = "#8C4FB0"
    badge_bg = "#241533"
    badge_text = (
        "🟣 CUSTOM — user-entered weather"
    )

elif pv.source.startswith("synthetic:"):

    scenario_id = pv.source.split(
        ":",
        1,
    )[1]

    badge_color = "#3B6FB0"
    badge_bg = "#12203A"

    badge_text = (
        "🔵 SYNTHETIC SCENARIO — "
        f"{scenario_id.replace('_', ' ').title()}"
    )

else:

    badge_color = "#B0803B"
    badge_bg = "#332510"

    badge_text = (
        "🟠 FALLBACK — live weather unreachable, "
        "showing a synthetic day instead"
    )


st.markdown(
    f"""
    <div style="
        display:inline-block;
        padding:6px 14px;
        border-radius:6px;
        border:1px solid {badge_color};
        background:{badge_bg};
        color:{badge_color};
        font-weight:600;
        font-size:0.95rem;
        margin-bottom:8px;
    ">
        {badge_text}
    </div>
    """,
    unsafe_allow_html=True,
)


st.caption(
    f"{pv.total_kwh:.2f} kWh of sun expected over "
    f"{plan.horizon}h · "
    f"solver {plan.status} in "
    f"{plan.solve_seconds}s"
)


# ======================================================================
# METRICS
# ======================================================================

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Critical deadlines missed",
    score.critical_deadline_misses,
    delta=(
        score.critical_deadline_misses
        - score_base.critical_deadline_misses
    ),
    delta_color="inverse",
)


c2.metric(
    "Load met directly by sun",
    f"{score.solar_share_pct}%",
    delta=(
        f"{score.solar_share_pct - score_base.solar_share_pct:+.1f} pts"
    ),
)


c3.metric(
    "Battery throughput",
    f"{score.battery_throughput_kwh:.2f} kWh",
    delta=(
        f"{score.battery_throughput_kwh - score_base.battery_throughput_kwh:+.2f}"
    ),
    delta_color="inverse",
)


c4.metric(
    "Lowest state of charge",
    f"{score.min_soc_pct}%",
    delta=(
        f"{score.min_soc_pct - score_base.min_soc_pct:+.1f} pts"
    ),
)


# ======================================================================
# TABS
# ======================================================================

tab_plan, tab_energy, tab_score, tab_ask = st.tabs(
    [
        "Plan",
        "Energy",
        "Scorecard",
        "Ask the coordinator",
    ]
)


# ======================================================================
# PLAN
# ======================================================================

with tab_plan:

    rows = []

    for tk in tasks:

        for slot in plan.schedule.get(
            tk.id,
            [],
        ):

            rows.append(
                dict(
                    Task=f"{tk.id} · {tk.name}",
                    Start=times[slot],
                    Finish=(
                        times[slot]
                        + timedelta(hours=1)
                    ),
                    Priority=tk.priority,
                    OnSun=(
                        pv.pv_kwh[slot]
                        >= tk.power_kw
                    ),
                )
            )


    fig = go.Figure()


    fig.add_trace(
        go.Bar(
            x=pv.pv_kwh,
            y=[
                t.strftime("%H:%M")
                for t in times
            ],
            orientation="h",
            marker_color=SOLAR,
            opacity=0.18,
            name="solar available",
            hovertemplate=(
                "%{y} · %{x:.2f} kWh"
                "<extra></extra>"
            ),
        )
    )


    if rows:

        df = pd.DataFrame(rows)

        order = {
            "critical": RISK,
            "high": STORED,
            "flexible": DRAW,
        }

        for priority, group in df.groupby(
            "Priority"
        ):

            fig.add_trace(
                go.Bar(
                    x=[1] * len(group),
                    y=group["Start"].dt.strftime(
                        "%H:%M"
                    ),
                    orientation="h",
                    name=priority,
                    marker_color=order[priority],
                    text=group["Task"],
                    textposition="inside",
                    hovertemplate=(
                        "%{text}"
                        "<extra></extra>"
                    ),
                )
            )


    fig.update_layout(
        barmode="stack",
        height=720,
        template="plotly_dark",
        paper_bgcolor=INK,
        plot_bgcolor=INK,
        yaxis=dict(
            autorange="reversed",
            title=None,
        ),
        xaxis_title=(
            "tasks running (stacked) "
            "over solar availability"
        ),
        legend=dict(
            orientation="h",
            y=1.05,
        ),
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
    )


    st.plotly_chart(
        fig,
        width="stretch",
    )


    st.subheader(
        "Task decisions"
    )


    table = []

    for tk in tasks:

        slots = plan.schedule.get(
            tk.id,
            [],
        )

        table.append(
            {
                "id": tk.id,
                "task": tk.name,
                "priority": tk.priority,
                "kW": tk.power_kw,
                "hours": tk.duration_h,
                "window": (
                    f"{tk.release_h}-{tk.deadline_h}"
                ),
                "slack": tk.slack_h,
                "scheduled": (
                    ", ".join(
                        times[s].strftime("%H:%M")
                        for s in slots
                    )
                    or "—"
                ),
                "done": (
                    "yes"
                    if plan.completed.get(tk.id)
                    else "no"
                ),
            }
        )


    st.dataframe(
        pd.DataFrame(table),
        width="stretch",
        hide_index=True,
    )


    pick = st.selectbox(
        "Explain one decision",
        [t.id for t in tasks],
    )


    if st.button(
        "Why this schedule?"
    ):

        st.info(
            explain_task(
                pick,
                trace,
                cfg,
            )
        )


# ======================================================================
# ENERGY
# ======================================================================

with tab_energy:

    H = plan.horizon

    labels = [
        t.strftime("%H:%M")
        for t in times[:H]
    ]


    pv_direct = [
        max(
            0.0,
            min(
                plan.pv_used_kwh[t]
                - plan.charge_kwh[t],
                plan.load_kwh[t],
            ),
        )
        for t in range(H)
    ]


    energy = go.Figure()


    energy.add_bar(
        x=labels,
        y=pv_direct,
        name="sun → load",
        marker_color=SOLAR,
    )


    energy.add_bar(
        x=labels,
        y=plan.discharge_kwh[:H],
        name="battery → load",
        marker_color=STORED,
    )


    energy.add_bar(
        x=labels,
        y=[
            -c
            for c in plan.charge_kwh[:H]
        ],
        name="→ battery",
        marker_color=DRAW,
    )


    energy.add_scatter(
        x=labels,
        y=plan.pv_available_kwh[:H],
        name="solar available",
        mode="lines",
        line=dict(
            color=SOLAR,
            dash="dot",
        ),
    )


    energy.add_scatter(
        x=labels,
        y=plan.load_kwh[:H],
        name="total load",
        mode="lines+markers",
        line=dict(
            color="#E9ECF1"
        ),
    )


    energy.update_layout(
        barmode="relative",
        template="plotly_dark",
        height=420,
        paper_bgcolor=INK,
        plot_bgcolor=INK,
        yaxis_title="kWh per hour",
        legend=dict(
            orientation="h",
            y=1.12,
        ),
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
    )


    st.plotly_chart(
        energy,
        width="stretch",
    )


    cap = cfg.battery.capacity_kwh


    soc = go.Figure()


    soc.add_scatter(
        x=labels + [labels[-1]],
        y=[
            100 * s / cap
            for s in plan.soc_kwh
        ],
        name="planned SoC",
        line=dict(
            color=STORED,
            width=3,
        ),
    )


    soc.add_scatter(
        x=labels + [labels[-1]],
        y=[
            100 * s / cap
            for s in base.soc_kwh
        ],
        name="greedy SoC",
        line=dict(
            color=DRAW,
            dash="dash",
        ),
    )


    soc.add_hline(
        y=100 * cfg.battery.soc_comfort_frac,
        line_dash="dot",
        line_color=SOLAR,
        annotation_text="comfort floor",
    )


    soc.add_hline(
        y=100 * cfg.battery.soc_min_frac,
        line_color=RISK,
        annotation_text="hard floor",
    )


    soc.update_layout(
        template="plotly_dark",
        height=360,
        paper_bgcolor=INK,
        plot_bgcolor=INK,
        yaxis_title="state of charge (%)",
        legend=dict(
            orientation="h",
            y=1.15,
        ),
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
    )


    st.plotly_chart(
        soc,
        width="stretch",
    )


# ======================================================================
# SCORECARD
# ======================================================================

with tab_score:

    comparison = pd.DataFrame(
        {
            "Coordinator (MILP)": score.to_dict(),
            "Greedy baseline": score_base.to_dict(),
        }
    )


    st.dataframe(
        comparison,
        width="stretch",
    )


    st.subheader(
        "What the objective paid for"
    )


    costs = pd.DataFrame(
        {
            "MILP": plan.cost_breakdown,
            "Greedy": base.cost_breakdown,
        }
    ).round(2)


    st.bar_chart(
        costs
    )


# ======================================================================
# ASK
# ======================================================================

with tab_ask:

    st.write(
        template_explanation(trace)
    )


    question = st.text_input(
        "Ask about the plan",
        placeholder=(
            "Why is the vibration sweep running "
            "at 07:00 instead of at sunrise?"
        ),
    )


    if question:

        with st.spinner(
            "Reading the decision trace..."
        ):

            st.markdown(
                answer_question(
                    question,
                    trace,
                    cfg,
                )
            )


    with st.expander(
        "Raw decision trace "
        "(what the LLM is allowed to see)"
    ):

        st.json(trace)
