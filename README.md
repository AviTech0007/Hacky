# Edge Energy Coordinator

A digital twin of an off-grid edge compute node — solar array, battery, and a
queue of sensor/compute jobs — with an optimization-based coordinator that
decides, hour by hour, what runs, what waits, and when the pack charges or
discharges.

Three objectives, enforced simultaneously by a single MILP:

1. critical-task deadlines are never missed
2. battery degradation (deep discharges, high C-rates, throughput) is minimised
3. solar energy is used preferentially over stored energy

## Run it

```bash
pip install -r requirements.txt
python -m scripts.run_sim --synthetic       # whole pipeline in the terminal
python -m scripts.run_sim --scenario overcast_day  # a named weather scenario
python -m scripts.run_sim --list-scenarios  # see all scenario names + blurbs
python -m scripts.compare_scenarios         # one table, all scenarios, MILP vs greedy
python -m pytest -q                         # 9 invariant tests
streamlit run app.py                        # the demo — scenario picker is in the sidebar
```

Six scenarios ship in `src/data/scenarios.py`: `clear_day`, `afternoon_clouds`
(the previous default), `overcast_day`, `storm_then_clear`, `winter_short_day`,
`intermittent_clouds`. Each is built to stress a different part of the
optimizer — run `compare_scenarios.py` once and you have your headline slide:
the coordinator holds zero critical-deadline misses across all six, while the
greedy baseline misses in half of them.

`--synthetic` skips the weather API entirely. The app also falls back to a
synthetic day automatically if Open-Meteo is unreachable, so bad conference
wifi cannot break your demo.

For the LLM explainer: `cp .env.example .env` and paste a free Groq key.
Without one, every explanation still works — it just uses the deterministic
template writer instead of the LLM.

## The pipeline, in execution order

| # | Stage | File | What it does |
|---|-------|------|--------------|
| 0 | Config | `config.yaml`, `src/config.py` | Every tunable number in one place |
| 1 | Data | `src/data/weather.py` | Open-Meteo hourly irradiance, with offline fallback |
| 1b | Data | `src/data/scenarios.py` | Named synthetic weather scenarios for demoing conditions |
| 2 | Data | `src/twin/tasks.py` | Task model + synthetic workload generator |
| 3 | Forecast | `src/coordinator/forecast.py` | Irradiance + temperature → kWh per slot |
| 4 | Optimize | `src/coordinator/optimizer.py` | **The MILP.** Schedule + charge plan |
| 4b | Baseline | `src/coordinator/baseline.py` | Greedy rule-based scheduler to beat |
| 5 | Twin | `src/twin/battery.py`, `src/twin/site.py` | Executes the plan against reality |
| 5b | Closed loop | `src/coordinator/loop.py` | Re-optimizes every hour (MPC) |
| 6 | Score | `src/metrics.py` | The KPIs judges will ask about |
| 7 | Explain | `src/coordinator/explainer.py` | Decision trace → plain language |
| 8 | Present | `app.py` | Streamlit dashboard |

Data flows strictly downhill: nothing in a lower layer imports from a higher
one. That is what makes each piece independently testable and independently
demoable.

## The MILP in one paragraph

Hourly slots over a 24-hour horizon. Binary `run[i,t]` puts task `i` in slot
`t`; non-preemptible tasks get start-variables so their slots stay contiguous.
Continuous variables handle PV use, charge, discharge and state of charge, with
a binary lock preventing simultaneous charge and discharge. Power balances
exactly each hour, SoC follows a round-trip-efficiency recursion, and the
objective prices five things against each other: unserved load, missed tasks
weighted by priority, battery throughput, depth-of-discharge below a comfort
band, and C-rate stress — plus a small penalty for curtailing free sunlight,
which is what makes the solver pull flexible work into the middle of the day.

An `unserved[t]` slack variable with a very large penalty guarantees the model
is always feasible. It never blacks out silently; it tells you it had to.

## Where to extend it

- **Better forecast** — swap the flat-plate model in `forecast.py` for a
  tilt/azimuth model, or blend Open-Meteo's cloud cover into an ensemble.
- **Thermal coupling** — compute load raises node temperature, which lowers
  panel and battery efficiency. Add it to `battery.py` and mirror it in the MILP.
- **Multi-node** — one MILP over several sites sharing a task queue.
- **Learned degradation** — fit `cycle_cost_per_kwh` from a real cell dataset
  instead of assuming it.
