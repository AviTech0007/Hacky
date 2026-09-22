"""A small library of named synthetic weather days.

`synthetic_weather()` in weather.py gives you exactly one shape: clear sky
plus an afternoon cloud bank. That is not enough to demo "what does the
coordinator do differently when conditions change" — which is one of the
most convincing things you can show a judge live. This file is that: a set
of reproducible, hand-tuned scenarios, each built to stress a different part
of the optimizer.

Every scenario returns a WeatherSeries, so it drops straight into
forecast_pv() exactly like the live API or the plain synthetic fallback does.
Nothing downstream needs to know the difference.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List

from src.data.weather import WeatherSeries


def _base_curve(hours: int, peak_ghi: float, sunrise_h: int, sunset_h: int) -> List[float]:
    curve = []
    for h in range(hours):
        if sunrise_h <= h <= sunset_h:
            phase = (h - sunrise_h) / (sunset_h - sunrise_h)
            curve.append(peak_ghi * math.sin(math.pi * phase))
        else:
            curve.append(0.0)
    return curve


def _to_series(
    ghi: List[float],
    clouds: List[float],
    start: datetime | None = None,
    temp_base: float = 22.0,
    temp_swing: float = 9.0,
    peak_ghi: float = 900.0,
    source: str = "synthetic",
) -> WeatherSeries:
    start = start or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    hours = len(ghi)
    timestamps = [start + timedelta(hours=i) for i in range(hours)]
    temps = [temp_base + temp_swing * (v / peak_ghi if peak_ghi else 0) for v in ghi]
    return WeatherSeries(timestamps, ghi, temps, clouds, source=source)


# --------------------------------------------------------------- presets ---
def clear_day(hours: int = 24, start: datetime | None = None) -> WeatherSeries:
    """Best case: cloudless, full solar budget. The floor for comparison."""
    ghi = _base_curve(hours, peak_ghi=950.0, sunrise_h=6, sunset_h=18)
    clouds = [3.0 if v > 0 else 0.0 for v in ghi]
    return _to_series(ghi, clouds, start, peak_ghi=950.0, source="synthetic:clear_day")


def afternoon_clouds(hours: int = 24, start: datetime | None = None) -> WeatherSeries:
    """The current default: clear morning, a cloud bank rolls in 13:00-16:00.

    Good for showing deferred flexible tasks pulled earlier to beat the dip.
    """
    ghi = _base_curve(hours, peak_ghi=900.0, sunrise_h=6, sunset_h=18)
    clouds = [5.0 if v > 0 else 0.0 for v in ghi]
    for h in range(13, min(16, hours)):
        ghi[h] *= 0.35
        clouds[h] = 65.0
    return _to_series(ghi, clouds, start, peak_ghi=900.0, source="synthetic:afternoon_clouds")


def overcast_day(hours: int = 24, start: datetime | None = None) -> WeatherSeries:
    """Heavy cloud cover all day. Forces the optimizer to lean on the battery
    and drop low-priority flexible tasks — the scenario that shows the
    deep-discharge penalty and missed-task trade-off doing real work."""
    ghi = _base_curve(hours, peak_ghi=900.0, sunrise_h=6, sunset_h=18)
    ghi = [v * 0.22 for v in ghi]
    clouds = [85.0 if v > 0 else 20.0 for v in ghi]
    return _to_series(ghi, clouds, start, peak_ghi=900.0, source="synthetic:overcast_day")


def storm_then_clear(hours: int = 24, start: datetime | None = None) -> WeatherSeries:
    """A storm knocks out generation for the morning, then it clears.
    Stresses the critical-deadline logic: T01 (release 0, deadline 5) has
    almost no sun in its window and must be served from the battery."""
    ghi = _base_curve(hours, peak_ghi=950.0, sunrise_h=6, sunset_h=18)
    clouds = [5.0 if v > 0 else 0.0 for v in ghi]
    for h in range(6, min(12, hours)):
        ghi[h] *= 0.08
        clouds[h] = 95.0
    return _to_series(ghi, clouds, start, peak_ghi=950.0, source="synthetic:storm_then_clear")


def winter_short_day(hours: int = 24, start: datetime | None = None) -> WeatherSeries:
    """Short daylight window, lower peak irradiance, cooler temperatures.
    Everything has to fit into a narrower sunny band."""
    ghi = _base_curve(hours, peak_ghi=550.0, sunrise_h=8, sunset_h=16)
    clouds = [10.0 if v > 0 else 0.0 for v in ghi]
    return _to_series(
        ghi, clouds, start, temp_base=8.0, temp_swing=6.0, peak_ghi=550.0,
        source="synthetic:winter_short_day",
    )


def intermittent_clouds(hours: int = 24, start: datetime | None = None, seed: int = 11) -> WeatherSeries:
    """Passing clouds all day - noisy, not a clean dip. The hardest case for
    a naive rule-based controller, which tends to thrash charge/discharge
    every time a cloud passes; the MILP plans through the noise instead."""
    rng = random.Random(seed)
    ghi = _base_curve(hours, peak_ghi=900.0, sunrise_h=6, sunset_h=18)
    clouds = []
    for h in range(hours):
        if ghi[h] > 0:
            factor = max(0.15, 1.0 - abs(rng.gauss(0.0, 0.28)))
            ghi[h] *= factor
            clouds.append(round((1 - factor) * 100, 1))
        else:
            clouds.append(0.0)
    return _to_series(ghi, clouds, start, peak_ghi=900.0, source="synthetic:intermittent_clouds")


SCENARIOS: Dict[str, Callable[..., WeatherSeries]] = {
    "clear_day": clear_day,
    "afternoon_clouds": afternoon_clouds,
    "overcast_day": overcast_day,
    "storm_then_clear": storm_then_clear,
    "winter_short_day": winter_short_day,
    "intermittent_clouds": intermittent_clouds,
}

SCENARIO_BLURBS: Dict[str, str] = {
    "clear_day": "Cloudless best case - the solar budget the other scenarios are measured against.",
    "afternoon_clouds": "Clear morning, cloud bank 13:00-16:00. Watch flexible tasks get pulled earlier to beat it.",
    "overcast_day": "Heavy cloud all day. Forces battery reliance and drops low-priority tasks.",
    "storm_then_clear": "Dead morning, clears by noon. Stress-tests early critical-task deadlines.",
    "winter_short_day": "Short, weak daylight window (08:00-16:00). Everything must fit in a narrow band.",
    "intermittent_clouds": "Noisy passing clouds all day. Shows the MILP planning through noise instead of thrashing.",
}


def get_scenario(name: str, hours: int = 24, start: datetime | None = None, **kwargs) -> WeatherSeries:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Choose from: {list(SCENARIOS)}")
    return SCENARIOS[name](hours=hours, start=start, **kwargs)


def list_scenarios() -> List[str]:
    return list(SCENARIOS.keys())
