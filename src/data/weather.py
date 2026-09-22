"""Data layer: pulls real hourly irradiance from Open-Meteo (free, no API key).

Falls back to a deterministic synthetic clear-sky day if the network is
unavailable, so a dead conference wifi never kills your demo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass
class WeatherSeries:
    """Hourly weather over the planning horizon."""

    timestamps: List[datetime]
    ghi_w_m2: List[float]        # global horizontal irradiance
    temp_c: List[float]
    cloud_cover_pct: List[float]
    source: str                  # "open-meteo" or "synthetic"

    def __len__(self) -> int:
        return len(self.timestamps)


def fetch_weather(
    latitude: float,
    longitude: float,
    hours: int = 24,
    timezone: str = "auto",
    timeout: float = 8.0,
    start_at_midnight: bool = True,
) -> WeatherSeries:
    """Try the live API first; degrade gracefully to a synthetic day.

    `start_at_midnight` keeps slot index equal to hour of day, which is what
    the default workload's release/deadline windows assume. Set it False for a
    live "from right now" horizon once you generate task windows dynamically.
    """
    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
                "forecast_days": 2,
                "timezone": timezone,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        hourly = response.json()["hourly"]
        times = [datetime.fromisoformat(t) for t in hourly["time"]]

        now = datetime.now()
        anchor = now.replace(minute=0, second=0, microsecond=0)
        if start_at_midnight:
            anchor = anchor.replace(hour=0)
        start = 0
        for i, t in enumerate(times):
            if t >= anchor:
                start = i
                break
        end = start + hours

        return WeatherSeries(
            timestamps=times[start:end],
            ghi_w_m2=[float(v or 0.0) for v in hourly["shortwave_radiation"][start:end]],
            temp_c=[float(v or 25.0) for v in hourly["temperature_2m"][start:end]],
            cloud_cover_pct=[float(v or 0.0) for v in hourly["cloud_cover"][start:end]],
            source="open-meteo",
        )
    except Exception:
        return synthetic_weather(hours=hours)


def custom_weather(
    ghi_w_m2: List[float],
    temp_c: List[float],
    cloud_cover_pct: List[float],
    start: datetime | None = None,
) -> WeatherSeries:
    """Build a WeatherSeries directly from user-entered hourly values.

    Backs the dashboard's manual weather input bar: a judge (or you, live)
    can type in exactly the GHI / temperature / cloud-cover numbers they
    want to test, using the same three fields the Open-Meteo and synthetic
    paths populate. Because the output is a plain WeatherSeries, forecast_pv,
    the optimizer, and the baseline don't need to special-case it at all.
    """
    n = len(ghi_w_m2)
    if not (len(temp_c) == n and len(cloud_cover_pct) == n):
        raise ValueError("ghi_w_m2, temp_c and cloud_cover_pct must all be the same length")

    start = start or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps = [start + timedelta(hours=i) for i in range(n)]

    return WeatherSeries(
        timestamps=timestamps,
        ghi_w_m2=[max(0.0, float(v)) for v in ghi_w_m2],
        temp_c=[float(v) for v in temp_c],
        cloud_cover_pct=[min(100.0, max(0.0, float(v))) for v in cloud_cover_pct],
        source="custom",
    )


CONDITIONS: dict[str, dict[str, float]] = {
    "Sunny": {"ghi_factor": 1.00, "cloud_pct": 0.0},
    "Clear": {"ghi_factor": 1.00, "cloud_pct": 5.0},
    "Partly cloudy": {"ghi_factor": 0.75, "cloud_pct": 35.0},
    "Overcast": {"ghi_factor": 0.35, "cloud_pct": 80.0},
    "Rainy": {"ghi_factor": 0.15, "cloud_pct": 95.0},
    "Stormy": {"ghi_factor": 0.05, "cloud_pct": 98.0},
}


def weather_from_conditions(
    conditions: List[str],
    temp_c: List[float],
    peak_ghi: float = 900.0,
    sunrise_h: int = 6,
    sunset_h: int = 18,
    start: datetime | None = None,
) -> WeatherSeries:
    """Build a WeatherSeries from hour-by-hour plain-language conditions.

    This is the human-friendly counterpart to custom_weather(): instead of
    asking someone to type irradiance in W/m^2 (almost nobody has an
    intuition for that number), they enter what they'd actually know - a
    condition ("Overcast") and a temperature - and this works out GHI and
    cloud cover on their behalf.

    Each condition maps to a GHI derating factor applied on top of the same
    sunrise/sunset clear-sky shape synthetic_weather() uses, plus a matching
    cloud-cover percentage, so the two stay physically consistent (unlike
    typing them in separately, where nothing stops "Overcast" + "GHI 900"
    from being entered together). Hours outside the daylight window are
    always zero regardless of the condition picked for them, so there is no
    way to accidentally generate power at night.
    """
    n = len(conditions)
    if len(temp_c) != n:
        raise ValueError("conditions and temp_c must be the same length")

    ghi, cloud = [], []
    for h in range(n):
        if sunrise_h <= h <= sunset_h and sunset_h != sunrise_h:
            phase = (h - sunrise_h) / (sunset_h - sunrise_h)
            clear_sky = peak_ghi * math.sin(math.pi * phase)
        else:
            clear_sky = 0.0

        profile = CONDITIONS.get(conditions[h], CONDITIONS["Clear"])
        ghi.append(round(max(0.0, clear_sky * profile["ghi_factor"]), 1))
        cloud.append(profile["cloud_pct"] if clear_sky > 0 else 0.0)

    return custom_weather(ghi, temp_c, cloud, start=start)


def synthetic_weather(
    hours: int = 24,
    start: datetime | None = None,
    peak_ghi: float = 850.0,
    sunrise_h: int = 6,
    sunset_h: int = 18,
    cloud_event: tuple[int, int, float] | None = (13, 16, 0.65),
) -> WeatherSeries:
    """A repeatable clear-sky bell curve with an optional afternoon cloud bank.

    The cloud event is what makes the demo interesting: the optimizer has to
    notice the dip coming and pull work earlier.
    """
    # Slot index == hour of day, which keeps task windows and the clock aligned.
    start = start or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps, ghi, temps, clouds = [], [], [], []

    for i in range(hours):
        ts = start + timedelta(hours=i)
        h = ts.hour
        if sunrise_h <= h <= sunset_h:
            phase = (h - sunrise_h) / (sunset_h - sunrise_h)
            value = peak_ghi * math.sin(math.pi * phase)
        else:
            value = 0.0

        cloud = 5.0
        if cloud_event and cloud_event[0] <= h < cloud_event[1]:
            value *= 1.0 - cloud_event[2]
            cloud = cloud_event[2] * 100

        timestamps.append(ts)
        ghi.append(max(0.0, value))
        temps.append(22.0 + 8.0 * (value / peak_ghi if peak_ghi else 0))
        clouds.append(cloud)

    return WeatherSeries(timestamps, ghi, temps, clouds, source="synthetic")
