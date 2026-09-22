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
