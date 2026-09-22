"""Forecast module: weather -> expected PV energy per hourly slot.

This is the first stage of the coordinator. The optimizer never sees raw
irradiance; it only sees kWh available in each slot, which is what this
produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from src.config import PVConfig
from src.data.weather import WeatherSeries


@dataclass
class PVForecast:
    timestamps: List[datetime]
    pv_kwh: List[float]          # usable PV energy per 1-hour slot
    ghi_w_m2: List[float]
    cell_temp_c: List[float]
    source: str

    @property
    def total_kwh(self) -> float:
        return sum(self.pv_kwh)

    def peak_hours(self, top_n: int = 4) -> List[int]:
        return sorted(range(len(self.pv_kwh)), key=lambda i: -self.pv_kwh[i])[:top_n]


def forecast_pv(weather: WeatherSeries, pv: PVConfig, dt_h: float = 1.0) -> PVForecast:
    """Flat-plate array model.

    P = P_peak * (GHI / 1000) * derate * [1 + temp_coeff * (T_cell - T_ref)]

    Cell temperature is approximated with the standard NOCT-style linear rule
    T_cell ~= T_air + 0.03 * GHI, which is accurate enough for planning and
    keeps the module dependency-free.
    """
    pv_kwh: List[float] = []
    cell_temps: List[float] = []

    for ghi, t_air in zip(weather.ghi_w_m2, weather.temp_c):
        t_cell = t_air + 0.03 * ghi
        temp_factor = 1.0 + pv.temp_coeff_per_c * (t_cell - pv.ref_temp_c)
        power_kw = pv.peak_power_kw * (ghi / 1000.0) * pv.derate * max(0.0, temp_factor)
        power_kw = min(max(0.0, power_kw), pv.peak_power_kw)
        pv_kwh.append(power_kw * dt_h)
        cell_temps.append(t_cell)

    return PVForecast(
        timestamps=list(weather.timestamps),
        pv_kwh=pv_kwh,
        ghi_w_m2=list(weather.ghi_w_m2),
        cell_temp_c=cell_temps,
        source=weather.source,
    )


def apply_forecast_error(pv_kwh: List[float], seed: int = 0, sigma: float = 0.15) -> List[float]:
    """Perturb a forecast to stand in for 'what actually happened'.

    Used by the rolling-horizon loop so the coordinator has to re-plan against
    reality rather than against its own prediction.
    """
    import random

    rng = random.Random(seed)
    return [max(0.0, v * (1.0 + rng.gauss(0.0, sigma))) for v in pv_kwh]
