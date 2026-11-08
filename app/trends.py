"""AI Function 3: Gradual increase/decrease detection & exact rate estimation.

Fits an Ordinary Least Squares (OLS) line y(t) = a + b*t over a sliding window
for each channel (CO2, CH4, Temperature, Humidity):
  - Slope b (converted to units/minute) is the exact rate of increase (>0) or
    decrease (<0).
  - Coefficient of determination R^2 measures how consistently monotonic the
    change is. Gating on R^2 >= TREND_MIN_R2 separates a genuine gradual leak or
    thermal rise from zero-mean random walk noise.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import numpy as np

from . import settings


@dataclass
class ChannelTrend:
    channel: str
    direction: str          # "INCREASING", "DECREASING", "STABLE"
    rate_per_min: float     # exact slope in units / minute
    rate_per_sec: float     # exact slope in units / second
    r_squared: float        # 0.0 .. 1.0 goodness of linear fit
    projected_5min: float   # forecasted value 5 minutes ahead
    is_gradual_hazard: bool
    unit: str


class TrendAnalyzer:
    def __init__(self) -> None:
        self._history: Dict[str, Deque[Tuple[float, float]]] = {
            "co2":  deque(maxlen=settings.TREND_WINDOW_SAMPLES),
            "ch4":  deque(maxlen=settings.TREND_WINDOW_SAMPLES),
            "temp": deque(maxlen=settings.TREND_WINDOW_SAMPLES),
            "hum":  deque(maxlen=settings.TREND_WINDOW_SAMPLES),
        }

    def reset(self) -> None:
        for q in self._history.values():
            q.clear()

    @staticmethod
    def _ols_slope_r2(samples: Deque[Tuple[float, float]]) -> Tuple[float, float]:
        n = len(samples)
        if n < settings.TREND_MIN_SAMPLES:
            return 0.0, 0.0
        arr = np.asarray(samples, dtype=float)
        t = arr[:, 0] - arr[0, 0]
        y = arr[:, 1]
        span = float(t[-1] - t[0])
        if span < 1.5:
            return 0.0, 0.0

        t_mean = float(np.mean(t))
        y_mean = float(np.mean(y))
        dt = t - t_mean
        dy = y - y_mean
        ss_tt = float(np.dot(dt, dt))
        ss_yy = float(np.dot(dy, dy))
        if ss_tt < 1e-9 or ss_yy < 1e-9:
            return 0.0, 0.0

        slope_per_sec = float(np.dot(dt, dy) / ss_tt)
        r = float(np.dot(dt, dy) / np.sqrt(ss_tt * ss_yy))
        return slope_per_sec, round(r * r, 3)

    def update(
        self,
        ts_sec: float,
        co2_ppm: float,
        ch4_ppm: float,
        temp_c: float,
        hum_pct: float,
    ) -> Dict[str, ChannelTrend]:
        values = {"co2": co2_ppm, "ch4": ch4_ppm, "temp": temp_c, "hum": hum_pct}
        units  = {"co2": "ppm/min", "ch4": "ppm/min", "temp": "°C/min", "hum": "%RH/min"}
        sig_thresholds = {
            "co2":  settings.RATE_SIGNIFICANT_CO2_PPM_MIN,
            "ch4":  settings.RATE_SIGNIFICANT_CH4_PPM_MIN,
            "temp": settings.RATE_SIGNIFICANT_TEMP_C_MIN,
            "hum":  settings.RATE_SIGNIFICANT_HUM_PCT_MIN,
        }

        out: Dict[str, ChannelTrend] = {}
        for key, val in values.items():
            self._history[key].append((ts_sec, val))
            slope_s, r2 = self._ols_slope_r2(self._history[key])
            rate_min = slope_s * 60.0
            proj_5m  = val + rate_min * 5.0

            sig = sig_thresholds[key]
            if abs(rate_min) >= sig and r2 >= settings.TREND_MIN_R2:
                direction = "INCREASING" if rate_min > 0 else "DECREASING"
            else:
                direction = "STABLE"

            # Gradual hazard: sustained upward creep on gas/temp or extreme RH drift
            if key in ("co2", "ch4"):
                gradual_hazard = (direction == "INCREASING")
            elif key == "temp":
                gradual_hazard = (direction != "STABLE") and (
                    proj_5m >= settings.TEMP_WARN_HIGH_C or proj_5m <= settings.TEMP_WARN_LOW_C
                )
            else:
                gradual_hazard = (direction != "STABLE") and (
                    proj_5m >= settings.HUM_WARN_HIGH_PCT or proj_5m <= settings.HUM_WARN_LOW_PCT
                )

            out[key] = ChannelTrend(
                channel=key,
                direction=direction,
                rate_per_min=round(rate_min, 2),
                rate_per_sec=round(slope_s, 3),
                r_squared=r2,
                projected_5min=round(proj_5m, 1),
                is_gradual_hazard=gradual_hazard,
                unit=units[key],
            )
        return out
