"""AI Function 2: Sudden spike detector across gas, temperature, and humidity.

Uses a two-tier detector on every incoming sample:
  1. Per-channel Modified Z-Score on first differences (dx = x[t] - x[t-1]):
         z_mod = 0.6745 * |dx - median(dx)| / MAD(dx)
     Because MAD (median absolute deviation) is immune to outliers, a single
     sharp jump produces a large z_mod immediately without inflating the scale
     estimate. A physical minimum step floor prevents sensor quantisation noise
     on a dead-flat baseline from triggering false spikes.
  2. Multivariate IsolationForest (trained in train_model.py) over the 8-vector
     [co2, ch4, temp, hum, d_co2, d_ch4, d_temp, d_hum] to catch correlated
     multi-sensor bursts (e.g. flash fire = simultaneous CO2 + Temp jump + RH drop).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from . import settings


@dataclass
class ChannelSpike:
    channel: str
    triggered: bool
    delta: float
    delta_per_sec: float
    z_score: float
    unit: str


@dataclass
class SpikeReport:
    any_spike: bool
    isolation_forest_anomaly: bool
    anomaly_score: float
    triggered_channels: List[str] = field(default_factory=list)
    channels: Dict[str, ChannelSpike] = field(default_factory=dict)


class SuddenSpikeDetector:
    def __init__(self, model_bundle: Optional[Dict[str, Any]] = None) -> None:
        self.iforest = model_bundle.get("spike_iforest") if model_bundle else None
        self.scaler  = model_bundle.get("spike_scaler") if model_bundle else None
        self._prev: Optional[Dict[str, float]] = None
        self._prev_ts: Optional[float] = None
        self._deltas: Dict[str, Deque[float]] = {
            "co2":  deque(maxlen=settings.SPIKE_WINDOW_SAMPLES),
            "ch4":  deque(maxlen=settings.SPIKE_WINDOW_SAMPLES),
            "temp": deque(maxlen=settings.SPIKE_WINDOW_SAMPLES),
            "hum":  deque(maxlen=settings.SPIKE_WINDOW_SAMPLES),
        }

    def reset(self) -> None:
        self._prev = None
        self._prev_ts = None
        for d in self._deltas.values():
            d.clear();

    @staticmethod
    def _modified_z(history: Deque[float], current_dx: float, floor_step: float) -> float:
        if len(history) < 4:
            return abs(current_dx) / max(1e-6, floor_step) * 2.0
        arr = np.asarray(history, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        robust_sigma = max(mad / 0.6745, floor_step * 0.25)
        return abs(current_dx - med) / robust_sigma

    def update(
        self,
        ts_sec: float,
        co2_ppm: float,
        ch4_ppm: float,
        temp_c: float,
        hum_pct: float,
    ) -> SpikeReport:
        current = {"co2": co2_ppm, "ch4": ch4_ppm, "temp": temp_c, "hum": hum_pct}
        units   = {"co2": "ppm", "ch4": "ppm", "temp": "°C", "hum": "%RH"}
        floors  = {
            "co2":  settings.SPIKE_MIN_DELTA_CO2,
            "ch4":  settings.SPIKE_MIN_DELTA_CH4,
            "temp": settings.SPIKE_MIN_DELTA_TEMP,
            "hum":  settings.SPIKE_MIN_DELTA_HUM,
        }

        if self._prev is None or self._prev_ts is None:
            self._prev = current
            self._prev_ts = ts_sec
            empty = {
                k: ChannelSpike(k, False, 0.0, 0.0, 0.0, units[k]) for k in current
            }
            return SpikeReport(False, False, 0.0, [], empty)

        dt = max(0.1, min(10.0, ts_sec - self._prev_ts))
        self._prev_ts = ts_sec

        ch_reports: Dict[str, ChannelSpike] = {}
        triggered: List[str] = []
        d_vec: Dict[str, float] = {}

        for key, val in current.items():
            dx = val - self._prev[key]
            d_vec[key] = dx / dt
            z = self._modified_z(self._deltas[key], dx, floors[key])
            self._deltas[key].append(dx)

            is_spike = (abs(dx) >= floors[key]) and (z >= settings.SPIKE_Z_THRESHOLD)
            if is_spike:
                triggered.append(key)

            ch_reports[key] = ChannelSpike(
                channel=key,
                triggered=is_spike,
                delta=round(dx, 2),
                delta_per_sec=round(dx / dt, 2),
                z_score=round(z, 2),
                unit=units[key],
            )

        self._prev = current

        iforest_hit = False
        anomaly_score = 0.0
        if self.iforest is not None and self.scaler is not None:
            x = np.array(
                [[
                    co2_ppm,
                    ch4_ppm,
                    temp_c,
                    hum_pct,
                    d_vec["co2"],
                    d_vec["ch4"],
                    d_vec["temp"],
                    d_vec["hum"],
                ]],
                dtype=float,
            )
            xs = self.scaler.transform(x)
            raw_score = float(self.iforest.decision_function(xs)[0])
            anomaly_score = round(max(0.0, min(1.0, 0.5 - raw_score)), 3)
            iforest_hit = raw_score < -0.08 and any(
                abs(current[k] - self._prev[k]) >= floors[k] * 0.7 or ch_reports[k].z_score >= 2.8
                for k in current
            )

        any_spike = bool(triggered) or iforest_hit
        return SpikeReport(
            any_spike=any_spike,
            isolation_forest_anomaly=iforest_hit,
            anomaly_score=anomaly_score,
            triggered_channels=triggered,
            channels=ch_reports,
        )
