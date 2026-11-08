"""Central thresholds, physics constants, and window parameters.

Thresholds match firmware/stm32/include/config.h so the local edge safety
supervisor and the cloud AI agree on static boundaries while the cloud adds:
  (1) T/H + room-volume gas compensation,
  (2) sudden spike detection,
  (3) gradual rate-of-change estimation and time-to-limit forecasting.
"""
from __future__ import annotations
import os

DEVICE_TOKEN: str = os.getenv("GASSENTINEL_TOKEN", "gassentinel-secret")
MODEL_PATH: str   = os.getenv("GASSENTINEL_MODEL", "models/gas_sentinel_ai.joblib")

# ---- Static thresholds (mirrored on the STM32) ------------------------------
TEMP_WARN_HIGH_C: float = 32.0
TEMP_CRIT_HIGH_C: float = 40.0
TEMP_WARN_LOW_C:  float = 12.0
TEMP_CRIT_LOW_C:  float = 5.0

HUM_WARN_HIGH_PCT: float = 70.0
HUM_CRIT_HIGH_PCT: float = 85.0
HUM_WARN_LOW_PCT:  float = 25.0
HUM_CRIT_LOW_PCT:  float = 15.0

CO2_AMBIENT_PPM: float = 420.0
CO2_WARN_PPM:    float = 1200.0
CO2_CRIT_PPM:    float = 2500.0

CH4_AMBIENT_PPM: float = 2.0
CH4_WARN_PPM:    float = 5000.0   # 10 % LEL (methane LEL = 50,000 ppm)
CH4_CRIT_PPM:    float = 10000.0  # 20 % LEL

# ---- Sudden spike detection (Function 2) ------------------------------------
SPIKE_WINDOW_SAMPLES: int   = 30   # 15 s at 2 Hz
SPIKE_Z_THRESHOLD:    float = 3.8  # modified Z-score threshold
SPIKE_MIN_DELTA_CO2:  float = 85.0     # ppm per sample step
SPIKE_MIN_DELTA_CH4:  float = 200.0    # ppm per sample step
SPIKE_MIN_DELTA_TEMP: float = 0.6      # deg C per sample step
SPIKE_MIN_DELTA_HUM:  float = 2.5      # %RH per sample step

# ---- Gradual trend & rate estimation (Function 3) ---------------------------
TREND_WINDOW_SAMPLES: int   = 60   # 30 s sliding regression window at 2 Hz
TREND_MIN_SAMPLES:    int   = 10
TREND_MIN_R2:         float = 0.55  # goodness-of-fit required to declare a trend

# Minimum slope (units / minute) to classify as a meaningful gradual drift
RATE_SIGNIFICANT_CO2_PPM_MIN:  float = 25.0
RATE_SIGNIFICANT_CH4_PPM_MIN:  float = 60.0
RATE_SIGNIFICANT_TEMP_C_MIN:   float = 0.25
RATE_SIGNIFICANT_HUM_PCT_MIN:  float = 1.0

# ---- Ideal-gas constants for room mass-balance (Function 1) -----------------
ATM_PRESSURE_PA: float = 101325.0
GAS_CONSTANT_R:  float = 8.3144626   # J / (mol * K)
MOLAR_MASS_CO2:  float = 44.01       # g / mol
MOLAR_MASS_CH4:  float = 16.04       # g / mol
