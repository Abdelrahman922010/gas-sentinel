"""Training, evaluation & serialization pipeline for the GasSentinel Cloud AI.

Trains and validates two models saved in `models/gas_sentinel_ai.joblib`:
  1. `co2_calibrator` & `ch4_calibrator` (PolynomialFeatures + Ridge Regression):
     Learns the inverse mapping from [ln(Rs/R0), Temp_C, RH_%, ln(Volume_m3)] to
     ln(true_ppm), compensating for SnO2 thermo-hygrometric drift and room-volume
     plume dispersion.
  2. `spike_iforest` (StandardScaler + IsolationForest):
     Learns the 8-dimensional normal operating manifold
     [CO2, CH4, Temp, RH, dCO2/dt, dCH4/dt, dTemp/dt, dRH/dt] so sudden multi-channel
     anomalies are flagged immediately.

Usage:
    python train_model.py --out models/gas_sentinel_ai.joblib
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from app.calibration import th_correction_mq135, th_correction_mq4


def build_calibration_dataset(n_samples: int = 8000, seed: int = 42):
    rng = np.random.default_rng(seed)
    true_co2 = np.exp(rng.uniform(np.log(380.0), np.log(8000.0), n_samples))
    true_ch4 = np.exp(rng.uniform(np.log(10.0), np.log(15000.0), n_samples))
    temp_c   = rng.uniform(5.0, 48.0, n_samples)
    hum_pct  = rng.uniform(15.0, 92.0, n_samples)
    vol_m3   = np.exp(rng.uniform(np.log(6.0), np.log(250.0), n_samples))

    # Forward sensor model: Rs/R0 = (ppm / A)^(1/B) * c(T, H) * vol_bias * noise
    c135 = np.array([th_correction_mq135(t, h) for t, h in zip(temp_c, hum_pct)])
    c4   = np.array([th_correction_mq4(t, h) for t, h in zip(temp_c, hum_pct)])
    vol_bias = 1.0 - 0.018 * np.log10(vol_m3 / 30.0)

    r135_ideal = (true_co2 / 110.47) ** (1.0 / -2.862)
    r4_ideal   = (true_ch4 / 1012.7) ** (1.0 / -2.786)

    r135_meas = r135_ideal * c135 * vol_bias * rng.normal(1.0, 0.015, n_samples)
    r4_meas   = r4_ideal * c4 * vol_bias * rng.normal(1.0, 0.015, n_samples)

    X_co2 = np.column_stack([np.log(r135_meas), temp_c, hum_pct, np.log(vol_m3)])
    y_co2 = np.log(true_co2)

    X_ch4 = np.column_stack([np.log(r4_meas), temp_c, hum_pct, np.log(vol_m3)])
    y_ch4 = np.log(true_ch4)
    return X_co2, y_co2, X_ch4, y_ch4


def build_spike_dataset(n_normal: int = 5000, seed: int = 7):
    rng = np.random.default_rng(seed)
    co2  = rng.normal(560.0, 120.0, n_normal).clip(380.0, 1150.0)
    ch4  = rng.exponential(8.0, n_normal).clip(0.0, 80.0)
    temp = rng.normal(24.0, 3.2, n_normal).clip(15.0, 31.5)
    hum  = rng.normal(48.0, 8.5, n_normal).clip(28.0, 68.0)

    d_co2  = rng.normal(0.0, 6.0, n_normal)
    d_ch4  = rng.normal(0.0, 3.0, n_normal)
    d_temp = rng.normal(0.0, 0.05, n_normal)
    d_hum  = rng.normal(0.0, 0.15, n_normal)

    X_norm = np.column_stack([co2, ch4, temp, hum, d_co2, d_ch4, d_temp, d_hum])

    # Synthetic spike bursts for evaluation
    n_anom = 400
    X_anom = np.column_stack([
        rng.uniform(1400.0, 4800.0, n_anom),
        rng.uniform(200.0, 12000.0, n_anom),
        rng.uniform(32.0, 46.0, n_anom),
        rng.uniform(15.0, 88.0, n_anom),
        rng.uniform(180.0, 950.0, n_anom),
        rng.uniform(350.0, 2500.0, n_anom),
        rng.uniform(1.2, 4.5, n_anom),
        rng.uniform(-6.0, 6.0, n_anom),
    ])
    return X_norm, X_anom


def train_and_save(out_path: str) -> Dict[str, Any]:
    X_co2, y_co2, X_ch4, y_ch4 = build_calibration_dataset()
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_co2, y_co2, test_size=0.2, random_state=42)
    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(X_ch4, y_ch4, test_size=0.2, random_state=42)

    co2_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=0.5))
    co2_model.fit(Xc_tr, yc_tr)
    yc_pred = co2_model.predict(Xc_te)
    co2_r2  = float(r2_score( np.exp(yc_te), np.exp(yc_pred) ))
    co2_mae = float(mean_absolute_error( np.exp(yc_te), np.exp(yc_pred) ))

    # Compare against uncompensated power-law curve to quantify accuracy gain
    uncomp_co2 = 110.47 * (np.exp(Xc_te[:, 0]) ** -2.862)
    uncomp_mae = float(mean_absolute_error(np.exp(yc_te), uncomp_co2))

    ch4_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=0.5))
    ch4_model.fit(Xm_tr, ym_tr)
    ym_pred = ch4_model.predict(Xm_te)
    ch4_r2  = float(r2_score(np.exp(ym_te), np.exp(ym_pred)))

    X_norm, X_anom = build_spike_dataset()
    scaler = StandardScaler().fit(X_norm)
    iforest = IsolationForest(
        n_estimators=150,
        contamination=0.015,
        random_state=42,
    )
    iforest.fit(scaler.transform(X_norm))
    spike_recall = float(np.mean(iforest.predict(scaler.transform(X_anom)) == -1))

    metrics = {
        "co2_r2": round(co2_r2, 4),
        "co2_mae_ppm": round(co2_mae, 1),
        "uncompensated_co2_mae_ppm": round(uncomp_mae, 1),
        "error_reduction_pct": round((1.0 - co2_mae / uncomp_mae) * 100.0, 1),
        "ch4_r2": round(ch4_r2, 4),
        "spike_recall_pct": round(spike_recall * 100.0, 1),
    }

    bundle = {
        "co2_calibrator": co2_model,
        "ch4_calibrator": ch4_model,
        "spike_scaler": scaler,
        "spike_iforest": iforest,
        "metrics": metrics,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_file)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GasSentinel Cloud AI models")
    parser.add_argument("--out", default="models/gas_sentinel_ai.joblib")
    args = parser.parse_args()
    m = train_and_save(args.out)
    print("Trained GasSentinel AI bundle ->", os.path.abspath(args.out))
    for k, v in m.items():
        print(f"  {k}: {v}")
