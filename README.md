---
title: GasSentinel Cloud AI
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
app_file: run_hf.py
app_port: 7860
---

# GasSentinel Cloud AI & Industrial SCADA Dashboard

Real-time cloud decision engine for the **GasSentinel** STM32 + ESP-01 gas and volumetric room analyzer:

1. **Volumetric & Thermo-Hygrometric Sensor Compensation (`app/calibration.py`)** — corrects MQ-135 / MQ-4 $R_s/R_0$ ratios for temperature, humidity, and measured room volume $V$, and computes total excess gas mass ($\text{g}$), leak rate ($\text{g/min}$), required purge ventilation ($\text{m}^3/\text{h}$), and time-to-critical.
2. **Sudden Spike Detection (`app/anomaly.py`)** — hybrid Modified Z-Score + multivariate `IsolationForest` anomaly detector across $\text{CO}_2$, $\text{CH}_4$, Temperature, and Humidity.
3. **Gradual Trend & Exact Rate Estimation (`app/trends.py`)** — Ordinary Least Squares (OLS) regression with $R^2$ gating, outputting exact rates ($\text{ppm/min}$, ${}^\circ\text{C/min}$, $\text{\%RH/min}$) and 5-minute projections.
4. **Containment & Safety Advisor (`app/advisor.py`)** — translates spikes, gradual drifts, and static threshold breaches into STM32 LED/buzzer actuation commands (`$C,...*CRC`) and step-by-step containment instructions.
