"""AI Containment & Safety Playbook Advisor.

Synthesizes:
  - Function 1 (T/H + room-volume calibrated gas concentrations & mass balance),
  - Function 2 (sudden spike detection),
  - Function 3 (gradual trend slopes & 5-minute projections)
into:
  1. Hardware actuation levels for the 3 STM32 LEDs (Temp, Humidity, Gas) and
     the Buzzer so physical indicators react immediately.
  2. Ranked, actionable containment & evacuation instructions tailored to the
     exact failure mode and scaled to the ultrasonic-measured room volume.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from . import settings
from .anomaly import SpikeReport
from .calibration import CalibratedGasReading
from .trends import ChannelTrend


@dataclass
class AdvisoryAction:
    priority: str      # "CRITICAL", "WARNING", "INFO"
    channel: str       # "GAS", "TEMP", "HUMIDITY", "SYSTEM"
    title: str
    containment_steps: List[str] = field(default_factory=list)


@dataclass
class SafetyDecision:
    lvl_temp: int      # 0=OK, 1=WARN (steady LED), 2=CRIT (fast-blinking LED)
    lvl_hum: int
    lvl_gas: int
    buzzer_mode: int
    overall_status: str
    headline: str
    advisories: List[AdvisoryAction] = field(default_factory=list)


def _static_temp_level(temp_c: float) -> int:
    if temp_c >= settings.TEMP_CRIT_HIGH_C or temp_c <= settings.TEMP_CRIT_LOW_C:
        return 2
    if temp_c >= settings.TEMP_WARN_HIGH_C or temp_c <= settings.TEMP_WARN_LOW_C:
        return 1
    return 0


def _static_hum_level(hum_pct: float) -> int:
    if hum_pct >= settings.HUM_CRIT_HIGH_PCT or hum_pct <= settings.HUM_CRIT_LOW_PCT:
        return 2
    if hum_pct >= settings.HUM_WARN_HIGH_PCT or hum_pct <= settings.HUM_WARN_LOW_PCT:
        return 1
    return 0


def _static_gas_level(co2_ppm: float, ch4_ppm: float) -> int:
    if co2_ppm >= settings.CO2_CRIT_PPM or ch4_ppm >= settings.CH4_CRIT_PPM:
        return 2
    if co2_ppm >= settings.CO2_WARN_PPM or ch4_ppm >= settings.CH4_WARN_PPM:
        return 1
    return 0


def build_safety_decision(
    temp_c: float,
    hum_pct: float,
    volume_m3: float,
    cal: CalibratedGasReading,
    spikes: SpikeReport,
    trends: Dict[str, ChannelTrend],
    dht_ok: bool = True,
) -> SafetyDecision:
    lvl_t = _static_temp_level(temp_c) if dht_ok else 1
    lvl_h = _static_hum_level(hum_pct) if dht_ok else 1
    lvl_g = _static_gas_level(cal.co2_ppm, cal.ch4_ppm)

    # Promote levels when the AI detects a sudden spike (Function 2)
    if spikes.channels["temp"].triggered:
        lvl_t = max(lvl_t, 2 if abs(spikes.channels["temp"].delta) >= 1.5 else 1)
    if spikes.channels["hum"].triggered:
        lvl_h = max(lvl_h, 2 if abs(spikes.channels["hum"].delta) >= 6.0 else 1)
    if spikes.channels["co2"].triggered or spikes.channels["ch4"].triggered:
        lvl_g = max(lvl_g, 2)

    # Promote levels on sustained gradual increases (Function 3) before static limits hit
    if trends["temp"].is_gradual_hazard:
        lvl_t = max(lvl_t, 1)
    if trends["hum"].is_gradual_hazard:
        lvl_h = max(lvl_h, 1)
    if trends["co2"].is_gradual_hazard or trends["ch4"].is_gradual_hazard:
        if (
            trends["co2"].projected_5min >= settings.CO2_WARN_PPM
            or trends["ch4"].projected_5min >= settings.CH4_WARN_PPM
            or cal.co2_ppm >= 900.0
        ):
            lvl_g = max(lvl_g, 1)

    buzzer = max(lvl_t, lvl_h, lvl_g)
    advisories: List[AdvisoryAction] = []

    # ---- 1. Combustible / Methane hazards ------------------------------------
    if cal.ch4_ppm >= settings.CH4_WARN_PPM or spikes.channels["ch4"].triggered:
        crit = cal.ch4_ppm >= settings.CH4_CRIT_PPM or spikes.channels["ch4"].triggered
        advisories.append(
            AdvisoryAction(
                priority="CRITICAL" if crit else "WARNING",
                channel="GAS",
                title=(
                    f"Combustible Gas Hazard: {cal.ch4_ppm:.0f} ppm CH4 eq "
                    f"({cal.ch4_mass_g:.1f} g in {volume_m3:.1f} m³ room)"
                ),
                containment_steps=[
                    "DO NOT toggle electrical switches, unplug appliances, or create any spark.",
                    "Close the main gas supply valve immediately if safely accessible.",
                    f"Open opposite windows/doors to establish >= {cal.required_ventilation_m3_h:.0f} m³/h "
                    f"({cal.recommended_ach:.1f} ACH) natural cross-ventilation for this {volume_m3:.1f} m³ room.",
                    "Evacuate all occupants to fresh air and leave doors propped open.",
                ],
            )
        )

    # ---- 2. Carbon Dioxide spike or threshold breach -------------------------
    if (
        cal.co2_ppm >= settings.CO2_WARN_PPM
        or spikes.channels["co2"].triggered
        or trends["co2"].is_gradual_hazard
    ):
        co2_tr = trends["co2"]
        co2_sp = spikes.channels["co2"]
        is_crit = cal.co2_ppm >= settings.CO2_CRIT_PPM or co2_sp.triggered

        if co2_sp.triggered:
            cause = (
                f"Sudden CO₂ Spike (+{co2_sp.delta:.0f} ppm, Z={co2_sp.z_score:.1f}) — "
                f"active release of ~{cal.co2_leak_rate_g_min:.1f} g/min"
            )
        elif co2_tr.direction == "INCREASING":
            ttc_str = (
                f" (reaches critical limit in ~{cal.time_to_crit_min:.1f} min)"
                if cal.time_to_crit_min is not None
                else ""
            )
            cause = (
                f"Gradual CO₂ Accumulation at +{co2_tr.rate_per_min:.1f} ppm/min "
                f"(R²={co2_tr.r_squared:.2f}){ttc_str}"
            )
        else:
            cause = f"Elevated CO₂ Concentration ({cal.co2_ppm:.0f} ppm)"

        advisories.append(
            AdvisoryAction(
                priority="CRITICAL" if is_crit else "WARNING",
                channel="GAS",
                title=f"{cause} | Excess CO₂ Mass: {cal.co2_excess_mass_g:.1f} g",
                containment_steps=[
                    f"Supply {cal.required_ventilation_m3_h:.0f} m³/h of outside air "
                    f"({cal.recommended_ach:.1f} air changes/hour for {volume_m3:.1f} m³ volume).",
                    "Inspect compressed CO₂ cylinders, dry-ice storage, combustion heaters, or blocked exhaust ducts.",
                    "Reduce room occupancy or pause fuel-burning equipment until CO₂ drops below 800 ppm.",
                    "If anyone experiences dizziness, headache, or shortness of breath, move them to fresh air immediately.",
                ],
            )
        )

    # ---- 3. Temperature spike, gradual drift, or static breach ---------------
    if lvl_t > 0:
        t_tr = trends["temp"]
        t_sp = spikes.channels["temp"]
        if not dht_ok:
            advisories.append(
                AdvisoryAction(
                    priority="WARNING",
                    channel="TEMP",
                    title="DHT-22 Climate Sensor Read Timeout (Temp/Humidity LEDs Latched)",
                    containment_steps=[
                        "Check the 3.3 V supply, GND, and 10 kΩ pull-up resistor on PB8.",
                        "Gas compensation is holding the last valid temperature/humidity sample.",
                    ],
                )
            )
        elif t_sp.triggered or temp_c >= settings.TEMP_WARN_HIGH_C or t_tr.rate_per_min > 0:
            advisories.append(
                AdvisoryAction(
                    priority="CRITICAL" if lvl_t == 2 else "WARNING",
                    channel="TEMP",
                    title=(
                        f"Thermal Anomaly: {temp_c:.1f} °C "
                        f"({'+' if t_tr.rate_per_min >= 0 else ''}{t_tr.rate_per_min:.2f} °C/min)"
                    ),
                    containment_steps=[
                        "Inspect high-load electrical panels, battery regulators, and heat sources in the room.",
                        "If paired with a CO₂ spike, treat as an incipient fire: cut equipment power and prepare fire suppression.",
                        "Engage HVAC cooling or exhaust fans to keep ambient temperature below 30 °C.",
                    ],
                )
            )
        else:
            advisories.append(
                AdvisoryAction(
                    priority="CRITICAL" if lvl_t == 2 else "WARNING",
                    channel="TEMP",
                    title=f"Low-Temperature Alert: {temp_c:.1f} °C ({t_tr.rate_per_min:.2f} °C/min)",
                    containment_steps=[
                        "Close exterior dampers to prevent pipe freezing and condensation on sensor optics.",
                        "Verify heating loop operation.",
                    ],
                )
            )

    # ---- 4. Humidity spike, gradual drift, or static breach ------------------
    if lvl_h > 0 and dht_ok:
        h_tr = trends["hum"]
        if hum_pct >= settings.HUM_WARN_HIGH_PCT or h_tr.rate_per_min > 0:
            advisories.append(
                AdvisoryAction(
                    priority="CRITICAL" if lvl_h == 2 else "WARNING",
                    channel="HUMIDITY",
                    title=(
                        f"High Humidity / Condensation Risk: {hum_pct:.1f} %RH "
                        f"({'+' if h_tr.rate_per_min >= 0 else ''}{h_tr.rate_per_min:.2f} %RH/min)"
                    ),
                    containment_steps=[
                        "Check for steam leaks, water ingress, or blocked dehumidifier drains.",
                        "Prevent condensation on exposed electronics and the MQ-135 sensing cap.",
                        "Increase dehumidification or exhaust airflow until RH returns to 35–60 %.",
                    ],
                )
            )
        else:
            advisories.append(
                AdvisoryAction(
                    priority="CRITICAL" if lvl_h == 2 else "WARNING",
                    channel="HUMIDITY",
                    title=f"Dry Air / ESD Risk: {hum_pct:.1f} %RH ({h_tr.rate_per_min:.2f} %RH/min)",
                    containment_steps=[
                        "Low humidity (<25 %RH) increases electrostatic discharge ignition risk around flammable gases.",
                        "Ground personnel and enable humidification before handling gas cylinders or electronics.",
                    ],
                )
            )

    # ---- 5. Nominal state summary --------------------------------------------
    if not advisories:
        advisories.append(
            AdvisoryAction(
                priority="INFO",
                channel="SYSTEM",
                title="All Environmental & Gas Channels Nominal",
                containment_steps=[
                    f"Room volume ({volume_m3:.2f} m³) is within safe air-quality limits "
                    f"(CO₂ {cal.co2_ppm:.0f} ppm, T {temp_c:.1f} °C, RH {hum_pct:.1f} %).",
                    f"Baseline fresh-air exchange of {cal.required_ventilation_m3_h:.0f} m³/h "
                    f"({cal.recommended_ach:.1f} ACH) maintains steady-state air quality.",
                ],
            )
        )

    if buzzer == 2:
        overall = "CRITICAL"
        headline = advisories[0].title
    elif buzzer == 1:
        overall = "WARNING"
        headline = advisories[0].title
    else:
        overall = "NOMINAL"
        headline = "Air quality, climate, and rates of change are nominal."

    return SafetyDecision(
        lvl_temp=lvl_t,
        lvl_hum=lvl_h,
        lvl_gas=lvl_g,
        buzzer_mode=buzzer,
        overall_status=overall,
        headline=headline,
        advisories=advisories,
    )
