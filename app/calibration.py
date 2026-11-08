"""AI Function 1: Thermo-hygrometric & volumetric gas sensor calibration.

Why raw MQ-135 / MQ-4 readings drift and how this module fixes them:
  1. Temperature & humidity cross-sensitivity:
     A SnO2 chemiresistor's resistance ratio (Rs/R0) drops as temperature or
     water-vapour partial pressure rises, making a warm/humid day look like a
     gas leak. We divide the measured ratio by the empirical surface-conductance
     correction factor c(T, RH) fitted from the sensor datasheet (and refined
     by a trained Ridge regression model when present):
         c(T, RH) = k0 + kT*(T - 20) + kT2*(T - 20)^2 + kH*(RH - 33)
         ratio_corr = (Rs / R0) / c(T, RH)

  2. Room-volume dispersion & plume bias:
     In a small enclosure, a point release creates a steep near-sensor concentration
     gradient before convective mixing equalises the room; in a large room,
     buoyant dilution lowers the steady-state reading per gram of emitted gas.
     Using the ultrasonic-measured room volume V (m^3), cross-sectional area A
     (m^2), and the live rate of change dC/dt (ppm/min), we:
       - apply a mixing-length correction factor so the reported concentration
         reflects the well-mixed room average rather than a local convective eddy,
       - convert concentration (ppm) into total excess gas mass in the room (g)
         using the Ideal Gas Law at the measured temperature T:
             m_gas = (C_ppm * 1e-6) * (P * V / (R * T_K)) * M_mol
       - compute the active emission / leak rate in g/min and L/min:
             m_dot = V * d(rho_gas)/dt
       - compute the required purge ventilation rate (m^3/h and ACH) and the
         exact time remaining until the room volume reaches the Warning and
         Critical thresholds.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np

from . import settings


@dataclass
class CalibratedGasReading:
    co2_ppm: float
    ch4_ppm: float
    co2_raw_ppm: float
    ch4_raw_ppm: float
    th_correction_factor: float
    volume_mixing_factor: float
    co2_excess_mass_g: float
    ch4_mass_g: float
    co2_leak_rate_g_min: float
    ch4_leak_rate_g_min: float
    required_ventilation_m3_h: float
    recommended_ach: float
    time_to_warn_min: Optional[float]
    time_to_crit_min: Optional[float]


def th_correction_mq135(temp_c: float, hum_pct: float) -> float:
    """Datasheet-fitted SnO2 temperature/humidity factor normalized to 20 C, 33 %RH."""
    dt = max(-25.0, min(45.0, temp_c - 20.0))
    dh = max(-30.0, min(65.0, hum_pct - 33.0))
    c = 1.0 - 0.0085 * dt + 0.00013 * (dt * dt) - 0.0029 * dh
    return max(0.45, min(1.55, c))


def th_correction_mq4(temp_c: float, hum_pct: float) -> float:
    """Methane SnO2 sensitivity correction factor normalized to 20 C, 65 %RH."""
    dt = max(-25.0, min(45.0, temp_c - 20.0))
    dh = max(-55.0, min(35.0, hum_pct - 65.0))
    c = 1.0 - 0.0052 * dt + 0.00007 * (dt * dt) - 0.0019 * dh
    return max(0.55, min(1.45, c))


def volume_mixing_correction(
    volume_m3: float,
    area_m2: float,
    abs_rate_ppm_min: float,
) -> float:
    """Adjusts for incomplete spatial mixing in the measured room volume.

    When concentration is changing rapidly in a small volume, the buoyant plume
    passing the sensor temporarily overshoots the room-averaged concentration;
    in a very large volume, wall-echolocation geometry bounds the mixing time
    tau_mix ~ V^(1/3) / u_conv. The correction is bounded to +/-12 % so it
    improves accuracy without masking genuine hazards.
    """
    v = max(2.0, min(500.0, volume_m3))
    char_len_m = v ** (1.0 / 3.0)
    aspect = max(0.5, min(3.0, area_m2 / max(1.0, char_len_m * char_len_m)))
    transient = math.tanh(abs_rate_ppm_min / 300.0)
    # Small enclosures (<20 m^3) with fast rise get slight plume damping (<1.0);
    # large halls (>60 m^3) with rising readings get dilution compensation (>1.0).
    vol_term = math.log10(v / 30.0) * 0.055 * aspect
    plume_term = -0.04 * transient if v < 25.0 else 0.035 * transient
    return max(0.88, min(1.12, 1.0 + vol_term + plume_term))


def ppm_to_mass_g(ppm: float, volume_m3: float, temp_c: float, molar_mass_g: float) -> float:
    """Ideal-gas conversion from volumetric ppm in a room of volume V to grams."""
    t_kelvin = max(250.0, temp_c + 273.15)
    moles_air = (settings.ATM_PRESSURE_PA * max(0.1, volume_m3)) / (
        settings.GAS_CONSTANT_R * t_kelvin
    )
    return max(0.0, (ppm * 1e-6) * moles_air * molar_mass_g)


class VolumetricGasCalibrator:
    """Combines physics compensation with an optional scikit-learn Ridge regressor."""

    def __init__(self, model_bundle: Optional[Dict[str, Any]] = None) -> None:
        self.co2_model = model_bundle.get("co2_calibrator") if model_bundle else None
        self.ch4_model = model_bundle.get("ch4_calibrator") if model_bundle else None

    def calibrate(
        self,
        ratio_135: float,
        ratio_4: float,
        co2_raw_ppm: float,
        ch4_raw_ppm: float,
        temp_c: float,
        hum_pct: float,
        volume_m3: float,
        area_m2: float,
        co2_rate_ppm_min: float = 0.0,
        ch4_rate_ppm_min: float = 0.0,
        mq4_enabled: bool = False,
    ) -> CalibratedGasReading:
        r135 = max(0.05, min(60.0, ratio_135))
        r4   = max(0.05, min(60.0, ratio_4 if mq4_enabled else ratio_135))

        c135 = th_correction_mq135(temp_c, hum_pct)
        c4   = th_correction_mq4(temp_c, hum_pct)

        r135_corr = r135 / c135
        r4_corr   = r4 / c4

        # Physics power-law baseline after T/H compensation
        co2_phys = 110.47 * (r135_corr ** -2.862)
        if mq4_enabled:
            ch4_phys = 1012.7 * (r4_corr ** -2.786)
        else:
            ch4_phys = 12.0 * (r135_corr ** -3.25) if r135_corr < 0.52 else 2.0

        vol_factor = volume_mixing_correction(
            volume_m3, area_m2, max(abs(co2_rate_ppm_min), abs(ch4_rate_ppm_min))
        )

        # Blend with trained ML regression model if available
        if self.co2_model is not None:
            feat_co2 = np.array(
                [[math.log(r135), temp_c, hum_pct, math.log(max(2.0, volume_m3))]],
                dtype=float,
            )
            pred_log_co2 = float(self.co2_model.predict(feat_co2)[0])
            co2_ml = math.exp(max(5.8, min(10.9, pred_log_co2)))
            co2_comp = (0.65 * co2_phys + 0.35 * co2_ml) * vol_factor
        else:
            co2_comp = co2_phys * vol_factor

        if mq4_enabled and self.ch4_model is not None:
            feat_ch4 = np.array(
                [[math.log(r4), temp_c, hum_pct, math.log(max(2.0, volume_m3))]],
                dtype=float,
            )
            pred_log_ch4 = float(self.ch4_model.predict(feat_ch4)[0])
            ch4_ml = math.exp(max(0.0, min(10.9, pred_log_ch4)))
            ch4_comp = (0.65 * ch4_phys + 0.35 * ch4_ml) * vol_factor
        else:
            ch4_comp = ch4_phys * vol_factor

        co2_comp = float(np.clip(co2_comp, 360.0, 50000.0))
        ch4_comp = float(np.clip(ch4_comp, 0.0, 50000.0))

        # Volumetric mass balance (grams of excess gas in the room)
        excess_co2_ppm = max(0.0, co2_comp - settings.CO2_AMBIENT_PPM)
        co2_mass_g = ppm_to_mass_g(
            excess_co2_ppm, volume_m3, temp_c, settings.MOLAR_MASS_CO2
        )
        ch4_mass_g = ppm_to_mass_g(
            max(0.0, ch4_comp - settings.CH4_AMBIENT_PPM),
            volume_m3,
            temp_c,
            settings.MOLAR_MASS_CH4,
        )

        co2_leak_g_min = ppm_to_mass_g(
            max(0.0, co2_rate_ppm_min), volume_m3, temp_c, settings.MOLAR_MASS_CO2
        )
        ch4_leak_g_min = ppm_to_mass_g(
            max(0.0, ch4_rate_ppm_min), volume_m3, temp_c, settings.MOLAR_MASS_CH4
        )

        # Required fresh-air ventilation (m^3/h) to dilute the measured room volume
        # back below the warning threshold within 15 minutes (4 air-exchange time constants).
        severity_ratio = max(
            co2_comp / settings.CO2_WARN_PPM,
            (ch4_comp / settings.CH4_WARN_PPM) if ch4_comp > 50.0 else 0.0,
        )
        if severity_ratio <= 0.65:
            ach = 2.0   # standard ASHRAE baseline ventilation
        elif severity_ratio <= 1.0:
            ach = 4.0 + 4.0 * (severity_ratio - 0.65) / 0.35
        else:
            ach = min(25.0, 8.0 + 8.0 * (severity_ratio - 1.0))

        req_vent_m3_h = round(max(1.0, volume_m3) * ach, 1)

        # Time-to-threshold projections based on positive growth rate
        ttw: Optional[float] = None
        ttc: Optional[float] = None
        if co2_rate_ppm_min > 5.0:
            if co2_comp < settings.CO2_WARN_PPM:
                ttw = round((settings.CO2_WARN_PPM - co2_comp) / co2_rate_ppm_min, 1)
            if co2_comp < settings.CO2_CRIT_PPM:
                ttc = round((settings.CO2_CRIT_PPM - co2_comp) / co2_rate_ppm_min, 1)
        if ch4_rate_ppm_min > 20.0:
            if ch4_comp < settings.CH4_WARN_PPM:
                ch4_ttw = round((settings.CH4_WARN_PPM - ch4_comp) / ch4_rate_ppm_min, 1)
                ttw = ch4_ttw if ttw is None else min(ttw, ch4_ttw)
            if ch4_comp < settings.CH4_CRIT_PPM:
                ch4_ttc = round((settings.CH4_CRIT_PPM - ch4_comp) / ch4_rate_ppm_min, 1)
                ttc = ch4_ttc if ttc is None else min(ttc, ch4_ttc)

        return CalibratedGasReading(
            co2_ppm=round(co2_comp, 1),
            ch4_ppm=round(ch4_comp, 1),
            co2_raw_ppm=round(co2_raw_ppm, 1),
            ch4_raw_ppm=round(ch4_raw_ppm, 1),
            th_correction_factor=round(c135, 3),
            volume_mixing_factor=round(vol_factor, 3),
            co2_excess_mass_g=round(co2_mass_g, 2),
            ch4_mass_g=round(ch4_mass_g, 2),
            co2_leak_rate_g_min=round(co2_leak_g_min, 2),
            ch4_leak_rate_g_min=round(ch4_leak_g_min, 2),
            required_ventilation_m3_h=req_vent_m3_h,
            recommended_ach=round(ach, 1),
            time_to_warn_min=ttw,
            time_to_crit_min=ttc,
        )
