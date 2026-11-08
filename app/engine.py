"""Real-time Cloud AI orchestrator & interactive scenario simulator.

Maintains the sliding state for the active STM32 node, executes:
  1. Volumetric & T/H gas sensor calibration (`VolumetricGasCalibrator`),
  2. Sudden spike detection (`SuddenSpikeDetector`),
  3. OLS gradual trend & exact rate analysis (`TrendAnalyzer`),
  4. Safety actuation & containment playbook generation (`build_safety_decision`),
and produces both the JSON state snapshot for the SCADA dashboard and the
compact `$C,...*CRC` actuation frame for the STM32.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict
import math
import os
import random
import tempfile
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import joblib

from . import settings
from .advisor import SafetyDecision, build_safety_decision
from .anomaly import SpikeReport, SuddenSpikeDetector
from .calibration import CalibratedGasReading, VolumetricGasCalibrator, th_correction_mq135
from .protocol import (
    GeometryPacket,
    TelemetryPacket,
    format_command_frame,
    parse_geometry_body,
    parse_telemetry_body,
    verify_and_strip,
)
from .trends import ChannelTrend, TrendAnalyzer


class GasSentinelEngine:
    def __init__(self, model_path: str = settings.MODEL_PATH) -> None:
        bundle: Optional[Dict[str, Any]] = None
        if os.path.exists(model_path):
            try:
                bundle = joblib.load(model_path)
            except Exception:
                bundle = None

        if bundle is None:
            try:
                from train_model import train_and_save
                tmp_model = os.path.join(tempfile.gettempdir(), "gas_sentinel_ai.joblib")
                train_and_save(tmp_model)
                bundle = joblib.load(tmp_model)
            except Exception:
                bundle = None

        self.model_loaded: bool = bundle is not None
        self.model_metrics: Dict[str, Any] = (
            bundle.get("metrics", {}) if isinstance(bundle, dict) else {}
        )
        self.calibrator = VolumetricGasCalibrator(bundle)
        self.spikes     = SuddenSpikeDetector(bundle)
        self.trends     = TrendAnalyzer()

        # Default room geometry (4.2 m x 3.6 m rectangular room, 2.75 m ceiling = 41.58 m^3)
        self.geometry = self._default_geometry()
        self.history: Deque[Dict[str, Any]] = deque(maxlen=180)
        self.event_log: Deque[Dict[str, Any]] = deque(maxlen=40)

        self.cmd_seq: int = 0
        self.last_device_rx_ts: float = 0.0
        self.device_connected: bool = False
        self.last_state: Dict[str, Any] = {}
        self._last_sent_levels: Tuple[int, int, int, int] = (-1, -1, -1, -1)
        self._last_cmd_tx_ts: float = 0.0
        self._last_headline: str = ""

        # Scenario simulator state (for testing / demo when no hardware is attached)
        self.sim_mode: str = "off"   # "off", "nominal", "gradual_co2", "spike_gas", "thermal_fire"
        self.sim_step: int = 0
        self.sim_seq: int = 0

    @staticmethod
    def _default_geometry() -> GeometryPacket:
        # Ray-cast a realistic 4.2 m x 3.6 m room from the centre at 10-degree steps
        hx, hy = 210.0, 180.0  # half-widths in cm
        radii: List[int] = []
        for i in range(36):
            ang = math.radians(i * 10.0)
            ca, sa = abs(math.cos(ang)), abs(math.sin(ang))
            rx = (hx / ca) if ca > 1e-4 else 1e6
            ry = (hy / sa) if sa > 1e-4 else 1e6
            r = min(rx, ry)
            radii.append(int(round(r)))
        area_m2 = round((2 * hx * 0.01) * (2 * hy * 0.01), 3)
        height_m = 2.75
        return GeometryPacket(
            seq=1,
            area_m2=area_m2,
            height_m=height_m,
            volume_m3=round(area_m2 * height_m, 3),
            valid_pct=100,
            radii_cm=radii,
        )

    def reset_detectors(self) -> None:
        self.spikes.reset()
        self.trends.reset()

    def process_wire_line(self, line: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Parses a raw `$...*CRC` frame from the ESP-01 and returns (state, downlink_cmd)."""
        body = verify_and_strip(line)
        if body is None:
            return None, None

        now = time.time()
        self.last_device_rx_ts = now
        self.device_connected = True

        if body.startswith("G,"):
            geom = parse_geometry_body(body)
            if geom is not None:
                self.geometry = geom
                self._log_event(
                    "INFO",
                    f"360° room scan updated: A={geom.area_m2:.2f} m², "
                    f"H={geom.height_m:.2f} m, V={geom.volume_m3:.2f} m³ ({geom.valid_pct}% valid)",
                )
                if self.last_state:
                    self.last_state["geometry"] = asdict(self.geometry)
                return self.last_state or None, None

        if body.startswith("T,"):
            pkt = parse_telemetry_body(body)
            if pkt is not None:
                return self.ingest_telemetry(pkt, now, source="hardware")

        return None, None

    def ingest_telemetry(
        self,
        pkt: TelemetryPacket,
        ts_sec: Optional[float] = None,
        source: str = "hardware",
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        t0 = time.perf_counter()
        now = ts_sec if ts_sec is not None else time.time()

        # Use previous slope estimate to inform the volumetric mixing correction
        prev_co2_rate = (
            self.last_state["trends"]["co2"]["rate_per_min"]
            if self.last_state and "trends" in self.last_state
            else 0.0
        )
        prev_ch4_rate = (
            self.last_state["trends"]["ch4"]["rate_per_min"]
            if self.last_state and "trends" in self.last_state
            else 0.0
        )

        # 1. Function 1: Temperature, humidity & room-volume calibration
        cal: CalibratedGasReading = self.calibrator.calibrate(
            ratio_135=pkt.ratio_135,
            ratio_4=pkt.ratio_4,
            co2_raw_ppm=pkt.co2_raw_ppm,
            ch4_raw_ppm=pkt.ch4_raw_ppm,
            temp_c=pkt.temp_c,
            hum_pct=pkt.hum_pct,
            volume_m3=self.geometry.volume_m3,
            area_m2=self.geometry.area_m2,
            co2_rate_ppm_min=prev_co2_rate,
            ch4_rate_ppm_min=prev_ch4_rate,
            mq4_enabled=pkt.mq4_enabled,
        )

        # 2. Function 2: Sudden spike detection on calibrated streams
        spikes: SpikeReport = self.spikes.update(
            ts_sec=now,
            co2_ppm=cal.co2_ppm,
            ch4_ppm=cal.ch4_ppm,
            temp_c=pkt.temp_c,
            hum_pct=pkt.hum_pct,
        )

        # 3. Function 3: OLS gradual increase/decrease & exact rate calculation
        trends: Dict[str, ChannelTrend] = self.trends.update(
            ts_sec=now,
            co2_ppm=cal.co2_ppm,
            ch4_ppm=cal.ch4_ppm,
            temp_c=pkt.temp_c,
            hum_pct=pkt.hum_pct,
        )

        # Refresh leak-rate and time-to-limit with the newly updated OLS slope
        cal = self.calibrator.calibrate(
            ratio_135=pkt.ratio_135,
            ratio_4=pkt.ratio_4,
            co2_raw_ppm=pkt.co2_raw_ppm,
            ch4_raw_ppm=pkt.ch4_raw_ppm,
            temp_c=pkt.temp_c,
            hum_pct=pkt.hum_pct,
            volume_m3=self.geometry.volume_m3,
            area_m2=self.geometry.area_m2,
            co2_rate_ppm_min=trends["co2"].rate_per_min,
            ch4_rate_ppm_min=trends["ch4"].rate_per_min,
            mq4_enabled=pkt.mq4_enabled,
        )

        # 4. Containment & hardware actuation decision
        decision: SafetyDecision = build_safety_decision(
            temp_c=pkt.temp_c,
            hum_pct=pkt.hum_pct,
            volume_m3=self.geometry.volume_m3,
            cal=cal,
            spikes=spikes,
            trends=trends,
            dht_ok=pkt.dht_ok,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        point = {
            "ts": round(now, 2),
            "temp_c": round(pkt.temp_c, 2),
            "hum_pct": round(pkt.hum_pct, 2),
            "co2_ppm": cal.co2_ppm,
            "co2_raw_ppm": cal.co2_raw_ppm,
            "ch4_ppm": cal.ch4_ppm,
            "spike": spikes.any_spike,
        }
        self.history.append(point)

        if decision.headline != self._last_headline and decision.overall_status != "NOMINAL":
            self._last_headline = decision.headline
            self._log_event(decision.overall_status, decision.headline)
        elif decision.overall_status == "NOMINAL":
            self._last_headline = ""

        state = {
            "timestamp": round(now, 3),
            "source": source,
            "sim_mode": self.sim_mode,
            "device_online": (now - self.last_device_rx_ts) < 5.0 if source == "hardware" else True,
            "ai_latency_ms": latency_ms,
            "model_loaded": self.model_loaded,
            "model_metrics": self.model_metrics,
            "telemetry": {
                "seq": pkt.seq,
                "uptime_s": round(pkt.uptime_ms * 0.001, 1),
                "temp_c": round(pkt.temp_c, 1),
                "hum_pct": round(pkt.hum_pct, 1),
                "dht_ok": pkt.dht_ok,
                "mq_warmed": pkt.mq_warmed,
                "scan_busy": pkt.scan_busy,
                "muted": pkt.muted,
                "ratio_135": round(pkt.ratio_135, 3),
                "adc_135": pkt.adc_135,
            },
            "calibration": asdict(cal),
            "spikes": {
                "any_spike": spikes.any_spike,
                "isolation_forest_anomaly": spikes.isolation_forest_anomaly,
                "anomaly_score": spikes.anomaly_score,
                "triggered_channels": spikes.triggered_channels,
                "channels": {k: asdict(v) for k, v in spikes.channels.items()},
            },
            "trends": {k: asdict(v) for k, v in trends.items()},
            "decision": {
                "lvl_temp": decision.lvl_temp,
                "lvl_hum": decision.lvl_hum,
                "lvl_gas": decision.lvl_gas,
                "buzzer_mode": decision.buzzer_mode,
                "overall_status": decision.overall_status,
                "headline": decision.headline,
                "advisories": [asdict(a) for a in decision.advisories],
            },
            "geometry": asdict(self.geometry),
            "history": list(self.history)[-90:],
            "events": list(self.event_log)[:15],
        }
        self.last_state = state

        # Send a downlink command to the STM32 immediately on any level change
        # or every 2.5 s as a keep-alive so the STM32 cloud lease stays fresh.
        levels_tuple = (
            decision.lvl_temp,
            decision.lvl_hum,
            decision.lvl_gas,
            decision.buzzer_mode,
        )
        cmd_frame: Optional[str] = None
        if levels_tuple != self._last_sent_levels or (now - self._last_cmd_tx_ts) >= 2.5:
            self._last_sent_levels = levels_tuple
            self._last_cmd_tx_ts = now
            self.cmd_seq = (self.cmd_seq + 1) & 0xFFFF
            cmd_frame = format_command_frame(
                cmd_id=self.cmd_seq,
                lvl_temp=decision.lvl_temp,
                lvl_hum=decision.lvl_hum,
                lvl_gas=decision.lvl_gas,
                buzzer_mode=decision.buzzer_mode,
            )

        return state, cmd_frame

    def build_manual_command(
        self,
        trigger_scan: bool = False,
        calibrate_r0: bool = False,
        mute_seconds: int = 0,
    ) -> str:
        self.cmd_seq = (self.cmd_seq + 1) & 0xFFFF
        d = self.last_state.get("decision", {})
        if trigger_scan:
            self._log_event("INFO", "Manual 360° ultrasonic room scan requested from SCADA UI.")
        if calibrate_r0:
            self._log_event("INFO", "Clean-air R0 calibration requested from SCADA UI.")
        if mute_seconds > 0:
            self._log_event("INFO", f"Buzzer muted for {mute_seconds} s by operator.")
        return format_command_frame(
            cmd_id=self.cmd_seq,
            lvl_temp=int(d.get("lvl_temp", 0)),
            lvl_hum=int(d.get("lvl_hum", 0)),
            lvl_gas=int(d.get("lvl_gas", 0)),
            buzzer_mode=int(d.get("buzzer_mode", 0)),
            trigger_scan=trigger_scan,
            calibrate_r0=calibrate_r0,
            mute_seconds=mute_seconds,
        )

    def _log_event(self, severity: str, message: str) -> None:
        self.event_log.appendleft(
            {
                "time": time.strftime("%H:%M:%S"),
                "severity": severity,
                "message": message,
            }
        )

    # ---- Built-in scenario simulator for live testing without hardware -------
    def set_sim_mode(self, mode: str) -> None:
        self.sim_mode = mode
        self.sim_step = 0
        self.reset_detectors()
        self._log_event("INFO", f"Simulation mode switched to '{mode.upper()}'.")

    def step_simulation(self) -> Optional[Dict[str, Any]]:
        if self.sim_mode == "off":
            return None

        self.sim_step += 1
        self.sim_seq += 1
        s = self.sim_step
        rng = random.Random(1000 + self.sim_seq)

        temp_c  = 24.2 + rng.uniform(-0.06, 0.06)
        hum_pct = 48.0 + rng.uniform(-0.15, 0.15)
        true_co2 = 435.0 + rng.uniform(-4.0, 4.0)

        if self.sim_mode == "gradual_co2":
            # Smooth +180 ppm/min leak (= +1.5 ppm per 0.5 s step)
            true_co2 = min(3400.0, 440.0 + s * 6.5 + rng.uniform(-3.0, 3.0))
            temp_c   = 24.4 + min(4.0, s * 0.02)
        elif self.sim_mode == "spike_gas":
            # 5 s of calm baseline, then a sudden compressed-gas release
            if s < 10:
                true_co2 = 435.0 + rng.uniform(-4.0, 4.0)
            else:
                true_co2 = min(4200.0, 435.0 + 1650.0 * (1.0 - math.exp(-(s - 9) * 0.35)))
        elif self.sim_mode == "thermal_fire":
            if s < 8:
                true_co2 = 440.0
            else:
                true_co2 = min(3800.0, 440.0 + (s - 7) * 140.0)
                temp_c   = min(46.0, 24.5 + (s - 7) * 1.1)
                hum_pct  = max(18.0, 48.0 - (s - 7) * 1.4)

        # Invert the MQ-135 curve with T/H distortion so the AI calibrator has
        # realistic uncompensated sensor ratios to correct.
        r_clean = (true_co2 / 110.47) ** (1.0 / -2.862)
        c_th    = th_correction_mq135(temp_c, hum_pct)
        ratio_135 = max(0.08, r_clean * c_th)
        raw_co2   = 110.47 * (ratio_135 ** -2.862)
        raw_ch4   = 12.0 * (ratio_135 ** -3.25) if ratio_135 < 0.52 else 2.0

        pkt = TelemetryPacket(
            seq=self.sim_seq,
            uptime_ms=self.sim_seq * 500,
            temp_c=temp_c,
            hum_pct=hum_pct,
            co2_raw_ppm=raw_co2,
            ch4_raw_ppm=raw_ch4,
            ratio_135=ratio_135,
            ratio_4=ratio_135,
            adc_135=int(min(4090, max(50, 4095 / (1.0 + ratio_135)))),
            adc_4=0,
            mcu_lvl_temp=0,
            mcu_lvl_hum=0,
            mcu_lvl_gas=0,
            flags=(1 << 0) | (1 << 1) | (1 << 3) | (1 << 4),
        )
        state, _ = self.ingest_telemetry(pkt, time.time(), source=f"sim:{self.sim_mode}")
        return state
