"""End-to-end verification suite for the GasSentinel Cloud AI & WebSocket pipeline."""
from __future__ import annotations
from fastapi.testclient import TestClient

from app.engine import GasSentinelEngine
from app.main import app
from app.protocol import (
    build_frame,
    crc16_ccitt,
    parse_geometry_body,
    parse_telemetry_body,
    verify_and_strip,
)
from train_model import train_and_save


def test_crc16_and_wire_frames() -> None:
    body = "T,1,1000,250,500,420,2,2590,2590,1200,0,0,0,0,11"
    framed = build_frame(body)
    stripped = verify_and_strip(framed)
    assert stripped == body
    pkt = parse_telemetry_body(stripped)
    assert pkt is not None
    assert pkt.temp_c == 25.0
    assert pkt.hum_pct == 50.0
    assert pkt.co2_raw_ppm == 420.0
    assert pkt.dht_ok and pkt.mq_warmed and pkt.scan_valid


def test_geometry_frame_parsing() -> None:
    radii = ",".join(["200"] * 36)
    body = f"G,1,125600,280,35168,100,{radii}"
    framed = build_frame(body)
    stripped = verify_and_strip(framed)
    assert stripped is not None
    geom = parse_geometry_body(stripped)
    assert geom is not None
    assert abs(geom.area_m2 - 12.56) < 0.01
    assert abs(geom.height_m - 2.80) < 0.01
    assert abs(geom.volume_m3 - 35.168) < 0.01
    assert len(geom.radii_cm) == 36


def test_model_training_and_three_ai_functions(tmp_path) -> None:
    model_file = tmp_path / "model.joblib"
    metrics = train_and_save(str(model_file))
    assert metrics["co2_r2"] > 0.98
    assert metrics["error_reduction_pct"] > 50.0

    eng = GasSentinelEngine(model_path=str(model_file))
    assert eng.model_loaded

    # 1. Feed 15 normal samples (Function 1: calibration + volumetric mass balance)
    # At 430 ppm CO2, clean Rs/R0 = (430 / 110.47)**(1 / -2.862) = 0.622 -> 622 milli
    t0 = 1000.0
    for i in range(15):
        body = f"T,{i+1},{(i+1)*500},245,480,430,2,622,622,1150,0,0,0,0,11"
        pkt = parse_telemetry_body(body)
        assert pkt is not None
        state, cmd = eng.ingest_telemetry(pkt, ts_sec=t0 + i * 0.5)
        assert state["decision"]["overall_status"] == "NOMINAL"

    # 2. Feed a gradual CO2 increase (+1800 ppm/min steady ramp) -> Function 3
    eng.reset_detectors()
    for i in range(25):
        co2_val = 500.0 + i * 15.0
        r_ratio = int(((co2_val / 110.47) ** (1.0 / -2.862)) * 1000)
        body = f"T,{100+i},{(100+i)*500},250,500,{int(co2_val)},2,{r_ratio},{r_ratio},1400,0,0,0,0,11"
        pkt = parse_telemetry_body(body)
        assert pkt is not None
        state, _ = eng.ingest_telemetry(pkt, ts_sec=2000.0 + i * 0.5)

    co2_trend = state["trends"]["co2"]
    assert co2_trend["direction"] == "INCREASING"
    assert co2_trend["rate_per_min"] > 800.0
    assert co2_trend["r_squared"] > 0.90
    assert state["calibration"]["co2_leak_rate_g_min"] > 0.0
    assert state["calibration"]["time_to_crit_min"] is not None

    # 3. Feed a sudden gas spike after calm baseline -> Function 2
    eng.reset_detectors()
    for i in range(12):
        r_base = int((((425.0 + (i % 2) * 3.0) / 110.47) ** (1.0 / -2.862)) * 1000)
        body = f"T,{200+i},{(200+i)*500},240,450,425,2,{r_base},{r_base},1120,0,0,0,0,11"
        pkt = parse_telemetry_body(body)
        assert pkt is not None
        eng.ingest_telemetry(pkt, ts_sec=3000.0 + i * 0.5)

    r_spike = int(((2650.0 / 110.47) ** (1.0 / -2.862)) * 1000)
    spike_body = f"T,250,125000,240,450,2650,2,{r_spike},{r_spike},2200,0,0,0,0,11"
    spike_pkt = parse_telemetry_body(spike_body)
    assert spike_pkt is not None
    spike_state, cmd_frame = eng.ingest_telemetry(spike_pkt, ts_sec=3006.5)

    assert spike_state["spikes"]["any_spike"] is True
    assert "co2" in spike_state["spikes"]["triggered_channels"]
    assert spike_state["decision"]["lvl_gas"] == 2
    assert spike_state["decision"]["buzzer_mode"] == 2
    assert cmd_frame is not None and cmd_frame.startswith("$C,")


def test_websocket_device_and_dashboard_roundtrip() -> None:
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    with client.websocket_connect("/ws/device?token=gassentinel-secret&device_id=test-mcu") as ws_dev:
        frame = build_frame("T,1,1000,252,495,430,2,2560,2560,1180,0,0,0,0,11")
        ws_dev.send_text(frame)
        reply = ws_dev.receive_text()
        assert reply.startswith("$C,")
        assert verify_and_strip(reply) is not None
