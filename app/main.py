"""FastAPI server for the GasSentinel Cloud AI and Industrial SCADA Dashboard.

Endpoints:
  GET  /                        -> Factory SCADA Web Dashboard
  GET  /api/health              -> Cloud & AI model readiness probe
  GET  /api/state               -> Latest full telemetry + AI analysis snapshot
  POST /api/telemetry           -> HTTP fallback for ingesting a `$T,...*CRC` frame
  POST /api/command             -> Trigger 360° room scan, R0 calibration, or buzzer mute
  POST /api/simulate/{mode}     -> Switch built-in scenario simulator
                                   ("off", "nominal", "gradual_co2", "spike_gas", "thermal_fire")
  WS   /ws/device               -> Persistent low-latency link for the ESP-01 module
  WS   /ws/dashboard            -> Real-time stream to all open SCADA browser tabs
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any, Dict, Set

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import settings
from .engine import GasSentinelEngine

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class ConnectionHub:
    def __init__(self) -> None:
        self.dashboards: Set[WebSocket] = set()
        self.device_ws: WebSocket | None = None

    async def broadcast_state(self, state: Dict[str, Any]) -> None:
        if not self.dashboards:
            return
        dead: list[WebSocket] = []
        for ws in list(self.dashboards):
            try:
                await ws.send_json(state)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.dashboards.discard(ws)

    async def send_to_device(self, frame: str) -> bool:
        if self.device_ws is None:
            return False
        try:
            await self.device_ws.send_text(frame)
            return True
        except Exception:
            self.device_ws = None
            return False


engine = GasSentinelEngine()
hub    = ConnectionHub()


async def _simulation_loop() -> None:
    # Seed the dashboard with one nominal reading at boot so charts are never blank
    engine.set_sim_mode("nominal")
    for _ in range(12):
        engine.step_simulation()
    engine.set_sim_mode("off")

    while True:
        await asyncio.sleep(0.5)
        if engine.sim_mode != "off" and hub.device_ws is None:
            state = engine.step_simulation()
            if state is not None:
                await hub.broadcast_state(state)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_simulation_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(
    title="GasSentinel Cloud AI",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class FrameIn(BaseModel):
    frame: str


class ManualCommandIn(BaseModel):
    trigger_scan: bool = False
    calibrate_r0: bool = False
    mute_seconds: int = 0


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/styles.css")
async def styles_css() -> FileResponse:
    return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
async def app_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": engine.model_loaded,
        "model_metrics": engine.model_metrics,
        "device_connected": hub.device_ws is not None,
        "sim_mode": engine.sim_mode,
    }


@app.get("/api/state")
async def get_state() -> Dict[str, Any]:
    return engine.last_state


@app.post("/api/telemetry")
async def post_telemetry(payload: FrameIn) -> Dict[str, Any]:
    state, cmd_frame = engine.process_wire_line(payload.frame)
    if state is None:
        raise HTTPException(status_code=400, detail="Invalid frame or CRC mismatch")
    await hub.broadcast_state(state)
    return {"state": state, "command_frame": cmd_frame}


@app.post("/api/command")
async def post_command(cmd: ManualCommandIn) -> Dict[str, Any]:
    frame = engine.build_manual_command(
        trigger_scan=cmd.trigger_scan,
        calibrate_r0=cmd.calibrate_r0,
        mute_seconds=cmd.mute_seconds,
    )
    delivered = await hub.send_to_device(frame)
    if engine.last_state:
        await hub.broadcast_state(engine.last_state)
    return {"sent_frame": frame.strip(), "delivered_to_device": delivered}


@app.post("/api/simulate/{mode}")
async def set_simulation(mode: str) -> Dict[str, Any]:
    allowed = {"off", "nominal", "gradual_co2", "spike_gas", "thermal_fire"}
    if mode not in allowed:
        raise HTTPException(status_code=400, detail=f"Mode must be one of {sorted(allowed)}")
    engine.set_sim_mode(mode)
    if mode != "off":
        state = engine.step_simulation()
        if state:
            await hub.broadcast_state(state)
    elif engine.last_state:
        engine.last_state["sim_mode"] = "off"
        await hub.broadcast_state(engine.last_state)
    return {"sim_mode": engine.sim_mode}


@app.websocket("/ws/device")
async def ws_device(
    websocket: WebSocket,
    token: str = Query(default=""),
    device_id: str = Query(default="stm32-node-01"),
) -> None:
    if settings.DEVICE_TOKEN and token != settings.DEVICE_TOKEN:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    hub.device_ws = websocket
    engine.sim_mode = "off"  # live hardware always overrides the simulator
    engine.device_connected = True
    try:
        while True:
            raw_text = await websocket.receive_text()
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                state, cmd_frame = engine.process_wire_line(line)
                if cmd_frame:
                    await websocket.send_text(cmd_frame)
                if state:
                    await hub.broadcast_state(state)
    except WebSocketDisconnect:
        pass
    finally:
        if hub.device_ws is websocket:
            hub.device_ws = None
        engine.device_connected = False


@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket) -> None:
    await websocket.accept()
    hub.dashboards.add(websocket)
    try:
        if engine.last_state:
            await websocket.send_json(engine.last_state)
        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")
            if action == "simulate":
                mode = str(msg.get("mode", "off"))
                engine.set_sim_mode(mode)
            elif action in ("scan", "calibrate", "mute"):
                frame = engine.build_manual_command(
                    trigger_scan=(action == "scan"),
                    calibrate_r0=(action == "calibrate"),
                    mute_seconds=120 if action == "mute" else 0,
                )
                await hub.send_to_device(frame)
                if engine.last_state:
                    await hub.broadcast_state(engine.last_state)
    except WebSocketDisconnect:
        pass
    finally:
        hub.dashboards.discard(websocket)
