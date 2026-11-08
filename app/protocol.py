"""Framed UART/WebSocket line protocol shared with the STM32 and ESP-01.

Every frame is a single ASCII line:
    $<TYPE>,<csv_fields>*<CRC16_HEX>
where CRC16_HEX is the 4-digit uppercase hex of CRC-16/CCITT-FALSE over the
characters between '$' and '*' (exclusive).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def build_frame(body: str) -> str:
    crc = crc16_ccitt(body.encode("ascii"))
    return f"${body}*{crc:04X}\n"


def verify_and_strip(line: str) -> Optional[str]:
    s = line.strip()
    if len(s) < 7 or not s.startswith("$") or "*" not in s:
        return None
    body, crc_hex = s[1:].rsplit("*", 1)
    if len(crc_hex) != 4:
        return None
    try:
        expected = int(crc_hex, 16)
    except ValueError:
        return None
    if crc16_ccitt(body.encode("ascii")) != expected:
        return None
    return body


@dataclass
class TelemetryPacket:
    seq: int
    uptime_ms: int
    temp_c: float
    hum_pct: float
    co2_raw_ppm: float
    ch4_raw_ppm: float
    ratio_135: float
    ratio_4: float
    adc_135: int
    adc_4: int
    mcu_lvl_temp: int
    mcu_lvl_hum: int
    mcu_lvl_gas: int
    flags: int

    @property
    def dht_ok(self) -> bool:
        return bool(self.flags & (1 << 0))

    @property
    def mq_warmed(self) -> bool:
        return bool(self.flags & (1 << 1))

    @property
    def scan_busy(self) -> bool:
        return bool(self.flags & (1 << 2))

    @property
    def scan_valid(self) -> bool:
        return bool(self.flags & (1 << 3))

    @property
    def muted(self) -> bool:
        return bool(self.flags & (1 << 5))

    @property
    def mq4_enabled(self) -> bool:
        return bool(self.flags & (1 << 6))


@dataclass
class GeometryPacket:
    seq: int
    area_m2: float
    height_m: float
    volume_m3: float
    valid_pct: int
    radii_cm: List[int] = field(default_factory=list)


def parse_telemetry_body(body: str) -> Optional[TelemetryPacket]:
    parts = body.split(",")
    if len(parts) != 15 or parts[0] != "T":
        return None
    try:
        return TelemetryPacket(
            seq=int(parts[1]),
            uptime_ms=int(parts[2]),
            temp_c=int(parts[3]) * 0.1,
            hum_pct=int(parts[4]) * 0.1,
            co2_raw_ppm=float(parts[5]),
            ch4_raw_ppm=float(parts[6]),
            ratio_135=int(parts[7]) * 0.001,
            ratio_4=int(parts[8]) * 0.001,
            adc_135=int(parts[9]),
            adc_4=int(parts[10]),
            mcu_lvl_temp=int(parts[11]),
            mcu_lvl_hum=int(parts[12]),
            mcu_lvl_gas=int(parts[13]),
            flags=int(parts[14]),
        )
    except ValueError:
        return None


def parse_geometry_body(body: str) -> Optional[GeometryPacket]:
    parts = body.split(",")
    if len(parts) < 10 or parts[0] != "G":
        return None
    try:
        area_cm2   = int(parts[2])
        height_cm  = int(parts[3])
        volume_dm3 = int(parts[4])
        radii      = [int(x) for x in parts[6:]]
        return GeometryPacket(
            seq=int(parts[1]),
            area_m2=round(area_cm2 * 1e-4, 3),
            height_m=round(height_cm * 1e-2, 2),
            volume_m3=round(volume_dm3 * 1e-3, 3),
            valid_pct=int(parts[5]),
            radii_cm=radii,
        )
    except ValueError:
        return None


def format_command_frame(
    cmd_id: int,
    lvl_temp: int,
    lvl_hum: int,
    lvl_gas: int,
    buzzer_mode: int,
    trigger_scan: bool = False,
    calibrate_r0: bool = False,
    mute_seconds: int = 0,
) -> str:
    body = (
        f"C,{cmd_id & 0xFFFF},{int(lvl_temp)},{int(lvl_hum)},{int(lvl_gas)},"
        f"{int(buzzer_mode)},{1 if trigger_scan else 0},"
        f"{1 if calibrate_r0 else 0},{int(mute_seconds)}"
    )
    return build_frame(body)
