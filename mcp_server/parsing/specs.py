# mcp_server/parsing/specs.py
from __future__ import annotations

import re
from dataclasses import dataclass
from math import pi
from typing import Any, Dict, List


def rebar_area_mm2(dia_mm: float) -> float:
    return pi * (float(dia_mm) ** 2) / 4.0


# Regexes
RC_RECT_RX = re.compile(
    r"(?:\b|^)(?:RC\s*)?(?:BEAM|COLUMN|B|C)?\s*(?P<b>\d{2,4})\s*([xX*])\s*(?P<h>\d{2,4})(?:\b|$)"
)

STEEL_H_RX = re.compile(
    r"(?:\b|^)\s*H\s*[-]?\s*(?P<H>\d{2,4})\s*([xX*])\s*(?P<B>\d{2,4})\s*([xX*])\s*(?P<tw>\d{1,3})\s*([xX*])\s*(?P<tf>\d{1,3})(?:\b|$)",
    re.IGNORECASE,
)

STEEL_BOX_RX = re.compile(
    r"(?:\b|^)\s*(?:RHS|SHS|BOX)\s*[-]?\s*(?P<H>\d{2,4}(?:\.\d+)?)\s*([xX*])\s*(?P<B>\d{2,4}(?:\.\d+)?)\s*([xX*])\s*(?P<t>\d{1,3}(?:\.\d+)?)(?:t|T)?(?:\b|$)",
    re.IGNORECASE,
)

STEEL_PIPE_RX = re.compile(
    r"(?:\b|^)\s*(?:CHS|PIPE|P)\s*[-]?\s*(?P<D>\d{2,4}(?:\.\d+)?)\s*([xX*])\s*(?P<t>\d{1,3}(?:\.\d+)?)(?:t|T)?(?:\b|$)",
    re.IGNORECASE,
)

STEEL_CHANNEL_RX = re.compile(
    r"(?:\b|^)\s*(?:CHANNEL|CH|C)\s*[-]\s*(?P<H>\d{2,4}(?:\.\d+)?)\s*([xX*])\s*(?P<B>\d{2,4}(?:\.\d+)?)\s*([xX*])\s*(?P<tw>\d{1,3}(?:\.\d+)?)\s*([xX*])\s*(?P<tf>\d{1,3}(?:\.\d+)?)(?:\b|$)",
    re.IGNORECASE,
)

STEEL_ANGLE_RX = re.compile(
    r"(?:\b|^)\s*L\s*[-]?\s*(?P<b>\d{2,4}(?:\.\d+)?)\s*([xX*])\s*(?P<d>\d{2,4}(?:\.\d+)?)\s*([xX*])\s*(?P<t>\d{1,3}(?:\.\d+)?)(?:\b|$)",
    re.IGNORECASE,
)

STIRRUP_RX = re.compile(
    r"(?:(?P<legs>\d+)\s*[-]?\s*)?(?P<prefix>HD|SD|D)?\s*(?P<dia>\d{2})\s*@\s*(?P<s>\d{2,4})",
    re.IGNORECASE,
)

MAIN_A_RX = re.compile(
    r"(?P<count>\d+)\s*[-]?\s*(?P<prefix>HD|SD|D)?\s*(?P<dia>\d{2})\b",
    re.IGNORECASE,
)
MAIN_B_RX = re.compile(
    r"(?P<prefix>HD|SD|D)?\s*(?P<dia>\d{2})\s*[-]?\s*(?P<count>\d+)\s*(?:EA|E\.A\.|EA\.|E\.A|EA)?",
    re.IGNORECASE,
)

TOP_KEYS = ["TOP", "UPPER", "상부", "상", "T/"]
BOT_KEYS = ["BOT", "BOTTOM", "LOWER", "하부", "하", "B/"]


def _pos_hint(text: str, idx: int, window: int = 20) -> str:
    s = text[max(0, idx - window):idx].upper()
    for k in TOP_KEYS:
        if k.upper() in s:
            return "TOP"
    for k in BOT_KEYS:
        if k.upper() in s:
            return "BOT"
    return "UNKNOWN"


def _near_has_at(text: str, end: int, window: int = 6) -> bool:
    return "@" in text[end:end + window]


def parse_specs_from_text(raw_text: str) -> List[Dict[str, Any]]:
    t0 = (raw_text or "").strip()
    if not t0:
        return []

    t = t0
    t = t.replace("\u00d7", "X").replace("\uff0a", "*").replace("\uff38", "X")
    t = t.replace("\u00d8", "D").replace("\u00f8", "D")
    t = t.replace("\u25a1", "BOX")

    specs: List[Dict[str, Any]] = []

    # 1) Steel H section
    m = STEEL_H_RX.search(t)
    if m:
        H = float(m.group("H"))
        B = float(m.group("B"))
        tw = float(m.group("tw"))
        tf = float(m.group("tf"))
        if 100 <= H <= 2000 and 50 <= B <= 1000:
            specs.append({
                "spec_kind": "steel_h_section",
                "shape": "H",
                "H_mm": H,
                "B_mm": B,
                "tw_mm": tw,
                "tf_mm": tf,
                "confidence": 0.9,
                "raw_fragment": m.group(0).strip(),
            })

    # 1.1) Steel BOX
    m = STEEL_BOX_RX.search(t)
    if m:
        H = float(m.group("H"))
        B = float(m.group("B"))
        tt = float(m.group("t"))
        specs.append({
            "spec_kind": "steel_box_section",
            "H_mm": H,
            "B_mm": B,
            "t_mm": tt,
            "confidence": 0.9,
            "raw_fragment": m.group(0).strip(),
        })

    # 1.2) Steel PIPE
    m = STEEL_PIPE_RX.search(t)
    if m:
        D = float(m.group("D"))
        tt = float(m.group("t"))
        if D >= 30:
            specs.append({
                "spec_kind": "steel_pipe_section",
                "D_mm": D,
                "t_mm": tt,
                "confidence": 0.9,
                "raw_fragment": m.group(0).strip(),
            })

    # 1.3) Steel CHANNEL
    m = STEEL_CHANNEL_RX.search(t)
    if m:
        H = float(m.group("H"))
        B = float(m.group("B"))
        tw = float(m.group("tw"))
        tf = float(m.group("tf"))
        specs.append({
            "spec_kind": "steel_channel_section",
            "H_mm": H,
            "B_mm": B,
            "tw_mm": tw,
            "tf_mm": tf,
            "confidence": 0.85,
            "raw_fragment": m.group(0).strip(),
        })

    # 1.4) Steel ANGLE
    m = STEEL_ANGLE_RX.search(t)
    if m:
        b = float(m.group("b"))
        dd = float(m.group("d"))
        tt = float(m.group("t"))
        specs.append({
            "spec_kind": "steel_angle_section",
            "b_mm": b,
            "d_mm": dd,
            "t_mm": tt,
            "confidence": 0.8,
            "raw_fragment": m.group(0).strip(),
        })

    # 2) RC rectangular section
    if not any(s["spec_kind"].startswith("steel_") for s in specs):
        for m in RC_RECT_RX.finditer(t):
            b = int(m.group("b"))
            h = int(m.group("h"))
            if 100 <= b <= 2000 and 100 <= h <= 3000:
                specs.append({
                    "spec_kind": "rc_rect_section",
                    "b_mm": b,
                    "h_mm": h,
                    "confidence": 0.8,
                    "raw_fragment": m.group(0).strip(),
                })

    # 3) Stirrups
    for m in STIRRUP_RX.finditer(t):
        dia = int(m.group("dia"))
        s = int(m.group("s"))
        legs = m.group("legs")
        legs_n = int(legs) if legs else 2
        if 6 <= dia <= 19 and 50 <= s <= 400:
            Av = legs_n * rebar_area_mm2(dia)
            specs.append({
                "spec_kind": "rebar_stirrup",
                "dia_mm": dia,
                "legs": legs_n,
                "s_mm": s,
                "Av_mm2": Av,
                "confidence": 0.8,
                "raw_fragment": m.group(0).strip(),
            })

    # 4) Main bars
    for rx in (MAIN_A_RX, MAIN_B_RX):
        for m in rx.finditer(t):
            if _near_has_at(t, m.end()):
                continue
            dia = int(m.group("dia"))
            count = int(m.group("count"))
            if not (10 <= dia <= 43 and 1 <= count <= 40):
                continue

            pos = _pos_hint(t, m.start())
            As = count * rebar_area_mm2(dia)
            specs.append({
                "spec_kind": "rebar_main",
                "pos": pos,
                "dia_mm": dia,
                "count": count,
                "As_mm2": As,
                "confidence": 0.75,
                "raw_fragment": m.group(0).strip(),
            })

    return specs
