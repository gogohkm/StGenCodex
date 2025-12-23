# mcp_server/parsing/story.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

BASEMENT_RX = re.compile(r"(?:\b|^)(?:B|\uc9c0\ud558)\s*0*(\d{1,2})\s*(?:\uce35|F)?(?:\b|$)", re.IGNORECASE)
FLOOR_RX = re.compile(r"(?:\b|^)(?:\uc9c0\uc0c1)?\s*0*(\d{1,2})\s*(?:\uce35|F)(?:\b|$)", re.IGNORECASE)
RF_RX = re.compile(r"(?:\b|^)(RF|ROOF|\uc625\uc0c1)(?:\b|$)", re.IGNORECASE)
LEVEL_RX = re.compile(r"(?:\b|^)(?:LV\.?|LEVEL)\s*0*(\d{1,2})(?:\b|$)", re.IGNORECASE)


def normalize_story(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None

    if RF_RX.search(t):
        return "RF"

    m = BASEMENT_RX.search(t)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 20:
            return f"B{n}"

    m = FLOOR_RX.search(t)
    if m:
        n = int(m.group(1))
        if 0 <= n <= 200:
            return f"{n}F"

    m = LEVEL_RX.search(t)
    if m:
        n = int(m.group(1))
        if 0 <= n <= 200:
            return f"{n}F"

    return None


def story_index(story_norm: str) -> int:
    s = (story_norm or "").upper().strip()
    if s == "RF":
        return 999
    if s.startswith("B"):
        try:
            return -int(s[1:])
        except Exception:
            return -999
    if s.endswith("F"):
        try:
            return int(s[:-1])
        except Exception:
            return 0
    return 0


def extract_story_candidates(text: str) -> List[Dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return []

    out: List[Dict[str, Any]] = []
    s = normalize_story(t)
    if s:
        out.append({"story_norm": s, "confidence": 0.85, "raw_fragment": t})
    return out
