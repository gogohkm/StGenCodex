# mcp_server/parsing/table_schema.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .story import normalize_story

TOKEN_RX = re.compile(r"\b[A-Z]{1,4}\s?-?\s?\d{1,4}\b", re.IGNORECASE)

KW_TOKEN = ["\ubd80\uc7ac", "\uae30\ud638", "\ubd80\uc7ac\uba85", "\ubd80\uc7ac\ubc88\ud638", "MARK", "MEMBER", "NAME", "ID", "NO"]
KW_STORY = ["\uce35", "STORY", "FLOOR", "LEVEL", "LV"]
KW_SECTION = ["\ub2e8\uba74", "\uaddc\uaca9", "SECTION", "SIZE", "\ud615\uac15", "H\ud615\uac15", "STEEL"]
KW_REBAR = ["\ucca0\uadfc", "REBAR", "\uc8fc\uadfc"]
KW_TOP = ["\uc0c1\ubd80", "TOP", "UPPER"]
KW_BOT = ["\ud558\ubd80", "BOT", "BOTTOM", "LOWER"]
KW_STIR = ["\ub760", "\uc2a4\ud130\ub7fd", "STIR", "@"]


def _has_any(s: str, kws: List[str]) -> bool:
    u = (s or "").upper()
    return any(k.upper() in u for k in kws)


def role_from_header_cell(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    if not t:
        return None

    if _has_any(t, KW_TOKEN):
        return "token"
    if _has_any(t, KW_STORY):
        return "story"
    if _has_any(t, KW_SECTION):
        return "section"

    if _has_any(t, KW_REBAR) and _has_any(t, KW_TOP):
        return "rebar_top"
    if _has_any(t, KW_REBAR) and _has_any(t, KW_BOT):
        return "rebar_bot"
    if _has_any(t, KW_TOP):
        return "rebar_top"
    if _has_any(t, KW_BOT):
        return "rebar_bot"
    if _has_any(t, KW_STIR):
        return "stirrup"
    if _has_any(t, KW_REBAR):
        return "rebar"

    return None


def infer_schema(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not cells:
        return {"header_row_idx": None, "columns": {}, "confidence": 0.0, "debug": {}}

    by_row: Dict[int, List[Dict[str, Any]]] = {}
    max_row = 0
    max_col = 0
    for c in cells:
        r = int(c["row_idx"])
        k = int(c["col_idx"])
        max_row = max(max_row, r)
        max_col = max(max_col, k)
        by_row.setdefault(r, []).append(c)

    best_row = None
    best_score = -1
    best_roles_count = 0
    row_debug = {}

    for r, row_cells in by_row.items():
        roles = {}
        for c in row_cells:
            role = role_from_header_cell(str(c.get("text") or ""))
            if role:
                roles.setdefault(role, 0)
                roles[role] += 1
        score = sum(roles.values())
        row_debug[r] = {"roles": roles, "score": score}
        if score > best_score:
            best_score = score
            best_row = r
            best_roles_count = len(roles)

    columns: Dict[str, int] = {}
    header_row_idx = None
    confidence = 0.3

    if best_row is not None and best_score >= 2:
        header_row_idx = best_row
        confidence = 0.6 + 0.1 * min(3, best_roles_count)

        header_cells = by_row.get(best_row, [])
        for c in header_cells:
            role = role_from_header_cell(str(c.get("text") or ""))
            if role and role not in columns:
                columns[role] = int(c["col_idx"])

    if not columns:
        col_token_hits = {}
        col_story_hits = {}
        for c in cells:
            col = int(c["col_idx"])
            txt = str(c.get("text") or "")
            if TOKEN_RX.search(txt):
                col_token_hits[col] = col_token_hits.get(col, 0) + 1
            if normalize_story(txt):
                col_story_hits[col] = col_story_hits.get(col, 0) + 1

        if col_token_hits:
            token_col = max(col_token_hits.items(), key=lambda kv: kv[1])[0]
            columns["token"] = token_col
            confidence = max(confidence, 0.55)

        if col_story_hits:
            story_col = max(col_story_hits.items(), key=lambda kv: kv[1])[0]
            columns.setdefault("story", story_col)
            confidence = max(confidence, 0.55)

        col_sec_hits = {}
        for c in cells:
            col = int(c["col_idx"])
            txt = str(c.get("text") or "").upper().replace("\u00d7", "X")
            if ("H" in txt and "X" in txt) or ("X" in txt and any(ch.isdigit() for ch in txt)):
                col_sec_hits[col] = col_sec_hits.get(col, 0) + 1
        if col_sec_hits:
            sec_col = max(col_sec_hits.items(), key=lambda kv: kv[1])[0]
            columns.setdefault("section", sec_col)
            confidence = max(confidence, 0.5)

    return {
        "header_row_idx": header_row_idx,
        "columns": columns,
        "confidence": float(min(0.95, confidence)),
        "debug": {"row_debug": row_debug},
    }


def parse_rows(cells: List[Dict[str, Any]], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not cells:
        return []
    cols = schema.get("columns") or {}
    header_row = schema.get("header_row_idx")

    by_row: Dict[int, List[Dict[str, Any]]] = {}
    for c in cells:
        by_row.setdefault(int(c["row_idx"]), []).append(c)

    out = []
    for r, row_cells in by_row.items():
        if header_row is not None and r == header_row:
            continue
        colmap = {int(c["col_idx"]): c for c in row_cells}

        fields: Dict[str, Any] = {}

        def pick(role: str) -> Optional[Dict[str, Any]]:
            if role not in cols:
                return None
            cc = colmap.get(int(cols[role]))
            if not cc:
                return None
            return {
                "text": str(cc.get("text") or "").strip(),
                "cad_entity_id": cc.get("cad_entity_id"),
                "x": cc.get("x"),
                "y": cc.get("y"),
            }

        token_cell = pick("token")
        story_cell = pick("story")
        section_cell = pick("section")
        rebar_cell = pick("rebar")
        rebar_top_cell = pick("rebar_top")
        rebar_bot_cell = pick("rebar_bot")
        stirrup_cell = pick("stirrup")

        token_norm = None
        if token_cell and token_cell["text"]:
            m = TOKEN_RX.search(token_cell["text"].upper())
            if m:
                token_norm = re.sub(r"[\s\-_]+", "", m.group(0).strip().upper())

        story_norm = None
        if story_cell and story_cell["text"]:
            story_norm = normalize_story(story_cell["text"])

        for role, cell in [
            ("token", token_cell),
            ("story", story_cell),
            ("section", section_cell),
            ("rebar", rebar_cell),
            ("rebar_top", rebar_top_cell),
            ("rebar_bot", rebar_bot_cell),
            ("stirrup", stirrup_cell),
        ]:
            if cell and cell.get("text"):
                fields[role] = cell

        conf = 0.4
        if token_norm:
            conf += 0.25
        if section_cell and section_cell.get("text"):
            conf += 0.2
        if story_norm:
            conf += 0.1
        if any(k in fields for k in ("rebar", "rebar_top", "rebar_bot", "stirrup")):
            conf += 0.1

        out.append({
            "row_idx": int(r),
            "token_norm": token_norm,
            "story_norm": story_norm,
            "fields": fields,
            "confidence": float(min(0.95, conf)),
        })
    return out
