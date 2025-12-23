# mcp_server/parsing/table_schema_v2.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .story import normalize_story

TOKEN_RX = re.compile(r"\b[A-Z]{1,4}\s?-?\s?\d{1,4}\b", re.IGNORECASE)

KW = {
    "token": ["\ubd80\uc7ac", "\uae30\ud638", "\ubd80\uc7ac\uba85", "\ubd80\uc7ac\ubc88\ud638", "MARK", "MEMBER", "NAME", "ID", "NO"],
    "story": ["\uce35", "STORY", "FLOOR", "LEVEL", "LV"],
    "section": ["\ub2e8\uba74", "\uaddc\uaca9", "SECTION", "SIZE", "\ud615\uac15", "H\ud615\uac15", "STEEL", "RHS", "SHS", "PIPE", "CHS", "BOX"],
    "rebar": ["\ucca0\uadfc", "REBAR", "\uc8fc\uadfc"],
    "top": ["\uc0c1\ubd80", "TOP", "UPPER"],
    "bot": ["\ud558\ubd80", "BOT", "BOTTOM", "LOWER"],
    "stir": ["\ub760", "\uc2a4\ud130\ub7fd", "STIR", "@"],
}


def _has_any(s: str, kws: List[str]) -> bool:
    u = (s or "").upper()
    return any(k.upper() in u for k in kws)


def role_from_header(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    if _has_any(t, KW["token"]):
        return "token"
    if _has_any(t, KW["story"]):
        return "story"
    if _has_any(t, KW["section"]):
        return "section"
    if _has_any(t, KW["rebar"]) and _has_any(t, KW["top"]):
        return "rebar_top"
    if _has_any(t, KW["rebar"]) and _has_any(t, KW["bot"]):
        return "rebar_bot"
    if _has_any(t, KW["top"]):
        return "rebar_top"
    if _has_any(t, KW["bot"]):
        return "rebar_bot"
    if _has_any(t, KW["stir"]):
        return "stirrup"
    if _has_any(t, KW["rebar"]):
        return "rebar"
    return None


def infer_schema_v2(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not cells:
        return {"header_rows": [], "columns": {}, "confidence": 0.0, "debug": {}}

    by_row: Dict[int, List[Dict[str, Any]]] = {}
    max_row = 0
    for c in cells:
        r = int(c["row_idx"])
        by_row.setdefault(r, []).append(c)
        max_row = max(max_row, r)

    row_scores = []
    for r, row_cells in by_row.items():
        score = 0
        for c in row_cells:
            if role_from_header(str(c.get("text") or "")):
                score += 1
        row_scores.append((score, r))
    row_scores.sort(reverse=True)

    header_rows = []
    if row_scores and row_scores[0][0] >= 2:
        best_r = row_scores[0][1]
        header_rows = [best_r]
        if (best_r + 1) in by_row:
            s2 = 0
            for c in by_row[best_r + 1]:
                if role_from_header(str(c.get("text") or "")) or _has_any(str(c.get("text") or ""), KW["top"] + KW["bot"]):
                    s2 += 1
            if s2 >= 1:
                header_rows.append(best_r + 1)

    col_text: Dict[int, List[str]] = {}
    for hr in header_rows:
        for c in by_row.get(hr, []):
            col = int(c["col_idx"])
            col_text.setdefault(col, [])
            txt = str(c.get("text") or "").strip()
            if txt:
                col_text[col].append(txt)

    columns: Dict[str, int] = {}
    for col, parts in col_text.items():
        combined = " ".join(parts)
        role = role_from_header(combined)
        if role and role not in columns:
            columns[role] = col

    confidence = 0.35
    if columns:
        confidence = min(0.95, 0.6 + 0.1 * len(columns))

    if not columns:
        col_token_hits = {}
        col_story_hits = {}
        col_section_hits = {}
        for c in cells:
            col = int(c["col_idx"])
            txt = str(c.get("text") or "")
            if TOKEN_RX.search(txt):
                col_token_hits[col] = col_token_hits.get(col, 0) + 1
            if normalize_story(txt):
                col_story_hits[col] = col_story_hits.get(col, 0) + 1
            u = txt.upper().replace("\u00d7", "X")
            if ("H-" in u) or ("RHS" in u) or ("SHS" in u) or ("PIPE" in u) or ("CHS" in u) or ("\u25a1" in txt) or ("\u00d8" in txt):
                col_section_hits[col] = col_section_hits.get(col, 0) + 1

        if col_token_hits:
            columns["token"] = max(col_token_hits.items(), key=lambda kv: kv[1])[0]
            confidence = max(confidence, 0.55)
        if col_story_hits:
            columns.setdefault("story", max(col_story_hits.items(), key=lambda kv: kv[1])[0])
            confidence = max(confidence, 0.55)
        if col_section_hits:
            columns.setdefault("section", max(col_section_hits.items(), key=lambda kv: kv[1])[0])
            confidence = max(confidence, 0.5)

    return {
        "header_rows": header_rows,
        "columns": columns,
        "confidence": confidence,
        "debug": {"row_scores": row_scores[:10]},
    }


def _extract_token_and_story(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = (text or "").strip()
    if not t:
        return None, None
    tok = None
    m = TOKEN_RX.search(t.upper())
    if m:
        tok = re.sub(r"[\s\-_]+", "", m.group(0).upper())
    st = normalize_story(t)
    return tok, st


def parse_rows_v2(cells: List[Dict[str, Any]], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    cols = schema.get("columns") or {}
    header_rows = set(schema.get("header_rows") or [])

    by_row: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for c in cells:
        r = int(c["row_idx"])
        col = int(c["col_idx"])
        by_row.setdefault(r, {})
        by_row[r][col] = dict(c)

    out = []
    for r, rowmap in by_row.items():
        if r in header_rows:
            continue

        def pick(role: str) -> Optional[Dict[str, Any]]:
            if role not in cols:
                return None
            cc = rowmap.get(int(cols[role]))
            if not cc:
                return None
            txt = str(cc.get("text") or "").strip()
            if not txt:
                return None
            return {
                "text": txt,
                "cad_entity_id": cc.get("cad_entity_id"),
                "x": cc.get("x"),
                "y": cc.get("y"),
            }

        token_cell = pick("token")
        story_cell = pick("story")

        token_norm = None
        story_norm = None

        if token_cell:
            token_norm, story_norm = _extract_token_and_story(token_cell["text"])

        if story_cell and story_cell.get("text"):
            story_norm = normalize_story(story_cell["text"]) or story_norm

        fields = {}
        for role in ("token", "story", "section", "rebar", "rebar_top", "rebar_bot", "stirrup"):
            cell = pick(role)
            if cell:
                fields[role] = cell

        conf = 0.4
        if token_norm:
            conf += 0.25
        if story_norm:
            conf += 0.1
        if "section" in fields:
            conf += 0.2
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
