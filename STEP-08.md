좋아. 이제 8단계를 “실무 정밀도/자동화가 확 올라가는 구간”으로 잡고, 네가 말한 3가지를 순서대로 실제 구현 가능한 수준으로 내려줄게.

형강 표준 단면 라이브러리 우선 적용(카탈로그 값 > 계산값) + alias/우선순위 시스템

표(Table) 인식 강화(헤더/컬럼 역할 추정 → 행(row) 구조화 → 토큰/층/단면/철근을 ‘한 줄’로 묶어 배정)

다층(층별 동일 라벨) 문제 해결(스토리 태그 감지 + token×story 매핑 테이블 + story-aware link 생성)

⚠️ 안전/품질 고지

“계산으로 만든 단면특성(필렛 무시)”은 fallback이어야 하고, 표준값(카탈로그/KS/제조사)을 우선으로 쓰는 구조가 정답이야.

테이블/스토리 자동 배정은 프로젝트 표기 관행에 따라 오배정 가능성이 있으니, 추천 → 자동확정(보수적 정책) → 수동확정 흐름이 맞다.

8-1) DB 스키마 추가(v0.0.9)

mcp_server/schema.sql 맨 아래에 붙여줘.

-- ===== schema patch v0.0.9 =====
UPDATE meta SET value='0.0.9' WHERE key='schema_version';

-- ---------------------------------------
-- Story tags detected from CAD texts
-- ---------------------------------------
CREATE TABLE IF NOT EXISTS cad_story_tags (
  tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  cad_entity_id INTEGER REFERENCES cad_entities(cad_entity_id) ON DELETE SET NULL,

  story_norm TEXT NOT NULL,          -- e.g., "B2", "3F", "RF"
  raw_text TEXT NOT NULL,

  x REAL, y REAL, z REAL,
  layer TEXT,
  confidence REAL NOT NULL DEFAULT 0.6,

  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_story_tags_artifact ON cad_story_tags(cad_artifact_id, story_norm);

-- ---------------------------------------
-- Table schema inference + row parses
-- ---------------------------------------
CREATE TABLE IF NOT EXISTS cad_table_schemas (
  schema_id INTEGER PRIMARY KEY AUTOINCREMENT,
  table_id INTEGER NOT NULL REFERENCES cad_tables(table_id) ON DELETE CASCADE,

  header_row_idx INTEGER,
  columns_json TEXT NOT NULL,        -- {"token":0,"story":1,"section":2,"rebar_top":3,...}
  confidence REAL NOT NULL DEFAULT 0.5,

  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_table_schemas_table ON cad_table_schemas(table_id);

CREATE TABLE IF NOT EXISTS cad_table_row_parses (
  row_parse_id INTEGER PRIMARY KEY AUTOINCREMENT,
  table_id INTEGER NOT NULL REFERENCES cad_tables(table_id) ON DELETE CASCADE,

  row_idx INTEGER NOT NULL,
  token_norm TEXT,
  story_norm TEXT,
  fields_json TEXT NOT NULL,         -- {"token":{text,cad_entity_id,x,y}, "section":{...}, ...}
  confidence REAL NOT NULL DEFAULT 0.5,

  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_table_row_parses_table ON cad_table_row_parses(table_id, row_idx);

-- ---------------------------------------
-- Token × Story map (multi-story disambiguation)
-- ---------------------------------------
CREATE TABLE IF NOT EXISTS token_story_maps (
  map_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,

  cad_token_norm TEXT NOT NULL,
  story_norm TEXT NOT NULL,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,

  confidence REAL NOT NULL DEFAULT 0.7,
  method TEXT NOT NULL DEFAULT 'inferred',  -- inferred|table|manual
  status TEXT NOT NULL DEFAULT 'suggested', -- suggested|confirmed|rejected
  evidence_json TEXT NOT NULL DEFAULT '{}',

  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),

  UNIQUE(cad_artifact_id, model_id, cad_token_norm, story_norm)
);

CREATE INDEX IF NOT EXISTS idx_token_story_maps_lookup
ON token_story_maps(cad_artifact_id, model_id, status, cad_token_norm, story_norm);

8-2) “스토리(층) 태그” 감지 모듈 추가
8-2-1) 새 파일: mcp_server/parsing/story.py
# mcp_server/parsing/story.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# 예시 커버:
# - "3F", "03F", "3층", "지상3층"
# - "B2", "B2F", "지하2층"
# - "RF", "ROOF", "옥상"
BASEMENT_RX = re.compile(r"(?:\b|^)(?:B|지하)\s*0*(\d{1,2})\s*(?:층|F)?(?:\b|$)", re.IGNORECASE)
FLOOR_RX    = re.compile(r"(?:\b|^)(?:지상)?\s*0*(\d{1,2})\s*(?:층|F)(?:\b|$)", re.IGNORECASE)
RF_RX       = re.compile(r"(?:\b|^)(RF|ROOF|옥상)(?:\b|$)", re.IGNORECASE)
LEVEL_RX    = re.compile(r"(?:\b|^)(?:LV\.?|LEVEL)\s*0*(\d{1,2})(?:\b|$)", re.IGNORECASE)

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
        # LEVEL 3 -> 3F로 취급(프로젝트별 다름; 필요하면 별도 규칙)
        n = int(m.group(1))
        if 0 <= n <= 200:
            return f"{n}F"

    # "3"만 단독으로 쓰는 경우는 오인식 위험이 커서 기본은 None
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
        # 문자열 길이가 너무 길어도(예: "3F 평면도")는 정상일 수 있음
        # confidence는 패턴 매칭 강도 기반
        conf = 0.85
        out.append({"story_norm": s, "confidence": conf, "raw_fragment": t})
    return out

8-3) 테이블 “헤더/컬럼 역할” 추정 + 행(row) 구조화
8-3-1) 새 파일: mcp_server/parsing/table_schema.py
# mcp_server/parsing/table_schema.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .story import normalize_story

TOKEN_RX = re.compile(r"\b[A-Z]{1,4}\s?-?\s?\d{1,4}\b", re.IGNORECASE)

# 헤더 키워드
KW_TOKEN = ["부재", "기호", "부재명", "부재번호", "MARK", "MEMBER", "NAME", "ID", "NO"]
KW_STORY = ["층", "STORY", "FLOOR", "LEVEL", "LV"]
KW_SECTION = ["단면", "규격", "SECTION", "SIZE", "형강", "H형강", "STEEL"]
KW_REBAR = ["철근", "REBAR", "주근"]
KW_TOP = ["상부", "TOP", "UPPER"]
KW_BOT = ["하부", "BOT", "BOTTOM", "LOWER"]
KW_STIR = ["띠", "스터럽", "STIR", "@"]

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

    # rebar columns
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
    """
    cells: [{row_idx,col_idx,text,cad_entity_id,x,y}]
    returns {header_row_idx, columns:{role:col}, confidence, debug}
    """
    if not cells:
        return {"header_row_idx": None, "columns": {}, "confidence": 0.0, "debug": {}}

    # group by row
    by_row: Dict[int, List[Dict[str, Any]]] = {}
    max_row = 0
    max_col = 0
    for c in cells:
        r = int(c["row_idx"]); k = int(c["col_idx"])
        max_row = max(max_row, r); max_col = max(max_col, k)
        by_row.setdefault(r, []).append(c)

    # 1) header row scoring
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

        # assign columns by header cell content
        header_cells = by_row.get(best_row, [])
        for c in header_cells:
            role = role_from_header_cell(str(c.get("text") or ""))
            if role and role not in columns:
                columns[role] = int(c["col_idx"])

    # 2) fallback content-based inference if header weak
    if not columns:
        # token col: most token matches
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

        # section col: heuristic: texts containing 'H-' or 'x' patterns
        col_sec_hits = {}
        for c in cells:
            col = int(c["col_idx"])
            txt = str(c.get("text") or "").upper().replace("×", "X")
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
    """
    schema.columns 기반으로 row 파싱 -> row objects
    """
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
        # row->col map
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

        # fields pack
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

        # confidence heuristic
        conf = 0.4
        if token_norm:
            conf += 0.25
        if section_cell and section_cell.get("text"):
            conf += 0.2
        if story_norm:
            conf += 0.1
        if any(k in fields for k in ("rebar","rebar_top","rebar_bot","stirrup")):
            conf += 0.1

        out.append({
            "row_idx": int(r),
            "token_norm": token_norm,
            "story_norm": story_norm,
            "fields": fields,
            "confidence": float(min(0.95, conf)),
        })
    return out

8-4) 서버(server.py)에 Step8용 MCP Tools 추가

아래는 **기존 Step7/6 서버에 “추가”**하면 되는 형태야.
(코드 블록 그대로 붙일 수 있게 구성)

8-4-1) import 추가

server.py 상단에 추가:

from mcp_server.parsing.story import extract_story_candidates, normalize_story
from mcp_server.parsing.table_schema import infer_schema, parse_rows
from mcp_server.parsing.specs import parse_specs_from_text

8-4-2) CAD 스토리 태그 감지 저장
@mcp.tool()
def structai_cad_detect_story_tags(
    cad_artifact_id: int,
    overwrite: bool = True,
    include_layers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    CAD 텍스트에서 층 표기(3F/B2F/RF/3층/지하2층 등)를 감지해 cad_story_tags에 저장
    """
    conn = _connect()
    try:
        if overwrite:
            conn.execute("DELETE FROM cad_story_tags WHERE cad_artifact_id=?", (int(cad_artifact_id),))

        rows = conn.execute(
            """
            SELECT cad_entity_id, layer, x,y,z, text
            FROM cad_entities
            WHERE artifact_id=? AND type IN ('TEXT','MTEXT','ATTRIB') AND text IS NOT NULL
              AND x IS NOT NULL AND y IS NOT NULL
            """,
            (int(cad_artifact_id),),
        ).fetchall()

        inserted = 0
        items = []

        for r in rows:
            layer = r["layer"] or ""
            if include_layers:
                if not any(k.upper() in layer.upper() for k in include_layers):
                    continue

            raw = str(r["text"] or "").strip()
            if not raw:
                continue

            cands = extract_story_candidates(raw)
            for c in cands:
                conn.execute(
                    """
                    INSERT INTO cad_story_tags(
                      cad_artifact_id, cad_entity_id, story_norm, raw_text,
                      x,y,z, layer, confidence
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        int(cad_artifact_id),
                        int(r["cad_entity_id"]),
                        str(c["story_norm"]),
                        raw,
                        r["x"], r["y"], r["z"],
                        r["layer"],
                        float(c.get("confidence", 0.6)),
                    ),
                )
                inserted += 1
                if len(items) < 50:
                    items.append({"story_norm": c["story_norm"], "raw_text": raw, "layer": r["layer"]})

        conn.commit()
        return {"ok": True, "cad_artifact_id": int(cad_artifact_id), "inserted": inserted, "sample": items}
    finally:
        conn.close()

8-4-3) (핵심) 테이블 스키마 추정 + 행 파싱 저장
@mcp.tool()
def structai_cad_infer_table_schemas(
    cad_artifact_id: int,
    overwrite: bool = True,
    min_table_confidence: float = 0.0
) -> Dict[str, Any]:
    """
    cad_tables -> 헤더/컬럼 역할 추정 -> cad_table_schemas 저장
    + 각 table의 row parse 결과를 cad_table_row_parses에 저장
    """
    conn = _connect()
    try:
        if overwrite:
            # table_id 기준 cascade 삭제
            conn.execute(
                """
                DELETE FROM cad_table_schemas
                WHERE table_id IN (SELECT table_id FROM cad_tables WHERE cad_artifact_id=?)
                """,
                (int(cad_artifact_id),),
            )
            conn.execute(
                """
                DELETE FROM cad_table_row_parses
                WHERE table_id IN (SELECT table_id FROM cad_tables WHERE cad_artifact_id=?)
                """,
                (int(cad_artifact_id),),
            )

        tables = conn.execute(
            "SELECT table_id, confidence, meta_json FROM cad_tables WHERE cad_artifact_id=? ORDER BY confidence DESC",
            (int(cad_artifact_id),),
        ).fetchall()

        saved_schema = 0
        saved_rows = 0
        sample = []

        for t in tables:
            if float(t["confidence"] or 0.0) < float(min_table_confidence):
                continue
            table_id = int(t["table_id"])

            cells = conn.execute(
                """
                SELECT row_idx, col_idx, cad_entity_id, text, x,y
                FROM cad_table_cells
                WHERE table_id=?
                """,
                (table_id,),
            ).fetchall()
            cell_list = [dict(r) for r in cells]

            sch = infer_schema(cell_list)
            conn.execute(
                """
                INSERT INTO cad_table_schemas(table_id, header_row_idx, columns_json, confidence)
                VALUES(?,?,?,?)
                """,
                (table_id, sch.get("header_row_idx"), json.dumps(sch.get("columns") or {}, ensure_ascii=False), float(sch.get("confidence", 0.5))),
            )
            saved_schema += 1

            rows_parsed = parse_rows(cell_list, sch)
            for rp in rows_parsed:
                conn.execute(
                    """
                    INSERT INTO cad_table_row_parses(table_id, row_idx, token_norm, story_norm, fields_json, confidence)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (
                        table_id,
                        int(rp["row_idx"]),
                        rp.get("token_norm"),
                        rp.get("story_norm"),
                        json.dumps(rp.get("fields") or {}, ensure_ascii=False),
                        float(rp.get("confidence", 0.5)),
                    ),
                )
                saved_rows += 1
                if len(sample) < 20 and rp.get("token_norm"):
                    sample.append({"table_id": table_id, "row_idx": rp["row_idx"], "token": rp["token_norm"], "story": rp.get("story_norm"), "fields": list((rp.get("fields") or {}).keys())})

        conn.commit()
        return {"ok": True, "cad_artifact_id": int(cad_artifact_id), "schemas": saved_schema, "rows": saved_rows, "sample": sample}
    finally:
        conn.close()

8-4-4) Token×Story map 생성(다층 동일 라벨 해결)

이 도구는 “테이블 row에 story가 있으면” 그걸 가장 신뢰하고, 없으면 “스토리 태그 근접”으로 보완해.

def _norm_story_model(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return normalize_story(str(s)) or str(s).strip().upper().replace(" ", "")

def _nearest_story_for_point(conn, cad_artifact_id: int, x: float, y: float, max_dist: float = 2000.0) -> Optional[str]:
    # 단순: 가장 가까운 story tag 1개
    rows = conn.execute(
        """
        SELECT story_norm, x, y
        FROM cad_story_tags
        WHERE cad_artifact_id=? AND x IS NOT NULL AND y IS NOT NULL
        """,
        (int(cad_artifact_id),),
    ).fetchall()
    best = None
    for r in rows:
        dx = float(r["x"]) - x
        dy = float(r["y"]) - y
        d = (dx*dx + dy*dy) ** 0.5
        if d <= max_dist and (best is None or d < best[0]):
            best = (d, r["story_norm"])
    return best[1] if best else None

@mcp.tool()
def structai_token_story_map_build(
    cad_artifact_id: int,
    model_id: int,
    mapping_status: str = "confirmed",
    overwrite: bool = True,
    max_story_tag_dist: float = 2000.0
) -> Dict[str, Any]:
    """
    token×story -> model_member_id 추천 생성(token_story_maps)
    우선순위:
      1) table row parse에서 token+story 얻음
      2) token 텍스트 위치에서 story tag 근접 추정
      3) model member 자체 story와 결합(가능 시)
    """
    conn = _connect()
    try:
        if overwrite:
            conn.execute("DELETE FROM token_story_maps WHERE cad_artifact_id=? AND model_id=?", (int(cad_artifact_id), int(model_id)))

        # token -> member candidates (from confirmed mappings)
        maps = conn.execute(
            """
            SELECT cad_token_norm, model_member_id
            FROM member_mappings
            WHERE cad_artifact_id=? AND model_id=? AND status=?
            """,
            (int(cad_artifact_id), int(model_id), str(mapping_status)),
        ).fetchall()

        token_to_members: Dict[str, List[int]] = {}
        for r in maps:
            token_to_members.setdefault(r["cad_token_norm"], []).append(int(r["model_member_id"]))

        # member story map
        mem_rows = conn.execute(
            "SELECT model_member_id, story FROM model_members WHERE model_id=?",
            (int(model_id),),
        ).fetchall()
        member_story = {int(r["model_member_id"]): _norm_story_model(r["story"]) for r in mem_rows}

        created = 0
        sample = []

        # (A) from table row parses
        table_rows = conn.execute(
            """
            SELECT trp.table_id, trp.row_idx, trp.token_norm, trp.story_norm, trp.fields_json, trp.confidence
            FROM cad_table_row_parses trp
            JOIN cad_tables t ON t.table_id = trp.table_id
            WHERE t.cad_artifact_id=? AND trp.token_norm IS NOT NULL AND trp.story_norm IS NOT NULL
            """,
            (int(cad_artifact_id),),
        ).fetchall()

        for r in table_rows:
            tok = str(r["token_norm"])
            st = str(r["story_norm"])
            candidates = token_to_members.get(tok, [])
            if not candidates:
                continue

            # story match 우선
            match = [mid for mid in candidates if (member_story.get(mid) == st)]
            chosen = None
            conf = 0.8 + 0.15 * float(r["confidence"] or 0.5)

            if len(match) == 1:
                chosen = match[0]
                conf = min(0.98, conf + 0.1)
            elif len(candidates) == 1:
                chosen = candidates[0]
                conf = min(0.9, conf)
            else:
                # 모호하면 저장 안 하거나 suggested만 낮게
                chosen = None

            if chosen is None:
                continue

            evidence = {"source": "table_row", "table_id": int(r["table_id"]), "row_idx": int(r["row_idx"])}
            conn.execute(
                """
                INSERT INTO token_story_maps(cad_artifact_id, model_id, cad_token_norm, story_norm, model_member_id,
                                             confidence, method, status, evidence_json, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?, datetime('now'))
                ON CONFLICT(cad_artifact_id, model_id, cad_token_norm, story_norm)
                DO UPDATE SET
                  model_member_id=excluded.model_member_id,
                  confidence=excluded.confidence,
                  method=excluded.method,
                  status=excluded.status,
                  evidence_json=excluded.evidence_json,
                  updated_at=datetime('now')
                """,
                (int(cad_artifact_id), int(model_id), tok, st, int(chosen), float(conf), "table", "suggested", json.dumps(evidence, ensure_ascii=False)),
            )
            created += 1
            if len(sample) < 30:
                sample.append({"token": tok, "story": st, "model_member_id": chosen, "method": "table", "confidence": conf})

        # (B) from token occurrence nearest story tag
        # 이미 table에서 채워진 key는 skip
        existing = conn.execute(
            "SELECT cad_token_norm, story_norm FROM token_story_maps WHERE cad_artifact_id=? AND model_id=?",
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()
        existing_keys = {(str(r["cad_token_norm"]), str(r["story_norm"])) for r in existing}

        for tok, candidates in token_to_members.items():
            # token CAD 위치들
            pts = conn.execute(
                """
                SELECT x,y
                FROM cad_entities
                WHERE artifact_id=? AND text IS NOT NULL AND x IS NOT NULL AND y IS NOT NULL
                  AND instr(upper(text), ?) > 0
                LIMIT 50
                """,
                (int(cad_artifact_id), tok.upper()),
            ).fetchall()
            if not pts:
                continue

            # 각 점에서 story 추정 -> 빈도
            counts: Dict[str, int] = {}
            for p in pts:
                st = _nearest_story_for_point(conn, cad_artifact_id, float(p["x"]), float(p["y"]), max_dist=max_story_tag_dist)
                if st:
                    counts[st] = counts.get(st, 0) + 1
            if not counts:
                continue

            # story별로 member 선택
            for st, n in counts.items():
                if (tok, st) in existing_keys:
                    continue
                match = [mid for mid in candidates if member_story.get(mid) == st]
                chosen = None
                if len(match) == 1:
                    chosen = match[0]
                    conf = min(0.95, 0.75 + 0.02 * n)
                elif len(candidates) == 1:
                    chosen = candidates[0]
                    conf = min(0.85, 0.70 + 0.01 * n)
                else:
                    continue

                evidence = {"source": "nearest_story_tag", "count": n, "max_dist": max_story_tag_dist}
                conn.execute(
                    """
                    INSERT INTO token_story_maps(cad_artifact_id, model_id, cad_token_norm, story_norm, model_member_id,
                                                 confidence, method, status, evidence_json, updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?, datetime('now'))
                    ON CONFLICT(cad_artifact_id, model_id, cad_token_norm, story_norm)
                    DO UPDATE SET
                      model_member_id=excluded.model_member_id,
                      confidence=excluded.confidence,
                      method=excluded.method,
                      status=excluded.status,
                      evidence_json=excluded.evidence_json,
                      updated_at=datetime('now')
                    """,
                    (int(cad_artifact_id), int(model_id), tok, st, int(chosen), float(conf), "inferred", "suggested", json.dumps(evidence, ensure_ascii=False)),
                )
                created += 1
                if len(sample) < 30:
                    sample.append({"token": tok, "story": st, "model_member_id": chosen, "method": "inferred", "confidence": conf})

        conn.commit()
        return {"ok": True, "created": created, "sample": sample}
    finally:
        conn.close()

8-4-5) (핵심) 테이블 행 → cad_specs 생성 + story-aware member_spec_links 생성

이 도구가 Step8의 “실무 체감”을 만든다.
즉, “부재표 한 줄”에서 token + story + section + rebar를 묶어서, 가장 정확하게 부재에 배정한다.

@mcp.tool()
def structai_specs_from_table_rows(
    cad_artifact_id: int,
    model_id: int,
    overwrite_specs: bool = False,
    overwrite_links_suggested: bool = False
) -> Dict[str, Any]:
    """
    cad_table_row_parses의 fields를 이용해:
      - cad_specs 생성(단면/철근/스터럽)
      - member_spec_links 생성(method='table_schema')
    story_norm이 있으면 token_story_maps(confirmed/suggested)로 target member를 먼저 결정.
    """
    conn = _connect()
    try:
        if overwrite_specs:
            conn.execute("DELETE FROM cad_specs WHERE cad_artifact_id=? AND (raw_text LIKE 'TABLE_ROW:%')", (int(cad_artifact_id),))

        if overwrite_links_suggested:
            conn.execute(
                "DELETE FROM member_spec_links WHERE cad_artifact_id=? AND model_id=? AND status='suggested' AND method='table_schema'",
                (int(cad_artifact_id), int(model_id)),
            )

        # token->members fallback (confirmed mappings)
        maps = conn.execute(
            """
            SELECT cad_token_norm, model_member_id
            FROM member_mappings
            WHERE cad_artifact_id=? AND model_id=? AND status='confirmed'
            """,
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()
        token_to_members: Dict[str, List[int]] = {}
        for r in maps:
            token_to_members.setdefault(r["cad_token_norm"], []).append(int(r["model_member_id"]))

        # token_story_maps (prefer confirmed)
        tsm = conn.execute(
            """
            SELECT cad_token_norm, story_norm, model_member_id, status, confidence
            FROM token_story_maps
            WHERE cad_artifact_id=? AND model_id=? AND status IN ('confirmed','suggested')
            """,
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()
        token_story_best: Dict[Tuple[str,str], Tuple[int, float, str]] = {}
        for r in tsm:
            key = (str(r["cad_token_norm"]), str(r["story_norm"]))
            cur = token_story_best.get(key)
            score = float(r["confidence"] or 0.7) + (0.2 if r["status"] == "confirmed" else 0.0)
            if (cur is None) or (score > cur[1]):
                token_story_best[key] = (int(r["model_member_id"]), score, str(r["status"]))

        # table rows
        rows = conn.execute(
            """
            SELECT trp.table_id, trp.row_idx, trp.token_norm, trp.story_norm, trp.fields_json, trp.confidence
            FROM cad_table_row_parses trp
            JOIN cad_tables t ON t.table_id = trp.table_id
            WHERE t.cad_artifact_id=? AND trp.token_norm IS NOT NULL
            ORDER BY trp.confidence DESC
            """,
            (int(cad_artifact_id),),
        ).fetchall()

        created_specs = 0
        created_links = 0
        sample = []

        def insert_spec(spec_kind: str, spec: dict, raw_text: str, cad_entity_id, x, y, layer=None, conf=0.85) -> int:
            nonlocal created_specs
            conn.execute(
                """
                INSERT INTO cad_specs(
                  cad_artifact_id, cad_entity_id, spec_kind, spec_json, raw_text,
                  x,y,z, layer, confidence
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (int(cad_artifact_id), cad_entity_id, spec_kind, json.dumps(spec, ensure_ascii=False), raw_text, x, y, None, layer, float(conf)),
            )
            created_specs += 1
            return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

        for r in rows:
            tok = str(r["token_norm"])
            story = r["story_norm"]
            fields = json.loads(r["fields_json"] or "{}")

            # target member 결정
            target_member_ids: List[int] = []
            if story:
                key = (tok, str(story))
                if key in token_story_best:
                    target_member_ids = [token_story_best[key][0]]
                else:
                    # story match 가능한 후보만 남김
                    candidates = token_to_members.get(tok, [])
                    if candidates:
                        # model story match
                        ms = conn.execute(
                            f"SELECT model_member_id, story FROM model_members WHERE model_member_id IN ({','.join(['?']*len(candidates))})",
                            candidates,
                        ).fetchall()
                        match = [int(x["model_member_id"]) for x in ms if _norm_story_model(x["story"]) == str(story)]
                        if len(match) == 1:
                            target_member_ids = [match[0]]
                        elif len(candidates) == 1:
                            target_member_ids = [candidates[0]]
                        else:
                            target_member_ids = candidates  # 모호하면 다 suggested로
            else:
                candidates = token_to_members.get(tok, [])
                if len(candidates) == 1:
                    target_member_ids = [candidates[0]]
                elif candidates:
                    target_member_ids = candidates

            if not target_member_ids:
                continue

            # ---- cad_specs 생성: section/rebar/stirrup ----
            new_spec_ids: List[Tuple[int, str]] = []

            # section
            if "section" in fields and fields["section"].get("text"):
                sec_text = fields["section"]["text"]
                sec_specs = parse_specs_from_text(sec_text)
                for s in sec_specs:
                    sid = insert_spec(
                        s["spec_kind"],
                        s,
                        raw_text=f"TABLE_ROW:section:{tok}:{story or ''}:{sec_text}",
                        cad_entity_id=fields["section"].get("cad_entity_id"),
                        x=fields["section"].get("x"),
                        y=fields["section"].get("y"),
                        layer=None,
                        conf=0.92,
                    )
                    new_spec_ids.append((sid, s["spec_kind"]))

            # rebar top/bot: pos 강제 힌트를 위해 prefix 붙여 파싱
            if "rebar_top" in fields and fields["rebar_top"].get("text"):
                txt = "TOP " + fields["rebar_top"]["text"]
                for s in parse_specs_from_text(txt):
                    sid = insert_spec(
                        s["spec_kind"],
                        s,
                        raw_text=f"TABLE_ROW:rebar_top:{tok}:{story or ''}:{fields['rebar_top']['text']}",
                        cad_entity_id=fields["rebar_top"].get("cad_entity_id"),
                        x=fields["rebar_top"].get("x"),
                        y=fields["rebar_top"].get("y"),
                        conf=0.9,
                    )
                    new_spec_ids.append((sid, s["spec_kind"]))

            if "rebar_bot" in fields and fields["rebar_bot"].get("text"):
                txt = "BOT " + fields["rebar_bot"]["text"]
                for s in parse_specs_from_text(txt):
                    sid = insert_spec(
                        s["spec_kind"],
                        s,
                        raw_text=f"TABLE_ROW:rebar_bot:{tok}:{story or ''}:{fields['rebar_bot']['text']}",
                        cad_entity_id=fields["rebar_bot"].get("cad_entity_id"),
                        x=fields["rebar_bot"].get("x"),
                        y=fields["rebar_bot"].get("y"),
                        conf=0.9,
                    )
                    new_spec_ids.append((sid, s["spec_kind"]))

            # generic rebar column
            if "rebar" in fields and fields["rebar"].get("text"):
                txt = fields["rebar"]["text"]
                for s in parse_specs_from_text(txt):
                    sid = insert_spec(
                        s["spec_kind"],
                        s,
                        raw_text=f"TABLE_ROW:rebar:{tok}:{story or ''}:{txt}",
                        cad_entity_id=fields["rebar"].get("cad_entity_id"),
                        x=fields["rebar"].get("x"),
                        y=fields["rebar"].get("y"),
                        conf=0.85,
                    )
                    new_spec_ids.append((sid, s["spec_kind"]))

            # stirrup
            if "stirrup" in fields and fields["stirrup"].get("text"):
                txt = fields["stirrup"]["text"]
                for s in parse_specs_from_text(txt):
                    sid = insert_spec(
                        s["spec_kind"],
                        s,
                        raw_text=f"TABLE_ROW:stirrup:{tok}:{story or ''}:{txt}",
                        cad_entity_id=fields["stirrup"].get("cad_entity_id"),
                        x=fields["stirrup"].get("x"),
                        y=fields["stirrup"].get("y"),
                        conf=0.9,
                    )
                    new_spec_ids.append((sid, s["spec_kind"]))

            # ---- link 생성 ----
            for (spec_id, spec_kind) in new_spec_ids:
                for mmid in target_member_ids:
                    evidence = {
                        "source": "table_schema",
                        "table_id": int(r["table_id"]),
                        "row_idx": int(r["row_idx"]),
                        "token_norm": tok,
                        "story_norm": story,
                        "row_confidence": float(r["confidence"] or 0.5),
                        "spec_kind": spec_kind,
                        "note": "multi-target if ambiguous token within story",
                    }
                    conn.execute(
                        """
                        INSERT INTO member_spec_links(
                          cad_artifact_id, spec_id, model_id, model_member_id,
                          cad_token_norm, distance, method, status, evidence_json, updated_at
                        ) VALUES(?,?,?,?,?,?,?,'suggested',?, datetime('now'))
                        ON CONFLICT(cad_artifact_id, spec_id, model_member_id)
                        DO UPDATE SET
                          method='table_schema',
                          status='suggested',
                          distance=0,
                          evidence_json=excluded.evidence_json,
                          updated_at=datetime('now')
                        """,
                        (int(cad_artifact_id), int(spec_id), int(model_id), int(mmid), tok, 0.0, "table_schema", json.dumps(evidence, ensure_ascii=False)),
                    )
                    created_links += 1

            if len(sample) < 20 and new_spec_ids:
                sample.append({"token": tok, "story": story, "specs": [k for _,k in new_spec_ids], "target_member_ids": target_member_ids})

        conn.commit()
        return {"ok": True, "created_specs": created_specs, "created_links": created_links, "sample": sample}
    finally:
        conn.close()

8-4-6) “보수적 자동 확정” 정책(실무용)

table_schema에서 생성된 링크는 정확도가 높으니 기본 confirm

단, token이 같은 story에서 여러 member로 모호하면 자동확정하지 않음(보수적)

@mcp.tool()
def structai_specs_auto_confirm_table_schema(
    cad_artifact_id: int,
    model_id: int
) -> Dict[str, Any]:
    """
    table_schema method 링크 중:
      - 동일 (token_norm, story_norm)에서 target member가 유일하면 confirmed
      - 모호하면 suggested 유지
    """
    conn = _connect()
    try:
        links = conn.execute(
            """
            SELECT l.link_id, l.model_member_id, l.evidence_json
            FROM member_spec_links l
            WHERE l.cad_artifact_id=? AND l.model_id=? AND l.status='suggested' AND l.method='table_schema'
            """,
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()

        # group by (token, story) from evidence
        groups: Dict[Tuple[str,str], List[int]] = {}
        link_info: Dict[int, Tuple[str,str,int]] = {}

        for r in links:
            lid = int(r["link_id"])
            ev = json.loads(r["evidence_json"] or "{}")
            tok = str(ev.get("token_norm") or "")
            st = str(ev.get("story_norm") or "")
            mmid = int(r["model_member_id"])
            if not tok:
                continue
            key = (tok, st)
            groups.setdefault(key, [])
            groups[key].append(mmid)
            link_info[lid] = (tok, st, mmid)

        confirm_ids = []
        for lid, (tok, st, mmid) in link_info.items():
            key = (tok, st)
            # 유일 member이면 confirm
            if len(set(groups.get(key, []))) == 1:
                confirm_ids.append(lid)

        if confirm_ids:
            conn.execute(
                f"UPDATE member_spec_links SET status='confirmed', updated_at=datetime('now') WHERE link_id IN ({','.join(['?']*len(confirm_ids))})",
                confirm_ids,
            )

        conn.commit()
        return {"ok": True, "confirmed": len(confirm_ids), "kept_suggested": len(links) - len(confirm_ids)}
    finally:
        conn.close()

8-5) 형강 “표준 라이브러리 우선 적용” (priority + alias)

Step7에서 section_catalog를 만들었는데, Step8에서는 여기의 핵심이:

**카탈로그 값(ks_catalog/csv_import)**이 있으면 그걸 우선

없을 때만 computed로 채우기

표기 흔들림(“H400x200x8x13”, “H-400×200×8×13” 등)은 alias로 해결

8-5-1) 섹션 테이블에 priority 컬럼 추가(마이그레이션 안전 도구)

SQLite에서 ALTER TABLE ... ADD COLUMN은 가능하니, 도구로 처리하면 DB reset 없이도 가능해.

@mcp.tool()
def structai_sections_migrate_add_priority() -> Dict[str, Any]:
    """
    section_catalog에 priority 컬럼이 없으면 추가.
    """
    conn = _connect()
    try:
        cols = conn.execute("PRAGMA table_info(section_catalog)").fetchall()
        names = {c["name"] for c in cols}
        if "priority" in names:
            return {"ok": True, "changed": False, "note": "priority already exists"}

        conn.execute("ALTER TABLE section_catalog ADD COLUMN priority INTEGER NOT NULL DEFAULT 50")
        conn.commit()
        return {"ok": True, "changed": True}
    finally:
        conn.close()

8-5-2) upsert 로직 개선(“우선순위 좋은 값만 덮어쓰기”)

Step7의 _upsert_section을 이 버전으로 교체해줘(중요).

def _upsert_section(conn, family: str, name: str, dims: dict, props: dict, source: str, priority: int = 50) -> int:
    name_norm = _norm(name)

    # 우선순위가 낮을수록(숫자 작을수록) 더 "신뢰도 높은 소스"
    conn.execute(
        """
        INSERT INTO section_catalog(family, name, name_norm, dims_json, props_json, source, priority)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(family, name_norm) DO UPDATE SET
          name=excluded.name,
          dims_json=excluded.dims_json,
          props_json=excluded.props_json,
          source=excluded.source,
          priority=excluded.priority
        WHERE excluded.priority <= section_catalog.priority
        """,
        (family, name, name_norm,
         json.dumps(dims, ensure_ascii=False),
         json.dumps(props, ensure_ascii=False),
         source, int(priority)),
    )
    r = conn.execute("SELECT section_id FROM section_catalog WHERE family=? AND name_norm=?", (family, name_norm)).fetchone()
    return int(r["section_id"])


이렇게 하면 “computed(80)”이 이미 들어가 있어도 나중에 “ks_catalog(10)”를 임포트하면 덮어쓰고,
반대로 ks_catalog가 있으면 computed가 와도 덮어쓰지 못함.

8-5-3) alias 등록 도구
@mcp.tool()
def structai_sections_add_alias(
    family: str,
    canonical_name: str,
    aliases: List[str]
) -> Dict[str, Any]:
    """
    alias_norm -> section_id 매핑 등록
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT section_id FROM section_catalog WHERE family=? AND name_norm=?",
            (family, _norm(canonical_name)),
        ).fetchone()
        if not row:
            raise ValueError("canonical section not found in catalog")

        sid = int(row["section_id"])
        added = 0
        for a in aliases:
            an = _norm(a)
            if not an:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO section_aliases(alias_norm, section_id) VALUES(?,?)",
                (an, sid),
            )
            added += 1

        conn.commit()
        return {"ok": True, "section_id": sid, "added": added}
    finally:
        conn.close()

8-5-4) 섹션 resolve 시 alias 우선 적용(개선)

Step7의 structai_sections_resolve_members에서 “catalog lookup”을 할 때,
name_norm 조회 전에 alias 조회를 먼저 넣어줘.

핵심 패치(개념):

# before: SELECT section_catalog by (family,name_norm)

alias = conn.execute("SELECT section_id FROM section_aliases WHERE alias_norm=?", (name_norm,)).fetchone()
if alias:
    section_id = int(alias["section_id"])
    r = conn.execute("SELECT dims_json, props_json FROM section_catalog WHERE section_id=?", (section_id,)).fetchone()
    ...

8-6) Step8 “한 번에 굴리는” 운영 순서

아래 순서대로 실행하면, 다층/표/단면라이브러리까지 한 바퀴 돈다.

(선택) 섹션 priority 컬럼 마이그레이션

#structai_sections_migrate_add_priority {}


(선택) 표준 단면 라이브러리 임포트
(네가 가진 KS/제조사 카탈로그 CSV/JSON을 임포트)

#structai_sections_import_catalog { "path": ".../steel_sections.csv" }


DXF 임포트(텍스트 + 가능하면 LINE/LWPOLYLINE까지)
(이건 Step7에서 importer 강화한 버전 기준)

스토리 태그 감지

#structai_cad_detect_story_tags { "cad_artifact_id": 1, "overwrite": true }


테이블 추출(이미 했어도 OK)

#structai_cad_extract_tables { "cad_artifact_id": 1, "min_cells": 16 }


테이블 스키마 추정 + row parse 저장

#structai_cad_infer_table_schemas { "cad_artifact_id": 1, "overwrite": true }


(필수) 도면 토큰 ↔ 모델 부재 매핑은 confirmed 상태여야 함 (Step3)

token×story map 생성(다층 해결)

#structai_token_story_map_build { "cad_artifact_id": 1, "model_id": 1 }


테이블 row 기반 specs + links 생성(정밀 배정)

#structai_specs_from_table_rows { "cad_artifact_id": 1, "model_id": 1 }


보수적 자동 확정(table_schema 기반)

#structai_specs_auto_confirm_table_schema { "cad_artifact_id": 1, "model_id": 1 }


확정된 specs → design_inputs 적용(Step6의 apply 도구 사용)

#structai_design_apply_specs_to_inputs { "cad_artifact_id": 1, "model_id": 1, "overwrite_keys": false }


섹션 resolve (모델 section 문자열 기반, CAD 기반이 부족할 때)

#structai_sections_resolve_members { "model_id": 1, "default_Fy_MPa": 325, "overwrite_design": false }


내력 계산(steel/rc) → checks → report
(Step5/6/7에서 만든 흐름 그대로)

8-7) 여기까지 하면 “대화형 구조검토”가 어떻게 달라지나

이제 AI 대화창에서 가능해지는 흐름은 아래야(결정론 기반 + 근거 추적 가능):

“3F 보 B12 휨 검토가 NA인데 왜 그래?”
→ validate_ready_for_checks가 **‘Mn_neg가 없음(상부근 누락)’**이라고 정확히 말해줌
→ “부재표에서 B12(3F) 상부근 컬럼을 확인해서 확정하세요”로 안내
→ table_schema 기반 링크가 있으면 자동으로 design_inputs 채워지고 체크가 PASS/WARN/FAIL로 바뀜

“H-400x200x8x13 단면특성이 프로젝트마다 달라”
→ 이제는 section_catalog에서 ks_catalog 우선값을 쓰고, 없으면 computed fallback
→ 보고서에 section_resolved.source/priority가 남아서 감사 가능

“B12 라벨이 2F/3F/4F에 다 있는데 혼동되던데?”
→ token_story_maps가 story별로 member를 분리하고, 테이블 row story로 배정되므로 혼동이 크게 줄어듦

8-8) 다음(9단계)에서 “완성도”가 급상승하는 것

Step8까지가 “정밀한 자동 배정”의 핵심 뼈대고, 다음 단계는 실무 적용성의 마지막 벽이야:

단면 라이브러리 데이터셋 품질 강화

KS/제조사 공칭값(필렛 포함)을 표준으로

형강/각형강관/원형강관/앵글/채널까지 확장

테이블 인식 고도화

헤더 다중행/병합셀/블록 속성(ATTRIB) 테이블 대응 강화

“열 역할 학습(룰 기반 + 통계)”로 프로젝트별 자동 적응

검증 하네스(Regression)

샘플 프로젝트 3~5개를 골라 상용 프로그램/수기와 결과 비교 자동화

이게 있어야 현장 적용 신뢰가 생김

원하면 다음 응답에서 **9단계(단면 라이브러리 확장 + 테이블 고도화 + 검증 하네스)**를 같은 방식(스키마/도구/코드/워크플로우)로 바로 이어서 구현해줄게.
