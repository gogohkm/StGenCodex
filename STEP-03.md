좋아. 이제 3단계: “도면(CAD) ↔ 해석모델(FE) 자동 매핑 + 패널에서 확정” 을 (1) DB 설계 → (2) 매칭 알고리즘/도구 → (3) VS Code 우측 패널 UI 흐름/구현 순서로 바로 구현 구상(=코드까지)해줄게.

참고: 지난 단계에서 내가 제안했던 doc_chunks_fts에 uri/artifact_kind 같은 “추가 컬럼”을 넣는 방식은 SQLite FTS5 외부 컨텐츠 모드(content='doc_chunks')에서 오류가 날 수 있어. 이번 단계에서 FTS 정의를 안전한 형태로 수정해서 같이 반영할게(아래 schema/server 코드에 포함).

1) DB 스키마: “해석모델/부재” + “매핑” 테이블 추가
왜 이런 테이블이 필요한가

CAD 도면에는 보통 라벨 텍스트(B12, C1 등) 가 있고,

해석모델에는 요소/부재 ID(uid) 와 라벨(label), 좌표(선분/절점), 단면(section), 층(story) 등이 있음

자동 매핑은 결국
(CAD token 후보들) → (모델 부재 후보들) 를 점수화해 “추천”하고
사람이 패널에서 “확정/수정”하는 흐름이 가장 안정적임.

1-1) schema.sql 교체(버전 0.0.4)

아래 파일로 mcp_server/schema.sql 전체 교체해줘.

-- mcp_server/schema.sql
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version','0.0.4');

-- =========================
-- Artifacts (pdf/md/dxf/etc)
-- =========================
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  uri TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,             -- 'pdf'|'md'|'dxf'|'model'|...
  title TEXT,
  source_path TEXT,
  sha256 TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- =========================
-- Documents chunks + FTS
-- =========================
CREATE TABLE IF NOT EXISTS doc_chunks (
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  page_start INTEGER,
  page_end INTEGER,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- NOTE: FTS5 external content 테이블은 content table(doc_chunks)의 컬럼과 맞아야 안정적임.
CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts
USING fts5(
  content,
  content='doc_chunks',
  content_rowid='chunk_id',
  tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS doc_chunks_ai AFTER INSERT ON doc_chunks BEGIN
  INSERT INTO doc_chunks_fts(rowid, content) VALUES (new.chunk_id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS doc_chunks_ad AFTER DELETE ON doc_chunks BEGIN
  INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content)
  VALUES('delete', old.chunk_id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS doc_chunks_au AFTER UPDATE ON doc_chunks BEGIN
  INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content)
  VALUES('delete', old.chunk_id, old.content);
  INSERT INTO doc_chunks_fts(rowid, content)
  VALUES (new.chunk_id, new.content);
END;

CREATE INDEX IF NOT EXISTS idx_doc_chunks_artifact ON doc_chunks(artifact_id, chunk_index);

-- =========================
-- CAD entities (DXF)
-- =========================
CREATE TABLE IF NOT EXISTS cad_entities (
  cad_entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  handle TEXT,
  type TEXT NOT NULL,             -- 'LINE'|'LWPOLYLINE'|'TEXT'|'MTEXT'...
  layer TEXT,
  text TEXT,
  x REAL,
  y REAL,
  z REAL,
  geom_json TEXT,                 -- e.g. {"points":[[x,y],[x,y],...]}
  layout TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_entities_artifact ON cad_entities(artifact_id, type);
CREATE INDEX IF NOT EXISTS idx_cad_entities_text ON cad_entities(artifact_id, type, text);

-- =========================
-- Structural model + members
-- =========================
CREATE TABLE IF NOT EXISTS models (
  model_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  source_path TEXT,
  units TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 해석모델의 부재/요소(beam/column 등)
CREATE TABLE IF NOT EXISTS model_members (
  model_member_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,

  member_uid TEXT NOT NULL,       -- 해석프로그램의 고유 ID(중복 X 권장)
  member_label TEXT,              -- 도면 라벨(B12, C1 등). 없으면 NULL
  label_norm TEXT,                -- 정규화 라벨(대문자 + 공백/하이픈 제거)
  type TEXT,                      -- 'beam'|'column'|'brace'|'wall'|'slab'|'unknown'

  x1 REAL, y1 REAL, z1 REAL,
  x2 REAL, y2 REAL, z2 REAL,

  section TEXT,
  story TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',

  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),

  UNIQUE(model_id, member_uid)
);

CREATE INDEX IF NOT EXISTS idx_model_members_label ON model_members(model_id, label_norm);
CREATE INDEX IF NOT EXISTS idx_model_members_type ON model_members(model_id, type);
CREATE INDEX IF NOT EXISTS idx_model_members_story ON model_members(model_id, story);

-- =========================
-- CAD token ↔ model member mapping
-- =========================
CREATE TABLE IF NOT EXISTS member_mappings (
  mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,

  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  cad_token TEXT NOT NULL,
  cad_token_norm TEXT NOT NULL,

  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,

  confidence REAL NOT NULL,         -- 0~1
  method TEXT NOT NULL,             -- 'label_exact'|'label_fuzzy'|'spatial'|'hybrid'|'manual'
  status TEXT NOT NULL DEFAULT 'suggested', -- suggested|confirmed|rejected
  evidence_json TEXT NOT NULL DEFAULT '{}',

  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),

  UNIQUE(cad_artifact_id, cad_token_norm, model_id, model_member_id)
);

CREATE INDEX IF NOT EXISTS idx_member_mappings_lookup
ON member_mappings(cad_artifact_id, model_id, status, cad_token_norm);

2) 매칭 구현: “라벨 기반 + 좌표 기반 + 레이어/타입 힌트” 하이브리드
2-1) 입력(매칭에 필요한 최소 데이터)

CAD 쪽: (token, occurrences[x,y,layer])
→ cad_entities에서 TEXT/MTEXT를 추출해 토큰화

모델 쪽: (member_uid, member_label, type, 좌표(시작/끝), section, story)
→ model_members

2-2) 점수화(Confidence) 구성

MVP에서 가장 잘 먹히는 조합:

Label Exact

normalize(token) == member.label_norm

가중치 가장 큼 (예: 0.95 근처)

Label Fuzzy (옵션)

token_norm이 member.label_norm에 포함되거나 반대(부분 문자열)

가중치 낮음 (예: 0.6~0.75)

Spatial (좌표 근접)

CAD 텍스트 위치(x,y) ↔ 모델 부재의 midpoint(xmid, ymid) 거리

dist <= tolerance면 점수 상승

동일 라벨이 여러 개(층/분절 요소) 있을 때 유용

Type Hint

토큰 prefix(C/B/BR/W 등) + 레이어 이름(“COL”, “BEAM” 등)을 힌트로

모델 member.type과 불일치하면 confidence 감점

2-3) MCP 서버 코드 업데이트(모델 임포트 + 매핑 추천/저장)

아래 내용으로 mcp_server/server.py 전체 교체해줘.

# mcp_server/server.py
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("structai")

DB_PATH = os.path.join(os.path.dirname(__file__), "structai.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


# -----------------------
# DB helpers
# -----------------------
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _connect()
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


_init_db()


# -----------------------
# Normalization & scoring
# -----------------------
LABEL_NORM_RX = re.compile(r"[\s\-_]+", re.UNICODE)


def normalize_label(s: str) -> str:
    s = (s or "").strip().upper()
    s = LABEL_NORM_RX.sub("", s)
    s = s.replace("(", "").replace(")", "")
    return s


def dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def midpoint(m: sqlite3.Row) -> Optional[Tuple[float, float]]:
    x1, y1, x2, y2 = m["x1"], m["y1"], m["x2"], m["y2"]
    if x1 is None or y1 is None:
        return None
    if x2 is None or y2 is None:
        return (float(x1), float(y1))
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def guess_type_from_token(token: str) -> Optional[str]:
    t = (token or "").strip().upper()
    if re.match(r"^C\d+", t):
        return "column"
    if re.match(r"^B\d+", t):
        return "beam"
    if re.match(r"^BR\d+", t):
        return "brace"
    if re.match(r"^W\d+", t):
        return "wall"
    if re.match(r"^S\d+", t):
        return "slab"
    return None


def guess_type_from_layer(layer: Optional[str], rules: Optional[Dict[str, str]] = None) -> Optional[str]:
    if not layer:
        return None
    u = layer.upper()

    # 사용자 규칙 우선(부분문자열 매칭)
    if rules:
        for k, v in rules.items():
            if k.upper() in u:
                return v

    if "COL" in u or "COLUMN" in u:
        return "column"
    if "BEAM" in u or "BM" in u or "GIRDER" in u:
        return "beam"
    if "BRACE" in u or "BR" in u:
        return "brace"
    if "WALL" in u:
        return "wall"
    if "SLAB" in u:
        return "slab"
    return None


@dataclass
class CadOccurrence:
    cad_entity_id: int
    x: float
    y: float
    layer: Optional[str]


def score_mapping(
    token_norm: str,
    token_type: Optional[str],
    occs: List[CadOccurrence],
    member: sqlite3.Row,
    spatial_tolerance: float,
) -> Tuple[float, Dict[str, Any], str]:
    """
    Returns: confidence, evidence, method
    """
    member_label_norm = member["label_norm"] or ""
    member_type = member["type"] or "unknown"

    # label score
    label_score = 0.0
    label_method = None
    if token_norm and member_label_norm and token_norm == member_label_norm:
        label_score = 1.0
        label_method = "label_exact"
    elif token_norm and member_label_norm and (token_norm in member_label_norm or member_label_norm in token_norm):
        label_score = 0.7
        label_method = "label_fuzzy"

    # spatial score
    m_mid = midpoint(member)
    min_d = None
    spatial_score = 0.0
    if m_mid and occs:
        ds = [dist2d((o.x, o.y), m_mid) for o in occs if o.x is not None and o.y is not None]
        if ds:
            min_d = min(ds)
            if spatial_tolerance and min_d <= spatial_tolerance:
                spatial_score = max(0.0, 1.0 - (min_d / spatial_tolerance))
            else:
                spatial_score = 0.0

    # type score
    type_score = 0.7  # unknown default
    if token_type and member_type and member_type != "unknown":
        type_score = 1.0 if token_type == member_type else 0.25

    # Combine
    # - label이 강하면 label 중심
    # - label이 약하면 spatial/type 반영
    if label_score >= 1.0:
        confidence = 0.75 * label_score + 0.20 * spatial_score + 0.05 * type_score
        method = "hybrid" if spatial_score > 0 else (label_method or "label_exact")
    elif label_score > 0:
        confidence = 0.55 * label_score + 0.35 * spatial_score + 0.10 * type_score
        method = "hybrid" if spatial_score > 0 else (label_method or "label_fuzzy")
    else:
        confidence = 0.15 * label_score + 0.70 * spatial_score + 0.15 * type_score
        method = "spatial" if spatial_score > 0 else "none"

    evidence = {
        "token_norm": token_norm,
        "member_label_norm": member_label_norm,
        "token_type": token_type,
        "member_type": member_type,
        "min_distance": min_d,
        "spatial_tolerance": spatial_tolerance,
        "label_score": label_score,
        "spatial_score": spatial_score,
        "type_score": type_score,
    }
    if label_method:
        evidence["label_method"] = label_method

    return float(max(0.0, min(1.0, confidence))), evidence, method


# -----------------------
# Existing: basic ops
# -----------------------
@mcp.tool()
def structai_reset_all() -> Dict[str, Any]:
    """(DEV) 모든 테이블 데이터 삭제"""
    conn = _connect()
    try:
        conn.executescript(
            """
            DELETE FROM member_mappings;
            DELETE FROM model_members;
            DELETE FROM models;
            DELETE FROM cad_entities;
            DELETE FROM doc_chunks;
            DELETE FROM artifacts;
            VACUUM;
            """
        )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@mcp.tool()
def structai_list_artifacts(kind: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """아티팩트 목록(문서, 도면, 모델 등)"""
    conn = _connect()
    try:
        if kind:
            rows = conn.execute(
                "SELECT artifact_id, uri, kind, title, source_path, created_at FROM artifacts WHERE kind=? ORDER BY artifact_id DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT artifact_id, uri, kind, title, source_path, created_at FROM artifacts ORDER BY artifact_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {"items": [dict(r) for r in rows]}
    finally:
        conn.close()


@mcp.tool()
def structai_search(query: str, limit: int = 8) -> Dict[str, Any]:
    """FTS로 문서 chunk 검색(pdf/md 등)"""
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT
              dc.chunk_id,
              a.uri,
              a.kind,
              a.title,
              dc.page_start,
              dc.page_end,
              snippet(doc_chunks_fts, 0, '[', ']', '…', 12) AS snippet,
              bm25(doc_chunks_fts) AS score
            FROM doc_chunks_fts
            JOIN doc_chunks dc ON dc.chunk_id = doc_chunks_fts.rowid
            JOIN artifacts a ON a.artifact_id = dc.artifact_id
            WHERE doc_chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return {"results": [dict(r) for r in rows]}
    finally:
        conn.close()


# -----------------------
# CAD: DXF import + token candidates
# -----------------------
@mcp.tool()
def structai_import_dxf(path: str, title: Optional[str] = None) -> Dict[str, Any]:
    """
    DXF를 읽어 TEXT/MTEXT/LINE/LWPOLYLINE 일부를 cad_entities로 저장
    """
    import ezdxf  # type: ignore

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    abs_path = os.path.abspath(path)
    sha = _sha256_file(abs_path)

    conn = _connect()
    try:
        # artifacts upsert
        cur = conn.execute(
            """
            INSERT INTO artifacts(uri, kind, title, source_path, sha256)
            VALUES(?, 'dxf', ?, ?, ?)
            ON CONFLICT(uri) DO UPDATE SET
              title=excluded.title,
              source_path=excluded.source_path,
              sha256=excluded.sha256
            """,
            (f"file://{abs_path}", title or os.path.basename(abs_path), abs_path, sha),
        )
        # sqlite3: INSERT..ON CONFLICT doesn't set lastrowid if update happened
        row = conn.execute("SELECT artifact_id FROM artifacts WHERE uri=?", (f"file://{abs_path}",)).fetchone()
        artifact_id = int(row["artifact_id"])

        # clean old
        conn.execute("DELETE FROM cad_entities WHERE artifact_id=?", (artifact_id,))

        doc = ezdxf.readfile(abs_path)
        msp = doc.modelspace()

        def insert_entity(handle, etype, layer, text=None, x=None, y=None, z=None, geom_json=None, layout="Model"):
            conn.execute(
                """
                INSERT INTO cad_entities(artifact_id, handle, type, layer, text, x, y, z, geom_json, layout)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (artifact_id, handle, etype, layer, text, x, y, z, geom_json, layout),
            )

        n = 0
        for e in msp:
            etype = e.dxftype()
            layer = getattr(e.dxf, "layer", None)
            handle = getattr(e.dxf, "handle", None)

            if etype in ("TEXT", "MTEXT"):
                try:
                    txt = e.plain_text() if etype == "MTEXT" else e.dxf.text
                except Exception:
                    txt = None
                try:
                    ins = e.dxf.insert if etype == "TEXT" else e.dxf.insert
                    x, y, z = float(ins.x), float(ins.y), float(ins.z)
                except Exception:
                    x = y = z = None
                insert_entity(handle, etype, layer, text=txt, x=x, y=y, z=z)
                n += 1

            elif etype == "LINE":
                try:
                    s = e.dxf.start
                    t = e.dxf.end
                    pts = [[float(s.x), float(s.y)], [float(t.x), float(t.y)]]
                    insert_entity(handle, etype, layer, geom_json=json.dumps({"points": pts}), x=float(s.x), y=float(s.y), z=float(s.z))
                    n += 1
                except Exception:
                    pass

            elif etype == "LWPOLYLINE":
                try:
                    pts = [[float(p[0]), float(p[1])] for p in e.get_points()]
                    x = pts[0][0] if pts else None
                    y = pts[0][1] if pts else None
                    insert_entity(handle, etype, layer, geom_json=json.dumps({"points": pts}), x=x, y=y, z=None)
                    n += 1
                except Exception:
                    pass

        conn.commit()
        return {"ok": True, "artifact_id": artifact_id, "entities_inserted": n, "uri": f"file://{abs_path}"}
    finally:
        conn.close()


@mcp.tool()
def structai_cad_extract_member_candidates(
    cad_artifact_id: int,
    token_regex: str = r"\b(?:C|B|BR|W|S)\s?-?\s?\d+\b",
    max_tokens: int = 200,
    min_occurrences: int = 1,
) -> Dict[str, Any]:
    """
    cad_entities(TEXT/MTEXT)에서 부재 라벨로 보이는 토큰을 후보로 추출
    """
    rx = re.compile(token_regex, re.IGNORECASE)
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT cad_entity_id, text, layer, x, y
            FROM cad_entities
            WHERE artifact_id=? AND type IN ('TEXT','MTEXT') AND text IS NOT NULL
            """,
            (cad_artifact_id,),
        ).fetchall()

        buckets: Dict[str, List[CadOccurrence]] = {}
        for r in rows:
            txt = r["text"] or ""
            for m in rx.finditer(txt):
                token = m.group(0)
                token_norm = normalize_label(token)
                if not token_norm:
                    continue
                occ = CadOccurrence(
                    cad_entity_id=int(r["cad_entity_id"]),
                    x=float(r["x"]) if r["x"] is not None else None,
                    y=float(r["y"]) if r["y"] is not None else None,
                    layer=r["layer"],
                )
                buckets.setdefault(token_norm, []).append(occ)

        items = []
        for token_norm, occs in buckets.items():
            if len(occs) < min_occurrences:
                continue
            items.append(
                {
                    "token": token_norm,
                    "token_norm": token_norm,
                    "occurrences": [
                        {"cad_entity_id": o.cad_entity_id, "x": o.x, "y": o.y, "layer": o.layer} for o in occs[:20]
                    ],
                    "occurrence_count": len(occs),
                    "type_guess": guess_type_from_token(token_norm),
                }
            )

        items.sort(key=lambda x: x["occurrence_count"], reverse=True)
        return {"cad_artifact_id": cad_artifact_id, "items": items[:max_tokens], "total_tokens": len(items)}
    finally:
        conn.close()


# -----------------------
# MODEL: import / list
# -----------------------
@mcp.tool()
def structai_model_import_members(
    path: str,
    model_name: Optional[str] = None,
    fmt: Optional[str] = None,
    units: Optional[str] = None,
    default_type: str = "unknown",
) -> Dict[str, Any]:
    """
    해석모델 부재 데이터를 JSON/CSV로 임포트

    JSON 형식 예:
    {
      "model_name": "ETABS Export",
      "units": "m",
      "members": [
        {"uid":"E1","label":"B12","type":"beam","x1":0,"y1":0,"z1":0,"x2":5,"y2":0,"z2":0,"section":"H400x200","story":"3F"}
      ]
    }

    CSV 헤더(권장):
    uid,label,type,x1,y1,z1,x2,y2,z2,section,story
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    abs_path = os.path.abspath(path)
    ext = os.path.splitext(abs_path)[1].lower()
    if fmt is None:
        fmt = "json" if ext in (".json",) else "csv"

    conn = _connect()
    try:
        # create model
        name = model_name or os.path.basename(abs_path)

        meta = {"source": "import", "format": fmt}
        cur = conn.execute(
            "INSERT INTO models(name, source_path, units, meta_json) VALUES(?,?,?,?)",
            (name, abs_path, units, json.dumps(meta, ensure_ascii=False)),
        )
        model_id = int(cur.lastrowid)

        imported = 0
        warnings: List[str] = []

        def insert_member(m: Dict[str, Any]) -> None:
            nonlocal imported
            uid = str(m.get("uid") or m.get("member_uid") or m.get("id") or "")
            if not uid:
                raise ValueError("member uid is required (uid/member_uid/id)")

            label = m.get("label") or m.get("member_label")
            label = str(label) if label is not None else None
            label_norm = normalize_label(label) if label else None

            mtype = str(m.get("type") or default_type or "unknown").lower()

            def to_f(v):
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            x1 = to_f(m.get("x1"))
            y1 = to_f(m.get("y1"))
            z1 = to_f(m.get("z1"))
            x2 = to_f(m.get("x2"))
            y2 = to_f(m.get("y2"))
            z2 = to_f(m.get("z2"))

            section = m.get("section")
            story = m.get("story")
            meta_json = m.get("meta") or m.get("meta_json") or {}

            conn.execute(
                """
                INSERT INTO model_members(
                  model_id, member_uid, member_label, label_norm, type,
                  x1,y1,z1,x2,y2,z2,
                  section, story, meta_json
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    model_id,
                    uid,
                    label,
                    label_norm,
                    mtype,
                    x1,
                    y1,
                    z1,
                    x2,
                    y2,
                    z2,
                    section,
                    story,
                    json.dumps(meta_json, ensure_ascii=False) if isinstance(meta_json, (dict, list)) else str(meta_json),
                ),
            )
            imported += 1

        if fmt == "json":
            with open(abs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "members" in data:
                if not units and data.get("units"):
                    conn.execute("UPDATE models SET units=? WHERE model_id=?", (data.get("units"), model_id))
                if not model_name and data.get("model_name"):
                    conn.execute("UPDATE models SET name=? WHERE model_id=?", (data.get("model_name"), model_id))
                members = data["members"]
            elif isinstance(data, list):
                members = data
            else:
                raise ValueError("JSON must be an object with 'members' or a list of members")
            for m in members:
                insert_member(m)

        elif fmt == "csv":
            with open(abs_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    insert_member(
                        {
                            "uid": row.get("uid"),
                            "label": row.get("label"),
                            "type": row.get("type") or default_type,
                            "x1": row.get("x1"),
                            "y1": row.get("y1"),
                            "z1": row.get("z1"),
                            "x2": row.get("x2"),
                            "y2": row.get("y2"),
                            "z2": row.get("z2"),
                            "section": row.get("section"),
                            "story": row.get("story"),
                            "meta_json": {},
                        }
                    )
        else:
            raise ValueError(f"unsupported fmt: {fmt}")

        conn.commit()
        return {"ok": True, "model_id": model_id, "name": name, "imported": imported, "warnings": warnings}
    finally:
        conn.close()


@mcp.tool()
def structai_model_list(limit: int = 50) -> Dict[str, Any]:
    """모델 목록"""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT model_id, name, source_path, units, created_at FROM models ORDER BY model_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {"items": [dict(r) for r in rows]}
    finally:
        conn.close()


@mcp.tool()
def structai_model_list_members(model_id: int, contains: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """모델 부재 목록(필터: label/uid 포함 검색)"""
    conn = _connect()
    try:
        if contains:
            q = f"%{contains.upper()}%"
            rows = conn.execute(
                """
                SELECT model_member_id, member_uid, member_label, type, section, story, x1,y1,z1,x2,y2,z2
                FROM model_members
                WHERE model_id=? AND (UPPER(member_uid) LIKE ? OR UPPER(COALESCE(member_label,'')) LIKE ?)
                ORDER BY model_member_id DESC
                LIMIT ?
                """,
                (model_id, q, q, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT model_member_id, member_uid, member_label, type, section, story, x1,y1,z1,x2,y2,z2
                FROM model_members
                WHERE model_id=?
                ORDER BY model_member_id DESC
                LIMIT ?
                """,
                (model_id, limit),
            ).fetchall()
        return {"model_id": model_id, "items": [dict(r) for r in rows]}
    finally:
        conn.close()


# -----------------------
# MAPPING: suggest / save / list
# -----------------------
@mcp.tool()
def structai_map_suggest_members(
    cad_artifact_id: int,
    model_id: int,
    token_regex: str = r"\b(?:C|B|BR|W|S)\s?-?\s?\d+\b",
    max_tokens: int = 200,
    min_occurrences: int = 1,
    max_candidates_per_token: int = 5,
    spatial_tolerance: float = 5.0,
    layer_type_rules: Optional[Dict[str, str]] = None,
    enable_fuzzy: bool = True,
) -> Dict[str, Any]:
    """
    CAD 토큰 후보와 모델 부재를 매칭해 추천 리스트를 반환(저장 X)
    """
    rx = re.compile(token_regex, re.IGNORECASE)

    conn = _connect()
    try:
        # CAD tokens (grouped by norm)
        cad_rows = conn.execute(
            """
            SELECT cad_entity_id, text, layer, x, y
            FROM cad_entities
            WHERE artifact_id=? AND type IN ('TEXT','MTEXT') AND text IS NOT NULL
            """,
            (cad_artifact_id,),
        ).fetchall()

        buckets: Dict[str, List[CadOccurrence]] = {}
        for r in cad_rows:
            txt = r["text"] or ""
            for m in rx.finditer(txt):
                token = m.group(0)
                token_norm = normalize_label(token)
                if not token_norm:
                    continue
                occ = CadOccurrence(
                    cad_entity_id=int(r["cad_entity_id"]),
                    x=float(r["x"]) if r["x"] is not None else None,
                    y=float(r["y"]) if r["y"] is not None else None,
                    layer=r["layer"],
                )
                buckets.setdefault(token_norm, []).append(occ)

        # Filter/sort tokens
        token_items = [(k, v) for k, v in buckets.items() if len(v) >= min_occurrences]
        token_items.sort(key=lambda kv: len(kv[1]), reverse=True)
        token_items = token_items[:max_tokens]

        # Model members
        members = conn.execute(
            """
            SELECT model_member_id, member_uid, member_label, label_norm, type,
                   x1,y1,z1,x2,y2,z2, section, story
            FROM model_members
            WHERE model_id=?
            """,
            (model_id,),
        ).fetchall()

        # Index by label_norm
        by_label: Dict[str, List[sqlite3.Row]] = {}
        for m in members:
            ln = m["label_norm"]
            if ln:
                by_label.setdefault(str(ln), []).append(m)

        # fallback list for fuzzy scan
        members_with_label = [m for m in members if m["label_norm"]]

        out_items = []
        unmatched = []

        for token_norm, occs in token_items:
            # type guess: token prefix + layer hints majority vote
            token_type = guess_type_from_token(token_norm)
            if token_type is None:
                # layer majority
                layer_votes: Dict[str, int] = {}
                for o in occs:
                    t = guess_type_from_layer(o.layer, layer_type_rules)
                    if t:
                        layer_votes[t] = layer_votes.get(t, 0) + 1
                if layer_votes:
                    token_type = sorted(layer_votes.items(), key=lambda kv: kv[1], reverse=True)[0][0]

            candidates: List[Dict[str, Any]] = []

            # 1) exact label candidates
            exact_members = by_label.get(token_norm, [])
            for m in exact_members:
                conf, ev, method = score_mapping(token_norm, token_type, occs, m, spatial_tolerance)
                candidates.append(
                    {
                        "model_member_id": int(m["model_member_id"]),
                        "member_uid": m["member_uid"],
                        "member_label": m["member_label"],
                        "type": m["type"],
                        "section": m["section"],
                        "story": m["story"],
                        "confidence": conf,
                        "method": method if method != "none" else "label_exact",
                        "evidence": ev,
                    }
                )

            # 2) fuzzy label candidates (optional)
            if not candidates and enable_fuzzy:
                fuzzy_hits = []
                for m in members_with_label:
                    ln = m["label_norm"]
                    if not ln:
                        continue
                    if token_norm in ln or ln in token_norm:
                        fuzzy_hits.append(m)
                # score and keep top
                for m in fuzzy_hits[:2000]:
                    conf, ev, method = score_mapping(token_norm, token_type, occs, m, spatial_tolerance)
                    if conf > 0.1:
                        candidates.append(
                            {
                                "model_member_id": int(m["model_member_id"]),
                                "member_uid": m["member_uid"],
                                "member_label": m["member_label"],
                                "type": m["type"],
                                "section": m["section"],
                                "story": m["story"],
                                "confidence": conf,
                                "method": method if method != "none" else "label_fuzzy",
                                "evidence": ev,
                            }
                        )

            # 3) spatial-only fallback (same type만 우선)
            if not candidates:
                for m in members:
                    # type filter (약하게)
                    if token_type and (m["type"] or "unknown") not in ("unknown", token_type):
                        continue
                    conf, ev, method = score_mapping(token_norm, token_type, occs, m, spatial_tolerance)
                    if conf > 0.15 and method in ("spatial", "hybrid"):
                        candidates.append(
                            {
                                "model_member_id": int(m["model_member_id"]),
                                "member_uid": m["member_uid"],
                                "member_label": m["member_label"],
                                "type": m["type"],
                                "section": m["section"],
                                "story": m["story"],
                                "confidence": conf,
                                "method": method,
                                "evidence": ev,
                            }
                        )

            candidates.sort(key=lambda c: c["confidence"], reverse=True)
            candidates = candidates[:max_candidates_per_token]

            if not candidates:
                unmatched.append({"token": token_norm, "occurrence_count": len(occs), "type_guess": token_type})
                continue

            out_items.append(
                {
                    "token": token_norm,
                    "token_norm": token_norm,
                    "type_guess": token_type,
                    "occurrence_count": len(occs),
                    "occurrences": [
                        {"cad_entity_id": o.cad_entity_id, "x": o.x, "y": o.y, "layer": o.layer} for o in occs[:20]
                    ],
                    "candidates": candidates,
                }
            )

        return {
            "cad_artifact_id": cad_artifact_id,
            "model_id": model_id,
            "items": out_items,
            "unmatched_tokens": unmatched,
        }
    finally:
        conn.close()


@mcp.tool()
def structai_map_save_mappings(mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    매핑을 DB에 저장(상태 포함). mappings 각 원소 예:
    {
      "cad_artifact_id": 1,
      "cad_token": "B12",
      "model_id": 2,
      "model_member_id": 101,
      "confidence": 0.93,
      "method": "hybrid",
      "status": "confirmed",
      "evidence": {...}
    }
    """
    conn = _connect()
    try:
        saved = 0
        for m in mappings:
            cad_artifact_id = int(m["cad_artifact_id"])
            cad_token = str(m["cad_token"])
            cad_token_norm = normalize_label(cad_token)
            model_id = int(m["model_id"])
            model_member_id = int(m["model_member_id"])
            confidence = float(m.get("confidence", 0.5))
            method = str(m.get("method", "manual"))
            status = str(m.get("status", "confirmed"))
            evidence = m.get("evidence", {}) or {}

            conn.execute(
                """
                INSERT INTO member_mappings(
                  cad_artifact_id, cad_token, cad_token_norm,
                  model_id, model_member_id,
                  confidence, method, status, evidence_json
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(cad_artifact_id, cad_token_norm, model_id, model_member_id)
                DO UPDATE SET
                  confidence=excluded.confidence,
                  method=excluded.method,
                  status=excluded.status,
                  evidence_json=excluded.evidence_json,
                  updated_at=datetime('now')
                """,
                (
                    cad_artifact_id,
                    cad_token,
                    cad_token_norm,
                    model_id,
                    model_member_id,
                    confidence,
                    method,
                    status,
                    json.dumps(evidence, ensure_ascii=False),
                ),
            )
            saved += 1

        conn.commit()
        return {"ok": True, "saved": saved}
    finally:
        conn.close()


@mcp.tool()
def structai_map_list_mappings(
    cad_artifact_id: Optional[int] = None,
    model_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    """저장된 매핑 조회"""
    conn = _connect()
    try:
        where = []
        params: List[Any] = []
        if cad_artifact_id is not None:
            where.append("mm.cad_artifact_id=?")
            params.append(int(cad_artifact_id))
        if model_id is not None:
            where.append("mm.model_id=?")
            params.append(int(model_id))
        if status is not None:
            where.append("mm.status=?")
            params.append(str(status))

        wsql = ("WHERE " + " AND ".join(where)) if where else ""
        rows = conn.execute(
            f"""
            SELECT
              mm.mapping_id,
              mm.cad_artifact_id,
              mm.cad_token,
              mm.cad_token_norm,
              mm.model_id,
              mm.model_member_id,
              mm.confidence,
              mm.method,
              mm.status,
              mm.evidence_json,
              mb.member_uid,
              mb.member_label,
              mb.type AS member_type,
              mb.section,
              mb.story
            FROM member_mappings mm
            JOIN model_members mb ON mb.model_member_id = mm.model_member_id
            {wsql}
            ORDER BY mm.updated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return {"items": [dict(r) for r in rows]}
    finally:
        conn.close()

3) VS Code 우측 패널(UI): “Auto-map → 수정 → Confirm”

여기서 목표는 Cursor처럼:

오른쪽 패널에서

DXF 선택 + 모델 선택

Auto-map 클릭 → 추천 리스트

임계치 이상만 일괄 확정 / 개별 드롭다운 수정 / 확정

결과는 DB(member_mappings)에 저장

이를 위해 extension에서 **MCP tool을 “프로그램적으로 호출”**해야 하는데, VS Code API에 vscode.lm.invokeTool이 있어서 가능해. 또한 lm.tools 목록에 있는 tool을 name으로 호출하는 형태야. 
Visual Studio Code
+1

3-1) mappingView.ts 업데이트 (웹뷰 ↔ MCP tool 브릿지)

src/mappingView.ts를 아래처럼 수정(핵심만)해줘.

기존 파일 구조가 다를 수 있으니, message handler + callTool helper만 동일하게 넣으면 돼.

// src/mappingView.ts
import * as vscode from "vscode";
import * as path from "path";

export class MappingViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "structai.mappingView";
  private _view?: vscode.WebviewView;

  constructor(private readonly _context: vscode.ExtensionContext) {}

  resolveWebviewView(webviewView: vscode.WebviewView) {
    this._view = webviewView;
    const webview = webviewView.webview;

    webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this._context.extensionUri, "media")],
    };

    webview.html = this._getHtml(webview);

    webview.onDidReceiveMessage(async (msg) => {
      try {
        switch (msg.type) {
          case "init":
            await this._sendState();
            break;

          case "refresh":
            await this._sendState();
            break;

          case "importModel": {
            const picked = await vscode.window.showOpenDialog({
              canSelectMany: false,
              openLabel: "Import model (CSV/JSON)",
              filters: { "Model Export": ["csv", "json"] },
            });
            if (!picked || picked.length === 0) return;
            const filePath = picked[0].fsPath;
            const result = await this._invokeStructAi("structai_model_import_members", {
              path: filePath,
            });
            webview.postMessage({ type: "modelImported", result });
            await this._sendState();
            break;
          }

          case "suggestMappings": {
            const result = await this._invokeStructAi("structai_map_suggest_members", msg.input);
            webview.postMessage({ type: "suggestions", result });
            break;
          }

          case "saveMappings": {
            const result = await this._invokeStructAi("structai_map_save_mappings", { mappings: msg.mappings });
            webview.postMessage({ type: "saved", result });
            await this._sendState();
            break;
          }

          default:
            break;
        }
      } catch (e: any) {
        webview.postMessage({ type: "error", message: String(e?.message ?? e) });
      }
    });
  }

  private async _sendState() {
    if (!this._view) return;
    const webview = this._view.webview;

    const artifacts = await this._invokeStructAi("structai_list_artifacts", {});
    const models = await this._invokeStructAi("structai_model_list", {});
    const mappings = await this._invokeStructAi("structai_map_list_mappings", { limit: 200 });

    webview.postMessage({
      type: "state",
      artifacts,
      models,
      mappings,
    });
  }

  /**
   * MCP tool을 VS Code API로 호출
   * - tool name은 vscode.lm.tools에 등록된 실제 name을 찾아서 invokeTool
   * - invokeTool은 extension 어디서든 호출 가능(채팅 문맥 없어도 됨) :contentReference[oaicite:1]{index=1}
   */
  private async _invokeStructAi(toolSuffix: string, input: any): Promise<any> {
    // 주의: MCP tool name이 'mcp.xxx.structai_map_suggest_members' 처럼 prefix가 붙을 수 있어서 includes로 찾음
    const toolInfo = vscode.lm.tools.find((t) => t.name.includes(toolSuffix));
    if (!toolInfo) {
      throw new Error(
        `StructAI MCP tool not found: ${toolSuffix}. ` +
          `팁: Chat을 한 번 열고(StructAI participant로 아무 메시지) 도구 discovery가 된 뒤 다시 시도하세요.`
      );
    }

    const result: any = await vscode.lm.invokeTool(toolInfo.name, { input });
    return this._unwrapToolResult(result);
  }

  /**
   * LanguageModelToolResult(파트 배열)를 최대한 JSON로 복원
   */
  private _unwrapToolResult(result: any): any {
    // MCP 툴 결과가 이미 오브젝트로 들어오는 경우
    if (result && typeof result === "object" && !Array.isArray(result)) {
      // 일부 구현체는 { content: [...] } 형태일 수 있음
      if (Array.isArray((result as any).content)) {
        return this._unwrapToolResult((result as any).content);
      }
    }

    if (Array.isArray(result)) {
      const text = result
        .map((p) => {
          if (typeof p === "string") return p;
          if (p && typeof p === "object") {
            // 흔한 케이스: { value: "..." }
            if (typeof (p as any).value === "string") return (p as any).value;
            if (typeof (p as any).text === "string") return (p as any).text;
            if (typeof (p as any).content === "string") return (p as any).content;
            return JSON.stringify(p);
          }
          return String(p);
        })
        .join("");

      // JSON이면 파싱
      try {
        return JSON.parse(text);
      } catch {
        return { raw: text };
      }
    }

    if (typeof result === "string") {
      try {
        return JSON.parse(result);
      } catch {
        return { raw: result };
      }
    }

    return result;
  }

  private _getHtml(webview: vscode.Webview): string {
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, "media", "mapping", "main.js")
    );

    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, "media", "mapping", "styles.css")
    );

    return /* html */ `
      <!doctype html>
      <html>
        <head>
          <meta charset="UTF-8" />
          <meta http-equiv="Content-Security-Policy"
                content="default-src 'none'; style-src ${webview.cspSource}; script-src ${webview.cspSource};" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <link href="${styleUri}" rel="stylesheet" />
          <title>StructAI Mapping</title>
        </head>
        <body>
          <div id="app"></div>
          <script src="${scriptUri}"></script>
        </body>
      </html>
    `;
  }
}

3-2) 웹뷰 JS: media/mapping/main.js (Auto-map UI)

media/mapping/main.js를 아래로 교체해줘.

// media/mapping/main.js
(function () {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("app");

  let state = {
    artifacts: { items: [] },
    models: { items: [] },
    mappings: { items: [] },
  };

  let suggestions = null;

  function h(tag, attrs = {}, children = []) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") el.className = v;
      else if (k === "onclick") el.onclick = v;
      else if (k === "onchange") el.onchange = v;
      else el.setAttribute(k, v);
    }
    for (const c of children) {
      if (typeof c === "string") el.appendChild(document.createTextNode(c));
      else if (c) el.appendChild(c);
    }
    return el;
  }

  function render() {
    app.innerHTML = "";

    const dxfArtifacts = (state.artifacts.items || []).filter((a) => a.kind === "dxf");
    const models = state.models.items || [];

    const cadSelect = h(
      "select",
      { id: "cadSelect" },
      dxfArtifacts.map((a) => h("option", { value: String(a.artifact_id) }, [`${a.title || a.uri}`]))
    );

    const modelSelect = h(
      "select",
      { id: "modelSelect" },
      models.map((m) => h("option", { value: String(m.model_id) }, [`${m.name}`]))
    );

    const thresholdInput = h("input", { id: "threshold", type: "number", step: "0.05", min: "0", max: "1", value: "0.85" });

    const btnRow = h("div", { class: "row" }, [
      h("button", { onclick: () => vscode.postMessage({ type: "refresh" }) }, ["Refresh"]),
      h("button", { onclick: () => vscode.postMessage({ type: "importModel" }) }, ["Import Model (CSV/JSON)"]),
    ]);

    const actionRow = h("div", { class: "row" }, [
      h("label", {}, ["CAD(DXF): "]),
      cadSelect,
      h("label", { style: "margin-left:10px" }, ["Model: "]),
      modelSelect,
    ]);

    const suggestRow = h("div", { class: "row" }, [
      h("label", {}, ["Auto-confirm threshold: "]),
      thresholdInput,
      h(
        "button",
        {
          style: "margin-left:10px",
          onclick: () => {
            const cadId = Number(document.getElementById("cadSelect").value);
            const modelId = Number(document.getElementById("modelSelect").value);
            vscode.postMessage({
              type: "suggestMappings",
              input: {
                cad_artifact_id: cadId,
                model_id: modelId,
                max_tokens: 200,
                max_candidates_per_token: 5,
                spatial_tolerance: 5.0,
                enable_fuzzy: true,
              },
            });
          },
        },
        ["Auto-map"]
      ),
      h(
        "button",
        {
          style: "margin-left:10px",
          onclick: () => saveSelected(),
        },
        ["Confirm selected"]
      ),
    ]);

    app.appendChild(h("h2", {}, ["StructAI Mapping"]));
    app.appendChild(btnRow);
    app.appendChild(actionRow);
    app.appendChild(suggestRow);

    app.appendChild(renderMappings());
    app.appendChild(renderSuggestions());
  }

  function renderMappings() {
    const items = state.mappings.items || [];
    const container = h("div", { class: "section" }, [
      h("h3", {}, [`Saved mappings (${items.length})`]),
    ]);

    if (items.length === 0) {
      container.appendChild(h("div", { class: "muted" }, ["No mappings yet."]));
      return container;
    }

    const table = h("table", { class: "table" }, []);
    table.appendChild(h("tr", {}, [
      h("th", {}, ["Token"]),
      h("th", {}, ["Member"]),
      h("th", {}, ["Type"]),
      h("th", {}, ["Confidence"]),
      h("th", {}, ["Status"]),
      h("th", {}, ["Method"]),
    ]));

    for (const m of items.slice(0, 200)) {
      table.appendChild(
        h("tr", {}, [
          h("td", {}, [m.cad_token]),
          h("td", {}, [`${m.member_label || ""} (${m.member_uid})`]),
          h("td", {}, [m.member_type || ""]),
          h("td", {}, [String(Number(m.confidence).toFixed(2))]),
          h("td", {}, [m.status]),
          h("td", {}, [m.method]),
        ])
      );
    }

    container.appendChild(table);
    return container;
  }

  function renderSuggestions() {
    const container = h("div", { class: "section" }, [h("h3", {}, ["Suggestions"])]);

    if (!suggestions) {
      container.appendChild(h("div", { class: "muted" }, ["Run Auto-map to see suggestions."]));
      return container;
    }

    const items = suggestions.items || [];
    const unmatched = suggestions.unmatched_tokens || [];

    container.appendChild(h("div", { class: "muted" }, [
      `Suggested tokens: ${items.length}, Unmatched: ${unmatched.length}`,
    ]));

    const table = h("table", { class: "table" }, []);
    table.appendChild(h("tr", {}, [
      h("th", {}, ["Confirm"]),
      h("th", {}, ["Token"]),
      h("th", {}, ["Type guess"]),
      h("th", {}, ["Occ."]),
      h("th", {}, ["Pick member"]),
      h("th", {}, ["Confidence"]),
      h("th", {}, ["Method"]),
    ]));

    const threshold = Number(document.getElementById("threshold")?.value || "0.85");

    for (const it of items) {
      const token = it.token;
      const cand = it.candidates || [];
      const top = cand[0];

      const select = h(
        "select",
        { class: "candSelect", "data-token": token },
        cand.map((c) =>
          h("option", { value: String(c.model_member_id) }, [
            `${c.member_label || ""} (${c.member_uid}) [${Number(c.confidence).toFixed(2)}]`,
          ])
        )
      );

      const chk = h("input", {
        type: "checkbox",
        class: "confirmChk",
        "data-token": token,
        checked: top && Number(top.confidence) >= threshold ? "checked" : null,
      });

      table.appendChild(
        h("tr", {}, [
          h("td", {}, [chk]),
          h("td", {}, [token]),
          h("td", {}, [it.type_guess || ""]),
          h("td", {}, [String(it.occurrence_count || 0)]),
          h("td", {}, [select]),
          h("td", {}, [top ? String(Number(top.confidence).toFixed(2)) : ""]),
          h("td", {}, [top ? top.method : ""]),
        ])
      );
    }

    container.appendChild(table);

    if (unmatched.length > 0) {
      container.appendChild(h("h4", {}, ["Unmatched tokens (top 20)"]));
      const ul = h("ul", {}, unmatched.slice(0, 20).map((u) => h("li", {}, [`${u.token} (occ:${u.occurrence_count})`])));
      container.appendChild(ul);
    }

    return container;
  }

  function saveSelected() {
    if (!suggestions) return;

    const cadId = Number(document.getElementById("cadSelect").value);
    const modelId = Number(document.getElementById("modelSelect").value);

    const rows = Array.from(document.querySelectorAll(".confirmChk"));
    const selects = new Map(
      Array.from(document.querySelectorAll(".candSelect")).map((s) => [s.getAttribute("data-token"), s])
    );

    const toSave = [];

    for (const chk of rows) {
      if (!chk.checked) continue;
      const token = chk.getAttribute("data-token");
      const sel = selects.get(token);
      const memberId = Number(sel.value);

      // selected candidate의 confidence/method/evidence 찾기
      const item = (suggestions.items || []).find((i) => i.token === token);
      const cand = (item?.candidates || []).find((c) => Number(c.model_member_id) === memberId);
      toSave.push({
        cad_artifact_id: cadId,
        cad_token: token,
        model_id: modelId,
        model_member_id: memberId,
        confidence: cand?.confidence ?? 0.5,
        method: cand?.method ?? "manual",
        status: "confirmed",
        evidence: cand?.evidence ?? {},
      });
    }

    vscode.postMessage({ type: "saveMappings", mappings: toSave });
  }

  window.addEventListener("message", (event) => {
    const msg = event.data;
    if (msg.type === "state") {
      state = {
        artifacts: msg.artifacts || { items: [] },
        models: msg.models || { items: [] },
        mappings: msg.mappings || { items: [] },
      };
      render();
    } else if (msg.type === "suggestions") {
      suggestions = msg.result;
      render();
    } else if (msg.type === "saved") {
      vscode.postMessage({ type: "refresh" });
    } else if (msg.type === "error") {
      console.error(msg.message);
      alert(msg.message);
    }
  });

  vscode.postMessage({ type: "init" });
})();

3-3) styles.css (간단 스타일)

media/mapping/styles.css를 아래로 추가/교체해줘.

/* media/mapping/styles.css */
body {
  font-family: var(--vscode-font-family);
  font-size: var(--vscode-font-size);
  color: var(--vscode-foreground);
  background: var(--vscode-sideBar-background);
  padding: 10px;
}

h2, h3, h4 {
  margin: 8px 0;
}

.row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 6px 0;
  flex-wrap: wrap;
}

.section {
  margin-top: 12px;
  padding-top: 8px;
  border-top: 1px solid var(--vscode-sideBar-border, rgba(255,255,255,0.08));
}

.muted {
  opacity: 0.8;
  margin: 6px 0;
}

button, select, input {
  background: var(--vscode-button-background);
  color: var(--vscode-button-foreground);
  border: 1px solid var(--vscode-button-border, transparent);
  padding: 4px 8px;
  border-radius: 4px;
}

select, input[type="number"] {
  background: var(--vscode-input-background);
  color: var(--vscode-input-foreground);
  border: 1px solid var(--vscode-input-border);
}

.table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 6px;
}

.table th, .table td {
  border-bottom: 1px solid rgba(255,255,255,0.08);
  padding: 6px;
  vertical-align: top;
}

.table th {
  text-align: left;
  opacity: 0.9;
}

4) 사용 흐름(지금 단계 MVP)

DXF 임포트

채팅(StructAI participant)에서: #structai_import_dxf path="..."

해석모델 export 준비

ETABS/SAP2000/MIDAS 등에서 부재 리스트를 CSV/JSON으로 뽑아(최소 uid,label,type,x1,y1,z1,x2,y2,z2,section,story)

채팅에서: #structai_model_import_members path="..."
또는 패널에서 Import Model 버튼

오른쪽 패널(StructAI Mapping) 열기

DXF 선택 + Model 선택

Auto-map → 추천 생성

임계치(예: 0.85) 이상 자동 체크

Confirm selected → DB 저장

5) 다음 단계로 이어지는 “구조안전성 평가” 연결 포인트(미리 설계만)

이제 member_mappings가 생겼기 때문에,

대화 중에 “B12 안전성 평가해줘” 하면:

CAD token “B12” → member_mappings에서 모델 부재(들) 찾기

모델 부재의 section/story/재료/하중조합 등(추가 테이블)로 해석결과 연결

기준서(doc_chunks_fts)에서 관련 조항 검색

패널에서 “근거/조항/결과/판정”을 함께 표시

이게 Claude Skills처럼 “대화로 체크를 수행하고 근거를 남기는” 구조의 핵심 골격이야.

원하면 다음(4단계)에서 바로:

해석결과(부재력/응력/변형) 임포트 테이블

부재별 설계검토(checks) 테이블 + 판정 로직(강도/서비스/내진)

기준서 조항을 체크 로직에 연결하는 “규칙 엔진(룰셋)”
까지 이어서 같은 방식으로 순서대로 구현해줄게.
