좋아. 이제 **4단계(해석결과 임포트 → 결정론 체크 엔진 → 기준 조항 연결(룰셋))**를 “바로 붙여서 굴러가는” 수준으로 내려줄게.
이번 단계에서 목표는 아래 3개야:

해석결과(부재력/변형/드리프트 등) 임포트 테이블 + MCP 도구

부재별 설계검토(checks) 테이블 + 결정론 판정 로직(Strength/Service/Seismic)

기준서(PDF) 조항을 체크 로직에 연결하는 룰셋(rulepack) + 인용(citation) 자동 첨부

주의(정직하게): 아래에 포함된 “체크 계산” 자체는 코드(예: KDS/ACI/AISC) 완전 구현이 아니라
룰셋 기반 프레임워크 + 예시(Generic) 체크야.
실제 KDS 체크는 이후 단계에서 각 체크별 결정론 로직을 확장하면 된다.
(중요한 건 아키텍처와 감사 가능한 결과 저장 구조를 먼저 완성하는 것)

4-1) DB 스키마 업데이트 (schema.sql v0.0.5)

기존 DB를 계속 마이그레이션하려면 ALTER가 필요하지만, 지금은 개발 단계니까 가장 안전한 방법은 DB 삭제 후 재생성이야.
(워크스페이스 .structai/structai.db 또는 설정한 DB 경로)

아래로 mcp_server/schema.sql 전체 교체:

-- mcp_server/schema.sql (v0.0.5)
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version','0.0.5');
UPDATE meta SET value='0.0.5' WHERE key='schema_version';

-- =========================================================
-- Artifacts: PDF/MD/DXF/Model exports/Result exports/Rulepacks
-- =========================================================
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  uri TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,             -- 'pdf'|'md'|'dxf'|'model_export'|'results_export'|'rulepack' ...
  title TEXT,
  source_path TEXT,
  sha256 TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- =========================================================
-- Document chunks (for PDF/MD + also CAD text chunks) + FTS5
-- =========================================================
CREATE TABLE IF NOT EXISTS doc_chunks (
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  page_start INTEGER,
  page_end INTEGER,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- External content FTS (safe pattern)
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

-- =========================================================
-- CAD entities (DXF)
--  - chunk_id로 doc_chunks와 연결해서 검색/근거로 재사용 가능
-- =========================================================
CREATE TABLE IF NOT EXISTS cad_entities (
  cad_entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  chunk_id INTEGER REFERENCES doc_chunks(chunk_id) ON DELETE SET NULL,

  handle TEXT,
  type TEXT NOT NULL,             -- TEXT|MTEXT|DIMENSION|ATTRIB|...
  layer TEXT,
  layout TEXT,

  text TEXT,
  x REAL, y REAL, z REAL,

  geom_json TEXT,
  raw_json TEXT NOT NULL DEFAULT '{}',

  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_entities_artifact ON cad_entities(artifact_id, type);
CREATE INDEX IF NOT EXISTS idx_cad_entities_layer ON cad_entities(artifact_id, layer);
CREATE INDEX IF NOT EXISTS idx_cad_entities_text ON cad_entities(artifact_id, type, text);

-- =========================================================
-- Structural models + members
-- =========================================================
CREATE TABLE IF NOT EXISTS models (
  model_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  source_path TEXT,
  units TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_members (
  model_member_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,

  member_uid TEXT NOT NULL,
  member_label TEXT,
  label_norm TEXT,
  type TEXT,                      -- beam|column|brace|wall|slab|unknown

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

-- =========================================================
-- CAD token ↔ model member mapping
-- =========================================================
CREATE TABLE IF NOT EXISTS member_mappings (
  mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,

  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  cad_token TEXT NOT NULL,
  cad_token_norm TEXT NOT NULL,

  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,

  confidence REAL NOT NULL,
  method TEXT NOT NULL,           -- label_exact|fuzzy|spatial|hybrid|manual
  status TEXT NOT NULL DEFAULT 'suggested',  -- suggested|confirmed|rejected
  evidence_json TEXT NOT NULL DEFAULT '{}',

  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),

  UNIQUE(cad_artifact_id, cad_token_norm, model_id, model_member_id)
);

CREATE INDEX IF NOT EXISTS idx_member_mappings_lookup
ON member_mappings(cad_artifact_id, model_id, status, cad_token_norm);

-- =========================================================
-- Analysis results: runs + per-member envelopes (combo별)
-- =========================================================
CREATE TABLE IF NOT EXISTS analysis_runs (
  analysis_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  engine TEXT,                    -- opensees|etabs|sap2000|midas|...
  units TEXT,
  source_path TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS member_results (
  result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(analysis_run_id) ON DELETE CASCADE,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,
  combo TEXT NOT NULL,
  envelope_json TEXT NOT NULL,    -- {"N_max":..,"N_min":..,"V2_max":..,"M3_max":..,"D_max":..}
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(analysis_run_id, model_member_id, combo)
);

CREATE INDEX IF NOT EXISTS idx_member_results_lookup
ON member_results(analysis_run_id, combo, model_member_id);

-- =========================================================
-- Design inputs per member (capacities/limits/params)
-- =========================================================
CREATE TABLE IF NOT EXISTS member_design_inputs (
  model_member_id INTEGER PRIMARY KEY REFERENCES model_members(model_member_id) ON DELETE CASCADE,
  design_json TEXT NOT NULL,      -- {"Mn":..,"Vn":..,"Pn":..,"D_allow":..,"drift_allow":..}
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- =========================================================
-- Rulepacks (룰셋) - 체크 정의 + 인용쿼리 포함
-- =========================================================
CREATE TABLE IF NOT EXISTS rulepacks (
  rulepack_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  rulepack_json TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_rulepacks_namever ON rulepacks(name, version);

-- =========================================================
-- Check runs + check results (audit friendly)
-- =========================================================
CREATE TABLE IF NOT EXISTS check_runs (
  check_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(analysis_run_id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  rulepack_name TEXT,
  rulepack_version TEXT,
  combos_json TEXT NOT NULL,
  check_types_json TEXT NOT NULL,
  scope_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS check_results (
  check_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  check_run_id INTEGER NOT NULL REFERENCES check_runs(check_run_id) ON DELETE CASCADE,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,

  combo TEXT NOT NULL,
  check_type TEXT NOT NULL,

  demand_value REAL,
  capacity_value REAL,
  ratio REAL,
  status TEXT NOT NULL,             -- PASS|WARN|FAIL|NA

  details_json TEXT NOT NULL DEFAULT '{}',    -- 중간값/단위/식/파라미터
  citations_json TEXT NOT NULL DEFAULT '[]',  -- [{query, uri, page, chunk_id, snippet}]
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_check_results_run ON check_results(check_run_id, status);
CREATE INDEX IF NOT EXISTS idx_check_results_member ON check_results(model_member_id);

4-2) MCP 서버 업데이트: “결과 임포트 / 설계입력 / 룰셋 / 체크 실행”

이제 mcp_server/server.py를 아래 코드로 교체해줘.
(앞 단계의 모델/매핑 도구는 유지 + 결과/체크 관련 도구 추가)

requirements.txt (최소)

mcp>=1.0.0
pypdf>=5.0.0
ezdxf>=1.4.0

mcp_server/server.py (통합본)
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("structai-mcp")

# -----------------------
# Paths / DB init
# -----------------------
WORKSPACE = Path(os.environ.get("STRUCTAI_WORKSPACE", os.getcwd())).resolve()
DB_PATH = Path(os.environ.get("STRUCTAI_DB_PATH", str(WORKSPACE / ".structai" / "structai.db"))).resolve()
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def _init_db() -> None:
    conn = _connect()
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()

_init_db()

# -----------------------
# Common helpers
# -----------------------
LABEL_NORM_RX = re.compile(r"[\s\-_]+", re.UNICODE)

def normalize_label(s: str) -> str:
    s = (s or "").strip().upper()
    s = LABEL_NORM_RX.sub("", s)
    s = s.replace("(", "").replace(")", "")
    return s

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _upsert_artifact(conn: sqlite3.Connection, *, uri: str, kind: str, title: str, source_path: str, sha256: str) -> int:
    conn.execute(
        """
        INSERT INTO artifacts(uri, kind, title, source_path, sha256, updated_at)
        VALUES(?,?,?,?,?, datetime('now'))
        ON CONFLICT(uri) DO UPDATE SET
          kind=excluded.kind,
          title=excluded.title,
          source_path=excluded.source_path,
          sha256=excluded.sha256,
          updated_at=datetime('now')
        """,
        (uri, kind, title, source_path, sha256),
    )
    row = conn.execute("SELECT artifact_id FROM artifacts WHERE uri=?", (uri,)).fetchone()
    assert row is not None
    return int(row["artifact_id"])

def _delete_doc_chunks_for_artifact(conn: sqlite3.Connection, artifact_id: int) -> None:
    conn.execute("DELETE FROM doc_chunks WHERE artifact_id=?", (artifact_id,))

def _chunk_text(text: str, max_chars: int = 2400, overlap: int = 200) -> List[str]:
    t = (text or "").replace("\r\n", "\n").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    out = []
    i = 0
    while i < len(t):
        j = min(i + max_chars, len(t))
        out.append(t[i:j].strip())
        if j == len(t):
            break
        i = max(0, j - overlap)
    return [c for c in out if c]

def _resolve_artifact_id(conn: sqlite3.Connection, artifact: str | int) -> int:
    if isinstance(artifact, int):
        return artifact
    s = str(artifact).strip()
    if s.isdigit():
        return int(s)
    row = conn.execute("SELECT artifact_id FROM artifacts WHERE uri=?", (s,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown artifact: {artifact}")
    return int(row["artifact_id"])

def _resolve_model_member_id(conn: sqlite3.Connection, model_id: int, uid: str) -> Optional[int]:
    r = conn.execute(
        "SELECT model_member_id FROM model_members WHERE model_id=? AND member_uid=?",
        (model_id, uid),
    ).fetchone()
    return int(r["model_member_id"]) if r else None

# -----------------------
# PDF / MD import + search
# -----------------------
@mcp.tool()
def structai_import_pdf(path: str, title: Optional[str] = None) -> Dict[str, Any]:
    """
    PDF를 페이지 단위로 텍스트 추출하여 doc_chunks에 저장 (page_start 유지)
    """
    from pypdf import PdfReader  # type: ignore

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    uri = p.as_uri()
    sha = _sha256_file(p)
    title0 = title or p.name

    reader = PdfReader(str(p))
    pages = reader.pages

    conn = _connect()
    try:
        artifact_id = _upsert_artifact(conn, uri=uri, kind="pdf", title=title0, source_path=str(p), sha256=sha)
        _delete_doc_chunks_for_artifact(conn, artifact_id)

        chunk_index = 0
        empty_pages = 0
        for i, pg in enumerate(pages, start=1):
            txt = (pg.extract_text() or "").strip()
            if not txt:
                empty_pages += 1
                continue
            # 기본은 page 전체를 1 chunk로 저장(페이지 인용 쉬움)
            conn.execute(
                "INSERT INTO doc_chunks(artifact_id, page_start, page_end, chunk_index, content) VALUES(?,?,?,?,?)",
                (artifact_id, i, i, chunk_index, txt),
            )
            chunk_index += 1

        conn.commit()
        return {"ok": True, "artifact_id": artifact_id, "uri": uri, "pages": len(pages), "empty_pages": empty_pages}
    finally:
        conn.close()

@mcp.tool()
def structai_import_md(path: str, title: Optional[str] = None) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    uri = p.as_uri()
    sha = _sha256_file(p)
    title0 = title or p.name

    text = p.read_text(encoding="utf-8", errors="ignore")
    chunks = _chunk_text(text)

    conn = _connect()
    try:
        artifact_id = _upsert_artifact(conn, uri=uri, kind="md", title=title0, source_path=str(p), sha256=sha)
        _delete_doc_chunks_for_artifact(conn, artifact_id)

        for idx, c in enumerate(chunks):
            conn.execute(
                "INSERT INTO doc_chunks(artifact_id, page_start, page_end, chunk_index, content) VALUES(?, NULL, NULL, ?, ?)",
                (artifact_id, idx, c),
            )
        conn.commit()
        return {"ok": True, "artifact_id": artifact_id, "uri": uri, "chunks": len(chunks)}
    finally:
        conn.close()

@mcp.tool()
def structai_search_docs(query: str, limit: int = 8, kind_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    FTS 검색. kind_filter='pdf' 같은 필터를 걸 수 있음.
    """
    conn = _connect()
    try:
        if kind_filter:
            rows = conn.execute(
                """
                SELECT
                  dc.chunk_id,
                  a.uri, a.kind, a.title,
                  dc.page_start,
                  snippet(doc_chunks_fts, 0, '[', ']', '…', 12) AS snippet,
                  bm25(doc_chunks_fts) AS score
                FROM doc_chunks_fts
                JOIN doc_chunks dc ON dc.chunk_id = doc_chunks_fts.rowid
                JOIN artifacts a ON a.artifact_id = dc.artifact_id
                WHERE doc_chunks_fts MATCH ? AND a.kind=?
                ORDER BY score
                LIMIT ?
                """,
                (query, kind_filter, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                  dc.chunk_id,
                  a.uri, a.kind, a.title,
                  dc.page_start,
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

        out = []
        for r in rows:
            page = r["page_start"]
            cite_uri = f"{r['uri']}#page={int(page)}" if (r["kind"] == "pdf" and page is not None) else f"{r['uri']}#chunk={int(r['chunk_id'])}"
            out.append({**dict(r), "cite_uri": cite_uri})
        return {"query": query, "results": out}
    finally:
        conn.close()

# -----------------------
# DXF import (text -> doc_chunks + cad_entities)
# -----------------------
def _vec_xyz(v: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        return float(v.x), float(v.y), float(v.z)
    except Exception:
        return None, None, None

@mcp.tool()
def structai_import_dxf(path: str, title: Optional[str] = None, include_virtual_text: bool = True) -> Dict[str, Any]:
    """
    DXF에서 TEXT/MTEXT/ATTRIB(+INSERT virtual text 일부)를 추출해:
    - doc_chunks (검색용) 저장
    - cad_entities (좌표/레이어 메타) 저장
    """
    import ezdxf  # type: ignore
    from ezdxf import recover  # type: ignore

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    uri = p.as_uri()
    sha = _sha256_file(p)
    title0 = title or p.name

    # robust load
    try:
        doc = ezdxf.readfile(str(p))
        load_mode = "readfile"
    except Exception:
        doc, _aud = recover.readfile(str(p))
        load_mode = "recover"

    msp = doc.modelspace()

    items: List[Dict[str, Any]] = []

    def add_text(type_: str, layer: Optional[str], text: str, x: Optional[float], y: Optional[float], z: Optional[float], handle: Optional[str], raw: Dict[str, Any]):
        t = (text or "").strip()
        if not t:
            return
        items.append({
            "type": type_,
            "layer": layer,
            "text": t,
            "x": x, "y": y, "z": z,
            "handle": handle,
            "raw": raw,
        })

    # TEXT
    for e in msp.query("TEXT"):
        ins = getattr(e.dxf, "insert", None)
        x, y, z = _vec_xyz(ins) if ins is not None else (None, None, None)
        add_text(
            "TEXT",
            getattr(e.dxf, "layer", None),
            getattr(e.dxf, "text", "") or "",
            x, y, z,
            getattr(e.dxf, "handle", None),
            {"rotation": getattr(e.dxf, "rotation", None), "height": getattr(e.dxf, "height", None)},
        )

    # MTEXT
    for e in msp.query("MTEXT"):
        ins = getattr(e.dxf, "insert", None)
        x, y, z = _vec_xyz(ins) if ins is not None else (None, None, None)
        try:
            t = e.plain_text()
        except Exception:
            t = getattr(e, "text", "") or ""
        add_text(
            "MTEXT",
            getattr(e.dxf, "layer", None),
            t,
            x, y, z,
            getattr(e.dxf, "handle", None),
            {"char_height": getattr(e.dxf, "char_height", None)},
        )

    # INSERT ATTRIB + virtual_entities
    for ins in msp.query("INSERT"):
        ins_handle = getattr(ins.dxf, "handle", None)
        ins_layer = getattr(ins.dxf, "layer", None)
        block_name = getattr(ins.dxf, "name", None)

        # ATTRIB
        for a in getattr(ins, "attribs", []) or []:
            pt = getattr(a.dxf, "insert", None)
            x, y, z = _vec_xyz(pt) if pt is not None else (None, None, None)
            add_text(
                "ATTRIB",
                getattr(a.dxf, "layer", ins_layer),
                getattr(a.dxf, "text", "") or "",
                x, y, z,
                getattr(a.dxf, "handle", None),
                {"tag": getattr(a.dxf, "tag", None), "source_insert": ins_handle, "block": block_name},
            )

        # virtual text inside block
        if include_virtual_text:
            try:
                for ve in ins.virtual_entities():
                    vtype = ve.dxftype()
                    if vtype not in ("TEXT", "MTEXT"):
                        continue
                    inner_layer = getattr(ve.dxf, "layer", None)
                    layer = ins_layer if (inner_layer in (None, "", "0")) else inner_layer

                    if vtype == "TEXT":
                        pt = getattr(ve.dxf, "insert", None)
                        x, y, z = _vec_xyz(pt) if pt is not None else (None, None, None)
                        add_text(
                            "TEXT",
                            layer,
                            getattr(ve.dxf, "text", "") or "",
                            x, y, z,
                            f"{ins_handle}:V",
                            {"virtual": True, "source_insert": ins_handle, "block": block_name, "inner_layer": inner_layer},
                        )
                    else:
                        pt = getattr(ve.dxf, "insert", None)
                        x, y, z = _vec_xyz(pt) if pt is not None else (None, None, None)
                        try:
                            t = ve.plain_text()
                        except Exception:
                            t = getattr(ve, "text", "") or ""
                        add_text(
                            "MTEXT",
                            layer,
                            t,
                            x, y, z,
                            f"{ins_handle}:V",
                            {"virtual": True, "source_insert": ins_handle, "block": block_name, "inner_layer": inner_layer},
                        )
            except Exception:
                pass

    conn = _connect()
    try:
        artifact_id = _upsert_artifact(conn, uri=uri, kind="dxf", title=title0, source_path=str(p), sha256=sha)

        # clear old
        conn.execute("DELETE FROM cad_entities WHERE artifact_id=?", (artifact_id,))
        _delete_doc_chunks_for_artifact(conn, artifact_id)

        # insert each text as doc_chunk + cad_entity
        for idx, it in enumerate(items):
            conn.execute(
                "INSERT INTO doc_chunks(artifact_id, page_start, page_end, chunk_index, content) VALUES(?, NULL, NULL, ?, ?)",
                (artifact_id, idx, it["text"]),
            )
            chunk_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

            conn.execute(
                """
                INSERT INTO cad_entities(
                  artifact_id, chunk_id, handle, type, layer, layout, text, x,y,z, geom_json, raw_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    artifact_id,
                    chunk_id,
                    it.get("handle"),
                    it["type"],
                    it.get("layer"),
                    "Model",
                    it["text"],
                    it.get("x"), it.get("y"), it.get("z"),
                    None,
                    json.dumps(it.get("raw", {}), ensure_ascii=False),
                ),
            )

        conn.commit()
        return {"ok": True, "artifact_id": artifact_id, "uri": uri, "text_items": len(items), "load_mode": load_mode}
    finally:
        conn.close()

@mcp.tool()
def structai_list_artifacts(kind: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    conn = _connect()
    try:
        if kind:
            rows = conn.execute(
                "SELECT artifact_id, uri, kind, title, source_path, created_at, updated_at FROM artifacts WHERE kind=? ORDER BY updated_at DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT artifact_id, uri, kind, title, source_path, created_at, updated_at FROM artifacts ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {"items": [dict(r) for r in rows]}
    finally:
        conn.close()

# -----------------------
# Model import/list (CSV/JSON)
# -----------------------
@mcp.tool()
def structai_model_import_members(path: str, model_name: Optional[str] = None, fmt: Optional[str] = None, units: Optional[str] = None) -> Dict[str, Any]:
    """
    모델 부재 임포트
    - JSON: { "model_name":..., "units":..., "members":[{uid,label,type,x1,y1,z1,x2,y2,z2,section,story}]}
    - CSV 헤더: uid,label,type,x1,y1,z1,x2,y2,z2,section,story
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower()
    if fmt is None:
        fmt = "json" if ext == ".json" else "csv"

    conn = _connect()
    try:
        name = model_name or p.name
        cur = conn.execute(
            "INSERT INTO models(name, source_path, units, meta_json) VALUES(?,?,?,?)",
            (name, str(p), units, "{}"),
        )
        model_id = int(cur.lastrowid)

        def to_f(v):
            if v is None or v == "":
                return None
            try:
                return float(v)
            except Exception:
                return None

        imported = 0
        if fmt == "json":
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict) and "members" in data:
                if data.get("model_name"):
                    conn.execute("UPDATE models SET name=? WHERE model_id=?", (data["model_name"], model_id))
                if data.get("units") and not units:
                    conn.execute("UPDATE models SET units=? WHERE model_id=?", (data["units"], model_id))
                members = data["members"]
            elif isinstance(data, list):
                members = data
            else:
                raise ValueError("JSON format invalid")
            for m in members:
                uid = str(m.get("uid") or m.get("member_uid") or "")
                if not uid:
                    continue
                label = m.get("label")
                label = str(label) if label is not None else None
                ln = normalize_label(label) if label else None
                mtype = str(m.get("type") or "unknown").lower()
                conn.execute(
                    """
                    INSERT INTO model_members(
                      model_id, member_uid, member_label, label_norm, type,
                      x1,y1,z1,x2,y2,z2, section, story, meta_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        model_id, uid, label, ln, mtype,
                        to_f(m.get("x1")), to_f(m.get("y1")), to_f(m.get("z1")),
                        to_f(m.get("x2")), to_f(m.get("y2")), to_f(m.get("z2")),
                        m.get("section"), m.get("story"),
                        json.dumps(m.get("meta", {}), ensure_ascii=False),
                    ),
                )
                imported += 1

        elif fmt == "csv":
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uid = str(row.get("uid") or "").strip()
                    if not uid:
                        continue
                    label = (row.get("label") or "").strip() or None
                    ln = normalize_label(label) if label else None
                    mtype = (row.get("type") or "unknown").strip().lower()
                    conn.execute(
                        """
                        INSERT INTO model_members(
                          model_id, member_uid, member_label, label_norm, type,
                          x1,y1,z1,x2,y2,z2, section, story, meta_json
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            model_id, uid, label, ln, mtype,
                            to_f(row.get("x1")), to_f(row.get("y1")), to_f(row.get("z1")),
                            to_f(row.get("x2")), to_f(row.get("y2")), to_f(row.get("z2")),
                            row.get("section"), row.get("story"),
                            "{}",
                        ),
                    )
                    imported += 1
        else:
            raise ValueError(f"unsupported fmt: {fmt}")

        conn.commit()
        return {"ok": True, "model_id": model_id, "imported": imported}
    finally:
        conn.close()

@mcp.tool()
def structai_model_list(limit: int = 50) -> Dict[str, Any]:
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
    conn = _connect()
    try:
        if contains:
            q = f"%{contains.upper()}%"
            rows = conn.execute(
                """
                SELECT model_member_id, member_uid, member_label, type, section, story, x1,y1,z1,x2,y2,z2
                FROM model_members
                WHERE model_id=? AND (UPPER(member_uid) LIKE ? OR UPPER(COALESCE(member_label,'')) LIKE ?)
                ORDER BY model_member_id DESC LIMIT ?
                """,
                (model_id, q, q, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT model_member_id, member_uid, member_label, type, section, story, x1,y1,z1,x2,y2,z2
                FROM model_members
                WHERE model_id=?
                ORDER BY model_member_id DESC LIMIT ?
                """,
                (model_id, limit),
            ).fetchall()
        return {"items": [dict(r) for r in rows]}
    finally:
        conn.close()

# -----------------------
# Mapping suggestion (same idea as step 3)
# -----------------------
_DEFAULT_TOKEN_RE = re.compile(r"\b[A-Z]{1,4}\s?-?\s?\d{1,4}\b")

@dataclass
class CadOccurrence:
    cad_entity_id: int
    x: Optional[float]
    y: Optional[float]
    layer: Optional[str]

def dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def midpoint(member_row: sqlite3.Row) -> Optional[Tuple[float, float]]:
    x1, y1, x2, y2 = member_row["x1"], member_row["y1"], member_row["x2"], member_row["y2"]
    if x1 is None or y1 is None:
        return None
    if x2 is None or y2 is None:
        return (float(x1), float(y1))
    return ((float(x1)+float(x2))/2.0, (float(y1)+float(y2))/2.0)

def guess_type_from_token(token: str) -> Optional[str]:
    t = token.upper()
    if t.startswith("C"):
        return "column"
    if t.startswith("B"):
        return "beam"
    if t.startswith("BR"):
        return "brace"
    if t.startswith("W"):
        return "wall"
    if t.startswith("S"):
        return "slab"
    return None

def score_mapping(token_norm: str, occs: List[CadOccurrence], member: sqlite3.Row, spatial_tol: float) -> Tuple[float, Dict[str, Any], str]:
    ln = member["label_norm"] or ""
    label_score = 0.0
    method = "none"

    if token_norm and ln and token_norm == ln:
        label_score = 1.0
        method = "label_exact"
    elif token_norm and ln and (token_norm in ln or ln in token_norm):
        label_score = 0.7
        method = "label_fuzzy"

    spatial_score = 0.0
    min_d = None
    mid = midpoint(member)
    if mid and occs:
        ds = [dist2d((o.x, o.y), mid) for o in occs if o.x is not None and o.y is not None]
        if ds:
            min_d = min(ds)
            if spatial_tol and min_d <= spatial_tol:
                spatial_score = max(0.0, 1.0 - (min_d / spatial_tol))

    # combine
    if label_score >= 1.0:
        conf = 0.8*label_score + 0.2*spatial_score
        method = "hybrid" if spatial_score > 0 else method
    elif label_score > 0:
        conf = 0.55*label_score + 0.45*spatial_score
        method = "hybrid" if spatial_score > 0 else method
    else:
        conf = 0.9*spatial_score
        method = "spatial" if spatial_score > 0 else "none"

    ev = {"token_norm": token_norm, "member_label_norm": ln, "min_distance": min_d, "spatial_tolerance": spatial_tol, "label_score": label_score, "spatial_score": spatial_score}
    return float(max(0.0, min(1.0, conf))), ev, method

@mcp.tool()
def structai_map_suggest_members(
    cad_artifact_id: int,
    model_id: int,
    max_tokens: int = 200,
    min_occurrences: int = 1,
    max_candidates_per_token: int = 5,
    spatial_tolerance: float = 5.0,
    enable_fuzzy: bool = True
) -> Dict[str, Any]:
    conn = _connect()
    try:
        cad_rows = conn.execute(
            "SELECT cad_entity_id, text, layer, x, y FROM cad_entities WHERE artifact_id=? AND text IS NOT NULL",
            (cad_artifact_id,),
        ).fetchall()

        buckets: Dict[str, List[CadOccurrence]] = {}
        for r in cad_rows:
            txt = str(r["text"] or "")
            for tok in _DEFAULT_TOKEN_RE.findall(txt.upper()):
                tnorm = normalize_label(tok)
                if not tnorm:
                    continue
                buckets.setdefault(tnorm, []).append(CadOccurrence(
                    cad_entity_id=int(r["cad_entity_id"]),
                    x=float(r["x"]) if r["x"] is not None else None,
                    y=float(r["y"]) if r["y"] is not None else None,
                    layer=r["layer"],
                ))

        token_items = [(k,v) for k,v in buckets.items() if len(v) >= min_occurrences]
        token_items.sort(key=lambda kv: len(kv[1]), reverse=True)
        token_items = token_items[:max_tokens]

        members = conn.execute(
            """
            SELECT model_member_id, member_uid, member_label, label_norm, type,
                   x1,y1,z1,x2,y2,z2, section, story
            FROM model_members
            WHERE model_id=?
            """,
            (model_id,),
        ).fetchall()

        by_label: Dict[str, List[sqlite3.Row]] = {}
        labeled = []
        for m in members:
            ln = m["label_norm"]
            if ln:
                by_label.setdefault(str(ln), []).append(m)
                labeled.append(m)

        out = []
        unmatched = []

        for token_norm, occs in token_items:
            candidates = []

            exact = by_label.get(token_norm, [])
            for m in exact:
                conf, ev, method = score_mapping(token_norm, occs, m, spatial_tolerance)
                candidates.append({"model_member_id": int(m["model_member_id"]), "member_uid": m["member_uid"], "member_label": m["member_label"],
                                   "type": m["type"], "section": m["section"], "story": m["story"], "confidence": conf, "method": method, "evidence": ev})

            if not candidates and enable_fuzzy:
                fuzzy = [m for m in labeled if (token_norm in (m["label_norm"] or "") or (m["label_norm"] or "") in token_norm)]
                for m in fuzzy[:2000]:
                    conf, ev, method = score_mapping(token_norm, occs, m, spatial_tolerance)
                    if conf > 0.1:
                        candidates.append({"model_member_id": int(m["model_member_id"]), "member_uid": m["member_uid"], "member_label": m["member_label"],
                                           "type": m["type"], "section": m["section"], "story": m["story"], "confidence": conf, "method": method, "evidence": ev})

            candidates.sort(key=lambda c: c["confidence"], reverse=True)
            candidates = candidates[:max_candidates_per_token]

            if not candidates:
                unmatched.append({"token": token_norm, "occurrence_count": len(occs)})
                continue

            out.append({
                "token": token_norm,
                "token_norm": token_norm,
                "type_guess": guess_type_from_token(token_norm),
                "occurrence_count": len(occs),
                "occurrences": [{"cad_entity_id": o.cad_entity_id, "x": o.x, "y": o.y, "layer": o.layer} for o in occs[:20]],
                "candidates": candidates,
            })

        return {"cad_artifact_id": cad_artifact_id, "model_id": model_id, "items": out, "unmatched_tokens": unmatched}
    finally:
        conn.close()

@mcp.tool()
def structai_map_save_mappings(mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
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
                  cad_artifact_id, cad_token, cad_token_norm, model_id, model_member_id,
                  confidence, method, status, evidence_json, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?, datetime('now'))
                ON CONFLICT(cad_artifact_id, cad_token_norm, model_id, model_member_id)
                DO UPDATE SET
                  confidence=excluded.confidence,
                  method=excluded.method,
                  status=excluded.status,
                  evidence_json=excluded.evidence_json,
                  updated_at=datetime('now')
                """,
                (cad_artifact_id, cad_token, cad_token_norm, model_id, model_member_id,
                 confidence, method, status, json.dumps(evidence, ensure_ascii=False)),
            )
            saved += 1
        conn.commit()
        return {"ok": True, "saved": saved}
    finally:
        conn.close()

@mcp.tool()
def structai_map_list_mappings(cad_artifact_id: Optional[int] = None, model_id: Optional[int] = None, status: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
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
              mm.mapping_id, mm.cad_artifact_id, mm.cad_token, mm.cad_token_norm, mm.model_id,
              mm.model_member_id, mm.confidence, mm.method, mm.status, mm.updated_at,
              mb.member_uid, mb.member_label, mb.type AS member_type, mb.section, mb.story
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

# -----------------------
# Analysis results import
# -----------------------
def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None

@mcp.tool()
def structai_results_import(
    model_id: int,
    path: str,
    run_name: Optional[str] = None,
    engine: Optional[str] = None,
    units: Optional[str] = None,
    fmt: Optional[str] = None
) -> Dict[str, Any]:
    """
    해석결과(부재별 envelope)를 임포트 (CSV/JSON)
    CSV 권장 헤더:
      uid, combo, N_max, N_min, V2_max, V2_min, V3_max, V3_min, M2_max, M2_min, M3_max, M3_min, D_max
    JSON 예:
    {
      "run_name":"ULS Run 1",
      "engine":"etabs",
      "units":"kN-m",
      "results":[{"uid":"E1","combo":"ULS1","envelope":{"M3_max":..., "M3_min":...}}]
    }
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower()
    if fmt is None:
        fmt = "json" if ext == ".json" else "csv"

    conn = _connect()
    try:
        name = run_name or p.name
        cur = conn.execute(
            "INSERT INTO analysis_runs(model_id, name, engine, units, source_path, meta_json) VALUES(?,?,?,?,?,?)",
            (int(model_id), name, engine, units, str(p), "{}"),
        )
        analysis_run_id = int(cur.lastrowid)

        inserted = 0
        missing_members = 0
        errors: List[Dict[str, Any]] = []

        if fmt == "json":
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                if data.get("run_name") and not run_name:
                    conn.execute("UPDATE analysis_runs SET name=? WHERE analysis_run_id=?", (data["run_name"], analysis_run_id))
                if data.get("engine") and not engine:
                    conn.execute("UPDATE analysis_runs SET engine=? WHERE analysis_run_id=?", (data["engine"], analysis_run_id))
                if data.get("units") and not units:
                    conn.execute("UPDATE analysis_runs SET units=? WHERE analysis_run_id=?", (data["units"], analysis_run_id))
                results = data.get("results") or []
            else:
                raise ValueError("JSON must be an object with 'results'.")

            for r in results:
                uid = str(r.get("uid") or r.get("member_uid") or "").strip()
                combo = str(r.get("combo") or "").strip()
                env = r.get("envelope") or {}
                if not uid or not combo:
                    continue
                mmid = _resolve_model_member_id(conn, int(model_id), uid)
                if mmid is None:
                    missing_members += 1
                    continue
                conn.execute(
                    """
                    INSERT INTO member_results(analysis_run_id, model_member_id, combo, envelope_json, updated_at)
                    VALUES(?,?,?,?, datetime('now'))
                    ON CONFLICT(analysis_run_id, model_member_id, combo)
                    DO UPDATE SET
                      envelope_json=excluded.envelope_json,
                      updated_at=datetime('now')
                    """,
                    (analysis_run_id, mmid, combo, json.dumps(env, ensure_ascii=False)),
                )
                inserted += 1

        elif fmt == "csv":
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uid = str(row.get("uid") or "").strip()
                    combo = str(row.get("combo") or "").strip()
                    if not uid or not combo:
                        continue
                    mmid = _resolve_model_member_id(conn, int(model_id), uid)
                    if mmid is None:
                        missing_members += 1
                        continue

                    env: Dict[str, Any] = {}
                    for k, v in row.items():
                        if k in ("uid", "combo"):
                            continue
                        if v is None or v == "":
                            continue
                        fv = _to_float(v)
                        env[k] = fv if fv is not None else v

                    conn.execute(
                        """
                        INSERT INTO member_results(analysis_run_id, model_member_id, combo, envelope_json, updated_at)
                        VALUES(?,?,?,?, datetime('now'))
                        ON CONFLICT(analysis_run_id, model_member_id, combo)
                        DO UPDATE SET
                          envelope_json=excluded.envelope_json,
                          updated_at=datetime('now')
                        """,
                        (analysis_run_id, mmid, combo, json.dumps(env, ensure_ascii=False)),
                    )
                    inserted += 1
        else:
            raise ValueError(f"unsupported fmt: {fmt}")

        conn.commit()
        return {
            "ok": True,
            "analysis_run_id": analysis_run_id,
            "inserted": inserted,
            "missing_members": missing_members,
            "errors": errors,
        }
    finally:
        conn.close()

@mcp.tool()
def structai_results_list_runs(model_id: int, limit: int = 50) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT analysis_run_id, model_id, name, engine, units, source_path, created_at FROM analysis_runs WHERE model_id=? ORDER BY analysis_run_id DESC LIMIT ?",
            (int(model_id), int(limit)),
        ).fetchall()
        return {"items": [dict(r) for r in rows]}
    finally:
        conn.close()

@mcp.tool()
def structai_results_list_combos(analysis_run_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT combo FROM member_results WHERE analysis_run_id=? ORDER BY combo ASC",
            (int(analysis_run_id),),
        ).fetchall()
        return {"analysis_run_id": int(analysis_run_id), "combos": [r["combo"] for r in rows]}
    finally:
        conn.close()

# -----------------------
# Design inputs import/set
# -----------------------
@mcp.tool()
def structai_design_set_member_inputs(model_id: int, member_uid: str, design: Dict[str, Any]) -> Dict[str, Any]:
    """
    특정 부재에 설계입력(내력/허용치 등) JSON 저장
    예: {"Mn":120.0,"Vn":80.0,"Pn":300.0,"D_allow":0.02}
    """
    conn = _connect()
    try:
        mmid = _resolve_model_member_id(conn, int(model_id), str(member_uid))
        if mmid is None:
            raise ValueError(f"member not found: model_id={model_id}, uid={member_uid}")

        conn.execute(
            """
            INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
            VALUES(?,?, datetime('now'))
            ON CONFLICT(model_member_id) DO UPDATE SET
              design_json=excluded.design_json,
              updated_at=datetime('now')
            """,
            (mmid, json.dumps(design, ensure_ascii=False)),
        )
        conn.commit()
        return {"ok": True, "model_member_id": mmid}
    finally:
        conn.close()

@mcp.tool()
def structai_design_import_inputs(model_id: int, path: str, fmt: Optional[str] = None) -> Dict[str, Any]:
    """
    부재별 설계입력(내력/허용치)을 CSV/JSON으로 임포트

    CSV 헤더 예:
      uid, Mn, Vn, Pn, D_allow, drift_allow
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower()
    if fmt is None:
        fmt = "json" if ext == ".json" else "csv"

    conn = _connect()
    try:
        upserted = 0
        missing = 0

        if fmt == "json":
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            items = data.get("items") if isinstance(data, dict) else data
            if not isinstance(items, list):
                raise ValueError("JSON must be list or {items:[...]}")

            for it in items:
                uid = str(it.get("uid") or it.get("member_uid") or "").strip()
                if not uid:
                    continue
                mmid = _resolve_model_member_id(conn, int(model_id), uid)
                if mmid is None:
                    missing += 1
                    continue
                design = {k: it[k] for k in it.keys() if k not in ("uid", "member_uid")}
                conn.execute(
                    """
                    INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
                    VALUES(?,?, datetime('now'))
                    ON CONFLICT(model_member_id) DO UPDATE SET
                      design_json=excluded.design_json,
                      updated_at=datetime('now')
                    """,
                    (mmid, json.dumps(design, ensure_ascii=False)),
                )
                upserted += 1

        elif fmt == "csv":
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uid = str(row.get("uid") or "").strip()
                    if not uid:
                        continue
                    mmid = _resolve_model_member_id(conn, int(model_id), uid)
                    if mmid is None:
                        missing += 1
                        continue
                    design: Dict[str, Any] = {}
                    for k, v in row.items():
                        if k == "uid":
                            continue
                        if v is None or v == "":
                            continue
                        fv = _to_float(v)
                        design[k] = fv if fv is not None else v

                    conn.execute(
                        """
                        INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
                        VALUES(?,?, datetime('now'))
                        ON CONFLICT(model_member_id) DO UPDATE SET
                          design_json=excluded.design_json,
                          updated_at=datetime('now')
                        """,
                        (mmid, json.dumps(design, ensure_ascii=False)),
                    )
                    upserted += 1
        else:
            raise ValueError(f"unsupported fmt: {fmt}")

        conn.commit()
        return {"ok": True, "upserted": upserted, "missing_members": missing}
    finally:
        conn.close()

# -----------------------
# Rulepack (룰셋) + safe eval
# -----------------------
ALLOWED_FUNCS = {"abs": abs, "max": max, "min": min, "round": round}

def safe_eval(expr: str, vars_: Dict[str, Any]) -> float:
    """
    안전한 산술 eval: 기본 산술 + abs/max/min/round만 허용.
    """
    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom)):
            raise ValueError(f"disallowed expression node: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                raise ValueError("only abs/max/min/round calls allowed")
        if isinstance(node, ast.Name):
            if node.id not in vars_ and node.id not in ALLOWED_FUNCS:
                # 없는 변수는 0으로 처리하면 오작동이 숨어버리므로 에러로 처리
                raise KeyError(f"unknown variable: {node.id}")

    val = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {} , **ALLOWED_FUNCS}, vars_)
    try:
        return float(val)
    except Exception:
        raise ValueError(f"expression did not evaluate to number: {expr}")

def enrich_envelope(env: Dict[str, Any]) -> Dict[str, Any]:
    """
    env에 *_absmax 계산값을 추가 (예: M3_absmax = max(|M3_max|,|M3_min|))
    """
    out = dict(env)
    bases = ["N", "V2", "V3", "M2", "M3", "T", "D", "DRIFT"]
    for b in bases:
        kmax = f"{b}_max"
        kmin = f"{b}_min"
        if kmax in out or kmin in out:
            a = abs(float(out.get(kmax) or 0.0))
            c = abs(float(out.get(kmin) or 0.0))
            out[f"{b}_absmax"] = max(a, c)
    # 흔한 키 alias도 보정(예: P_max -> N_max)
    if "P_max" in out and "N_max" not in out:
        out["N_max"] = out["P_max"]
    if "P_min" in out and "N_min" not in out:
        out["N_min"] = out["P_min"]
    return out

BUILTIN_RULEPACK = {
    "name": "generic",
    "version": "0.1",
    "checks": {
        # Ratio = demand/capacity, PASS if <= limit
        "strength.flexure": {
            "demand_expr": "M3_absmax",
            "capacity_expr": "Mn",
            "limit": 1.0,
            "warn": 0.95,
            "unit": "force*length",
            "citations": [{"query": "flexure strength", "kind": "pdf", "note": "replace with your code clause id"}],
        },
        "strength.shear": {
            "demand_expr": "V2_absmax",
            "capacity_expr": "Vn",
            "limit": 1.0,
            "warn": 0.95,
            "unit": "force",
            "citations": [{"query": "shear strength", "kind": "pdf", "note": "replace with your code clause id"}],
        },
        "strength.axial": {
            "demand_expr": "N_absmax",
            "capacity_expr": "Pn",
            "limit": 1.0,
            "warn": 0.95,
            "unit": "force",
            "citations": [{"query": "axial strength", "kind": "pdf", "note": "replace with your code clause id"}],
        },
        "service.deflection": {
            "demand_expr": "D_max",
            "capacity_expr": "D_allow",
            "limit": 1.0,
            "warn": 0.9,
            "unit": "length",
            "citations": [{"query": "deflection limit", "kind": "pdf", "note": "replace with your code clause id"}],
        },
        "seismic.drift": {
            "demand_expr": "DRIFT_max",
            "capacity_expr": "drift_allow",
            "limit": 1.0,
            "warn": 0.9,
            "unit": "ratio",
            "citations": [{"query": "drift limit", "kind": "pdf", "note": "replace with your code clause id"}],
        },
    }
}

def _get_active_rulepack(conn: sqlite3.Connection) -> Dict[str, Any]:
    row = conn.execute("SELECT rulepack_json FROM rulepacks WHERE is_active=1 ORDER BY rulepack_id DESC LIMIT 1").fetchone()
    if row:
        try:
            return json.loads(row["rulepack_json"])
        except Exception:
            pass
    return BUILTIN_RULEPACK

def _code_citations(conn: sqlite3.Connection, queries: List[Dict[str, Any]], limit_each: int = 1) -> List[Dict[str, Any]]:
    """
    룰셋의 citations.query로 PDF에서 FTS 검색하여 1개 근거(페이지 포함)를 반환.
    """
    out: List[Dict[str, Any]] = []
    for q in (queries or []):
        query = str(q.get("query") or "").strip()
        if not query:
            continue
        kind = q.get("kind")  # e.g. 'pdf'
        if kind:
            rows = conn.execute(
                """
                SELECT dc.chunk_id, a.uri, a.title, a.kind, dc.page_start,
                       snippet(doc_chunks_fts, 0, '[', ']', '…', 12) AS snippet,
                       bm25(doc_chunks_fts) AS score
                FROM doc_chunks_fts
                JOIN doc_chunks dc ON dc.chunk_id = doc_chunks_fts.rowid
                JOIN artifacts a ON a.artifact_id = dc.artifact_id
                WHERE doc_chunks_fts MATCH ? AND a.kind=?
                ORDER BY score
                LIMIT ?
                """,
                (query, kind, limit_each),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT dc.chunk_id, a.uri, a.title, a.kind, dc.page_start,
                       snippet(doc_chunks_fts, 0, '[', ']', '…', 12) AS snippet,
                       bm25(doc_chunks_fts) AS score
                FROM doc_chunks_fts
                JOIN doc_chunks dc ON dc.chunk_id = doc_chunks_fts.rowid
                JOIN artifacts a ON a.artifact_id = dc.artifact_id
                WHERE doc_chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query, limit_each),
            ).fetchall()

        for r in rows:
            page = r["page_start"]
            cite_uri = f"{r['uri']}#page={int(page)}" if (r["kind"] == "pdf" and page is not None) else f"{r['uri']}#chunk={int(r['chunk_id'])}"
            out.append({
                "query": query,
                "note": q.get("note"),
                "chunk_id": int(r["chunk_id"]),
                "uri": r["uri"],
                "title": r["title"],
                "kind": r["kind"],
                "page": int(page) if page is not None else None,
                "snippet": r["snippet"],
                "cite_uri": cite_uri,
            })
    return out

@mcp.tool()
def structai_rules_import_rulepack(path: str) -> Dict[str, Any]:
    """
    룰셋(JSON) 임포트:
    {
      "name":"kds-rc",
      "version":"2025.01",
      "checks":{
        "strength.flexure":{"demand_expr":"M3_absmax","capacity_expr":"Mn","limit":1.0,"citations":[{"query":"KDS ... 4.3.1"}]}
      }
    }
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    name = data.get("name") or p.stem
    ver = data.get("version") or "0.0"
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO rulepacks(name, version, rulepack_json, is_active)
            VALUES(?,?,?,0)
            ON CONFLICT(name,version) DO UPDATE SET
              rulepack_json=excluded.rulepack_json
            """,
            (name, ver, json.dumps(data, ensure_ascii=False)),
        )
        conn.commit()
        return {"ok": True, "name": name, "version": ver}
    finally:
        conn.close()

@mcp.tool()
def structai_rules_list() -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT rulepack_id, name, version, is_active, created_at FROM rulepacks ORDER BY rulepack_id DESC"
        ).fetchall()
        return {"items": [dict(r) for r in rows], "builtin": {"name": BUILTIN_RULEPACK["name"], "version": BUILTIN_RULEPACK["version"]}}
    finally:
        conn.close()

@mcp.tool()
def structai_rules_set_active(rulepack_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute("UPDATE rulepacks SET is_active=0")
        conn.execute("UPDATE rulepacks SET is_active=1 WHERE rulepack_id=?", (int(rulepack_id),))
        conn.commit()
        return {"ok": True, "active_rulepack_id": int(rulepack_id)}
    finally:
        conn.close()

# -----------------------
# Checks: deterministic run
# -----------------------
def _status_from_ratio(ratio: Optional[float], limit: float, warn: float) -> str:
    if ratio is None:
        return "NA"
    if ratio <= warn:
        return "PASS"
    if ratio <= limit:
        return "WARN"
    return "FAIL"

@mcp.tool()
def structai_check_run(
    model_id: int,
    analysis_run_id: int,
    name: Optional[str] = None,
    combos: Optional[List[str]] = None,
    check_types: Optional[List[str]] = None,
    only_mapped_from_cad_artifact_id: Optional[int] = None,
    mapping_status: str = "confirmed"
) -> Dict[str, Any]:
    """
    체크 실행:
    - rulepack(활성 룰셋 또는 builtin)을 사용
    - member_results의 envelope + member_design_inputs의 capacity/limit로 ratio 계산
    - citations(query)로 PDF 근거(페이지 포함) 자동 첨부
    """
    conn = _connect()
    try:
        rulepack = _get_active_rulepack(conn)
        checks_def: Dict[str, Any] = rulepack.get("checks") or {}

        if not checks_def:
            raise ValueError("rulepack has no checks")

        if check_types is None or len(check_types) == 0:
            check_types = list(checks_def.keys())
        else:
            for ct in check_types:
                if ct not in checks_def:
                    raise ValueError(f"unknown check_type in rulepack: {ct}")

        # combos default: distinct combos in results
        if combos is None or len(combos) == 0:
            rows = conn.execute("SELECT DISTINCT combo FROM member_results WHERE analysis_run_id=? ORDER BY combo ASC", (int(analysis_run_id),)).fetchall()
            combos = [r["combo"] for r in rows]

        # member scope
        member_ids: List[int] = []
        if only_mapped_from_cad_artifact_id is not None:
            rows = conn.execute(
                """
                SELECT DISTINCT mm.model_member_id
                FROM member_mappings mm
                JOIN model_members mb ON mb.model_member_id = mm.model_member_id
                WHERE mm.cad_artifact_id=? AND mm.model_id=? AND mm.status=?
                """,
                (int(only_mapped_from_cad_artifact_id), int(model_id), str(mapping_status)),
            ).fetchall()
            member_ids = [int(r["model_member_id"]) for r in rows]
        else:
            rows = conn.execute("SELECT model_member_id FROM model_members WHERE model_id=?", (int(model_id),)).fetchall()
            member_ids = [int(r["model_member_id"]) for r in rows]

        run_name = name or f"check_{model_id}_{analysis_run_id}"
        conn.execute(
            """
            INSERT INTO check_runs(model_id, analysis_run_id, name, rulepack_name, rulepack_version, combos_json, check_types_json, scope_json)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                int(model_id),
                int(analysis_run_id),
                run_name,
                rulepack.get("name"),
                rulepack.get("version"),
                json.dumps(combos, ensure_ascii=False),
                json.dumps(check_types, ensure_ascii=False),
                json.dumps({"only_mapped_from_cad_artifact_id": only_mapped_from_cad_artifact_id, "mapping_status": mapping_status}, ensure_ascii=False),
            ),
        )
        check_run_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

        # Preload design inputs
        di_rows = conn.execute(
            """
            SELECT model_member_id, design_json
            FROM member_design_inputs
            WHERE model_member_id IN (%s)
            """ % ",".join(["?"] * max(1, len(member_ids))),
            member_ids if member_ids else [0],
        ).fetchall()
        design_map = {int(r["model_member_id"]): json.loads(r["design_json"]) for r in di_rows}

        inserted = 0
        stat = {"PASS": 0, "WARN": 0, "FAIL": 0, "NA": 0}

        # iterate members x combos
        for mid in member_ids:
            design = design_map.get(mid)
            if design is None:
                # 설계입력 없으면 NA로 남기거나 skip. 여기선 NA로 기록.
                design = {}

            for combo in combos:
                rr = conn.execute(
                    "SELECT envelope_json FROM member_results WHERE analysis_run_id=? AND model_member_id=? AND combo=?",
                    (int(analysis_run_id), int(mid), str(combo)),
                ).fetchone()
                if rr is None:
                    # 결과 없으면 NA
                    env = {}
                else:
                    env = json.loads(rr["envelope_json"])
                env = enrich_envelope(env)

                # 각 체크 타입 계산
                for ct in check_types:
                    cd = checks_def[ct]
                    demand_expr = str(cd.get("demand_expr") or "").strip()
                    capacity_expr = str(cd.get("capacity_expr") or "").strip()
                    limit = float(cd.get("limit", 1.0))
                    warn = float(cd.get("warn", limit))

                    demand_val = None
                    cap_val = None
                    ratio = None
                    status = "NA"
                    details = {"rule": {"demand_expr": demand_expr, "capacity_expr": capacity_expr, "limit": limit, "warn": warn}}

                    try:
                        if demand_expr:
                            demand_val = safe_eval(demand_expr, {k: float(v) for k, v in env.items() if isinstance(v, (int, float))})
                        if capacity_expr:
                            # design inputs는 key가 문자열로 들어오므로, 숫자인 값만 eval 변수로 제공
                            dv = {k: float(v) for k, v in design.items() if isinstance(v, (int, float))}
                            cap_val = safe_eval(capacity_expr, dv)
                        if demand_val is not None and cap_val is not None and cap_val != 0:
                            ratio = float(demand_val / cap_val)
                            status = _status_from_ratio(ratio, limit, warn)
                        else:
                            status = "NA"
                    except Exception as e:
                        details["error"] = str(e)
                        status = "NA"

                    citations = _code_citations(conn, cd.get("citations") or [], limit_each=1)

                    conn.execute(
                        """
                        INSERT INTO check_results(
                          check_run_id, model_member_id, combo, check_type,
                          demand_value, capacity_value, ratio, status,
                          details_json, citations_json
                        ) VALUES(?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            int(check_run_id),
                            int(mid),
                            str(combo),
                            str(ct),
                            demand_val,
                            cap_val,
                            ratio,
                            status,
                            json.dumps(details, ensure_ascii=False),
                            json.dumps(citations, ensure_ascii=False),
                        ),
                    )
                    inserted += 1
                    stat[status] = stat.get(status, 0) + 1

        conn.commit()
        return {
            "ok": True,
            "check_run_id": check_run_id,
            "name": run_name,
            "rulepack": {"name": rulepack.get("name"), "version": rulepack.get("version")},
            "combos": combos,
            "check_types": check_types,
            "inserted": inserted,
            "summary": stat,
        }
    finally:
        conn.close()

@mcp.tool()
def structai_check_list_runs(model_id: int, limit: int = 50) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT check_run_id, model_id, analysis_run_id, name, rulepack_name, rulepack_version, created_at
            FROM check_runs
            WHERE model_id=?
            ORDER BY check_run_id DESC
            LIMIT ?
            """,
            (int(model_id), int(limit)),
        ).fetchall()
        return {"items": [dict(r) for r in rows]}
    finally:
        conn.close()

@mcp.tool()
def structai_check_get_results(check_run_id: int, status: Optional[str] = None, limit: int = 500) -> Dict[str, Any]:
    conn = _connect()
    try:
        if status:
            rows = conn.execute(
                """
                SELECT
                  cr.check_result_id, cr.combo, cr.check_type, cr.demand_value, cr.capacity_value, cr.ratio, cr.status,
                  mb.member_uid, mb.member_label, mb.type, mb.section, mb.story,
                  cr.citations_json
                FROM check_results cr
                JOIN model_members mb ON mb.model_member_id = cr.model_member_id
                WHERE cr.check_run_id=? AND cr.status=?
                ORDER BY cr.ratio DESC NULLS LAST
                LIMIT ?
                """,
                (int(check_run_id), str(status), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT
                  cr.check_result_id, cr.combo, cr.check_type, cr.demand_value, cr.capacity_value, cr.ratio, cr.status,
                  mb.member_uid, mb.member_label, mb.type, mb.section, mb.story,
                  cr.citations_json
                FROM check_results cr
                JOIN model_members mb ON mb.model_member_id = cr.model_member_id
                WHERE cr.check_run_id=?
                ORDER BY cr.status DESC, cr.ratio DESC NULLS LAST
                LIMIT ?
                """,
                (int(check_run_id), int(limit)),
            ).fetchall()

        out = []
        for r in rows:
            item = dict(r)
            try:
                item["citations"] = json.loads(item.pop("citations_json") or "[]")
            except Exception:
                item["citations"] = []
            out.append(item)
        return {"check_run_id": int(check_run_id), "count": len(out), "items": out}
    finally:
        conn.close()

@mcp.tool()
def structai_reset_all() -> Dict[str, Any]:
    """
    DEV: 전부 삭제
    """
    conn = _connect()
    try:
        conn.executescript(
            """
            DELETE FROM check_results;
            DELETE FROM check_runs;
            DELETE FROM member_design_inputs;
            DELETE FROM member_results;
            DELETE FROM analysis_runs;
            DELETE FROM member_mappings;
            DELETE FROM model_members;
            DELETE FROM models;
            DELETE FROM cad_entities;
            DELETE FROM doc_chunks;
            DELETE FROM artifacts;
            DELETE FROM rulepacks;
            """
        )
        conn.commit()
        return {"ok": True, "db": str(DB_PATH)}
    finally:
        conn.close()

if __name__ == "__main__":
    mcp.run()

4-3) 임포트 파일 포맷 예시
A) 해석결과 CSV 예시 (member envelope)

results.csv

uid,combo,N_max,N_min,V2_max,V2_min,M3_max,M3_min,D_max
E1,ULS1,200,-180,60,-55,120,-110,
E1,SLS1,120,-100,35,-30,70,-65,0.012
E2,ULS1,150,-140,50,-45,90,-85,


키 이름은 자유지만, **룰셋(demand_expr)**에서 그 키를 쓰면 된다.

absmax는 엔진이 자동 생성: M3_absmax, V2_absmax, N_absmax 등.

B) 설계입력 CSV 예시 (capacities/limits)

design_inputs.csv

uid,Mn,Vn,Pn,D_allow,drift_allow
E1,150,80,300,0.02,0.015
E2,120,70,250,0.02,0.015


이 값들이 룰셋의 capacity_expr에서 참조되는 변수들이다.

C) 룰셋 JSON 예시 (코드 조항 쿼리 연결)

rulepack_kds_stub.json

{
  "name": "kds-stub",
  "version": "2025.01",
  "checks": {
    "strength.flexure": {
      "demand_expr": "M3_absmax",
      "capacity_expr": "Mn",
      "limit": 1.0,
      "warn": 0.95,
      "citations": [
        { "query": "KDS 휨 강도", "kind": "pdf", "note": "실제 조항번호/키워드로 교체" }
      ]
    },
    "service.deflection": {
      "demand_expr": "D_max",
      "capacity_expr": "D_allow",
      "limit": 1.0,
      "warn": 0.9,
      "citations": [
        { "query": "처짐 허용", "kind": "pdf", "note": "실제 조항번호/키워드로 교체" }
      ]
    }
  }
}


핵심: citations.query를 “조항번호/정확한 문구”로 잡으면 FTS가 거의 결정론적으로 동일 chunk를 찾아서 감사 가능하게 됨.

4-4) VS Code “Results” 패널 UI 붙이기 (선택이지만 추천)

너가 원하는 “대화 + 패널 통합”을 위해, 우측 패널에 결과 임포트/체크 실행/결과 열람 UI를 붙이는 게 좋아.

아래는 최소 구현.

4-4-1) extension.ts에 Results View 등록
// src/extension.ts
import * as vscode from "vscode";
import { MappingViewProvider } from "./views/mappingView";
import { ResultsViewProvider } from "./views/resultsView";

export function activate(context: vscode.ExtensionContext) {
  // ... (기존 MCP provider, chat participant 등록 등)

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      "structai.mapping",
      new MappingViewProvider(context)
    )
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      "structai.results",
      new ResultsViewProvider(context),
      { webviewOptions: { retainContextWhenHidden: true } }
    )
  );
}

4-4-2) src/views/resultsView.ts
import * as vscode from "vscode";

export class ResultsViewProvider implements vscode.WebviewViewProvider {
  constructor(private readonly ctx: vscode.ExtensionContext) {}

  resolveWebviewView(view: vscode.WebviewView) {
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.ctx.extensionUri, "media")]
    };

    const jsUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.ctx.extensionUri, "media", "results", "main.js"));
    const cssUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.ctx.extensionUri, "media", "results", "styles.css"));

    view.webview.html = `<!doctype html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src ${view.webview.cspSource}; script-src ${view.webview.cspSource};" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="${cssUri}" rel="stylesheet" />
  <title>StructAI Results</title>
</head>
<body>
  <div id="app"></div>
  <script src="${jsUri}"></script>
</body>
</html>`;

    const invoke = async (toolSuffix: string, input: any) => {
      const tool = vscode.lm.tools.find(t => t.name.includes(toolSuffix));
      if (!tool) throw new Error(`Tool not found: ${toolSuffix}`);
      const res: any = await vscode.lm.invokeTool(tool.name, { input });
      return unwrap(res);
    };

    const unwrap = (res: any): any => {
      if (res && typeof res === "object" && Array.isArray(res.content)) return unwrap(res.content);
      if (Array.isArray(res)) {
        const txt = res.map((p:any)=> (typeof p === "string" ? p : (p?.value ?? p?.text ?? JSON.stringify(p)))).join("");
        try { return JSON.parse(txt); } catch { return { raw: txt }; }
      }
      if (typeof res === "string") { try { return JSON.parse(res); } catch { return { raw: res }; } }
      return res;
    };

    const sendState = async () => {
      const models = await invoke("structai_model_list", {});
      view.webview.postMessage({ type: "models", models });
    };

    view.webview.onDidReceiveMessage(async (msg) => {
      try {
        switch (msg.type) {
          case "init":
            await sendState();
            break;

          case "loadRuns": {
            const runs = await invoke("structai_results_list_runs", { model_id: msg.model_id });
            const checks = await invoke("structai_check_list_runs", { model_id: msg.model_id });
            view.webview.postMessage({ type: "runs", runs, checks });
            break;
          }

          case "importResults": {
            const pick = await vscode.window.showOpenDialog({ canSelectMany: false, filters: { "Results": ["csv","json"] }});
            if (!pick?.length) return;
            const result = await invoke("structai_results_import", { model_id: msg.model_id, path: pick[0].fsPath });
            view.webview.postMessage({ type: "importResultsDone", result });
            break;
          }

          case "importDesignInputs": {
            const pick = await vscode.window.showOpenDialog({ canSelectMany: false, filters: { "Design Inputs": ["csv","json"] }});
            if (!pick?.length) return;
            const result = await invoke("structai_design_import_inputs", { model_id: msg.model_id, path: pick[0].fsPath });
            view.webview.postMessage({ type: "importDesignDone", result });
            break;
          }

          case "runChecks": {
            const result = await invoke("structai_check_run", msg.input);
            view.webview.postMessage({ type: "runChecksDone", result });
            break;
          }

          case "loadCheckResults": {
            const result = await invoke("structai_check_get_results", { check_run_id: msg.check_run_id, status: msg.status || null, limit: 500 });
            view.webview.postMessage({ type: "checkResults", result });
            break;
          }
        }
      } catch (e:any) {
        view.webview.postMessage({ type: "error", message: String(e?.message ?? e) });
      }
    });
  }
}

4-4-3) media/results/main.js
(function () {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("app");

  let models = [];
  let runs = [];
  let checks = [];
  let checkResults = null;

  function h(tag, attrs = {}, children = []) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") el.className = v;
      else if (k === "onclick") el.onclick = v;
      else el.setAttribute(k, v);
    }
    for (const c of children) el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    return el;
  }

  function render() {
    app.innerHTML = "";
    app.appendChild(h("h2", {}, ["StructAI Results / Checks"]));

    const modelSelect = h("select", { id: "modelSel" }, models.map(m =>
      h("option", { value: String(m.model_id) }, [`${m.name} (#${m.model_id})`])
    ));

    const btnRow = h("div", { class: "row" }, [
      h("button", { onclick: () => vscode.postMessage({ type: "init" }) }, ["Refresh Models"]),
      h("button", { onclick: () => loadRuns() }, ["Load Runs/Checks"]),
      h("button", { onclick: () => importResults() }, ["Import Results"]),
      h("button", { onclick: () => importDesign() }, ["Import Design Inputs"]),
      h("button", { onclick: () => runChecks() }, ["Run Checks"]),
    ]);

    app.appendChild(h("div", { class: "row" }, [h("label", {}, ["Model: "]), modelSelect]));
    app.appendChild(btnRow);

    app.appendChild(h("h3", {}, ["Analysis Runs"]));
    app.appendChild(h("div", { class: "muted" }, [runs.length ? `Runs: ${runs.length}` : "No runs loaded."]));
    app.appendChild(h("ul", {}, runs.map(r => h("li", {}, [`#${r.analysis_run_id} ${r.name} (${r.engine||""} ${r.units||""})`]))));

    app.appendChild(h("h3", {}, ["Check Runs"]));
    const ul = h("ul", {}, []);
    for (const c of checks) {
      ul.appendChild(h("li", {}, [
        h("button", { onclick: () => loadCheck(c.check_run_id) }, [`Open #${c.check_run_id}`]),
        document.createTextNode(` ${c.name} (rule: ${c.rulepack_name||"builtin"}/${c.rulepack_version||""})`)
      ]));
    }
    app.appendChild(ul);

    app.appendChild(h("h3", {}, ["Check Results (top 200)"]));
    if (!checkResults) {
      app.appendChild(h("div", { class: "muted" }, ["No check results loaded."]));
    } else {
      const items = checkResults.items || [];
      const table = h("table", { class: "table" }, []);
      table.appendChild(h("tr", {}, [
        h("th", {}, ["Status"]),
        h("th", {}, ["UID"]),
        h("th", {}, ["Label"]),
        h("th", {}, ["Combo"]),
        h("th", {}, ["Type"]),
        h("th", {}, ["Ratio"]),
        h("th", {}, ["Citations"]),
      ]));
      for (const it of items.slice(0, 200)) {
        const cites = (it.citations || []).map(c => (c.page ? `${c.title||""} p.${c.page}` : (c.title||""))).join("; ");
        table.appendChild(h("tr", {}, [
          h("td", {}, [it.status]),
          h("td", {}, [it.member_uid]),
          h("td", {}, [it.member_label || ""]),
          h("td", {}, [it.combo]),
          h("td", {}, [it.check_type]),
          h("td", {}, [it.ratio != null ? Number(it.ratio).toFixed(3) : ""]),
          h("td", {}, [cites]),
        ]));
      }
      app.appendChild(table);
    }
  }

  function getModelId() {
    const el = document.getElementById("modelSel");
    return el ? Number(el.value) : null;
  }

  function loadRuns() {
    const model_id = getModelId();
    if (!model_id) return;
    vscode.postMessage({ type: "loadRuns", model_id });
  }

  function importResults() {
    const model_id = getModelId();
    if (!model_id) return;
    vscode.postMessage({ type: "importResults", model_id });
  }

  function importDesign() {
    const model_id = getModelId();
    if (!model_id) return;
    vscode.postMessage({ type: "importDesignInputs", model_id });
  }

  function runChecks() {
    const model_id = getModelId();
    if (!model_id) return;

    // 가장 최근 분석 run을 자동 선택(없으면 실패)
    const latest = runs[0];
    if (!latest) {
      alert("먼저 결과(analysis run)를 import 하세요.");
      return;
    }

    vscode.postMessage({
      type: "runChecks",
      input: {
        model_id,
        analysis_run_id: latest.analysis_run_id,
        name: `check_${Date.now()}`,
        // combos/check_types 생략하면 자동
        // only_mapped_from_cad_artifact_id: 123  // 필요하면 여기서 제한 가능
      }
    });
  }

  function loadCheck(check_run_id) {
    vscode.postMessage({ type: "loadCheckResults", check_run_id, status: null });
  }

  window.addEventListener("message", (ev) => {
    const msg = ev.data;
    if (msg.type === "models") {
      models = msg.models.items || [];
      render();
    } else if (msg.type === "runs") {
      runs = msg.runs.items || [];
      checks = msg.checks.items || [];
      render();
    } else if (msg.type === "checkResults") {
      checkResults = msg.result;
      render();
    } else if (msg.type === "importResultsDone" || msg.type === "importDesignDone" || msg.type === "runChecksDone") {
      // 완료 후 새로고침
      loadRuns();
    } else if (msg.type === "error") {
      console.error(msg.message);
      alert(msg.message);
    }
  });

  vscode.postMessage({ type: "init" });
})();

4-4-4) media/results/styles.css
body {
  font-family: var(--vscode-font-family);
  color: var(--vscode-foreground);
  background: var(--vscode-sideBar-background);
  padding: 10px;
}

.row {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  margin: 8px 0;
}

button, select {
  background: var(--vscode-button-background);
  color: var(--vscode-button-foreground);
  border: 1px solid var(--vscode-button-border, transparent);
  padding: 4px 8px;
  border-radius: 4px;
}

select {
  background: var(--vscode-input-background);
  color: var(--vscode-input-foreground);
  border: 1px solid var(--vscode-input-border);
}

.muted { opacity: 0.8; }

.table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 8px;
  font-size: 12px;
}
.table th, .table td {
  border-bottom: 1px solid rgba(255,255,255,0.08);
  padding: 6px;
  vertical-align: top;
  text-align: left;
}

4-5) 실제 사용 순서 (지금 단계 끝내는 “행동 순서”)

기준서 PDF 임포트 (근거 인용용)

#structai_import_pdf { "path": "…/KDS.pdf" }

DXF 임포트 (도면 라벨 후보/근거)

#structai_import_dxf { "path": "…/S-101.dxf" }

모델 임포트

#structai_model_import_members { "path": "…/model.csv" }

해석 결과 임포트

#structai_results_import { "model_id": 1, "path": "…/results.csv" }

설계입력(내력/허용치) 임포트

#structai_design_import_inputs { "model_id": 1, "path": "…/design_inputs.csv" }

체크 실행

#structai_check_run { "model_id": 1, "analysis_run_id": 1 }

결과 조회

#structai_check_get_results { "check_run_id": 1 }

(선택) 룰셋 적용

#structai_rules_import_rulepack { "path": "…/rulepack_kds_stub.json" }

#structai_rules_list

#structai_rules_set_active { "rulepack_id": 2 }

다시 structai_check_run 실행

4-6) 이 단계에서 “Claude Skills 같은 구조”가 성립되는 이유

도구 호출(임포트/검색/체크)은 결정론적이고 DB에 저장됨

체크 결과는 check_results에 **입력(설계입력) + 수요(해석결과) + 비율 + 상태 + 근거(PDF chunk/page)**가 같이 저장됨

따라서 채팅은 “계산”이 아니라:

어떤 부재/조합/체크를 실행할지 계획

누락(설계입력/결과/기준문서) 감지

결과 해설 + 링크 제시
를 담당하게 됨

다음(5단계)에서 바로 이어서 하면 제일 효과 큰 게 두 가지야:

“부재별 코드체크(예: RC 보/기둥) 결정론 엔진”을 실제 KDS 기반으로 확장

지금은 capacity(Mn/Vn/Pn)를 외부에서 주입하는 구조지만

다음 단계에서는 단면/재료/철근/설계계수 입력 → Mn/Vn 계산을 결정론으로 만들면 됨

리포트(계산서/검토서) 자동 생성

check_run_id 기반으로 부재별 표/근거/요약/불합격 Top10 등을 MD/PDF로 생성

원하면 바로 다음 응답에서 **5단계(실제 RC/강구조 체크 모듈화 + 보고서 생성 파이프라인)**도 같은 방식으로 순서대로 구현해줄게.
