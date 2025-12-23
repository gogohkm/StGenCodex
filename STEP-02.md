좋아. 2단계: DXF(도면) 파이프라인을 “바로 돌아가는 형태”로 구현해볼게.
이번 단계의 핵심은:

DXF에서 레이어/블록/치수/텍스트(및 블록 내부 텍스트) 추출

추출된 텍스트를 1단계에서 만든 ‘공통 메모리(DB+FTS)’에 넣어서
PDF/MD와 똑같이 검색/근거 인용/부재 후보 생성까지 가능하게 만들기

아래 순서대로 적용하면 돼.

2-0. 이번 단계에서 추가되는 것
✅ 기능

DXF 로딩 → TEXT / MTEXT / DIMENSION / (INSERT의 ATTRIB) 추출

INSERT 블록 안에 박혀있는 텍스트도 virtual_entities()로 추출 (중요)

도면에서 나온 텍스트들을 doc_chunks에 넣어 FTS 검색 통합

CAD 전용 테이블 cad_entities에 좌표/레이어/블록정보 저장

MCP 툴 추가:

structai_cad_get_summary

structai_cad_list_layers

structai_cad_list_text

structai_cad_extract_member_candidates

structai_cad_get_entity

virtual_entities()로 블록 내부 텍스트를 꺼내는 방식은 ezdxf 공식 문서에서도 “블록 레퍼런스 내용 조회” 방법으로 안내돼.

2-1. mcp_server/requirements.txt 업데이트

기존에 pypdf가 들어있을 텐데, 여기에 ezdxf를 추가해줘.

mcp>=1.0.0
pypdf>=5.0.0
ezdxf>=1.4.0


ezdxf는 DXF 읽기/엔티티 쿼리/MTEXT plain text/INSERT 속성/virtual entity 등을 제공해. 
ezdxf

2-2. mcp_server/schema.sql 업데이트 (스키마 0.0.3)

아래로 교체하는 걸 추천해. (idempotent라 기존 DB에도 안전)

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '0.0.3');
UPDATE meta SET value='0.0.3' WHERE key='schema_version';

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  uri TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,               -- pdf | md | dxf | (future: fem, etc.)
  file_hash TEXT,
  title TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS doc_chunks (
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  page_start INTEGER,               -- PDF pages; NULL for md/dxf
  page_end INTEGER,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_artifact ON doc_chunks(artifact_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_pages ON doc_chunks(artifact_id, page_start, page_end);

-- ✅ CAD(도면) 엔티티/텍스트 메타
CREATE TABLE IF NOT EXISTS cad_entities (
  cad_entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  chunk_id INTEGER REFERENCES doc_chunks(chunk_id) ON DELETE SET NULL,

  layout TEXT,                      -- "Model" or paperspace layout name
  is_virtual INTEGER NOT NULL DEFAULT 0, -- 1 if derived from INSERT.virtual_entities()

  handle TEXT,                      -- may be synthetic for virtual text
  dxf_type TEXT NOT NULL,           -- TEXT|MTEXT|DIMENSION|ATTRIB|MLEADER|...
  layer TEXT,
  block_name TEXT,
  tag TEXT,                         -- ATTRIB tag

  x REAL, y REAL, z REAL,
  rotation REAL,
  height REAL,
  measurement REAL,                 -- DIMENSION.get_measurement()

  text TEXT,
  raw_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_entities_artifact ON cad_entities(artifact_id);
CREATE INDEX IF NOT EXISTS idx_cad_entities_layer ON cad_entities(artifact_id, layer);
CREATE INDEX IF NOT EXISTS idx_cad_entities_type ON cad_entities(artifact_id, dxf_type);
CREATE INDEX IF NOT EXISTS idx_cad_entities_chunk ON cad_entities(chunk_id);

2-3. mcp_server/server.py 업데이트 (DXF 추출 + CAD 툴 추가 + FTS 안정화)

아래는 이전 단계(1단계) server.py에 바로 덮어쓸 수 있게 “통합본” 형태로 제공할게.
(※ 핵심 변경: _connect()에서 FTS를 미리 준비해서 “import 후 search 했는데 안 나오는 문제” 방지)

FTS는 “테이블을 나중에 만들면 기존 row가 인덱싱 안 되는” 케이스가 흔해서, 서버 시작 시점에 항상 준비시키는 게 안전해.

from __future__ import annotations

import os
import re
import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

ROOT = Path(os.environ.get("STRUCTAI_ROOT", str(Path.home() / ".structai"))).expanduser()
ROOT.mkdir(parents=True, exist_ok=True)

DB_PATH = ROOT / "structai.sqlite"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

mcp = FastMCP(
    name="structai-mcp",
    instructions=(
        "Structural AI MCP server. Stores PDFs/MD/DXF into a shared memory (SQLite + FTS5) "
        "and exposes tools for search & CAD extraction."
    ),
)

# ----------------------------
# Helpers
# ----------------------------

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _path_to_uri(path: Path) -> str:
    return path.resolve().as_uri()

def _infer_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".md", ".markdown", ".txt"):
        return "md"
    if ext == ".dxf":
        return "dxf"
    # fallback:
    return ext.lstrip(".") or "unknown"

def _chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    out: List[str] = []
    i = 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        out.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return out

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    _ensure_base_schema(conn)
    _ensure_fts(conn)  # ✅ always ensure FTS exists BEFORE imports
    return conn

def _ensure_base_schema(conn: sqlite3.Connection) -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()

def _fts_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='doc_chunks_fts' LIMIT 1"
    ).fetchone()
    return row is not None

def _ensure_fts(conn: sqlite3.Connection) -> None:
    existed = _fts_exists(conn)

    # External-content FTS5: doc_chunks is source of truth
    conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts
    USING fts5(
      content,
      uri UNINDEXED,
      artifact_kind UNINDEXED,
      page_start UNINDEXED,
      content='doc_chunks',
      content_rowid='chunk_id',
      tokenize='unicode61'
    );
    """)

    # Triggers to keep FTS in sync:
    conn.executescript("""
    CREATE TRIGGER IF NOT EXISTS doc_chunks_ai AFTER INSERT ON doc_chunks BEGIN
      INSERT INTO doc_chunks_fts(rowid, content, uri, artifact_kind, page_start)
      VALUES (
        new.chunk_id,
        new.content,
        (SELECT uri FROM artifacts WHERE artifact_id=new.artifact_id),
        (SELECT kind FROM artifacts WHERE artifact_id=new.artifact_id),
        new.page_start
      );
    END;

    CREATE TRIGGER IF NOT EXISTS doc_chunks_ad AFTER DELETE ON doc_chunks BEGIN
      INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content, uri, artifact_kind, page_start)
      VALUES('delete', old.chunk_id, old.content,
        (SELECT uri FROM artifacts WHERE artifact_id=old.artifact_id),
        (SELECT kind FROM artifacts WHERE artifact_id=old.artifact_id),
        old.page_start
      );
    END;

    CREATE TRIGGER IF NOT EXISTS doc_chunks_au AFTER UPDATE ON doc_chunks BEGIN
      INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content, uri, artifact_kind, page_start)
      VALUES('delete', old.chunk_id, old.content,
        (SELECT uri FROM artifacts WHERE artifact_id=old.artifact_id),
        (SELECT kind FROM artifacts WHERE artifact_id=old.artifact_id),
        old.page_start
      );
      INSERT INTO doc_chunks_fts(rowid, content, uri, artifact_kind, page_start)
      VALUES (
        new.chunk_id,
        new.content,
        (SELECT uri FROM artifacts WHERE artifact_id=new.artifact_id),
        (SELECT kind FROM artifacts WHERE artifact_id=new.artifact_id),
        new.page_start
      );
    END;
    """)

    # If FTS table was just created, rebuild from existing doc_chunks:
    if not existed:
        conn.execute("INSERT INTO doc_chunks_fts(doc_chunks_fts) VALUES('rebuild');")
    conn.commit()

def _upsert_artifact(
    conn: sqlite3.Connection,
    uri: str,
    kind: str,
    file_hash: Optional[str],
    title: Optional[str],
    meta_json: Dict[str, Any],
) -> int:
    meta_s = json.dumps(meta_json, ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO artifacts(uri, kind, file_hash, title, meta_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(uri) DO UPDATE SET
          kind=excluded.kind,
          file_hash=excluded.file_hash,
          title=excluded.title,
          meta_json=excluded.meta_json,
          updated_at=datetime('now')
        """,
        (uri, kind, file_hash, title, meta_s),
    )
    row = conn.execute("SELECT artifact_id FROM artifacts WHERE uri=?", (uri,)).fetchone()
    assert row is not None
    return int(row["artifact_id"])

def _delete_artifact_content(conn: sqlite3.Connection, artifact_id: int) -> None:
    # doc_chunks cascade to FTS via triggers
    conn.execute("DELETE FROM cad_entities WHERE artifact_id=?", (artifact_id,))
    conn.execute("DELETE FROM doc_chunks WHERE artifact_id=?", (artifact_id,))

def _resolve_artifact_id(conn: sqlite3.Connection, artifact: str | int) -> int:
    if isinstance(artifact, int):
        return artifact
    s = str(artifact).strip()
    if s.isdigit():
        return int(s)
    row = conn.execute("SELECT artifact_id FROM artifacts WHERE uri=?", (s,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown artifact reference: {artifact}")
    return int(row["artifact_id"])

# ----------------------------
# PDF extraction
# ----------------------------

def _extract_pdf_pages(path: Path) -> List[str]:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("pypdf is required for PDF import. pip install pypdf") from e

    reader = PdfReader(str(path))
    pages: List[str] = []
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t)
    return pages

# ----------------------------
# DXF extraction
# ----------------------------

def _vec_to_xyz(v: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if v is None:
        return None, None, None
    # ezdxf Vec3 supports x,y,z
    for attr in ("x", "y", "z"):
        if not hasattr(v, attr):
            return None, None, None
    try:
        return float(v.x), float(v.y), float(v.z)
    except Exception:
        return None, None, None

def _safe_dxf_get(entity: Any, attr: str, default: Any = None) -> Any:
    try:
        dxf = getattr(entity, "dxf", None)
        if dxf is None:
            return default
        if hasattr(dxf, "get"):
            return dxf.get(attr, default)
        return getattr(dxf, attr, default)
    except Exception:
        return default

def _entity_handle(entity: Any) -> Optional[str]:
    # DXF entities usually have dxf.handle
    h = _safe_dxf_get(entity, "handle", None)
    if h:
        return str(h)
    try:
        return str(entity.handle)
    except Exception:
        return None

def _extract_dxf(
    path: Path,
    include_virtual_text: bool = True,
    include_paperspace: bool = False,
    virtual_text_limit: int = 50000,
) -> Dict[str, Any]:
    """
    Extract text-bearing entities from DXF:
    - TEXT, MTEXT
    - DIMENSION (override text + measurement)
    - INSERT.ATTRIB (block attributes)
    - INSERT.virtual_entities() -> TEXT/MTEXT inside blocks (optional)
    - MULTILEADER/MLEADER mtext content (if present)
    """
    try:
        import ezdxf
    except Exception as e:
        raise RuntimeError("ezdxf is required for DXF import. pip install ezdxf") from e

    # robust loading: try normal readfile, fallback to recover.readfile
    try:
        doc = ezdxf.readfile(str(path))
        load_mode = "readfile"
    except Exception:
        # handle DXFStructureError & other structural problems:
        from ezdxf import recover
        doc, _auditor = recover.readfile(str(path))
        load_mode = "recover.readfile"  # recover module described in docs 

    layers_all = []
    try:
        layers_all = [layer.dxf.name for layer in doc.layers]
    except Exception:
        pass

    blocks_all = []
    try:
        for blk in doc.blocks:
            name = getattr(blk, "name", None)
            if name and not str(name).startswith("*"):
                blocks_all.append(str(name))
    except Exception:
        pass

    # choose layouts
    layouts: List[Tuple[str, Any]] = [("Model", doc.modelspace())]
    if include_paperspace:
        try:
            # doc.layout_names() documented in getting_data tutorial 
            for lname in doc.layout_names():
                if str(lname).lower() == "model":
                    continue
                try:
                    layouts.append((str(lname), doc.paperspace(str(lname))))
                except Exception:
                    continue
        except Exception:
            pass

    extracted: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    def bump(k: str, n: int = 1) -> None:
        counts[k] = counts.get(k, 0) + n

    def add_item(
        *,
        layout: str,
        is_virtual: int,
        handle: Optional[str],
        dxf_type: str,
        layer: Optional[str],
        text: str,
        x: Optional[float],
        y: Optional[float],
        z: Optional[float],
        rotation: Optional[float] = None,
        height: Optional[float] = None,
        measurement: Optional[float] = None,
        block_name: Optional[str] = None,
        tag: Optional[str] = None,
        raw: Optional[Dict[str, Any]] = None,
    ) -> None:
        t = (text or "").strip()
        if not t:
            return
        extracted.append(
            dict(
                layout=layout,
                is_virtual=int(is_virtual),
                handle=handle,
                dxf_type=dxf_type,
                layer=layer,
                block_name=block_name,
                tag=tag,
                x=x, y=y, z=z,
                rotation=rotation,
                height=height,
                measurement=measurement,
                text=t,
                raw_json=raw or {},
            )
        )
        bump(dxf_type, 1)

    virtual_text_count = 0
    virtual_text_truncated = False

    for layout_name, space in layouts:
        # TEXT
        for e in space.query("TEXT"):
            pt = _safe_dxf_get(e, "insert", None)
            x, y, z = _vec_to_xyz(pt)
            add_item(
                layout=layout_name,
                is_virtual=0,
                handle=_entity_handle(e),
                dxf_type="TEXT",
                layer=_safe_dxf_get(e, "layer", None),
                text=_safe_dxf_get(e, "text", ""),
                x=x, y=y, z=z,
                rotation=_safe_dxf_get(e, "rotation", None),
                height=_safe_dxf_get(e, "height", None),
            )

        # MTEXT
        for e in space.query("MTEXT"):
            pt = _safe_dxf_get(e, "insert", None)
            x, y, z = _vec_to_xyz(pt)
            # plain_text() is recommended by ezdxf for formatting-free text :contentReference[oaicite:4]{index=4}
            try:
                t = e.plain_text()
            except Exception:
                t = getattr(e, "text", "") or ""
            add_item(
                layout=layout_name,
                is_virtual=0,
                handle=_entity_handle(e),
                dxf_type="MTEXT",
                layer=_safe_dxf_get(e, "layer", None),
                text=t,
                x=x, y=y, z=z,
                rotation=_safe_dxf_get(e, "rotation", None),
                height=_safe_dxf_get(e, "char_height", None),
            )

        # DIMENSION: real measurement + override text 
        for e in space.query("DIMENSION"):
            override = _safe_dxf_get(e, "text", "") or ""
            try:
                meas = float(e.get_measurement())
            except Exception:
                meas = None

            # location (best-effort): text_midpoint exists in ezdxf index 
            pt = _safe_dxf_get(e, "text_midpoint", None)
            x, y, z = _vec_to_xyz(pt)

            # Build searchable text:
            ov = override.strip()
            if ov in ("<>",):
                ov = ""
            parts = []
            if ov:
                parts.append(ov)
            if meas is not None:
                parts.append(f"{meas}")
            text = " | ".join(parts)
            if not text and meas is not None:
                text = str(meas)

            add_item(
                layout=layout_name,
                is_virtual=0,
                handle=_entity_handle(e),
                dxf_type="DIMENSION",
                layer=_safe_dxf_get(e, "layer", None),
                text=text,
                x=x, y=y, z=z,
                measurement=meas,
                raw={"override": override, "dimtype": getattr(e, "dimtype", None)},
            )

        # MULTILEADER / MLEADER text content 
        for e in space.query("MLEADER MULTILEADER"):
            layer = _safe_dxf_get(e, "layer", None)
            # Prefer context.mtext.default_content
            ctx = getattr(e, "context", None)
            if ctx is not None:
                mtext = getattr(ctx, "mtext", None)
                if mtext is not None:
                    try:
                        pt = getattr(mtext, "insert", None)
                        x, y, z = _vec_to_xyz(pt)
                        content = getattr(mtext, "default_content", "") or ""
                        content = content.replace("\\P", "\n")
                        add_item(
                            layout=layout_name,
                            is_virtual=0,
                            handle=_entity_handle(e),
                            dxf_type="MLEADER",
                            layer=layer,
                            text=content,
                            x=x, y=y, z=z,
                        )
                    except Exception:
                        pass
            else:
                # fallback method exists: get_mtext_content() 
                try:
                    content = e.get_mtext_content() or ""
                    if content.strip():
                        add_item(
                            layout=layout_name,
                            is_virtual=0,
                            handle=_entity_handle(e),
                            dxf_type="MLEADER",
                            layer=layer,
                            text=content,
                            x=None, y=None, z=None,
                        )
                except Exception:
                    pass

        # INSERT + ATTRIB (block attributes) 
        for ins in space.query("INSERT"):
            ins_handle = _entity_handle(ins)
            ins_layer = _safe_dxf_get(ins, "layer", None)
            block_name = _safe_dxf_get(ins, "name", None)  # dxf.name is block name 

            # attached ATTRIB entities
            attribs = getattr(ins, "attribs", None) or []
            for a in attribs:
                pt = _safe_dxf_get(a, "insert", None)
                x, y, z = _vec_to_xyz(pt)
                add_item(
                    layout=layout_name,
                    is_virtual=0,
                    handle=_entity_handle(a) or (f"{ins_handle}:ATTR" if ins_handle else None),
                    dxf_type="ATTRIB",
                    layer=_safe_dxf_get(a, "layer", ins_layer),
                    text=_safe_dxf_get(a, "text", ""),
                    x=x, y=y, z=z,
                    rotation=_safe_dxf_get(a, "rotation", None),
                    height=_safe_dxf_get(a, "height", None),
                    block_name=block_name,
                    tag=_safe_dxf_get(a, "tag", None),
                    raw={"source_insert_handle": ins_handle},
                )

            # (optional) virtual entities: TEXT/MTEXT inside block reference 
            if include_virtual_text:
                try:
                    for ve in ins.virtual_entities():
                        if virtual_text_count >= virtual_text_limit:
                            virtual_text_truncated = True
                            break
                        vtype = ve.dxftype()
                        if vtype not in ("TEXT", "MTEXT"):
                            continue
                        inner_layer = _safe_dxf_get(ve, "layer", None)
                        # layer '0' inside block often inherits insert layer:
                        effective_layer = ins_layer if (inner_layer in (None, "", "0")) else inner_layer

                        if vtype == "TEXT":
                            txt = _safe_dxf_get(ve, "text", "") or ""
                            pt = _safe_dxf_get(ve, "insert", None)
                            x, y, z = _vec_to_xyz(pt)
                            add_item(
                                layout=layout_name,
                                is_virtual=1,
                                handle=f"{ins_handle}:V{virtual_text_count}" if ins_handle else f"V{virtual_text_count}",
                                dxf_type="TEXT",
                                layer=effective_layer,
                                text=txt,
                                x=x, y=y, z=z,
                                rotation=_safe_dxf_get(ve, "rotation", None),
                                height=_safe_dxf_get(ve, "height", None),
                                block_name=block_name,
                                raw={
                                    "source_insert_handle": ins_handle,
                                    "source_block_name": block_name,
                                    "inner_layer": inner_layer,
                                },
                            )
                            virtual_text_count += 1
                        elif vtype == "MTEXT":
                            try:
                                txt = ve.plain_text()
                            except Exception:
                                txt = getattr(ve, "text", "") or ""
                            pt = _safe_dxf_get(ve, "insert", None)
                            x, y, z = _vec_to_xyz(pt)
                            add_item(
                                layout=layout_name,
                                is_virtual=1,
                                handle=f"{ins_handle}:V{virtual_text_count}" if ins_handle else f"V{virtual_text_count}",
                                dxf_type="MTEXT",
                                layer=effective_layer,
                                text=txt,
                                x=x, y=y, z=z,
                                height=_safe_dxf_get(ve, "char_height", None),
                                block_name=block_name,
                                raw={
                                    "source_insert_handle": ins_handle,
                                    "source_block_name": block_name,
                                    "inner_layer": inner_layer,
                                },
                            )
                            virtual_text_count += 1
                    if virtual_text_truncated:
                        break
                except Exception:
                    # some INSERT may not support virtual_entities() if malformed
                    pass

    return dict(
        load_mode=load_mode,
        layers_all=layers_all,
        blocks_all=blocks_all,
        counts=counts,
        extracted=extracted,
        virtual_text_count=virtual_text_count,
        virtual_text_truncated=virtual_text_truncated,
        include_virtual_text=include_virtual_text,
        include_paperspace=include_paperspace,
    )

# ----------------------------
# MCP Tools
# ----------------------------

@mcp.tool(annotations=ToolAnnotations(title="Import files into StructAI memory"))
def structai_import_files(
    paths: List[str],
    dxf_include_virtual_text: bool = True,
    dxf_include_paperspace: bool = False,
    dxf_virtual_text_limit: int = 50000,
) -> Dict[str, Any]:
    """
    Import PDF/MD/DXF files.
    DXF: extracts text + layer/block/dimension info, stores into doc_chunks (search) + cad_entities (metadata).
    """
    conn = _connect()
    results: List[Dict[str, Any]] = []

    for p in paths:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            results.append({"path": p, "ok": False, "error": "file not found"})
            continue

        uri = _path_to_uri(path)
        kind = _infer_kind(path)
        file_hash = _file_sha256(path)

        try:
            if kind == "pdf":
                pages = _extract_pdf_pages(path)
                meta = {"pages": len(pages)}
                artifact_id = _upsert_artifact(conn, uri, kind, file_hash, path.name, meta)
                _delete_artifact_content(conn, artifact_id)

                chunk_index = 0
                for i, page_text in enumerate(pages, start=1):
                    # page-level chunks (keep page citations simple)
                    text = (page_text or "").strip()
                    if not text:
                        continue
                    conn.execute(
                        "INSERT INTO doc_chunks(artifact_id, page_start, page_end, chunk_index, content) VALUES (?, ?, ?, ?, ?)",
                        (artifact_id, i, i, chunk_index, text),
                    )
                    chunk_index += 1
                conn.commit()
                results.append({"path": str(path), "uri": uri, "kind": kind, "ok": True, "artifact_id": artifact_id})

            elif kind == "md":
                text = path.read_text(encoding="utf-8", errors="ignore")
                meta = {"chars": len(text)}
                artifact_id = _upsert_artifact(conn, uri, kind, file_hash, path.name, meta)
                _delete_artifact_content(conn, artifact_id)

                chunks = _chunk_text(text)
                for idx, c in enumerate(chunks):
                    conn.execute(
                        "INSERT INTO doc_chunks(artifact_id, page_start, page_end, chunk_index, content) VALUES (?, NULL, NULL, ?, ?)",
                        (artifact_id, idx, c),
                    )
                conn.commit()
                results.append({"path": str(path), "uri": uri, "kind": kind, "ok": True, "artifact_id": artifact_id})

            elif kind == "dxf":
                dxf = _extract_dxf(
                    path,
                    include_virtual_text=dxf_include_virtual_text,
                    include_paperspace=dxf_include_paperspace,
                    virtual_text_limit=dxf_virtual_text_limit,
                )
                meta = {
                    "load_mode": dxf["load_mode"],
                    "layers_all_count": len(dxf["layers_all"]),
                    "blocks_all_count": len(dxf["blocks_all"]),
                    "counts": dxf["counts"],
                    "virtual_text_count": dxf["virtual_text_count"],
                    "virtual_text_truncated": dxf["virtual_text_truncated"],
                    "include_virtual_text": dxf["include_virtual_text"],
                    "include_paperspace": dxf["include_paperspace"],
                }
                # keep lists too (optional but useful)
                meta["layers_all"] = dxf["layers_all"]
                meta["blocks_all"] = dxf["blocks_all"]

                artifact_id = _upsert_artifact(conn, uri, kind, file_hash, path.name, meta)
                _delete_artifact_content(conn, artifact_id)

                for idx, item in enumerate(dxf["extracted"]):
                    # One chunk per CAD entity text
                    content = item["text"]
                    conn.execute(
                        "INSERT INTO doc_chunks(artifact_id, page_start, page_end, chunk_index, content) VALUES (?, NULL, NULL, ?, ?)",
                        (artifact_id, idx, content),
                    )
                    chunk_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

                    conn.execute(
                        """
                        INSERT INTO cad_entities(
                          artifact_id, chunk_id, layout, is_virtual, handle, dxf_type, layer,
                          block_name, tag, x, y, z, rotation, height, measurement, text, raw_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            artifact_id,
                            chunk_id,
                            item.get("layout"),
                            int(item.get("is_virtual", 0)),
                            item.get("handle"),
                            item.get("dxf_type"),
                            item.get("layer"),
                            item.get("block_name"),
                            item.get("tag"),
                            item.get("x"), item.get("y"), item.get("z"),
                            item.get("rotation"),
                            item.get("height"),
                            item.get("measurement"),
                            item.get("text"),
                            json.dumps(item.get("raw_json", {}), ensure_ascii=False),
                        ),
                    )

                conn.commit()
                results.append({
                    "path": str(path),
                    "uri": uri,
                    "kind": kind,
                    "ok": True,
                    "artifact_id": artifact_id,
                    "cad_text_items": len(dxf["extracted"]),
                    "virtual_text_count": dxf["virtual_text_count"],
                    "virtual_text_truncated": dxf["virtual_text_truncated"],
                })

            else:
                results.append({"path": str(path), "uri": uri, "kind": kind, "ok": False, "error": "unsupported kind"})
        except Exception as e:
            results.append({"path": str(path), "uri": uri, "kind": kind, "ok": False, "error": str(e)})

    return {"ok": True, "results": results, "db": str(DB_PATH)}


@mcp.tool(annotations=ToolAnnotations(title="List imported artifacts"))
def structai_list_artifacts() -> Dict[str, Any]:
    conn = _connect()
    rows = conn.execute(
        "SELECT artifact_id, uri, kind, title, file_hash, meta_json, created_at, updated_at FROM artifacts ORDER BY updated_at DESC"
    ).fetchall()
    items = []
    for r in rows:
        items.append(
            dict(
                artifact_id=int(r["artifact_id"]),
                uri=r["uri"],
                kind=r["kind"],
                title=r["title"],
                file_hash=r["file_hash"],
                meta_json=json.loads(r["meta_json"] or "{}"),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
        )
    return {"count": len(items), "items": items}


@mcp.tool(annotations=ToolAnnotations(title="Search memory (PDF/MD/DXF)"))
def structai_search_knowledge(query: str, limit: int = 8) -> Dict[str, Any]:
    conn = _connect()

    # Join FTS -> doc_chunks -> artifacts, left join cad_entities to enrich DXF hits
    rows = conn.execute(
        """
        SELECT
          a.uri, a.kind,
          d.chunk_id, d.page_start, d.page_end,
          snippet(doc_chunks_fts, 0, '[', ']', '…', 12) AS snippet,
          bm25(doc_chunks_fts) AS score,
          c.cad_entity_id, c.dxf_type AS cad_type, c.layer AS cad_layer, c.x AS cad_x, c.y AS cad_y,
          c.layout AS cad_layout, c.block_name AS cad_block, c.tag AS cad_tag, c.is_virtual AS cad_is_virtual
        FROM doc_chunks_fts
        JOIN doc_chunks d ON d.chunk_id = doc_chunks_fts.rowid
        JOIN artifacts a ON a.artifact_id = d.artifact_id
        LEFT JOIN cad_entities c ON c.chunk_id = d.chunk_id
        WHERE doc_chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()

    items = []
    for r in rows:
        kind = r["kind"]
        uri = r["uri"]
        chunk_id = int(r["chunk_id"])
        page_start = r["page_start"]

        if kind == "pdf" and page_start is not None:
            cite_uri = f"{uri}#page={int(page_start)}"
        else:
            cite_uri = f"{uri}#chunk={chunk_id}"

        item = dict(
            uri=uri,
            kind=kind,
            chunk_id=chunk_id,
            page_start=r["page_start"],
            page_end=r["page_end"],
            snippet=r["snippet"],
            score=float(r["score"]),
            cite_uri=cite_uri,
        )
        # DXF enrichment
        if r["cad_entity_id"] is not None:
            item["cad"] = dict(
                cad_entity_id=int(r["cad_entity_id"]),
                type=r["cad_type"],
                layer=r["cad_layer"],
                x=r["cad_x"],
                y=r["cad_y"],
                layout=r["cad_layout"],
                block=r["cad_block"],
                tag=r["cad_tag"],
                is_virtual=bool(r["cad_is_virtual"]),
            )
        items.append(item)

    return {"query": query, "count": len(items), "items": items}


@mcp.tool(annotations=ToolAnnotations(title="Reset project memory (dangerous)"))
def structai_reset_project() -> Dict[str, Any]:
    conn = _connect()
    conn.execute("DELETE FROM cad_entities;")
    conn.execute("DELETE FROM doc_chunks;")
    conn.execute("DELETE FROM artifacts;")
    conn.commit()
    return {"ok": True, "db": str(DB_PATH)}


# ----------------------------
# CAD Tools
# ----------------------------

@mcp.tool(annotations=ToolAnnotations(title="CAD: get summary for a DXF artifact"))
def structai_cad_get_summary(artifact: str | int, top_layers: int = 20) -> Dict[str, Any]:
    conn = _connect()
    artifact_id = _resolve_artifact_id(conn, artifact)

    a = conn.execute("SELECT artifact_id, uri, title, meta_json FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
    if a is None:
        raise ValueError("artifact not found")
    meta = json.loads(a["meta_json"] or "{}")

    layer_rows = conn.execute(
        """
        SELECT layer, COUNT(*) as n
        FROM cad_entities
        WHERE artifact_id=?
        GROUP BY layer
        ORDER BY n DESC
        LIMIT ?
        """,
        (artifact_id, top_layers),
    ).fetchall()

    type_rows = conn.execute(
        """
        SELECT dxf_type, COUNT(*) as n
        FROM cad_entities
        WHERE artifact_id=?
        GROUP BY dxf_type
        ORDER BY n DESC
        """,
        (artifact_id,),
    ).fetchall()

    return {
        "artifact_id": int(a["artifact_id"]),
        "uri": a["uri"],
        "title": a["title"],
        "meta": meta,
        "top_layers": [{"layer": r["layer"], "count": int(r["n"])} for r in layer_rows],
        "type_counts": [{"type": r["dxf_type"], "count": int(r["n"])} for r in type_rows],
    }


@mcp.tool(annotations=ToolAnnotations(title="CAD: list layers (with extracted text counts)"))
def structai_cad_list_layers(artifact: str | int, include_empty: bool = True) -> Dict[str, Any]:
    conn = _connect()
    artifact_id = _resolve_artifact_id(conn, artifact)
    a = conn.execute("SELECT uri, meta_json FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
    if a is None:
        raise ValueError("artifact not found")
    meta = json.loads(a["meta_json"] or "{}")
    all_layers = meta.get("layers_all", []) if include_empty else []

    counts = conn.execute(
        """
        SELECT layer, COUNT(*) AS n
        FROM cad_entities
        WHERE artifact_id=?
        GROUP BY layer
        """,
        (artifact_id,),
    ).fetchall()
    count_map = {r["layer"]: int(r["n"]) for r in counts}

    if include_empty and all_layers:
        items = [{"layer": L, "count": int(count_map.get(L, 0))} for L in all_layers]
        items.sort(key=lambda x: x["count"], reverse=True)
    else:
        items = [{"layer": k, "count": v} for k, v in sorted(count_map.items(), key=lambda kv: kv[1], reverse=True)]

    return {"artifact_id": artifact_id, "uri": a["uri"], "count": len(items), "items": items}


@mcp.tool(annotations=ToolAnnotations(title="CAD: list extracted text entities"))
def structai_cad_list_text(
    artifact: str | int,
    layer: Optional[str] = None,
    contains: Optional[str] = None,
    dxf_types: Optional[List[str]] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    conn = _connect()
    artifact_id = _resolve_artifact_id(conn, artifact)
    a = conn.execute("SELECT uri FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
    if a is None:
        raise ValueError("artifact not found")
    uri = a["uri"]

    sql = """
    SELECT cad_entity_id, chunk_id, layout, is_virtual, handle, dxf_type, layer, block_name, tag, x, y, z, text
    FROM cad_entities
    WHERE artifact_id=?
      AND text IS NOT NULL
      AND trim(text) != ''
    """
    params: List[Any] = [artifact_id]

    if layer:
        sql += " AND layer=?"
        params.append(layer)

    if dxf_types:
        sql += " AND dxf_type IN (%s)" % ",".join(["?"] * len(dxf_types))
        params.extend(dxf_types)

    if contains:
        sql += " AND instr(text, ?) > 0"
        params.append(contains)

    sql += " ORDER BY cad_entity_id ASC LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    items = []
    for r in rows:
        cad_id = int(r["cad_entity_id"])
        items.append(
            dict(
                cad_entity_id=cad_id,
                chunk_id=int(r["chunk_id"]) if r["chunk_id"] is not None else None,
                layout=r["layout"],
                is_virtual=bool(r["is_virtual"]),
                handle=r["handle"],
                dxf_type=r["dxf_type"],
                layer=r["layer"],
                block_name=r["block_name"],
                tag=r["tag"],
                x=r["x"],
                y=r["y"],
                z=r["z"],
                text=r["text"],
                cite_uri=f"{uri}#cad_entity={cad_id}",
            )
        )

    return {"artifact_id": artifact_id, "uri": uri, "count": len(items), "items": items}


@mcp.tool(annotations=ToolAnnotations(title="CAD: get one cad entity"))
def structai_cad_get_entity(cad_entity_id: int) -> Dict[str, Any]:
    conn = _connect()
    r = conn.execute(
        """
        SELECT c.*, a.uri
        FROM cad_entities c
        JOIN artifacts a ON a.artifact_id=c.artifact_id
        WHERE c.cad_entity_id=?
        """,
        (cad_entity_id,),
    ).fetchone()
    if r is None:
        raise ValueError("cad_entity not found")
    return {
        "cad_entity_id": int(r["cad_entity_id"]),
        "artifact_id": int(r["artifact_id"]),
        "uri": r["uri"],
        "chunk_id": int(r["chunk_id"]) if r["chunk_id"] is not None else None,
        "layout": r["layout"],
        "is_virtual": bool(r["is_virtual"]),
        "handle": r["handle"],
        "dxf_type": r["dxf_type"],
        "layer": r["layer"],
        "block_name": r["block_name"],
        "tag": r["tag"],
        "x": r["x"], "y": r["y"], "z": r["z"],
        "rotation": r["rotation"],
        "height": r["height"],
        "measurement": r["measurement"],
        "text": r["text"],
        "raw_json": json.loads(r["raw_json"] or "{}"),
        "cite_uri": f"{r['uri']}#cad_entity={int(r['cad_entity_id'])}",
    }


_DEFAULT_TOKEN_RE = re.compile(r"\b[A-Z]{1,4}[-_]?\d{1,4}\b")

def _guess_member_type(token: str) -> str:
    # very rough heuristic – later 단계에서 프로젝트별 규칙으로 강화
    t = token.upper()
    if t.startswith("C"):
        return "column"
    if t.startswith(("B", "G")):
        return "beam"
    if t.startswith(("W",)):
        return "wall"
    if t.startswith(("BR", "K")):
        return "brace"
    return "unknown"


@mcp.tool(annotations=ToolAnnotations(title="CAD: extract member label candidates"))
def structai_cad_extract_member_candidates(
    artifact: str | int,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Heuristically extract member label candidates from CAD texts.
    Returns aggregated tokens with occurrences.
    """
    conn = _connect()
    artifact_id = _resolve_artifact_id(conn, artifact)
    a = conn.execute("SELECT uri FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
    if a is None:
        raise ValueError("artifact not found")
    uri = a["uri"]

    rows = conn.execute(
        """
        SELECT cad_entity_id, layer, dxf_type, x, y, text
        FROM cad_entities
        WHERE artifact_id=?
          AND text IS NOT NULL
          AND trim(text) != ''
        """,
        (artifact_id,),
    ).fetchall()

    agg: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        text = str(r["text"])
        for tok in _DEFAULT_TOKEN_RE.findall(text.upper()):
            # filter out common non-member tokens if needed (later: config)
            if tok.isdigit():
                continue
            entry = agg.get(tok)
            if entry is None:
                entry = {
                    "token": tok,
                    "member_type_guess": _guess_member_type(tok),
                    "occurrences": [],
                }
                agg[tok] = entry
            entry["occurrences"].append({
                "cad_entity_id": int(r["cad_entity_id"]),
                "layer": r["layer"],
                "dxf_type": r["dxf_type"],
                "x": r["x"],
                "y": r["y"],
                "cite_uri": f"{uri}#cad_entity={int(r['cad_entity_id'])}",
            })

    items = list(agg.values())
    items.sort(key=lambda e: len(e["occurrences"]), reverse=True)
    items = items[: max(0, int(limit))]

    return {"artifact_id": artifact_id, "uri": uri, "count": len(items), "items": items}


if __name__ == "__main__":
    mcp.run()

DXF 관련 구현에서 참고한 ezdxf 포인트

DXF 로딩/모델스페이스/엔티티 쿼리/레이아웃 개념:

INSERT의 dxf.name, attribs, get_attrib_text(), virtual_entities() 등:

MTEXT plain_text() 사용: 
ezdxf

DIMENSION 측정치 추출 get_measurement():

MULTILEADER 텍스트/컨텍스트:

손상 DXF 복구 로더 recover.readfile():

2-4. 사용 예시 (VS Code 채팅에서 MCP 툴 호출)
1) DXF 가져오기
#structai_import_files {
  "paths": ["C:/proj/struct/drawings/S-101.dxf"],
  "dxf_include_virtual_text": true,
  "dxf_include_paperspace": false
}

2) 레이어 요약 보기
#structai_cad_list_layers { "artifact": "file:///C:/proj/struct/drawings/S-101.dxf" }

3) 도면에서 “C1” 같은 라벨 검색
#structai_search_knowledge { "query": "C1", "limit": 8 }

4) 부재 라벨 후보 자동 추출
#structai_cad_extract_member_candidates {
  "artifact": "file:///C:/proj/struct/drawings/S-101.dxf",
  "limit": 50
}

5) 특정 레이어에서 텍스트 리스트
#structai_cad_list_text {
  "artifact": "file:///C:/proj/struct/drawings/S-101.dxf",
  "layer": "S-COL-TEXT",
  "limit": 200
}

2-5. 이 단계 구현의 “의미” (다음 단계 연결)

이제부터는 AI가:

PDF(기준/설계서), MD(메모/설계노트), **DXF(도면 텍스트/부재기호)**를
전부 같은 검색/근거 시스템으로 접근할 수 있어.

즉, 다음 단계(3단계)에서 할 일이 명확해져:

도면 부재기호(C1, B12, G3 등) ↔ 해석모델 부재ID 자동 매핑

그리고 매핑된 부재에 대해 기준/도면/해석결과를 함께 가져와서
“클로드 스킬처럼” 대화형 구조안전성 평가로 확장

원하면 다음 응답에서는 **3단계(도면↔해석모델 자동 매핑)**로 바로 넘어가서,

member_map 테이블 설계

“도면 라벨 후보 + 모델 부재/단면/좌표”를 이용한 매칭 알고리즘(거리/레이어/규칙 기반)

매핑 결과를 VS Code 오른쪽 패널에서 수정/확정하는 UI 흐름

까지 순서대로 구현해줄게.
