# mcp_server/server.py
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

from mcp.server.fastmcp import FastMCP

from mcp_server.design.rc import RcBeamRectInputs, RcColumnAxialInputs, rc_beam_rect_capacity, rc_column_axial_capacity
from mcp_server.design.steel import SteelBeamInputs, steel_beam_capacity
from mcp_server.design.steel_props import SteelHSectionDims, compute_h_section_props
from mcp_server.design.steel_props_more import (
    SteelBoxDims,
    compute_box_props,
    SteelPipeDims,
    compute_pipe_props,
    SteelChannelDims,
    compute_channel_props,
    SteelAngleDims,
    compute_angle_props,
)
from mcp_server.parsing.specs import parse_specs_from_text
from mcp_server.parsing.story import extract_story_candidates, normalize_story, story_index
from mcp_server.parsing.table_schema import infer_schema, parse_rows
from mcp_server.parsing.table_schema_v2 import infer_schema_v2, parse_rows_v2
from mcp_server.parsing.tables import TextPoint, extract_grid_tables
from mcp_server.reporting.md import build_markdown_report
from mcp_server.reporting.pdf import build_pdf_report

ROOT = Path(os.environ.get("STRUCTAI_ROOT", str(Path.cwd() / ".structai"))).resolve()
DB_PATH = Path(os.environ.get("STRUCTAI_DB", str(ROOT / "structai.sqlite")))
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DB_PATH_OVERRIDE: Optional[Path] = None

try:
    mcp = FastMCP("structai-mcp", version="0.1.4")
except TypeError:
    # Older MCP SDKs do not accept a version kwarg.
    mcp = FastMCP("structai-mcp")

_VAR_RX = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _apply_schema_if_needed(db_path: Path) -> None:
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        sql = SCHEMA_PATH.read_text(encoding="utf-8", errors="ignore")
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()


def _fts_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1 FROM doc_chunks_fts LIMIT 1")
        return True
    except sqlite3.OperationalError:
        return False


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    if _fts_available(conn):
        return True
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts
            USING fts5(
              content,
              content='doc_chunks',
              content_rowid='chunk_id',
              tokenize='unicode61'
            )
            """
        )
        conn.executescript(
            """
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
            """
        )
        conn.commit()
        return True
    except sqlite3.OperationalError:
        return False


def _connect() -> sqlite3.Connection:
    path = DB_PATH_OVERRIDE or DB_PATH
    _apply_schema_if_needed(path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_fts(conn)
    return conn


@contextmanager
def _db_override(db_path: Path):
    global DB_PATH_OVERRIDE
    prev = DB_PATH_OVERRIDE
    DB_PATH_OVERRIDE = db_path
    try:
        yield
    finally:
        DB_PATH_OVERRIDE = prev


def _resolve(obj: Any, vars: Dict[str, Any]) -> Any:
    if isinstance(obj, str):
        def repl(m: re.Match) -> str:
            key = m.group(1)
            if key not in vars:
                raise KeyError(f"fixture var not found: {key}")
            return str(vars[key])
        return _VAR_RX.sub(repl, obj)
    if isinstance(obj, list):
        return [_resolve(x, vars) for x in obj]
    if isinstance(obj, dict):
        return {k: _resolve(v, vars) for k, v in obj.items()}
    return obj


def _get_by_path(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            raise KeyError(f"output path not found: {path}")
    return cur


def _tool_registry() -> Dict[str, Any]:
    reg: Dict[str, Any] = {}
    for name, obj in globals().items():
        if callable(obj) and name.startswith("structai_"):
            reg[name] = obj
    return reg


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().upper()).replace("-", "").replace("_", "")


def _norm_story_model(story: str) -> str:
    s = (story or "").strip().upper().replace(" ", "")
    if s.endswith("F") or s.startswith("B") or s == "RF":
        return s
    if s.isdigit():
        return f"{int(s)}F"
    return s


def _actor_from_env() -> str:
    return (
        os.environ.get("STRUCTAI_ACTOR")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or "unknown"
    )


def _safe_eval(expr: str, env: Dict[str, Any]) -> Optional[float]:
    if not expr:
        return None

    allowed_funcs = {"abs": abs, "max": max, "min": min, "round": round}
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Load,
        ast.Name,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError("expression contains disallowed nodes")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs:
                raise ValueError("expression contains disallowed function")

    compiled = compile(tree, "<expr>", "eval")
    safe_env = {**allowed_funcs, **env}
    return float(eval(compiled, {"__builtins__": {}}, safe_env))


def extract_names(expr: str) -> List[str]:
    if not expr:
        return []
    tree = ast.parse(expr, mode="eval")
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
    return [n for n in names if n not in ("abs", "max", "min", "round")]


def _status_from_ratio(ratio: Optional[float], limit: float, warn: float) -> str:
    if ratio is None:
        return "NA"
    try:
        r = float(ratio)
    except Exception:
        return "NA"
    if r <= warn:
        return "PASS"
    if r <= limit:
        return "WARN"
    return "FAIL"

BUILTIN_RULEPACK = {
    "name": "builtin-generic",
    "version": "0.2",
    "checks": {
        "strength.flexure": {
            "ratio_expr": "max(abs(M3_max)/Mn_pos, abs(M3_min)/Mn_neg)",
            "limit": 1.0,
            "warn": 0.95,
            "citations": [{"query": "flexure design", "kind": "pdf", "note": "update with code clause"}],
        },
        "strength.shear": {
            "ratio_expr": "abs(V2_max)/Vn",
            "limit": 1.0,
            "warn": 0.95,
            "citations": [{"query": "shear strength", "kind": "pdf", "note": "update with code clause"}],
        },
        "strength.axial": {
            "ratio_expr": "abs(N_max)/Pn",
            "limit": 1.0,
            "warn": 0.95,
            "citations": [{"query": "axial strength", "kind": "pdf", "note": "update with code clause"}],
        },
        "service.deflection": {
            "ratio_expr": "abs(D_max)/D_allow",
            "limit": 1.0,
            "warn": 0.9,
            "citations": [{"query": "deflection limit", "kind": "pdf", "note": "update with code clause"}],
        },
        "service.drift": {
            "ratio_expr": "abs(drift_max)/drift_allow",
            "limit": 1.0,
            "warn": 0.9,
            "citations": [{"query": "drift limit", "kind": "pdf", "note": "update with code clause"}],
        },
    },
}


def _get_active_rulepack(conn: sqlite3.Connection) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT rulepack_json FROM rulepacks WHERE is_active=1 ORDER BY rulepack_id DESC LIMIT 1"
    ).fetchone()
    if row:
        try:
            return json.loads(row["rulepack_json"] or "{}")
        except Exception:
            return BUILTIN_RULEPACK
    return BUILTIN_RULEPACK


def _get_active_codebook(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT codebook_json FROM codebooks WHERE is_active=1 ORDER BY codebook_id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["codebook_json"] or "{}")
    except Exception:
        return None


def _get_effective_qa_profile(conn: sqlite3.Connection, model_id: Optional[int]) -> Optional[Dict[str, Any]]:
    profile_row = None
    if model_id is not None:
        profile_row = conn.execute(
            """
            SELECT qp.profile_json
            FROM model_qa_profiles mp
            JOIN qa_profiles qp ON qp.qa_profile_id = mp.qa_profile_id
            WHERE mp.model_id=?
            """,
            (int(model_id),),
        ).fetchone()
    if not profile_row:
        profile_row = conn.execute(
            "SELECT profile_json FROM qa_profiles WHERE is_active=1 ORDER BY qa_profile_id DESC LIMIT 1"
        ).fetchone()
    if not profile_row:
        return None
    try:
        return json.loads(profile_row["profile_json"] or "{}")
    except Exception:
        return None


def _upsert_artifact(conn: sqlite3.Connection, uri: str, kind: str, title: Optional[str], source_path: str, sha256: str) -> int:
    row = conn.execute("SELECT artifact_id FROM artifacts WHERE uri=?", (uri,)).fetchone()
    if row:
        conn.execute(
            """
            UPDATE artifacts SET kind=?, title=?, source_path=?, sha256=?, updated_at=datetime('now')
            WHERE artifact_id=?
            """,
            (kind, title, source_path, sha256, int(row["artifact_id"])),
        )
        return int(row["artifact_id"])
    cur = conn.execute(
        """
        INSERT INTO artifacts(uri, kind, title, source_path, sha256, created_at, updated_at)
        VALUES(?,?,?,?,?, datetime('now'), datetime('now'))
        """,
        (uri, kind, title, source_path, sha256),
    )
    return int(cur.lastrowid)


def _chunk_text(text: str, max_len: int = 1200, overlap: int = 200) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    chunks = []
    i = 0
    while i < len(t):
        chunk = t[i:i + max_len].strip()
        if chunk:
            chunks.append(chunk)
        i += max_len - overlap
    return chunks


def _insert_doc_chunks(conn: sqlite3.Connection, artifact_id: int, chunks: List[str], page: Optional[int]) -> None:
    for idx, content in enumerate(chunks):
        conn.execute(
            """
            INSERT INTO doc_chunks(artifact_id, page_start, page_end, chunk_index, content)
            VALUES(?,?,?,?,?)
            """,
            (int(artifact_id), page, page, int(idx), content),
        )


def _resolve_path(path_or_uri: str) -> Path:
    p = Path(path_or_uri)
    if p.exists():
        return p.expanduser().resolve()
    if path_or_uri.startswith("file://"):
        parsed = urlparse(path_or_uri)
        return Path(unquote(parsed.path)).resolve()
    return Path(path_or_uri).expanduser().resolve()


def _search_docs(conn: sqlite3.Connection, query: str, limit: int = 8, kind_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    hits: List[Dict[str, Any]] = []
    fts = _fts_available(conn)

    if fts:
        sql = """
        SELECT dc.chunk_id, dc.content, dc.page_start, dc.page_end, a.uri, a.kind,
               snippet(doc_chunks_fts, 0, '[', ']', '...', 8) AS snip
        FROM doc_chunks_fts
        JOIN doc_chunks dc ON dc.chunk_id = doc_chunks_fts.rowid
        JOIN artifacts a ON a.artifact_id = dc.artifact_id
        WHERE doc_chunks_fts MATCH ?
        """
        params: List[Any] = [q]
        if kind_filter:
            sql += " AND a.kind=?"
            params.append(kind_filter)
        sql += " LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
    else:
        sql = """
        SELECT dc.chunk_id, dc.content, dc.page_start, dc.page_end, a.uri, a.kind
        FROM doc_chunks dc
        JOIN artifacts a ON a.artifact_id = dc.artifact_id
        WHERE dc.content LIKE ?
        """
        params = [f"%{q}%"]
        if kind_filter:
            sql += " AND a.kind=?"
            params.append(kind_filter)
        sql += " LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()

    for r in rows:
        snippet = r["snip"] if "snip" in r.keys() else (r["content"] or "")[:240]
        hits.append(
            {
                "chunk_id": int(r["chunk_id"]),
                "uri": r["uri"],
                "kind": r["kind"],
                "page": r["page_start"],
                "snippet": snippet,
            }
        )
    return hits


def _code_citations(conn: sqlite3.Connection, citations: List[Dict[str, Any]], limit_each: int = 1) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for c in citations:
        query = str(c.get("query") or "").strip()
        if not query:
            continue
        kind = c.get("kind")
        rows = _search_docs(conn, query, limit=limit_each, kind_filter=kind)
        for r in rows:
            hits.append(
                {
                    "query": query,
                    "uri": r.get("uri"),
                    "page": r.get("page"),
                    "snippet": r.get("snippet"),
                    "kind": r.get("kind"),
                }
            )
    return hits


def enrich_envelope(env: Dict[str, Any]) -> Dict[str, Any]:
    return env or {}


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
    return ext.lstrip(".") or "unknown"


def _delete_artifact_content(conn: sqlite3.Connection, artifact_id: int) -> None:
    conn.execute("DELETE FROM cad_entities WHERE artifact_id=?", (int(artifact_id),))
    conn.execute("DELETE FROM doc_chunks WHERE artifact_id=?", (int(artifact_id),))


def _resolve_artifact_id(conn: sqlite3.Connection, artifact: Any) -> int:
    if isinstance(artifact, int):
        return int(artifact)
    s = str(artifact).strip()
    if s.isdigit():
        return int(s)
    row = conn.execute("SELECT artifact_id FROM artifacts WHERE uri=?", (s,)).fetchone()
    if not row:
        raise ValueError("unknown artifact reference")
    return int(row["artifact_id"])


def _extract_pdf_pages(path: Path) -> List[str]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("pypdf is required for PDF import") from exc

    reader = PdfReader(str(path))
    pages: List[str] = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return pages


def _extract_dxf(path: Path, include_virtual_text: bool = True) -> Dict[str, Any]:
    try:
        import ezdxf
    except Exception as exc:
        raise RuntimeError("ezdxf is required for DXF import") from exc

    try:
        doc = ezdxf.readfile(str(path))
    except Exception:
        from ezdxf import recover
        doc, _ = recover.readfile(str(path))

    msp = doc.modelspace()
    items: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    def bump(kind: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1

    def add_item(kind: str, layer: Optional[str], text: Optional[str], x: Optional[float], y: Optional[float], z: Optional[float],
                 handle: Optional[str] = None, geom: Optional[Dict[str, Any]] = None, raw: Optional[Dict[str, Any]] = None) -> None:
        items.append({
            "type": kind,
            "layer": layer,
            "text": text,
            "x": x,
            "y": y,
            "z": z,
            "handle": handle,
            "geom_json": geom or None,
            "raw_json": raw or {},
        })
        bump(kind)

    for e in msp.query("TEXT"):
        pt = getattr(e.dxf, "insert", None)
        add_item(
            "TEXT",
            getattr(e.dxf, "layer", None),
            getattr(e.dxf, "text", None),
            float(pt.x) if pt is not None else None,
            float(pt.y) if pt is not None else None,
            float(pt.z) if pt is not None else None,
            handle=getattr(e.dxf, "handle", None),
        )

    for e in msp.query("MTEXT"):
        pt = getattr(e.dxf, "insert", None)
        try:
            text = e.plain_text()
        except Exception:
            text = getattr(e, "text", None)
        add_item(
            "MTEXT",
            getattr(e.dxf, "layer", None),
            text,
            float(pt.x) if pt is not None else None,
            float(pt.y) if pt is not None else None,
            float(pt.z) if pt is not None else None,
            handle=getattr(e.dxf, "handle", None),
        )

    for e in msp.query("DIMENSION"):
        try:
            meas = float(e.get_measurement())
        except Exception:
            meas = None
        override = (getattr(e.dxf, "text", "") or "").strip()
        if override in ("<>", ""):
            override = ""
        text = override
        if meas is not None:
            text = f"{override} {meas}".strip()
        pt = getattr(e.dxf, "text_midpoint", None)
        add_item(
            "DIMENSION",
            getattr(e.dxf, "layer", None),
            text,
            float(pt.x) if pt is not None else None,
            float(pt.y) if pt is not None else None,
            float(pt.z) if pt is not None else None,
            handle=getattr(e.dxf, "handle", None),
            raw={"measurement": meas},
        )

    for ins in msp.query("INSERT"):
        ins_layer = getattr(ins.dxf, "layer", None)
        ins_handle = getattr(ins.dxf, "handle", None)
        for a in getattr(ins, "attribs", []) or []:
            pt = getattr(a.dxf, "insert", None)
            add_item(
                "ATTRIB",
                getattr(a.dxf, "layer", None) or ins_layer,
                getattr(a.dxf, "text", None),
                float(pt.x) if pt is not None else None,
                float(pt.y) if pt is not None else None,
                float(pt.z) if pt is not None else None,
                handle=getattr(a.dxf, "handle", None) or ins_handle,
                raw={"tag": getattr(a.dxf, "tag", None), "insert_handle": ins_handle},
            )

        if include_virtual_text:
            try:
                for ve in ins.virtual_entities():
                    vtype = ve.dxftype()
                    if vtype not in ("TEXT", "MTEXT"):
                        continue
                    pt = getattr(ve.dxf, "insert", None)
                    if vtype == "TEXT":
                        vtext = getattr(ve.dxf, "text", None)
                    else:
                        try:
                            vtext = ve.plain_text()
                        except Exception:
                            vtext = getattr(ve, "text", None)
                    add_item(
                        vtype,
                        getattr(ve.dxf, "layer", None) or ins_layer,
                        vtext,
                        float(pt.x) if pt is not None else None,
                        float(pt.y) if pt is not None else None,
                        float(pt.z) if pt is not None else None,
                        handle=str(ins_handle) + ":V" if ins_handle else None,
                        raw={"insert_handle": ins_handle, "virtual": True},
                    )
            except Exception:
                pass

    for e in msp.query("LINE"):
        try:
            s = e.dxf.start
            t = e.dxf.end
            pts = [[float(s.x), float(s.y)], [float(t.x), float(t.y)]]
            add_item(
                "LINE",
                getattr(e.dxf, "layer", None),
                None,
                float(s.x),
                float(s.y),
                float(s.z) if hasattr(s, "z") else None,
                handle=getattr(e.dxf, "handle", None),
                geom={"points": pts},
                raw={"linetype": getattr(e.dxf, "linetype", None)},
            )
        except Exception:
            continue

    for e in msp.query("LWPOLYLINE"):
        try:
            pts = [[float(p[0]), float(p[1])] for p in e.get_points()]
            if not pts:
                continue
            add_item(
                "LWPOLYLINE",
                getattr(e.dxf, "layer", None),
                None,
                float(pts[0][0]),
                float(pts[0][1]),
                None,
                handle=getattr(e.dxf, "handle", None),
                geom={"points": pts},
                raw={"closed": bool(getattr(e, "closed", False))},
            )
        except Exception:
            continue

    return {"items": items, "counts": counts}

@mcp.tool()
def structai_import_pdf(path: str, title: Optional[str] = None) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    pages = _extract_pdf_pages(p)
    sha = _sha256_file(p)
    uri = _path_to_uri(p)

    conn = _connect()
    try:
        artifact_id = _upsert_artifact(conn, uri=uri, kind="pdf", title=title or p.name, source_path=str(p), sha256=sha)
        _delete_artifact_content(conn, artifact_id)
        for i, text in enumerate(pages, start=1):
            chunks = _chunk_text(text, max_len=1800, overlap=200)
            _insert_doc_chunks(conn, artifact_id, chunks, page=i)
        conn.commit()
        return {"ok": True, "artifact_id": artifact_id, "pages": len(pages), "uri": uri}
    finally:
        conn.close()


@mcp.tool()
def structai_import_md(path: str, title: Optional[str] = None) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8", errors="ignore")
    sha = _sha256_file(p)
    uri = _path_to_uri(p)

    conn = _connect()
    try:
        artifact_id = _upsert_artifact(conn, uri=uri, kind="md", title=title or p.name, source_path=str(p), sha256=sha)
        _delete_artifact_content(conn, artifact_id)
        chunks = _chunk_text(text, max_len=1800, overlap=200)
        _insert_doc_chunks(conn, artifact_id, chunks, page=None)
        conn.commit()
        return {"ok": True, "artifact_id": artifact_id, "chunks": len(chunks), "uri": uri}
    finally:
        conn.close()


@mcp.tool()
def structai_import_dxf(path: str, title: Optional[str] = None, include_virtual_text: bool = True) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    data = _extract_dxf(p, include_virtual_text=include_virtual_text)
    sha = _sha256_file(p)
    uri = _path_to_uri(p)

    conn = _connect()
    try:
        artifact_id = _upsert_artifact(conn, uri=uri, kind="dxf", title=title or p.name, source_path=str(p), sha256=sha)
        _delete_artifact_content(conn, artifact_id)

        text_entities = 0
        for it in data["items"]:
            text = it.get("text")
            chunk_id = None
            if text:
                chunks = _chunk_text(text, max_len=400, overlap=0)
                if chunks:
                    _insert_doc_chunks(conn, artifact_id, [chunks[0]], page=None)
                    chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    text_entities += 1

            conn.execute(
                """
                INSERT INTO cad_entities(artifact_id, chunk_id, handle, type, layer, layout, text, x,y,z, geom_json, raw_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    int(artifact_id),
                    chunk_id,
                    it.get("handle"),
                    it.get("type"),
                    it.get("layer"),
                    "Model",
                    text,
                    it.get("x"),
                    it.get("y"),
                    it.get("z"),
                    json.dumps(it.get("geom_json") or {}, ensure_ascii=False) if it.get("geom_json") is not None else None,
                    json.dumps(it.get("raw_json") or {}, ensure_ascii=False),
                ),
            )

        conn.commit()
        return {
            "ok": True,
            "artifact_id": artifact_id,
            "uri": uri,
            "counts": data.get("counts"),
            "text_entities": text_entities,
        }
    finally:
        conn.close()


@mcp.tool()
def structai_import_files(paths: List[str], dxf_include_virtual_text: bool = True) -> Dict[str, Any]:
    results = []
    for p in paths:
        kind = _infer_kind(Path(p))
        if kind == "pdf":
            results.append(structai_import_pdf(p))
        elif kind in ("md", "markdown", "txt"):
            results.append(structai_import_md(p))
        elif kind == "dxf":
            results.append(structai_import_dxf(p, include_virtual_text=dxf_include_virtual_text))
        else:
            results.append({"ok": False, "path": p, "error": "unsupported type"})
    return {"ok": True, "items": results}


@mcp.tool()
def structai_list_artifacts(kind: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    conn = _connect()
    try:
        if kind:
            rows = conn.execute(
                "SELECT artifact_id, uri, kind, title, source_path, created_at, updated_at FROM artifacts WHERE kind=? ORDER BY artifact_id DESC LIMIT ?",
                (kind, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT artifact_id, uri, kind, title, source_path, created_at, updated_at FROM artifacts ORDER BY artifact_id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
    finally:
        conn.close()


@mcp.tool()
def structai_search_docs(query: str, limit: int = 8, kind_filter: Optional[str] = None) -> Dict[str, Any]:
    conn = _connect()
    try:
        hits = _search_docs(conn, query, limit=limit, kind_filter=kind_filter)
        return {"ok": True, "query": query, "hits": hits}
    finally:
        conn.close()


@mcp.tool()
def structai_search(query: str, limit: int = 8) -> Dict[str, Any]:
    return structai_search_docs(query=query, limit=limit)

TOKEN_RX = re.compile(r"\b[A-Z]{1,4}\s?-?\s?\d{1,4}\b", re.IGNORECASE)


@mcp.tool()
def structai_cad_get_summary(artifact: Any, top_layers: int = 20) -> Dict[str, Any]:
    conn = _connect()
    try:
        aid = _resolve_artifact_id(conn, artifact)
        total = conn.execute(
            "SELECT COUNT(*) as n FROM cad_entities WHERE artifact_id=?",
            (int(aid),),
        ).fetchone()
        by_type = conn.execute(
            "SELECT type, COUNT(*) as n FROM cad_entities WHERE artifact_id=? GROUP BY type ORDER BY n DESC",
            (int(aid),),
        ).fetchall()
        by_layer = conn.execute(
            "SELECT layer, COUNT(*) as n FROM cad_entities WHERE artifact_id=? GROUP BY layer ORDER BY n DESC LIMIT ?",
            (int(aid), int(top_layers)),
        ).fetchall()
        return {
            "ok": True,
            "artifact_id": int(aid),
            "total": int(total["n"]),
            "by_type": [dict(r) for r in by_type],
            "top_layers": [dict(r) for r in by_layer],
        }
    finally:
        conn.close()


@mcp.tool()
def structai_cad_list_layers(artifact: Any, include_empty: bool = True) -> Dict[str, Any]:
    conn = _connect()
    try:
        aid = _resolve_artifact_id(conn, artifact)
        rows = conn.execute(
            "SELECT layer, COUNT(*) as n FROM cad_entities WHERE artifact_id=? GROUP BY layer ORDER BY layer ASC",
            (int(aid),),
        ).fetchall()
        items = []
        for r in rows:
            if (not include_empty) and (not r["layer"]):
                continue
            items.append({"layer": r["layer"], "count": int(r["n"])})
        return {"ok": True, "artifact_id": int(aid), "items": items}
    finally:
        conn.close()


@mcp.tool()
def structai_cad_list_text(
    artifact: Any,
    contains: Optional[str] = None,
    layer: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        aid = _resolve_artifact_id(conn, artifact)
        sql = """
        SELECT cad_entity_id, type, layer, x, y, z, text
        FROM cad_entities
        WHERE artifact_id=? AND text IS NOT NULL
        """
        params: List[Any] = [int(aid)]
        if layer:
            sql += " AND layer=?"
            params.append(layer)
        if contains:
            sql += " AND text LIKE ?"
            params.append(f"%{contains}%")
        sql += " ORDER BY cad_entity_id ASC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return {"ok": True, "artifact_id": int(aid), "items": [dict(r) for r in rows]}
    finally:
        conn.close()


@mcp.tool()
def structai_cad_get_entity(cad_entity_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM cad_entities WHERE cad_entity_id=?", (int(cad_entity_id),)).fetchone()
        if not row:
            raise ValueError("entity not found")
        out = dict(row)
        if out.get("geom_json"):
            out["geom"] = json.loads(out.get("geom_json") or "{}")
        if out.get("raw_json"):
            out["raw"] = json.loads(out.get("raw_json") or "{}")
        return {"ok": True, "entity": out}
    finally:
        conn.close()


@mcp.tool()
def structai_cad_extract_member_candidates(
    cad_artifact_id: int,
    limit: int = 200,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT cad_entity_id, text FROM cad_entities WHERE artifact_id=? AND text IS NOT NULL",
            (int(cad_artifact_id),),
        ).fetchall()
        candidates: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            text = str(r["text"] or "")
            for m in TOKEN_RX.finditer(text.upper()):
                tok = re.sub(r"[\s\-_]+", "", m.group(0))
                if tok not in candidates:
                    candidates[tok] = {"token": tok, "count": 0}
                candidates[tok]["count"] += 1

        items = sorted(candidates.values(), key=lambda x: (-x["count"], x["token"]))
        return {"ok": True, "cad_artifact_id": int(cad_artifact_id), "items": items[: int(limit)]}
    finally:
        conn.close()

@mcp.tool()
def structai_model_import_members(
    path: str,
    model_name: Optional[str] = None,
    fmt: Optional[str] = None,
    units: Optional[str] = None,
) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    ext = p.suffix.lower()
    if fmt is None:
        fmt = "json" if ext == ".json" else "csv"

    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO models(name, source_path, units, meta_json, created_at) VALUES(?,?,?,?, datetime('now'))",
            (model_name or p.stem, str(p), units, "{}"),
        )
        model_id = int(cur.lastrowid)

        members: List[Dict[str, Any]] = []
        if fmt == "json":
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            members = data.get("members") if isinstance(data, dict) else data
            if not isinstance(members, list):
                raise ValueError("invalid json members")
        else:
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                members = [row for row in reader]

        for m in members:
            uid = str(m.get("member_uid") or m.get("uid") or "").strip()
            if not uid:
                continue
            label = str(m.get("member_label") or m.get("label") or uid).strip()
            label_norm = _norm(label)
            conn.execute(
                """
                INSERT INTO model_members(
                  model_id, member_uid, member_label, label_norm, type,
                  x1,y1,z1,x2,y2,z2, section, story, meta_json, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?, datetime('now'), datetime('now'))
                """,
                (
                    model_id,
                    uid,
                    label,
                    label_norm,
                    str(m.get("type") or ""),
                    m.get("x1"), m.get("y1"), m.get("z1"),
                    m.get("x2"), m.get("y2"), m.get("z2"),
                    str(m.get("section") or ""),
                    str(m.get("story") or ""),
                    json.dumps(m.get("meta") or {}, ensure_ascii=False),
                ),
            )

        conn.commit()
        return {"ok": True, "model_id": model_id, "members": len(members)}
    finally:
        conn.close()


@mcp.tool()
def structai_model_import_members_json(path: str, model_name: str = "regression_model") -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    members = data.get("members") if isinstance(data, dict) else data
    if not isinstance(members, list):
        raise ValueError("members.json must be list or {members:[...]}")

    conn = _connect()
    try:
        cur = conn.execute("INSERT INTO models(name, created_at) VALUES(?, datetime('now'))", (model_name,))
        model_id = int(cur.lastrowid)
        for m in members:
            conn.execute(
                """
                INSERT INTO model_members(model_id, member_uid, member_label, type, story, section)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    int(model_id),
                    str(m.get("member_uid") or ""),
                    str(m.get("member_label") or m.get("member_uid") or ""),
                    str(m.get("type") or ""),
                    str(m.get("story") or ""),
                    str(m.get("section") or ""),
                ),
            )
        conn.commit()
        return {"ok": True, "model_id": model_id, "members": len(members)}
    finally:
        conn.close()


@mcp.tool()
def structai_model_list(limit: int = 50) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT model_id, name, source_path, units, created_at FROM models ORDER BY model_id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
    finally:
        conn.close()


@mcp.tool()
def structai_model_list_members(model_id: int, contains: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    conn = _connect()
    try:
        sql = "SELECT model_member_id, member_uid, member_label, type, story, section FROM model_members WHERE model_id=?"
        params: List[Any] = [int(model_id)]
        if contains:
            sql += " AND (member_uid LIKE ? OR member_label LIKE ?)"
            params.extend([f"%{contains}%", f"%{contains}%"])
        sql += " ORDER BY model_member_id ASC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return {"ok": True, "model_id": int(model_id), "items": [dict(r) for r in rows]}
    finally:
        conn.close()

@mcp.tool()
def structai_map_suggest_members(
    cad_artifact_id: int,
    model_id: int,
    limit: int = 500,
    max_tokens: Optional[int] = None,
    max_candidates_per_token: Optional[int] = None,
    spatial_tolerance: Optional[float] = None,
    enable_fuzzy: bool = True,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        if max_tokens is not None:
            try:
                limit = min(int(limit), int(max_tokens))
            except Exception:
                pass
        # load model members
        rows = conn.execute(
            "SELECT model_member_id, member_uid, member_label, label_norm FROM model_members WHERE model_id=?",
            (int(model_id),),
        ).fetchall()
        members = [dict(r) for r in rows]
        by_label = {m["label_norm"]: m for m in members if m.get("label_norm")}
        by_uid = {str(m["member_uid"]).upper(): m for m in members if m.get("member_uid")}

        text_rows = conn.execute(
            "SELECT cad_entity_id, text, x, y, layer FROM cad_entities WHERE artifact_id=? AND text IS NOT NULL",
            (int(cad_artifact_id),),
        ).fetchall()

        suggestions = []
        seen = set()
        for r in text_rows:
            text = str(r["text"] or "")
            for m in TOKEN_RX.finditer(text.upper()):
                token = re.sub(r"[\s\-_]+", "", m.group(0))
                if not token:
                    continue
                cand = by_uid.get(token) or by_label.get(_norm(token))
                if not cand:
                    continue
                key = (token, int(cand["model_member_id"]))
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append(
                    {
                        "cad_artifact_id": int(cad_artifact_id),
                        "cad_token": token,
                        "cad_token_norm": _norm(token),
                        "model_id": int(model_id),
                        "model_member_id": int(cand["model_member_id"]),
                        "confidence": 0.85,
                        "method": "label_exact",
                        "status": "suggested",
                        "evidence": {
                            "cad_entity_id": int(r["cad_entity_id"]),
                            "layer": r["layer"],
                            "x": r["x"],
                            "y": r["y"],
                            "text": text,
                        },
                    }
                )
                if len(suggestions) >= limit:
                    break
            if len(suggestions) >= limit:
                break

        return {"ok": True, "suggestions": suggestions}
    finally:
        conn.close()


@mcp.tool()
def structai_map_save_mappings(mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
    conn = _connect()
    try:
        saved = 0
        for m in mappings:
            conn.execute(
                """
                INSERT INTO member_mappings(
                  cad_artifact_id, cad_token, cad_token_norm,
                  model_id, model_member_id, confidence, method, status, evidence_json, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?, datetime('now'))
                ON CONFLICT(cad_artifact_id, cad_token_norm, model_id, model_member_id) DO UPDATE SET
                  confidence=excluded.confidence,
                  method=excluded.method,
                  status=excluded.status,
                  evidence_json=excluded.evidence_json,
                  updated_at=datetime('now')
                """,
                (
                    int(m.get("cad_artifact_id")),
                    str(m.get("cad_token") or ""),
                    str(m.get("cad_token_norm") or _norm(m.get("cad_token") or "")),
                    int(m.get("model_id")),
                    int(m.get("model_member_id")),
                    float(m.get("confidence") or 0.5),
                    str(m.get("method") or "manual"),
                    str(m.get("status") or "suggested"),
                    json.dumps(m.get("evidence") or {}, ensure_ascii=False),
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
    conn = _connect()
    try:
        sql = "SELECT * FROM member_mappings WHERE 1=1"
        params: List[Any] = []
        if cad_artifact_id is not None:
            sql += " AND cad_artifact_id=?"
            params.append(int(cad_artifact_id))
        if model_id is not None:
            sql += " AND model_id=?"
            params.append(int(model_id))
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY mapping_id DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        items = []
        for r in rows:
            it = dict(r)
            it["evidence"] = json.loads(it.get("evidence_json") or "{}")
            items.append(it)
        return {"ok": True, "items": items}
    finally:
        conn.close()


@mcp.tool()
def structai_member_mapping_autofill(cad_artifact_id: int, model_id: int) -> Dict[str, Any]:
    res = structai_map_suggest_members(cad_artifact_id=cad_artifact_id, model_id=model_id)
    mappings = res.get("suggestions") or []
    if mappings:
        structai_map_save_mappings(mappings)
    return {"ok": True, "suggested": len(mappings)}

@mcp.tool()
def structai_results_import(
    model_id: int,
    path: str,
    run_name: Optional[str] = None,
    fmt: Optional[str] = None,
) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    ext = p.suffix.lower()
    if fmt is None:
        fmt = "json" if ext == ".json" else "csv"

    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO analysis_runs(model_id, name, created_at) VALUES(?,?, datetime('now'))",
            (int(model_id), run_name or p.stem),
        )
        analysis_run_id = int(cur.lastrowid)

        rows = conn.execute(
            "SELECT model_member_id, member_uid FROM model_members WHERE model_id=?",
            (int(model_id),),
        ).fetchall()
        uid2id = {r["member_uid"]: int(r["model_member_id"]) for r in rows}

        inserted = 0
        if fmt == "json":
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            items = data.get("results") if isinstance(data, dict) else data
            if not isinstance(items, list):
                raise ValueError("results json must be list or {results:[...]}")
            for it in items:
                uid = str(it.get("member_uid") or "").strip()
                combo = str(it.get("combo") or "LC1")
                env = it.get("env") or it.get("envelope") or {}
                mmid = uid2id.get(uid)
                if not mmid:
                    continue
                conn.execute(
                    """
                    INSERT INTO member_results(analysis_run_id, model_member_id, combo, envelope_json, created_at, updated_at)
                    VALUES(?,?,?,?, datetime('now'), datetime('now'))
                    """,
                    (analysis_run_id, mmid, combo, json.dumps(env, ensure_ascii=False)),
                )
                inserted += 1
        else:
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uid = str(row.get("member_uid") or row.get("uid") or "").strip()
                    combo = str(row.get("combo") or "LC1")
                    mmid = uid2id.get(uid)
                    if not mmid:
                        continue
                    env: Dict[str, Any] = {}
                    for k, v in row.items():
                        if k in ("member_uid", "uid", "combo"):
                            continue
                        if v is None or v == "":
                            continue
                        try:
                            env[k] = float(v)
                        except Exception:
                            env[k] = v
                    conn.execute(
                        """
                        INSERT INTO member_results(analysis_run_id, model_member_id, combo, envelope_json, created_at, updated_at)
                        VALUES(?,?,?,?, datetime('now'), datetime('now'))
                        """,
                        (analysis_run_id, mmid, combo, json.dumps(env, ensure_ascii=False)),
                    )
                    inserted += 1

        conn.commit()
        return {"ok": True, "analysis_run_id": analysis_run_id, "inserted": inserted}
    finally:
        conn.close()


@mcp.tool()
def structai_results_import_envelopes_json(
    model_id: int,
    path: str,
    run_name: str = "regression_run",
) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        raise ValueError("results.json must be list or {results:[...]}")

    conn = _connect()
    try:
        cur = conn.execute("INSERT INTO analysis_runs(model_id, name, created_at) VALUES(?,?, datetime('now'))", (int(model_id), run_name))
        analysis_run_id = int(cur.lastrowid)

        rows = conn.execute("SELECT model_member_id, member_uid FROM model_members WHERE model_id=?", (int(model_id),)).fetchall()
        uid2id = {r["member_uid"]: int(r["model_member_id"]) for r in rows}

        inserted = 0
        for r in results:
            uid = str(r.get("member_uid") or "")
            env = r.get("env") or {}
            mmid = uid2id.get(uid)
            if not mmid:
                continue
            conn.execute(
                "INSERT INTO member_results(analysis_run_id, model_member_id, combo, envelope_json, created_at, updated_at) VALUES(?,?,?,?, datetime('now'), datetime('now'))",
                (analysis_run_id, int(mmid), str(r.get("combo") or "LC1"), json.dumps(env, ensure_ascii=False)),
            )
            inserted += 1
        conn.commit()
        return {"ok": True, "analysis_run_id": analysis_run_id, "inserted": inserted}
    finally:
        conn.close()


@mcp.tool()
def structai_results_list_runs(model_id: int, limit: int = 50) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT analysis_run_id, name, engine, units, created_at FROM analysis_runs WHERE model_id=? ORDER BY analysis_run_id DESC LIMIT ?",
            (int(model_id), int(limit)),
        ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
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
        return {"ok": True, "analysis_run_id": int(analysis_run_id), "combos": [r["combo"] for r in rows]}
    finally:
        conn.close()

@mcp.tool()
def structai_design_set_member_inputs(model_id: int, member_uid: str, design: Dict[str, Any]) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT model_member_id FROM model_members WHERE model_id=? AND member_uid=?",
            (int(model_id), str(member_uid)),
        ).fetchone()
        if not row:
            raise ValueError("member not found")
        mmid = int(row["model_member_id"])
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
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    ext = p.suffix.lower()
    if fmt is None:
        fmt = "json" if ext == ".json" else "csv"

    conn = _connect()
    try:
        rows = conn.execute("SELECT model_member_id, member_uid FROM model_members WHERE model_id=?", (int(model_id),)).fetchall()
        uid2id = {r["member_uid"]: int(r["model_member_id"]) for r in rows}
        imported = 0

        if fmt == "json":
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            items = data.get("items") if isinstance(data, dict) else data
            if not isinstance(items, list):
                raise ValueError("design json must be list or {items:[...]} ")
            for it in items:
                uid = str(it.get("member_uid") or "")
                mmid = uid2id.get(uid)
                if not mmid:
                    continue
                design = it.get("design") or {k: v for k, v in it.items() if k != "member_uid"}
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
                imported += 1
        else:
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uid = str(row.get("member_uid") or row.get("uid") or "")
                    mmid = uid2id.get(uid)
                    if not mmid:
                        continue
                    design: Dict[str, Any] = {}
                    for k, v in row.items():
                        if k in ("member_uid", "uid"):
                            continue
                        if v in (None, ""):
                            continue
                        try:
                            design[k] = float(v)
                        except Exception:
                            design[k] = v
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
                    imported += 1

        conn.commit()
        return {"ok": True, "imported": imported}
    finally:
        conn.close()


@mcp.tool()
def structai_design_get_member_inputs(model_id: int, member_uid: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT model_member_id FROM model_members WHERE model_id=? AND member_uid=?",
            (int(model_id), str(member_uid)),
        ).fetchone()
        if not row:
            raise ValueError("member not found")
        mmid = int(row["model_member_id"])
        di = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
        design = json.loads(di["design_json"]) if di else {}
        return {"ok": True, "model_member_id": mmid, "design": design}
    finally:
        conn.close()

@mcp.tool()
def structai_codebook_import(path: str) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    name = data.get("name") or p.stem
    ver = data.get("version") or "0.0"

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO codebooks(name, version, codebook_json, is_active)
            VALUES(?,?,?,0)
            ON CONFLICT(name,version) DO UPDATE SET
              codebook_json=excluded.codebook_json
            """,
            (name, ver, json.dumps(data, ensure_ascii=False)),
        )
        conn.commit()
        return {"ok": True, "name": name, "version": ver}
    finally:
        conn.close()


@mcp.tool()
def structai_codebook_list() -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT codebook_id, name, version, is_active, created_at FROM codebooks ORDER BY codebook_id DESC"
        ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
    finally:
        conn.close()


@mcp.tool()
def structai_codebook_set_active(codebook_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute("UPDATE codebooks SET is_active=0")
        conn.execute("UPDATE codebooks SET is_active=1 WHERE codebook_id=?", (int(codebook_id),))
        conn.commit()
        return {"ok": True, "active_codebook_id": int(codebook_id)}
    finally:
        conn.close()


@mcp.tool()
def structai_rules_generate_from_active_codebook(
    new_rulepack_name: str,
    new_rulepack_version: str,
    base: str = "active",
    activate: bool = False,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        if base == "active":
            base_rp = _get_active_rulepack(conn)
        else:
            base_rp = BUILTIN_RULEPACK

        cb = _get_active_codebook(conn)
        if not cb:
            raise ValueError("No active codebook")
        cite_map = cb.get("citations") or {}

        rp = json.loads(json.dumps(base_rp))
        rp["name"] = new_rulepack_name
        rp["version"] = new_rulepack_version

        checks = rp.get("checks") or {}
        changed = 0
        for ct, cd in checks.items():
            if ct in cite_map:
                cd["citations"] = cite_map[ct]
                changed += 1

        conn.execute(
            """
            INSERT INTO rulepacks(name, version, rulepack_json, is_active)
            VALUES(?,?,?,0)
            ON CONFLICT(name,version) DO UPDATE SET
              rulepack_json=excluded.rulepack_json
            """,
            (new_rulepack_name, new_rulepack_version, json.dumps(rp, ensure_ascii=False)),
        )
        if activate:
            rp_id = conn.execute(
                "SELECT rulepack_id FROM rulepacks WHERE name=? AND version=?",
                (new_rulepack_name, new_rulepack_version),
            ).fetchone()
            if rp_id:
                conn.execute("UPDATE rulepacks SET is_active=0")
                conn.execute("UPDATE rulepacks SET is_active=1 WHERE rulepack_id=?", (int(rp_id["rulepack_id"]),))

        conn.commit()
        return {
            "ok": True,
            "changed_checks": changed,
            "new_rulepack": {"name": new_rulepack_name, "version": new_rulepack_version},
            "activated": activate,
        }
    finally:
        conn.close()


@mcp.tool()
def structai_rules_test_citations(limit_each: int = 1) -> Dict[str, Any]:
    conn = _connect()
    try:
        rp = _get_active_rulepack(conn)
        checks = rp.get("checks") or {}
        report = []
        for ct, cd in checks.items():
            cites = cd.get("citations") or []
            found_any = False
            found = []
            for c in cites:
                q = str(c.get("query") or "").strip()
                if not q:
                    continue
                hits = _code_citations(conn, [c], limit_each=limit_each)
                if hits:
                    found_any = True
                    found.extend(hits)
            report.append({
                "check_type": ct,
                "citation_queries": [c.get("query") for c in cites],
                "found": found_any,
                "hits": found[:limit_each],
            })
        return {"ok": True, "rulepack": {"name": rp.get("name"), "version": rp.get("version")}, "items": report}
    finally:
        conn.close()

@mcp.tool()
def structai_rules_import_rulepack(path: str) -> Dict[str, Any]:
    p = _resolve_path(path)
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
        return {"ok": True, "items": [dict(r) for r in rows]}
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


def _apply_qa_thresholds(qa: Optional[Dict[str, Any]], check_type: str, base_limit: float, base_warn: float) -> Tuple[float, float]:
    if not qa:
        return base_limit, base_warn
    thresholds = qa.get("thresholds") or {}
    default = thresholds.get("default") or {}
    per = (thresholds.get("per_check_type") or {}).get(check_type) or {}
    limit = float(per.get("limit") or default.get("limit") or base_limit)
    warn = float(per.get("warn") or default.get("warn") or base_warn)
    return limit, warn


def _check_disabled(qa: Optional[Dict[str, Any]], check_type: str, member: Dict[str, Any]) -> bool:
    if not qa:
        return False
    scope = qa.get("check_scope") or {}
    disabled = set(scope.get("disabled_check_types") or [])
    enabled = set(scope.get("enabled_check_types") or [])
    if enabled and check_type not in enabled:
        return True
    if check_type in disabled:
        return True
    exceptions = qa.get("exceptions") or []
    for ex in exceptions:
        sel = ex.get("selector") or {}
        if sel.get("check_type") and sel.get("check_type") != check_type:
            continue
        uid_re = sel.get("member_uid_regex")
        st = sel.get("story_norm")
        if uid_re:
            try:
                if not re.search(uid_re, str(member.get("member_uid") or "")):
                    continue
            except Exception:
                continue
        if st and str(st) != str(member.get("story") or ""):
            continue
        action = ex.get("action") or {}
        if action.get("skip_check_types") and check_type in action.get("skip_check_types"):
            return True
    return False


@mcp.tool()
def structai_check_run(
    model_id: int,
    analysis_run_id: int,
    name: str = "check_run",
    check_types: Optional[List[str]] = None,
    combos: Optional[List[str]] = None,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        rp = _get_active_rulepack(conn)
        checks_def = rp.get("checks") or {}
        if not check_types:
            check_types = list(checks_def.keys())

        if not combos:
            rows = conn.execute(
                "SELECT DISTINCT combo FROM member_results WHERE analysis_run_id=?",
                (int(analysis_run_id),),
            ).fetchall()
            combos = [r["combo"] for r in rows]

        qa = _get_effective_qa_profile(conn, model_id)

        cur = conn.execute(
            """
            INSERT INTO check_runs(model_id, analysis_run_id, name, rulepack_name, rulepack_version, combos_json, check_types_json, scope_json, created_at)
            VALUES(?,?,?,?,?,?,?, ?, datetime('now'))
            """,
            (
                int(model_id),
                int(analysis_run_id),
                name,
                rp.get("name"),
                rp.get("version"),
                json.dumps(combos, ensure_ascii=False),
                json.dumps(check_types, ensure_ascii=False),
                json.dumps({"qa_profile": qa.get("name") if qa else None}, ensure_ascii=False),
            ),
        )
        check_run_id = int(cur.lastrowid)

        members = conn.execute(
            "SELECT model_member_id, member_uid, member_label, type, story FROM model_members WHERE model_id=?",
            (int(model_id),),
        ).fetchall()
        member_rows = {int(r["model_member_id"]): dict(r) for r in members}

        results = conn.execute(
            "SELECT model_member_id, combo, envelope_json FROM member_results WHERE analysis_run_id=?",
            (int(analysis_run_id),),
        ).fetchall()

        for res in results:
            mmid = int(res["model_member_id"])
            combo = res["combo"]
            env = json.loads(res["envelope_json"] or "{}")
            env = enrich_envelope(env)

            di = conn.execute(
                "SELECT design_json FROM member_design_inputs WHERE model_member_id=?",
                (mmid,),
            ).fetchone()
            design = json.loads(di["design_json"]) if di else {}

            member = member_rows.get(mmid) or {}

            for ct in check_types:
                if ct not in checks_def:
                    continue
                if _check_disabled(qa, ct, member):
                    status = "NA"
                    conn.execute(
                        """
                        INSERT INTO check_results(check_run_id, model_member_id, combo, check_type, status, details_json, citations_json)
                        VALUES(?,?,?,?,?,?,?)
                        """,
                        (check_run_id, mmid, combo, ct, status, json.dumps({"disabled": True}, ensure_ascii=False), "[]"),
                    )
                    continue

                cd = checks_def.get(ct) or {}
                limit = float(cd.get("limit", 1.0))
                warn = float(cd.get("warn", limit))
                limit, warn = _apply_qa_thresholds(qa, ct, limit, warn)

                demand_expr = cd.get("demand_expr")
                capacity_expr = cd.get("capacity_expr")
                ratio_expr = cd.get("ratio_expr")

                numeric_env = {k: v for k, v in env.items() if isinstance(v, (int, float))}
                numeric_design = {k: v for k, v in design.items() if isinstance(v, (int, float))}

                demand = None
                capacity = None
                ratio = None
                details = {"demand_expr": demand_expr, "capacity_expr": capacity_expr, "ratio_expr": ratio_expr}

                try:
                    if ratio_expr:
                        ratio = _safe_eval(str(ratio_expr), {**numeric_env, **numeric_design})
                    if demand_expr:
                        demand = _safe_eval(str(demand_expr), numeric_env)
                    if capacity_expr:
                        capacity = _safe_eval(str(capacity_expr), numeric_design)
                    if ratio is None and demand is not None and capacity not in (None, 0):
                        ratio = float(demand) / float(capacity)
                except Exception as exc:
                    details["error"] = str(exc)

                status = _status_from_ratio(ratio, limit=limit, warn=warn)
                cites = _code_citations(conn, cd.get("citations") or [], limit_each=1)

                conn.execute(
                    """
                    INSERT INTO check_results(
                      check_run_id, model_member_id, combo, check_type,
                      demand_value, capacity_value, ratio, status, details_json, citations_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        check_run_id,
                        mmid,
                        combo,
                        ct,
                        demand,
                        capacity,
                        ratio,
                        status,
                        json.dumps(details, ensure_ascii=False),
                        json.dumps(cites, ensure_ascii=False),
                    ),
                )

        conn.commit()
        return {"ok": True, "check_run_id": check_run_id}
    finally:
        conn.close()


@mcp.tool()
def structai_check_list_runs(model_id: int, limit: int = 50) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT check_run_id, name, rulepack_name, rulepack_version, created_at FROM check_runs WHERE model_id=? ORDER BY check_run_id DESC LIMIT ?",
            (int(model_id), int(limit)),
        ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
    finally:
        conn.close()


@mcp.tool()
def structai_check_get_results(check_run_id: int, status: Optional[str] = None, limit: int = 500) -> Dict[str, Any]:
    conn = _connect()
    try:
        sql = """
        SELECT cr.*, mb.member_uid, mb.member_label, mb.type
        FROM check_results cr
        JOIN model_members mb ON mb.model_member_id = cr.model_member_id
        WHERE cr.check_run_id=?
        """
        params: List[Any] = [int(check_run_id)]
        if status:
            sql += " AND cr.status=?"
            params.append(status)
        sql += " ORDER BY cr.check_result_id ASC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        items = []
        for r in rows:
            it = dict(r)
            it["details"] = json.loads(it.get("details_json") or "{}")
            it["citations"] = json.loads(it.get("citations_json") or "[]")
            items.append(it)
        return {"ok": True, "check_run_id": int(check_run_id), "items": items}
    finally:
        conn.close()


def _match_where(member: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    if not where:
        return True
    for k, v in where.items():
        if v is None:
            continue
        mv = member.get(k)
        if isinstance(v, (list, tuple, set)):
            if mv not in v:
                return False
        else:
            if str(mv).lower() != str(v).lower():
                return False
    return True


@mcp.tool()
def structai_design_compute_rc_beam_rect(
    model_id: int,
    where: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    defaults = defaults or {}
    conn = _connect()
    try:
        members = conn.execute(
            "SELECT model_member_id, member_uid, member_label, type FROM model_members WHERE model_id=?",
            (int(model_id),),
        ).fetchall()
        updated = 0
        for m in members:
            member = dict(m)
            if not _match_where(member, where):
                continue
            mmid = int(member["model_member_id"])
            row = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
            dj = json.loads(row["design_json"]) if row else {}
            if not overwrite and (dj.get("Mn") or dj.get("Vn")):
                continue

            data = {**defaults, **dj}
            try:
                inp = RcBeamRectInputs(
                    b=float(data.get("b")),
                    h=data.get("h"),
                    d=data.get("d"),
                    cover=data.get("cover"),
                    stirrup_dia=data.get("stirrup_dia"),
                    bar_dia=data.get("bar_dia"),
                    fc=float(data.get("fc", 24.0)),
                    fy=float(data.get("fy", 400.0)),
                    As=data.get("As"),
                    As_top=data.get("As_top"),
                    As_bot=data.get("As_bot"),
                    Av=data.get("Av"),
                    s=data.get("s"),
                    phi_flex=float(data.get("phi_flex", 0.9)),
                    phi_shear=float(data.get("phi_shear", 0.75)),
                )
            except Exception:
                continue

            try:
                cap = rc_beam_rect_capacity(inp)
            except Exception:
                continue

            dj.update({
                "Mn": cap.get("Mn"),
                "Mn_pos": cap.get("Mn_pos"),
                "Mn_neg": cap.get("Mn_neg"),
                "Vn": cap.get("Vn"),
                "trace_rc_beam": cap.get("trace"),
            })

            conn.execute(
                """
                INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
                VALUES(?,?, datetime('now'))
                ON CONFLICT(model_member_id) DO UPDATE SET
                  design_json=excluded.design_json,
                  updated_at=datetime('now')
                """,
                (mmid, json.dumps(dj, ensure_ascii=False)),
            )
            updated += 1

        conn.commit()
        return {"ok": True, "updated": updated}
    finally:
        conn.close()


@mcp.tool()
def structai_design_compute_rc_column_axial(
    model_id: int,
    where: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    defaults = defaults or {}
    conn = _connect()
    try:
        members = conn.execute(
            "SELECT model_member_id, member_uid, member_label, type FROM model_members WHERE model_id=?",
            (int(model_id),),
        ).fetchall()
        updated = 0
        for m in members:
            member = dict(m)
            if not _match_where(member, where):
                continue
            mmid = int(member["model_member_id"])
            row = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
            dj = json.loads(row["design_json"]) if row else {}
            if not overwrite and dj.get("Pn"):
                continue

            data = {**defaults, **dj}
            try:
                inp = RcColumnAxialInputs(
                    Ag=data.get("Ag"),
                    As=float(data.get("As", 0.0)),
                    fc=float(data.get("fc", 24.0)),
                    fy=float(data.get("fy", 400.0)),
                    phi_axial=float(data.get("phi_axial", 0.65)),
                )
            except Exception:
                continue
            try:
                cap = rc_column_axial_capacity(inp)
            except Exception:
                continue

            dj.update({"Pn": cap.get("Pn"), "trace_rc_column": cap.get("trace")})
            conn.execute(
                """
                INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
                VALUES(?,?, datetime('now'))
                ON CONFLICT(model_member_id) DO UPDATE SET
                  design_json=excluded.design_json,
                  updated_at=datetime('now')
                """,
                (mmid, json.dumps(dj, ensure_ascii=False)),
            )
            updated += 1

        conn.commit()
        return {"ok": True, "updated": updated}
    finally:
        conn.close()


@mcp.tool()
def structai_design_compute_steel_beam_simple(
    model_id: int,
    where: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    defaults = defaults or {}
    conn = _connect()
    try:
        members = conn.execute(
            "SELECT model_member_id, member_uid, member_label, type FROM model_members WHERE model_id=?",
            (int(model_id),),
        ).fetchall()
        updated = 0
        for m in members:
            member = dict(m)
            if not _match_where(member, where):
                continue
            mmid = int(member["model_member_id"])
            row = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
            dj = json.loads(row["design_json"]) if row else {}
            if not overwrite and (dj.get("Mn") or dj.get("Vn")):
                continue

            data = {**defaults, **dj}
            try:
                inp = SteelBeamInputs(
                    Fy=float(data.get("Fy", 345.0)),
                    Zx=data.get("Zx"),
                    Aw=data.get("Aw"),
                    phi_flex=float(data.get("phi_flex", 0.9)),
                    phi_shear=float(data.get("phi_shear", 0.9)),
                )
            except Exception:
                continue

            try:
                cap = steel_beam_capacity(inp)
            except Exception:
                continue

            dj.update({"Mn": cap.get("Mn"), "Vn": cap.get("Vn"), "trace_steel": cap.get("trace")})
            conn.execute(
                """
                INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
                VALUES(?,?, datetime('now'))
                ON CONFLICT(model_member_id) DO UPDATE SET
                  design_json=excluded.design_json,
                  updated_at=datetime('now')
                """,
                (mmid, json.dumps(dj, ensure_ascii=False)),
            )
            updated += 1

        conn.commit()
        return {"ok": True, "updated": updated}
    finally:
        conn.close()

@mcp.tool()
def structai_report_generate(
    check_run_id: int,
    formats: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    formats = formats or ["md"]
    out_dir_path = Path(out_dir).expanduser().resolve() if out_dir else (DB_PATH.parent / "reports")
    out_dir_path.mkdir(parents=True, exist_ok=True)

    conn = _connect()
    try:
        run = conn.execute(
            """
            SELECT cr.check_run_id, cr.name as check_run_name, cr.model_id, cr.analysis_run_id, cr.rulepack_name, cr.rulepack_version,
                   m.name as model_name, a.name as analysis_run_name
            FROM check_runs cr
            JOIN models m ON m.model_id = cr.model_id
            JOIN analysis_runs a ON a.analysis_run_id = cr.analysis_run_id
            WHERE cr.check_run_id=?
            """,
            (int(check_run_id),),
        ).fetchone()
        if not run:
            raise ValueError("check_run not found")

        rows = conn.execute(
            """
            SELECT cr.*, mb.member_uid, mb.member_label, mb.type
            FROM check_results cr
            JOIN model_members mb ON mb.model_member_id = cr.model_member_id
            WHERE cr.check_run_id=?
            ORDER BY cr.ratio DESC
            """,
            (int(check_run_id),),
        ).fetchall()

        items = []
        summary: Dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0, "NA": 0}
        for r in rows:
            it = dict(r)
            it["details"] = json.loads(it.get("details_json") or "{}")
            it["citations"] = json.loads(it.get("citations_json") or "[]")
            items.append(it)
            status = it.get("status") or "NA"
            summary[status] = summary.get(status, 0) + 1

        meta = {
            "check_run_id": run["check_run_id"],
            "check_run_name": run["check_run_name"],
            "model_id": run["model_id"],
            "model_name": run["model_name"],
            "analysis_run_id": run["analysis_run_id"],
            "analysis_run_name": run["analysis_run_name"],
            "rulepack_name": run["rulepack_name"],
            "rulepack_version": run["rulepack_version"],
        }

        outputs = []
        if "md" in formats:
            md_text = build_markdown_report(meta, summary, items[:300])
            md_path = out_dir_path / f"check_run_{check_run_id}.md"
            md_path.write_text(md_text, encoding="utf-8")
            sha = _sha256_file(md_path)
            uri = md_path.as_uri()
            art_id = _upsert_artifact(conn, uri=uri, kind="report_md", title=f"Check Report #{check_run_id}", source_path=str(md_path), sha256=sha)
            conn.execute("INSERT INTO reports(check_run_id, artifact_id, format) VALUES(?,?,?)", (int(check_run_id), art_id, "md"))
            outputs.append({"format": "md", "path": str(md_path), "uri": uri, "artifact_id": art_id})

        if "pdf" in formats:
            pdf_path = out_dir_path / f"check_run_{check_run_id}.pdf"
            build_pdf_report(str(pdf_path), meta, summary, items)
            sha = _sha256_file(pdf_path)
            uri = pdf_path.as_uri()
            art_id = _upsert_artifact(conn, uri=uri, kind="report_pdf", title=f"Check Report #{check_run_id}", source_path=str(pdf_path), sha256=sha)
            conn.execute("INSERT INTO reports(check_run_id, artifact_id, format) VALUES(?,?,?)", (int(check_run_id), art_id, "pdf"))
            outputs.append({"format": "pdf", "path": str(pdf_path), "uri": uri, "artifact_id": art_id})

        conn.commit()
        return {"ok": True, "check_run_id": int(check_run_id), "outputs": outputs}
    finally:
        conn.close()


@mcp.tool()
def structai_report_list(check_run_id: Optional[int] = None, limit: int = 50) -> Dict[str, Any]:
    conn = _connect()
    try:
        if check_run_id is not None:
            rows = conn.execute(
                """
                SELECT r.report_id, r.check_run_id, r.format, r.created_at, a.uri, a.source_path
                FROM reports r
                JOIN artifacts a ON a.artifact_id = r.artifact_id
                WHERE r.check_run_id=?
                ORDER BY r.report_id DESC LIMIT ?
                """,
                (int(check_run_id), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT r.report_id, r.check_run_id, r.format, r.created_at, a.uri, a.source_path
                FROM reports r
                JOIN artifacts a ON a.artifact_id = r.artifact_id
                ORDER BY r.report_id DESC LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
    finally:
        conn.close()


def _dist2d(ax: Optional[float], ay: Optional[float], bx: Optional[float], by: Optional[float]) -> Optional[float]:
    if ax is None or ay is None or bx is None or by is None:
        return None
    try:
        return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)
    except Exception:
        return None


@mcp.tool()
def structai_cad_parse_specs(
    cad_artifact_id: int,
    overwrite: bool = True,
    include_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    include_types = include_types or ["TEXT", "MTEXT", "ATTRIB"]
    conn = _connect()
    try:
        if overwrite:
            conn.execute("DELETE FROM cad_specs WHERE cad_artifact_id=?", (int(cad_artifact_id),))

        rows = conn.execute(
            f"""
            SELECT cad_entity_id, type, layer, x, y, z, text
            FROM cad_entities
            WHERE artifact_id=? AND type IN ({','.join(['?'] * len(include_types))}) AND text IS NOT NULL
            """,
            (int(cad_artifact_id), *include_types),
        ).fetchall()

        inserted = 0
        counts: Dict[str, int] = {}
        for r in rows:
            raw = str(r["text"] or "").strip()
            if not raw:
                continue
            specs = parse_specs_from_text(raw)
            if not specs:
                continue
            for s in specs:
                kind = s["spec_kind"]
                counts[kind] = counts.get(kind, 0) + 1
                conn.execute(
                    """
                    INSERT INTO cad_specs(
                      cad_artifact_id, cad_entity_id, spec_kind, spec_json, raw_text,
                      x,y,z, layer, confidence
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        int(cad_artifact_id),
                        int(r["cad_entity_id"]),
                        kind,
                        json.dumps(s, ensure_ascii=False),
                        raw,
                        r["x"], r["y"], r["z"],
                        r["layer"],
                        float(s.get("confidence", 0.5)),
                    ),
                )
                inserted += 1

        conn.commit()
        return {"ok": True, "cad_artifact_id": int(cad_artifact_id), "inserted": inserted, "counts": counts}
    finally:
        conn.close()


@mcp.tool()
def structai_cad_list_specs(
    cad_artifact_id: int,
    spec_kind: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        if spec_kind:
            rows = conn.execute(
                """
                SELECT spec_id, spec_kind, raw_text, spec_json, x, y, layer, confidence
                FROM cad_specs
                WHERE cad_artifact_id=? AND spec_kind=?
                ORDER BY confidence DESC, spec_id ASC
                LIMIT ?
                """,
                (int(cad_artifact_id), str(spec_kind), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT spec_id, spec_kind, raw_text, spec_json, x, y, layer, confidence
                FROM cad_specs
                WHERE cad_artifact_id=?
                ORDER BY confidence DESC, spec_id ASC
                LIMIT ?
                """,
                (int(cad_artifact_id), int(limit)),
            ).fetchall()

        out = []
        for r in rows:
            it = dict(r)
            it["spec"] = json.loads(it.pop("spec_json") or "{}")
            out.append(it)
        return {"ok": True, "cad_artifact_id": int(cad_artifact_id), "count": len(out), "items": out}
    finally:
        conn.close()


@mcp.tool()
def structai_specs_suggest_links(
    cad_artifact_id: int,
    model_id: int,
    mapping_status: str = "confirmed",
    max_dist: float = 500.0,
    overwrite_suggested: bool = True,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        if overwrite_suggested:
            conn.execute(
                "DELETE FROM member_spec_links WHERE cad_artifact_id=? AND model_id=? AND status='suggested'",
                (int(cad_artifact_id), int(model_id)),
            )

        maps = conn.execute(
            """
            SELECT mm.cad_token_norm, mm.model_member_id, mb.member_uid, mb.member_label, mb.type
            FROM member_mappings mm
            JOIN model_members mb ON mb.model_member_id = mm.model_member_id
            WHERE mm.cad_artifact_id=? AND mm.model_id=? AND mm.status=?
            """,
            (int(cad_artifact_id), int(model_id), str(mapping_status)),
        ).fetchall()

        specs = conn.execute(
            """
            SELECT spec_id, cad_entity_id, spec_kind, spec_json, raw_text, x, y, layer, confidence
            FROM cad_specs
            WHERE cad_artifact_id=?
            """,
            (int(cad_artifact_id),),
        ).fetchall()

        def token_points(token_norm: str) -> List[Tuple[int, float, float, Optional[str]]]:
            rows = conn.execute(
                """
                SELECT cad_entity_id, x, y, layer, text
                FROM cad_entities
                WHERE artifact_id=? AND text IS NOT NULL AND instr(upper(text), ?) > 0
                """,
                (int(cad_artifact_id), str(token_norm).upper()),
            ).fetchall()
            pts = []
            for r in rows:
                if r["x"] is None or r["y"] is None:
                    continue
                pts.append((int(r["cad_entity_id"]), float(r["x"]), float(r["y"]), r["layer"]))
            return pts

        created = 0
        suggestions: List[Dict[str, Any]] = []

        for m in maps:
            token_norm = m["cad_token_norm"]
            mmid = int(m["model_member_id"])
            pts = token_points(token_norm)
            if not pts:
                continue

            best_by_kind: Dict[str, Tuple[sqlite3.Row, float]] = {}
            for s in specs:
                sx, sy = s["x"], s["y"]
                if sx is None or sy is None:
                    continue
                ds = []
                for (_eid, px, py, _layer) in pts:
                    d = _dist2d(px, py, float(sx), float(sy))
                    if d is not None:
                        ds.append(d)
                if not ds:
                    continue
                dmin = min(ds)
                if dmin > max_dist:
                    continue
                kind = s["spec_kind"]
                prev = best_by_kind.get(kind)
                if prev is None or dmin < prev[1]:
                    best_by_kind[kind] = (s, dmin)

            for kind, (s, dmin) in best_by_kind.items():
                evidence = {
                    "token_norm": token_norm,
                    "distance": dmin,
                    "spec_confidence": float(s["confidence"] or 0.5),
                }
                conn.execute(
                    """
                    INSERT INTO member_spec_links(
                      cad_artifact_id, spec_id, model_id, model_member_id,
                      cad_token_norm, distance, method, status, evidence_json, updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?, datetime('now'))
                    ON CONFLICT(cad_artifact_id, spec_id, model_member_id)
                    DO UPDATE SET
                      status='suggested',
                      distance=excluded.distance,
                      evidence_json=excluded.evidence_json,
                      updated_at=datetime('now')
                    """,
                    (
                        int(cad_artifact_id),
                        int(s["spec_id"]),
                        int(model_id),
                        int(mmid),
                        str(token_norm),
                        float(dmin),
                        "spatial",
                        "suggested",
                        json.dumps(evidence, ensure_ascii=False),
                    ),
                )
                created += 1
                suggestions.append({
                    "model_member_id": mmid,
                    "member_uid": m["member_uid"],
                    "member_label": m["member_label"],
                    "member_type": m["type"],
                    "token_norm": token_norm,
                    "spec_id": int(s["spec_id"]),
                    "spec_kind": kind,
                    "distance": float(dmin),
                    "raw_text": s["raw_text"],
                    "layer": s["layer"],
                    "confidence": float(s["confidence"]),
                    "spec": json.loads(s["spec_json"] or "{}"),
                })

        conn.commit()
        return {"ok": True, "created_links": created, "suggestions": suggestions[:200]}
    finally:
        conn.close()


@mcp.tool()
def structai_specs_confirm_all(
    cad_artifact_id: int,
    model_id: int,
    from_status: str = "suggested",
) -> Dict[str, Any]:
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE member_spec_links SET status='confirmed', updated_at=datetime('now') WHERE cad_artifact_id=? AND model_id=? AND status=?",
            (int(cad_artifact_id), int(model_id), str(from_status)),
        )
        conn.commit()
        return {"ok": True, "confirmed": int(cur.rowcount)}
    finally:
        conn.close()


@mcp.tool()
def structai_specs_list_links(
    cad_artifact_id: int,
    model_id: int,
    status: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        sql = """
        SELECT l.link_id, l.status, l.method, l.distance, l.cad_token_norm, l.evidence_json,
               s.spec_kind, s.spec_json, s.raw_text, mb.member_uid, mb.member_label
        FROM member_spec_links l
        JOIN cad_specs s ON s.spec_id = l.spec_id
        JOIN model_members mb ON mb.model_member_id = l.model_member_id
        WHERE l.cad_artifact_id=? AND l.model_id=?
        """
        params: List[Any] = [int(cad_artifact_id), int(model_id)]
        if status:
            sql += " AND l.status=?"
            params.append(status)
        sql += " ORDER BY l.link_id DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        items = []
        for r in rows:
            it = dict(r)
            it["evidence"] = json.loads(it.get("evidence_json") or "{}")
            it["spec"] = json.loads(it.get("spec_json") or "{}")
            items.append(it)
        return {"ok": True, "items": items}
    finally:
        conn.close()


@mcp.tool()
def structai_specs_set_link_status(link_id: int, to_status: str, reason: str = "") -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT status FROM member_spec_links WHERE link_id=?",
            (int(link_id),),
        ).fetchone()
        if not row:
            raise ValueError("link not found")
        from_status = row["status"]
        conn.execute(
            "UPDATE member_spec_links SET status=?, updated_at=datetime('now') WHERE link_id=?",
            (str(to_status), int(link_id)),
        )
        conn.execute(
            """INSERT INTO decision_logs(entity_type, entity_id, from_status, to_status, reason, meta_json)
             VALUES(?,?,?,?,?,?)""",
            ("member_spec_links", int(link_id), str(from_status), str(to_status), reason, "{}"),
        )
        conn.commit()
        return {"ok": True, "link_id": int(link_id), "status": to_status}
    finally:
        conn.close()


@mcp.tool()
def structai_design_apply_specs_to_inputs(
    cad_artifact_id: int,
    model_id: int,
    overwrite_keys: bool = False,
    note: str = "apply specs",
) -> Dict[str, Any]:
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO design_patch_runs(model_id, cad_artifact_id, note, params_json, created_at) VALUES(?,?,?,?, datetime('now'))",
            (int(model_id), int(cad_artifact_id), note, json.dumps({"overwrite_keys": overwrite_keys}, ensure_ascii=False)),
        )
        patch_run_id = int(cur.lastrowid)

        rows = conn.execute(
            """
            SELECT l.model_member_id, s.spec_kind, s.spec_json, s.raw_text, l.distance
            FROM member_spec_links l
            JOIN cad_specs s ON s.spec_id = l.spec_id
            WHERE l.cad_artifact_id=? AND l.model_id=? AND l.status='confirmed'
            ORDER BY l.model_member_id ASC
            """,
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()

        by_member: Dict[int, List[sqlite3.Row]] = {}
        for r in rows:
            by_member.setdefault(int(r["model_member_id"]), []).append(r)

        applied = 0
        details = []

        for mmid, items in by_member.items():
            di = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (int(mmid),)).fetchone()
            dj = json.loads(di["design_json"]) if di else {}
            before = json.loads(json.dumps(dj))

            patch: Dict[str, Any] = {}
            sources: List[Dict[str, Any]] = []

            for r in items:
                kind = r["spec_kind"]
                spec = json.loads(r["spec_json"] or "{}")

                if kind == "rc_rect_section":
                    patch.setdefault("b", spec.get("b_mm"))
                    patch.setdefault("h", spec.get("h_mm"))
                if kind == "steel_h_section":
                    patch.setdefault("steel_section", spec)
                if kind == "steel_box_section":
                    patch.setdefault("steel_section", spec)
                if kind == "steel_pipe_section":
                    patch.setdefault("steel_section", spec)
                if kind == "steel_channel_section":
                    patch.setdefault("steel_section", spec)
                if kind == "steel_angle_section":
                    patch.setdefault("steel_section", spec)

                if kind == "rebar_main":
                    pos = spec.get("pos", "UNKNOWN")
                    As = spec.get("As_mm2")
                    if As:
                        if pos == "TOP":
                            patch["As_top"] = float(As)
                        elif pos == "BOT":
                            patch["As_bot"] = float(As)
                        else:
                            patch.setdefault("As", float(As))

                if kind == "rebar_stirrup":
                    if spec.get("Av_mm2"):
                        patch.setdefault("Av", float(spec.get("Av_mm2")))
                    if spec.get("s_mm"):
                        patch.setdefault("s", float(spec.get("s_mm")))

                sources.append({
                    "spec_id": None,
                    "spec_kind": kind,
                    "raw_text": r["raw_text"],
                    "distance": float(r["distance"]) if r["distance"] is not None else None,
                })

            merged = dict(dj)
            changed_keys = []
            for k, v in patch.items():
                if v is None:
                    continue
                if (not overwrite_keys) and (k in merged and merged[k] not in (None, "", 0)):
                    continue
                merged[k] = v
                changed_keys.append(k)

            merged["spec_sources"] = sources
            merged.setdefault("units", {"length": "mm", "stress": "MPa"})

            conn.execute(
                """
                INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
                VALUES(?,?, datetime('now'))
                ON CONFLICT(model_member_id) DO UPDATE SET
                  design_json=excluded.design_json,
                  updated_at=datetime('now')
                """,
                (int(mmid), json.dumps(merged, ensure_ascii=False)),
            )

            conn.execute(
                """
                INSERT INTO design_patch_items(patch_run_id, model_member_id, before_json, after_json, changed_keys_json)
                VALUES(?,?,?,?,?)
                """,
                (
                    patch_run_id,
                    int(mmid),
                    json.dumps(before, ensure_ascii=False),
                    json.dumps(merged, ensure_ascii=False),
                    json.dumps(changed_keys, ensure_ascii=False),
                ),
            )
            applied += 1
            details.append({"model_member_id": mmid, "patch_keys": list(patch.keys())})

        conn.commit()
        return {"ok": True, "patch_run_id": patch_run_id, "members_applied": applied, "details": details[:200]}
    finally:
        conn.close()


@mcp.tool()
def structai_design_list_patch_runs(model_id: int, limit: int = 30) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT patch_run_id, cad_artifact_id, note, params_json, created_at FROM design_patch_runs WHERE model_id=? ORDER BY patch_run_id DESC LIMIT ?",
            (int(model_id), int(limit)),
        ).fetchall()
        items = []
        for r in rows:
            it = dict(r)
            it["params"] = json.loads(it.get("params_json") or "{}")
            items.append(it)
        return {"ok": True, "items": items}
    finally:
        conn.close()


@mcp.tool()
def structai_design_rollback_patch(patch_run_id: int, mode: str = "keys_only") -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT model_member_id, before_json, after_json, changed_keys_json FROM design_patch_items WHERE patch_run_id=?",
            (int(patch_run_id),),
        ).fetchall()
        rolled = 0
        for r in rows:
            before = json.loads(r["before_json"] or "{}")
            after = json.loads(r["after_json"] or "{}")
            keys = json.loads(r["changed_keys_json"] or "[]")
            if mode == "full":
                new_design = before
            else:
                new_design = dict(after)
                for k in keys:
                    if k in before:
                        new_design[k] = before[k]
                    elif k in new_design:
                        new_design.pop(k, None)

            conn.execute(
                """
                INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
                VALUES(?,?, datetime('now'))
                ON CONFLICT(model_member_id) DO UPDATE SET
                  design_json=excluded.design_json,
                  updated_at=datetime('now')
                """,
                (int(r["model_member_id"]), json.dumps(new_design, ensure_ascii=False)),
            )
            rolled += 1

        conn.commit()
        return {"ok": True, "rolled_back": rolled, "mode": mode}
    finally:
        conn.close()


def _nearest_story_for_point(conn: sqlite3.Connection, cad_artifact_id: int, x: float, y: float, max_dist: float = 800.0) -> Optional[str]:
    rows = conn.execute(
        "SELECT story_norm, x, y FROM cad_story_tags WHERE cad_artifact_id=?",
        (int(cad_artifact_id),),
    ).fetchall()
    best = None
    best_d = None
    for r in rows:
        d = _dist2d(x, y, r["x"], r["y"])
        if d is None:
            continue
        if best_d is None or d < best_d:
            best_d = d
            best = r["story_norm"]
    if best_d is not None and best_d <= max_dist:
        return str(best)
    return None


@mcp.tool()
def structai_cad_extract_tables(
    cad_artifact_id: int,
    layer_filter: Optional[str] = None,
    min_cells: int = 16,
    overwrite: bool = True,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        if overwrite:
            conn.execute("DELETE FROM cad_tables WHERE cad_artifact_id=?", (int(cad_artifact_id),))

        rows = conn.execute(
            """
            SELECT cad_entity_id, x, y, text, layer
            FROM cad_entities
            WHERE artifact_id=? AND type IN ('TEXT','MTEXT','ATTRIB') AND text IS NOT NULL AND x IS NOT NULL AND y IS NOT NULL
            """,
            (int(cad_artifact_id),),
        ).fetchall()

        by_layer: Dict[str, List[TextPoint]] = {}
        for r in rows:
            layer = r["layer"] or ""
            if layer_filter and layer_filter not in layer:
                continue
            by_layer.setdefault(layer, []).append(TextPoint(
                cad_entity_id=int(r["cad_entity_id"]),
                x=float(r["x"]),
                y=float(r["y"]),
                text=str(r["text"] or ""),
                layer=layer,
            ))

        tables_saved = 0
        cells_saved = 0
        for layer, pts in by_layer.items():
            tables = extract_grid_tables(pts, min_cells=min_cells)
            for t in tables:
                cur = conn.execute(
                    """
                    INSERT INTO cad_tables(cad_artifact_id, method, bbox_json, rows, cols, confidence, meta_json)
                    VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        int(cad_artifact_id),
                        t.get("method"),
                        json.dumps(t.get("bbox") or {}, ensure_ascii=False),
                        int(t.get("rows") or 0),
                        int(t.get("cols") or 0),
                        float(t.get("confidence") or 0.5),
                        json.dumps(t.get("meta") or {}, ensure_ascii=False),
                    ),
                )
                table_id = int(cur.lastrowid)
                tables_saved += 1
                for c in t.get("cells") or []:
                    conn.execute(
                        """
                        INSERT INTO cad_table_cells(table_id, row_idx, col_idx, cad_entity_id, text, x, y)
                        VALUES(?,?,?,?,?,?,?)
                        """,
                        (
                            table_id,
                            int(c.get("row_idx")),
                            int(c.get("col_idx")),
                            int(c.get("cad_entity_id")),
                            str(c.get("text") or ""),
                            float(c.get("x")) if c.get("x") is not None else None,
                            float(c.get("y")) if c.get("y") is not None else None,
                        ),
                    )
                    cells_saved += 1

        conn.commit()
        return {"ok": True, "tables": tables_saved, "cells": cells_saved}
    finally:
        conn.close()


@mcp.tool()
def structai_cad_infer_table_schemas(
    cad_artifact_id: int,
    overwrite: bool = True,
    min_table_confidence: float = 0.0,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        if overwrite:
            conn.execute(
                "DELETE FROM cad_table_schemas WHERE table_id IN (SELECT table_id FROM cad_tables WHERE cad_artifact_id=?)",
                (int(cad_artifact_id),),
            )
            conn.execute(
                "DELETE FROM cad_table_row_parses WHERE table_id IN (SELECT table_id FROM cad_tables WHERE cad_artifact_id=?)",
                (int(cad_artifact_id),),
            )

        tables = conn.execute(
            "SELECT table_id, confidence FROM cad_tables WHERE cad_artifact_id=? ORDER BY confidence DESC",
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
                "SELECT row_idx, col_idx, cad_entity_id, text, x, y FROM cad_table_cells WHERE table_id=?",
                (table_id,),
            ).fetchall()
            cell_list = [dict(r) for r in cells]
            sch = infer_schema(cell_list)
            conn.execute(
                "INSERT INTO cad_table_schemas(table_id, header_row_idx, columns_json, confidence) VALUES(?,?,?,?)",
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
                    sample.append({"table_id": table_id, "row_idx": rp["row_idx"], "token": rp.get("token_norm")})

        conn.commit()
        return {"ok": True, "schemas": saved_schema, "rows": saved_rows, "sample": sample}
    finally:
        conn.close()


@mcp.tool()
def structai_cad_infer_table_schemas_v2(
    cad_artifact_id: int,
    overwrite: bool = True,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        if overwrite:
            conn.execute("DELETE FROM cad_table_schemas WHERE table_id IN (SELECT table_id FROM cad_tables WHERE cad_artifact_id=?)", (int(cad_artifact_id),))
            conn.execute("DELETE FROM cad_table_row_parses WHERE table_id IN (SELECT table_id FROM cad_tables WHERE cad_artifact_id=?)", (int(cad_artifact_id),))

        tables = conn.execute(
            "SELECT table_id, confidence FROM cad_tables WHERE cad_artifact_id=? ORDER BY confidence DESC",
            (int(cad_artifact_id),),
        ).fetchall()

        schemas = 0
        rows_saved = 0
        sample = []

        for t in tables:
            table_id = int(t["table_id"])
            cells = conn.execute(
                "SELECT row_idx, col_idx, cad_entity_id, text, x, y FROM cad_table_cells WHERE table_id=?",
                (table_id,),
            ).fetchall()
            cell_list = [dict(r) for r in cells]

            sch = infer_schema_v2(cell_list)
            conn.execute(
                "INSERT INTO cad_table_schemas(table_id, header_row_idx, columns_json, confidence) VALUES(?,?,?,?)",
                (table_id, (sch.get("header_rows") or [None])[0], json.dumps(sch.get("columns") or {}, ensure_ascii=False), float(sch.get("confidence", 0.5))),
            )
            schemas += 1

            parsed = parse_rows_v2(cell_list, sch)
            for rp in parsed:
                conn.execute(
                    "INSERT INTO cad_table_row_parses(table_id, row_idx, token_norm, story_norm, fields_json, confidence) VALUES(?,?,?,?,?,?)",
                    (table_id, rp["row_idx"], rp.get("token_norm"), rp.get("story_norm"), json.dumps(rp.get("fields") or {}, ensure_ascii=False), float(rp.get("confidence", 0.5))),
                )
                rows_saved += 1
                if len(sample) < 20 and rp.get("token_norm"):
                    sample.append({"table_id": table_id, "row_idx": rp["row_idx"], "token": rp.get("token_norm"), "story": rp.get("story_norm")})

        conn.commit()
        return {"ok": True, "schemas": schemas, "rows": rows_saved, "sample": sample}
    finally:
        conn.close()


@mcp.tool()
def structai_cad_detect_story_tags(
    cad_artifact_id: int,
    overwrite: bool = True,
    include_layers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        if overwrite:
            conn.execute("DELETE FROM cad_story_tags WHERE cad_artifact_id=?", (int(cad_artifact_id),))

        rows = conn.execute(
            """
            SELECT cad_entity_id, layer, x, y, z, text
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


@mcp.tool()
def structai_token_story_map_build(
    cad_artifact_id: int,
    model_id: int,
    max_story_tag_dist: float = 800.0,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        # token->members from confirmed mappings
        maps = conn.execute(
            "SELECT cad_token_norm, model_member_id FROM member_mappings WHERE cad_artifact_id=? AND model_id=? AND status='confirmed'",
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()
        token_to_members: Dict[str, List[int]] = {}
        for r in maps:
            token_to_members.setdefault(r["cad_token_norm"], []).append(int(r["model_member_id"]))

        member_story = {}
        for r in conn.execute("SELECT model_member_id, story FROM model_members WHERE model_id=?", (int(model_id),)).fetchall():
            member_story[int(r["model_member_id"])] = _norm_story_model(r["story"])

        # from table rows
        rows = conn.execute(
            """
            SELECT table_id, row_idx, token_norm, story_norm, fields_json
            FROM cad_table_row_parses
            WHERE token_norm IS NOT NULL AND story_norm IS NOT NULL
            """,
        ).fetchall()

        created = 0
        sample = []

        for r in rows:
            tok = str(r["token_norm"])
            st = str(r["story_norm"])
            candidates = token_to_members.get(tok, [])
            if not candidates:
                continue
            match = [mid for mid in candidates if member_story.get(mid) == st]
            chosen = None
            conf = 0.7
            if len(match) == 1:
                chosen = match[0]
                conf = min(0.98, conf + 0.1)
            elif len(candidates) == 1:
                chosen = candidates[0]
                conf = min(0.9, conf)
            else:
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

        # infer from nearest story tag
        existing = conn.execute(
            "SELECT cad_token_norm, story_norm FROM token_story_maps WHERE cad_artifact_id=? AND model_id=?",
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()
        existing_keys = {(str(r["cad_token_norm"]), str(r["story_norm"])) for r in existing}

        for tok, candidates in token_to_members.items():
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

            counts: Dict[str, int] = {}
            for p in pts:
                st = _nearest_story_for_point(conn, cad_artifact_id, float(p["x"]), float(p["y"]), max_dist=max_story_tag_dist)
                if st:
                    counts[st] = counts.get(st, 0) + 1
            if not counts:
                continue

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


@mcp.tool()
def structai_specs_from_table_rows(
    cad_artifact_id: int,
    model_id: int,
    overwrite_specs: bool = False,
    overwrite_links_suggested: bool = False,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        if overwrite_specs:
            conn.execute("DELETE FROM cad_specs WHERE cad_artifact_id=? AND raw_text LIKE 'TABLE_ROW:%'", (int(cad_artifact_id),))
        if overwrite_links_suggested:
            conn.execute(
                "DELETE FROM member_spec_links WHERE cad_artifact_id=? AND model_id=? AND status='suggested' AND method='table_schema'",
                (int(cad_artifact_id), int(model_id)),
            )

        maps = conn.execute(
            "SELECT cad_token_norm, model_member_id FROM member_mappings WHERE cad_artifact_id=? AND model_id=? AND status='confirmed'",
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()
        token_to_members: Dict[str, List[int]] = {}
        for r in maps:
            token_to_members.setdefault(r["cad_token_norm"], []).append(int(r["model_member_id"]))

        tsm = conn.execute(
            "SELECT cad_token_norm, story_norm, model_member_id, status, confidence FROM token_story_maps WHERE cad_artifact_id=? AND model_id=? AND status IN ('confirmed','suggested')",
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()
        token_story_best: Dict[Tuple[str, str], Tuple[int, float, str]] = {}
        for r in tsm:
            key = (str(r["cad_token_norm"]), str(r["story_norm"]))
            score = float(r["confidence"] or 0.7) + (0.2 if r["status"] == "confirmed" else 0.0)
            cur = token_story_best.get(key)
            if cur is None or score > cur[1]:
                token_story_best[key] = (int(r["model_member_id"]), score, str(r["status"]))

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

            target_member_ids: List[int] = []
            if story:
                key = (tok, str(story))
                if key in token_story_best:
                    target_member_ids = [token_story_best[key][0]]
                else:
                    candidates = token_to_members.get(tok, [])
                    if candidates:
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
                            target_member_ids = candidates
            else:
                candidates = token_to_members.get(tok, [])
                if len(candidates) == 1:
                    target_member_ids = [candidates[0]]
                elif candidates:
                    target_member_ids = candidates

            if not target_member_ids:
                continue

            new_spec_ids: List[Tuple[int, str]] = []

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
                        conf=0.92,
                    )
                    new_spec_ids.append((sid, s["spec_kind"]))

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
                sample.append({"token": tok, "story": story, "specs": [k for _, k in new_spec_ids], "target_member_ids": target_member_ids})

        conn.commit()
        return {"ok": True, "created_specs": created_specs, "created_links": created_links, "sample": sample}
    finally:
        conn.close()


@mcp.tool()
def structai_specs_auto_confirm_table_schema(cad_artifact_id: int, model_id: int) -> Dict[str, Any]:
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

        groups: Dict[Tuple[str, str], List[int]] = {}
        link_info: Dict[int, Tuple[str, str, int]] = {}

        for r in links:
            lid = int(r["link_id"])
            ev = json.loads(r["evidence_json"] or "{}")
            tok = str(ev.get("token_norm") or "")
            st = str(ev.get("story_norm") or "")
            mmid = int(r["model_member_id"])
            if not tok:
                continue
            key = (tok, st)
            groups.setdefault(key, []).append(mmid)
            link_info[lid] = (tok, st, mmid)

        confirm_ids = []
        for lid, (tok, st, mmid) in link_info.items():
            key = (tok, st)
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


@mcp.tool()
def structai_token_story_auto_confirm(cad_artifact_id: int, model_id: int, min_confidence: float = 0.85) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT cad_token_norm, story_norm, COUNT(*) as n
            FROM token_story_maps
            WHERE cad_artifact_id=? AND model_id=? AND status='suggested'
            GROUP BY cad_token_norm, story_norm
            """,
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()

        confirmed = 0
        for r in rows:
            tok = r["cad_token_norm"]
            st = r["story_norm"]
            cand = conn.execute(
                """
                SELECT map_id, confidence
                FROM token_story_maps
                WHERE cad_artifact_id=? AND model_id=? AND status='suggested' AND cad_token_norm=? AND story_norm=?
                ORDER BY confidence DESC
                """,
                (int(cad_artifact_id), int(model_id), tok, st),
            ).fetchall()

            if len(cand) == 1 and float(cand[0]["confidence"] or 0.0) >= float(min_confidence):
                conn.execute(
                    "UPDATE token_story_maps SET status='confirmed', updated_at=datetime('now') WHERE map_id=?",
                    (int(cand[0]["map_id"]),),
                )
                confirmed += 1

        conn.commit()
        return {"ok": True, "confirmed": confirmed}
    finally:
        conn.close()


@mcp.tool()
def structai_token_story_conflicts(cad_artifact_id: int, model_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT cad_token_norm, story_norm, COUNT(*) as n
            FROM token_story_maps
            WHERE cad_artifact_id=? AND model_id=? AND status IN ('suggested','confirmed')
            GROUP BY cad_token_norm, story_norm
            HAVING COUNT(*) >= 2
            ORDER BY n DESC
            """,
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()

        items = []
        for r in rows[:200]:
            tok = r["cad_token_norm"]
            st = r["story_norm"]
            cand = conn.execute(
                """
                SELECT tsm.model_member_id, tsm.confidence, tsm.status, mb.member_uid, mb.member_label, mb.story
                FROM token_story_maps tsm
                JOIN model_members mb ON mb.model_member_id = tsm.model_member_id
                WHERE tsm.cad_artifact_id=? AND tsm.model_id=? AND tsm.cad_token_norm=? AND tsm.story_norm=?
                ORDER BY tsm.status DESC, tsm.confidence DESC
                """,
                (int(cad_artifact_id), int(model_id), tok, st),
            ).fetchall()
            items.append({"token": tok, "story": st, "candidates": [dict(x) for x in cand]})

        return {"ok": True, "conflicts": len(items), "items": items}
    finally:
        conn.close()

@mcp.tool()
def structai_sections_migrate_add_priority() -> Dict[str, Any]:
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


def _upsert_section(conn: sqlite3.Connection, family: str, name: str, dims: dict, props: dict, source: str, priority: int = 50) -> int:
    name_norm = _norm(name)
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
        (family, name, name_norm, json.dumps(dims, ensure_ascii=False), json.dumps(props, ensure_ascii=False), source, int(priority)),
    )
    r = conn.execute(
        "SELECT section_id FROM section_catalog WHERE family=? AND name_norm=?",
        (family, name_norm),
    ).fetchone()
    return int(r["section_id"])


@mcp.tool()
def structai_sections_add_alias(section_id: int, aliases: List[str]) -> Dict[str, Any]:
    conn = _connect()
    try:
        for a in aliases:
            if not a:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO section_aliases(alias_norm, section_id) VALUES(?,?)",
                (_norm(a), int(section_id)),
            )
        conn.commit()
        return {"ok": True, "section_id": int(section_id), "aliases": len(aliases)}
    finally:
        conn.close()


@mcp.tool()
def structai_sections_import_catalog(path: str, fmt: Optional[str] = None) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    ext = p.suffix.lower()
    if fmt is None:
        fmt = "json" if ext == ".json" else "csv"

    conn = _connect()
    try:
        imported = 0
        computed = 0

        def priority_for(source: str) -> int:
            s = (source or "").lower()
            if "ks" in s or "catalog" in s:
                return 10
            if "computed" in s:
                return 80
            return 50

        if fmt == "json":
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            items = data.get("items") if isinstance(data, dict) else data
            if not isinstance(items, list):
                raise ValueError("JSON must be list or {items:[...]}")

            for it in items:
                family = str(it.get("family") or "").strip()
                name = str(it.get("name") or "").strip()
                dims = it.get("dims") or {}
                props = it.get("props") or {}
                source = str(it.get("source") or "json_import")
                priority = int(it.get("priority") or priority_for(source))
                if not family or not name:
                    continue

                if family == "steel_h" and ("Zx_mm3" not in props or "Aw_mm2" not in props):
                    d = SteelHSectionDims(
                        H=float(dims.get("H_mm")),
                        B=float(dims.get("B_mm")),
                        tw=float(dims.get("tw_mm")),
                        tf=float(dims.get("tf_mm")),
                    )
                    calc = compute_h_section_props(d)
                    props = {**calc["props"], **props}
                    computed += 1
                if family == "steel_box" and ("Zx_mm3" not in props or "Aw_mm2" not in props):
                    calc = compute_box_props(SteelBoxDims(H=float(dims.get("H_mm")), B=float(dims.get("B_mm")), t=float(dims.get("t_mm"))))
                    props = {**calc["props"], **props}
                    computed += 1
                if family == "steel_pipe" and ("Zx_mm3" not in props or "Aw_mm2" not in props):
                    calc = compute_pipe_props(SteelPipeDims(D=float(dims.get("D_mm")), t=float(dims.get("t_mm"))))
                    props = {**calc["props"], **props}
                    computed += 1
                if family == "steel_channel" and ("Zx_mm3" not in props or "Aw_mm2" not in props):
                    calc = compute_channel_props(SteelChannelDims(H=float(dims.get("H_mm")), B=float(dims.get("B_mm")), tw=float(dims.get("tw_mm")), tf=float(dims.get("tf_mm"))))
                    props = {**calc["props"], **props}
                    computed += 1
                if family == "steel_angle" and ("Zx_mm3" not in props or "Aw_mm2" not in props):
                    calc = compute_angle_props(SteelAngleDims(b=float(dims.get("b_mm")), d=float(dims.get("d_mm")), t=float(dims.get("t_mm"))))
                    props = {**calc["props"], **props}
                    computed += 1

                _upsert_section(conn, family, name, dims, props, source, priority=priority)
                imported += 1

        elif fmt == "csv":
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    family = str(row.get("family") or "").strip()
                    name = str(row.get("name") or "").strip()
                    if not family or not name:
                        continue
                    dims: Dict[str, Any] = {}
                    props: Dict[str, Any] = {}
                    source = row.get("source") or "csv_import"
                    priority = int(row.get("priority") or priority_for(source))

                    for k in ("H_mm", "B_mm", "tw_mm", "tf_mm", "t_mm", "D_mm", "b_mm", "d_mm"):
                        if row.get(k) not in (None, ""):
                            dims[k] = float(row[k])

                    for k in ("Fy_MPa", "Zx_mm3", "Aw_mm2", "Ix_mm4", "Iy_mm4", "A_mm2"):
                        if row.get(k) not in (None, ""):
                            props[k] = float(row[k])

                    if family == "steel_h" and ("Zx_mm3" not in props or "Aw_mm2" not in props):
                        d = SteelHSectionDims(
                            H=float(dims.get("H_mm")),
                            B=float(dims.get("B_mm")),
                            tw=float(dims.get("tw_mm")),
                            tf=float(dims.get("tf_mm")),
                        )
                        calc = compute_h_section_props(d)
                        props = {**calc["props"], **props}
                        computed += 1
                    if family == "steel_box" and ("Zx_mm3" not in props or "Aw_mm2" not in props):
                        calc = compute_box_props(SteelBoxDims(H=float(dims.get("H_mm")), B=float(dims.get("B_mm")), t=float(dims.get("t_mm"))))
                        props = {**calc["props"], **props}
                        computed += 1
                    if family == "steel_pipe" and ("Zx_mm3" not in props or "Aw_mm2" not in props):
                        calc = compute_pipe_props(SteelPipeDims(D=float(dims.get("D_mm")), t=float(dims.get("t_mm"))))
                        props = {**calc["props"], **props}
                        computed += 1
                    if family == "steel_channel" and ("Zx_mm3" not in props or "Aw_mm2" not in props):
                        calc = compute_channel_props(SteelChannelDims(H=float(dims.get("H_mm")), B=float(dims.get("B_mm")), tw=float(dims.get("tw_mm")), tf=float(dims.get("tf_mm"))))
                        props = {**calc["props"], **props}
                        computed += 1
                    if family == "steel_angle" and ("Zx_mm3" not in props or "Aw_mm2" not in props):
                        calc = compute_angle_props(SteelAngleDims(b=float(dims.get("b_mm")), d=float(dims.get("d_mm")), t=float(dims.get("t_mm"))))
                        props = {**calc["props"], **props}
                        computed += 1

                    _upsert_section(conn, family, name, dims, props, source, priority=priority)
                    imported += 1
        else:
            raise ValueError("unsupported fmt")

        conn.commit()
        return {"ok": True, "imported": imported, "computed_props": computed}
    finally:
        conn.close()


@mcp.tool()
def structai_sections_resolve_members(
    model_id: int,
    default_Fy_MPa: Optional[float] = None,
    overwrite_design: bool = False,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        members = conn.execute(
            "SELECT model_member_id, member_uid, type, section FROM model_members WHERE model_id=?",
            (int(model_id),),
        ).fetchall()

        updated = 0
        created_sections = 0
        skipped = []

        spec_to_family = {
            "steel_h_section": "steel_h",
            "steel_box_section": "steel_box",
            "steel_pipe_section": "steel_pipe",
            "steel_channel_section": "steel_channel",
            "steel_angle_section": "steel_angle",
        }

        for m in members:
            mmid = int(m["model_member_id"])
            sec = (m["section"] or "").strip()
            if not sec:
                continue

            row = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
            dj = json.loads(row["design_json"]) if row else {}
            if not overwrite_design and (dj.get("Zx") or dj.get("Aw")):
                skipped.append({"uid": m["member_uid"], "reason": "already has Zx/Aw"})
                continue

            specs = parse_specs_from_text(sec)
            found = None
            for s in specs:
                if s.get("spec_kind") in spec_to_family:
                    found = s
                    break
            if not found:
                continue

            family = spec_to_family[found["spec_kind"]]
            name_norm = _norm(sec)
            section_id = None
            resolved = None
            method = "parsed"
            confidence = 0.6

            alias = conn.execute("SELECT section_id FROM section_aliases WHERE alias_norm=?", (name_norm,)).fetchone()
            if alias:
                section_id = int(alias["section_id"])
                row = conn.execute("SELECT name, dims_json, props_json FROM section_catalog WHERE section_id=?", (section_id,)).fetchone()
                if row:
                    resolved = {"family": family, "name": row["name"], "dims": json.loads(row["dims_json"] or "{}"), "props": json.loads(row["props_json"] or "{}")} 
                    method = "alias"
                    confidence = 0.85

            if not resolved:
                row = conn.execute(
                    "SELECT section_id, name, dims_json, props_json, source FROM section_catalog WHERE family=? AND name_norm=?",
                    (family, name_norm),
                ).fetchone()
                if row:
                    section_id = int(row["section_id"])
                    resolved = {"family": family, "name": row["name"], "dims": json.loads(row["dims_json"] or "{}"), "props": json.loads(row["props_json"] or "{}")} 
                    method = "catalog"
                    confidence = 0.85

            if not resolved:
                dims = {}
                props = {}
                if family == "steel_h":
                    dims = {"H_mm": found.get("H_mm"), "B_mm": found.get("B_mm"), "tw_mm": found.get("tw_mm"), "tf_mm": found.get("tf_mm")}
                    calc = compute_h_section_props(SteelHSectionDims(H=float(dims["H_mm"]), B=float(dims["B_mm"]), tw=float(dims["tw_mm"]), tf=float(dims["tf_mm"])))
                    props = calc["props"]
                elif family == "steel_box":
                    dims = {"H_mm": found.get("H_mm"), "B_mm": found.get("B_mm"), "t_mm": found.get("t_mm")}
                    props = compute_box_props(SteelBoxDims(H=float(dims["H_mm"]), B=float(dims["B_mm"]), t=float(dims["t_mm"])))["props"]
                elif family == "steel_pipe":
                    dims = {"D_mm": found.get("D_mm"), "t_mm": found.get("t_mm")}
                    props = compute_pipe_props(SteelPipeDims(D=float(dims["D_mm"]), t=float(dims["t_mm"])))["props"]
                elif family == "steel_channel":
                    dims = {"H_mm": found.get("H_mm"), "B_mm": found.get("B_mm"), "tw_mm": found.get("tw_mm"), "tf_mm": found.get("tf_mm")}
                    props = compute_channel_props(SteelChannelDims(H=float(dims["H_mm"]), B=float(dims["B_mm"]), tw=float(dims["tw_mm"]), tf=float(dims["tf_mm"])))["props"]
                elif family == "steel_angle":
                    dims = {"b_mm": found.get("b_mm"), "d_mm": found.get("d_mm"), "t_mm": found.get("t_mm")}
                    props = compute_angle_props(SteelAngleDims(b=float(dims["b_mm"]), d=float(dims["d_mm"]), t=float(dims["t_mm"])))["props"]

                section_name = sec
                section_id = _upsert_section(conn, family, section_name, dims, props, "computed", priority=80)
                resolved = {"family": family, "name": section_name, "dims": dims, "props": props}
                method = "parsed+computed"
                confidence = 0.7
                created_sections += 1

            props = resolved.get("props") or {}
            if "Zx_mm3" in props:
                dj["Zx"] = float(props["Zx_mm3"])
            if "Aw_mm2" in props:
                dj["Aw"] = float(props["Aw_mm2"])
            if "Ix_mm4" in props:
                dj["Ix"] = float(props["Ix_mm4"])
            if "A_mm2" in props:
                dj["A"] = float(props["A_mm2"])
            if default_Fy_MPa is not None:
                dj.setdefault("Fy", float(default_Fy_MPa))
            elif "Fy_MPa" in props:
                dj.setdefault("Fy", float(props["Fy_MPa"]))

            dj["section_resolved"] = resolved
            dj.setdefault("units", {})
            dj["units"].update({"length": "mm", "stress": "MPa"})

            conn.execute(
                """
                INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
                VALUES(?,?, datetime('now'))
                ON CONFLICT(model_member_id) DO UPDATE SET
                  design_json=excluded.design_json,
                  updated_at=datetime('now')
                """,
                (mmid, json.dumps(dj, ensure_ascii=False)),
            )

            conn.execute(
                """
                INSERT INTO member_section_resolutions(model_member_id, section_id, resolved_name, confidence, method, updated_at)
                VALUES(?,?,?,?,?, datetime('now'))
                ON CONFLICT(model_member_id) DO UPDATE SET
                  section_id=excluded.section_id,
                  resolved_name=excluded.resolved_name,
                  confidence=excluded.confidence,
                  method=excluded.method,
                  updated_at=datetime('now')
                """,
                (mmid, section_id, resolved.get("name"), float(confidence), str(method)),
            )

            updated += 1

        conn.commit()
        return {"ok": True, "updated": updated, "created_sections": created_sections, "skipped": skipped[:200]}
    finally:
        conn.close()

@mcp.tool()
def structai_validate_ready_for_checks(
    model_id: int,
    analysis_run_id: int,
    check_types: Optional[List[str]] = None,
    only_mapped_from_cad_artifact_id: Optional[int] = None,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        rulepack = _get_active_rulepack(conn)
        checks_def = rulepack.get("checks") or {}
        if not check_types:
            check_types = list(checks_def.keys())

        if only_mapped_from_cad_artifact_id is not None:
            mids = conn.execute(
                """
                SELECT DISTINCT mm.model_member_id
                FROM member_mappings mm
                WHERE mm.model_id=? AND mm.cad_artifact_id=? AND mm.status='confirmed'
                """,
                (int(model_id), int(only_mapped_from_cad_artifact_id)),
            ).fetchall()
            member_ids = [int(r["model_member_id"]) for r in mids]
        else:
            mids = conn.execute("SELECT model_member_id FROM model_members WHERE model_id=?", (int(model_id),)).fetchall()
            member_ids = [int(r["model_member_id"]) for r in mids]

        issues = []
        ok_count = 0

        for mid in member_ids:
            mb = conn.execute(
                "SELECT member_uid, member_label, type FROM model_members WHERE model_member_id=?",
                (int(mid),),
            ).fetchone()

            di = conn.execute(
                "SELECT design_json FROM member_design_inputs WHERE model_member_id=?",
                (int(mid),),
            ).fetchone()
            design = json.loads(di["design_json"]) if di else {}
            design_keys = set(design.keys())

            rr = conn.execute(
                "SELECT envelope_json FROM member_results WHERE analysis_run_id=? AND model_member_id=? LIMIT 1",
                (int(analysis_run_id), int(mid)),
            ).fetchone()
            if rr:
                env = json.loads(rr["envelope_json"])
                env = enrich_envelope(env)
                env_keys = set(env.keys())
            else:
                env = {}
                env_keys = set()

            member_issue = {
                "member_uid": mb["member_uid"],
                "member_label": mb["member_label"],
                "type": mb["type"],
                "missing_results": rr is None,
                "checks": [],
            }

            any_problem = False

            for ct in check_types:
                cd = checks_def.get(ct) or {}
                ratio_expr = cd.get("ratio_expr")
                demand_expr = cd.get("demand_expr")
                capacity_expr = cd.get("capacity_expr")

                missing_env = []
                missing_design = []

                if ratio_expr:
                    for n in extract_names(str(ratio_expr)):
                        if n in design_keys:
                            continue
                        if n in env_keys:
                            continue
                        if "_" in n or n.isupper():
                            missing_env.append(n)
                        else:
                            missing_design.append(n)
                else:
                    for n in extract_names(str(demand_expr or "")):
                        if n not in env_keys:
                            missing_env.append(n)
                    for n in extract_names(str(capacity_expr or "")):
                        if n not in design_keys:
                            missing_design.append(n)

                if missing_env or missing_design or rr is None:
                    any_problem = True

                member_issue["checks"].append({
                    "check_type": ct,
                    "missing_env": sorted(set(missing_env)),
                    "missing_design": sorted(set(missing_design)),
                })

            if any_problem:
                issues.append(member_issue)
            else:
                ok_count += 1

        return {
            "ok": True,
            "model_id": int(model_id),
            "analysis_run_id": int(analysis_run_id),
            "rulepack": {"name": rulepack.get("name"), "version": rulepack.get("version")},
            "ready_members": ok_count,
            "problem_members": len(issues),
            "issues": issues[:200],
        }
    finally:
        conn.close()


@mcp.tool()
def structai_quality_summary(model_id: int, analysis_run_id: Optional[int] = None) -> Dict[str, Any]:
    conn = _connect()
    try:
        members = conn.execute(
            "SELECT model_member_id, member_uid, member_label, type, section FROM model_members WHERE model_id=?",
            (int(model_id),),
        ).fetchall()

        steel_missing = []
        rc_missing = []
        results_missing = []

        for m in members:
            mmid = int(m["model_member_id"])
            di = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
            dj = json.loads(di["design_json"]) if di else {}

            mtype = (m["type"] or "").lower()
            if mtype in ("beam", "girder", "steel_beam", "steel"):
                if not dj.get("Zx") or not dj.get("Aw"):
                    steel_missing.append({"uid": m["member_uid"], "label": m["member_label"], "need": ["Zx", "Aw"], "section": m["section"]})

            if mtype in ("beam", "rc_beam"):
                need = []
                for k in ("b", "h"):
                    if not dj.get(k):
                        need.append(k)
                if not (dj.get("As_top") or dj.get("As")):
                    need.append("As_top(or As)")
                if not (dj.get("As_bot") or dj.get("As")):
                    need.append("As_bot(or As)")
                if not dj.get("Av"):
                    need.append("Av")
                if not dj.get("s"):
                    need.append("s")
                if need:
                    rc_missing.append({"uid": m["member_uid"], "label": m["member_label"], "need": need})

            if analysis_run_id is not None:
                rr = conn.execute(
                    "SELECT 1 FROM member_results WHERE analysis_run_id=? AND model_member_id=?",
                    (int(analysis_run_id), mmid),
                ).fetchone()
                if not rr:
                    results_missing.append({"uid": m["member_uid"], "label": m["member_label"]})

        return {
            "ok": True,
            "model_id": int(model_id),
            "analysis_run_id": int(analysis_run_id) if analysis_run_id is not None else None,
            "steel_missing_count": len(steel_missing),
            "rc_missing_count": len(rc_missing),
            "results_missing_count": len(results_missing),
            "steel_missing_sample": steel_missing[:50],
            "rc_missing_sample": rc_missing[:50],
            "results_missing_sample": results_missing[:50],
        }
    finally:
        conn.close()

@mcp.tool()
def structai_benchmark_import(
    path: str,
    name: str,
    version: str,
    kind: str = "commercial",
    source: str = "",
    fmt: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    ext = p.suffix.lower()
    if fmt is None:
        fmt = "json" if ext == ".json" else "csv"

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO benchmarks(name, version, kind, source, meta_json)
            VALUES(?,?,?,?,?)
            ON CONFLICT(name, version) DO UPDATE SET
              kind=excluded.kind,
              source=excluded.source
            """,
            (name, version, kind, source, "{}"),
        )
        row = conn.execute("SELECT benchmark_id FROM benchmarks WHERE name=? AND version=?", (name, version)).fetchone()
        benchmark_id = int(row["benchmark_id"])

        if overwrite:
            conn.execute("DELETE FROM benchmark_results WHERE benchmark_id=?", (benchmark_id,))

        inserted = 0
        if fmt == "json":
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            items = data.get("items") if isinstance(data, dict) else data
            if not isinstance(items, list):
                raise ValueError("benchmark json must be list or {items:[...]}")
            for it in items:
                conn.execute(
                    """
                    INSERT INTO benchmark_results(
                      benchmark_id, member_uid, story_norm, check_type, combo,
                      demand_value, capacity_value, ratio, status, meta_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(benchmark_id, member_uid, COALESCE(story_norm,''), check_type, combo) DO UPDATE SET
                      demand_value=excluded.demand_value,
                      capacity_value=excluded.capacity_value,
                      ratio=excluded.ratio,
                      status=excluded.status,
                      meta_json=excluded.meta_json
                    """,
                    (
                        benchmark_id,
                        str(it.get("member_uid") or ""),
                        str(it.get("story") or it.get("story_norm") or ""),
                        str(it.get("check_type") or ""),
                        str(it.get("combo") or "LC1"),
                        it.get("demand"),
                        it.get("capacity"),
                        it.get("ratio"),
                        it.get("status"),
                        json.dumps(it.get("meta") or {}, ensure_ascii=False),
                    ),
                )
                inserted += 1
        else:
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    conn.execute(
                        """
                        INSERT INTO benchmark_results(
                          benchmark_id, member_uid, story_norm, check_type, combo,
                          demand_value, capacity_value, ratio, status, meta_json
                        ) VALUES(?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(benchmark_id, member_uid, COALESCE(story_norm,''), check_type, combo) DO UPDATE SET
                          demand_value=excluded.demand_value,
                          capacity_value=excluded.capacity_value,
                          ratio=excluded.ratio,
                          status=excluded.status,
                          meta_json=excluded.meta_json
                        """,
                        (
                            benchmark_id,
                            str(row.get("member_uid") or ""),
                            str(row.get("story") or row.get("story_norm") or ""),
                            str(row.get("check_type") or ""),
                            str(row.get("combo") or "LC1"),
                            float(row.get("demand")) if row.get("demand") not in (None, "") else None,
                            float(row.get("capacity")) if row.get("capacity") not in (None, "") else None,
                            float(row.get("ratio")) if row.get("ratio") not in (None, "") else None,
                            row.get("status"),
                            json.dumps({"source_note": row.get("source_note")}, ensure_ascii=False),
                        ),
                    )
                    inserted += 1

        conn.commit()
        return {"ok": True, "benchmark_id": benchmark_id, "inserted": inserted}
    finally:
        conn.close()


@mcp.tool()
def structai_compare_check_run_to_benchmark(
    check_run_id: int,
    benchmark_id: int,
    ratio_tol: float = 0.01,
    ratio_warn: float = 0.03,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO compare_runs(check_run_id, benchmark_id, ratio_tol, ratio_warn, summary_json) VALUES(?,?,?,?, '{}')",
            (int(check_run_id), int(benchmark_id), float(ratio_tol), float(ratio_warn)),
        )
        compare_id = int(cur.lastrowid)

        # actual results
        actual_rows = conn.execute(
            """
            SELECT cr.check_type, cr.combo, cr.ratio, mb.member_uid, mb.story
            FROM check_results cr
            JOIN model_members mb ON mb.model_member_id = cr.model_member_id
            WHERE cr.check_run_id=?
            """,
            (int(check_run_id),),
        ).fetchall()
        actual_map = {}
        for r in actual_rows:
            key = (r["member_uid"], str(r["story"] or ""), r["check_type"], r["combo"])
            actual_map[key] = float(r["ratio"]) if r["ratio"] is not None else None

        bench_rows = conn.execute(
            "SELECT member_uid, story_norm, check_type, combo, ratio FROM benchmark_results WHERE benchmark_id=?",
            (int(benchmark_id),),
        ).fetchall()

        items = []
        for r in bench_rows:
            key = (r["member_uid"], str(r["story_norm"] or ""), r["check_type"], r["combo"])
            expected = float(r["ratio"]) if r["ratio"] is not None else None
            actual = actual_map.get(key)
            if actual is None:
                sev = "MISSING_ACTUAL"
                abs_diff = None
                rel_diff = None
            else:
                abs_diff = abs(actual - expected) if expected is not None else None
                rel_diff = abs_diff / expected if expected not in (None, 0) and abs_diff is not None else None
                if abs_diff is None:
                    sev = "OK"
                elif abs_diff > ratio_warn:
                    sev = "DIFF"
                elif abs_diff > ratio_tol:
                    sev = "WARN"
                else:
                    sev = "OK"

            items.append((key, expected, actual, abs_diff, rel_diff, sev))

        # missing expected
        for key, actual in actual_map.items():
            if not any(key == i[0] for i in items):
                items.append((key, None, actual, None, None, "MISSING_EXPECTED"))

        summary = {"OK": 0, "WARN": 0, "DIFF": 0, "MISSING_EXPECTED": 0, "MISSING_ACTUAL": 0}
        for key, expected, actual, abs_diff, rel_diff, sev in items:
            summary[sev] = summary.get(sev, 0) + 1
            conn.execute(
                """
                INSERT INTO compare_items(compare_id, member_uid, story_norm, check_type, combo, expected_ratio, actual_ratio, abs_diff, rel_diff, severity)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    compare_id,
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    expected,
                    actual,
                    abs_diff,
                    rel_diff,
                    sev,
                ),
            )

        conn.execute(
            "UPDATE compare_runs SET summary_json=? WHERE compare_id=?",
            (json.dumps(summary, ensure_ascii=False), int(compare_id)),
        )
        conn.commit()
        return {"ok": True, "compare_id": compare_id, "summary": summary}
    finally:
        conn.close()


@mcp.tool()
def structai_compare_report_generate(
    compare_id: int,
    formats: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    formats = formats or ["md"]
    out_dir_path = Path(out_dir).expanduser().resolve() if out_dir else (DB_PATH.parent / "compare_reports")
    out_dir_path.mkdir(parents=True, exist_ok=True)

    conn = _connect()
    try:
        run = conn.execute(
            "SELECT compare_id, check_run_id, benchmark_id, summary_json, created_at FROM compare_runs WHERE compare_id=?",
            (int(compare_id),),
        ).fetchone()
        if not run:
            raise ValueError("compare run not found")

        rows = conn.execute(
            "SELECT * FROM compare_items WHERE compare_id=? ORDER BY severity DESC, abs_diff DESC LIMIT 200",
            (int(compare_id),),
        ).fetchall()
        items = [dict(r) for r in rows]
        summary = json.loads(run["summary_json"] or "{}")

        outputs = []
        if "md" in formats:
            md = []
            md.append("# Compare Report")
            md.append("")
            md.append(f"- Compare ID: {compare_id}")
            md.append(f"- Check Run: {run['check_run_id']}")
            md.append(f"- Benchmark: {run['benchmark_id']}")
            md.append("")
            md.append("## Summary")
            for k, v in summary.items():
                md.append(f"- {k}: {v}")
            md.append("")
            md.append("## Items")
            md.append("| Severity | Member | Story | Check | Combo | Expected | Actual | Abs Diff | Rel Diff |")
            md.append("|---|---|---|---|---|---:|---:|---:|---:|")
            for it in items:
                md.append(
                    f"| {it.get('severity')} | {it.get('member_uid')} | {it.get('story_norm') or ''} | {it.get('check_type')} | {it.get('combo')} | "
                    f"{it.get('expected_ratio')} | {it.get('actual_ratio')} | {it.get('abs_diff')} | {it.get('rel_diff')} |"
                )
            md_text = "\n".join(md)
            md_path = out_dir_path / f"compare_{compare_id}.md"
            md_path.write_text(md_text, encoding="utf-8")
            sha = _sha256_file(md_path)
            uri = md_path.as_uri()
            art_id = _upsert_artifact(conn, uri=uri, kind="compare_report_md", title=f"Compare Report #{compare_id}", source_path=str(md_path), sha256=sha)
            conn.execute("INSERT INTO compare_reports(compare_id, artifact_id, format) VALUES(?,?,?)", (int(compare_id), art_id, "md"))
            outputs.append({"format": "md", "path": str(md_path), "uri": uri, "artifact_id": art_id})

        conn.commit()
        return {"ok": True, "compare_id": int(compare_id), "outputs": outputs}
    finally:
        conn.close()


@mcp.tool()
def structai_compare_list_runs(limit: int = 20) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT compare_id, check_run_id, benchmark_id, summary_json, created_at FROM compare_runs ORDER BY compare_id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        items = []
        for r in rows:
            it = dict(r)
            it["summary"] = json.loads(it.get("summary_json") or "{}")
            items.append(it)
        return {"ok": True, "items": items}
    finally:
        conn.close()


@mcp.tool()
def structai_compare_read_run(compare_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        cr = conn.execute(
            "SELECT compare_id, check_run_id, benchmark_id, summary_json, created_at FROM compare_runs WHERE compare_id=?",
            (int(compare_id),),
        ).fetchone()
        if not cr:
            raise ValueError("compare run not found")

        items = conn.execute(
            "SELECT * FROM compare_items WHERE compare_id=? ORDER BY severity DESC, abs_diff DESC LIMIT 200",
            (int(compare_id),),
        ).fetchall()

        reports = conn.execute(
            """
            SELECT cr.compare_report_id, cr.format, a.uri
            FROM compare_reports cr
            JOIN artifacts a ON a.artifact_id = cr.artifact_id
            WHERE cr.compare_id=?
            ORDER BY cr.compare_report_id DESC
            """,
            (int(compare_id),),
        ).fetchall()

        return {
            "ok": True,
            "compare": dict(cr),
            "reports": [dict(r) for r in reports],
            "top_items": [dict(r) for r in items],
        }
    finally:
        conn.close()


@mcp.tool()
def structai_templates_import(path: str) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    name = data.get("name") or p.stem
    ver = data.get("version") or "0.0"
    templates = data.get("templates") or data

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO check_template_sets(name, version, templates_json, is_active)
            VALUES(?,?,?,0)
            ON CONFLICT(name, version) DO UPDATE SET
              templates_json=excluded.templates_json
            """,
            (name, ver, json.dumps(templates, ensure_ascii=False)),
        )
        conn.commit()
        return {"ok": True, "name": name, "version": ver}
    finally:
        conn.close()


@mcp.tool()
def structai_templates_set_active(name: str, version: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute("UPDATE check_template_sets SET is_active=0")
        conn.execute("UPDATE check_template_sets SET is_active=1 WHERE name=? AND version=?", (name, version))
        conn.commit()
        return {"ok": True, "active": {"name": name, "version": version}}
    finally:
        conn.close()


@mcp.tool()
def structai_rulepack_generate_from_templates(
    new_rulepack_name: str,
    new_rulepack_version: str,
    activate: bool = False,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT templates_json FROM check_template_sets WHERE is_active=1 ORDER BY template_set_id DESC LIMIT 1"
        ).fetchone()
        if not row:
            raise ValueError("no active template set")
        templates = json.loads(row["templates_json"] or "{}");

        cb = _get_active_codebook(conn) or {}
        cite_map = cb.get("citations") or {}

        checks = {}
        if isinstance(templates, dict):
            items = templates.get("items") or templates.get("checks") or templates
        else:
            items = templates
        if isinstance(items, list):
            for t in items:
                ct = t.get("check_type") or t.get("name")
                if not ct:
                    continue
                checks[ct] = {
                    "demand_expr": t.get("demand_expr"),
                    "capacity_expr": t.get("capacity_expr"),
                    "ratio_expr": t.get("ratio_expr"),
                    "limit": t.get("limit", 1.0),
                    "warn": t.get("warn", 0.95),
                    "citations": cite_map.get(ct) or t.get("citations") or [],
                }
        elif isinstance(items, dict):
            for ct, t in items.items():
                checks[ct] = {
                    "demand_expr": t.get("demand_expr"),
                    "capacity_expr": t.get("capacity_expr"),
                    "ratio_expr": t.get("ratio_expr"),
                    "limit": t.get("limit", 1.0),
                    "warn": t.get("warn", 0.95),
                    "citations": cite_map.get(ct) or t.get("citations") or [],
                }

        rulepack = {"name": new_rulepack_name, "version": new_rulepack_version, "checks": checks}

        conn.execute(
            """
            INSERT INTO rulepacks(name, version, rulepack_json, is_active)
            VALUES(?,?,?,0)
            ON CONFLICT(name, version) DO UPDATE SET
              rulepack_json=excluded.rulepack_json
            """,
            (new_rulepack_name, new_rulepack_version, json.dumps(rulepack, ensure_ascii=False)),
        )
        if activate:
            rp_id = conn.execute("SELECT rulepack_id FROM rulepacks WHERE name=? AND version=?", (new_rulepack_name, new_rulepack_version)).fetchone()
            if rp_id:
                conn.execute("UPDATE rulepacks SET is_active=0")
                conn.execute("UPDATE rulepacks SET is_active=1 WHERE rulepack_id=?", (int(rp_id["rulepack_id"]),))

        conn.commit()
        return {"ok": True, "rulepack": {"name": new_rulepack_name, "version": new_rulepack_version}, "activated": activate}
    finally:
        conn.close()


def _metrics_for_check_run(conn: sqlite3.Connection, check_run_id: int) -> Dict[str, Any]:
    overall = {r["status"]: int(r["n"]) for r in conn.execute(
        "SELECT status, COUNT(*) as n FROM check_results WHERE check_run_id=? GROUP BY status",
        (int(check_run_id),),
    ).fetchall()}

    by_check_type: Dict[str, Any] = {}
    cts = conn.execute(
        "SELECT DISTINCT check_type FROM check_results WHERE check_run_id=?",
        (int(check_run_id),),
    ).fetchall()
    for ctrow in cts:
        ct = ctrow["check_type"]
        counts = {r["status"]: int(r["n"]) for r in conn.execute(
            "SELECT status, COUNT(*) as n FROM check_results WHERE check_run_id=? AND check_type=? GROUP BY status",
            (int(check_run_id), ct),
        ).fetchall()}
        worst = conn.execute(
            "SELECT ratio FROM check_results WHERE check_run_id=? AND check_type=? AND ratio IS NOT NULL ORDER BY ratio DESC LIMIT 1",
            (int(check_run_id), ct),
        ).fetchone()
        by_check_type[ct] = {**counts, "worst_ratio": float(worst["ratio"]) if worst else None}
    return {"overall": overall, "by_check_type": by_check_type}


def _compare_metrics(golden: Dict[str, Any], actual: Dict[str, Any], ratio_tol: float = 1e-3) -> Dict[str, Any]:
    diff = {"ok": True, "overall": {}, "by_check_type": {}}

    g_over = golden.get("overall", {})
    a_over = actual.get("overall", {})
    for k in set(g_over.keys()) | set(a_over.keys()):
        if int(g_over.get(k, 0)) != int(a_over.get(k, 0)):
            diff["overall"][k] = {"golden": int(g_over.get(k, 0)), "actual": int(a_over.get(k, 0))}
            diff["ok"] = False

    g_ct = golden.get("by_check_type", {})
    a_ct = actual.get("by_check_type", {})
    for ct in set(g_ct.keys()) | set(a_ct.keys()):
        gd = g_ct.get(ct, {})
        ad = a_ct.get(ct, {})
        cd: Dict[str, Any] = {}

        for k in ("PASS", "WARN", "FAIL", "NA"):
            if int(gd.get(k, 0)) != int(ad.get(k, 0)):
                cd[k] = {"golden": int(gd.get(k, 0)), "actual": int(ad.get(k, 0))}

        gw = gd.get("worst_ratio")
        aw = ad.get("worst_ratio")
        if (gw is not None) and (aw is not None):
            if abs(float(gw) - float(aw)) > float(ratio_tol):
                cd["worst_ratio"] = {"golden": float(gw), "actual": float(aw), "tol": float(ratio_tol)}
        elif gw != aw:
            cd["worst_ratio"] = {"golden": gw, "actual": aw}

        if cd:
            diff["by_check_type"][ct] = cd
            diff["ok"] = False

    return diff


@mcp.tool()
def structai_regression_run_case(
    fixture_json: Dict[str, Any],
    isolated_db: bool = True,
    keep_db: bool = False,
) -> Dict[str, Any]:
    reg = _tool_registry()
    run_id = uuid.uuid4().hex[:10]

    if isolated_db:
        db_path = (DB_PATH.parent / "regression_dbs" / f"reg_{run_id}.sqlite")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _apply_schema_if_needed(db_path)
    else:
        db_path = DB_PATH

    vars_ = dict(fixture_json.get("vars") or {})
    steps = fixture_json.get("steps") or []
    if not isinstance(steps, list):
        raise ValueError("fixture_json.steps must be a list")

    def _execute():
        for st in steps:
            tool = str(st.get("tool") or "")
            if tool not in reg:
                raise ValueError(f"tool not found: {tool}")
            args = _resolve(st.get("args") or {}, vars_)
            out = reg[tool](**args)
            save = st.get("save") or {}
            if isinstance(save, dict):
                for var_name, path in save.items():
                    vars_[str(var_name)] = _get_by_path(out, str(path))

        final = fixture_json.get("final") or {}
        check_run_id = None
        if isinstance(final, dict) and "check_run_id" in final:
            check_run_id = int(_resolve(final["check_run_id"], vars_))
        elif "check_run_id" in vars_:
            check_run_id = int(vars_["check_run_id"])
        if not check_run_id:
            raise ValueError("check_run_id not produced by fixture")

        conn = _connect()
        try:
            metrics = _metrics_for_check_run(conn, check_run_id)
        finally:
            conn.close()

        return check_run_id, metrics

    if isolated_db:
        with _db_override(db_path):
            check_run_id, metrics = _execute()
    else:
        check_run_id, metrics = _execute()

    if isolated_db and (not keep_db):
        try:
            db_path.unlink()
        except Exception:
            pass

    return {
        "ok": True,
        "isolated_db": isolated_db,
        "db_path": str(db_path) if (isolated_db and keep_db) else None,
        "check_run_id": int(check_run_id),
        "metrics": metrics,
    }


@mcp.tool()
def structai_regression_update_golden_case(
    suite_name: str,
    case_name: str,
    isolated_db: bool = True,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        s = conn.execute("SELECT suite_id FROM regression_suites WHERE name=?", (suite_name,)).fetchone()
        if not s:
            raise ValueError("suite not found")
        sid = int(s["suite_id"])
        c = conn.execute(
            "SELECT case_id, fixture_json FROM regression_cases WHERE suite_id=? AND name=?",
            (sid, case_name),
        ).fetchone()
        if not c:
            raise ValueError("case not found")
        fixture = json.loads(c["fixture_json"] or "{}")
    finally:
        conn.close()

    r = structai_regression_run_case(fixture_json=fixture, isolated_db=isolated_db, keep_db=False)
    metrics = r["metrics"]

    conn = _connect()
    try:
        conn.execute(
            "UPDATE regression_cases SET golden_json=? WHERE case_id=?",
            (json.dumps(metrics, ensure_ascii=False), int(c["case_id"])),
        )
        conn.commit()
        return {"ok": True, "suite": suite_name, "case": case_name, "golden": metrics}
    finally:
        conn.close()


@mcp.tool()
def structai_regression_update_golden_suite(suite_name: str, isolated_db: bool = True) -> Dict[str, Any]:
    conn = _connect()
    try:
        s = conn.execute("SELECT suite_id FROM regression_suites WHERE name=?", (suite_name,)).fetchone()
        if not s:
            raise ValueError("suite not found")
        sid = int(s["suite_id"])
        cases = conn.execute(
            "SELECT case_id, name, fixture_json FROM regression_cases WHERE suite_id=? ORDER BY case_id ASC",
            (sid,),
        ).fetchall()
    finally:
        conn.close()

    updated = []
    for c in cases:
        fixture = json.loads(c["fixture_json"] or "{}")
        r = structai_regression_run_case(fixture_json=fixture, isolated_db=isolated_db, keep_db=False)
        metrics = r["metrics"]
        updated.append({"case": c["name"], "golden": metrics})

        conn = _connect()
        try:
            conn.execute(
                "UPDATE regression_cases SET golden_json=? WHERE case_id=?",
                (json.dumps(metrics, ensure_ascii=False), int(c["case_id"])),
            )
            conn.commit()
        finally:
            conn.close()

    return {"ok": True, "suite": suite_name, "updated": len(updated), "items": updated[:20]}


@mcp.tool()
def structai_regression_run_suite_v2(
    suite_name: str,
    isolated_db: bool = True,
    ratio_tol: float = 1e-3,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        s = conn.execute("SELECT suite_id FROM regression_suites WHERE name=?", (suite_name,)).fetchone()
        if not s:
            raise ValueError("suite not found")
        sid = int(s["suite_id"])
        run_cur = conn.execute(
            "INSERT INTO regression_runs(suite_id, status, report_json) VALUES(?, 'RUNNING', '{}')",
            (sid,),
        )
        run_id = int(run_cur.lastrowid)
        cases = conn.execute(
            "SELECT case_id, name, fixture_json, golden_json FROM regression_cases WHERE suite_id=? ORDER BY case_id ASC",
            (sid,),
        ).fetchall()
        conn.commit()
    finally:
        conn.close()

    case_reports = []
    all_ok = True

    for c in cases:
        case_id = int(c["case_id"])
        name = c["name"]
        fixture = json.loads(c["fixture_json"] or "{}")
        golden = json.loads(c["golden_json"] or "{}")

        try:
            if not golden:
                raise ValueError("golden_json is empty")
            r = structai_regression_run_case(fixture_json=fixture, isolated_db=isolated_db, keep_db=False)
            actual = r["metrics"]
            diff = _compare_metrics(golden, actual, ratio_tol=ratio_tol)

            status = "PASS" if diff["ok"] else "FAIL"
            if status != "PASS":
                all_ok = False

            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO regression_case_results(run_id, case_id, status, diff_json) VALUES(?,?,?,?)",
                    (run_id, case_id, status, json.dumps({"diff": diff, "actual": actual}, ensure_ascii=False)),
                )
                conn.commit()
            finally:
                conn.close()

            case_reports.append({"case": name, "status": status, "diff": diff})

        except Exception as exc:
            all_ok = False
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO regression_case_results(run_id, case_id, status, diff_json) VALUES(?,?,?,?)",
                    (run_id, case_id, "ERROR", json.dumps({"error": str(exc)}, ensure_ascii=False)),
                )
                conn.commit()
            finally:
                conn.close()
            case_reports.append({"case": name, "status": "ERROR", "error": str(exc)})

    status = "PASS" if all_ok else "FAIL"
    conn = _connect()
    try:
        summary = {"suite": suite_name, "status": status, "ratio_tol": ratio_tol, "cases": case_reports}
        conn.execute(
            "UPDATE regression_runs SET status=?, finished_at=datetime('now'), report_json=? WHERE run_id=?",
            (status, json.dumps(summary, ensure_ascii=False), run_id),
        )
        conn.commit()
    finally:
        conn.close()

    return {"ok": True, "run_id": run_id, "status": status, "summary": summary}


@mcp.tool()
def structai_regression_report_generate(
    run_id: int,
    formats: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    formats = formats or ["md"]
    out_dir_path = Path(out_dir).expanduser().resolve() if out_dir else (DB_PATH.parent / "regression_reports")
    out_dir_path.mkdir(parents=True, exist_ok=True)

    conn = _connect()
    try:
        rr = conn.execute(
            """
            SELECT r.run_id, r.status, r.started_at, r.finished_at, r.report_json, s.name as suite_name
            FROM regression_runs r
            JOIN regression_suites s ON s.suite_id = r.suite_id
            WHERE r.run_id=?
            """,
            (int(run_id),),
        ).fetchone()
        if not rr:
            raise ValueError("run not found")

        rows = conn.execute(
            """
            SELECT c.name as case_name, rcr.status, rcr.diff_json
            FROM regression_case_results rcr
            JOIN regression_cases c ON c.case_id = rcr.case_id
            WHERE rcr.run_id=?
            ORDER BY rcr.case_result_id ASC
            """,
            (int(run_id),),
        ).fetchall()
        items = []
        for r in rows:
            items.append({"case": r["case_name"], "status": r["status"], "detail": json.loads(r["diff_json"] or "{}")})

        md = []
        md.append("# Regression Report")
        md.append("")
        md.append(f"- Suite: {rr['suite_name']}")
        md.append(f"- Run ID: {run_id}")
        md.append(f"- Status: {rr['status']}")
        md.append(f"- Started: {rr['started_at']}")
        md.append(f"- Finished: {rr['finished_at']}")
        md.append("")
        md.append("## Cases")
        md.append("")
        md.append("| Case | Status | Notes |")
        md.append("|---|---|---|")
        for it in items:
            note = ""
            if it["status"] == "ERROR":
                note = str(it["detail"].get("error") or "")
            elif it["status"] == "FAIL":
                diff = it["detail"].get("diff") or {}
                note = f"overall_diff={len((diff.get('overall') or {}).keys())}, check_types={len((diff.get('by_check_type') or {}).keys())}"
            md.append(f"| {it['case']} | {it['status']} | {note} |")

        md_text = "\n".join(md)
        outputs = []

        if "md" in formats:
            md_path = out_dir_path / f"regression_run_{run_id}.md"
            md_path.write_text(md_text, encoding="utf-8")
            sha = _sha256_file(md_path)
            uri = md_path.as_uri()
            art_id = _upsert_artifact(conn, uri=uri, kind="regression_report_md", title=f"Regression Report #{run_id}", source_path=str(md_path), sha256=sha)
            conn.execute("INSERT INTO regression_reports(run_id, artifact_id, format) VALUES(?,?,?)", (int(run_id), art_id, "md"))
            outputs.append({"format": "md", "path": str(md_path), "uri": uri, "artifact_id": art_id})

        conn.commit()
        return {"ok": True, "run_id": run_id, "outputs": outputs}
    finally:
        conn.close()


@mcp.tool()
def structai_regression_list_runs(suite_name: str, limit: int = 20) -> Dict[str, Any]:
    conn = _connect()
    try:
        s = conn.execute("SELECT suite_id FROM regression_suites WHERE name=?", (suite_name,)).fetchone()
        if not s:
            raise ValueError("suite not found")
        sid = int(s["suite_id"])
        rows = conn.execute(
            "SELECT run_id, status, started_at, finished_at FROM regression_runs WHERE suite_id=? ORDER BY run_id DESC LIMIT ?",
            (sid, int(limit)),
        ).fetchall()
        return {"ok": True, "suite": suite_name, "items": [dict(r) for r in rows]}
    finally:
        conn.close()


@mcp.tool()
def structai_regression_read_run(run_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        rr = conn.execute(
            "SELECT run_id, suite_id, status, report_json, started_at, finished_at FROM regression_runs WHERE run_id=?",
            (int(run_id),),
        ).fetchone()
        if not rr:
            raise ValueError("run not found")
        cases = conn.execute(
            """
            SELECT c.name as case_name, rcr.status, rcr.diff_json
            FROM regression_case_results rcr
            JOIN regression_cases c ON c.case_id = rcr.case_id
            WHERE rcr.run_id=?
            ORDER BY rcr.case_result_id ASC
            """,
            (int(run_id),),
        ).fetchall()
        reports = conn.execute(
            """
            SELECT rr.regression_report_id, rr.format, a.uri
            FROM regression_reports rr
            JOIN artifacts a ON a.artifact_id = rr.artifact_id
            WHERE rr.run_id=?
            ORDER BY rr.regression_report_id DESC
            """,
            (int(run_id),),
        ).fetchall()
        return {
            "ok": True,
            "run": dict(rr),
            "cases": [{"case": c["case_name"], "status": c["status"], "detail": json.loads(c["diff_json"] or "{}")}
                      for c in cases],
            "reports": [dict(r) for r in reports],
        }
    finally:
        conn.close()

@mcp.tool()
def structai_qa_profile_import(path: str, activate: bool = False) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    name = data.get("name") or p.stem
    ver = data.get("version") or "0.0"

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO qa_profiles(name, version, profile_json, is_active)
            VALUES(?,?,?,0)
            ON CONFLICT(name, version) DO UPDATE SET
              profile_json=excluded.profile_json
            """,
            (name, ver, json.dumps(data, ensure_ascii=False)),
        )
        if activate:
            conn.execute("UPDATE qa_profiles SET is_active=0")
            conn.execute("UPDATE qa_profiles SET is_active=1 WHERE name=? AND version=?", (name, ver))
        conn.commit()
        return {"ok": True, "name": name, "version": ver, "activated": activate}
    finally:
        conn.close()


@mcp.tool()
def structai_qa_profile_set_active(name: str, version: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute("UPDATE qa_profiles SET is_active=0")
        conn.execute("UPDATE qa_profiles SET is_active=1 WHERE name=? AND version=?", (name, version))
        conn.commit()
        return {"ok": True, "active": {"name": name, "version": version}}
    finally:
        conn.close()


@mcp.tool()
def structai_qa_profile_bind_model(model_id: int, name: str, version: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute("SELECT qa_profile_id FROM qa_profiles WHERE name=? AND version=?", (name, version)).fetchone()
        if not row:
            raise ValueError("qa profile not found")
        qa_id = int(row["qa_profile_id"])
        conn.execute(
            "INSERT INTO model_qa_profiles(model_id, qa_profile_id, bound_at) VALUES(?,?, datetime('now'))",
            (int(model_id), qa_id),
        )
        conn.commit()
        return {"ok": True, "model_id": int(model_id), "qa_profile_id": qa_id}
    finally:
        conn.close()


@mcp.tool()
def structai_qa_profile_get_effective(model_id: Optional[int] = None) -> Dict[str, Any]:
    conn = _connect()
    try:
        qa = _get_effective_qa_profile(conn, model_id)
        return {"ok": True, "profile": qa}
    finally:
        conn.close()


@mcp.tool()
def structai_dataset_register(
    type: str,
    name: str,
    version: str,
    artifact_path: Optional[str] = None,
    activate: bool = False,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO dataset_defs(type, name) VALUES(?,?) ON CONFLICT(type, name) DO NOTHING",
            (type, name),
        )
        ds = conn.execute("SELECT dataset_id FROM dataset_defs WHERE type=? AND name=?", (type, name)).fetchone()
        dataset_id = int(ds["dataset_id"])

        artifact_id = None
        sha = None
        if artifact_path:
            p = _resolve_path(artifact_path)
            if p.exists():
                sha = _sha256_file(p)
                uri = p.as_uri()
                artifact_id = _upsert_artifact(conn, uri=uri, kind=f"dataset_{type}", title=name, source_path=str(p), sha256=sha)

        conn.execute(
            """
            INSERT INTO dataset_versions(dataset_id, version, artifact_id, sha256, meta_json, is_active)
            VALUES(?,?,?,?,?,0)
            ON CONFLICT(dataset_id, version) DO UPDATE SET
              artifact_id=excluded.artifact_id,
              sha256=excluded.sha256
            """,
            (dataset_id, version, artifact_id, sha, "{}"),
        )

        if activate:
            conn.execute("UPDATE dataset_versions SET is_active=0 WHERE dataset_id=?", (dataset_id,))
            conn.execute(
                "UPDATE dataset_versions SET is_active=1 WHERE dataset_id=? AND version=?",
                (dataset_id, version),
            )
            conn.execute(
                "INSERT INTO dataset_activation_events(dataset_id, from_version, to_version, actor, reason) VALUES(?,?,?,?,?)",
                (dataset_id, None, version, _actor_from_env(), "activate"),
            )

        conn.commit()
        return {"ok": True, "dataset_id": dataset_id, "version": version, "activated": activate}
    finally:
        conn.close()


@mcp.tool()
def structai_dataset_set_active(type: str, name: str, version: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        ds = conn.execute("SELECT dataset_id FROM dataset_defs WHERE type=? AND name=?", (type, name)).fetchone()
        if not ds:
            raise ValueError("dataset not found")
        dataset_id = int(ds["dataset_id"])

        prev = conn.execute(
            "SELECT version FROM dataset_versions WHERE dataset_id=? AND is_active=1 ORDER BY dataset_version_id DESC LIMIT 1",
            (dataset_id,),
        ).fetchone()
        prev_ver = prev["version"] if prev else None

        conn.execute("UPDATE dataset_versions SET is_active=0 WHERE dataset_id=?", (dataset_id,))
        conn.execute("UPDATE dataset_versions SET is_active=1 WHERE dataset_id=? AND version=?", (dataset_id, version))
        conn.execute(
            "INSERT INTO dataset_activation_events(dataset_id, from_version, to_version, actor, reason) VALUES(?,?,?,?,?)",
            (dataset_id, prev_ver, version, _actor_from_env(), "set_active"),
        )
        conn.commit()
        return {"ok": True, "dataset_id": dataset_id, "from": prev_ver, "to": version}
    finally:
        conn.close()


@mcp.tool()
def structai_dataset_get_active_all() -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT dd.type, dd.name, dv.version, dv.artifact_id
            FROM dataset_defs dd
            JOIN dataset_versions dv ON dv.dataset_id = dd.dataset_id
            WHERE dv.is_active=1
            ORDER BY dd.type, dd.name
            """,
        ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
    finally:
        conn.close()


@mcp.tool()
def structai_report_sign(
    artifact_id: int,
    signer: str,
    method: str = "sha256",
    note: str = "",
    key_id: Optional[str] = None,
    secret: Optional[str] = None,
) -> Dict[str, Any]:
    conn = _connect()
    try:
        art = conn.execute("SELECT source_path FROM artifacts WHERE artifact_id=?", (int(artifact_id),)).fetchone()
        if not art:
            raise ValueError("artifact not found")
        p = Path(art["source_path"])
        if not p.exists():
            raise FileNotFoundError(str(p))

        digest = _sha256_file(p)
        signature = None
        if method == "hmac-sha256" and secret:
            import hmac
            signature = hmac.new(secret.encode("utf-8"), digest.encode("utf-8"), hashlib.sha256).hexdigest()

        conn.execute(
            """
            INSERT INTO report_signatures(artifact_id, method, digest_sha256, signature_b64, key_id, signer, note)
            VALUES(?,?,?,?,?,?,?)
            """,
            (int(artifact_id), method, digest, signature, key_id, signer, note),
        )
        conn.commit()
        return {"ok": True, "artifact_id": int(artifact_id), "digest": digest}
    finally:
        conn.close()


@mcp.tool()
def structai_report_verify(signature_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT rs.signature_id, rs.artifact_id, rs.method, rs.digest_sha256, rs.signature_b64, a.source_path
            FROM report_signatures rs
            JOIN artifacts a ON a.artifact_id = rs.artifact_id
            WHERE rs.signature_id=?
            """,
            (int(signature_id),),
        ).fetchone()
        if not row:
            raise ValueError("signature not found")
        p = Path(row["source_path"])
        if not p.exists():
            raise FileNotFoundError(str(p))
        digest = _sha256_file(p)
        ok = digest == row["digest_sha256"]
        return {"ok": True, "signature_id": int(signature_id), "valid": ok}
    finally:
        conn.close()


@mcp.tool()
def structai_approval_request(entity_type: str, entity_id: int, actor: str, comment: str = "") -> Dict[str, Any]:
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO approvals(entity_type, entity_id, status, actor, comment, meta_json, created_at, updated_at)
            VALUES(?,?,?,?,?, '{}', datetime('now'), datetime('now'))
            """,
            (entity_type, int(entity_id), "requested", actor, comment),
        )
        conn.commit()
        return {"ok": True, "approval_id": int(cur.lastrowid)}
    finally:
        conn.close()


@mcp.tool()
def structai_approval_set_status(approval_id: int, status: str, actor: str, comment: str = "") -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE approvals SET status=?, actor=?, comment=?, updated_at=datetime('now') WHERE approval_id=?",
            (status, actor, comment, int(approval_id)),
        )
        conn.commit()
        return {"ok": True, "approval_id": int(approval_id), "status": status}
    finally:
        conn.close()


@mcp.tool()
def structai_approval_list(entity_type: str, entity_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT approval_id, status, actor, comment, created_at, updated_at FROM approvals WHERE entity_type=? AND entity_id=? ORDER BY approval_id DESC",
            (entity_type, int(entity_id)),
        ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
    finally:
        conn.close()

@mcp.tool()
def structai_actor_whoami() -> Dict[str, Any]:
    return {"ok": True, "actor": _actor_from_env()}


@mcp.tool()
def structai_role_upsert(name: str, permissions: List[str], description: str = "") -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO roles(name, description, permissions_json)
            VALUES(?,?,?)
            ON CONFLICT(name) DO UPDATE SET
              description=excluded.description,
              permissions_json=excluded.permissions_json
            """,
            (name, description, json.dumps(permissions, ensure_ascii=False)),
        )
        conn.commit()
        return {"ok": True, "name": name}
    finally:
        conn.close()


@mcp.tool()
def structai_user_upsert(actor: str, display_name: str = "", email: str = "", is_active: bool = True) -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO users(actor, display_name, email, is_active)
            VALUES(?,?,?,?)
            ON CONFLICT(actor) DO UPDATE SET
              display_name=excluded.display_name,
              email=excluded.email,
              is_active=excluded.is_active
            """,
            (actor, display_name, email, 1 if is_active else 0),
        )
        conn.commit()
        return {"ok": True, "actor": actor}
    finally:
        conn.close()


@mcp.tool()
def structai_project_create(name: str, description: str = "") -> Dict[str, Any]:
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO projects(name, description) VALUES(?,?)",
            (name, description),
        )
        conn.commit()
        return {"ok": True, "project_id": int(cur.lastrowid)}
    finally:
        conn.close()


@mcp.tool()
def structai_project_list(limit: int = 50) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT project_id, name, description, created_at FROM projects ORDER BY project_id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
    finally:
        conn.close()


@mcp.tool()
def structai_project_bind_model(project_id: int, model_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO project_models(project_id, model_id, bound_at) VALUES(?,?, datetime('now'))",
            (int(project_id), int(model_id)),
        )
        conn.commit()
        return {"ok": True, "project_id": int(project_id), "model_id": int(model_id)}
    finally:
        conn.close()


@mcp.tool()
def structai_project_add_member(project_id: int, actor: str, role_name: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        user = conn.execute("SELECT user_id FROM users WHERE actor=?", (actor,)).fetchone()
        if not user:
            raise ValueError("user not found; run structai_user_upsert first")
        role = conn.execute("SELECT role_id FROM roles WHERE name=?", (role_name,)).fetchone()
        if not role:
            raise ValueError("role not found; run structai_role_upsert first")

        conn.execute(
            "INSERT OR REPLACE INTO project_memberships(project_id, user_id, role_id, status, joined_at) VALUES(?,?,?,?, datetime('now'))",
            (int(project_id), int(user["user_id"]), int(role["role_id"]), "active"),
        )
        conn.commit()
        return {"ok": True, "project_id": int(project_id), "actor": actor, "role": role_name}
    finally:
        conn.close()


@mcp.tool()
def structai_workflow_import(path: str, activate: bool = False) -> Dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    name = data.get("name") or p.stem
    ver = data.get("version") or "0.0"
    steps = data.get("steps") or []
    entity_types = data.get("entity_types") or []

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO approval_workflow_defs(name, version, entity_types_json, steps_json, is_active)
            VALUES(?,?,?,?,0)
            ON CONFLICT(name, version) DO UPDATE SET
              entity_types_json=excluded.entity_types_json,
              steps_json=excluded.steps_json
            """,
            (name, ver, json.dumps(entity_types, ensure_ascii=False), json.dumps(steps, ensure_ascii=False)),
        )
        if activate:
            conn.execute("UPDATE approval_workflow_defs SET is_active=0")
            conn.execute("UPDATE approval_workflow_defs SET is_active=1 WHERE name=? AND version=?", (name, ver))
        conn.commit()
        return {"ok": True, "name": name, "version": ver, "activated": activate}
    finally:
        conn.close()


@mcp.tool()
def structai_workflow_bind_project(project_id: int, name: str, version: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT workflow_def_id FROM approval_workflow_defs WHERE name=? AND version=?",
            (name, version),
        ).fetchone()
        if not row:
            raise ValueError("workflow not found")
        wf_id = int(row["workflow_def_id"])
        conn.execute(
            "INSERT OR REPLACE INTO project_workflows(project_id, workflow_def_id, bound_at) VALUES(?,?, datetime('now'))",
            (int(project_id), wf_id),
        )
        conn.commit()
        return {"ok": True, "project_id": int(project_id), "workflow_def_id": wf_id}
    finally:
        conn.close()


@mcp.tool()
def structai_approval_request_v2(project_id: int, entity_type: str, entity_id: int, comment: str = "") -> Dict[str, Any]:
    conn = _connect()
    try:
        wf = conn.execute(
            "SELECT workflow_def_id FROM project_workflows WHERE project_id=?",
            (int(project_id),),
        ).fetchone()
        if not wf:
            raise ValueError("project workflow not bound")
        wf_id = int(wf["workflow_def_id"])

        cur = conn.execute(
            """
            INSERT INTO approval_instances(workflow_def_id, entity_type, entity_id, project_id, status, current_step_idx, requested_by)
            VALUES(?,?,?,?, 'in_progress', 0, ?)
            ON CONFLICT(workflow_def_id, entity_type, entity_id) DO UPDATE SET
              status='in_progress', current_step_idx=0, requested_by=excluded.requested_by, updated_at=datetime('now')
            """,
            (wf_id, entity_type, int(entity_id), int(project_id), _actor_from_env()),
        )
        instance_id = int(cur.lastrowid or conn.execute(
            "SELECT instance_id FROM approval_instances WHERE workflow_def_id=? AND entity_type=? AND entity_id=?",
            (wf_id, entity_type, int(entity_id)),
        ).fetchone()["instance_id"])

        conn.execute(
            "INSERT INTO project_events(project_id, event_type, message, payload_json, actor) VALUES(?,?,?,?,?)",
            (int(project_id), "approval_request", comment, json.dumps({"entity_type": entity_type, "entity_id": entity_id}, ensure_ascii=False), _actor_from_env()),
        )
        conn.commit()
        return {"ok": True, "instance_id": instance_id}
    finally:
        conn.close()


@mcp.tool()
def structai_approval_vote(instance_id: int, decision: str, comment: str = "") -> Dict[str, Any]:
    conn = _connect()
    try:
        inst = conn.execute(
            "SELECT instance_id, workflow_def_id, current_step_idx, status, project_id FROM approval_instances WHERE instance_id=?",
            (int(instance_id),),
        ).fetchone()
        if not inst:
            raise ValueError("approval instance not found")
        if inst["status"] not in ("in_progress",):
            return {"ok": True, "instance_id": int(instance_id), "status": inst["status"]}

        conn.execute(
            "INSERT INTO approval_votes(instance_id, step_idx, actor, decision, comment) VALUES(?,?,?,?,?)",
            (int(instance_id), int(inst["current_step_idx"]), _actor_from_env(), decision, comment),
        )

        # advance or close
        if decision == "reject":
            conn.execute(
                "UPDATE approval_instances SET status='rejected', updated_at=datetime('now') WHERE instance_id=?",
                (int(instance_id),),
            )
        else:
            wf = conn.execute(
                "SELECT steps_json FROM approval_workflow_defs WHERE workflow_def_id=?",
                (int(inst["workflow_def_id"]),),
            ).fetchone()
            steps = json.loads(wf["steps_json"] or "[]") if wf else []
            next_idx = int(inst["current_step_idx"]) + 1
            if next_idx >= len(steps):
                conn.execute(
                    "UPDATE approval_instances SET status='approved', current_step_idx=?, updated_at=datetime('now') WHERE instance_id=?",
                    (next_idx, int(instance_id)),
                )
            else:
                conn.execute(
                    "UPDATE approval_instances SET current_step_idx=?, updated_at=datetime('now') WHERE instance_id=?",
                    (next_idx, int(instance_id)),
                )

        conn.commit()
        return {"ok": True, "instance_id": int(instance_id)}
    finally:
        conn.close()


@mcp.tool()
def structai_approval_read(entity_type: str, entity_id: int, project_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        inst = conn.execute(
            """
            SELECT instance_id, status, current_step_idx, workflow_def_id
            FROM approval_instances
            WHERE entity_type=? AND entity_id=? AND project_id=?
            """,
            (entity_type, int(entity_id), int(project_id)),
        ).fetchone()
        if not inst:
            return {"ok": True, "instance": None}
        votes = conn.execute(
            "SELECT step_idx, actor, decision, comment, created_at FROM approval_votes WHERE instance_id=? ORDER BY vote_id ASC",
            (int(inst["instance_id"]),),
        ).fetchall()
        return {"ok": True, "instance": dict(inst), "votes": [dict(v) for v in votes]}
    finally:
        conn.close()


@mcp.tool()
def structai_context_capture(entity_type: str, entity_id: int, model_id: Optional[int] = None, project_id: Optional[int] = None) -> Dict[str, Any]:
    conn = _connect()
    try:
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "actor": _actor_from_env(),
            "model_id": model_id,
            "project_id": project_id,
            "datasets": structai_dataset_get_active_all().get("items"),
            "qa_profile": structai_qa_profile_get_effective(model_id).get("profile") if model_id else None,
        }
        conn.execute(
            """
            INSERT INTO entity_contexts(entity_type, entity_id, context_json, created_at, updated_at)
            VALUES(?,?,?,?, datetime('now'))
            ON CONFLICT(entity_type, entity_id) DO UPDATE SET
              context_json=excluded.context_json,
              updated_at=datetime('now')
            """,
            (entity_type, int(entity_id), json.dumps(context, ensure_ascii=False), datetime.utcnow().isoformat()),
        )
        conn.commit()
        return {"ok": True, "entity_type": entity_type, "entity_id": int(entity_id)}
    finally:
        conn.close()


@mcp.tool()
def structai_context_get(entity_type: str, entity_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT context_json FROM entity_contexts WHERE entity_type=? AND entity_id=?",
            (entity_type, int(entity_id)),
        ).fetchone()
        if not row:
            return {"ok": True, "context": None}
        return {"ok": True, "context": json.loads(row["context_json"] or "{}")}
    finally:
        conn.close()


@mcp.tool()
def structai_dataset_set_active_notify(
    dataset_type: str,
    dataset_name: str,
    from_version: Optional[str],
    to_version: str,
    project_id: Optional[int] = None,
    reason: str = "",
) -> Dict[str, Any]:
    conn = _connect()
    try:
        ds = conn.execute("SELECT dataset_id FROM dataset_defs WHERE type=? AND name=?", (dataset_type, dataset_name)).fetchone()
        if not ds:
            raise ValueError("dataset not found")
        dataset_id = int(ds["dataset_id"])
        conn.execute(
            "INSERT INTO dataset_activation_events(dataset_id, from_version, to_version, actor, reason) VALUES(?,?,?,?,?)",
            (dataset_id, from_version, to_version, _actor_from_env(), reason),
        )
        if project_id is not None:
            conn.execute(
                "INSERT INTO project_events(project_id, event_type, message, payload_json, actor) VALUES(?,?,?,?,?)",
                (int(project_id), "dataset_change", reason, json.dumps({"dataset_id": dataset_id, "from": from_version, "to": to_version}, ensure_ascii=False), _actor_from_env()),
            )
        conn.commit()
        return {"ok": True, "dataset_id": dataset_id}
    finally:
        conn.close()


@mcp.tool()
def structai_project_dashboard(project_id: int, limit_events: int = 30) -> Dict[str, Any]:
    conn = _connect()
    try:
        events = conn.execute(
            "SELECT event_type, message, payload_json, actor, created_at FROM project_events WHERE project_id=? ORDER BY project_event_id DESC LIMIT ?",
            (int(project_id), int(limit_events)),
        ).fetchall()
        approvals = conn.execute(
            "SELECT instance_id, status, entity_type, entity_id, updated_at FROM approval_instances WHERE project_id=? ORDER BY updated_at DESC LIMIT 20",
            (int(project_id),),
        ).fetchall()
        models = conn.execute(
            "SELECT model_id FROM project_models WHERE project_id=?",
            (int(project_id),),
        ).fetchall()
        return {
            "ok": True,
            "project_id": int(project_id),
            "models": [dict(m) for m in models],
            "events": [dict(e) for e in events],
            "approvals": [dict(a) for a in approvals],
        }
    finally:
        conn.close()

@mcp.tool()
def structai_reset_all() -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute("DELETE FROM cad_entities")
        conn.execute("DELETE FROM doc_chunks")
        conn.execute("DELETE FROM artifacts")
        conn.execute("DELETE FROM member_spec_links")
        conn.execute("DELETE FROM cad_specs")
        conn.execute("DELETE FROM member_mappings")
        conn.execute("DELETE FROM model_members")
        conn.execute("DELETE FROM models")
        conn.execute("DELETE FROM analysis_runs")
        conn.execute("DELETE FROM member_results")
        conn.execute("DELETE FROM member_design_inputs")
        conn.execute("DELETE FROM rulepacks")
        conn.execute("DELETE FROM check_runs")
        conn.execute("DELETE FROM check_results")
        conn.execute("DELETE FROM reports")
        conn.execute("DELETE FROM cad_tables")
        conn.execute("DELETE FROM cad_table_cells")
        conn.execute("DELETE FROM cad_story_tags")
        conn.execute("DELETE FROM cad_table_schemas")
        conn.execute("DELETE FROM cad_table_row_parses")
        conn.execute("DELETE FROM token_story_maps")
        conn.execute("DELETE FROM section_catalog")
        conn.execute("DELETE FROM section_aliases")
        conn.execute("DELETE FROM member_section_resolutions")
        conn.execute("DELETE FROM regression_suites")
        conn.execute("DELETE FROM regression_cases")
        conn.execute("DELETE FROM regression_runs")
        conn.execute("DELETE FROM regression_case_results")
        conn.execute("DELETE FROM regression_reports")
        conn.execute("DELETE FROM benchmarks")
        conn.execute("DELETE FROM benchmark_results")
        conn.execute("DELETE FROM compare_runs")
        conn.execute("DELETE FROM compare_items")
        conn.execute("DELETE FROM compare_reports")
        conn.execute("DELETE FROM check_template_sets")
        conn.execute("DELETE FROM design_patch_runs")
        conn.execute("DELETE FROM design_patch_items")
        conn.execute("DELETE FROM decision_logs")
        conn.execute("DELETE FROM qa_profiles")
        conn.execute("DELETE FROM model_qa_profiles")
        conn.execute("DELETE FROM dataset_defs")
        conn.execute("DELETE FROM dataset_versions")
        conn.execute("DELETE FROM dataset_changelogs")
        conn.execute("DELETE FROM report_signatures")
        conn.execute("DELETE FROM approvals")
        conn.execute("DELETE FROM users")
        conn.execute("DELETE FROM roles")
        conn.execute("DELETE FROM projects")
        conn.execute("DELETE FROM project_models")
        conn.execute("DELETE FROM project_memberships")
        conn.execute("DELETE FROM approval_workflow_defs")
        conn.execute("DELETE FROM project_workflows")
        conn.execute("DELETE FROM approval_instances")
        conn.execute("DELETE FROM approval_votes")
        conn.execute("DELETE FROM entity_contexts")
        conn.execute("DELETE FROM dataset_activation_events")
        conn.execute("DELETE FROM project_events")
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()

if __name__ == "__main__":
    mcp.run()
