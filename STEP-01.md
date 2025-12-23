좋아. **1단계(문서 인덱싱 강화: PDF 페이지 단위 인용 + FTS5 검색)**부터 실제로 “바로 붙여 넣어서 동작”하는 수준으로 구체화해서 진행할게.
(이 단계가 끝나면, 너가 말한 “대화 중에 KDS/KBC/도면/PDF 근거를 페이지까지 찍어서 답변”의 기반이 생긴다.)

1단계 목표

PDF를 페이지 단위로 텍스트 추출 → chunking → DB 저장

검색은 SQLite FTS5로 빠르게 + 스니펫/랭킹 제공

검색 결과에 **“file URI + page 정보(#page=…)”**를 같이 반환해서
→ AI 대화창에서 답할 때 근거 페이지를 바로 붙일 수 있게 한다.

참고: VS Code는 MCP 도구의 readOnlyHint가 true이면 “승인 팝업 없이” 실행되도록 힌트를 줄 수 있어. 
Visual Studio Code

또한 VS Code는 충돌 방지를 위해 “fully qualified tool name” 개념을 지원한다는 점도 염두에 두면 좋아. 
Visual Studio Code

1-1) DB 스키마 업데이트 (schema.sql)

중요 포인트

스키마 파일에는 FTS5 생성문을 넣지 않는 걸 추천해. (환경에 따라 FTS5가 없는 SQLite 빌드가 있을 수 있어서, schema.sql 실행 자체가 실패할 수 있음)

대신 서버 코드에서 try/except로 FTS5를 “가능하면 켜고, 아니면 LIKE로 fallback” 하게 한다.

아래처럼 schema.sql을 베이스 테이블만 포함하도록 업데이트해줘:

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '0.0.2');

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  uri TEXT NOT NULL UNIQUE,            -- file:///... 형태 권장
  kind TEXT NOT NULL,                  -- pdf|md|txt|dxf|...
  sha256 TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  meta_json TEXT                       -- {"page_count":..., ...}
);

CREATE TABLE IF NOT EXISTS doc_chunks (
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  page_start INTEGER,                  -- PDF는 1-based 페이지, 문서형은 NULL 가능
  page_end INTEGER,
  chunk_index INTEGER NOT NULL DEFAULT 0,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fe_models (
  model_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  source_uri TEXT,
  meta_json TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS mappings (
  mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
  from_ref TEXT NOT NULL,              -- 예: "dxf://...#entity=..."
  to_ref TEXT NOT NULL,                -- 예: "fem://model/beam/12"
  confidence REAL NOT NULL DEFAULT 0.5,
  rationale TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS checks (
  check_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER,
  member_ref TEXT NOT NULL,
  check_type TEXT NOT NULL,            -- e.g. "beam-flexure"
  result_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

1-2) MCP 서버 업데이트 (PDF 추출 + FTS5 + 페이지 인용)

아래는 server.py를 통째로 교체해도 되는 수준의 구현 예시야.
(기존에 FastMCP로 만들었던 structai_* 도구들과 호환되도록 구성했어.)

mcp_server/server.py
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

ROOT = Path(os.environ.get("STRUCTAI_ROOT", str(Path.home() / ".structai")))
DB_PATH = ROOT / "structai.sqlite"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

mcp = FastMCP("StructAI Local MCP", version="0.0.2")


# -----------------------------
# DB / schema / FTS helpers
# -----------------------------
def _connect() -> sqlite3.Connection:
    ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _ensure_base_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def _fts_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1 FROM doc_chunks_fts LIMIT 1;")
        return True
    except sqlite3.OperationalError:
        return False


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    """
    Create FTS5 index & triggers if possible.
    Return True if FTS enabled, False otherwise.
    """
    if _fts_available(conn):
        return True

    try:
        # External content table pattern:
        # - rowid maps to doc_chunks.chunk_id
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5(
              content,
              content='doc_chunks',
              content_rowid='chunk_id',
              tokenize='unicode61'
            );
            """
        )

        conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS doc_chunks_ai AFTER INSERT ON doc_chunks BEGIN
              INSERT INTO doc_chunks_fts(rowid, content) VALUES (new.chunk_id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS doc_chunks_ad AFTER DELETE ON doc_chunks BEGIN
              INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content) VALUES('delete', old.chunk_id, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS doc_chunks_au AFTER UPDATE OF content ON doc_chunks BEGIN
              INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, content) VALUES('delete', old.chunk_id, old.content);
              INSERT INTO doc_chunks_fts(rowid, content) VALUES (new.chunk_id, new.content);
            END;
            """
        )

        # Build index from existing doc_chunks (if any)
        conn.execute("INSERT INTO doc_chunks_fts(doc_chunks_fts) VALUES('rebuild');")
        return True
    except sqlite3.OperationalError:
        # SQLite build without FTS5
        return False


def _ensure_db() -> Tuple[sqlite3.Connection, bool]:
    conn = _connect()
    with conn:
        _ensure_base_schema(conn)
        fts = _ensure_fts(conn)
    return conn, fts


# -----------------------------
# File helpers
# -----------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _to_path_and_uri(p: str) -> Tuple[Path, str]:
    """
    Accepts a local path or file:// URI.
    Returns (Path, uri).
    """
    if p.startswith("file:"):
        uri = p
        pr = urlparse(p)
        # urlparse().path is url-encoded and may start with /C:/ on Windows
        raw = unquote(pr.path)
        if os.name == "nt" and raw.startswith("/") and len(raw) >= 3 and raw[2] == ":":
            raw = raw[1:]
        path = Path(raw)
        return path, uri

    path = Path(p).expanduser().resolve()
    return path, path.as_uri()


def _infer_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in (".md", ".markdown"):
        return "md"
    if ext in (".txt", ".text"):
        return "txt"
    if ext == ".dxf":
        return "dxf"
    return "bin"


def _chunk_text(text: str, max_chars: int = 1400, overlap: int = 200) -> List[str]:
    """
    Simple chunker (character based).
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap)
    return chunks


def _extract_pdf_pages(path: Path) -> Tuple[int, List[Tuple[int, str]], int]:
    """
    Returns (page_count, [(page_no, text)], empty_page_count)
    """
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("pypdf가 설치되어 있지 않습니다. requirements에 pypdf를 추가하세요.") from e

    reader = PdfReader(str(path))
    page_count = len(reader.pages)

    pages: List[Tuple[int, str]] = []
    empty_pages = 0
    for idx, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if not txt.strip():
            empty_pages += 1
        pages.append((idx, txt))

    return page_count, pages, empty_pages


def _upsert_artifact(
    conn: sqlite3.Connection,
    uri: str,
    kind: str,
    sha256: Optional[str],
    meta: Optional[Dict[str, Any]],
) -> int:
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    row = conn.execute("SELECT artifact_id FROM artifacts WHERE uri=?", (uri,)).fetchone()
    if row:
        artifact_id = int(row["artifact_id"])
        conn.execute(
            "UPDATE artifacts SET kind=?, sha256=?, meta_json=? WHERE artifact_id=?",
            (kind, sha256, meta_json, artifact_id),
        )
        return artifact_id

    cur = conn.execute(
        "INSERT INTO artifacts(uri, kind, sha256, meta_json) VALUES(?,?,?,?)",
        (uri, kind, sha256, meta_json),
    )
    return int(cur.lastrowid)


def _cite_uri(uri: str, page_start: Optional[int]) -> str:
    """
    PDF 페이지 링크는 일반적으로 #page=12 형태가 호환이 좋음.
    """
    if page_start:
        return f"{uri}#page={page_start}"
    return uri


# -----------------------------
# Tools
# -----------------------------
@mcp.tool(
    description="프로젝트(로컬 DB)에 저장된 아티팩트/청크/모델/매핑/체크 개수를 요약합니다.",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
)
def structai_get_project_summary() -> Dict[str, Any]:
    conn, fts = _ensure_db()
    with conn:
        a = conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"]
        c = conn.execute("SELECT COUNT(*) AS n FROM doc_chunks").fetchone()["n"]
        m = conn.execute("SELECT COUNT(*) AS n FROM fe_models").fetchone()["n"]
        mp = conn.execute("SELECT COUNT(*) AS n FROM mappings").fetchone()["n"]
        ck = conn.execute("SELECT COUNT(*) AS n FROM checks").fetchone()["n"]
    conn.close()
    return {
        "schema_version": "0.0.2",
        "fts_enabled": bool(fts),
        "counts": {
            "artifacts": int(a),
            "doc_chunks": int(c),
            "fe_models": int(m),
            "mappings": int(mp),
            "checks": int(ck),
        },
    }


@mcp.tool(
    description="현재 프로젝트에 인덱싱/등록된 문서(아티팩트) 목록을 반환합니다.",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
)
def structai_list_artifacts(limit: int = 200) -> Dict[str, Any]:
    conn, fts = _ensure_db()
    rows = conn.execute(
        "SELECT artifact_id, uri, kind, sha256, created_at, meta_json FROM artifacts ORDER BY artifact_id DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    conn.close()
    artifacts = []
    for r in rows:
        meta = {}
        try:
            meta = json.loads(r["meta_json"] or "{}")
        except Exception:
            meta = {}
        artifacts.append(
            {
                "artifact_id": int(r["artifact_id"]),
                "uri": r["uri"],
                "kind": r["kind"],
                "sha256": r["sha256"],
                "created_at": r["created_at"],
                "meta": meta,
            }
        )
    return {"fts_enabled": bool(fts), "artifacts": artifacts}


@mcp.tool(
    description="파일들을 프로젝트 DB에 인덱싱(텍스트/MD/PDF)합니다. PDF는 페이지 단위로 저장하고 page_start/page_end를 유지합니다.",
    annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False, destructiveHint=False),
)
def structai_import_files(paths: List[str]) -> Dict[str, Any]:
    conn, fts = _ensure_db()

    imported: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for p in paths:
        try:
            path, uri = _to_path_and_uri(p)
            if not path.exists():
                raise FileNotFoundError(str(path))

            kind = _infer_kind(path)
            sha = _sha256_file(path) if kind != "bin" else None

            if kind == "pdf":
                page_count, pages, empty_pages = _extract_pdf_pages(path)
                meta = {
                    "path": str(path),
                    "page_count": page_count,
                    "empty_pages": empty_pages,
                    "extractor": "pypdf",
                }

                with conn:
                    artifact_id = _upsert_artifact(conn, uri, kind, sha, meta)
                    # Re-import: clear old chunks
                    conn.execute("DELETE FROM doc_chunks WHERE artifact_id=?", (artifact_id,))

                    chunk_index = 0
                    inserted = 0
                    for page_no, page_text in pages:
                        for chunk in _chunk_text(page_text):
                            conn.execute(
                                """
                                INSERT INTO doc_chunks(artifact_id, page_start, page_end, chunk_index, content)
                                VALUES(?,?,?,?,?)
                                """,
                                (artifact_id, page_no, page_no, chunk_index, chunk),
                            )
                            chunk_index += 1
                            inserted += 1

                imported.append(
                    {
                        "uri": uri,
                        "kind": kind,
                        "artifact_id": artifact_id,
                        "chunks_inserted": inserted,
                        "meta": meta,
                    }
                )
                continue

            # md/txt 등 텍스트
            if kind in ("md", "txt"):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = path.read_text(errors="ignore")

                meta = {"path": str(path), "extractor": "read_text"}

                with conn:
                    artifact_id = _upsert_artifact(conn, uri, kind, sha, meta)
                    conn.execute("DELETE FROM doc_chunks WHERE artifact_id=?", (artifact_id,))

                    chunk_index = 0
                    inserted = 0
                    for chunk in _chunk_text(text):
                        conn.execute(
                            """
                            INSERT INTO doc_chunks(artifact_id, page_start, page_end, chunk_index, content)
                            VALUES(?,?,?,?,?)
                            """,
                            (artifact_id, None, None, chunk_index, chunk),
                        )
                        chunk_index += 1
                        inserted += 1

                imported.append(
                    {
                        "uri": uri,
                        "kind": kind,
                        "artifact_id": artifact_id,
                        "chunks_inserted": inserted,
                        "meta": meta,
                    }
                )
                continue

            # dxf/bin은 2단계에서 본격 처리. 여기선 등록만.
            meta = {"path": str(path)}
            with conn:
                artifact_id = _upsert_artifact(conn, uri, kind, sha, meta)
            imported.append({"uri": uri, "kind": kind, "artifact_id": artifact_id, "chunks_inserted": 0, "meta": meta})

        except Exception as e:
            errors.append({"input": p, "error": str(e)})

    conn.close()
    return {"fts_enabled": bool(fts), "imported": imported, "errors": errors}


@mcp.tool(
    description="인덱싱된 문서에서 쿼리로 검색합니다. 가능한 경우 FTS5를 사용하고, 아니면 LIKE로 대체합니다. 결과는 페이지/URI 인용 정보를 포함합니다.",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
)
def structai_search_knowledge(query: str, limit: int = 8) -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"fts_enabled": False, "query": query, "hits": []}

    conn, fts = _ensure_db()
    hits: List[Dict[str, Any]] = []

    if fts:
        rows = conn.execute(
            """
            SELECT
              c.chunk_id,
              a.uri,
              a.kind,
              c.page_start,
              c.page_end,
              snippet(doc_chunks_fts, 0, '[', ']', '…', 12) AS snippet,
              bm25(doc_chunks_fts) AS rank
            FROM doc_chunks_fts
            JOIN doc_chunks c ON c.chunk_id = doc_chunks_fts.rowid
            JOIN artifacts a ON a.artifact_id = c.artifact_id
            WHERE doc_chunks_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?;
            """,
            (query, int(limit)),
        ).fetchall()

        for r in rows:
            page_start = r["page_start"]
            cite = _cite_uri(r["uri"], int(page_start) if page_start is not None else None)
            hits.append(
                {
                    "chunk_id": int(r["chunk_id"]),
                    "artifact_uri": r["uri"],
                    "kind": r["kind"],
                    "page_start": int(page_start) if page_start is not None else None,
                    "page_end": int(r["page_end"]) if r["page_end"] is not None else None,
                    "snippet": r["snippet"],
                    "rank": float(r["rank"]),
                    "cite_uri": cite,
                }
            )

        conn.close()
        return {"fts_enabled": True, "query": query, "hits": hits}

    # FTS5 fallback (LIKE)
    like = f"%{query}%"
    rows = conn.execute(
        """
        SELECT
          c.chunk_id, a.uri, a.kind, c.page_start, c.page_end, substr(c.content, 1, 400) AS snippet
        FROM doc_chunks c
        JOIN artifacts a ON a.artifact_id = c.artifact_id
        WHERE c.content LIKE ?
        ORDER BY c.chunk_id DESC
        LIMIT ?;
        """,
        (like, int(limit)),
    ).fetchall()

    for r in rows:
        page_start = r["page_start"]
        cite = _cite_uri(r["uri"], int(page_start) if page_start is not None else None)
        hits.append(
            {
                "chunk_id": int(r["chunk_id"]),
                "artifact_uri": r["uri"],
                "kind": r["kind"],
                "page_start": int(page_start) if page_start is not None else None,
                "page_end": int(r["page_end"]) if r["page_end"] is not None else None,
                "snippet": r["snippet"],
                "rank": None,
                "cite_uri": cite,
            }
        )

    conn.close()
    return {"fts_enabled": False, "query": query, "hits": hits}


@mcp.tool(
    description="검색 결과의 chunk_id로 원문 청크 텍스트(및 페이지 인용 링크)를 가져옵니다.",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
)
def structai_get_chunk(chunk_id: int) -> Dict[str, Any]:
    conn, fts = _ensure_db()
    r = conn.execute(
        """
        SELECT c.chunk_id, c.content, c.page_start, c.page_end, a.uri, a.kind
        FROM doc_chunks c
        JOIN artifacts a ON a.artifact_id = c.artifact_id
        WHERE c.chunk_id = ?;
        """,
        (int(chunk_id),),
    ).fetchone()
    conn.close()

    if not r:
        return {"found": False, "chunk_id": int(chunk_id)}

    page_start = r["page_start"]
    cite = _cite_uri(r["uri"], int(page_start) if page_start is not None else None)

    return {
        "found": True,
        "chunk_id": int(r["chunk_id"]),
        "artifact_uri": r["uri"],
        "kind": r["kind"],
        "page_start": int(page_start) if page_start is not None else None,
        "page_end": int(r["page_end"]) if r["page_end"] is not None else None,
        "cite_uri": cite,
        "content": r["content"],
    }


@mcp.tool(
    description="프로젝트 DB를 초기화합니다(모든 artifacts/chunks/mappings/checks 삭제). 주의해서 사용하세요.",
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False),
)
def structai_reset_project() -> Dict[str, Any]:
    conn, fts = _ensure_db()
    with conn:
        conn.execute("DELETE FROM checks;")
        conn.execute("DELETE FROM mappings;")
        conn.execute("DELETE FROM fe_models;")
        conn.execute("DELETE FROM doc_chunks;")
        conn.execute("DELETE FROM artifacts;")
        # FTS table exists? then rebuild to empty
        try:
            conn.execute("INSERT INTO doc_chunks_fts(doc_chunks_fts) VALUES('rebuild');")
        except Exception:
            pass
    conn.close()
    return {"ok": True, "fts_enabled": bool(fts)}

1-3) requirements.txt 업데이트

mcp_server/requirements.txt에 PDF 파서 추가:

mcp>=1.0.0
pypdf>=5.0.0


(추후 OCR을 붙일 때는 pytesseract/easyocr 같은 걸 고려할 수 있는데, 1단계에서는 “텍스트 내장 PDF”만 제대로 잡아도 효과가 커.)

1-4) VS Code 확장 코드 수정: MCP 도구 필터 버그 해결

네가 지금 확장 코드에서 도구 이름을 startsWith('structai_')로 필터링하면,
VS Code가 fully qualified name(예: github/github-mcp-server/list_issues 같은 형태)로 이름을 갖는 경우 
Visual Studio Code

너의 MCP 도구도 비슷하게 앞에 prefix가 붙어 startsWith가 실패할 가능성이 높아.

그래서 필터를 이렇게 바꿔줘:

src/extension.ts (핵심만)
const tools = vscode.lm.tools.filter(t => {
  const name = (t as any).name as string | undefined;
  if (!name) return false;

  // fully qualified name에서도 잡히도록
  return name.includes('structai_') || name.includes('/structai_');
});


그리고 “근거 인용을 하라”는 지시도 prompt에 같이 넣어주는 걸 추천:

prompt: `
너는 건축구조(구조설계/구조해석) 도우미다.
추측하지 말고, 프로젝트에 인덱싱된 문서(PDF/MD/TXT)를 structai_search_knowledge로 먼저 찾아라.
답변에 문서 근거가 있으면 반드시 (문서명 p.xx) 형태로 페이지를 인용하고, 가능하면 cite_uri 링크를 같이 제시하라.
근거가 없으면 '근거를 찾지 못했다'고 명확히 말하라.
`,

1-5) 동작 테스트 체크리스트

VS Code에서 MCP 서버 실행/연결 (mcp.json 설정)

채팅에서 아래처럼 실행:

#structai_import_files로 PDF/MD 넣기

#structai_get_project_summary

#structai_search_knowledge "KDS 전단강도"

결과에 cite_uri가 file:///.../xxx.pdf#page=12 형태로 뜨는지 확인

#structai_get_chunk로 원문 청크를 가져오고, 답변에서 페이지 인용이 자연스럽게 붙는지 확인

1-6) 다음 단계 예고 (2단계로 자연스럽게 이어짐)

1단계가 되면 “문서 근거 인용”이 잡히고, 다음은 자연스럽게:

2단계: DXF(도면) → 텍스트/레이어/블록/치수 추출 → 부재/단면/부호/그리드 인식

3단계: 도면 요소 ↔ 해석 모델 요소 자동 매핑 + confidence

4단계: 결과(내력비/변위/응력) + 기준(KDS) 근거를 묶어서 ‘대화형 안전성 평가’

원하면, **바로 다음 응답에서 2단계(DXF 파이프라인: ezdxf 기반 “도면 텍스트/레이어/블록 이름 추출 + 부재 후보 생성”)**도 같은 방식으로 “코드부터” 순서대로 이어서 진행할게.
