좋아. 이제 **7단계(단면 라이브러리 + 테이블/리더 기반 정밀 배정 + 형강 단면특성 자동 계산)**를 **“바로 붙여서 굴러가는 형태”**로 순서대로 정리하고, 필요한 **DB/서버/파서/도구/패널(UI)**까지 한 번에 제시할게.

⚠️ 다시 한 번 중요한 안전 고지
여기서 구현하는 **형강 단면특성 계산(Zx, Ix 등)**은 “기하학적 이상형(필렛/라운드 무시, 대칭 단면 가정)” 기반의 결정론 계산이야.
실제 표준 형강의 공칭 단면특성(제조사/KS/KSD/KSF 등)과 미세하게 다를 수 있으니, 실무 적용 시에는

표준 단면 라이브러리(카탈로그 값) 우선, 2) 계산값은 fallback/검증용으로 쓰는 구조로 가는 게 맞다.
(이번 단계에 “표준 단면 라이브러리 테이블”을 같이 넣는 이유가 그거야.)

7-1) DB 스키마 추가(v0.0.8)

Step6(v0.0.7)까지의 DB에 아래를 추가해줘.
mcp_server/schema.sql 맨 아래에 붙이면 돼.

-- ===== schema patch v0.0.8 =====
UPDATE meta SET value='0.0.8' WHERE key='schema_version';

-- -----------------------------
-- Section catalog (library)
-- -----------------------------
CREATE TABLE IF NOT EXISTS section_catalog (
  section_id INTEGER PRIMARY KEY AUTOINCREMENT,
  family TEXT NOT NULL,              -- 'steel_h'|'steel_box'|'rc_rect'|...
  name TEXT NOT NULL,                -- display name (e.g., H-400x200x8x13)
  name_norm TEXT NOT NULL,           -- normalized key (e.g., H400X200X8X13)
  dims_json TEXT NOT NULL DEFAULT '{}',
  props_json TEXT NOT NULL DEFAULT '{}',
  source TEXT,                       -- 'computed'|'csv_import'|'ks'...
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_section_catalog_family_name
ON section_catalog(family, name_norm);

CREATE TABLE IF NOT EXISTS section_aliases (
  alias_norm TEXT PRIMARY KEY,       -- normalized alias
  section_id INTEGER NOT NULL REFERENCES section_catalog(section_id) ON DELETE CASCADE
);

-- Member -> resolved section record (audit)
CREATE TABLE IF NOT EXISTS member_section_resolutions (
  model_member_id INTEGER PRIMARY KEY REFERENCES model_members(model_member_id) ON DELETE CASCADE,
  section_id INTEGER REFERENCES section_catalog(section_id) ON DELETE SET NULL,
  resolved_name TEXT,
  confidence REAL NOT NULL DEFAULT 0.6,
  method TEXT NOT NULL DEFAULT 'parsed',   -- parsed|catalog|manual
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- -----------------------------
-- CAD tables (extracted grids)
-- -----------------------------
CREATE TABLE IF NOT EXISTS cad_tables (
  table_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  method TEXT NOT NULL DEFAULT 'grid',
  bbox_json TEXT NOT NULL DEFAULT '{}',     -- {"minx":..,"miny":..,"maxx":..,"maxy":..}
  rows INTEGER NOT NULL DEFAULT 0,
  cols INTEGER NOT NULL DEFAULT 0,
  confidence REAL NOT NULL DEFAULT 0.5,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cad_table_cells (
  cell_id INTEGER PRIMARY KEY AUTOINCREMENT,
  table_id INTEGER NOT NULL REFERENCES cad_tables(table_id) ON DELETE CASCADE,
  row_idx INTEGER NOT NULL,
  col_idx INTEGER NOT NULL,
  cad_entity_id INTEGER REFERENCES cad_entities(cad_entity_id) ON DELETE SET NULL,
  text TEXT NOT NULL,
  x REAL, y REAL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_table_cells_table
ON cad_table_cells(table_id, row_idx, col_idx);

7-2) 형강 단면특성 계산 모듈 추가
7-2-1) 새 파일: mcp_server/design/steel_props.py
# mcp_server/design/steel_props.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


def norm_section_name(s: str) -> str:
    return (s or "").strip().upper().replace(" ", "").replace("-", "").replace("×", "X")


@dataclass
class SteelHSectionDims:
    H: float   # overall depth (mm)
    B: float   # flange width (mm)
    tw: float  # web thickness (mm)
    tf: float  # flange thickness (mm)


def compute_h_section_props(d: SteelHSectionDims) -> Dict[str, Any]:
    """
    Symmetric I/H section, sharp corners, no fillets.
    Returns:
      A (mm2), Ix/Iy (mm4), Sx/Sy (mm3), Zx/Zy (mm3), Aw (mm2)
    """
    H, B, tw, tf = float(d.H), float(d.B), float(d.tw), float(d.tf)
    if H <= 0 or B <= 0 or tw <= 0 or tf <= 0:
        raise ValueError("H,B,tw,tf must be positive.")
    if 2 * tf >= H:
        raise ValueError("Invalid dims: 2*tf must be < H.")
    if tw >= B:
        # 이론상 가능하지만 일반 H형강으론 비정상
        raise ValueError("Invalid dims: tw must be < B for typical H-section.")

    hw = H - 2.0 * tf  # web clear height

    # Areas
    Af = B * tf
    Aw = tw * hw
    A = 2.0 * Af + Aw

    # Second moments (centroidal)
    # Ix: strong axis (about horizontal axis through centroid)
    # flanges: local + parallel axis
    y_f = (H / 2.0 - tf / 2.0)
    Ix_flange_each = (B * tf**3) / 12.0 + Af * (y_f**2)
    Ix_web = (tw * hw**3) / 12.0
    Ix = 2.0 * Ix_flange_each + Ix_web

    # Iy: weak axis (about vertical axis through centroid)
    Iy_flange_each = (tf * B**3) / 12.0
    Iy_web = (hw * tw**3) / 12.0
    Iy = 2.0 * Iy_flange_each + Iy_web

    # Elastic section modulus
    Sx = Ix / (H / 2.0)
    Sy = Iy / (B / 2.0)

    # Plastic section modulus (symmetric -> PNA at centroid)
    # Zx = 2*(Af*y_f + Aw_top*y_w)
    Aw_top = tw * (H / 2.0 - tf)
    y_w = (H / 4.0 - tf / 2.0)
    Zx = 2.0 * (Af * y_f + Aw_top * y_w)

    # Zy: sum of plastic moduli of rectangles about y-axis
    # rectangle: Z = h*b^2/4 (b in x-direction)
    Zy = 2.0 * (tf * B**2 / 4.0) + (hw * tw**2 / 4.0)

    # Shear area (web)
    shear_area = Aw

    return {
        "family": "steel_h",
        "dims": {"H_mm": H, "B_mm": B, "tw_mm": tw, "tf_mm": tf},
        "props": {
            "A_mm2": A,
            "Ix_mm4": Ix,
            "Iy_mm4": Iy,
            "Sx_mm3": Sx,
            "Sy_mm3": Sy,
            "Zx_mm3": Zx,
            "Zy_mm3": Zy,
            "Aw_mm2": shear_area,
        },
    }


def canonical_h_name(H: int, B: int, tw: int, tf: int) -> str:
    return f"H-{H}x{B}x{tw}x{tf}"

7-3) “표준 단면 라이브러리(Section Catalog)” 도구 추가
7-3-1) 섹션 카탈로그 임포트 포맷
CSV 예시 (steel_h)
family,name,H_mm,B_mm,tw_mm,tf_mm,Fy_MPa,Zx_mm3,Aw_mm2,source
steel_h,H-400x200x8x13,400,200,8,13,325,1210000,2992,ks_catalog


표준 카탈로그 값이 있으면 Zx_mm3, Aw_mm2까지 넣고

없으면 dims만 넣고 서버에서 계산해도 됨

JSON 예시
{
  "items": [
    {
      "family": "steel_h",
      "name": "H-400x200x8x13",
      "dims": {"H_mm":400,"B_mm":200,"tw_mm":8,"tf_mm":13},
      "props": {"Zx_mm3": 1210000, "Aw_mm2": 2992},
      "source": "ks_catalog"
    }
  ]
}

7-3-2) server.py에 섹션 관련 MCP Tool 추가

아래는 “추가 도구들”이야. (기존 server.py에 붙여 넣기)

(A) 섹션 이름 정규화 + upsert 헬퍼
def _norm(s: str) -> str:
    return (s or "").strip().upper().replace(" ", "").replace("-", "").replace("×", "X")

def _upsert_section(conn, family: str, name: str, dims: dict, props: dict, source: str) -> int:
    name_norm = _norm(name)
    conn.execute(
        """
        INSERT INTO section_catalog(family, name, name_norm, dims_json, props_json, source)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(family, name_norm) DO UPDATE SET
          name=excluded.name,
          dims_json=excluded.dims_json,
          props_json=excluded.props_json,
          source=excluded.source
        """,
        (family, name, name_norm, json.dumps(dims, ensure_ascii=False), json.dumps(props, ensure_ascii=False), source),
    )
    r = conn.execute("SELECT section_id FROM section_catalog WHERE family=? AND name_norm=?", (family, name_norm)).fetchone()
    return int(r["section_id"])

(B) 섹션 카탈로그 임포트
@mcp.tool()
def structai_sections_import_catalog(path: str, fmt: Optional[str] = None) -> Dict[str, Any]:
    """
    section_catalog 임포트 (CSV/JSON)
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    ext = p.suffix.lower()
    if fmt is None:
        fmt = "json" if ext == ".json" else "csv"

    from mcp_server.design.steel_props import compute_h_section_props, SteelHSectionDims

    conn = _connect()
    try:
        imported = 0
        computed = 0

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

                if not family or not name:
                    continue

                # steel_h: if dims exist but props missing -> compute
                if family == "steel_h":
                    if ("Zx_mm3" not in props) or ("Aw_mm2" not in props):
                        d = SteelHSectionDims(
                            H=float(dims.get("H_mm")),
                            B=float(dims.get("B_mm")),
                            tw=float(dims.get("tw_mm")),
                            tf=float(dims.get("tf_mm")),
                        )
                        calc = compute_h_section_props(d)
                        props = {**calc["props"], **props}
                        computed += 1

                _upsert_section(conn, family, name, dims, props, source)
                imported += 1

        elif fmt == "csv":
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    family = str(row.get("family") or "").strip()
                    name = str(row.get("name") or "").strip()
                    if not family or not name:
                        continue

                    dims = {}
                    props = {}
                    source = row.get("source") or "csv_import"

                    # dims
                    for k in ("H_mm","B_mm","tw_mm","tf_mm","b_mm","h_mm"):
                        if row.get(k) not in (None, ""):
                            dims[k] = float(row[k])

                    # props
                    for k in ("Fy_MPa","Zx_mm3","Aw_mm2","Ix_mm4","Iy_mm4","A_mm2"):
                        if row.get(k) not in (None, ""):
                            props[k] = float(row[k])

                    if family == "steel_h":
                        if ("Zx_mm3" not in props) or ("Aw_mm2" not in props):
                            from mcp_server.design.steel_props import SteelHSectionDims, compute_h_section_props
                            d = SteelHSectionDims(
                                H=float(dims.get("H_mm")),
                                B=float(dims.get("B_mm")),
                                tw=float(dims.get("tw_mm")),
                                tf=float(dims.get("tf_mm")),
                            )
                            calc = compute_h_section_props(d)
                            props = {**calc["props"], **props}
                            computed += 1

                    _upsert_section(conn, family, name, dims, props, source)
                    imported += 1
        else:
            raise ValueError(f"unsupported fmt: {fmt}")

        conn.commit()
        return {"ok": True, "imported": imported, "computed_props": computed}
    finally:
        conn.close()

(C) 모델 부재 section 문자열 → 섹션 해석/적용

목표: model_members.section에 "H-400x200x8x13" 같은 문자열이 있으면

섹션 카탈로그에서 찾거나

직접 파싱해서 props 계산

member_design_inputs에 Zx, Aw 등 채우기

member_section_resolutions에 감사 기록

@mcp.tool()
def structai_sections_resolve_members(
    model_id: int,
    default_Fy_MPa: Optional[float] = None,
    overwrite_design: bool = False
) -> Dict[str, Any]:
    """
    model_members.section을 보고 steel_h 등 섹션을 resolve해서 design_inputs(Zx,Aw 등)에 채운다.
    """
    from mcp_server.parsing.specs import parse_specs_from_text
    from mcp_server.design.steel_props import SteelHSectionDims, compute_h_section_props, canonical_h_name

    conn = _connect()
    try:
        members = conn.execute(
            "SELECT model_member_id, member_uid, type, section FROM model_members WHERE model_id=?",
            (int(model_id),),
        ).fetchall()

        updated = 0
        skipped = []
        created_sections = 0

        for m in members:
            mmid = int(m["model_member_id"])
            sec = (m["section"] or "").strip()
            if not sec:
                continue

            # existing design json
            row = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
            dj = json.loads(row["design_json"]) if row else {}

            if not overwrite_design and (("Zx" in dj and dj.get("Zx")) or ("Aw" in dj and dj.get("Aw"))):
                skipped.append({"uid": m["member_uid"], "reason": "already has Zx/Aw"})
                continue

            # 1) try parse steel_h from section string
            specs = parse_specs_from_text(sec)
            steel_h = next((s for s in specs if s.get("spec_kind") == "steel_h_section"), None)

            resolved = None
            method = None
            confidence = 0.6
            section_id = None

            if steel_h:
                H = int(steel_h["H_mm"]); B = int(steel_h["B_mm"]); tw = int(steel_h["tw_mm"]); tf = int(steel_h["tf_mm"])
                name = canonical_h_name(H, B, tw, tf)
                name_norm = _norm(name)

                # try catalog
                r = conn.execute(
                    "SELECT section_id, dims_json, props_json FROM section_catalog WHERE family='steel_h' AND name_norm=?",
                    (name_norm,),
                ).fetchone()

                if r:
                    section_id = int(r["section_id"])
                    dims = json.loads(r["dims_json"] or "{}")
                    props = json.loads(r["props_json"] or "{}")
                    resolved = {"family":"steel_h","name":name,"dims":dims,"props":props}
                    method = "catalog"
                    confidence = 0.85
                else:
                    # compute and store as catalog (source='computed')
                    calc = compute_h_section_props(SteelHSectionDims(H=H, B=B, tw=tw, tf=tf))
                    dims = calc["dims"]
                    props = calc["props"]
                    section_id = _upsert_section(conn, "steel_h", name, dims, props, "computed")
                    created_sections += 1
                    resolved = {"family":"steel_h","name":name,"dims":dims,"props":props}
                    method = "parsed+computed"
                    confidence = 0.7

            if not resolved:
                continue

            # apply to design json
            props = resolved["props"] or {}
            # keys for design engine
            if "Zx_mm3" in props:
                dj["Zx"] = float(props["Zx_mm3"])   # (mm3) used by steel capacity formula
            if "Aw_mm2" in props:
                dj["Aw"] = float(props["Aw_mm2"])   # (mm2)
            if default_Fy_MPa is not None:
                dj.setdefault("Fy", float(default_Fy_MPa))
            else:
                # catalog may have Fy_MPa
                if "Fy_MPa" in props:
                    dj.setdefault("Fy", float(props["Fy_MPa"]))

            dj["section_resolved"] = resolved
            dj.setdefault("units", {})
            dj["units"].update({"length":"mm","stress":"MPa","steel_props":"mm/mm2/mm3/mm4"})

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

7-4) DXF Importer를 “리더/테이블” 분석 가능한 형태로 강화

Step4의 structai_import_dxf는 텍스트 중심이라, LINE/LWPOLYLINE 같은 “리더(화살표선)” 정보가 DB에 없을 수 있어.
Step7에서는 아래를 반영해:

cad_entities에 LINE, LWPOLYLINE도 저장(geom_json에 점 목록)

(가능하면) LEADER, MLEADER도 저장하려고 시도(실패해도 넘어가게)

7-4-1) structai_import_dxf 수정 포인트(요약)

기존 텍스트 수집 로직 유지

아래 루프 추가

# LINE
for e in msp.query("LINE"):
    try:
        s = e.dxf.start; t = e.dxf.end
        pts = [[float(s.x), float(s.y)], [float(t.x), float(t.y)]]
        conn.execute(
            """
            INSERT INTO cad_entities(artifact_id, chunk_id, handle, type, layer, layout, text, x,y,z, geom_json, raw_json)
            VALUES(?, NULL, ?, 'LINE', ?, 'Model', NULL, ?,?,?, ?, ?)
            """,
            (artifact_id, getattr(e.dxf,"handle",None), getattr(e.dxf,"layer",None),
             float(s.x), float(s.y), float(s.z),
             json.dumps({"points": pts}, ensure_ascii=False),
             json.dumps({"linetype": getattr(e.dxf,"linetype",None)}, ensure_ascii=False)),
        )
    except Exception:
        pass

# LWPOLYLINE
for e in msp.query("LWPOLYLINE"):
    try:
        pts = [[float(p[0]), float(p[1])] for p in e.get_points()]
        if not pts:
            continue
        x, y = pts[0][0], pts[0][1]
        conn.execute(
            """
            INSERT INTO cad_entities(artifact_id, chunk_id, handle, type, layer, layout, text, x,y,z, geom_json, raw_json)
            VALUES(?, NULL, ?, 'LWPOLYLINE', ?, 'Model', NULL, ?,?,NULL, ?, ?)
            """,
            (artifact_id, getattr(e.dxf,"handle",None), getattr(e.dxf,"layer",None),
             float(x), float(y),
             json.dumps({"points": pts}, ensure_ascii=False),
             json.dumps({"closed": bool(getattr(e,"closed",False))}, ensure_ascii=False)),
        )
    except Exception:
        pass


실제로는 네 server.py 버전이 여러 번 바뀌었으니,
“핵심은 cad_entities에 LINE/LWPOLYLINE이 들어오게 만들기”만 맞추면 돼.

7-5) CAD “표(Table)” 추출기 추가 (정밀 배정의 핵심)
7-5-1) 새 파일: mcp_server/parsing/tables.py
# mcp_server/parsing/tables.py
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TextPoint:
    cad_entity_id: int
    x: float
    y: float
    text: str
    layer: Optional[str] = None


def _auto_tol(vals: List[float], fallback: float) -> float:
    if len(vals) < 10:
        return fallback
    v = sorted(vals)
    diffs = []
    for i in range(1, len(v)):
        d = abs(v[i] - v[i-1])
        if d > 1e-6:
            diffs.append(d)
    if not diffs:
        return fallback
    m = median(diffs)
    # 도면 mm 스케일에서 row/col 간격이 5~20mm인 경우가 많아 clamp
    return max(2.0, min(25.0, float(m) * 0.35))


def _cluster_1d(values: List[float], tol: float) -> List[float]:
    """
    1D clustering by tolerance -> return cluster centers
    """
    if not values:
        return []
    vals = sorted(values)
    centers = [vals[0]]
    counts = [1]
    for v in vals[1:]:
        if abs(v - centers[-1]) <= tol:
            # update running mean
            c = centers[-1] * counts[-1] + v
            counts[-1] += 1
            centers[-1] = c / counts[-1]
        else:
            centers.append(v)
            counts.append(1)
    return centers


def extract_grid_tables(
    points: List[TextPoint],
    min_rows: int = 3,
    min_cols: int = 2,
    min_cells: int = 12,
    row_tol: Optional[float] = None,
    col_tol: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    매우 단순한 '그리드 테이블' 추정:
    - 같은 layer 안에서 x/y 정렬이 강한 텍스트 덩어리를 table로 본다.
    """
    if len(points) < min_cells:
        return []

    # row/col tolerance 자동 추정
    ys = [p.y for p in points]
    xs = [p.x for p in points]
    rt = row_tol if row_tol is not None else _auto_tol(ys, fallback=6.0)
    ct = col_tol if col_tol is not None else _auto_tol(xs, fallback=8.0)

    # 1) row clustering by y
    row_centers = _cluster_1d(ys, rt)
    if len(row_centers) < min_rows:
        return []

    # 2) col clustering by x
    col_centers = _cluster_1d(xs, ct)
    if len(col_centers) < min_cols:
        return []

    # build grid
    # cell assignment: nearest center if within tol
    cells: Dict[Tuple[int, int], List[TextPoint]] = {}
    for p in points:
        # row idx: closest y center
        rbest = None
        rdist = None
        for ri, yc in enumerate(row_centers):
            d = abs(p.y - yc)
            if rdist is None or d < rdist:
                rdist = d
                rbest = ri
        # col idx: closest x center
        cbest = None
        cdist = None
        for ci, xc in enumerate(col_centers):
            d = abs(p.x - xc)
            if cdist is None or d < cdist:
                cdist = d
                cbest = ci
        if rdist is None or cdist is None:
            continue
        if rdist > rt or cdist > ct:
            continue
        cells.setdefault((rbest, cbest), []).append(p)

    rows = len(row_centers)
    cols = len(col_centers)
    filled = len(cells)
    occupancy = filled / max(1, rows * cols)

    if rows < min_rows or cols < min_cols or filled < min_cells:
        return []

    # bbox
    minx = min(xs); maxx = max(xs)
    miny = min(ys); maxy = max(ys)

    # materialize table
    table_cells = []
    for (ri, ci), ps in cells.items():
        # multiple texts in same cell -> join by space (x 정렬)
        ps_sorted = sorted(ps, key=lambda p: p.x)
        text = " ".join([p.text.strip() for p in ps_sorted if p.text.strip()])
        # store one representative point
        rep = ps_sorted[0]
        table_cells.append({
            "row_idx": int(ri),
            "col_idx": int(ci),
            "cad_entity_id": int(rep.cad_entity_id),
            "text": text,
            "x": float(rep.x),
            "y": float(rep.y),
        })

    # confidence heuristic
    conf = 0.4 + 0.6 * min(1.0, occupancy)

    return [{
        "method": "grid",
        "rows": rows,
        "cols": cols,
        "confidence": conf,
        "bbox": {"minx": float(minx), "miny": float(miny), "maxx": float(maxx), "maxy": float(maxy)},
        "meta": {"row_tol": rt, "col_tol": ct, "occupancy": occupancy},
        "cells": table_cells,
    }]

7-5-2) table 추출 MCP Tool: structai_cad_extract_tables
@mcp.tool()
def structai_cad_extract_tables(
    cad_artifact_id: int,
    layer_filter: Optional[str] = None,
    min_cells: int = 16,
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    CAD 텍스트를 layer별로 모아서 grid table을 추정하고 cad_tables/cad_table_cells로 저장
    """
    from mcp_server.parsing.tables import TextPoint, extract_grid_tables

    conn = _connect()
    try:
        if overwrite:
            # cascade로 cell도 삭제됨
            conn.execute("DELETE FROM cad_tables WHERE cad_artifact_id=?", (int(cad_artifact_id),))

        rows = conn.execute(
            """
            SELECT cad_entity_id, x, y, text, layer
            FROM cad_entities
            WHERE artifact_id=? AND type IN ('TEXT','MTEXT','ATTRIB') AND text IS NOT NULL AND x IS NOT NULL AND y IS NOT NULL
            """,
            (int(cad_artifact_id),),
        ).fetchall()

        # group by layer
        by_layer: Dict[str, List[TextPoint]] = {}
        for r in rows:
            layer = (r["layer"] or "NO_LAYER")
            if layer_filter and (layer_filter.upper() not in layer.upper()):
                continue
            by_layer.setdefault(layer, []).append(TextPoint(
                cad_entity_id=int(r["cad_entity_id"]),
                x=float(r["x"]),
                y=float(r["y"]),
                text=str(r["text"]),
                layer=layer
            ))

        saved_tables = 0
        saved_cells = 0

        for layer, pts in by_layer.items():
            if len(pts) < min_cells:
                continue
            tables = extract_grid_tables(pts, min_cells=min_cells)
            for t in tables:
                cur = conn.execute(
                    """
                    INSERT INTO cad_tables(cad_artifact_id, method, bbox_json, rows, cols, confidence, meta_json)
                    VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        int(cad_artifact_id),
                        t["method"],
                        json.dumps(t["bbox"], ensure_ascii=False),
                        int(t["rows"]),
                        int(t["cols"]),
                        float(t["confidence"]),
                        json.dumps({**t.get("meta", {}), "layer": layer}, ensure_ascii=False),
                    ),
                )
                table_id = int(cur.lastrowid)
                saved_tables += 1

                for c in t["cells"]:
                    conn.execute(
                        """
                        INSERT INTO cad_table_cells(table_id, row_idx, col_idx, cad_entity_id, text, x, y)
                        VALUES(?,?,?,?,?,?,?)
                        """,
                        (table_id, c["row_idx"], c["col_idx"], c.get("cad_entity_id"), c["text"], c["x"], c["y"]),
                    )
                    saved_cells += 1

        conn.commit()
        return {"ok": True, "cad_artifact_id": int(cad_artifact_id), "tables": saved_tables, "cells": saved_cells}
    finally:
        conn.close()

7-6) “테이블/리더” 기반으로 Spec 링크를 정밀 추천

Step6는 “근접거리(spatial)” 기반이었지.
Step7은 다음 순서로 정밀도를 올린다:

table method: 같은 행(row)에 있는 token+spec은 거의 확실

leader method: 라벨/치수/철근을 리더선으로 연결한 경우

spatial fallback: 둘 다 없으면 거리 기반

7-6-1) 테이블 기반 링크 추천: structai_specs_suggest_links_from_tables
@mcp.tool()
def structai_specs_suggest_links_from_tables(
    cad_artifact_id: int,
    model_id: int,
    mapping_status: str = "confirmed",
    overwrite_suggested: bool = True
) -> Dict[str, Any]:
    """
    cad_tables/cells에서 행(row) 단위로 token을 찾고, 같은 행의 spec cad_entity_id로 cad_specs를 찾아 link 생성.
    """
    token_rx = re.compile(r"\b[A-Z]{1,4}\s?-?\s?\d{1,4}\b", re.IGNORECASE)

    conn = _connect()
    try:
        if overwrite_suggested:
            conn.execute(
                "DELETE FROM member_spec_links WHERE cad_artifact_id=? AND model_id=? AND status='suggested' AND method='table'",
                (int(cad_artifact_id), int(model_id)),
            )

        # token -> member ids
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

        # tables
        tables = conn.execute(
            "SELECT table_id, meta_json, confidence FROM cad_tables WHERE cad_artifact_id=? ORDER BY confidence DESC",
            (int(cad_artifact_id),),
        ).fetchall()

        created = 0
        sample = []

        for t in tables:
            table_id = int(t["table_id"])

            cells = conn.execute(
                "SELECT row_idx, col_idx, cad_entity_id, text FROM cad_table_cells WHERE table_id=?",
                (table_id,),
            ).fetchall()

            # group by row
            by_row: Dict[int, List[sqlite3.Row]] = {}
            for c in cells:
                by_row.setdefault(int(c["row_idx"]), []).append(c)

            for ri, row_cells in by_row.items():
                # row text list
                row_texts = [str(c["text"]) for c in row_cells]
                row_join = " | ".join(row_texts)

                # find token in row
                tok = None
                for m in token_rx.finditer(row_join.upper()):
                    cand = normalize_label(m.group(0))
                    if cand in token_to_members:
                        tok = cand
                        break
                if not tok:
                    continue

                member_ids = token_to_members.get(tok, [])
                if not member_ids:
                    continue

                # find cad_specs within same row by cad_entity_id
                row_entity_ids = [int(c["cad_entity_id"]) for c in row_cells if c["cad_entity_id"] is not None]
                if not row_entity_ids:
                    continue

                spec_rows = conn.execute(
                    f"""
                    SELECT spec_id, cad_entity_id, spec_kind, raw_text, confidence
                    FROM cad_specs
                    WHERE cad_artifact_id=? AND cad_entity_id IN ({",".join(["?"]*len(row_entity_ids))})
                    """,
                    (int(cad_artifact_id), *row_entity_ids),
                ).fetchall()

                for sp in spec_rows:
                    spec_id = int(sp["spec_id"])
                    for mmid in member_ids:
                        evidence = {
                            "table_id": table_id,
                            "row_idx": ri,
                            "token_norm": tok,
                            "spec_confidence": float(sp["confidence"] or 0.5),
                            "table_confidence": float(t["confidence"] or 0.5),
                        }
                        conn.execute(
                            """
                            INSERT INTO member_spec_links(
                              cad_artifact_id, spec_id, model_id, model_member_id,
                              cad_token_norm, distance, method, status, evidence_json, updated_at
                            ) VALUES(?,?,?,?,?,?,?,'suggested',?, datetime('now'))
                            ON CONFLICT(cad_artifact_id, spec_id, model_member_id)
                            DO UPDATE SET
                              method='table',
                              status='suggested',
                              distance=0,
                              evidence_json=excluded.evidence_json,
                              updated_at=datetime('now')
                            """,
                            (
                                int(cad_artifact_id),
                                spec_id,
                                int(model_id),
                                int(mmid),
                                tok,
                                0.0,
                                "table",
                                json.dumps(evidence, ensure_ascii=False),
                            ),
                        )
                        created += 1
                        if len(sample) < 100:
                            sample.append({"token": tok, "model_member_id": mmid, "spec_id": spec_id, "spec_kind": sp["spec_kind"], "raw": sp["raw_text"]})
        conn.commit()
        return {"ok": True, "created": created, "sample": sample}
    finally:
        conn.close()

7-6-2) 리더 기반 링크 추천(라인/폴리라인 endpoint 연결)

DXF에서 리더가 “그냥 LINE”인 경우도 많아서,
이 로직은 “라인 한쪽 끝이 token 근처 + 다른 끝이 spec 근처”면 연결로 본다.

def _grid_key(x: float, y: float, cell: float) -> Tuple[int, int]:
    return (int(x // cell), int(y // cell))

def _neighbor_keys(k: Tuple[int,int]) -> List[Tuple[int,int]]:
    i, j = k
    return [(i+di, j+dj) for di in (-1,0,1) for dj in (-1,0,1)]

def _endpoints_from_geom(geom_json: str) -> Optional[Tuple[Tuple[float,float], Tuple[float,float]]]:
    try:
        g = json.loads(geom_json or "{}")
        pts = g.get("points") or []
        if len(pts) < 2:
            return None
        a = pts[0]; b = pts[-1]
        return (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
    except Exception:
        return None

@mcp.tool()
def structai_specs_suggest_links_by_leaders(
    cad_artifact_id: int,
    model_id: int,
    mapping_status: str = "confirmed",
    snap_tol: float = 20.0,
    overwrite_suggested: bool = True
) -> Dict[str, Any]:
    """
    LINE/LWPOLYLINE endpoints로 token text <-> spec text 연결을 추정하여 link 생성
    """
    conn = _connect()
    try:
        if overwrite_suggested:
            conn.execute(
                "DELETE FROM member_spec_links WHERE cad_artifact_id=? AND model_id=? AND status='suggested' AND method='leader'",
                (int(cad_artifact_id), int(model_id)),
            )

        # 1) token points (from confirmed mappings)
        maps = conn.execute(
            """
            SELECT DISTINCT cad_token_norm
            FROM member_mappings
            WHERE cad_artifact_id=? AND model_id=? AND status=?
            """,
            (int(cad_artifact_id), int(model_id), str(mapping_status)),
        ).fetchall()
        tokens = [r["cad_token_norm"] for r in maps]

        token_points = []  # (token_norm, x, y)
        for tok in tokens:
            rows = conn.execute(
                """
                SELECT x,y
                FROM cad_entities
                WHERE artifact_id=? AND text IS NOT NULL AND x IS NOT NULL AND y IS NOT NULL
                  AND instr(upper(text), ?) > 0
                """,
                (int(cad_artifact_id), tok.upper()),
            ).fetchall()
            for r in rows[:10]:
                token_points.append((tok, float(r["x"]), float(r["y"])))

        # map token->member ids
        rows = conn.execute(
            """
            SELECT cad_token_norm, model_member_id
            FROM member_mappings
            WHERE cad_artifact_id=? AND model_id=? AND status=?
            """,
            (int(cad_artifact_id), int(model_id), str(mapping_status)),
        ).fetchall()
        token_to_members: Dict[str, List[int]] = {}
        for r in rows:
            token_to_members.setdefault(r["cad_token_norm"], []).append(int(r["model_member_id"]))

        # 2) spec points
        specs = conn.execute(
            """
            SELECT spec_id, x, y, spec_kind, confidence
            FROM cad_specs
            WHERE cad_artifact_id=? AND x IS NOT NULL AND y IS NOT NULL
            """,
            (int(cad_artifact_id),),
        ).fetchall()
        spec_points = [(int(s["spec_id"]), float(s["x"]), float(s["y"]), s["spec_kind"], float(s["confidence"] or 0.5)) for s in specs]

        # 3) leader candidates = LINE/LWPOLYLINE
        lines = conn.execute(
            """
            SELECT cad_entity_id, type, layer, geom_json
            FROM cad_entities
            WHERE artifact_id=? AND type IN ('LINE','LWPOLYLINE') AND geom_json IS NOT NULL
            """,
            (int(cad_artifact_id),),
        ).fetchall()

        # spatial hash
        cell = float(snap_tol)
        tok_grid: Dict[Tuple[int,int], List[Tuple[str,float,float]]] = {}
        for tok, x, y in token_points:
            tok_grid.setdefault(_grid_key(x,y,cell), []).append((tok,x,y))

        spec_grid: Dict[Tuple[int,int], List[Tuple[int,float,float,str,float]]] = {}
        for sid, x, y, kind, conf in spec_points:
            spec_grid.setdefault(_grid_key(x,y,cell), []).append((sid,x,y,kind,conf))

        created = 0
        sample = []

        def find_near_token(x: float, y: float) -> Optional[Tuple[str, float]]:
            k = _grid_key(x,y,cell)
            best = None
            for nk in _neighbor_keys(k):
                for tok, tx, ty in tok_grid.get(nk, []):
                    d = ((tx-x)**2 + (ty-y)**2) ** 0.5
                    if d <= snap_tol and (best is None or d < best[1]):
                        best = (tok, d)
            return best

        def find_near_spec(x: float, y: float) -> Optional[Tuple[int, float, str, float]]:
            k = _grid_key(x,y,cell)
            best = None
            for nk in _neighbor_keys(k):
                for sid, sx, sy, kind, conf in spec_grid.get(nk, []):
                    d = ((sx-x)**2 + (sy-y)**2) ** 0.5
                    if d <= snap_tol and (best is None or d < best[1]):
                        best = (sid, d, kind, conf)
            return best

        for ln in lines:
            ep = _endpoints_from_geom(ln["geom_json"])
            if not ep:
                continue
            (x1,y1), (x2,y2) = ep

            # endpoint A token, endpoint B spec
            t1 = find_near_token(x1,y1)
            s2 = find_near_spec(x2,y2)

            # swap
            t2 = find_near_token(x2,y2)
            s1 = find_near_spec(x1,y1)

            pairs = []
            if t1 and s2:
                pairs.append((t1, s2))
            if t2 and s1:
                pairs.append((t2, s1))

            for (tok, dt), (sid, ds, kind, sconf) in pairs:
                for mmid in token_to_members.get(tok, []):
                    # confidence heuristic
                    conf = max(0.7, min(0.98, 0.85 + 0.05*(1 - dt/snap_tol) + 0.05*(1 - ds/snap_tol) + 0.03*(sconf-0.5)))
                    evidence = {
                        "line_entity_id": int(ln["cad_entity_id"]),
                        "line_type": ln["type"],
                        "line_layer": ln["layer"],
                        "tok_dist": dt,
                        "spec_dist": ds,
                        "snap_tol": snap_tol,
                        "spec_confidence": sconf,
                        "confidence": conf
                    }
                    conn.execute(
                        """
                        INSERT INTO member_spec_links(
                          cad_artifact_id, spec_id, model_id, model_member_id,
                          cad_token_norm, distance, method, status, evidence_json, updated_at
                        ) VALUES(?,?,?,?,?,?,?,'suggested',?, datetime('now'))
                        ON CONFLICT(cad_artifact_id, spec_id, model_member_id)
                        DO UPDATE SET
                          method='leader',
                          status='suggested',
                          distance=excluded.distance,
                          evidence_json=excluded.evidence_json,
                          updated_at=datetime('now')
                        """,
                        (int(cad_artifact_id), int(sid), int(model_id), int(mmid),
                         tok, float(dt+ds), "leader", json.dumps(evidence, ensure_ascii=False)),
                    )
                    created += 1
                    if len(sample) < 100:
                        sample.append({"token": tok, "model_member_id": mmid, "spec_id": sid, "spec_kind": kind, "confidence": conf})

        conn.commit()
        return {"ok": True, "created": created, "sample": sample}
    finally:
        conn.close()

7-6-3) “통합 추천” 도구: table + leader + spatial

table/leader 추천을 먼저 넣고

마지막에 spatial(Step6의 기존 추천)을 호출하거나, 기존 spatial 로직을 재사용

가장 깔끔한 형태는 새 도구로 묶는 거야:

@mcp.tool()
def structai_specs_suggest_links_advanced(
    cad_artifact_id: int,
    model_id: int,
    mapping_status: str = "confirmed",
    max_dist: float = 500.0,
    snap_tol: float = 20.0
) -> Dict[str, Any]:
    """
    1) table 기반
    2) leader 기반
    3) spatial 기반 (fallback)
    """
    r1 = structai_specs_suggest_links_from_tables(cad_artifact_id, model_id, mapping_status, overwrite_suggested=False)
    r2 = structai_specs_suggest_links_by_leaders(cad_artifact_id, model_id, mapping_status, snap_tol, overwrite_suggested=False)
    r3 = structai_specs_suggest_links(cad_artifact_id, model_id, mapping_status=mapping_status, max_dist=max_dist, overwrite_suggested=False)  # Step6의 spatial 함수

    return {
        "ok": True,
        "table": r1,
        "leader": r2,
        "spatial": {"created_links": r3.get("created_links"), "sample": (r3.get("suggestions") or [])[:50]},
    }


structai_specs_suggest_links(...)는 Step6에 만든 spatial 추천 도구를 그대로 재사용하면 됨.

7-7) “자동 확정” 정책 도구 (table/leader 우선 확정)

초기에는 사람이 UI에서 확정하는 게 정답이지만, 생산성을 위해 정책 기반 auto-confirm을 넣는 게 좋아.

@mcp.tool()
def structai_specs_auto_confirm(
    cad_artifact_id: int,
    model_id: int,
    confirm_table: bool = True,
    confirm_leader: bool = True,
    max_distance_for_spatial: float = 80.0
) -> Dict[str, Any]:
    """
    - table: 전부 confirmed
    - leader: evidence.confidence >= 0.9 정도만 confirmed
    - spatial: distance 매우 가까운 것만 confirmed(선택)
    """
    conn = _connect()
    try:
        confirmed = 0

        if confirm_table:
            cur = conn.execute(
                """
                UPDATE member_spec_links
                SET status='confirmed', updated_at=datetime('now')
                WHERE cad_artifact_id=? AND model_id=? AND status='suggested' AND method='table'
                """,
                (int(cad_artifact_id), int(model_id)),
            )
            confirmed += int(cur.rowcount)

        if confirm_leader:
            rows = conn.execute(
                """
                SELECT link_id, evidence_json
                FROM member_spec_links
                WHERE cad_artifact_id=? AND model_id=? AND status='suggested' AND method='leader'
                """,
                (int(cad_artifact_id), int(model_id)),
            ).fetchall()
            ids = []
            for r in rows:
                try:
                    ev = json.loads(r["evidence_json"] or "{}")
                    conf = float(ev.get("confidence", 0.0))
                    if conf >= 0.9:
                        ids.append(int(r["link_id"]))
                except Exception:
                    pass
            if ids:
                conn.execute(
                    f"UPDATE member_spec_links SET status='confirmed', updated_at=datetime('now') WHERE link_id IN ({','.join(['?']*len(ids))})",
                    ids,
                )
                confirmed += len(ids)

        # spatial very close
        rows = conn.execute(
            """
            SELECT link_id, distance
            FROM member_spec_links
            WHERE cad_artifact_id=? AND model_id=? AND status='suggested' AND method='spatial'
            """,
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()
        ids = [int(r["link_id"]) for r in rows if (r["distance"] is not None and float(r["distance"]) <= float(max_distance_for_spatial))]
        if ids:
            conn.execute(
                f"UPDATE member_spec_links SET status='confirmed', updated_at=datetime('now') WHERE link_id IN ({','.join(['?']*len(ids))})",
                ids,
            )
            confirmed += len(ids)

        conn.commit()
        return {"ok": True, "confirmed": confirmed}
    finally:
        conn.close()

7-8) CAD spec 적용 시 “형강 props 자동 채움”까지 반영

Step6의 structai_design_apply_specs_to_inputs에 아래를 추가하면, CAD에서 파싱된 steel_h_section이 들어오는 순간:

Zx, Aw 자동 세팅

section_resolved 같이 저장

from mcp_server.design.steel_props import SteelHSectionDims, compute_h_section_props, canonical_h_name

# ... inside kind == "steel_h_section":
H = int(spec.get("H_mm"))
B = int(spec.get("B_mm"))
tw = int(spec.get("tw_mm"))
tf = int(spec.get("tf_mm"))

calc = compute_h_section_props(SteelHSectionDims(H=H,B=B,tw=tw,tf=tf))
props = calc["props"]

patch["Zx"] = float(props["Zx_mm3"])
patch["Aw"] = float(props["Aw_mm2"])
patch["section_resolved"] = {"family":"steel_h","name": canonical_h_name(H,B,tw,tf), "dims": calc["dims"], "props": props}


이렇게 하면 CAD 표(부재표)에 형강 규격이 있는 프로젝트는 모델 section이 부정확해도 CAD에서 바로 들어온다.

7-9) (중요) 체크 엔진은 Step6의 ratio_expr 지원을 반드시 포함

Step7에서 상/하부근, (+)/(-)휨, 조합별 envelope 처리까지 제대로 하려면:

룰셋에 ratio_expr 지원

예: max(abs(M3_max)/Mn_pos, abs(M3_min)/Mn_neg)

이 패치는 Step6에서 이미 제안했는데, 아직 안 넣었다면 이번 Step7에서 꼭 같이 넣어줘.

7-10) Step7 “실사용 순서”(이대로 하면 한 바퀴 돈다)

DXF 재임포트(라인 포함되게 업데이트한 importer로)

#structai_import_dxf { "path": ".../S-101.dxf" }


CAD 스펙 파싱

#structai_cad_parse_specs { "cad_artifact_id": 1, "overwrite": true }


테이블 추출(부재표/철근표가 있는 레이어 중심으로)

#structai_cad_extract_tables { "cad_artifact_id": 1, "min_cells": 16 }


매핑(도면 token ↔ 모델 member) 확정되어 있어야 함 (Step3)

스펙 링크 정밀 추천(테이블+리더+공간)

#structai_specs_suggest_links_advanced {
  "cad_artifact_id": 1,
  "model_id": 1,
  "max_dist": 500.0,
  "snap_tol": 20.0
}


자동 확정(table/leader 우선)

#structai_specs_auto_confirm { "cad_artifact_id": 1, "model_id": 1 }


확정된 spec → design_inputs 반영

#structai_design_apply_specs_to_inputs { "cad_artifact_id": 1, "model_id": 1, "overwrite_keys": false }


(선택) 모델 section 기준 섹션 resolve (CAD가 없는 경우 대비)

#structai_sections_resolve_members { "model_id": 1, "default_Fy_MPa": 325, "overwrite_design": false }


내력 계산(steel/rc)

RC/Steel 계산 도구(Step5) 실행 → 체크 → 리포트 생성

7-11) VS Code 패널(UI)까지 한 번에 붙이고 싶으면

가장 빠른 방법은 기존 Mapping/Results 패널에 버튼을 추가하는 거야:

Mapping 패널에:

Parse Specs

Extract Tables

Suggest Spec Links (Advanced)

Auto Confirm

Apply Specs to Inputs

이 버튼들은 모두 “MCP tool 호출”만 하면 되니까 구현 난이도 낮아.

원하면 다음 메시지에서 내가:

SpecsViewProvider(별도 탭) 버전 /

기존 mapping 뷰 확장 버전
둘 중 하나를 골라 바로 코드로 제공해줄게. (확인 질문은 안 하고, 추천은 “기존 mapping 뷰 확장”이 가장 빠르고 유지보수도 쉬움)

다음 Step 8(정밀도 폭증 구간) 예고

Step7까지 하면 “도면 표/리더 기반 배정 + 형강 props 자동”이 되어서 체감이 확 오는데,
실무 정밀도가 진짜 올라가는 건 다음이야:

형강 표준 라이브러리(공칭 단면특성) 우선 적용 + 제조사/KS 데이터셋 import

리더/표 인식 강화

블록 속성(ATTRIB) 테이블

행/열 규칙(헤더 인식)

같은 행의 “토큰/단면/철근” 컬럼 자동 추정

다층(층별 동일 라벨) 문제 해결

story/층 표기, 도면 시트/뷰 범위(레이아웃)로 disambiguation

원하면 바로 다음 응답에서 Step 8을 위 3개 중심으로 “결정론 + DB + UI”까지 같이 이어서 구현해줄게.