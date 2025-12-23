좋아. 이제 6단계를 “실제로 자동화가 체감되는 단계”로 잡고, 아래 3가지를 순서대로 구현해줄게.

도면(CAD) 텍스트에서 단면/철근 스펙 파싱 → 구조화 저장 → 부재에 자동 할당

부재 타입별 입력 스키마/검증(Validation) → 누락/단위 이상/룰셋 요구 변수 자동 점검

KDS 조항번호 ↔ 체크타입 매핑(Codebook) → 룰셋 citations.query 자동 생성/검증

⚠️ 안전 고지: 아래 파서는 “현장 표기 관행”을 최대한 커버하도록 설계한 자동화 기반이지만, 도면 표기 방식이 프로젝트마다 달라서 100% 자동은 불가능해. 그래서 설계는 “추천 → 패널에서 확정” 흐름으로 짜는 게 정답이야.
또한 내력식/조항 적용은 최종 설계 책임을 대체하지 못함(검증/리뷰 필수).

6-1) 데이터 흐름(완성형)

CAD → Specs → Member → Design Inputs → Capacity Compute → Check → Report

cad_entities(text)에서 스펙 텍스트 파싱

파싱 결과를 cad_specs로 저장(감사/수정 가능)

member_mappings(확정된 도면라벨↔모델부재) 기준으로
라벨 위치 주변의 spec을 찾아 member_spec_links(추천) 생성

확정/자동 승인된 링크를 member_design_inputs에 반영

설계엔진(결정론)으로 Mn_pos/Mn_neg/Vn/Pn… 계산

룰셋(rulepack)에서 ratio_expr로 실제 체크 수행

리포트 생성

6-2) DB 스키마 추가(v0.0.7)

mcp_server/schema.sql 맨 아래에 추가해줘.

-- ===== schema patch v0.0.7 =====
UPDATE meta SET value='0.0.7' WHERE key='schema_version';

-- 1) CAD text -> parsed specs
CREATE TABLE IF NOT EXISTS cad_specs (
  spec_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  cad_entity_id INTEGER REFERENCES cad_entities(cad_entity_id) ON DELETE SET NULL,

  spec_kind TEXT NOT NULL,          -- rc_rect_section | rebar_main | rebar_stirrup | steel_h_section | ...
  spec_json TEXT NOT NULL,
  raw_text TEXT NOT NULL,

  x REAL, y REAL, z REAL,
  layer TEXT,
  confidence REAL NOT NULL DEFAULT 0.5,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_specs_artifact ON cad_specs(cad_artifact_id, spec_kind);
CREATE INDEX IF NOT EXISTS idx_cad_specs_entity ON cad_specs(cad_entity_id);

-- 2) Spec <-> Member link (suggested/confirmed)
CREATE TABLE IF NOT EXISTS member_spec_links (
  link_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  spec_id INTEGER NOT NULL REFERENCES cad_specs(spec_id) ON DELETE CASCADE,

  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,

  cad_token_norm TEXT,
  distance REAL,
  method TEXT NOT NULL,             -- spatial | table | manual
  status TEXT NOT NULL DEFAULT 'suggested', -- suggested | confirmed | rejected
  evidence_json TEXT NOT NULL DEFAULT '{}',

  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),

  UNIQUE(cad_artifact_id, spec_id, model_member_id)
);

CREATE INDEX IF NOT EXISTS idx_member_spec_links_lookup
ON member_spec_links(cad_artifact_id, model_id, status, model_member_id);

-- 3) Codebook (KDS clause mapping)
CREATE TABLE IF NOT EXISTS codebooks (
  codebook_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  codebook_json TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_codebooks_namever ON codebooks(name, version);

6-3) 파서 모듈 추가: mcp_server/parsing/specs.py
6-3-1) 패키지 준비

mcp_server/__init__.py (빈 파일) 추가
mcp_server/parsing/__init__.py (빈 파일) 추가

6-3-2) mcp_server/parsing/specs.py
# mcp_server/parsing/specs.py
from __future__ import annotations

import re
from dataclasses import dataclass
from math import pi
from typing import Any, Dict, List, Optional, Tuple


def rebar_area_mm2(dia_mm: float) -> float:
    return pi * (float(dia_mm) ** 2) / 4.0


# --- Regexes (최대한 관용 표기 커버) ---
# 1) RC 직사각 단면: 400x600, C400x600, B400×600 등
RC_RECT_RX = re.compile(
    r"(?:\b|^)(?:RC\s*)?(?:BEAM|COLUMN|B|C)?\s*(?P<b>\d{2,4})\s*([xX×*])\s*(?P<h>\d{2,4})(?:\b|$)"
)

# 2) 강재 H형강: H-400x200x8x13, H400×200×8×13
STEEL_H_RX = re.compile(
    r"(?:\b|^)\s*H\s*[-]?\s*(?P<H>\d{2,4})\s*([xX×*])\s*(?P<B>\d{2,4})\s*([xX×*])\s*(?P<tw>\d{1,3})\s*([xX×*])\s*(?P<tf>\d{1,3})(?:\b|$)",
    re.IGNORECASE
)

# 3) 띠철근/스터럽: D10@150, 2-D10@150(다리=2로 보는 관용)
STIRRUP_RX = re.compile(
    r"(?:(?P<legs>\d+)\s*[-]?\s*)?(?P<prefix>HD|SD|D|Ø|∅)\s*(?P<dia>\d{2})\s*@\s*(?P<s>\d{2,4})",
    re.IGNORECASE
)

# 4) 주근: 4-D25, D25-4EA, D25 4EA, 4Ø25 등
MAIN_A_RX = re.compile(
    r"(?P<count>\d+)\s*[-]?\s*(?P<prefix>HD|SD|D|Ø|∅)\s*(?P<dia>\d{2})\b",
    re.IGNORECASE
)
MAIN_B_RX = re.compile(
    r"(?P<prefix>HD|SD|D|Ø|∅)\s*(?P<dia>\d{2})\s*[-]?\s*(?P<count>\d+)\s*(?:EA|E\.A\.|개)?",
    re.IGNORECASE
)

# 상/하부 키워드 (도면/표기 관용)
TOP_KEYS = ["TOP", "UPPER", "상부", "상", "T/"]
BOT_KEYS = ["BOT", "BOTTOM", "LOWER", "하부", "하", "B/"]


def _pos_hint(text: str, idx: int, window: int = 20) -> str:
    """매치 시작점 주변(앞쪽)에서 상/하부 힌트를 찾는다."""
    s = text[max(0, idx - window):idx].upper()
    # 한글은 upper 영향 없으니 그대로 포함됨
    for k in TOP_KEYS:
        if k.upper() in s:
            return "TOP"
    for k in BOT_KEYS:
        if k.upper() in s:
            return "BOT"
    return "UNKNOWN"


def _near_has_at(text: str, end: int, window: int = 6) -> bool:
    """주근 매치 뒤에 @가 붙어있으면(예: 2-D10@150) 스터럽으로 취급."""
    return "@" in text[end:end + window]


def parse_specs_from_text(raw_text: str) -> List[Dict[str, Any]]:
    """
    한 줄/한 엔티티 텍스트에서 스펙들을 최대한 추출.
    반환: [{spec_kind, fields..., confidence, raw_text_fragment}]
    """
    t0 = (raw_text or "").strip()
    if not t0:
        return []

    # 분석 편의용 (영문은 upper, 한글은 그대로)
    t = t0.replace("＊", "*").replace("Ｘ", "X").replace("×", "×")

    specs: List[Dict[str, Any]] = []

    # 1) Steel H section
    m = STEEL_H_RX.search(t)
    if m:
        H = int(m.group("H"))
        B = int(m.group("B"))
        tw = int(m.group("tw"))
        tf = int(m.group("tf"))
        # sanity
        if 100 <= H <= 2000 and 50 <= B <= 1000:
            specs.append({
                "spec_kind": "steel_h_section",
                "shape": "H",
                "H_mm": H, "B_mm": B, "tw_mm": tw, "tf_mm": tf,
                "confidence": 0.9,
                "raw_fragment": m.group(0).strip(),
            })

    # 2) RC rectangular section
    # steel H가 이미 잡혔다면, 동일 텍스트에서 400x200 부분을 RC로 중복 파싱하는 것을 약하게 억제
    if not any(s["spec_kind"] == "steel_h_section" for s in specs):
        for m in RC_RECT_RX.finditer(t):
            b = int(m.group("b"))
            h = int(m.group("h"))
            # RC 단면 plausibility
            if 100 <= b <= 2000 and 100 <= h <= 3000:
                specs.append({
                    "spec_kind": "rc_rect_section",
                    "b_mm": b, "h_mm": h,
                    "confidence": 0.8,
                    "raw_fragment": m.group(0).strip(),
                })

    # 3) Stirrups (can be multiple)
    for m in STIRRUP_RX.finditer(t):
        dia = int(m.group("dia"))
        s = int(m.group("s"))
        legs = m.group("legs")
        legs_n = int(legs) if legs else 2  # 관용 기본 2다리
        if 6 <= dia <= 19 and 50 <= s <= 400:
            Av = legs_n * rebar_area_mm2(dia)
            specs.append({
                "spec_kind": "rebar_stirrup",
                "dia_mm": dia,
                "legs": legs_n,
                "s_mm": s,
                "Av_mm2": Av,
                "confidence": 0.8,
                "raw_fragment": m.group(0).strip(),
            })

    # 4) Main bars (two patterns)
    for rx in (MAIN_A_RX, MAIN_B_RX):
        for m in rx.finditer(t):
            # exclude stirrup-like (2-D10@150)
            if _near_has_at(t, m.end()):
                continue
            dia = int(m.group("dia"))
            count = int(m.group("count"))
            if not (10 <= dia <= 43 and 1 <= count <= 40):
                continue

            pos = _pos_hint(t, m.start())
            As = count * rebar_area_mm2(dia)
            specs.append({
                "spec_kind": "rebar_main",
                "pos": pos,          # TOP/BOT/UNKNOWN
                "dia_mm": dia,
                "count": count,
                "As_mm2": As,
                "confidence": 0.75,
                "raw_fragment": m.group(0).strip(),
            })

    return specs

6-4) server.py에 “Spec 파싱/추천/적용” 도구 추가

아래는 Step 5까지 구현된 server.py에 추가하는 형태로 적었어.
(이미 존재하는 함수/테이블과 충돌하지 않게 이름을 맞췄음)

6-4-1) import 추가

server.py 상단에 추가:

from mcp_server.parsing.specs import parse_specs_from_text

6-4-2) CAD 스펙 파싱 저장 도구
@mcp.tool()
def structai_cad_parse_specs(
    cad_artifact_id: int,
    overwrite: bool = True,
    include_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    cad_entities.text에서 단면/철근 스펙을 파싱해 cad_specs에 저장
    """
    include_types = include_types or ["TEXT", "MTEXT", "ATTRIB"]

    conn = _connect()
    try:
        # clear old
        if overwrite:
            conn.execute("DELETE FROM cad_specs WHERE cad_artifact_id=?", (int(cad_artifact_id),))

        rows = conn.execute(
            f"""
            SELECT cad_entity_id, type, layer, x, y, z, text
            FROM cad_entities
            WHERE artifact_id=? AND type IN ({",".join(["?"]*len(include_types))}) AND text IS NOT NULL
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

                spec_json = json.dumps(s, ensure_ascii=False)
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
                        spec_json,
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

6-4-3) CAD 스펙 목록 도구(검수용)
@mcp.tool()
def structai_cad_list_specs(
    cad_artifact_id: int,
    spec_kind: Optional[str] = None,
    limit: int = 200
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

6-4-4) “확정 매핑된 부재” 기준으로 스펙 링크 추천

핵심 아이디어:

member_mappings(status=confirmed)로 member 리스트 확보

각 token을 cad_entities에서 찾아 “라벨 위치(점)” 확보

그 점에서 가까운 cad_specs를 찾아 추천

def _dist2d(ax, ay, bx, by) -> Optional[float]:
    if ax is None or ay is None or bx is None or by is None:
        return None
    try:
        return float(((ax - bx)**2 + (ay - by)**2) ** 0.5)
    except Exception:
        return None


@mcp.tool()
def structai_specs_suggest_links(
    cad_artifact_id: int,
    model_id: int,
    mapping_status: str = "confirmed",
    max_dist: float = 500.0,          # DXF 단위(보통 mm). 프로젝트에 맞춰 조정
    overwrite_suggested: bool = True
) -> Dict[str, Any]:
    """
    member_mappings(확정) + token 위치 주변 spec으로 link 추천을 생성(member_spec_links 저장)
    """
    conn = _connect()
    try:
        if overwrite_suggested:
            conn.execute(
                """
                DELETE FROM member_spec_links
                WHERE cad_artifact_id=? AND model_id=? AND status='suggested'
                """,
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

        # preload specs
        specs = conn.execute(
            """
            SELECT spec_id, cad_entity_id, spec_kind, spec_json, raw_text, x, y, layer, confidence
            FROM cad_specs
            WHERE cad_artifact_id=?
            """,
            (int(cad_artifact_id),),
        ).fetchall()

        # helper: token points 찾기 (간단히 upper(text) contains token_norm)
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

            # member마다 spec_kind별로 하나씩만(가장 가까운 것)
            best_by_kind: Dict[str, Tuple[sqlite3.Row, float]] = {}

            for s in specs:
                sx, sy = s["x"], s["y"]
                if sx is None or sy is None:
                    continue
                # min dist to any token point
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
                if (prev is None) or (dmin < prev[1]):
                    best_by_kind[kind] = (s, dmin)

            # 저장
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

6-4-5) 링크(confirmed) → design_inputs 패치 적용

“추천→확정” UI가 아직 없다면, 일단 suggested를 일괄 confirmed로 바꾸는 옵션까지 같이 넣어줄게.

@mcp.tool()
def structai_specs_confirm_all(
    cad_artifact_id: int,
    model_id: int
) -> Dict[str, Any]:
    conn = _connect()
    try:
        cur = conn.execute(
            """
            UPDATE member_spec_links
            SET status='confirmed', updated_at=datetime('now')
            WHERE cad_artifact_id=? AND model_id=? AND status='suggested'
            """,
            (int(cad_artifact_id), int(model_id)),
        )
        conn.commit()
        return {"ok": True, "confirmed": int(cur.rowcount)}
    finally:
        conn.close()


@mcp.tool()
def structai_design_apply_specs_to_inputs(
    cad_artifact_id: int,
    model_id: int,
    overwrite_keys: bool = False
) -> Dict[str, Any]:
    """
    confirmed member_spec_links를 읽어서 member_design_inputs에 b/h/As_top/As_bot/Av/s 등을 채움
    """
    conn = _connect()
    try:
        links = conn.execute(
            """
            SELECT l.model_member_id, l.spec_id, s.spec_kind, s.spec_json, s.raw_text, l.distance
            FROM member_spec_links l
            JOIN cad_specs s ON s.spec_id = l.spec_id
            WHERE l.cad_artifact_id=? AND l.model_id=? AND l.status='confirmed'
            ORDER BY l.model_member_id ASC
            """,
            (int(cad_artifact_id), int(model_id)),
        ).fetchall()

        # group by member
        by_member: Dict[int, List[sqlite3.Row]] = {}
        for r in links:
            by_member.setdefault(int(r["model_member_id"]), []).append(r)

        applied = 0
        details = []

        for mmid, rows in by_member.items():
            row = conn.execute(
                "SELECT design_json FROM member_design_inputs WHERE model_member_id=?",
                (int(mmid),),
            ).fetchone()
            dj = json.loads(row["design_json"]) if row else {}

            patch: Dict[str, Any] = {}
            sources: List[Dict[str, Any]] = dj.get("spec_sources", [])

            for r in rows:
                spec = json.loads(r["spec_json"] or "{}")
                kind = r["spec_kind"]

                # RC section
                if kind == "rc_rect_section":
                    b = spec.get("b_mm")
                    h = spec.get("h_mm")
                    if b and h:
                        patch.setdefault("b", b)
                        patch.setdefault("h", h)

                # Steel H
                if kind == "steel_h_section":
                    patch.setdefault("Fy", dj.get("Fy"))  # Fy는 보통 재료에서 오니까 defaults로 처리 권장
                    patch.setdefault("Zx", dj.get("Zx"))
                    patch.setdefault("Aw", dj.get("Aw"))
                    # 실제로는 H치수에서 Zx, Aw를 계산/조회해야 하므로,
                    # 여기서는 "형상 파악"만 하고, 단면 DB를 붙이는 Step 7로 넘김.
                    patch.setdefault("steel_section", spec)

                # Main bars
                if kind == "rebar_main":
                    pos = spec.get("pos", "UNKNOWN")
                    As = spec.get("As_mm2")
                    if As:
                        if pos == "TOP":
                            patch["As_top"] = float(As)
                        elif pos == "BOT":
                            patch["As_bot"] = float(As)
                        else:
                            # UNKNOWN은 As로 보관(또는 둘 다 비었을 때만 채우기)
                            patch.setdefault("As", float(As))

                # Stirrups
                if kind == "rebar_stirrup":
                    patch.setdefault("Av", float(spec.get("Av_mm2")) if spec.get("Av_mm2") else None)
                    patch.setdefault("s", float(spec.get("s_mm")) if spec.get("s_mm") else None)

                sources.append({
                    "spec_id": int(r["spec_id"]),
                    "spec_kind": kind,
                    "raw_text": r["raw_text"],
                    "distance": float(r["distance"]) if r["distance"] is not None else None,
                })

            # overwrite 정책
            merged = dict(dj)
            for k, v in patch.items():
                if v is None:
                    continue
                if (not overwrite_keys) and (k in merged and merged[k] not in (None, "", 0)):
                    continue
                merged[k] = v

            merged["spec_sources"] = sources
            merged.setdefault("units", {"length": "mm", "stress": "MPa"})  # 최소 단위 힌트

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
            applied += 1
            details.append({"model_member_id": mmid, "patch_keys": list(patch.keys())})

        conn.commit()
        return {"ok": True, "members_applied": applied, "details": details[:200]}
    finally:
        conn.close()

6-5) 체크 엔진 업그레이드: ratio_expr 지원(필수)

Mn_pos / Mn_neg 같은 “부호별 내력”을 다루려면 기존 demand/capacity 1쌍으로는 부족해.
그래서 룰셋에 ratio_expr를 추가 지원하게 만들어야 해.

6-5-1) safe_eval 허용 함수에 이미 abs/max/min 있음 → OK

(Step 4에서 abs/max/min/round 허용했던 구조 그대로면 된다)

6-5-2) structai_check_run 내부 로직 변경 포인트

기존:

demand_expr / capacity_expr 계산 → ratio = demand/capacity

변경:

ratio_expr가 있으면 ratio를 직접 계산

demand/capacity는 보조 필드로만(있으면 계산)

의사 코드:

ratio_expr = cd.get("ratio_expr")
if ratio_expr:
    vars_ = {**numeric_env, **numeric_design}
    ratio = safe_eval(ratio_expr, vars_)
    status = _status_from_ratio(ratio, limit, warn)
else:
    # 기존 방식 유지

6-6) RC 보 내력 계산도 업그레이드(As_top/As_bot → Mn_pos/Mn_neg)

Step 5에서 structai_design_compute_rc_beam_rect가 As만 읽었다면, 다음처럼 확장해줘:

As_bot 있으면 Mn_pos 계산(양의 휨)

As_top 있으면 Mn_neg 계산(음의 휨)

둘 다 있으면 Mn = min(Mn_pos, Mn_neg)로 “단일 Mn”도 유지(기존 룰셋 호환)

즉, 저장되는 design_json 예:

{
  "b": 400, "h": 600,
  "As_top": 1963.5,
  "As_bot": 1472.6,
  "Mn_pos": 145.2,
  "Mn_neg": 132.7,
  "Vn": 81.4
}

6-7) 룰셋 예시: 휨 체크를 ratio_expr로

예시(빌트인 generic v0.2 같은 느낌):

"strength.flexure": {
  "ratio_expr": "max(abs(M3_max)/Mn_pos, abs(M3_min)/Mn_neg)",
  "limit": 1.0,
  "warn": 0.95,
  "citations": [
    { "query": "KDS 휨강도 설계", "kind": "pdf", "note": "조항번호로 교체 권장" }
  ]
}


이렇게 하면 “상부근/하부근이 다른 보”도 자동으로 설계비를 잡아낼 수 있어.

6-8) 입력 스키마/검증(Validation) 도구 추가

목표:

“체크 실행 전에” 누락을 잡아내고,

“어떤 체크가 NA가 될지” 미리 설명 가능하게 만들기

6-8-1) 룰셋 기반 자동 요구 변수 추출

check 정의의 demand_expr/capacity_expr/ratio_expr에서 변수명을 AST로 추출

envelope에 있어야 할 변수/ design_json에 있어야 할 변수 분리

누락이면 “이 멤버는 이 체크가 NA가 된다”라고 정확히 리포트

이건 이후에 Claude-like 대화형 플로우에서 엄청 중요해져.
(“지금 B12는 상부근 정보가 없어서 음의 휨 체크가 NA입니다. 도면에서 상부근 표기(4-D25)를 확정해 주세요.” 같은 대화가 가능)

6-8-2) 구현 도구 (핵심만)
import ast

def extract_names(expr: str) -> List[str]:
    if not expr:
        return []
    tree = ast.parse(expr, mode="eval")
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
    # abs/max/min/round 같은 함수명은 제외
    return [n for n in names if n not in ("abs","max","min","round")]


@mcp.tool()
def structai_validate_ready_for_checks(
    model_id: int,
    analysis_run_id: int,
    check_types: Optional[List[str]] = None,
    only_mapped_from_cad_artifact_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    활성 룰셋 기준으로:
    - 결과(env) 변수 누락
    - 설계입력(design_json) 변수 누락
    - 결과 자체 누락
    을 멤버별로 리포트
    """
    conn = _connect()
    try:
        rulepack = _get_active_rulepack(conn)
        checks_def = rulepack.get("checks") or {}
        if not check_types:
            check_types = list(checks_def.keys())

        # scope members
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
                "checks": []
            }

            any_problem = False

            for ct in check_types:
                cd = checks_def.get(ct) or {}
                ratio_expr = cd.get("ratio_expr")
                demand_expr = cd.get("demand_expr")
                capacity_expr = cd.get("capacity_expr")

                need = set()
                if ratio_expr:
                    need |= set(extract_names(str(ratio_expr)))
                if demand_expr:
                    need |= set(extract_names(str(demand_expr)))
                if capacity_expr:
                    need |= set(extract_names(str(capacity_expr)))

                missing = []
                for name in need:
                    if (name not in env_keys) and (name not in design_keys):
                        missing.append(name)

                # 더 친절하게: env/design 중 어디가 비었는지 분해
                missing_env = []
                missing_design = []
                if ratio_expr:
                    for n in extract_names(str(ratio_expr)):
                        if n in design_keys:
                            continue
                        if n in env_keys:
                            continue
                        # heuristic: 대문자 수요키(M3_max 등)는 env일 가능성이 높음
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

6-9) KDS 조항 매핑(Codebook) → 룰셋 citations 자동화

여기서 핵심은:

조항번호/키워드는 사람이 정하는 게 정확함

하지만 “룰셋의 citations 배열을 매번 손으로 쓰는 작업”은 자동화해야 함

6-9-1) Codebook JSON 포맷(추천)

예: codebook_kds_rc.json

{
  "name": "kds-rc",
  "version": "2025.01",
  "citations": {
    "strength.flexure": [
      { "clause_id": "KDS ... 8.3", "query": "KDS 8.3 휨 강도", "kind": "pdf" }
    ],
    "strength.shear": [
      { "clause_id": "KDS ... 8.4", "query": "KDS 8.4 전단 강도", "kind": "pdf" }
    ],
    "service.deflection": [
      { "clause_id": "KDS ... 7.2", "query": "처짐 허용", "kind": "pdf" }
    ]
  }
}


query는 PDF 텍스트에서 실제로 검색되는 문자열로 잡는 게 중요

스캔 PDF라면 pypdf 추출이 빈약해서 검색이 안 될 수 있음 → 그 경우 OCR 단계(후속 Step) 필요

6-9-2) codebook 테이블 임포트/활성화 도구
@mcp.tool()
def structai_codebook_import(path: str) -> Dict[str, Any]:
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
        return {"items": [dict(r) for r in rows]}
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

6-9-3) “활성 codebook → 새로운 rulepack 생성” 도구
@mcp.tool()
def structai_rules_generate_from_active_codebook(
    new_rulepack_name: str,
    new_rulepack_version: str,
    base: str = "active",     # 'active' or 'builtin'
    activate: bool = False
) -> Dict[str, Any]:
    """
    활성 codebook의 citations를 base rulepack에 덮어써서 새 rulepack 생성
    """
    conn = _connect()
    try:
        # load base rulepack
        if base == "active":
            base_rp = _get_active_rulepack(conn)
        else:
            base_rp = BUILTIN_RULEPACK

        # load active codebook
        cb_row = conn.execute(
            "SELECT codebook_json FROM codebooks WHERE is_active=1 ORDER BY codebook_id DESC LIMIT 1"
        ).fetchone()
        if not cb_row:
            raise ValueError("No active codebook. Use structai_codebook_set_active first.")
        cb = json.loads(cb_row["codebook_json"])
        cite_map = cb.get("citations") or {}

        rp = json.loads(json.dumps(base_rp))  # deep copy
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
            # newly inserted id
            rp_id = conn.execute("SELECT rulepack_id FROM rulepacks WHERE name=? AND version=?", (new_rulepack_name, new_rulepack_version)).fetchone()
            if rp_id:
                conn.execute("UPDATE rulepacks SET is_active=0")
                conn.execute("UPDATE rulepacks SET is_active=1 WHERE rulepack_id=?", (int(rp_id["rulepack_id"]),))

        conn.commit()
        return {"ok": True, "changed_checks": changed, "new_rulepack": {"name": new_rulepack_name, "version": new_rulepack_version}, "activated": activate}
    finally:
        conn.close()

6-9-4) (강력 추천) citations 실제 검색 성공 여부 점검 도구
@mcp.tool()
def structai_rules_test_citations(limit_each: int = 1) -> Dict[str, Any]:
    """
    활성 rulepack의 citations.query가 실제 PDF/MD에서 검색되는지 테스트
    """
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
                kind = c.get("kind")

                # reuse existing citation search helper
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

6-10) “지금 단계” 실제 사용 순서(한 번에 굴러가게)

DXF + 모델 매핑까지 완료(Step 3)

CAD 스펙 파싱:

#structai_cad_parse_specs { "cad_artifact_id": 1, "overwrite": true }


파싱된 스펙 검수(샘플로 200개):

#structai_cad_list_specs { "cad_artifact_id": 1, "limit": 200 }


매핑된 부재에 스펙 링크 추천:

#structai_specs_suggest_links { "cad_artifact_id": 1, "model_id": 1, "max_dist": 500.0 }


일괄 확정(초기엔 이렇게 빠르게):

#structai_specs_confirm_all { "cad_artifact_id": 1, "model_id": 1 }


design_inputs 채우기:

#structai_design_apply_specs_to_inputs { "cad_artifact_id": 1, "model_id": 1, "overwrite_keys": false }


RC 보 내력 계산(As_top/bot 반영 버전으로):

#structai_design_compute_rc_beam_rect {
  "model_id": 1,
  "where": { "type": "beam" },
  "defaults": { "fc": 27, "fy": 400, "phi_flex": 0.9, "phi_shear": 0.75 },
  "overwrite": true
}


체크 전 검증:

#structai_validate_ready_for_checks { "model_id": 1, "analysis_run_id": 1 }


(선택) Codebook → 룰셋 citations 자동화:

#structai_codebook_import { "path": "…/codebook_kds_rc.json" }
#structai_codebook_set_active { "codebook_id": 1 }
#structai_rules_generate_from_active_codebook {
  "new_rulepack_name": "kds-rc-auto",
  "new_rulepack_version": "2025.01",
  "base": "active",
  "activate": true
}
#structai_rules_test_citations { "limit_each": 1 }

6-11) 다음 Step 7을 위한 “정답 포인트”(미리 연결)

Step 6까지 하면 “도면에서 철근/치수 읽어서 As/Av/s까지 채우고 체크”는 MVP가 된다.
하지만 실무에서 가장 강력해지는 지점은 Step 7이야:

단면 DB(강재 형강/RC 단면 라이브러리) 연결

H400x200x8x13 → 자동으로 Zx, Aw, Iy 등을 계산/조회

표(table) 인식 강화

도면은 “부재표/철근표” 형태가 많으니
단순 근접거리만으로는 오배정이 생김

해결: 블록/테이블 영역 추정 + 같은 행/열 룰 + 리더라인(MLEADER) 따라가기

모델(section) 문자열 파싱 + CAD 파싱을 결합

CAD에 철근이 없고 모델에 있는 프로젝트도 많음 → 둘 다 사용

원하면 다음 응답에서 **Step 7: “단면 라이브러리 + 테이블/리더 기반 정밀 배정 + 형강 단면특성 자동 계산”**을 같은 방식으로 순서대로 구현해줄게.