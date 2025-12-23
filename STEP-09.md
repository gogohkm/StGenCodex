좋아. 이제 **9단계(단면 라이브러리 확장 + 테이블 인식 고도화 + Regression 검증 하네스)**를 바로 구현 가능한 형태로 순서대로 정리해서 제공할게.

⚠️ (중요) 구조 안전/설계 책임 고지
아래는 “자동화/검증/감사 가능”한 도구를 만드는 설계이며, 결정론 계산은 코드/조항 검증이 완료되기 전까지는 참고용이야.
특히 **비대칭 단면(앵글/채널)**의 Z(소성단면계수)·좌굴·세부조항은 표준 카탈로그(공칭값) 우선으로 가야 안전하다.
그래서 Step9의 핵심은 (1) 카탈로그 우선 적용, (2) 계산은 fallback, (3) 자동 검증(Regression)으로 품질 유지야.

9-0) 이번 단계에서 완성되는 것

형강 라이브러리 확장(실무형)

steel_h(H형강)

steel_box(각형강관 RHS/SHS)

steel_pipe(원형강관 CHS/PIPE)

steel_channel(채널)

steel_angle(앵글)

표준 단면 우선 적용

ks_catalog/csv_import 같은 표준 소스는 priority가 높게(숫자 낮게)

computed는 priority 낮게(숫자 크게) → 표준값이 있으면 자동으로 덮어씀

테이블 인식 고도화

헤더 다중행(예: “철근” 아래 “상부/하부”) 결합

헤더 없이도 통계 기반으로 컬럼 역할 추정 강화

(보수적) 빈 셀/병합셀 처리 옵션 제공

Regression 검증 하네스

“샘플 프로젝트(소형)” 기준으로 결과 요약(golden) 저장

매 커밋마다 자동 비교(FAIL/WARN/NA 카운트 + worst ratio)

도구 변경 시 품질이 깨지면 바로 감지

9-1) DB 스키마 추가 (v0.1.0)

mcp_server/schema.sql 맨 아래에 추가:

-- ===== schema patch v0.1.0 =====
UPDATE meta SET value='0.1.0' WHERE key='schema_version';

-- Regression harness tables
CREATE TABLE IF NOT EXISTS regression_suites (
  suite_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS regression_cases (
  case_id INTEGER PRIMARY KEY AUTOINCREMENT,
  suite_id INTEGER NOT NULL REFERENCES regression_suites(suite_id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  fixture_json TEXT NOT NULL,         -- how to build/run
  golden_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(suite_id, name)
);

CREATE TABLE IF NOT EXISTS regression_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  suite_id INTEGER NOT NULL REFERENCES regression_suites(suite_id) ON DELETE CASCADE,
  started_at TEXT NOT NULL DEFAULT (datetime('now')),
  finished_at TEXT,
  status TEXT NOT NULL DEFAULT 'RUNNING', -- RUNNING|PASS|FAIL|ERROR
  report_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS regression_case_results (
  case_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES regression_runs(run_id) ON DELETE CASCADE,
  case_id INTEGER NOT NULL REFERENCES regression_cases(case_id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  diff_json TEXT NOT NULL DEFAULT '{}'
);


참고: Step8에서 section_catalog.priority 컬럼을 migration 도구로 추가했지.
새 DB를 스키마로 생성하는 경우에는 schema.sql에 priority까지 포함된 최신 버전으로 통합해두는 게 좋아.

9-2) 형강 단면 라이브러리 확장
9-2-1) 철골/강관/채널/앵글 파싱 추가 (parsing/specs.py 패치)

mcp_server/parsing/specs.py에 아래 Regex들을 추가하고, parse_specs_from_text()에서 steel_h 다음 우선순위로 매칭되게 넣어줘.

# --- add to mcp_server/parsing/specs.py ---
STEEL_BOX_RX = re.compile(
    r"(?:\b|^)\s*(?:RHS|SHS|BOX|□)\s*[-]?\s*(?P<H>\d{2,4}(?:\.\d+)?)\s*([xX×*])\s*(?P<B>\d{2,4}(?:\.\d+)?)\s*([xX×*])\s*(?P<t>\d{1,3}(?:\.\d+)?)(?:t|T)?(?:\b|$)",
    re.IGNORECASE
)

STEEL_PIPE_RX = re.compile(
    r"(?:\b|^)\s*(?:CHS|PIPE|P|Ø|∅)\s*[-]?\s*(?P<D>\d{2,4}(?:\.\d+)?)\s*([xX×*])\s*(?P<t>\d{1,3}(?:\.\d+)?)(?:t|T)?(?:\b|$)",
    re.IGNORECASE
)

# 채널은 RC "C400x600"과 충돌을 피하려고 "C-" 또는 "CHANNEL"을 요구
STEEL_CHANNEL_RX = re.compile(
    r"(?:\b|^)\s*(?:CHANNEL|CH|C)\s*[-]\s*(?P<H>\d{2,4}(?:\.\d+)?)\s*([xX×*])\s*(?P<B>\d{2,4}(?:\.\d+)?)\s*([xX×*])\s*(?P<tw>\d{1,3}(?:\.\d+)?)\s*([xX×*])\s*(?P<tf>\d{1,3}(?:\.\d+)?)(?:\b|$)",
    re.IGNORECASE
)

STEEL_ANGLE_RX = re.compile(
    r"(?:\b|^)\s*L\s*[-]?\s*(?P<b>\d{2,4}(?:\.\d+)?)\s*([xX×*])\s*(?P<d>\d{2,4}(?:\.\d+)?)\s*([xX×*])\s*(?P<t>\d{1,3}(?:\.\d+)?)(?:\b|$)",
    re.IGNORECASE
)


그리고 parse_specs_from_text() 안에 아래를 steel_h → rc_rect보다 먼저 넣는 걸 추천(단면이 RC로 오인식되지 않도록):

# 1.1) Steel BOX
m = STEEL_BOX_RX.search(t)
if m:
    H = float(m.group("H")); B = float(m.group("B")); tt = float(m.group("t"))
    specs.append({
        "spec_kind":"steel_box_section",
        "H_mm": H, "B_mm": B, "t_mm": tt,
        "confidence": 0.9,
        "raw_fragment": m.group(0).strip(),
    })

# 1.2) Steel PIPE
m = STEEL_PIPE_RX.search(t)
if m:
    D = float(m.group("D")); tt = float(m.group("t"))
    # rebar(D10)와 충돌을 피하려고 D가 너무 작으면 무시
    if D >= 30:
        specs.append({
            "spec_kind":"steel_pipe_section",
            "D_mm": D, "t_mm": tt,
            "confidence": 0.9,
            "raw_fragment": m.group(0).strip(),
        })

# 1.3) Steel CHANNEL
m = STEEL_CHANNEL_RX.search(t)
if m:
    H = float(m.group("H")); B = float(m.group("B"))
    tw = float(m.group("tw")); tf = float(m.group("tf"))
    specs.append({
        "spec_kind":"steel_channel_section",
        "H_mm": H, "B_mm": B, "tw_mm": tw, "tf_mm": tf,
        "confidence": 0.85,
        "raw_fragment": m.group(0).strip(),
    })

# 1.4) Steel ANGLE
m = STEEL_ANGLE_RX.search(t)
if m:
    b = float(m.group("b")); d = float(m.group("d")); tt = float(m.group("t"))
    specs.append({
        "spec_kind":"steel_angle_section",
        "b_mm": b, "d_mm": d, "t_mm": tt,
        "confidence": 0.8,
        "raw_fragment": m.group(0).strip(),
    })

9-2-2) 단면특성 계산 모듈 추가 (design/steel_props_more.py)

새 파일 mcp_server/design/steel_props_more.py:

# mcp_server/design/steel_props_more.py
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any, Dict


@dataclass
class SteelBoxDims:
    H: float
    B: float
    t: float

def compute_box_props(d: SteelBoxDims) -> Dict[str, Any]:
    H, B, t = float(d.H), float(d.B), float(d.t)
    if H <= 0 or B <= 0 or t <= 0:
        raise ValueError("H,B,t must be positive.")
    Hi = H - 2*t
    Bi = B - 2*t
    if Hi <= 0 or Bi <= 0:
        raise ValueError("Invalid box dims: H-2t and B-2t must be positive.")

    A = B*H - Bi*Hi
    Ix = (B*H**3 - Bi*Hi**3) / 12.0
    Iy = (H*B**3 - Hi*Bi**3) / 12.0
    Sx = Ix / (H/2.0)
    Sy = Iy / (B/2.0)

    Zx = (B*H**2 - Bi*Hi**2) / 4.0
    Zy = (H*B**2 - Hi*Bi**2) / 4.0

    Avx = 2.0 * t * Hi  # shear area for Vx (major axis shear)
    Avy = 2.0 * t * Bi  # shear area for Vy

    return {
        "family": "steel_box",
        "dims": {"H_mm": H, "B_mm": B, "t_mm": t},
        "props": {
            "A_mm2": A,
            "Ix_mm4": Ix,
            "Iy_mm4": Iy,
            "Sx_mm3": Sx,
            "Sy_mm3": Sy,
            "Zx_mm3": Zx,
            "Zy_mm3": Zy,
            "Aw_mm2": Avx,
            "Avx_mm2": Avx,
            "Avy_mm2": Avy,
        },
    }


@dataclass
class SteelPipeDims:
    D: float
    t: float

def compute_pipe_props(d: SteelPipeDims) -> Dict[str, Any]:
    D, t = float(d.D), float(d.t)
    if D <= 0 or t <= 0:
        raise ValueError("D,t must be positive.")
    Di = D - 2*t
    if Di <= 0:
        raise ValueError("Invalid pipe dims: D-2t must be positive.")

    Ro = D/2.0
    Ri = Di/2.0

    A = (pi/4.0) * (D**2 - Di**2)
    Ix = (pi/64.0) * (D**4 - Di**4)
    Iy = Ix
    Sx = Ix / (D/2.0)
    Sy = Sx

    # plastic modulus (exact; no pi)
    Zx = (4.0/3.0) * (Ro**3 - Ri**3)
    Zy = Zx

    # circular: use area as shear area (common conservative fallback)
    Aw = A

    return {
        "family": "steel_pipe",
        "dims": {"D_mm": D, "t_mm": t},
        "props": {
            "A_mm2": A,
            "Ix_mm4": Ix,
            "Iy_mm4": Iy,
            "Sx_mm3": Sx,
            "Sy_mm3": Sy,
            "Zx_mm3": Zx,
            "Zy_mm3": Zy,
            "Aw_mm2": Aw,
        },
    }


@dataclass
class SteelChannelDims:
    H: float
    B: float
    tw: float
    tf: float

def compute_channel_props(d: SteelChannelDims) -> Dict[str, Any]:
    """
    Channel modeled as 3 rectangles (no overlap):
      - web: tw x (H-2tf)
      - two flanges: B x tf (top & bottom)
    Coordinate:
      x from web back face (0..B), y from bottom (0..H)
    """
    H, B, tw, tf = float(d.H), float(d.B), float(d.tw), float(d.tf)
    if H <= 0 or B <= 0 or tw <= 0 or tf <= 0:
        raise ValueError("H,B,tw,tf must be positive.")
    hw = H - 2.0*tf
    if hw <= 0:
        raise ValueError("Invalid dims: H-2tf must be positive.")
    if tw >= B:
        raise ValueError("Invalid dims: tw must be < B (typical).")

    Aweb = tw*hw
    Af = B*tf
    A = Aweb + 2.0*Af

    # centroids
    x_web = tw/2.0
    y_web = H/2.0
    x_f = B/2.0
    y_top = H - tf/2.0
    y_bot = tf/2.0

    xbar = (Aweb*x_web + Af*x_f + Af*x_f) / A
    ybar = H/2.0  # symmetric about x-axis

    # Ix about centroidal x-axis
    Ix_web = (tw*hw**3)/12.0  # centered at ybar already
    y_f = abs(y_top - ybar)
    Ix_f_each = (B*tf**3)/12.0 + Af*(y_f**2)
    Ix = Ix_web + 2.0*Ix_f_each

    # Iy about centroidal y-axis (unsymmetric)
    Iy_web = (hw*tw**3)/12.0 + Aweb*((x_web - xbar)**2)
    Iy_f_each = (tf*B**3)/12.0 + Af*((x_f - xbar)**2)
    Iy = Iy_web + 2.0*Iy_f_each

    # elastic section modulus
    Sx = Ix / (H/2.0)
    # for y-axis, extreme distances are different left/right
    c_left = xbar
    c_right = B - xbar
    Sy_left = Iy / c_left if c_left > 0 else None
    Sy_right = Iy / c_right if c_right > 0 else None
    Sy_min = min([v for v in (Sy_left, Sy_right) if v is not None])

    # plastic modulus about x-axis (exact due to symmetry about x)
    # top half: top flange + top half of web
    Aweb_top = tw*(hw/2.0)
    y_w = hw/4.0
    y_f_pl = (H/2.0 - tf/2.0)
    Zx = 2.0 * (Af*y_f_pl + Aweb_top*y_w)

    Aw = tw*hw  # web area

    return {
        "family": "steel_channel",
        "dims": {"H_mm": H, "B_mm": B, "tw_mm": tw, "tf_mm": tf},
        "props": {
            "A_mm2": A,
            "Ix_mm4": Ix,
            "Iy_mm4": Iy,
            "Sx_mm3": Sx,
            "Sy_min_mm3": Sy_min,
            "Zx_mm3": Zx,
            "Aw_mm2": Aw,
            "centroid_x_mm": xbar,
            "centroid_y_mm": ybar,
        },
        "warnings": [
            "Channel is unsymmetric about y-axis; use catalog values if available for Sy/Zy and design checks about minor/principal axes."
        ],
    }


@dataclass
class SteelAngleDims:
    b: float
    d: float
    t: float

def compute_angle_props(d: SteelAngleDims) -> Dict[str, Any]:
    """
    L-angle as union of two rectangles minus overlap square.
    No plastic modulus exact (PNA not at centroid). Provide conservative Zx= Sx_min if needed elsewhere.
    """
    b, dd, t = float(d.b), float(d.d), float(d.t)
    if b <= 0 or dd <= 0 or t <= 0:
        raise ValueError("b,d,t must be positive.")
    if t >= b or t >= dd:
        raise ValueError("Invalid angle dims: t must be < b and < d.")

    A1 = b*t
    x1, y1 = b/2.0, t/2.0
    A2 = t*dd
    x2, y2 = t/2.0, dd/2.0
    A3 = t*t
    x3, y3 = t/2.0, t/2.0

    A = A1 + A2 - A3
    xbar = (A1*x1 + A2*x2 - A3*x3) / A
    ybar = (A1*y1 + A2*y2 - A3*y3) / A

    # rectangle centroidal moments
    def Ix_rect(w, h):  # about x-axis through rect centroid
        return w*h**3 / 12.0
    def Iy_rect(w, h):
        return h*w**3 / 12.0

    # Ix about global centroid
    Ix = (Ix_rect(b, t) + A1*(y1 - ybar)**2) + (Ix_rect(t, dd) + A2*(y2 - ybar)**2) - (Ix_rect(t, t) + A3*(y3 - ybar)**2)
    Iy = (Iy_rect(b, t) + A1*(x1 - xbar)**2) + (Iy_rect(t, dd) + A2*(x2 - xbar)**2) - (Iy_rect(t, t) + A3*(x3 - xbar)**2)

    # elastic section modulus (min of two sides)
    c_top = dd - ybar
    c_bot = ybar
    Sx_top = Ix / c_top if c_top > 0 else None
    Sx_bot = Ix / c_bot if c_bot > 0 else None
    Sx_min = min([v for v in (Sx_top, Sx_bot) if v is not None])

    c_right = b - xbar
    c_left = xbar
    Sy_right = Iy / c_right if c_right > 0 else None
    Sy_left = Iy / c_left if c_left > 0 else None
    Sy_min = min([v for v in (Sy_right, Sy_left) if v is not None])

    # conservative fallback: Zx = Sx_min (plastic >= elastic)
    Zx_cons = Sx_min

    return {
        "family": "steel_angle",
        "dims": {"b_mm": b, "d_mm": dd, "t_mm": t},
        "props": {
            "A_mm2": A,
            "Ix_mm4": Ix,
            "Iy_mm4": Iy,
            "Sx_min_mm3": Sx_min,
            "Sy_min_mm3": Sy_min,
            "Zx_mm3": Zx_cons,   # conservative fallback
            "Aw_mm2": A,         # axial/shear fallback
            "centroid_x_mm": xbar,
            "centroid_y_mm": ybar,
        },
        "warnings": [
            "Angle plastic modulus is not computed exactly; Zx is set conservatively to Sx_min. Prefer catalog values for design."
        ],
    }

9-2-3) section_catalog 임포트/계산 로직 확장

Step7의 structai_sections_import_catalog에서 family별로 props 계산을 확장해줘:

steel_h → 기존 compute_h_section_props

steel_box → compute_box_props

steel_pipe → compute_pipe_props

steel_channel → compute_channel_props

steel_angle → compute_angle_props (단, 표준값이 있으면 그걸 우선)

예시 패치(핵심만):

from mcp_server.design.steel_props import SteelHSectionDims, compute_h_section_props
from mcp_server.design.steel_props_more import (
  SteelBoxDims, compute_box_props,
  SteelPipeDims, compute_pipe_props,
  SteelChannelDims, compute_channel_props,
  SteelAngleDims, compute_angle_props
)

if family == "steel_box" and ("Zx_mm3" not in props or "Aw_mm2" not in props):
    calc = compute_box_props(SteelBoxDims(H=dims["H_mm"], B=dims["B_mm"], t=dims["t_mm"]))
    props = {**calc["props"], **props}

# ... 동일하게 pipe/channel/angle


그리고 _upsert_section()은 Step8에서 만든 priority 비교 업데이트 버전을 반드시 사용해줘.

ks_catalog/csv_import → priority 10~20

computed → priority 80

9-2-4) “모델 section 문자열” resolve 확장

Step7의 structai_sections_resolve_members도 parse_specs_from_text()의 새 spec_kind를 읽어
family를 맞춰 resolve하도록 확장하면 된다.

권장 매핑:

spec_kind	section_catalog.family
steel_h_section	steel_h
steel_box_section	steel_box
steel_pipe_section	steel_pipe
steel_channel_section	steel_channel
steel_angle_section	steel_angle

그리고 design_inputs에 최소한 아래는 채우도록:

Zx (mm3) → 휨강도 계산용

Aw (mm2) → 전단강도 계산용

Ix (mm4) → 처짐/강성 체크용(추후)

A (mm2) → 축력 체크용(추후)

안전하게 가려면:
“angle/channel의 Zx가 catalog가 아니라 fallback일 때”는 design_json에 warnings를 기록하고,
validate_ready_for_checks에서 WARN을 띄우는 게 좋다.

9-3) 테이블 인식 고도화 v2 (다중 헤더/병합셀 대응)

Step8의 infer_schema/parse_rows는 “헤더 1줄” 가정이 강했어.
여기서 “실무 부재표”는 아래가 많아:

헤더 2줄 이상:
1행: 철근 / 2행: 상부/하부

헤더 없이 데이터만 있는 표

story가 token과 같은 셀에 들어감: B12(3F)

그래서 v2를 별도 파일로 두고 교체하는 걸 추천해.

9-3-1) 새 파일: mcp_server/parsing/table_schema_v2.py

핵심 아이디어:

헤더 후보 상위 2~3줄을 결합하여 “컬럼 역할”을 더 정확히 추정

token/story는 셀 내부에서도 추출

(길어질 수 있으니 핵심 함수만 제공)

# mcp_server/parsing/table_schema_v2.py
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

from .story import normalize_story

TOKEN_RX = re.compile(r"\b[A-Z]{1,4}\s?-?\s?\d{1,4}\b", re.IGNORECASE)

KW = {
  "token": ["부재","기호","부재명","부재번호","MARK","MEMBER","NAME","ID","NO"],
  "story": ["층","STORY","FLOOR","LEVEL","LV"],
  "section": ["단면","규격","SECTION","SIZE","형강","H형강","STEEL","RHS","SHS","PIPE","CHS","BOX"],
  "rebar": ["철근","REBAR","주근"],
  "top": ["상부","TOP","UPPER"],
  "bot": ["하부","BOT","BOTTOM","LOWER"],
  "stir": ["띠","스터럽","STIR","@"],
}

def _has_any(s: str, kws: List[str]) -> bool:
    u = (s or "").upper()
    return any(k.upper() in u for k in kws)

def role_from_header(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    if _has_any(t, KW["token"]): return "token"
    if _has_any(t, KW["story"]): return "story"
    if _has_any(t, KW["section"]): return "section"
    if _has_any(t, KW["rebar"]) and _has_any(t, KW["top"]): return "rebar_top"
    if _has_any(t, KW["rebar"]) and _has_any(t, KW["bot"]): return "rebar_bot"
    if _has_any(t, KW["top"]): return "rebar_top"
    if _has_any(t, KW["bot"]): return "rebar_bot"
    if _has_any(t, KW["stir"]): return "stirrup"
    if _has_any(t, KW["rebar"]): return "rebar"
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

    # score rows as header candidates
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
        # allow multi-row header: include next row if it has header-like tokens
        if (best_r + 1) in by_row:
            s2 = 0
            for c in by_row[best_r + 1]:
                if role_from_header(str(c.get("text") or "")) or _has_any(str(c.get("text") or ""), KW["top"]+KW["bot"]):
                    s2 += 1
            if s2 >= 1:
                header_rows.append(best_r + 1)

    # build combined header text per column
    col_text: Dict[int, List[str]] = {}
    for hr in header_rows:
        for c in by_row.get(hr, []):
            col = int(c["col_idx"])
            col_text.setdefault(col, [])
            txt = str(c.get("text") or "").strip()
            if txt:
                col_text[col].append(txt)

    columns: Dict[str, int] = {}
    # assign roles from combined header strings
    for col, parts in col_text.items():
        combined = " ".join(parts)
        role = role_from_header(combined)
        if role and role not in columns:
            columns[role] = col

    confidence = 0.35
    if columns:
        confidence = min(0.95, 0.6 + 0.1 * len(columns))

    # fallback: content-based inference if headers weak
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
            u = txt.upper().replace("×","X")
            if ("H-" in u) or ("RHS" in u) or ("SHS" in u) or ("PIPE" in u) or ("CHS" in u) or ("□" in txt) or ("Ø" in txt) or ("∅" in txt):
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

    return {"header_rows": header_rows, "columns": columns, "confidence": confidence, "debug": {"row_scores": row_scores[:10]}}

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
        r = int(c["row_idx"]); col = int(c["col_idx"])
        by_row.setdefault(r, {})
        by_row[r][col] = dict(c)

    out = []
    for r, rowmap in by_row.items():
        if r in header_rows:
            continue

        def pick(role: str) -> Optional[Dict[str, Any]]:
            if role not in cols: return None
            cc = rowmap.get(int(cols[role]))
            if not cc: return None
            txt = str(cc.get("text") or "").strip()
            if not txt: return None
            return {"text": txt, "cad_entity_id": cc.get("cad_entity_id"), "x": cc.get("x"), "y": cc.get("y")}

        token_cell = pick("token")
        story_cell = pick("story")

        token_norm = None
        story_norm = None

        # token column에서 token+story 같이 있는 경우도 처리
        if token_cell:
            token_norm, story_norm = _extract_token_and_story(token_cell["text"])

        # story column이 있으면 우선
        if story_cell and story_cell.get("text"):
            story_norm = normalize_story(story_cell["text"]) or story_norm

        fields = {}
        for role in ("token","story","section","rebar","rebar_top","rebar_bot","stirrup"):
            cell = pick(role)
            if cell:
                fields[role] = cell

        conf = 0.4
        if token_norm: conf += 0.25
        if story_norm: conf += 0.1
        if "section" in fields: conf += 0.2
        if any(k in fields for k in ("rebar","rebar_top","rebar_bot","stirrup")): conf += 0.1

        out.append({
            "row_idx": int(r),
            "token_norm": token_norm,
            "story_norm": story_norm,
            "fields": fields,
            "confidence": float(min(0.95, conf)),
        })
    return out

9-3-2) MCP Tool 교체: structai_cad_infer_table_schemas_v2

Step8의 structai_cad_infer_table_schemas를 그대로 두고, v2 도구를 새로 추가하는 게 안전해.

@mcp.tool()
def structai_cad_infer_table_schemas_v2(
    cad_artifact_id: int,
    overwrite: bool = True
) -> Dict[str, Any]:
    from mcp_server.parsing.table_schema_v2 import infer_schema_v2, parse_rows_v2

    conn = _connect()
    try:
        if overwrite:
            conn.execute("DELETE FROM cad_table_schemas WHERE table_id IN (SELECT table_id FROM cad_tables WHERE cad_artifact_id=?)", (int(cad_artifact_id),))
            conn.execute("DELETE FROM cad_table_row_parses WHERE table_id IN (SELECT table_id FROM cad_tables WHERE cad_artifact_id=?)", (int(cad_artifact_id),))

        tables = conn.execute("SELECT table_id, confidence FROM cad_tables WHERE cad_artifact_id=? ORDER BY confidence DESC", (int(cad_artifact_id),)).fetchall()

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
                    sample.append({"table_id": table_id, "row_idx": rp["row_idx"], "token": rp["token_norm"], "story": rp.get("story_norm")})

        conn.commit()
        return {"ok": True, "schemas": schemas, "rows": rows_saved, "sample": sample}
    finally:
        conn.close()

9-4) 다층(층별 동일 라벨) 품질을 “검증 가능한 상태”로 만드는 도구들

Step8에서 token_story_maps를 만들었지. Step9는 이걸 확정(confirmed) 자동화하고, 충돌을 검출한다.

9-4-1) token×story 자동 확정(유일성 기반)
@mcp.tool()
def structai_token_story_auto_confirm(
    cad_artifact_id: int,
    model_id: int,
    min_confidence: float = 0.85
) -> Dict[str, Any]:
    """
    같은 (token, story)에서 model_member_id가 유일하고 confidence>=min이면 confirmed.
    """
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
            tok = r["cad_token_norm"]; st = r["story_norm"]
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
                conn.execute("UPDATE token_story_maps SET status='confirmed', updated_at=datetime('now') WHERE map_id=?", (int(cand[0]["map_id"]),))
                confirmed += 1

        conn.commit()
        return {"ok": True, "confirmed": confirmed}
    finally:
        conn.close()

9-4-2) 충돌 리포트(사람이 봐야 하는 것만 뽑기)
@mcp.tool()
def structai_token_story_conflicts(
    cad_artifact_id: int,
    model_id: int
) -> Dict[str, Any]:
    """
    (token, story) 하나에 member 후보가 2개 이상인 것만 추출
    """
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
            tok = r["cad_token_norm"]; st = r["story_norm"]
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

9-5) “품질 대시보드(coverage)” 도구

이건 VS Code 패널에 붙이면 바로 체감이 생겨.
(“뭘 더 채워야 체크가 돌아가나?”를 한 번에 보여줌)

@mcp.tool()
def structai_quality_summary(
    model_id: int,
    analysis_run_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    - 철골: Zx/Aw 없는 부재 수
    - RC 보: b/h/As_top/As_bot/Av/s 부족
    - (선택) 결과값 누락
    """
    conn = _connect()
    try:
        members = conn.execute(
            "SELECT model_member_id, member_uid, member_label, type, section FROM model_members WHERE model_id=?",
            (int(model_id),),
        ).fetchall()

        steel_missing = []
        rc_missing = []

        for m in members:
            mmid = int(m["model_member_id"])
            di = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
            dj = json.loads(di["design_json"]) if di else {}

            if (m["type"] or "").lower() in ("beam","girder","steel_beam","steel"):
                # steel beam-like
                if not dj.get("Zx") or not dj.get("Aw"):
                    steel_missing.append({"uid": m["member_uid"], "label": m["member_label"], "need": ["Zx","Aw"], "section": m["section"]})

            if (m["type"] or "").lower() in ("beam","rc_beam"):
                need = []
                for k in ("b","h"):
                    if not dj.get(k): need.append(k)
                if not (dj.get("As_top") or dj.get("As")): need.append("As_top(or As)")
                if not (dj.get("As_bot") or dj.get("As")): need.append("As_bot(or As)")
                if not dj.get("Av"): need.append("Av")
                if not dj.get("s"): need.append("s")
                if need:
                    rc_missing.append({"uid": m["member_uid"], "label": m["member_label"], "need": need})

        results_missing = []
        if analysis_run_id is not None:
            for m in members:
                mmid = int(m["model_member_id"])
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

9-6) Regression 검증 하네스(진짜 중요한 부분)

핵심 철학:

기능이 늘어날수록 “조용히 깨지는 케이스”가 생긴다

그래서 샘플 케이스를 고정하고

“체크 결과 요약(golden)”과 “실행 결과(actual)”을 비교해서

깨지면 CI에서 막는다

9-6-1) golden(정답) 포맷 추천

각 케이스는 아래만 맞춰도 충분히 강력해:

{
  "overall": { "PASS": 120, "WARN": 3, "FAIL": 1, "NA": 0 },
  "by_check_type": {
    "strength.flexure": { "PASS": 50, "WARN": 2, "FAIL": 0, "NA": 0, "worst_ratio": 0.98 },
    "strength.shear":   { "PASS": 50, "WARN": 1, "FAIL": 1, "NA": 0, "worst_ratio": 1.05 }
  }
}


ratio는 부동소수 오차가 있으니 tol=1e-3~1e-2로 비교하는 게 일반적.

9-6-2) MCP Tool: regression suite/case 관리 + 실행
(A) suite 생성
@mcp.tool()
def structai_regression_suite_create(name: str, description: str = "") -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute("INSERT OR IGNORE INTO regression_suites(name, description) VALUES(?,?)", (name, description))
        conn.commit()
        row = conn.execute("SELECT suite_id, name, description FROM regression_suites WHERE name=?", (name,)).fetchone()
        return {"ok": True, "suite": dict(row)}
    finally:
        conn.close()

(B) case 추가/업데이트

fixture_json은 “이 케이스를 어떻게 돌릴지”를 담는 실행 스펙이야. 예:

{
  "model_members_path": "fixtures/case01/members.json",
  "results_path": "fixtures/case01/results.json",
  "design_inputs_path": "fixtures/case01/design_inputs.json",
  "rulepack_name": "kds-rc-auto",
  "defaults": { "fc": 27, "fy": 400, "Fy": 325 }
}

@mcp.tool()
def structai_regression_case_upsert(
    suite_name: str,
    case_name: str,
    fixture_json: Dict[str, Any],
    golden_json: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    conn = _connect()
    try:
        s = conn.execute("SELECT suite_id FROM regression_suites WHERE name=?", (suite_name,)).fetchone()
        if not s:
            raise ValueError("suite not found; create suite first")

        sid = int(s["suite_id"])
        gj = golden_json or {}
        conn.execute(
            """
            INSERT INTO regression_cases(suite_id, name, fixture_json, golden_json)
            VALUES(?,?,?,?)
            ON CONFLICT(suite_id, name) DO UPDATE SET
              fixture_json=excluded.fixture_json,
              golden_json=excluded.golden_json
            """,
            (sid, case_name, json.dumps(fixture_json, ensure_ascii=False), json.dumps(gj, ensure_ascii=False)),
        )
        conn.commit()
        row = conn.execute("SELECT case_id, name FROM regression_cases WHERE suite_id=? AND name=?", (sid, case_name)).fetchone()
        return {"ok": True, "case": dict(row)}
    finally:
        conn.close()

(C) check_run_id에서 golden 자동 캡처(실무형)
def _metrics_for_check_run(conn, check_run_id: int) -> Dict[str, Any]:
    overall = {r["status"]: int(r["n"]) for r in conn.execute(
        "SELECT status, COUNT(*) as n FROM check_results WHERE check_run_id=? GROUP BY status",
        (int(check_run_id),),
    ).fetchall()}

    by_check_type = {}
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

@mcp.tool()
def structai_regression_capture_golden(
    suite_name: str,
    case_name: str,
    check_run_id: int
) -> Dict[str, Any]:
    conn = _connect()
    try:
        s = conn.execute("SELECT suite_id FROM regression_suites WHERE name=?", (suite_name,)).fetchone()
        if not s:
            raise ValueError("suite not found")
        sid = int(s["suite_id"])

        golden = _metrics_for_check_run(conn, int(check_run_id))
        conn.execute(
            """
            UPDATE regression_cases
            SET golden_json=?
            WHERE suite_id=? AND name=?
            """,
            (json.dumps(golden, ensure_ascii=False), sid, case_name),
        )
        conn.commit()
        return {"ok": True, "suite": suite_name, "case": case_name, "golden": golden}
    finally:
        conn.close()

(D) regression 실행(케이스마다 실제 파이프라인 실행 후 비교)

여기서 “케이스 실행”은 네 기존 도구를 그대로 호출하면 된다:

import members

import results

parse/apply specs (옵션)

compute capacities

run checks

metrics 추출 → golden 비교

도구 내부에서 그걸 전부 다 구현하면 길어지니, 원칙만 정확히 잡은 실행기 스켈레톤을 제공할게(너의 기존 함수 이름 기준):

def _compare_metrics(golden: Dict[str,Any], actual: Dict[str,Any], ratio_tol: float = 1e-3) -> Dict[str,Any]:
    diff = {"overall": {}, "by_check_type": {}, "ok": True}

    # overall status counts
    g_over = golden.get("overall", {})
    a_over = actual.get("overall", {})
    for k in set(g_over.keys()) | set(a_over.keys()):
        if int(g_over.get(k,0)) != int(a_over.get(k,0)):
            diff["overall"][k] = {"golden": int(g_over.get(k,0)), "actual": int(a_over.get(k,0))}
            diff["ok"] = False

    # per check type
    g_ct = golden.get("by_check_type", {})
    a_ct = actual.get("by_check_type", {})
    for ct in set(g_ct.keys()) | set(a_ct.keys()):
        gd = g_ct.get(ct, {})
        ad = a_ct.get(ct, {})
        cdiff = {}
        # counts
        for k in ("PASS","WARN","FAIL","NA"):
            if int(gd.get(k,0)) != int(ad.get(k,0)):
                cdiff[k] = {"golden": int(gd.get(k,0)), "actual": int(ad.get(k,0))}
        # worst ratio
        gw = gd.get("worst_ratio")
        aw = ad.get("worst_ratio")
        if (gw is not None) and (aw is not None):
            if abs(float(gw) - float(aw)) > float(ratio_tol):
                cdiff["worst_ratio"] = {"golden": float(gw), "actual": float(aw), "tol": float(ratio_tol)}
        elif gw != aw:
            cdiff["worst_ratio"] = {"golden": gw, "actual": aw}

        if cdiff:
            diff["by_check_type"][ct] = cdiff
            diff["ok"] = False

    return diff


그리고 MCP Tool:

@mcp.tool()
def structai_regression_run_suite(
    suite_name: str,
    ratio_tol: float = 1e-3
) -> Dict[str, Any]:
    """
    suite 내 모든 case를 실행하고 golden과 비교
    (주의) 이 도구를 쓰려면 fixture_json에 맞춘 '케이스 실행 함수'를 연결해줘야 함.
    """
    conn = _connect()
    try:
        s = conn.execute("SELECT suite_id FROM regression_suites WHERE name=?", (suite_name,)).fetchone()
        if not s:
            raise ValueError("suite not found")
        sid = int(s["suite_id"])

        run_cur = conn.execute("INSERT INTO regression_runs(suite_id) VALUES(?)", (sid,))
        run_id = int(run_cur.lastrowid)

        cases = conn.execute(
            "SELECT case_id, name, fixture_json, golden_json FROM regression_cases WHERE suite_id=? ORDER BY case_id ASC",
            (sid,),
        ).fetchall()

        all_ok = True
        case_reports = []

        for c in cases:
            case_id = int(c["case_id"])
            fixture = json.loads(c["fixture_json"] or "{}")
            golden = json.loads(c["golden_json"] or "{}")

            try:
                # ---- 여기에서 네 파이프라인을 실제로 실행해야 함 ----
                # 예:
                # model_id = structai_model_import_members(...)
                # analysis_run_id = structai_results_import(...)
                # (옵션) cad parse/apply/sections resolve
                # check_run_id = structai_check_run(...)
                # actual = _metrics_for_check_run(conn, check_run_id)

                # 임시 placeholder:
                raise NotImplementedError("Connect your pipeline here to produce a check_run_id and metrics.")

            except Exception as e:
                all_ok = False
                diff = {"error": str(e)}
                conn.execute(
                    "INSERT INTO regression_case_results(run_id, case_id, status, diff_json) VALUES(?,?,?,?)",
                    (run_id, case_id, "ERROR", json.dumps(diff, ensure_ascii=False)),
                )
                case_reports.append({"case": c["name"], "status": "ERROR", "diff": diff})

        status = "PASS" if all_ok else "FAIL"
        conn.execute(
            "UPDATE regression_runs SET status=?, finished_at=datetime('now'), report_json=? WHERE run_id=?",
            (status, json.dumps({"cases": case_reports}, ensure_ascii=False), run_id),
        )
        conn.commit()
        return {"ok": True, "run_id": run_id, "status": status, "cases": case_reports}
    finally:
        conn.close()


위 NotImplementedError 부분이 “너의 기존 import/check 도구를 실제로 호출하는 연결부”인데,
너는 이미 Step5~8에서 import/check 흐름이 있으니, 그걸 그대로 꽂으면 된다.
(이 단계에서 질문하지 않고) 내가 추천하는 연결 방식은 “회귀용 임시 모델/런 이름에 UUID를 붙여 생성 → 실행 후 DB에서 해당 run/model 삭제”로 깨끗하게 유지하는 거야.

9-7) Step9 “한 번에 굴리는” 추천 실행 순서

표준 단면 라이브러리 임포트(가능하면 먼저)

#structai_sections_migrate_add_priority {}
#structai_sections_import_catalog { "path": ".../steel_sections.csv" }


DXF → table/spec/story 파이프라인

#structai_import_dxf { "path": ".../S-101.dxf" }
#structai_cad_parse_specs { "cad_artifact_id": 1, "overwrite": true }
#structai_cad_detect_story_tags { "cad_artifact_id": 1, "overwrite": true }
#structai_cad_extract_tables { "cad_artifact_id": 1, "min_cells": 16 }
#structai_cad_infer_table_schemas_v2 { "cad_artifact_id": 1, "overwrite": true }
#structai_token_story_map_build { "cad_artifact_id": 1, "model_id": 1 }
#structai_token_story_auto_confirm { "cad_artifact_id": 1, "model_id": 1 }
#structai_specs_from_table_rows { "cad_artifact_id": 1, "model_id": 1 }
#structai_specs_auto_confirm_table_schema { "cad_artifact_id": 1, "model_id": 1 }
#structai_design_apply_specs_to_inputs { "cad_artifact_id": 1, "model_id": 1, "overwrite_keys": false }


섹션 resolve(모델 section 문자열 기반 fallback)

#structai_sections_resolve_members { "model_id": 1, "default_Fy_MPa": 325, "overwrite_design": false }


품질 체크(뭐가 부족한지 바로 확인)

#structai_quality_summary { "model_id": 1, "analysis_run_id": 1 }
#structai_token_story_conflicts { "cad_artifact_id": 1, "model_id": 1 }


내력 계산 → checks → report (이전 단계 그대로)

9-8) 여기서부터 “프로덕션 품질”을 만드는 핵심 원칙 3개

표준값 우선

computed는 언제나 fallback(우선순위 낮게)

표준 카탈로그(우선순위 높게)를 지속적으로 확장

NA를 두려워하지 않기

정보 없는데 억지로 추정해서 PASS 만드는 것보다
“NA 이유를 정확히 설명”하는 것이 도구 신뢰성을 만든다

Regression으로 품질 고정

파서/매핑/룰셋/내력식이 늘어날수록
회귀 테스트 없으면 어느 순간 ‘조용히’ 결과가 변한다

golden 비교는 개발 속도를 오히려 올린다

원하면 다음 단계(10단계)로는 “실제 현장 적용에서 가장 강력한 것”인

상용 프로그램/수기 계산서 비교 자동화(케이스별 리포트 생성)

룰셋/KDS 조항 적용을 ‘체크 타입별 템플릿’으로 표준화

VS Code 패널에서 ‘충돌/누락/확정’ 워크플로우 완성(일괄 승인/되돌리기/근거 표시)
이걸 같은 방식으로 바로 이어서 구현해줄게.
