좋아. 5단계는 네가 말한 “실제 구조설계 로직(결정론) + 계산서/검토서 자동 생성”을 MCP 도구 + DB + VS Code 패널까지 한 번에 실행 가능한 형태로 끌어올리는 단계야.

다만 먼저 아주 중요한 전제 1개만 분명히 할게:

아래에 넣는 **RC/강재 내력 계산식은 ‘프레임워크용(샘플/기본형)’**이야.

실제 KDS/KBC/ACI/AISC 설계 검토에 그대로 쓰면 위험하고, 반드시 **검증/보정(단위, 계수, 한계상태 정의, 세부 조항)**이 필요해.

우리가 만드는 도구의 방향은 “LLM이 계산하는 게 아니라, 결정론 엔진이 계산 + AI는 근거/흐름/자동화”가 맞고, 그 뼈대를 이번 단계에서 완성한다.

5-0) 이번 단계에서 완성되는 것

결정론 내력 산정 모듈화

RC 보(직사각 단면, 단철근 가정) phiMn, phiVn

RC 기둥(축압 중심, 단순) phiPn

강재 보(단순) phiMn, phiVn

계산 결과를 member_design_inputs.design_json에 **계산 추적(trace)**과 함께 저장

체크 엔진은 기존처럼 ratio = demand/capacity로 통일

capacity는 이제 “외부 입력(Mn/Vn/Pn)”이 아니라 내력산정 도구가 자동으로 채워줌

리포트 자동 생성 파이프라인

check_run_id 하나로 Markdown + PDF 생성

PDF는 reportlab로 생성 (VS Code/도구에서 바로 저장/열기)

생성된 보고서는 artifacts에 등록해서 “프로젝트 메모리”에 포함 (MD는 검색까지 가능)

VS Code Results 패널 버튼 1개로

“Generate Report” → MD/PDF 생성 → 바로 열기

5-1) DB 스키마 업데이트: reports 테이블 추가 (v0.0.6)

지금은 개발단계라면 가장 확실한 방법은 DB를 새로 만드는 거야.
(이미 데이터가 많다면 나중에 migration 스크립트를 추가하면 되고, 지금은 속도/확실성이 우선)

mcp_server/schema.sql 맨 아래에 추가해줘(기존 v0.0.5 유지 + 아래만 추가해도 됨):

-- schema.sql patch for v0.0.6
UPDATE meta SET value='0.0.6' WHERE key='schema_version';

CREATE TABLE IF NOT EXISTS reports (
  report_id INTEGER PRIMARY KEY AUTOINCREMENT,
  check_run_id INTEGER NOT NULL REFERENCES check_runs(check_run_id) ON DELETE CASCADE,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  format TEXT NOT NULL,             -- 'md' | 'pdf'
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_reports_check_run ON reports(check_run_id, format);

5-2) “결정론 설계 엔진”을 서버에서 분리(모듈화)하기

이게 Cursor/Claude 스타일로 확장할 때 제일 중요해.
LLM은 바뀌어도 설계 계산 모듈은 그대로 유지돼야 하거든.

5-2-1) 폴더 구조 추가

mcp_server/ 아래에 다음 폴더/파일을 추가해줘:

mcp_server/
  server.py
  schema.sql
  design/
    __init__.py
    units.py
    rc.py
    steel.py
  reporting/
    __init__.py
    md.py
    pdf.py

5-2-2) design/units.py
# mcp_server/design/units.py

def N_to_kN(x: float) -> float:
    return x / 1e3

def Nmm_to_kNm(x: float) -> float:
    # 1 kN-m = 1e6 N-mm
    return x / 1e6

def MPa_is_N_per_mm2(x: float) -> float:
    # 1 MPa == 1 N/mm^2
    return x

5-2-3) design/rc.py (RC 기본형)
# mcp_server/design/rc.py
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, Optional, Tuple

from .units import N_to_kN, Nmm_to_kNm


@dataclass
class RcBeamRectInputs:
    # Geometry (mm)
    b: float
    h: Optional[float] = None
    d: Optional[float] = None
    cover: Optional[float] = None
    stirrup_dia: Optional[float] = None
    bar_dia: Optional[float] = None

    # Materials (MPa)
    fc: float = 24.0
    fy: float = 400.0

    # Reinforcement
    As: float = 0.0      # mm2
    Av: float = 0.0      # mm2 per spacing s (e.g., 2 legs * area)
    s: Optional[float] = None  # mm

    # Strength reduction factors
    phi_flex: float = 0.9
    phi_shear: float = 0.75


def _compute_effective_d(inp: RcBeamRectInputs) -> float:
    """
    d is preferred.
    If missing, tries: d = h - cover - stirrup_dia - bar_dia/2
    """
    if inp.d is not None and inp.d > 0:
        return float(inp.d)

    if inp.h is None or inp.cover is None:
        raise ValueError("Need either d, or (h and cover) to compute effective depth d.")
    h = float(inp.h)
    cover = float(inp.cover)
    stir = float(inp.stirrup_dia or 0.0)
    bar = float(inp.bar_dia or 0.0)
    return h - cover - stir - 0.5 * bar


def rc_beam_rect_flexure_phiMn_kNm(inp: RcBeamRectInputs) -> Tuple[float, Dict[str, Any]]:
    """
    Rectangular, singly reinforced beam (basic):
      a = As*fy / (0.85*fc*b)
      Mn = As*fy*(d - a/2)    [N-mm]
      phiMn = phi_flex * Mn   [kN-m]
    Units:
      fc, fy: MPa = N/mm2
      b, d: mm
      As: mm2
    """
    b = float(inp.b)
    d = _compute_effective_d(inp)
    fc = float(inp.fc)
    fy = float(inp.fy)
    As = float(inp.As)
    if b <= 0 or d <= 0 or fc <= 0 or fy <= 0 or As <= 0:
        raise ValueError("b,d,fc,fy,As must be positive for flexure capacity.")

    a = As * fy / (0.85 * fc * b)
    Mn_Nmm = As * fy * (d - a / 2.0)
    phiMn_kNm = float(inp.phi_flex) * Nmm_to_kNm(Mn_Nmm)

    trace = {
        "method": "rc_beam_rect_flexure_v1",
        "inputs": {
            "b_mm": b, "d_mm": d,
            "fc_MPa": fc, "fy_MPa": fy,
            "As_mm2": As,
            "phi_flex": float(inp.phi_flex),
        },
        "intermediate": {
            "a_mm": a,
            "Mn_Nmm": Mn_Nmm,
            "Mn_kNm": Nmm_to_kNm(Mn_Nmm),
        },
        "outputs": {
            "phiMn_kNm": phiMn_kNm,
        },
    }
    return phiMn_kNm, trace


def rc_beam_rect_shear_phiVn_kN(inp: RcBeamRectInputs) -> Tuple[float, Dict[str, Any]]:
    """
    Basic shear capacity:
      Vc = 0.17*sqrt(fc)*b*d   [N]   (common form; must be calibrated to your code)
      Vs = Av*fy*d/s           [N]   (if s provided)
      Vn = Vc + Vs
      phiVn = phi_shear * Vn   [kN]
    """
    b = float(inp.b)
    d = _compute_effective_d(inp)
    fc = float(inp.fc)
    fy = float(inp.fy)

    if b <= 0 or d <= 0 or fc <= 0:
        raise ValueError("b,d,fc must be positive for shear capacity.")

    Vc_N = 0.17 * sqrt(fc) * b * d

    Vs_N = 0.0
    if inp.Av and inp.s:
        Av = float(inp.Av)
        s = float(inp.s)
        if Av > 0 and s > 0 and fy > 0:
            Vs_N = Av * fy * d / s

    Vn_N = Vc_N + Vs_N
    phiVn_kN = float(inp.phi_shear) * N_to_kN(Vn_N)

    trace = {
        "method": "rc_beam_rect_shear_v1",
        "inputs": {
            "b_mm": b, "d_mm": d,
            "fc_MPa": fc, "fy_MPa": fy,
            "Av_mm2": float(inp.Av), "s_mm": float(inp.s) if inp.s is not None else None,
            "phi_shear": float(inp.phi_shear),
        },
        "intermediate": {
            "Vc_N": Vc_N,
            "Vs_N": Vs_N,
            "Vn_N": Vn_N,
            "Vn_kN": N_to_kN(Vn_N),
        },
        "outputs": {"phiVn_kN": phiVn_kN},
    }
    return phiVn_kN, trace


@dataclass
class RcColumnAxialInputs:
    # Areas (mm2) or from b*h
    Ag: Optional[float] = None
    b: Optional[float] = None
    h: Optional[float] = None
    As: float = 0.0   # mm2

    # Materials (MPa)
    fc: float = 24.0
    fy: float = 400.0

    # Strength reduction
    phi_axial: float = 0.65


def _compute_Ag(inp: RcColumnAxialInputs) -> float:
    if inp.Ag is not None and inp.Ag > 0:
        return float(inp.Ag)
    if inp.b is None or inp.h is None:
        raise ValueError("Need Ag or (b and h) for column axial capacity.")
    return float(inp.b) * float(inp.h)


def rc_column_axial_phiPn_kN(inp: RcColumnAxialInputs) -> Tuple[float, Dict[str, Any]]:
    """
    Very simplified concentric axial strength:
      P0 = 0.85*fc*(Ag - As) + fy*As  [N]
      phiPn = phi_axial * P0         [kN]
    """
    Ag = _compute_Ag(inp)
    As = float(inp.As)
    fc = float(inp.fc)
    fy = float(inp.fy)
    if Ag <= 0 or fc <= 0:
        raise ValueError("Ag and fc must be positive.")
    if As < 0:
        raise ValueError("As must be >= 0.")

    P0_N = 0.85 * fc * max(Ag - As, 0.0) + fy * As
    phiPn_kN = float(inp.phi_axial) * N_to_kN(P0_N)

    trace = {
        "method": "rc_column_axial_v1",
        "inputs": {
            "Ag_mm2": Ag, "As_mm2": As,
            "fc_MPa": fc, "fy_MPa": fy,
            "phi_axial": float(inp.phi_axial),
        },
        "intermediate": {"P0_N": P0_N, "P0_kN": N_to_kN(P0_N)},
        "outputs": {"phiPn_kN": phiPn_kN},
    }
    return phiPn_kN, trace

5-2-4) design/steel.py (강재 기본형)
# mcp_server/design/steel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .units import N_to_kN, Nmm_to_kNm


@dataclass
class SteelBeamInputs:
    Fy: float          # MPa = N/mm2
    Zx: float          # mm3
    Aw: float          # mm2 (shear area)
    phi_flex: float = 0.9
    phi_shear: float = 1.0


def steel_beam_phiMn_kNm(inp: SteelBeamInputs) -> Tuple[float, Dict[str, Any]]:
    """
    Basic plastic/section moment strength (simplified):
      Mn = Fy * Zx  [N-mm]
      phiMn in kN-m
    """
    Fy = float(inp.Fy)
    Zx = float(inp.Zx)
    if Fy <= 0 or Zx <= 0:
        raise ValueError("Fy and Zx must be positive.")
    Mn_Nmm = Fy * Zx
    phiMn_kNm = float(inp.phi_flex) * Nmm_to_kNm(Mn_Nmm)
    trace = {
        "method": "steel_beam_flexure_v1",
        "inputs": {"Fy_MPa": Fy, "Zx_mm3": Zx, "phi_flex": float(inp.phi_flex)},
        "intermediate": {"Mn_Nmm": Mn_Nmm, "Mn_kNm": Nmm_to_kNm(Mn_Nmm)},
        "outputs": {"phiMn_kNm": phiMn_kNm},
    }
    return phiMn_kNm, trace


def steel_beam_phiVn_kN(inp: SteelBeamInputs) -> Tuple[float, Dict[str, Any]]:
    """
    Basic shear strength (simplified):
      Vn = 0.6 * Fy * Aw [N]
      phiVn [kN]
    """
    Fy = float(inp.Fy)
    Aw = float(inp.Aw)
    if Fy <= 0 or Aw <= 0:
        raise ValueError("Fy and Aw must be positive.")
    Vn_N = 0.6 * Fy * Aw
    phiVn_kN = float(inp.phi_shear) * N_to_kN(Vn_N)
    trace = {
        "method": "steel_beam_shear_v1",
        "inputs": {"Fy_MPa": Fy, "Aw_mm2": Aw, "phi_shear": float(inp.phi_shear)},
        "intermediate": {"Vn_N": Vn_N, "Vn_kN": N_to_kN(Vn_N)},
        "outputs": {"phiVn_kN": phiVn_kN},
    }
    return phiVn_kN, trace

5-3) 보고서 생성 모듈 (Markdown + PDF)
5-3-1) reporting/md.py
# mcp_server/reporting/md.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def build_markdown_report(meta: Dict[str, Any], summary: Dict[str, int], items: List[Dict[str, Any]]) -> str:
    """
    meta: {model_name, analysis_name, rulepack, created_at, check_run_id, ...}
    items: list of check results rows (already denormalized)
    """
    lines: List[str] = []
    lines.append(f"# Structural Check Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Check run: #{meta.get('check_run_id')} - {meta.get('check_run_name')}")
    lines.append(f"- Model: {meta.get('model_name')} (id={meta.get('model_id')})")
    lines.append(f"- Analysis run: {meta.get('analysis_run_name')} (id={meta.get('analysis_run_id')})")
    lines.append(f"- Rulepack: {meta.get('rulepack_name')}/{meta.get('rulepack_version')}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- PASS: {summary.get('PASS',0)}")
    lines.append(f"- WARN: {summary.get('WARN',0)}")
    lines.append(f"- FAIL: {summary.get('FAIL',0)}")
    lines.append(f"- NA: {summary.get('NA',0)}")
    lines.append("")

    lines.append("## Results (top)")
    lines.append("")
    lines.append("| Status | Label | UID | Type | Check | Combo | Demand | Capacity | Ratio | Citations |")
    lines.append("|---|---|---|---|---|---|---:|---:|---:|---|")

    def fmt(x):
        if x is None:
            return ""
        try:
            return f"{float(x):.3f}"
        except Exception:
            return str(x)

    for r in items:
        cites = r.get("citations") or []
        cite_txts = []
        for c in cites[:2]:
            pg = c.get("page")
            title = c.get("title") or c.get("uri") or ""
            if pg is not None:
                cite_txts.append(f"{title} p.{pg}")
            else:
                cite_txts.append(f"{title}")
        cite_cell = "; ".join(cite_txts)

        lines.append(
            f"| {r.get('status')} | {r.get('member_label','') or ''} | {r.get('member_uid','')} | "
            f"{r.get('type','')} | {r.get('check_type','')} | {r.get('combo','')} | "
            f"{fmt(r.get('demand_value'))} | {fmt(r.get('capacity_value'))} | {fmt(r.get('ratio'))} | {cite_cell} |"
        )

    lines.append("")
    lines.append("## Details (FAIL/WARN)")
    lines.append("")

    for r in items:
        if r.get("status") not in ("FAIL", "WARN"):
            continue

        lines.append(f"### {r.get('status')} - {r.get('member_label','')} ({r.get('member_uid','')}) - {r.get('check_type','')} / {r.get('combo','')}")
        lines.append("")
        lines.append(f"- Demand: {r.get('demand_value')}")
        lines.append(f"- Capacity: {r.get('capacity_value')}")
        lines.append(f"- Ratio: {r.get('ratio')}")
        lines.append("")

        # calc trace (if exists)
        trace = r.get("design_trace")
        if isinstance(trace, dict):
            lines.append("**Design trace**")
            lines.append("")
            for k, v in trace.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        cites = r.get("citations") or []
        if cites:
            lines.append("**Citations**")
            lines.append("")
            for c in cites[:3]:
                lines.append(f"- {c.get('cite_uri') or c.get('uri')}")
                snip = (c.get('snippet') or "").strip()
                if snip:
                    lines.append(f"  - {snip}")
            lines.append("")

    return "\n".join(lines)

5-3-2) reporting/pdf.py (reportlab PDF)
# mcp_server/reporting/pdf.py
from __future__ import annotations

from typing import Any, Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillGray(0.35)
    canvas.drawRightString(A4[0] - 15 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_report(pdf_path: str, meta: Dict[str, Any], summary: Dict[str, int], items: List[Dict[str, Any]]) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontName="Helvetica", fontSize=9, leading=11))
    styles.add(ParagraphStyle(name="H2", fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=16*mm,
        rightMargin=16*mm,
        topMargin=16*mm,
        bottomMargin=16*mm,
        title="Structural Check Report",
        author="StructAI"
    )

    story: List[Any] = []

    story.append(Paragraph("Structural Check Report", styles["Title"]))
    story.append(Spacer(1, 6*mm))

    meta_lines = [
        f"Check run: #{meta.get('check_run_id')} - {meta.get('check_run_name')}",
        f"Model: {meta.get('model_name')} (id={meta.get('model_id')})",
        f"Analysis run: {meta.get('analysis_run_name')} (id={meta.get('analysis_run_id')})",
        f"Rulepack: {meta.get('rulepack_name')}/{meta.get('rulepack_version')}",
    ]
    story.append(Paragraph("<br/>".join(meta_lines), styles["Normal"]))
    story.append(Spacer(1, 6*mm))

    story.append(Paragraph("Summary", styles["H2"]))
    sum_table = Table(
        [
            ["PASS", summary.get("PASS", 0), "WARN", summary.get("WARN", 0)],
            ["FAIL", summary.get("FAIL", 0), "NA", summary.get("NA", 0)],
        ],
        colWidths=[22*mm, 18*mm, 22*mm, 18*mm],
    )
    sum_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
        ("BOX", (0,0), (-1,-1), 0.6, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("FONT", (0,0), (-1,-1), "Helvetica", 10),
        ("ALIGN", (1,0), (1,-1), "RIGHT"),
        ("ALIGN", (3,0), (3,-1), "RIGHT"),
    ]))
    story.append(sum_table)
    story.append(Spacer(1, 6*mm))

    story.append(Paragraph("Top results", styles["H2"]))

    header = ["Status", "Label", "UID", "Check", "Combo", "Demand", "Capacity", "Ratio", "Citations"]
    rows = [header]

    def fmt(x):
        if x is None:
            return ""
        try:
            return f"{float(x):.3f}"
        except Exception:
            return str(x)

    for r in items[:120]:
        cites = r.get("citations") or []
        cite_txts = []
        for c in cites[:2]:
            title = c.get("title") or ""
            pg = c.get("page")
            if pg is not None:
                cite_txts.append(f"{title} p.{pg}")
            else:
                cite_txts.append(title or (c.get("uri") or ""))
        cite_cell = "; ".join([t for t in cite_txts if t])

        rows.append([
            r.get("status",""),
            r.get("member_label","") or "",
            r.get("member_uid","") or "",
            r.get("check_type","") or "",
            r.get("combo","") or "",
            fmt(r.get("demand_value")),
            fmt(r.get("capacity_value")),
            fmt(r.get("ratio")),
            cite_cell
        ])

    tbl = Table(rows, colWidths=[16*mm, 18*mm, 18*mm, 26*mm, 16*mm, 16*mm, 18*mm, 14*mm, 40*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
        ("FONT", (0,1), (-1,-1), "Helvetica", 8),
        ("BOX", (0,0), (-1,-1), 0.6, colors.black),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (5,1), (7,-1), "RIGHT"),
    ]))
    story.append(tbl)

    story.append(PageBreak())
    story.append(Paragraph("Details (FAIL/WARN)", styles["H2"]))

    # detail pages for failures
    for r in items:
        if r.get("status") not in ("FAIL", "WARN"):
            continue

        title = f"{r.get('status')} - {r.get('member_label','')} ({r.get('member_uid','')}) - {r.get('check_type','')} / {r.get('combo','')}"
        story.append(Paragraph(title, styles["Normal"]))
        story.append(Spacer(1, 2*mm))

        story.append(Paragraph(f"Demand: {r.get('demand_value')}", styles["Small"]))
        story.append(Paragraph(f"Capacity: {r.get('capacity_value')}", styles["Small"]))
        story.append(Paragraph(f"Ratio: {r.get('ratio')}", styles["Small"]))
        story.append(Spacer(1, 2*mm))

        cites = r.get("citations") or []
        if cites:
            story.append(Paragraph("Citations:", styles["Small"]))
            for c in cites[:3]:
                uri = c.get("cite_uri") or c.get("uri") or ""
                snip = (c.get("snippet") or "").strip()
                story.append(Paragraph(f"- {uri}", styles["Small"]))
                if snip:
                    story.append(Paragraph(f"&nbsp;&nbsp;{snip}", styles["Small"]))
            story.append(Spacer(1, 2*mm))

        story.append(Spacer(1, 4*mm))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

5-3-3) reporting/init.py / design/init.py
# mcp_server/design/__init__.py
# (empty)

# mcp_server/reporting/__init__.py
# (empty)

5-4) server.py에 “내력 자동 산정 + 보고서 생성” 도구 추가

아래는 Step 4의 server.py를 기준으로 “추가/수정해야 할 부분만” 제공할게.
(너의 server.py가 조금 달라도, 아래 도구/헬퍼를 같은 방식으로 넣으면 된다)

5-4-1) import 추가

server.py 상단 import 근처에 추가:

from mcp_server.design.rc import (
    RcBeamRectInputs,
    RcColumnAxialInputs,
    rc_beam_rect_flexure_phiMn_kNm,
    rc_beam_rect_shear_phiVn_kN,
    rc_column_axial_phiPn_kN,
)
from mcp_server.design.steel import (
    SteelBeamInputs,
    steel_beam_phiMn_kNm,
    steel_beam_phiVn_kN,
)
from mcp_server.reporting.md import build_markdown_report
from mcp_server.reporting.pdf import build_pdf_report


만약 import 경로 문제가 있으면(패키지로 실행하지 않을 때)
from design.rc import ... 처럼 상대 폴더 import로 바꿔줘.

5-4-2) 설계 입력 조회 도구(편의)
@mcp.tool()
def structai_design_get_member_inputs(model_id: int, member_uid: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        mm = conn.execute(
            "SELECT model_member_id FROM model_members WHERE model_id=? AND member_uid=?",
            (int(model_id), str(member_uid)),
        ).fetchone()
        if not mm:
            raise ValueError("member not found")
        mmid = int(mm["model_member_id"])
        row = conn.execute(
            "SELECT design_json FROM member_design_inputs WHERE model_member_id=?",
            (mmid,),
        ).fetchone()
        return {"model_member_id": mmid, "design": json.loads(row["design_json"]) if row else {}}
    finally:
        conn.close()

5-4-3) RC 보 내력 자동 산정(핵심)

design_json에서 아래 키들을 읽는다(없으면 defaults로 보정):

b, h, d, cover, stirrup_dia, bar_dia (mm)

fc, fy (MPa)

As, Av, s (mm2, mm2, mm)

phi_flex, phi_shear

산정 결과는 design_json에:

Mn (kN-m), Vn (kN)

calc_trace에 상세 추적값 저장

@mcp.tool()
def structai_design_compute_rc_beam_rect(
    model_id: int,
    where: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    RC 보(직사각 단면, 기본형) phiMn(kN-m), phiVn(kN) 계산 후 design_inputs에 저장.
    where 예:
      {"type":"beam", "story":"3F", "label_contains":"B"}
    defaults 예:
      {"fc":27, "fy":400, "phi_flex":0.9, "phi_shear":0.75}
    """
    where = where or {}
    defaults = defaults or {}

    conn = _connect()
    try:
        sql = "SELECT model_member_id, member_uid, member_label, type, story FROM model_members WHERE model_id=?"
        params = [int(model_id)]
        if where.get("type"):
            sql += " AND type=?"
            params.append(str(where["type"]))
        if where.get("story"):
            sql += " AND story=?"
            params.append(str(where["story"]))
        if where.get("label_contains"):
            sql += " AND UPPER(COALESCE(member_label,'')) LIKE ?"
            params.append(f"%{str(where['label_contains']).upper()}%")

        members = conn.execute(sql, params).fetchall()

        updated = 0
        skipped = []
        for m in members:
            mmid = int(m["model_member_id"])
            row = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
            dj = json.loads(row["design_json"]) if row else {}

            # overwrite 정책
            if not overwrite and ("Mn" in dj or "Vn" in dj):
                skipped.append({"uid": m["member_uid"], "reason": "already has Mn/Vn (overwrite=false)"})
                continue

            # merge defaults
            merged = {**defaults, **dj}

            try:
                inp = RcBeamRectInputs(
                    b=float(merged["b"]),
                    h=float(merged["h"]) if "h" in merged else None,
                    d=float(merged["d"]) if "d" in merged else None,
                    cover=float(merged["cover"]) if "cover" in merged else None,
                    stirrup_dia=float(merged["stirrup_dia"]) if "stirrup_dia" in merged else None,
                    bar_dia=float(merged["bar_dia"]) if "bar_dia" in merged else None,
                    fc=float(merged.get("fc", 24.0)),
                    fy=float(merged.get("fy", 400.0)),
                    As=float(merged.get("As", 0.0)),
                    Av=float(merged.get("Av", 0.0)),
                    s=float(merged["s"]) if "s" in merged else None,
                    phi_flex=float(merged.get("phi_flex", 0.9)),
                    phi_shear=float(merged.get("phi_shear", 0.75)),
                )

                phiMn, tr_f = rc_beam_rect_flexure_phiMn_kNm(inp)
                phiVn, tr_v = rc_beam_rect_shear_phiVn_kN(inp)

                merged["Mn"] = phiMn
                merged["Vn"] = phiVn
                merged["calc_trace"] = {
                    "flexure": tr_f,
                    "shear": tr_v,
                }
                merged["design_method"] = "rc_beam_rect_v1"
                merged["units"] = {"moment": "kN-m", "force": "kN", "length": "mm", "stress": "MPa"}

                conn.execute(
                    """
                    INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
                    VALUES(?,?, datetime('now'))
                    ON CONFLICT(model_member_id) DO UPDATE SET
                      design_json=excluded.design_json,
                      updated_at=datetime('now')
                    """,
                    (mmid, json.dumps(merged, ensure_ascii=False)),
                )
                updated += 1
            except Exception as e:
                skipped.append({"uid": m["member_uid"], "reason": str(e)})

        conn.commit()
        return {"ok": True, "updated": updated, "skipped": skipped}
    finally:
        conn.close()

5-4-4) RC 기둥 축압 내력 자동 산정(기본형)
@mcp.tool()
def structai_design_compute_rc_column_axial(
    model_id: int,
    where: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    RC 기둥(축압 중심, 단순형) phiPn(kN) 계산.
    입력: Ag 또는 (b,h), As, fc, fy, phi_axial
    결과: Pn (kN)
    """
    where = where or {}
    defaults = defaults or {}

    conn = _connect()
    try:
        sql = "SELECT model_member_id, member_uid, member_label, type, story FROM model_members WHERE model_id=?"
        params = [int(model_id)]
        if where.get("type"):
            sql += " AND type=?"
            params.append(str(where["type"]))
        if where.get("story"):
            sql += " AND story=?"
            params.append(str(where["story"]))

        members = conn.execute(sql, params).fetchall()

        updated = 0
        skipped = []
        for m in members:
            mmid = int(m["model_member_id"])
            row = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
            dj = json.loads(row["design_json"]) if row else {}
            if not overwrite and ("Pn" in dj):
                skipped.append({"uid": m["member_uid"], "reason": "already has Pn (overwrite=false)"})
                continue

            merged = {**defaults, **dj}
            try:
                inp = RcColumnAxialInputs(
                    Ag=float(merged["Ag"]) if "Ag" in merged else None,
                    b=float(merged["b"]) if "b" in merged else None,
                    h=float(merged["h"]) if "h" in merged else None,
                    As=float(merged.get("As", 0.0)),
                    fc=float(merged.get("fc", 24.0)),
                    fy=float(merged.get("fy", 400.0)),
                    phi_axial=float(merged.get("phi_axial", 0.65)),
                )
                phiPn, tr = rc_column_axial_phiPn_kN(inp)
                merged["Pn"] = phiPn
                merged["calc_trace"] = {"axial": tr}
                merged["design_method"] = "rc_column_axial_v1"
                merged["units"] = {"force": "kN", "length": "mm", "stress": "MPa"}

                conn.execute(
                    """
                    INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
                    VALUES(?,?, datetime('now'))
                    ON CONFLICT(model_member_id) DO UPDATE SET
                      design_json=excluded.design_json,
                      updated_at=datetime('now')
                    """,
                    (mmid, json.dumps(merged, ensure_ascii=False)),
                )
                updated += 1
            except Exception as e:
                skipped.append({"uid": m["member_uid"], "reason": str(e)})

        conn.commit()
        return {"ok": True, "updated": updated, "skipped": skipped}
    finally:
        conn.close()

5-4-5) 강재 보 내력 자동 산정(기본형)
@mcp.tool()
def structai_design_compute_steel_beam_simple(
    model_id: int,
    where: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    강재 보(단순형) phiMn(kN-m), phiVn(kN) 계산.
    입력: Fy(MPa), Zx(mm3), Aw(mm2), phi_flex, phi_shear
    """
    where = where or {}
    defaults = defaults or {}

    conn = _connect()
    try:
        sql = "SELECT model_member_id, member_uid, member_label, type, story FROM model_members WHERE model_id=?"
        params = [int(model_id)]
        if where.get("type"):
            sql += " AND type=?"
            params.append(str(where["type"]))

        members = conn.execute(sql, params).fetchall()

        updated = 0
        skipped = []
        for m in members:
            mmid = int(m["model_member_id"])
            row = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
            dj = json.loads(row["design_json"]) if row else {}

            if not overwrite and ("Mn" in dj or "Vn" in dj):
                skipped.append({"uid": m["member_uid"], "reason": "already has Mn/Vn (overwrite=false)"})
                continue

            merged = {**defaults, **dj}
            try:
                inp = SteelBeamInputs(
                    Fy=float(merged["Fy"]),
                    Zx=float(merged["Zx"]),
                    Aw=float(merged["Aw"]),
                    phi_flex=float(merged.get("phi_flex", 0.9)),
                    phi_shear=float(merged.get("phi_shear", 1.0)),
                )

                phiMn, tr_m = steel_beam_phiMn_kNm(inp)
                phiVn, tr_v = steel_beam_phiVn_kN(inp)

                merged["Mn"] = phiMn
                merged["Vn"] = phiVn
                merged["calc_trace"] = {"flexure": tr_m, "shear": tr_v}
                merged["design_method"] = "steel_beam_simple_v1"
                merged["units"] = {"moment": "kN-m", "force": "kN", "stress": "MPa", "length": "mm"}

                conn.execute(
                    """
                    INSERT INTO member_design_inputs(model_member_id, design_json, updated_at)
                    VALUES(?,?, datetime('now'))
                    ON CONFLICT(model_member_id) DO UPDATE SET
                      design_json=excluded.design_json,
                      updated_at=datetime('now')
                    """,
                    (mmid, json.dumps(merged, ensure_ascii=False)),
                )
                updated += 1
            except Exception as e:
                skipped.append({"uid": m["member_uid"], "reason": str(e)})

        conn.commit()
        return {"ok": True, "updated": updated, "skipped": skipped}
    finally:
        conn.close()

5-4-6) 보고서 생성 도구: structai_report_generate
@mcp.tool()
def structai_report_generate(
    check_run_id: int,
    formats: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    include_pass: bool = False,
    limit_items: int = 500
) -> Dict[str, Any]:
    """
    check_run_id 기반으로 보고서 생성:
    - Markdown(.md) + PDF(.pdf)
    - 생성된 파일을 artifacts + reports 테이블에 등록
    """
    formats = formats or ["md", "pdf"]
    out_dir_path = Path(out_dir).expanduser().resolve() if out_dir else (DB_PATH.parent / "reports")
    out_dir_path.mkdir(parents=True, exist_ok=True)

    conn = _connect()
    try:
        # meta
        cr = conn.execute(
            """
            SELECT check_run_id, model_id, analysis_run_id, name, rulepack_name, rulepack_version, created_at
            FROM check_runs
            WHERE check_run_id=?
            """,
            (int(check_run_id),),
        ).fetchone()
        if not cr:
            raise ValueError("check_run not found")

        model = conn.execute("SELECT name FROM models WHERE model_id=?", (int(cr["model_id"]),)).fetchone()
        ar = conn.execute("SELECT name FROM analysis_runs WHERE analysis_run_id=?", (int(cr["analysis_run_id"]),)).fetchone()

        meta = {
            "check_run_id": int(cr["check_run_id"]),
            "check_run_name": cr["name"],
            "model_id": int(cr["model_id"]),
            "model_name": (model["name"] if model else f"model_{cr['model_id']}"),
            "analysis_run_id": int(cr["analysis_run_id"]),
            "analysis_run_name": (ar["name"] if ar else f"analysis_{cr['analysis_run_id']}"),
            "rulepack_name": cr["rulepack_name"] or "builtin",
            "rulepack_version": cr["rulepack_version"] or "",
        }

        # summary counts
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM check_results WHERE check_run_id=? GROUP BY status",
            (int(check_run_id),),
        ).fetchall()
        summary = {r["status"]: int(r["n"]) for r in rows}

        # denormalized results
        where_clause = "" if include_pass else "AND cr.status IN ('FAIL','WARN','NA')"
        items_rows = conn.execute(
            f"""
            SELECT
              cr.status, cr.combo, cr.check_type, cr.demand_value, cr.capacity_value, cr.ratio,
              mb.model_member_id, mb.member_uid, mb.member_label, mb.type, mb.section, mb.story,
              cr.citations_json,
              mdi.design_json
            FROM check_results cr
            JOIN model_members mb ON mb.model_member_id = cr.model_member_id
            LEFT JOIN member_design_inputs mdi ON mdi.model_member_id = mb.model_member_id
            WHERE cr.check_run_id=? {where_clause}
            ORDER BY
              CASE cr.status WHEN 'FAIL' THEN 0 WHEN 'WARN' THEN 1 WHEN 'NA' THEN 2 ELSE 3 END,
              cr.ratio DESC
            LIMIT ?
            """,
            (int(check_run_id), int(limit_items)),
        ).fetchall()

        items: List[Dict[str, Any]] = []
        for r in items_rows:
            it = dict(r)
            try:
                it["citations"] = json.loads(it.pop("citations_json") or "[]")
            except Exception:
                it["citations"] = []
            try:
                design = json.loads(it.pop("design_json") or "{}")
            except Exception:
                design = {}
            # calc_trace만 요약해서 붙임(너무 길면 보고서가 과대해짐)
            it["design_trace"] = design.get("calc_trace")
            items.append(it)

        outputs = []

        # Markdown
        if "md" in formats:
            md_text = build_markdown_report(meta, summary, items)
            md_path = out_dir_path / f"check_run_{check_run_id}.md"
            md_path.write_text(md_text, encoding="utf-8")

            sha = _sha256_file(md_path)
            uri = md_path.as_uri()
            art_id = _upsert_artifact(conn, uri=uri, kind="report_md", title=f"Report MD #{check_run_id}", source_path=str(md_path), sha256=sha)

            # MD를 doc_chunks에 인덱싱(검색 가능)
            _delete_doc_chunks_for_artifact(conn, art_id)
            for idx, chunk in enumerate(_chunk_text(md_text)):
                conn.execute(
                    "INSERT INTO doc_chunks(artifact_id, page_start, page_end, chunk_index, content) VALUES(?, NULL, NULL, ?, ?)",
                    (art_id, idx, chunk),
                )

            conn.execute("INSERT INTO reports(check_run_id, artifact_id, format) VALUES(?,?,?)", (int(check_run_id), art_id, "md"))
            outputs.append({"format": "md", "path": str(md_path), "uri": uri, "artifact_id": art_id})

        # PDF
        if "pdf" in formats:
            pdf_path = out_dir_path / f"check_run_{check_run_id}.pdf"
            build_pdf_report(str(pdf_path), meta, summary, items)

            sha = _sha256_file(pdf_path)
            uri = pdf_path.as_uri()
            art_id = _upsert_artifact(conn, uri=uri, kind="report_pdf", title=f"Report PDF #{check_run_id}", source_path=str(pdf_path), sha256=sha)

            conn.execute("INSERT INTO reports(check_run_id, artifact_id, format) VALUES(?,?,?)", (int(check_run_id), art_id, "pdf"))
            outputs.append({"format": "pdf", "path": str(pdf_path), "uri": uri, "artifact_id": art_id})

        conn.commit()
        return {"ok": True, "check_run_id": int(check_run_id), "outputs": outputs, "summary": summary}
    finally:
        conn.close()

5-4-7) 보고서 목록 도구(선택)
@mcp.tool()
def structai_report_list(check_run_id: Optional[int] = None, limit: int = 50) -> Dict[str, Any]:
    conn = _connect()
    try:
        if check_run_id is None:
            rows = conn.execute(
                """
                SELECT r.report_id, r.check_run_id, r.format, r.created_at, a.uri, a.title, a.kind
                FROM reports r
                JOIN artifacts a ON a.artifact_id = r.artifact_id
                ORDER BY r.report_id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT r.report_id, r.check_run_id, r.format, r.created_at, a.uri, a.title, a.kind
                FROM reports r
                JOIN artifacts a ON a.artifact_id = r.artifact_id
                WHERE r.check_run_id=?
                ORDER BY r.report_id DESC
                LIMIT ?
                """,
                (int(check_run_id), int(limit)),
            ).fetchall()
        return {"items": [dict(r) for r in rows]}
    finally:
        conn.close()

5-5) “실제 사용 순서” (Step 5 기준)

모델/결과/설계입력까지 준비 (Step 4와 동일)

structai_model_import_members

structai_results_import

structai_design_import_inputs (또는 수동 set)

내력 자동 산정 실행

RC 보:

#structai_design_compute_rc_beam_rect {
  "model_id": 1,
  "where": { "type": "beam" },
  "defaults": { "fc": 27, "fy": 400, "phi_flex": 0.9, "phi_shear": 0.75 },
  "overwrite": true
}


RC 기둥:

#structai_design_compute_rc_column_axial {
  "model_id": 1,
  "where": { "type": "column" },
  "defaults": { "fc": 27, "fy": 400, "phi_axial": 0.65 },
  "overwrite": true
}


강재 보:

#structai_design_compute_steel_beam_simple {
  "model_id": 1,
  "where": { "type": "beam" },
  "defaults": { "Fy": 325, "phi_flex": 0.9, "phi_shear": 1.0 },
  "overwrite": true
}


체크 실행

이제 rulepack의 capacity_expr가 Mn, Vn, Pn을 참조하면 자동으로 채워져 있음.

보고서 생성

#structai_report_generate {
  "check_run_id": 1,
  "formats": ["md", "pdf"],
  "include_pass": false
}

5-6) Results 패널에 “Generate Report” 버튼 붙이기 (권장)

너가 이미 만들어둔 ResultsView에 버튼 하나만 추가하면 사용성이 급상승해.

(1) media/results/main.js에 버튼 추가

버튼 row에 추가:

h("button", { onclick: () => generateReport() }, ["Generate Report"]),


그리고 함수 추가:

function generateReport() {
  const latestCheck = checks[0];
  if (!latestCheck) {
    alert("먼저 Run Checks를 실행해서 check run을 생성하세요.");
    return;
  }
  vscode.postMessage({ type: "generateReport", check_run_id: latestCheck.check_run_id });
}

(2) resultsView.ts message handler에 추가
case "generateReport": {
  const result = await invoke("structai_report_generate", {
    check_run_id: msg.check_run_id,
    formats: ["md", "pdf"],
    include_pass: false
  });
  // md는 VS Code에서 열기, pdf는 외부 열기
  const outs = result.outputs || [];
  const md = outs.find((o:any)=> o.format === "md");
  const pdf = outs.find((o:any)=> o.format === "pdf");

  if (md?.path) {
    const doc = await vscode.workspace.openTextDocument(md.path);
    await vscode.window.showTextDocument(doc, { preview: false });
  }
  if (pdf?.uri) {
    await vscode.env.openExternal(vscode.Uri.parse(pdf.uri));
  }

  view.webview.postMessage({ type: "reportDone", result });
  break;
}

5-7) 다음으로 확장할 때의 “정답 루트” (Step 6 예고)

이제 뼈대는 완성됐고, 다음은 정확도를 올리는 단계야.

KDS용 결정론 모듈을 체크 단위로 쪼개서 추가

rc_beam_kds.py, rc_column_kds.py, steel_kds.py 같은 식으로

“입력 스키마(필수 키)”를 고정하고, 누락이면 체크를 NA로 명확히 처리

단면/철근 자동 파싱

도면/모델에서 “B400x600 / D25-4EA / D10@150” 같은 표기 읽어서
As, Av, s, b, h 자동 생성
→ 이게 되면 사람이 입력해야 할 게 급격히 줄어듦

검증 하네스

“상용 프로그램 계산서/수기 계산서”와 결과를 비교하는 regression test 세트 구축
(이걸 안 하면 도구 신뢰성을 확보하기 어렵다)

원하면 다음 응답에서는 Step 6으로 바로 이어서,

“도면 텍스트에서 철근/단면 스펙 파싱(정규식 + 룰 기반)”

“부재 타입별 입력 스키마/검증(Validation)”

“KDS 조항-체크 타입 매핑(조항번호 → 룰셋 citations.query 자동화)”
까지 순서대로 붙여줄게.
