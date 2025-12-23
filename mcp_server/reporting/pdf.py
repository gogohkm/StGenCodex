# mcp_server/reporting/pdf.py
from __future__ import annotations

from typing import Any, Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


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
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Structural Check Report",
        author="StructAI",
    )

    story: List[Any] = []

    story.append(Paragraph("Structural Check Report", styles["Title"]))
    story.append(Spacer(1, 6 * mm))

    meta_lines = [
        f"Check run: #{meta.get('check_run_id')} - {meta.get('check_run_name')}",
        f"Model: {meta.get('model_name')} (id={meta.get('model_id')})",
        f"Analysis run: {meta.get('analysis_run_name')} (id={meta.get('analysis_run_id')})",
        f"Rulepack: {meta.get('rulepack_name')}/{meta.get('rulepack_version')}",
    ]
    story.append(Paragraph("<br/>".join(meta_lines), styles["Normal"]))
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Summary", styles["H2"]))
    sum_table = Table(
        [
            ["PASS", summary.get("PASS", 0), "WARN", summary.get("WARN", 0)],
            ["FAIL", summary.get("FAIL", 0), "NA", summary.get("NA", 0)],
        ],
        colWidths=[22 * mm, 18 * mm, 22 * mm, 18 * mm],
    )
    sum_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.black),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("ALIGN", (3, 0), (3, -1), "RIGHT"),
            ]
        )
    )
    story.append(sum_table)
    story.append(Spacer(1, 6 * mm))

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

        rows.append(
            [
                r.get("status", ""),
                r.get("member_label", "") or "",
                r.get("member_uid", "") or "",
                r.get("check_type", "") or "",
                r.get("combo", "") or "",
                fmt(r.get("demand_value")),
                fmt(r.get("capacity_value")),
                fmt(r.get("ratio")),
                cite_cell,
            ]
        )

    tbl = Table(rows, colWidths=[16 * mm, 18 * mm, 18 * mm, 26 * mm, 16 * mm, 16 * mm, 18 * mm, 14 * mm, 40 * mm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.black),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (5, 1), (7, -1), "RIGHT"),
            ]
        )
    )
    story.append(tbl)

    story.append(PageBreak())
    story.append(Paragraph("Details (FAIL/WARN)", styles["H2"]))

    for r in items:
        if r.get("status") not in ("FAIL", "WARN"):
            continue

        title = f"{r.get('status')} - {r.get('member_label','')} ({r.get('member_uid','')}) - {r.get('check_type','')} / {r.get('combo','')}"
        story.append(Paragraph(title, styles["Normal"]))
        story.append(Spacer(1, 2 * mm))

        story.append(Paragraph(f"Demand: {r.get('demand_value')}", styles["Small"]))
        story.append(Paragraph(f"Capacity: {r.get('capacity_value')}", styles["Small"]))
        story.append(Paragraph(f"Ratio: {r.get('ratio')}", styles["Small"]))
        story.append(Spacer(1, 2 * mm))

        cites = r.get("citations") or []
        if cites:
            story.append(Paragraph("Citations:", styles["Small"]))
            for c in cites[:3]:
                uri = c.get("cite_uri") or c.get("uri") or ""
                snip = (c.get("snippet") or "").strip()
                story.append(Paragraph(f"- {uri}", styles["Small"]))
                if snip:
                    story.append(Paragraph(f"&nbsp;&nbsp;{snip}", styles["Small"]))
            story.append(Spacer(1, 2 * mm))

        story.append(Spacer(1, 4 * mm))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
