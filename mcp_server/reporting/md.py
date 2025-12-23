# mcp_server/reporting/md.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def build_markdown_report(meta: Dict[str, Any], summary: Dict[str, int], items: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Structural Check Report")
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

        lines.append(
            f"### {r.get('status')} - {r.get('member_label','')} ({r.get('member_uid','')}) - {r.get('check_type','')} / {r.get('combo','')}"
        )
        lines.append("")
        lines.append(f"- Demand: {r.get('demand_value')}")
        lines.append(f"- Capacity: {r.get('capacity_value')}")
        lines.append(f"- Ratio: {r.get('ratio')}")
        lines.append("")

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
