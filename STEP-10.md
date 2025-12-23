좋아. 이제 10단계는 “현장 적용 레벨로 마무리”하는 단계야.
너가 요청한 3가지를 순서대로, 바로 구현 가능한 형태(DB/서버(MCP)/리포트/VS Code UI/되돌리기 포함)로 제시할게.

10-1) 이번 단계에서 완성되는 것
A. 상용 프로그램/수기 계산서 비교 자동화(케이스별 리포트)

외부 결과(ETABS/MIDAS/STAAD/수기 등)를 Benchmark 데이터셋으로 DB에 적재

우리 엔진의 check_run 결과와 자동 비교

차이(Δratio/Δdemand/Δcapacity) 정리한 Diff Report(MD/PDF) 자동 생성

B. 룰셋/KDS 조항 적용을 “체크 타입별 템플릿”으로 표준화

프로젝트마다 rulepack JSON을 그때그때 편집하는 방식에서 벗어나

Template Set(체크 타입별 표준 템플릿) + Codebook(KDS 조항 매핑) 조합으로

rulepack을 항상 일관되게 생성

요구변수/표현식/조항 인용 구조 표준화

C. VS Code 패널에서 “충돌/누락/확정” 워크플로우 완성

Suggested/Conflict 목록 → 한 화면에서 확인/확정/거절/되돌리기

스펙 적용은 Patch Run으로 기록 → 언제든 rollback 가능

근거(evidence)와 CAD context도 같이 노출

10-2) DB 스키마 추가(v0.1.1)

mcp_server/schema.sql 맨 아래에 추가해줘.

-- ===== schema patch v0.1.1 =====
UPDATE meta SET value='0.1.1' WHERE key='schema_version';

-- -----------------------------
-- Benchmark (external baseline)
-- -----------------------------
CREATE TABLE IF NOT EXISTS benchmarks (
  benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'commercial',      -- commercial|manual|other
  source TEXT,                                 -- ETABS|MIDAS|HandCalc|...
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS benchmark_results (
  benchmark_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  benchmark_id INTEGER NOT NULL REFERENCES benchmarks(benchmark_id) ON DELETE CASCADE,

  member_uid TEXT NOT NULL,
  story_norm TEXT,                             -- optional
  check_type TEXT NOT NULL,
  combo TEXT NOT NULL,

  demand_value REAL,
  capacity_value REAL,
  ratio REAL,
  status TEXT,                                 -- optional

  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),

  UNIQUE(benchmark_id, member_uid, COALESCE(story_norm,''), check_type, combo)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_results_lookup
ON benchmark_results(benchmark_id, member_uid, story_norm, check_type, combo);

-- -----------------------------
-- Compare runs (check_run vs benchmark)
-- -----------------------------
CREATE TABLE IF NOT EXISTS compare_runs (
  compare_id INTEGER PRIMARY KEY AUTOINCREMENT,
  check_run_id INTEGER NOT NULL REFERENCES check_runs(check_run_id) ON DELETE CASCADE,
  benchmark_id INTEGER NOT NULL REFERENCES benchmarks(benchmark_id) ON DELETE CASCADE,

  name TEXT,
  ratio_tol REAL NOT NULL DEFAULT 0.01,
  ratio_warn REAL NOT NULL DEFAULT 0.03,

  summary_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS compare_items (
  compare_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
  compare_id INTEGER NOT NULL REFERENCES compare_runs(compare_id) ON DELETE CASCADE,

  model_member_id INTEGER REFERENCES model_members(model_member_id) ON DELETE SET NULL,
  member_uid TEXT,
  story_norm TEXT,

  check_type TEXT NOT NULL,
  combo TEXT NOT NULL,

  expected_ratio REAL,
  actual_ratio REAL,
  abs_diff REAL,
  rel_diff REAL,

  severity TEXT NOT NULL,                       -- OK|WARN|DIFF|MISSING_EXPECTED|MISSING_ACTUAL
  note TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_compare_items_sev
ON compare_items(compare_id, severity);

CREATE TABLE IF NOT EXISTS compare_reports (
  compare_report_id INTEGER PRIMARY KEY AUTOINCREMENT,
  compare_id INTEGER NOT NULL REFERENCES compare_runs(compare_id) ON DELETE CASCADE,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  format TEXT NOT NULL,                          -- md|pdf
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- -----------------------------
-- Check template sets
-- -----------------------------
CREATE TABLE IF NOT EXISTS check_template_sets (
  template_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  templates_json TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(name, version)
);

-- -----------------------------
-- Patch log for undo/redo
-- -----------------------------
CREATE TABLE IF NOT EXISTS design_patch_runs (
  patch_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  cad_artifact_id INTEGER REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
  note TEXT,
  params_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS design_patch_items (
  patch_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
  patch_run_id INTEGER NOT NULL REFERENCES design_patch_runs(patch_run_id) ON DELETE CASCADE,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,

  before_json TEXT NOT NULL,
  after_json TEXT NOT NULL,
  changed_keys_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_design_patch_items_run
ON design_patch_items(patch_run_id);

-- -----------------------------
-- Decision log (confirm/reject audits)
-- -----------------------------
CREATE TABLE IF NOT EXISTS decision_logs (
  decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity_type TEXT NOT NULL,                     -- member_spec_links|token_story_maps|member_mappings
  entity_id INTEGER NOT NULL,
  from_status TEXT,
  to_status TEXT,
  reason TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

10-3) A. 상용 프로그램/수기 비교 자동화
10-3-1) Benchmark 파일 포맷(권장)
CSV (가장 쉬움)
member_uid,story,check_type,combo,demand,capacity,ratio,status,source_note
B12,3F,strength.flexure,LC1,120.5,130.0,0.927,PASS,ETABS design comb LC1
B12,3F,strength.shear,LC1,80.0,75.0,1.067,FAIL,ETABS


story는 없어도 되지만, 다층 동일 라벨이 있으면 꼭 넣는 게 정답

ratio가 없으면 demand/capacity로 계산 가능

JSON
{
  "name": "ETABS",
  "version": "2025-01-Case01",
  "items": [
    {"member_uid":"B12","story":"3F","check_type":"strength.flexure","combo":"LC1","ratio":0.927},
    {"member_uid":"B12","story":"3F","check_type":"strength.shear","combo":"LC1","demand":80,"capacity":75}
  ]
}

10-3-2) MCP Tool: benchmark import

server.py에 추가(상단에 import csv 포함):

@mcp.tool()
def structai_benchmark_import(
    path: str,
    name: str,
    version: str,
    kind: str = "commercial",
    source: str = "",
    fmt: Optional[str] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    외부(상용/수기) benchmark 결과를 DB에 적재
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower()
    if fmt is None:
        fmt = "json" if ext == ".json" else "csv"

    from mcp_server.parsing.story import normalize_story

    conn = _connect()
    try:
        # upsert benchmark header
        conn.execute(
            """
            INSERT INTO benchmarks(name, version, kind, source, meta_json)
            VALUES(?,?,?,?, '{}')
            ON CONFLICT(name, version) DO UPDATE SET
              kind=excluded.kind,
              source=excluded.source
            """,
            (name, version, kind, source),
        )
        b = conn.execute("SELECT benchmark_id FROM benchmarks WHERE name=? AND version=?", (name, version)).fetchone()
        benchmark_id = int(b["benchmark_id"])

        if overwrite:
            conn.execute("DELETE FROM benchmark_results WHERE benchmark_id=?", (benchmark_id,))

        inserted = 0
        skipped = 0

        def insert_item(it: Dict[str, Any]):
            nonlocal inserted, skipped
            member_uid = str(it.get("member_uid") or "").strip()
            if not member_uid:
                skipped += 1
                return
            story = it.get("story")
            story_norm = normalize_story(str(story)) if story else None
            check_type = str(it.get("check_type") or "").strip()
            combo = str(it.get("combo") or "").strip()
            if not check_type or not combo:
                skipped += 1
                return

            demand = it.get("demand")
            cap = it.get("capacity")
            ratio = it.get("ratio")
            status = it.get("status")

            # ratio 계산(가능할 때)
            try:
                if ratio in (None, "", 0) and demand not in (None, "") and cap not in (None, "", 0):
                    ratio = float(demand) / float(cap)
            except Exception:
                pass

            conn.execute(
                """
                INSERT INTO benchmark_results(
                  benchmark_id, member_uid, story_norm, check_type, combo,
                  demand_value, capacity_value, ratio, status, meta_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(benchmark_id, member_uid, COALESCE(story_norm,''), check_type, combo)
                DO UPDATE SET
                  demand_value=excluded.demand_value,
                  capacity_value=excluded.capacity_value,
                  ratio=excluded.ratio,
                  status=excluded.status,
                  meta_json=excluded.meta_json
                """,
                (
                    benchmark_id,
                    member_uid,
                    story_norm,
                    check_type,
                    combo,
                    float(demand) if demand not in (None, "") else None,
                    float(cap) if cap not in (None, "") else None,
                    float(ratio) if ratio not in (None, "") else None,
                    str(status) if status not in (None, "") else None,
                    json.dumps({k:v for k,v in it.items() if k not in ("member_uid","story","check_type","combo","demand","capacity","ratio","status")}, ensure_ascii=False),
                ),
            )
            inserted += 1

        if fmt == "csv":
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    insert_item({
                        "member_uid": row.get("member_uid"),
                        "story": row.get("story"),
                        "check_type": row.get("check_type"),
                        "combo": row.get("combo"),
                        "demand": row.get("demand"),
                        "capacity": row.get("capacity"),
                        "ratio": row.get("ratio"),
                        "status": row.get("status"),
                        "source_note": row.get("source_note"),
                    })

        elif fmt == "json":
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            items = data.get("items") if isinstance(data, dict) else data
            if not isinstance(items, list):
                raise ValueError("JSON must be a list or {items:[...]} format")
            for it in items:
                insert_item(it)
        else:
            raise ValueError("fmt must be csv or json")

        conn.commit()
        return {"ok": True, "benchmark_id": benchmark_id, "inserted": inserted, "skipped": skipped}
    finally:
        conn.close()

10-3-3) MCP Tool: check_run vs benchmark 비교 실행
@mcp.tool()
def structai_compare_check_run_to_benchmark(
    check_run_id: int,
    benchmark_id: int,
    name: str = "",
    ratio_tol: float = 0.01,
    ratio_warn: float = 0.03
) -> Dict[str, Any]:
    """
    check_results(check_run_id) vs benchmark_results(benchmark_id) 비교 후 compare_runs/compare_items 저장
    """
    from mcp_server.parsing.story import normalize_story

    conn = _connect()
    try:
        # create compare run
        cur = conn.execute(
            """
            INSERT INTO compare_runs(check_run_id, benchmark_id, name, ratio_tol, ratio_warn, summary_json)
            VALUES(?,?,?,?,?, '{}')
            """,
            (int(check_run_id), int(benchmark_id), name or f"compare_{check_run_id}_{benchmark_id}", float(ratio_tol), float(ratio_warn)),
        )
        compare_id = int(cur.lastrowid)

        # load actual
        actual_rows = conn.execute(
            """
            SELECT
              mb.model_member_id, mb.member_uid, mb.story,
              cr.check_type, cr.combo, cr.ratio, cr.status
            FROM check_results cr
            JOIN model_members mb ON mb.model_member_id = cr.model_member_id
            WHERE cr.check_run_id=?
            """,
            (int(check_run_id),),
        ).fetchall()

        # build expected index
        exp_rows = conn.execute(
            """
            SELECT member_uid, story_norm, check_type, combo, ratio
            FROM benchmark_results
            WHERE benchmark_id=?
            """,
            (int(benchmark_id),),
        ).fetchall()

        exp = {}
        for r in exp_rows:
            key = (str(r["member_uid"]), str(r["story_norm"] or ""), str(r["check_type"]), str(r["combo"]))
            exp[key] = float(r["ratio"]) if r["ratio"] is not None else None

        used_expected_keys = set()

        counts = {"OK":0, "WARN":0, "DIFF":0, "MISSING_EXPECTED":0, "MISSING_ACTUAL":0}
        top_diffs = []

        # compare actual -> expected
        for r in actual_rows:
            member_uid = str(r["member_uid"])
            story_norm = normalize_story(str(r["story"])) or (str(r["story"]).strip().upper() if r["story"] else "")
            key = (member_uid, story_norm, str(r["check_type"]), str(r["combo"]))
            # fallback: expected may not have story
            key2 = (member_uid, "", str(r["check_type"]), str(r["combo"]))

            expected_ratio = exp.get(key, None)
            expected_key_used = key
            if expected_ratio is None:
                expected_ratio = exp.get(key2, None)
                expected_key_used = key2

            actual_ratio = float(r["ratio"]) if r["ratio"] is not None else None

            if expected_ratio is None:
                severity = "MISSING_EXPECTED"
                counts[severity] += 1
                conn.execute(
                    """
                    INSERT INTO compare_items(compare_id, model_member_id, member_uid, story_norm, check_type, combo,
                                              expected_ratio, actual_ratio, abs_diff, rel_diff, severity, note)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (compare_id, int(r["model_member_id"]), member_uid, story_norm, r["check_type"], r["combo"],
                     None, actual_ratio, None, None, severity, "no benchmark record"),
                )
                continue

            used_expected_keys.add(expected_key_used)

            if actual_ratio is None:
                severity = "MISSING_ACTUAL"
                counts[severity] += 1
                conn.execute(
                    """
                    INSERT INTO compare_items(compare_id, model_member_id, member_uid, story_norm, check_type, combo,
                                              expected_ratio, actual_ratio, abs_diff, rel_diff, severity, note)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (compare_id, int(r["model_member_id"]), member_uid, story_norm, r["check_type"], r["combo"],
                     expected_ratio, None, None, None, severity, "no actual ratio"),
                )
                continue

            abs_diff = actual_ratio - expected_ratio
            rel_diff = abs_diff / expected_ratio if expected_ratio not in (None, 0) else None

            ad = abs(abs_diff)
            if ad <= ratio_tol:
                severity = "OK"
            elif ad <= ratio_warn:
                severity = "WARN"
            else:
                severity = "DIFF"
            counts[severity] += 1

            conn.execute(
                """
                INSERT INTO compare_items(compare_id, model_member_id, member_uid, story_norm, check_type, combo,
                                          expected_ratio, actual_ratio, abs_diff, rel_diff, severity, note)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (compare_id, int(r["model_member_id"]), member_uid, story_norm, r["check_type"], r["combo"],
                 expected_ratio, actual_ratio, abs_diff, rel_diff, severity, ""),
            )

            if severity in ("DIFF","WARN"):
                top_diffs.append({
                    "member_uid": member_uid,
                    "story_norm": story_norm,
                    "check_type": r["check_type"],
                    "combo": r["combo"],
                    "expected_ratio": expected_ratio,
                    "actual_ratio": actual_ratio,
                    "abs_diff": abs_diff
                })

        # expected-only entries (missing actual rows)
        for key, expected_ratio in exp.items():
            if key in used_expected_keys:
                continue
            member_uid, story_norm, check_type, combo = key
            counts["MISSING_ACTUAL"] += 1
            conn.execute(
                """
                INSERT INTO compare_items(compare_id, member_uid, story_norm, check_type, combo,
                                          expected_ratio, actual_ratio, abs_diff, rel_diff, severity, note)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (compare_id, member_uid, story_norm, check_type, combo,
                 expected_ratio, None, None, None, "MISSING_ACTUAL", "missing in actual check_run"),
            )

        # summary + store
        top_diffs_sorted = sorted(top_diffs, key=lambda x: abs(x["abs_diff"]), reverse=True)[:20]
        summary = {"counts": counts, "top_diffs": top_diffs_sorted, "ratio_tol": ratio_tol, "ratio_warn": ratio_warn}
        conn.execute("UPDATE compare_runs SET summary_json=? WHERE compare_id=?", (json.dumps(summary, ensure_ascii=False), compare_id))

        conn.commit()
        return {"ok": True, "compare_id": compare_id, "summary": summary}
    finally:
        conn.close()

10-3-4) Diff Report 생성(MD/PDF)

mcp_server/reporting/에 새 파일 2개 추가하는 걸 추천:

compare_md.py

compare_pdf.py

하지만 빠르게 붙이려면 tool 안에서 MD 텍스트를 만들고, reportlab로 PDF를 만드는 게 가장 간단해. 아래는 “tool 한 방” 버전이야.

@mcp.tool()
def structai_compare_report_generate(
    compare_id: int,
    formats: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    limit_items: int = 500
) -> Dict[str, Any]:
    formats = formats or ["md","pdf"]
    out_dir_path = Path(out_dir).expanduser().resolve() if out_dir else (DB_PATH.parent / "compare_reports")
    out_dir_path.mkdir(parents=True, exist_ok=True)

    conn = _connect()
    try:
        cr = conn.execute(
            "SELECT compare_id, check_run_id, benchmark_id, name, ratio_tol, ratio_warn, summary_json, created_at FROM compare_runs WHERE compare_id=?",
            (int(compare_id),),
        ).fetchone()
        if not cr:
            raise ValueError("compare_run not found")

        chk = conn.execute("SELECT name FROM check_runs WHERE check_run_id=?", (int(cr["check_run_id"]),)).fetchone()
        bm = conn.execute("SELECT name, version, source FROM benchmarks WHERE benchmark_id=?", (int(cr["benchmark_id"]),)).fetchone()

        summary = json.loads(cr["summary_json"] or "{}")

        items = conn.execute(
            """
            SELECT severity, member_uid, story_norm, check_type, combo,
                   expected_ratio, actual_ratio, abs_diff, rel_diff, note
            FROM compare_items
            WHERE compare_id=?
            ORDER BY
              CASE severity WHEN 'DIFF' THEN 0 WHEN 'WARN' THEN 1 WHEN 'MISSING_EXPECTED' THEN 2 WHEN 'MISSING_ACTUAL' THEN 3 ELSE 9 END,
              abs(abs_diff) DESC
            LIMIT ?
            """,
            (int(compare_id), int(limit_items)),
        ).fetchall()
        items = [dict(r) for r in items]

        outputs = []

        # ---- Markdown build ----
        def fmt(x):
            if x is None: return ""
            try: return f"{float(x):.4f}"
            except Exception: return str(x)

        md_lines = []
        md_lines.append("# Compare Report (Check Run vs Benchmark)")
        md_lines.append("")
        md_lines.append(f"- Compare ID: {compare_id}")
        md_lines.append(f"- Check run: #{cr['check_run_id']} - {(chk['name'] if chk else '')}")
        md_lines.append(f"- Benchmark: #{cr['benchmark_id']} - {(bm['name'] if bm else '')}/{(bm['version'] if bm else '')} ({(bm['source'] if bm else '')})")
        md_lines.append(f"- ratio_tol: {cr['ratio_tol']}, ratio_warn: {cr['ratio_warn']}")
        md_lines.append("")

        md_lines.append("## Summary")
        md_lines.append("")
        counts = (summary.get("counts") or {})
        for k in ("OK","WARN","DIFF","MISSING_EXPECTED","MISSING_ACTUAL"):
            md_lines.append(f"- {k}: {counts.get(k,0)}")
        md_lines.append("")

        md_lines.append("## Top Diffs")
        md_lines.append("")
        md_lines.append("| Severity | UID | Story | Check | Combo | Expected | Actual | Δ |")
        md_lines.append("|---|---|---|---|---|---:|---:|---:|")
        for r in items[:200]:
            md_lines.append(
                f"| {r['severity']} | {r.get('member_uid','')} | {r.get('story_norm','')} | {r.get('check_type','')} | {r.get('combo','')} | "
                f"{fmt(r.get('expected_ratio'))} | {fmt(r.get('actual_ratio'))} | {fmt(r.get('abs_diff'))} |"
            )

        md_text = "\n".join(md_lines)

        if "md" in formats:
            md_path = out_dir_path / f"compare_{compare_id}.md"
            md_path.write_text(md_text, encoding="utf-8")
            sha = _sha256_file(md_path)
            uri = md_path.as_uri()
            art_id = _upsert_artifact(conn, uri=uri, kind="compare_report_md", title=f"Compare Report MD #{compare_id}", source_path=str(md_path), sha256=sha)
            conn.execute("INSERT INTO compare_reports(compare_id, artifact_id, format) VALUES(?,?,?)", (int(compare_id), art_id, "md"))
            outputs.append({"format":"md","path":str(md_path),"uri":uri,"artifact_id":art_id})

        if "pdf" in formats:
            # reuse Step5 pdf builder style: quick and simple
            pdf_path = out_dir_path / f"compare_{compare_id}.pdf"
            # minimal PDF generator
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import mm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet

            styles = getSampleStyleSheet()
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=16*mm, bottomMargin=16*mm)
            story = []
            story.append(Paragraph("Compare Report (Check Run vs Benchmark)", styles["Title"]))
            story.append(Spacer(1, 4*mm))
            story.append(Paragraph(f"Compare ID: {compare_id}", styles["Normal"]))
            story.append(Paragraph(f"Check run: #{cr['check_run_id']} - {(chk['name'] if chk else '')}", styles["Normal"]))
            story.append(Paragraph(f"Benchmark: #{cr['benchmark_id']} - {(bm['name'] if bm else '')}/{(bm['version'] if bm else '')}", styles["Normal"]))
            story.append(Spacer(1, 4*mm))

            # summary table
            sum_rows = [["OK", counts.get("OK",0), "WARN", counts.get("WARN",0)],
                        ["DIFF", counts.get("DIFF",0), "MISSING_EXPECTED", counts.get("MISSING_EXPECTED",0)],
                        ["MISSING_ACTUAL", counts.get("MISSING_ACTUAL",0), "", ""]]
            tbl = Table(sum_rows, colWidths=[30*mm, 15*mm, 45*mm, 15*mm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1), colors.whitesmoke),
                ("BOX",(0,0),(-1,-1), 0.6, colors.black),
                ("INNERGRID",(0,0),(-1,-1), 0.3, colors.grey),
            ]))
            story.append(Paragraph("Summary", styles["Heading2"]))
            story.append(tbl)
            story.append(Spacer(1, 5*mm))

            # top diffs
            rows = [["Sev","UID","Story","Check","Combo","Exp","Act","Δ"]]
            for r in items[:80]:
                rows.append([r["severity"], r.get("member_uid",""), r.get("story_norm",""), r.get("check_type",""), r.get("combo",""),
                             fmt(r.get("expected_ratio")), fmt(r.get("actual_ratio")), fmt(r.get("abs_diff"))])
            dt = Table(rows, colWidths=[14*mm, 18*mm, 12*mm, 30*mm, 14*mm, 14*mm, 14*mm, 14*mm])
            dt.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
                ("FONT",(0,0),(-1,0), "Helvetica-Bold", 8),
                ("FONT",(0,1),(-1,-1), "Helvetica", 7),
                ("BOX",(0,0),(-1,-1), 0.6, colors.black),
                ("INNERGRID",(0,0),(-1,-1), 0.25, colors.grey),
            ]))
            story.append(Paragraph("Top diffs", styles["Heading2"]))
            story.append(dt)
            doc.build(story)

            sha = _sha256_file(pdf_path)
            uri = pdf_path.as_uri()
            art_id = _upsert_artifact(conn, uri=uri, kind="compare_report_pdf", title=f"Compare Report PDF #{compare_id}", source_path=str(pdf_path), sha256=sha)
            conn.execute("INSERT INTO compare_reports(compare_id, artifact_id, format) VALUES(?,?,?)", (int(compare_id), art_id, "pdf"))
            outputs.append({"format":"pdf","path":str(pdf_path),"uri":uri,"artifact_id":art_id})

        conn.commit()
        return {"ok": True, "compare_id": compare_id, "outputs": outputs, "summary": summary}
    finally:
        conn.close()

10-3-5) 비교 자동화 실제 실행 순서

외부 benchmark 임포트

#structai_benchmark_import {
  "path": "fixtures/etabs_case01.csv",
  "name": "ETABS",
  "version": "case01",
  "source": "ETABS",
  "overwrite": true
}


check_run 결과와 비교

#structai_compare_check_run_to_benchmark {
  "check_run_id": 1,
  "benchmark_id": 1,
  "ratio_tol": 0.01,
  "ratio_warn": 0.03
}


Diff Report 생성(MD/PDF)

#structai_compare_report_generate {
  "compare_id": 1,
  "formats": ["md","pdf"]
}

10-4) B. 룰셋/KDS를 “체크 템플릿”으로 표준화
10-4-1) Template Set JSON 포맷(추천)

파일 예: templates_kds_rc_core.json

{
  "name": "kds-rc-core",
  "version": "2025.01",
  "checks": {
    "strength.flexure": {
      "ratio_expr": "max(abs(M3_max)/Mn_pos, abs(M3_min)/Mn_neg)",
      "limit": 1.0,
      "warn": 0.95,
      "requires": {
        "env": ["M3_max", "M3_min"],
        "design": ["Mn_pos", "Mn_neg"]
      },
      "citations_key": "strength.flexure"
    },
    "strength.shear": {
      "ratio_expr": "abs(V2_max)/Vn",
      "limit": 1.0,
      "warn": 0.95,
      "requires": { "env": ["V2_max"], "design": ["Vn"] },
      "citations_key": "strength.shear"
    }
  }
}


requires는 Step6의 validation을 더 정확하게 만들기 위한 “표준 요구변수”

citations_key는 Codebook의 citations 매핑 키로 쓰임(조항 변경/버전 변경을 codebook만 바꿔서 해결)

10-4-2) MCP Tool: 템플릿 임포트/활성화/룰팩 생성
(1) 템플릿 임포트
@mcp.tool()
def structai_templates_import(path: str) -> Dict[str, Any]:
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
            INSERT INTO check_template_sets(name, version, templates_json, is_active)
            VALUES(?,?,?,0)
            ON CONFLICT(name,version) DO UPDATE SET
              templates_json=excluded.templates_json
            """,
            (name, ver, json.dumps(data, ensure_ascii=False)),
        )
        conn.commit()
        return {"ok": True, "name": name, "version": ver}
    finally:
        conn.close()

(2) 템플릿 활성화
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

(3) 템플릿 + codebook → rulepack 생성
@mcp.tool()
def structai_rulepack_generate_from_templates(
    new_rulepack_name: str,
    new_rulepack_version: str,
    activate: bool = True
) -> Dict[str, Any]:
    conn = _connect()
    try:
        # load active template set
        ts = conn.execute(
            "SELECT templates_json FROM check_template_sets WHERE is_active=1 ORDER BY template_set_id DESC LIMIT 1"
        ).fetchone()
        if not ts:
            raise ValueError("No active template set")

        tpl = json.loads(ts["templates_json"])
        checks_tpl = tpl.get("checks") or {}

        # load active codebook (Step6/8에서 만든 codebooks)
        cb = conn.execute(
            "SELECT codebook_json FROM codebooks WHERE is_active=1 ORDER BY codebook_id DESC LIMIT 1"
        ).fetchone()
        if not cb:
            raise ValueError("No active codebook")
        codebook = json.loads(cb["codebook_json"])
        cite_map = codebook.get("citations") or {}

        # build rulepack checks
        checks = {}
        for ct, cd in checks_tpl.items():
            out = dict(cd)
            cite_key = out.pop("citations_key", None)
            # 템플릿의 requires는 룰팩엔 남겨도 되고(검증용), 제거해도 됨
            # 여기서는 남겨서 validation에서 활용하도록 유지
            if cite_key and cite_key in cite_map:
                out["citations"] = cite_map[cite_key]
            checks[ct] = out

        rp = {
            "name": new_rulepack_name,
            "version": new_rulepack_version,
            "checks": checks
        }

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
            rid = conn.execute("SELECT rulepack_id FROM rulepacks WHERE name=? AND version=?", (new_rulepack_name, new_rulepack_version)).fetchone()
            if rid:
                conn.execute("UPDATE rulepacks SET is_active=0")
                conn.execute("UPDATE rulepacks SET is_active=1 WHERE rulepack_id=?", (int(rid["rulepack_id"]),))
        conn.commit()

        return {"ok": True, "rulepack": {"name": new_rulepack_name, "version": new_rulepack_version}, "activated": activate, "checks": list(checks.keys())}
    finally:
        conn.close()

10-5) C. VS Code “Resolve(확정/충돌/누락)” 패널 완성

여기서는 기능 5개만 제대로 잡으면 실무 워크플로우가 완성돼.

Suggested Spec Links 목록 보기

Confirm / Reject / Undo(되돌리기)

token-story conflicts 보기 + 확정

missing inputs(왜 NA인지) 보기

Apply Specs가 만든 patch_run 롤백

이를 위해 서버에 “상태 변경/목록 조회/패치 롤백” 도구를 추가하고, VS Code 웹뷰에서 호출하면 된다.

10-5-1) 서버 도구: spec link 목록/상태 변경/감사로그
(1) spec links list
@mcp.tool()
def structai_specs_list_links(
    cad_artifact_id: int,
    model_id: int,
    status: str = "suggested",
    limit: int = 200
) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT
              l.link_id, l.model_member_id, mb.member_uid, mb.member_label, mb.story, mb.type,
              l.spec_id, s.spec_kind, s.raw_text, s.spec_json, s.confidence,
              l.method, l.status, l.distance, l.evidence_json, l.updated_at
            FROM member_spec_links l
            JOIN model_members mb ON mb.model_member_id = l.model_member_id
            JOIN cad_specs s ON s.spec_id = l.spec_id
            WHERE l.cad_artifact_id=? AND l.model_id=? AND l.status=?
            ORDER BY
              CASE l.method WHEN 'table_schema' THEN 0 WHEN 'table' THEN 1 WHEN 'leader' THEN 2 ELSE 9 END,
              COALESCE(l.distance, 999999) ASC,
              s.confidence DESC
            LIMIT ?
            """,
            (int(cad_artifact_id), int(model_id), str(status), int(limit)),
        ).fetchall()

        items = []
        for r in rows:
            it = dict(r)
            it["spec"] = json.loads(it.pop("spec_json") or "{}")
            it["evidence"] = json.loads(it.pop("evidence_json") or "{}")
            items.append(it)
        return {"ok": True, "items": items}
    finally:
        conn.close()

(2) link status set + decision log
@mcp.tool()
def structai_specs_set_link_status(
    link_id: int,
    to_status: str,
    reason: str = ""
) -> Dict[str, Any]:
    if to_status not in ("suggested","confirmed","rejected"):
        raise ValueError("to_status must be suggested|confirmed|rejected")

    conn = _connect()
    try:
        row = conn.execute("SELECT status FROM member_spec_links WHERE link_id=?", (int(link_id),)).fetchone()
        if not row:
            raise ValueError("link not found")
        from_status = row["status"]

        conn.execute(
            "UPDATE member_spec_links SET status=?, updated_at=datetime('now') WHERE link_id=?",
            (to_status, int(link_id)),
        )
        conn.execute(
            """
            INSERT INTO decision_logs(entity_type, entity_id, from_status, to_status, reason, meta_json)
            VALUES(?,?,?,?,?, '{}')
            """,
            ("member_spec_links", int(link_id), from_status, to_status, reason),
        )
        conn.commit()
        return {"ok": True, "link_id": int(link_id), "from": from_status, "to": to_status}
    finally:
        conn.close()

10-5-2) 서버 도구: patch run 기록 + 롤백
(1) structai_design_apply_specs_to_inputs에 patch run 기록 추가(중요)

Step6에서 만들었던 structai_design_apply_specs_to_inputs를 이 방식으로 업그레이드해:

시작 시 design_patch_runs에 1행 생성 → patch_run_id

각 member마다 before/after JSON 저장 + changed_keys 기록

결과로 patch_run_id 반환

핵심 패치 아이디어(코드 조각):

# apply 시작 전에
cur = conn.execute(
  "INSERT INTO design_patch_runs(model_id, cad_artifact_id, note, params_json) VALUES(?,?,?,?)",
  (int(model_id), int(cad_artifact_id), "apply_specs_to_inputs", json.dumps({"overwrite_keys": overwrite_keys}, ensure_ascii=False))
)
patch_run_id = int(cur.lastrowid)

# member 처리에서 before/after 저장
before = json.dumps(dj, ensure_ascii=False)
after  = json.dumps(merged, ensure_ascii=False)
changed_keys = sorted(list(set(patch.keys())))  # 실제로는 덮어쓴 키 기준으로 정교화 가능
conn.execute(
  "INSERT INTO design_patch_items(patch_run_id, model_member_id, before_json, after_json, changed_keys_json) VALUES(?,?,?,?,?)",
  (patch_run_id, mmid, before, after, json.dumps(changed_keys, ensure_ascii=False))
)


그리고 최종 반환에 patch_run_id 넣기:

return {"ok": True, "patch_run_id": patch_run_id, ...}

(2) patch runs list
@mcp.tool()
def structai_design_list_patch_runs(model_id: int, limit: int = 30) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT patch_run_id, cad_artifact_id, note, params_json, created_at
            FROM design_patch_runs
            WHERE model_id=?
            ORDER BY patch_run_id DESC
            LIMIT ?
            """,
            (int(model_id), int(limit)),
        ).fetchall()
        items = []
        for r in rows:
            it = dict(r)
            it["params"] = json.loads(it.pop("params_json") or "{}")
            items.append(it)
        return {"ok": True, "items": items}
    finally:
        conn.close()

(3) rollback tool (keys_only 기본)
@mcp.tool()
def structai_design_rollback_patch(
    patch_run_id: int,
    mode: str = "keys_only"   # keys_only | hard
) -> Dict[str, Any]:
    if mode not in ("keys_only","hard"):
        raise ValueError("mode must be keys_only|hard")

    conn = _connect()
    try:
        items = conn.execute(
            "SELECT model_member_id, before_json, changed_keys_json FROM design_patch_items WHERE patch_run_id=?",
            (int(patch_run_id),),
        ).fetchall()

        restored = 0
        for it in items:
            mmid = int(it["model_member_id"])
            before = json.loads(it["before_json"] or "{}")
            changed_keys = json.loads(it["changed_keys_json"] or "[]")

            cur = conn.execute("SELECT design_json FROM member_design_inputs WHERE model_member_id=?", (mmid,)).fetchone()
            now = json.loads(cur["design_json"]) if cur else {}

            if mode == "hard":
                merged = before
            else:
                merged = dict(now)
                for k in changed_keys:
                    if k in before:
                        merged[k] = before[k]
                    else:
                        # 원래 없던 키면 제거
                        if k in merged:
                            del merged[k]

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
            restored += 1

        conn.commit()
        return {"ok": True, "patch_run_id": int(patch_run_id), "restored_members": restored, "mode": mode}
    finally:
        conn.close()

10-6) VS Code 확장: “Resolve” Webview 추가(최소 구현)
10-6-1) 뷰 구성(추천)

Resolve(새 탭): 충돌/누락/추천 링크/패치 롤백을 한 화면에

기존 Results 탭은 “분석 결과/리포트”에 집중

기존 Mapping 탭은 “token↔member 매핑”에 집중

10-6-2) resolveView.ts(스켈레톤)

src/views/resolveView.ts (개념 코드)

import * as vscode from "vscode";
import { invoke } from "../mcp"; // 네가 쓰는 invoke 래퍼

export class ResolveViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "structai.resolveView";
  constructor(private readonly context: vscode.ExtensionContext) {}

  resolveWebviewView(view: vscode.WebviewView) {
    view.webview.options = { enableScripts: true };
    view.webview.html = this.getHtml();

    view.webview.onDidReceiveMessage(async (msg) => {
      try {
        switch (msg.type) {
          case "refresh": {
            const { cad_artifact_id, model_id, analysis_run_id } = msg;
            const [links, conflicts, quality, patches] = await Promise.all([
              invoke("structai_specs_list_links", { cad_artifact_id, model_id, status: "suggested", limit: 300 }),
              invoke("structai_token_story_conflicts", { cad_artifact_id, model_id }),
              invoke("structai_quality_summary", { model_id, analysis_run_id }),
              invoke("structai_design_list_patch_runs", { model_id, limit: 30 }),
            ]);
            view.webview.postMessage({ type: "data", links, conflicts, quality, patches });
            break;
          }

          case "setLinkStatus": {
            const res = await invoke("structai_specs_set_link_status", { link_id: msg.link_id, to_status: msg.to_status, reason: msg.reason || "" });
            view.webview.postMessage({ type: "toast", ok: true, text: `link ${res.from} -> ${res.to}` });
            break;
          }

          case "autoConfirmTableSchema": {
            const res = await invoke("structai_specs_auto_confirm_table_schema", { cad_artifact_id: msg.cad_artifact_id, model_id: msg.model_id });
            view.webview.postMessage({ type: "toast", ok: true, text: `confirmed=${res.confirmed}, kept=${res.kept_suggested}` });
            break;
          }

          case "applySpecs": {
            const res = await invoke("structai_design_apply_specs_to_inputs", { cad_artifact_id: msg.cad_artifact_id, model_id: msg.model_id, overwrite_keys: false });
            view.webview.postMessage({ type: "toast", ok: true, text: `applied. patch_run_id=${res.patch_run_id}` });
            break;
          }

          case "rollbackPatch": {
            const res = await invoke("structai_design_rollback_patch", { patch_run_id: msg.patch_run_id, mode: msg.mode || "keys_only" });
            view.webview.postMessage({ type: "toast", ok: true, text: `rollback done. restored=${res.restored_members}` });
            break;
          }

          default:
            break;
        }
      } catch (e: any) {
        view.webview.postMessage({ type: "toast", ok: false, text: String(e?.message || e) });
      }
    });
  }

  private getHtml(): string {
    // media/resolve/main.js 를 로딩하는 방식으로 구현하는 걸 추천
    return `<!DOCTYPE html>
      <html><body>
        <div id="app"></div>
        <script>
          const vscode = acquireVsCodeApi();
          // main.js 로 분리 권장
        </script>
      </body></html>`;
  }
}


그리고 package.json의 contributes.views에 view 등록(개념):

"views": {
  "explorer": [
    { "id": "structai.resolveView", "name": "StructAI Resolve" }
  ]
}

10-6-3) Resolve 웹뷰 UI(핵심 UX만)

웹뷰(main.js)에서는 다음 4개 섹션만 있으면 된다:

Quality Summary: missing counts

Token-Story Conflicts: 후보 목록

Suggested Spec Links: confirm/reject 버튼

Patch Runs: rollback 버튼

초기 MVP는 HTML table + 버튼만으로 충분해.

10-7) Step10 운영 순서(실무용 “한 바퀴”)

(템플릿/코드북 기반) rulepack 생성 및 활성화

#structai_templates_import { "path": "templates_kds_rc_core.json" }
#structai_templates_set_active { "name": "kds-rc-core", "version": "2025.01" }
#structai_rulepack_generate_from_templates {
  "new_rulepack_name": "kds-rc-generated",
  "new_rulepack_version": "2025.01",
  "activate": true
}


DXF/테이블/스토리/스펙 파이프라인 → suggested 생성
(이건 Step8/9 흐름 그대로)

VS Code Resolve 패널에서

auto confirm(table_schema) → apply specs → (필요시) rollback patch

내력 계산 → 체크 실행 → 리포트 생성

외부 benchmark 비교 + diff 리포트

#structai_benchmark_import { "path": "fixtures/etabs_case01.csv", "name": "ETABS", "version": "case01", "overwrite": true }
#structai_compare_check_run_to_benchmark { "check_run_id": 1, "benchmark_id": 1, "ratio_tol": 0.01, "ratio_warn": 0.03 }
#structai_compare_report_generate { "compare_id": 1, "formats": ["md","pdf"] }

10-8) Step10에서 바로 “현장 적용 신뢰”가 생기는 이유

Diff report로 “상용 프로그램 대비 어디가 얼마나 다른지”가 숫자로 드러남

템플릿 + codebook으로 rulepack이 흔들리지 않음(조항/버전 변경이 구조적으로 관리됨)

패치 롤백이 있으니 자동화가 실수해도 바로 되돌릴 수 있음

Resolve UI로 사람이 최종 확정하는 흐름이 정착됨(자동화 + 인간 검증)

원하면 다음 단계(11단계)로는, Step9의 Regression harness를 실제로 “완성”시켜서

fixture 기반 자동 실행

golden 캡처/비교

변경 시 자동 Fail

그리고 비교/회귀 리포트를 VS Code에서 한 번에 보는 QA 탭
까지 묶어서 “CI/품질관리” 단계로 올려줄게.