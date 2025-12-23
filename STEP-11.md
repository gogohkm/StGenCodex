좋아. 이제 **11단계 = “회귀(Regression) 하네스 완성 + CI 자동 Fail + VS Code QA 탭”**을 바로 붙여서 동작하도록, 순서대로 구현안을 줄게.
(이 단계는 품질관리/현장 적용 신뢰를 만드는 마지막 퍼즐이야.)

11-1) DB 스키마 추가 (v0.1.2)

Step9~10에서 regression_* 테이블은 이미 만들었고, Step11에서는 리포트 파일(artifact) 연결이 필요해.
mcp_server/schema.sql 맨 아래에 추가:

-- ===== schema patch v0.1.2 =====
UPDATE meta SET value='0.1.2' WHERE key='schema_version';

CREATE TABLE IF NOT EXISTS regression_reports (
  regression_report_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES regression_runs(run_id) ON DELETE CASCADE,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  format TEXT NOT NULL,                    -- md|pdf
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_regression_reports_run
ON regression_reports(run_id, format);

11-2) “격리 DB(Isolated DB)” 지원(회귀는 사용자 DB를 절대 오염시키면 안 됨)
11-2-1) server.py의 _connect()에 DB override 추가

server.py 상단 근처(전역)에 추가:

DB_PATH_OVERRIDE: Optional[Path] = None


그리고 _connect()를 다음처럼 변경:

def _connect():
    path = DB_PATH_OVERRIDE or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

11-2-2) DB override 컨텍스트 매니저 + 스키마 보장

server.py에 추가:

from contextlib import contextmanager

def _apply_schema_if_needed(db_path: Path):
    # schema.sql 기반 초기화/업데이트(이미 CREATE IF NOT EXISTS라 여러번 실행 안전)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        schema_path = (Path(__file__).resolve().parent / "schema.sql")
        sql = schema_path.read_text(encoding="utf-8", errors="ignore")
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()

@contextmanager
def _db_override(db_path: Path):
    global DB_PATH_OVERRIDE
    prev = DB_PATH_OVERRIDE
    DB_PATH_OVERRIDE = db_path
    try:
        yield
    finally:
        DB_PATH_OVERRIDE = prev

11-3) 회귀 케이스 Fixture 포맷(v2) — “도구 호출 시퀀스”로 통일(가장 튼튼)

회귀가 망가지는 이유는 “하네스가 특정 내부 구현에 결박”되기 때문이야.
그래서 fixture는 **MCP tool 호출 목록(steps)**으로 정의하는 게 가장 안정적이야.

11-3-1) fixture_json 예시(통합 케이스)
{
  "vars": {
    "dxf": "fixtures/case01/S-101.dxf",
    "members": "fixtures/case01/members.json",
    "results": "fixtures/case01/results.json"
  },
  "steps": [
    { "tool": "structai_import_dxf", "args": { "path": "${dxf}" }, "save": { "cad_artifact_id": "artifact_id" } },

    { "tool": "structai_model_import_members_json", "args": { "path": "${members}", "model_name": "case01" }, "save": { "model_id": "model_id" } },
    { "tool": "structai_results_import_envelopes_json", "args": { "model_id": "${model_id}", "path": "${results}", "run_name": "case01" }, "save": { "analysis_run_id": "analysis_run_id" } },

    { "tool": "structai_cad_parse_specs", "args": { "cad_artifact_id": "${cad_artifact_id}" } },
    { "tool": "structai_cad_detect_story_tags", "args": { "cad_artifact_id": "${cad_artifact_id}" } },
    { "tool": "structai_cad_extract_tables", "args": { "cad_artifact_id": "${cad_artifact_id}", "min_cells": 16 } },
    { "tool": "structai_cad_infer_table_schemas_v2", "args": { "cad_artifact_id": "${cad_artifact_id}" } },

    { "tool": "structai_member_mapping_autofill", "args": { "cad_artifact_id": "${cad_artifact_id}", "model_id": "${model_id}" } },

    { "tool": "structai_token_story_map_build", "args": { "cad_artifact_id": "${cad_artifact_id}", "model_id": "${model_id}" } },
    { "tool": "structai_token_story_auto_confirm", "args": { "cad_artifact_id": "${cad_artifact_id}", "model_id": "${model_id}" } },

    { "tool": "structai_specs_from_table_rows", "args": { "cad_artifact_id": "${cad_artifact_id}", "model_id": "${model_id}" } },
    { "tool": "structai_specs_auto_confirm_table_schema", "args": { "cad_artifact_id": "${cad_artifact_id}", "model_id": "${model_id}" } },

    { "tool": "structai_design_apply_specs_to_inputs", "args": { "cad_artifact_id": "${cad_artifact_id}", "model_id": "${model_id}", "overwrite_keys": false }, "save": { "patch_run_id": "patch_run_id" } },

    { "tool": "structai_design_compute_rc_beam_rect", "args": { "model_id": "${model_id}", "overwrite": true, "defaults": { "fc": 27, "fy": 400, "phi_flex": 0.9, "phi_shear": 0.75 } } },

    { "tool": "structai_check_run", "args": { "model_id": "${model_id}", "analysis_run_id": "${analysis_run_id}" }, "save": { "check_run_id": "check_run_id" } }
  ],
  "final": {
    "check_run_id": "${check_run_id}"
  }
}


포인트: regression 하네스는 “도구를 어떻게 호출할지”만 알고 있으면 된다.
내부 구현(파서/DB 구조)이 바뀌어도 도구 호출 시퀀스만 유지되면 회귀가 유지된다.

11-4) 회귀 실행 엔진(steps 실행 + 변수치환 + 결과 저장)
11-4-1) tool registry 자동 생성(서버 내부에서 호출 가능하게)

server.py 맨 아래쪽에 추가(또는 tool 정의 이후):

def _tool_registry() -> Dict[str, Any]:
    # @mcp.tool 로 등록된 함수들 이름 규칙이 structai_로 시작한다는 가정
    reg = {}
    for name, obj in globals().items():
        if callable(obj) and name.startswith("structai_"):
            reg[name] = obj
    return reg

11-4-2) 변수 치환 + path 추출 헬퍼

server.py에 추가:

import re
_VAR_RX = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

def _resolve(obj: Any, vars: Dict[str, Any]) -> Any:
    if isinstance(obj, str):
        def repl(m):
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
    # dot path 지원: "a.b.c"
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            raise KeyError(f"output path not found: {path}")
    return cur

11-5) (필수) 회귀용 최소 Importer 2개 제공(없으면 케이스를 못 돌림)

너의 기존 프로젝트에 이미 import 도구가 있으면 이건 “대체”가 아니라 “회귀용 보조”로 두면 된다.

11-5-1) members.json → model 생성/적재
@mcp.tool()
def structai_model_import_members_json(path: str, model_name: str = "regression_model") -> Dict[str, Any]:
    """
    fixtures용 간단 importer
    members.json 형식 예:
    {
      "members":[
        {"member_uid":"B12", "member_label":"B12", "type":"beam", "story":"3F", "section":"H-400x200x8x13"}
      ]
    }
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    members = data.get("members") or data
    if not isinstance(members, list):
        raise ValueError("members.json must be list or {members:[...]}")

    conn = _connect()
    try:
        # models 테이블 스키마는 프로젝트마다 다를 수 있음 → 여기서는 최소 컬럼(name) 가정
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

11-5-2) results.json → analysis_run + envelope 적재
@mcp.tool()
def structai_results_import_envelopes_json(
    model_id: int,
    path: str,
    run_name: str = "regression_run"
) -> Dict[str, Any]:
    """
    results.json 형식 예:
    {
      "results":[
        {"member_uid":"B12","env":{"M3_max":120,"M3_min":-90,"V2_max":80}}
      ]
    }
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    results = data.get("results") or data
    if not isinstance(results, list):
        raise ValueError("results.json must be list or {results:[...]}")

    conn = _connect()
    try:
        cur = conn.execute("INSERT INTO analysis_runs(model_id, name, created_at) VALUES(?,?, datetime('now'))", (int(model_id), run_name))
        analysis_run_id = int(cur.lastrowid)

        # member_uid -> model_member_id
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
                """
                INSERT INTO member_results(analysis_run_id, model_member_id, envelope_json, created_at)
                VALUES(?,?,?, datetime('now'))
                """,
                (int(analysis_run_id), int(mmid), json.dumps(env, ensure_ascii=False)),
            )
            inserted += 1

        conn.commit()
        return {"ok": True, "analysis_run_id": analysis_run_id, "inserted": inserted}
    finally:
        conn.close()


⚠️ 위 importer는 “너의 DB 스키마(models, analysis_runs)” 컬럼명이 다르면 맞춰서 바꿔야 함.
하지만 회귀 하네스 자체 구조(steps 실행/compare)는 그대로 유지된다.

11-6) Regression 실행 MCP Tools “완성판”
11-6-1) 케이스 1개 실행(격리 DB + steps 실행 + metrics 반환)

Step9에서 만든 _metrics_for_check_run()를 그대로 재사용한다고 가정할게.

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


이제 케이스 실행 tool:

@mcp.tool()
def structai_regression_run_case(
    fixture_json: Dict[str, Any],
    isolated_db: bool = True,
    keep_db: bool = False
) -> Dict[str, Any]:
    """
    fixture_json.steps를 실행해 check_run_id와 metrics를 반환.
    isolated_db=True면 임시 sqlite에 실행.
    """
    reg = _tool_registry()

    # choose db path
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
        for i, st in enumerate(steps):
            tool = str(st.get("tool") or "")
            if tool not in reg:
                raise ValueError(f"tool not found: {tool}")
            args = _resolve(st.get("args") or {}, vars_)
            out = reg[tool](**args)  # direct call

            # save variables
            save = st.get("save") or {}
            if isinstance(save, dict):
                for var_name, path in save.items():
                    vars_[str(var_name)] = _get_by_path(out, str(path))

        final = fixture_json.get("final") or {}
        check_run_id = None
        if isinstance(final, dict) and "check_run_id" in final:
            check_run_id = int(_resolve(final["check_run_id"], vars_))
        else:
            if "check_run_id" in vars_:
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

    # cleanup db unless keep_db
    if isolated_db and (not keep_db):
        try:
            db_path.unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "ok": True,
        "isolated_db": isolated_db,
        "db_path": str(db_path) if (isolated_db and keep_db) else None,
        "check_run_id": int(check_run_id),
        "metrics": metrics
    }

11-6-2) Golden 업데이트(케이스/스위트)
(A) 케이스 1개 업데이트
@mcp.tool()
def structai_regression_update_golden_case(
    suite_name: str,
    case_name: str,
    isolated_db: bool = True
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

    # run case -> metrics
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

(B) 스위트 전체 업데이트
@mcp.tool()
def structai_regression_update_golden_suite(
    suite_name: str,
    isolated_db: bool = True
) -> Dict[str, Any]:
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
        updated.append({"case": c["name"], "golden": r["metrics"]})

        conn = _connect()
        try:
            conn.execute(
                "UPDATE regression_cases SET golden_json=? WHERE case_id=?",
                (json.dumps(r["metrics"], ensure_ascii=False), int(c["case_id"])),
            )
            conn.commit()
        finally:
            conn.close()

    return {"ok": True, "suite": suite_name, "updated": len(updated), "items": updated[:20]}

11-6-3) “비교 실행 + 자동 Fail” (회귀 Run)

Step9의 _compare_metrics()를 그대로 쓰거나 아래 버전으로:

def _compare_metrics(golden: Dict[str, Any], actual: Dict[str, Any], ratio_tol: float = 1e-3) -> Dict[str, Any]:
    diff = {"ok": True, "overall": {}, "by_check_type": {}}

    g_over = golden.get("overall", {})
    a_over = actual.get("overall", {})
    for k in set(g_over.keys()) | set(a_over.keys()):
        if int(g_over.get(k,0)) != int(a_over.get(k,0)):
            diff["overall"][k] = {"golden": int(g_over.get(k,0)), "actual": int(a_over.get(k,0))}
            diff["ok"] = False

    g_ct = golden.get("by_check_type", {})
    a_ct = actual.get("by_check_type", {})
    for ct in set(g_ct.keys()) | set(a_ct.keys()):
        gd = g_ct.get(ct, {})
        ad = a_ct.get(ct, {})
        cd = {}

        for k in ("PASS","WARN","FAIL","NA"):
            if int(gd.get(k,0)) != int(ad.get(k,0)):
                cd[k] = {"golden": int(gd.get(k,0)), "actual": int(ad.get(k,0))}

        gw = gd.get("worst_ratio")
        aw = ad.get("worst_ratio")
        if (gw is not None) and (aw is not None):
            if abs(float(gw)-float(aw)) > float(ratio_tol):
                cd["worst_ratio"] = {"golden": float(gw), "actual": float(aw), "tol": float(ratio_tol)}
        elif gw != aw:
            cd["worst_ratio"] = {"golden": gw, "actual": aw}

        if cd:
            diff["by_check_type"][ct] = cd
            diff["ok"] = False

    return diff


이제 suite run:

@mcp.tool()
def structai_regression_run_suite_v2(
    suite_name: str,
    isolated_db: bool = True,
    ratio_tol: float = 1e-3
) -> Dict[str, Any]:
    """
    suite 전체 케이스를 실행하고 golden과 비교.
    하나라도 diff/에러가 있으면 FAIL.
    """
    conn = _connect()
    try:
        s = conn.execute("SELECT suite_id FROM regression_suites WHERE name=?", (suite_name,)).fetchone()
        if not s:
            raise ValueError("suite not found")
        sid = int(s["suite_id"])

        run_cur = conn.execute("INSERT INTO regression_runs(suite_id, status, report_json) VALUES(?, 'RUNNING', '{}')", (sid,))
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
                raise ValueError("golden_json is empty. Run structai_regression_update_golden_suite first.")

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

        except Exception as e:
            all_ok = False
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO regression_case_results(run_id, case_id, status, diff_json) VALUES(?,?,?,?)",
                    (run_id, case_id, "ERROR", json.dumps({"error": str(e)}, ensure_ascii=False)),
                )
                conn.commit()
            finally:
                conn.close()
            case_reports.append({"case": name, "status": "ERROR", "error": str(e)})

    status = "PASS" if all_ok else "FAIL"

    # store summary report_json
    conn = _connect()
    try:
        summary = {
            "suite": suite_name,
            "status": status,
            "ratio_tol": ratio_tol,
            "cases": case_reports,
        }
        conn.execute(
            "UPDATE regression_runs SET status=?, finished_at=datetime('now'), report_json=? WHERE run_id=?",
            (status, json.dumps(summary, ensure_ascii=False), run_id),
        )
        conn.commit()
    finally:
        conn.close()

    return {"ok": True, "run_id": run_id, "status": status, "summary": summary}

11-7) Regression 리포트 생성(MD/PDF) + QA 탭에서 열기

Step10의 compare report 생성과 같은 패턴으로 만들면 된다.

@mcp.tool()
def structai_regression_report_generate(
    run_id: int,
    formats: Optional[List[str]] = None,
    out_dir: Optional[str] = None
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

        report = json.loads(rr["report_json"] or "{}")

        # fetch case results
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
                diff = (it["detail"].get("diff") or {})
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
            outputs.append({"format":"md","path":str(md_path),"uri":uri,"artifact_id":art_id})

        # pdf는 필요하면 Step10 방식 그대로 붙이면 됨(동일 패턴)

        conn.commit()
        return {"ok": True, "run_id": run_id, "outputs": outputs}
    finally:
        conn.close()

11-8) CI 자동 Fail(깃헙 액션/스크립트)
11-8-1) CLI 스크립트: mcp_server/regression_cli.py

새 파일로 추가:

# mcp_server/regression_cli.py
import argparse
import json
import sys

from mcp_server.server import (
    structai_regression_run_suite_v2,
    structai_regression_report_generate,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--ratio_tol", type=float, default=1e-3)
    ap.add_argument("--isolated_db", action="store_true", default=True)
    ap.add_argument("--no_isolated_db", action="store_true", default=False)
    ap.add_argument("--report", action="store_true", default=True)
    args = ap.parse_args()

    isolated = args.isolated_db and (not args.no_isolated_db)

    r = structai_regression_run_suite_v2(suite_name=args.suite, isolated_db=isolated, ratio_tol=args.ratio_tol)
    run_id = r["run_id"]

    if args.report:
        structai_regression_report_generate(run_id=run_id, formats=["md"])

    print(json.dumps(r, ensure_ascii=False, indent=2))

    if r["status"] != "PASS":
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())

11-8-2) GitHub Actions 예시: .github/workflows/regression.yml
name: StructAI Regression

on:
  push:
  pull_request:

jobs:
  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install
        run: |
          python -m pip install -U pip
          pip install -r requirements.txt

      - name: Run regression
        run: |
          python -m mcp_server.regression_cli --suite core --ratio_tol 0.001


이러면 PR에서 파서/룰/내력식이 바뀌어서 결과가 변하면 즉시 FAIL로 잡힌다.

11-9) VS Code QA 탭(회귀 + 비교를 한 화면에)
11-9-1) 서버에 “QA용 조회 도구” 추가
(A) regression runs list/read
@mcp.tool()
def structai_regression_list_runs(suite_name: str, limit: int = 20) -> Dict[str, Any]:
    conn = _connect()
    try:
        s = conn.execute("SELECT suite_id FROM regression_suites WHERE name=?", (suite_name,)).fetchone()
        if not s:
            raise ValueError("suite not found")
        sid = int(s["suite_id"])
        rows = conn.execute(
            """
            SELECT run_id, status, started_at, finished_at
            FROM regression_runs
            WHERE suite_id=?
            ORDER BY run_id DESC
            LIMIT ?
            """,
            (sid, int(limit)),
        ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
    finally:
        conn.close()

@mcp.tool()
def structai_regression_read_run(run_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        rr = conn.execute("SELECT run_id, suite_id, status, report_json, started_at, finished_at FROM regression_runs WHERE run_id=?", (int(run_id),)).fetchone()
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

        reports = conn.execute(
            """
            SELECT a.artifact_id, a.uri, a.title, rr.format
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
            "cases": [{"case": r["case_name"], "status": r["status"], "detail": json.loads(r["diff_json"] or "{}")} for r in rows],
            "reports": [dict(x) for x in reports],
        }
    finally:
        conn.close()

(B) compare runs list/read (Step10 결과를 QA 탭에 같이 보여주기)
@mcp.tool()
def structai_compare_list_runs(limit: int = 20) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT compare_id, check_run_id, benchmark_id, name, created_at
            FROM compare_runs
            ORDER BY compare_id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
    finally:
        conn.close()

@mcp.tool()
def structai_compare_read_run(compare_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        cr = conn.execute("SELECT * FROM compare_runs WHERE compare_id=?", (int(compare_id),)).fetchone()
        if not cr:
            raise ValueError("compare run not found")

        reports = conn.execute(
            """
            SELECT a.artifact_id, a.uri, a.title, r.format
            FROM compare_reports r
            JOIN artifacts a ON a.artifact_id = r.artifact_id
            WHERE r.compare_id=?
            ORDER BY r.compare_report_id DESC
            """,
            (int(compare_id),),
        ).fetchall()

        # top diff items
        items = conn.execute(
            """
            SELECT severity, member_uid, story_norm, check_type, combo, expected_ratio, actual_ratio, abs_diff
            FROM compare_items
            WHERE compare_id=? AND severity IN ('DIFF','WARN')
            ORDER BY abs(abs_diff) DESC
            LIMIT 50
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

11-9-2) VS Code QA ViewProvider 스켈레톤

src/views/qaView.ts (개념):

import * as vscode from "vscode";
import { invoke } from "../mcp";

export class QaViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "structai.qaView";
  constructor(private readonly context: vscode.ExtensionContext) {}

  resolveWebviewView(view: vscode.WebviewView) {
    view.webview.options = { enableScripts: true };

    view.webview.html = this.html();
    view.webview.onDidReceiveMessage(async (msg) => {
      try {
        if (msg.type === "refresh") {
          const [runs, compares] = await Promise.all([
            invoke("structai_regression_list_runs", { suite_name: msg.suite_name, limit: 20 }),
            invoke("structai_compare_list_runs", { limit: 20 }),
          ]);
          view.webview.postMessage({ type: "data", runs, compares });
        }

        if (msg.type === "runSuite") {
          const res = await invoke("structai_regression_run_suite_v2", { suite_name: msg.suite_name, isolated_db: true, ratio_tol: msg.ratio_tol || 0.001 });
          await invoke("structai_regression_report_generate", { run_id: res.run_id, formats: ["md"] });
          view.webview.postMessage({ type: "toast", ok: true, text: `run=${res.run_id} status=${res.status}` });
        }

        if (msg.type === "readRun") {
          const run = await invoke("structai_regression_read_run", { run_id: msg.run_id });
          view.webview.postMessage({ type: "runDetail", run });
        }

        if (msg.type === "readCompare") {
          const cmp = await invoke("structai_compare_read_run", { compare_id: msg.compare_id });
          view.webview.postMessage({ type: "compareDetail", cmp });
        }

      } catch (e: any) {
        view.webview.postMessage({ type: "toast", ok: false, text: String(e?.message || e) });
      }
    });
  }

  private html(): string {
    return `<!doctype html>
<html>
<body>
  <h3>StructAI QA</h3>
  <div>
    Suite: <input id="suite" value="core"/>
    RatioTol: <input id="tol" value="0.001"/>
    <button id="refresh">Refresh</button>
    <button id="run">Run Suite</button>
  </div>
  <hr/>
  <div id="runs"></div>
  <hr/>
  <div id="compares"></div>
  <pre id="detail"></pre>

<script>
  const vscode = acquireVsCodeApi();
  const $ = (id)=>document.getElementById(id);

  $("refresh").onclick = () => vscode.postMessage({type:"refresh", suite_name:$("suite").value});
  $("run").onclick = () => vscode.postMessage({type:"runSuite", suite_name:$("suite").value, ratio_tol: parseFloat($("tol").value)});

  window.addEventListener("message", (ev) => {
    const msg = ev.data;
    if (msg.type === "data") {
      const runs = msg.runs.items || [];
      $("runs").innerHTML = "<h4>Regression runs</h4>" + runs.map(r =>
        \`<div><button onclick="vscode.postMessage({type:'readRun', run_id:\${r.run_id}})">Open</button>
        #\${r.run_id} \${r.status} \${r.started_at || ""}</div>\`
      ).join("");

      const comps = msg.compares.items || [];
      $("compares").innerHTML = "<h4>Compare runs</h4>" + comps.map(c =>
        \`<div><button onclick="vscode.postMessage({type:'readCompare', compare_id:\${c.compare_id}})">Open</button>
        #\${c.compare_id} (check:\${c.check_run_id}, bench:\${c.benchmark_id})</div>\`
      ).join("");
    }
    if (msg.type === "runDetail") {
      $("detail").textContent = JSON.stringify(msg.run, null, 2);
    }
    if (msg.type === "compareDetail") {
      $("detail").textContent = JSON.stringify(msg.cmp, null, 2);
    }
    if (msg.type === "toast") {
      // 최소 토스트
      console.log(msg);
    }
  });

  vscode.postMessage({type:"refresh", suite_name:$("suite").value});
</script>
</body>
</html>`;
  }
}


그리고 package.json에 view 추가:

"views": {
  "explorer": [
    { "id": "structai.qaView", "name": "StructAI QA" }
  ]
}

11-10) Step11 “운영 순서” (현장 품질관리 루틴)

새 기능 개발/파서 개선/룰 변경

로컬에서 회귀 실행

#structai_regression_run_suite_v2 { "suite_name": "core", "isolated_db": true, "ratio_tol": 0.001 }
#structai_regression_report_generate { "run_id": 1, "formats": ["md"] }


FAIL이면 QA 탭에서 실패 케이스 diff 확인

의도된 변경이면 golden 업데이트

#structai_regression_update_golden_suite { "suite_name": "core", "isolated_db": true }


PR 올리면 GitHub Actions가 자동으로 돌고, FAIL이면 머지 차단

여기까지가 Step11 완료 상태(정리)

fixture 기반 자동 실행 ✅

golden 캡처/비교 ✅

변경 시 자동 Fail(CI) ✅

VS Code QA 탭에서 회귀/비교 리포트 확인 ✅

원하면 다음(12단계)로는 **“현장 적용 최종형”**인

프로젝트별 “QA 프로파일”(허용 오차/체크 범위/예외 규칙)

데이터셋 버전 관리(섹션 카탈로그/코드북/템플릿의 SemVer + changelog)

리포트 서명/감사 로그(누가 무엇을 승인/거절했는지)
까지 묶어서 “정말로 배포 가능한 제품” 레벨로 마감해줄게.
