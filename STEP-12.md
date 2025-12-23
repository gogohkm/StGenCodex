좋아. 이제 **12단계 = “현장 배포 가능한 제품 마감(거버넌스/버전관리/서명·감사)”**를 순서대로 구현 설계 + 코드(스키마/서버 도구/UI 포인트)까지 한 번에 줄게.

이 단계의 목표는 딱 3가지야:

프로젝트(모델)별 QA 프로파일로 허용오차/검토범위/예외규칙을 “표준화 + 재현 가능”하게 만들기

**데이터셋(단면/코드북/템플릿/룰팩/픽스처)**의 SemVer + 활성버전 + 변경이력을 관리하기

생성된 리포트/결과에 대해 **서명(무결성) + 승인(누가/언제/무엇을)**을 남겨 감사(audit) 가능한 상태 만들기

12-1) DB 스키마 추가 (v0.1.3)

mcp_server/schema.sql 맨 아래에 추가해줘.

-- ===== schema patch v0.1.3 =====
UPDATE meta SET value='0.1.3' WHERE key='schema_version';

-- -----------------------------
-- QA profiles
-- -----------------------------
CREATE TABLE IF NOT EXISTS qa_profiles (
  qa_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,                   -- semver recommended
  profile_json TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS model_qa_profiles (
  model_id INTEGER PRIMARY KEY REFERENCES models(model_id) ON DELETE CASCADE,
  qa_profile_id INTEGER NOT NULL REFERENCES qa_profiles(qa_profile_id) ON DELETE RESTRICT,
  bound_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- -----------------------------
-- Dataset registry (versioned artifacts)
-- -----------------------------
CREATE TABLE IF NOT EXISTS dataset_defs (
  dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
  type TEXT NOT NULL,                      -- section_catalog|codebook|template_set|rulepack|fixture_suite|other
  name TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(type, name)
);

CREATE TABLE IF NOT EXISTS dataset_versions (
  dataset_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id INTEGER NOT NULL REFERENCES dataset_defs(dataset_id) ON DELETE CASCADE,
  version TEXT NOT NULL,                   -- semver
  artifact_id INTEGER REFERENCES artifacts(artifact_id) ON DELETE SET NULL,
  sha256 TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  is_active INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(dataset_id, version)
);

CREATE INDEX IF NOT EXISTS idx_dataset_versions_active
ON dataset_versions(dataset_id, is_active, created_at);

CREATE TABLE IF NOT EXISTS dataset_changelogs (
  changelog_id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_version_id INTEGER NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE CASCADE,
  change_type TEXT NOT NULL DEFAULT 'change',   -- change|fix|breaking|note
  summary TEXT NOT NULL,
  details TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- -----------------------------
-- Signatures (report integrity)
-- -----------------------------
CREATE TABLE IF NOT EXISTS report_signatures (
  signature_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  method TEXT NOT NULL DEFAULT 'sha256',        -- sha256|hmac-sha256|ed25519(optional)
  digest_sha256 TEXT NOT NULL,
  signature_b64 TEXT,                          -- for hmac/ed25519
  key_id TEXT,                                 -- identifier (not secret)
  signer TEXT,                                 -- "user@org" or machine id
  note TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_report_signatures_artifact
ON report_signatures(artifact_id);

-- -----------------------------
-- Approvals (governance)
-- -----------------------------
CREATE TABLE IF NOT EXISTS approvals (
  approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity_type TEXT NOT NULL,                   -- report|check_run|compare_run|regression_run|patch_run|other
  entity_id INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'requested',    -- requested|approved|rejected|revoked
  actor TEXT,                                  -- who decided
  comment TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_approvals_entity
ON approvals(entity_type, entity_id, status);

12-2) A. 프로젝트별 QA 프로파일(허용오차/검토범위/예외규칙)
12-2-1) QA 프로파일 JSON 포맷(권장)

파일 예: qa_profiles/qa_kds_prod.json

{
  "name": "kds-prod",
  "version": "1.0.0",

  "check_scope": {
    "enabled_check_types": ["strength.flexure", "strength.shear"],
    "disabled_check_types": []
  },

  "thresholds": {
    "default": { "limit": 1.0, "warn": 0.95 },
    "per_check_type": {
      "strength.flexure": { "warn": 0.97 },
      "strength.shear":   { "warn": 0.95 }
    }
  },

  "exceptions": [
    {
      "selector": { "member_uid_regex": "^B12$", "story_norm": "3F", "check_type": "strength.shear" },
      "action": { "treat_as": "WARN", "max_ratio": 1.05, "note": "현장 검토: 전단은 5%까지 WARN으로 관리" }
    },
    {
      "selector": { "member_uid_regex": "^(G|GB)\\d+$" },
      "action": { "skip_check_types": ["deflection.service"] }
    }
  ],

  "compare": {
    "ratio_tol": 0.01,
    "ratio_warn": 0.03
  },

  "regression": {
    "ratio_tol": 0.001,
    "fail_on_warn": false
  }
}

해석 규칙(결정)

check_scope: 어떤 check_type를 실행할지(룰팩에 있어도 QA에서 끌 수 있음)

thresholds: 룰팩의 limit/warn을 프로젝트 정책으로 오버라이드 가능

exceptions: “선택자(selector)”로 특정 부재/층/체크에 대해

skip

허용 한도 조정

FAIL을 WARN으로 완화(단, 상한 max_ratio 안에서만)

compare/regression: 비교/회귀에서의 허용 오차 정책을 프로젝트 단위로 고정

12-2-2) MCP Tool: QA 프로파일 임포트/활성/모델 바인딩/조회
(1) 임포트
@mcp.tool()
def structai_qa_profile_import(path: str, activate: bool = False) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    name = data.get("name") or p.stem
    ver = data.get("version") or "0.0.0"

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
        row = conn.execute("SELECT qa_profile_id, name, version, is_active FROM qa_profiles WHERE name=? AND version=?", (name, ver)).fetchone()
        return {"ok": True, "profile": dict(row)}
    finally:
        conn.close()

(2) 활성화
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

(3) 모델 바인딩
@mcp.tool()
def structai_qa_profile_bind_model(model_id: int, name: str, version: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        r = conn.execute("SELECT qa_profile_id FROM qa_profiles WHERE name=? AND version=?", (name, version)).fetchone()
        if not r:
            raise ValueError("qa profile not found")
        pid = int(r["qa_profile_id"])
        conn.execute(
            """
            INSERT INTO model_qa_profiles(model_id, qa_profile_id, bound_at)
            VALUES(?,?, datetime('now'))
            ON CONFLICT(model_id) DO UPDATE SET
              qa_profile_id=excluded.qa_profile_id,
              bound_at=datetime('now')
            """,
            (int(model_id), pid),
        )
        conn.commit()
        return {"ok": True, "model_id": int(model_id), "qa_profile_id": pid}
    finally:
        conn.close()

(4) 유효 QA 프로파일 resolve (모델 우선 → 없으면 active)
def _get_effective_qa_profile(conn, model_id: Optional[int]) -> Dict[str, Any]:
    row = None
    if model_id is not None:
        row = conn.execute(
            """
            SELECT p.profile_json, p.name, p.version
            FROM model_qa_profiles mp
            JOIN qa_profiles p ON p.qa_profile_id = mp.qa_profile_id
            WHERE mp.model_id=?
            """,
            (int(model_id),),
        ).fetchone()

    if not row:
        row = conn.execute(
            "SELECT profile_json, name, version FROM qa_profiles WHERE is_active=1 ORDER BY qa_profile_id DESC LIMIT 1"
        ).fetchone()

    if not row:
        # default minimal profile
        return {"name":"default","version":"0.0.0","check_scope":{"enabled_check_types":[],"disabled_check_types":[]},
                "thresholds":{"default":{"limit":1.0,"warn":0.95},"per_check_type":{}},
                "exceptions":[], "compare":{"ratio_tol":0.01,"ratio_warn":0.03}, "regression":{"ratio_tol":0.001,"fail_on_warn":False}}

    prof = json.loads(row["profile_json"] or "{}")
    prof.setdefault("name", row["name"])
    prof.setdefault("version", row["version"])
    return prof

@mcp.tool()
def structai_qa_profile_get_effective(model_id: Optional[int] = None) -> Dict[str, Any]:
    conn = _connect()
    try:
        prof = _get_effective_qa_profile(conn, model_id)
        return {"ok": True, "model_id": model_id, "profile": prof}
    finally:
        conn.close()

12-2-3) QA 프로파일을 체크 엔진에 적용(핵심 변경점)

기존 structai_check_run(model_id, analysis_run_id, ...)가 있다고 가정하면, 내부에서 다음을 적용해:

(1) check_type 실행 범위 필터

enabled_check_types가 비어있지 않으면 그 목록만 실행

disabled_check_types는 언제나 제외

(2) warn/limit 오버라이드

rulepack의 warn/limit를

thresholds.default

thresholds.per_check_type[check_type]
순서로 덮어씀

(3) 예외(exception) 적용

각 결과(부재×체크×조합)마다 selector가 매칭되면 action 수행:

skip_check_types: 그 체크 타입은 결과 생성 자체를 생략

treat_as: FAIL→WARN 등 상태 강제(단, max_ratio 초과면 FAIL 유지 권장)

max_ratio: 이 이상이면 예외 적용 불가(보수적 안전장치)

선택자 매칭은 결정론 규칙으로 충분해:

member_uid_regex, story_norm, type, check_type

이 변경은 “AI가 억지로 PASS 만들기”를 막고,
“프로젝트 정책으로 승인된 예외만” 반영하게 해준다.

12-3) B. 데이터셋 버전 관리(SemVer + 활성버전 + changelog)
12-3-1) 무엇을 데이터셋으로 관리할까(추천)
dataset.type	내용	생성/임포트 시점
section_catalog	형강/단면 테이블 CSV/JSON	structai_sections_import_catalog 이후
codebook	KDS 조항/인용/버전	structai_codebook_import 이후
template_set	체크 템플릿 세트	structai_templates_import 이후
rulepack	템플릿+코드북으로 생성된 룰팩	structai_rulepack_generate_from_templates 이후
fixture_suite	regression fixture 묶음(폴더/zip)	QA 데이터셋 등록 시
12-3-2) MCP Tool: 데이터셋 등록/활성/조회
(1) dataset_version 등록(artifact로 저장)
@mcp.tool()
def structai_dataset_register(
    type: str,
    name: str,
    version: str,
    artifact_path: str,
    activate: bool = False,
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    p = Path(artifact_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    conn = _connect()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO dataset_defs(type, name) VALUES(?,?)",
            (type, name),
        )
        d = conn.execute("SELECT dataset_id FROM dataset_defs WHERE type=? AND name=?", (type, name)).fetchone()
        dataset_id = int(d["dataset_id"])

        sha = _sha256_file(p)
        uri = p.as_uri()
        art_id = _upsert_artifact(
            conn,
            uri=uri,
            kind=f"dataset_{type}",
            title=f"Dataset {type}:{name}@{version}",
            source_path=str(p),
            sha256=sha,
        )

        conn.execute(
            """
            INSERT INTO dataset_versions(dataset_id, version, artifact_id, sha256, meta_json, is_active)
            VALUES(?,?,?,?,?,0)
            ON CONFLICT(dataset_id, version) DO UPDATE SET
              artifact_id=excluded.artifact_id,
              sha256=excluded.sha256,
              meta_json=excluded.meta_json
            """,
            (dataset_id, version, art_id, sha, json.dumps(meta or {}, ensure_ascii=False)),
        )

        if activate:
            conn.execute("UPDATE dataset_versions SET is_active=0 WHERE dataset_id=?", (dataset_id,))
            conn.execute("UPDATE dataset_versions SET is_active=1 WHERE dataset_id=? AND version=?", (dataset_id, version))

        conn.commit()

        dv = conn.execute(
            "SELECT dataset_version_id, dataset_id, version, artifact_id, sha256, is_active FROM dataset_versions WHERE dataset_id=? AND version=?",
            (dataset_id, version),
        ).fetchone()

        return {"ok": True, "dataset": {"type": type, "name": name}, "version": dict(dv)}
    finally:
        conn.close()

(2) 활성 버전 설정
@mcp.tool()
def structai_dataset_set_active(type: str, name: str, version: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        d = conn.execute("SELECT dataset_id FROM dataset_defs WHERE type=? AND name=?", (type, name)).fetchone()
        if not d:
            raise ValueError("dataset not found")
        dataset_id = int(d["dataset_id"])
        conn.execute("UPDATE dataset_versions SET is_active=0 WHERE dataset_id=?", (dataset_id,))
        conn.execute("UPDATE dataset_versions SET is_active=1 WHERE dataset_id=? AND version=?", (dataset_id, version))
        conn.commit()
        return {"ok": True, "active": {"type": type, "name": name, "version": version}}
    finally:
        conn.close()

(3) active 목록 한 번에 보기(대시보드용)
@mcp.tool()
def structai_dataset_get_active_all() -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT d.type, d.name, v.version, v.sha256, v.artifact_id, a.uri
            FROM dataset_defs d
            JOIN dataset_versions v ON v.dataset_id=d.dataset_id
            LEFT JOIN artifacts a ON a.artifact_id=v.artifact_id
            WHERE v.is_active=1
            ORDER BY d.type, d.name
            """
        ).fetchall()
        return {"ok": True, "items": [dict(r) for r in rows]}
    finally:
        conn.close()

12-3-3) “임포트 도구들”에 dataset 자동 등록 붙이기(추천)

structai_sections_import_catalog(...) 실행 후:

원본 파일이 있다면 structai_dataset_register(type="section_catalog", name="ks-steel", version="2025.01.0", artifact_path=path, activate=True)

structai_templates_import(...) 실행 후:

type="template_set"

structai_codebook_import(...) 실행 후:

type="codebook"

이렇게 하면 “어떤 데이터셋/버전으로 결과가 만들어졌는지”를 나중에 정확히 재현할 수 있어.

12-4) C. 리포트 서명(무결성) + 승인 워크플로우
12-4-1) 서명(무결성) 방식 권장

실무에서 “서명”은 크게 3레벨이 있어:

sha256 digest 기록: 파일이 변조되지 않았는지만 확인(가장 간단)

HMAC-SHA256: 조직 내부 공유키로 “누가 만들었는지”까지 어느 정도 보장(키 관리 필요)

Ed25519/RSA: 개인키/공개키 기반의 강한 서명(가장 안전, 라이브러리 필요)

여기서는 우선 (1) sha256, (2)는 옵션으로 넣는 걸 추천할게.

12-4-2) MCP Tool: report sign / verify
(1) sign
import base64, hmac, hashlib, os

def _read_bytes_from_uri_or_path(uri_or_path: str) -> bytes:
    # artifact.uri가 file:// URI일 수도 있고, path일 수도 있음
    if uri_or_path.startswith("file:"):
        p = Path(uri_or_path.replace("file://", "")).resolve()
    else:
        p = Path(uri_or_path).expanduser().resolve()
    return p.read_bytes()

@mcp.tool()
def structai_report_sign(
    artifact_id: int,
    signer: str,
    method: str = "sha256",            # sha256|hmac-sha256
    key_id: str = "",
    note: str = ""
) -> Dict[str, Any]:
    """
    artifact 파일을 digest/서명하여 report_signatures에 저장
    method:
      - sha256: digest only
      - hmac-sha256: STRUCTAI_SIGNING_HMAC_KEY 환경변수 사용(바이너리/base64 가능)
    """
    if method not in ("sha256", "hmac-sha256"):
        raise ValueError("method must be sha256|hmac-sha256")

    conn = _connect()
    try:
        a = conn.execute("SELECT uri, sha256 FROM artifacts WHERE artifact_id=?", (int(artifact_id),)).fetchone()
        if not a:
            raise ValueError("artifact not found")

        content = _read_bytes_from_uri_or_path(a["uri"])
        digest = hashlib.sha256(content).hexdigest()

        sig_b64 = None
        if method == "hmac-sha256":
            key = os.environ.get("STRUCTAI_SIGNING_HMAC_KEY", "")
            if not key:
                raise ValueError("STRUCTAI_SIGNING_HMAC_KEY is not set for hmac-sha256")
            key_bytes = key.encode("utf-8")
            mac = hmac.new(key_bytes, content, hashlib.sha256).digest()
            sig_b64 = base64.b64encode(mac).decode("ascii")

        conn.execute(
            """
            INSERT INTO report_signatures(artifact_id, method, digest_sha256, signature_b64, key_id, signer, note)
            VALUES(?,?,?,?,?,?,?)
            """,
            (int(artifact_id), method, digest, sig_b64, key_id or None, signer, note),
        )
        conn.commit()
        sid = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        return {"ok": True, "signature_id": sid, "artifact_id": int(artifact_id), "method": method, "digest_sha256": digest}
    finally:
        conn.close()

(2) verify
@mcp.tool()
def structai_report_verify(signature_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        s = conn.execute(
            """
            SELECT rs.*, a.uri
            FROM report_signatures rs
            JOIN artifacts a ON a.artifact_id = rs.artifact_id
            WHERE rs.signature_id=?
            """,
            (int(signature_id),),
        ).fetchone()
        if not s:
            raise ValueError("signature not found")

        content = _read_bytes_from_uri_or_path(s["uri"])
        digest = hashlib.sha256(content).hexdigest()

        ok = (digest == s["digest_sha256"])
        details = {"digest_match": ok}

        if s["method"] == "hmac-sha256":
            key = os.environ.get("STRUCTAI_SIGNING_HMAC_KEY", "")
            if not key:
                details["hmac"] = "missing key env"
                ok = False
            else:
                mac = hmac.new(key.encode("utf-8"), content, hashlib.sha256).digest()
                sig_b64 = base64.b64encode(mac).decode("ascii")
                details["hmac_match"] = (sig_b64 == (s["signature_b64"] or ""))
                ok = ok and details["hmac_match"]

        return {"ok": True, "verified": bool(ok), "signature_id": int(signature_id), "details": details}
    finally:
        conn.close()

12-4-3) 승인(Approval) 워크플로우 도구
(1) 승인 요청
@mcp.tool()
def structai_approval_request(
    entity_type: str,
    entity_id: int,
    actor: str,
    comment: str = "",
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO approvals(entity_type, entity_id, status, actor, comment, meta_json, updated_at)
            VALUES(?,?,'requested',?,?,?, datetime('now'))
            """,
            (entity_type, int(entity_id), actor, comment, json.dumps(meta or {}, ensure_ascii=False)),
        )
        conn.commit()
        aid = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        return {"ok": True, "approval_id": aid, "status": "requested"}
    finally:
        conn.close()

(2) 승인/반려/철회
@mcp.tool()
def structai_approval_set_status(
    approval_id: int,
    status: str,                 # approved|rejected|revoked
    actor: str,
    comment: str = ""
) -> Dict[str, Any]:
    if status not in ("approved","rejected","revoked"):
        raise ValueError("status must be approved|rejected|revoked")

    conn = _connect()
    try:
        conn.execute(
            """
            UPDATE approvals
            SET status=?, actor=?, comment=?, updated_at=datetime('now')
            WHERE approval_id=?
            """,
            (status, actor, comment, int(approval_id)),
        )
        conn.commit()
        return {"ok": True, "approval_id": int(approval_id), "status": status}
    finally:
        conn.close()

(3) 조회
@mcp.tool()
def structai_approval_list(entity_type: str, entity_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT approval_id, status, actor, comment, created_at, updated_at, meta_json
            FROM approvals
            WHERE entity_type=? AND entity_id=?
            ORDER BY approval_id DESC
            """,
            (entity_type, int(entity_id)),
        ).fetchall()
        items = []
        for r in rows:
            it = dict(r)
            it["meta"] = json.loads(it.pop("meta_json") or "{}")
            items.append(it)
        return {"ok": True, "items": items}
    finally:
        conn.close()

12-5) “결과 재현성”을 100%로 만드는 마지막 연결(아주 중요)

이제부터 리포트/체크 실행 시에 아래 3가지를 메타로 반드시 남겨야 감사가 가능해져.

12-5-1) check_run 생성 시 저장할 메타

active dataset versions(섹션/코드북/템플릿/룰팩)

effective QA profile(name/version)

실행 환경(엔진 버전, git commit hash 가능하면)

추천: check_runs 테이블의 meta_json 같은 컬럼이 있으면 거기에 저장.
없다면 check_runs에 meta_json TEXT DEFAULT '{}' 추가하는 마이그레이션을 Step12에 더해도 돼.

active dataset snapshot을 가져오는 도구를 이미 만들었으니,

structai_check_run 시작 시:

active = structai_dataset_get_active_all()
qa = structai_qa_profile_get_effective(model_id)
# -> meta_json에 저장

12-6) VS Code UI 확장: QA 탭에 “거버넌스” 섹션 추가

Step11 QA 탭에서 아래를 추가하면 현장 운영이 된다:

Active datasets (type/name/version/sha256)

Effective QA profile (모델별 바인딩 보여주기)

Approval 상태

check_run 승인 요청/승인/반려

regression run 승인

compare report 승인

Signature

report artifact 선택 → sign/verify 버튼

필요한 서버 조회 도구

structai_dataset_get_active_all ✅ (Step12에 넣음)

structai_qa_profile_get_effective ✅

structai_approval_list/request/set_status ✅

structai_report_sign/verify ✅

UI는 Step11 QA 뷰의 웹뷰에 버튼 몇 개만 더 달면 바로 된다.

12-7) Step12 운영 순서(현장 배포 루틴)
1) 데이터셋 버전 등록/활성
#structai_dataset_register {
  "type": "section_catalog",
  "name": "ks-steel",
  "version": "2025.01.0",
  "artifact_path": "datasets/ks_steel_sections.csv",
  "activate": true
}

2) QA 프로파일 임포트/모델 바인딩
#structai_qa_profile_import { "path": "qa_profiles/qa_kds_prod.json", "activate": true }
#structai_qa_profile_bind_model { "model_id": 1, "name": "kds-prod", "version": "1.0.0" }

3) 체크 실행(체크 엔진이 effective QA 적용)
#structai_check_run { "model_id": 1, "analysis_run_id": 1 }

4) 리포트 생성 → 서명
#structai_report_sign { "artifact_id": 123, "signer": "kim@company", "method": "sha256", "note": "1차 검토 리포트" }
#structai_report_verify { "signature_id": 1 }

5) 승인 요청/승인
#structai_approval_request { "entity_type": "report", "entity_id": 123, "actor": "kim@company", "comment": "검토 완료, 승인 요청" }
#structai_approval_set_status { "approval_id": 1, "status": "approved", "actor": "lead@company", "comment": "승인" }

6) 변경 발생 시(데이터셋/룰 변경)

regression suite 실행 → FAIL이면 QA 탭에서 diff 확인 → 의도된 변경이면 golden 업데이트 → 다시 PASS 확인

새 데이터셋 버전 등록 → 활성 변경 → 결과 재현성 유지

12-8) 이 단계에서 “제품”이 되는 이유(핵심 정리)

QA 프로파일: 프로젝트 정책이 코드/사람 기억이 아니라 데이터로 고정됨

데이터셋 버전 관리: “어떤 단면/조항/템플릿으로 계산했는지”가 재현 가능

서명 + 승인: 결과를 변조/책임 소재/감사 관점에서 관리 가능

즉, 현장/감리/발주처 대응이 가능한 형태로 올라간다

원하면 다음(13단계)은 “팀/조직 운영”에 직접 꽂히는 것들인데, 예를 들면:

멀티프로젝트/멀티유저 권한(roles)

승인 단계(1차/2차/최종) 워크플로우

변경관리(데이터셋/룰 변경 시 자동 공지 + 영향 분석: 어떤 프로젝트 결과가 영향을 받는지)

“프로젝트 상태 대시보드(진행률, 누락, FAIL 부재, 승인 대기)”

이것도 같은 방식으로(스키마 + MCP + UI) 바로 이어서 구현해줄 수 있어.
