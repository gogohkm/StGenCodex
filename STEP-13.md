좋아. 이제 **13단계 = “조직/팀 운영 레벨(멀티프로젝트·멀티유저 권한 + 다단계 승인 + 변경관리/영향분석 + 프로젝트 대시보드)”**를 **바로 구현 가능한 수준(스키마 + MCP tools + VS Code UI 포인트)**로 순서대로 정리해서 제공할게.

핵심 목표

“누가(권한) → 무엇을(승인/확정) → 어떤 버전(데이터셋/룰/QA)으로 → 어떤 결과를 → 언제 승인했는지”가 감사 가능해야 함

변경이 생겼을 때(단면/룰/코드북 변경) “어떤 프로젝트/결과가 영향을 받는지”가 즉시 보이는 구조여야 함

현장 운영은 결국 대시보드와 워크플로우가 좌우함

13-1) DB 스키마 추가 (v0.1.4)

mcp_server/schema.sql 맨 아래에 추가해줘.

-- ===== schema patch v0.1.4 =====
UPDATE meta SET value='0.1.4' WHERE key='schema_version';

-- -----------------------------
-- Users / Roles
-- -----------------------------
CREATE TABLE IF NOT EXISTS users (
  user_id INTEGER PRIMARY KEY AUTOINCREMENT,
  actor TEXT NOT NULL UNIQUE,               -- logs use this stable key
  display_name TEXT,
  email TEXT,
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS roles (
  role_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,                -- admin|lead|reviewer|contributor|viewer...
  description TEXT,
  permissions_json TEXT NOT NULL DEFAULT '[]'
);

-- -----------------------------
-- Projects (multi-project)
-- -----------------------------
CREATE TABLE IF NOT EXISTS projects (
  project_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS project_models (
  project_id INTEGER NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  bound_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY(project_id, model_id)
);

CREATE TABLE IF NOT EXISTS project_memberships (
  project_id INTEGER NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  role_id INTEGER NOT NULL REFERENCES roles(role_id) ON DELETE RESTRICT,
  status TEXT NOT NULL DEFAULT 'active',     -- active|disabled
  joined_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY(project_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_project_memberships_role
ON project_memberships(project_id, role_id, status);

-- -----------------------------
-- Approval workflow (multi-step)
-- -----------------------------
CREATE TABLE IF NOT EXISTS approval_workflow_defs (
  workflow_def_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  entity_types_json TEXT NOT NULL DEFAULT '[]',  -- ["report","check_run","compare_run","regression_run","patch_run"]
  steps_json TEXT NOT NULL,                      -- list of steps
  is_active INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS project_workflows (
  project_id INTEGER PRIMARY KEY REFERENCES projects(project_id) ON DELETE CASCADE,
  workflow_def_id INTEGER NOT NULL REFERENCES approval_workflow_defs(workflow_def_id) ON DELETE RESTRICT,
  bound_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS approval_instances (
  instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
  workflow_def_id INTEGER NOT NULL REFERENCES approval_workflow_defs(workflow_def_id) ON DELETE RESTRICT,
  entity_type TEXT NOT NULL,
  entity_id INTEGER NOT NULL,
  project_id INTEGER REFERENCES projects(project_id) ON DELETE SET NULL,

  status TEXT NOT NULL DEFAULT 'in_progress',     -- in_progress|approved|rejected|revoked
  current_step_idx INTEGER NOT NULL DEFAULT 0,

  requested_by TEXT,
  requested_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),

  UNIQUE(workflow_def_id, entity_type, entity_id)
);

CREATE TABLE IF NOT EXISTS approval_votes (
  vote_id INTEGER PRIMARY KEY AUTOINCREMENT,
  instance_id INTEGER NOT NULL REFERENCES approval_instances(instance_id) ON DELETE CASCADE,
  step_idx INTEGER NOT NULL,
  actor TEXT NOT NULL,
  decision TEXT NOT NULL,                         -- approve|reject
  comment TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(instance_id, step_idx, actor)
);

CREATE INDEX IF NOT EXISTS idx_approval_instances_project
ON approval_instances(project_id, status, updated_at);

-- -----------------------------
-- Context snapshots for reproducibility (any entity)
-- -----------------------------
CREATE TABLE IF NOT EXISTS entity_contexts (
  entity_type TEXT NOT NULL,                      -- check_run|compare_run|regression_run|artifact|patch_run|...
  entity_id INTEGER NOT NULL,
  context_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY(entity_type, entity_id)
);

-- -----------------------------
-- Change management / notifications
-- -----------------------------
CREATE TABLE IF NOT EXISTS dataset_activation_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id INTEGER NOT NULL REFERENCES dataset_defs(dataset_id) ON DELETE CASCADE,
  from_version TEXT,
  to_version TEXT NOT NULL,
  actor TEXT,
  reason TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS project_events (
  project_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id INTEGER NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,                       -- dataset_change|approval_request|approval_decision|...
  message TEXT,
  payload_json TEXT NOT NULL DEFAULT '{}',
  actor TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_project_events_project
ON project_events(project_id, created_at);


참고

Step12의 approvals 테이블(단일 승인)은 그대로 두고, Step13의 approval_workflow_*를 “정식 승인 시스템”으로 쓰는 걸 추천(기존 approvals는 레거시/단순용으로 남겨둬도 됨).

13-2) 멀티유저 “ID(Actor)” 규칙부터 고정

VS Code 확장은 로그인 시스템이 없으니, 현실적으로는 환경변수/설정값으로 actor를 고정하는 방식이 가장 단단해.

env: STRUCTAI_ACTOR=kim@company

(없으면) OS 사용자명(USER/USERNAME) fallback

server.py에 추가:

import os

def _actor_from_env() -> str:
    return (os.environ.get("STRUCTAI_ACTOR")
            or os.environ.get("USER")
            or os.environ.get("USERNAME")
            or "unknown")


MCP tool:

@mcp.tool()
def structai_actor_whoami() -> Dict[str, Any]:
    return {"ok": True, "actor": _actor_from_env()}

13-3) 권한 모델(roles → permissions)
13-3-1) 권장 Role/Permission 매트릭스

admin: 모든 권한

lead: 승인/데이터셋 활성화/프로파일 바인딩/롤백

reviewer: 승인(리뷰 단계), 리포트 서명, suggested 확정

contributor: 매핑/스펙 추출/적용(패치 생성)

viewer: 조회만

권장 permission 문자열(예시):

project.manage

dataset.activate

qa.bind

workflow.manage

approval.request

approval.vote

mapping.confirm

spec.confirm

patch.apply

patch.rollback

report.sign

13-3-2) role upsert 도구
@mcp.tool()
def structai_role_upsert(name: str, permissions: List[str], description: str = "") -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO roles(name, description, permissions_json)
            VALUES(?,?,?)
            ON CONFLICT(name) DO UPDATE SET
              description=excluded.description,
              permissions_json=excluded.permissions_json
            """,
            (name, description, json.dumps(sorted(list(set(permissions))), ensure_ascii=False)),
        )
        conn.commit()
        r = conn.execute("SELECT role_id, name, description, permissions_json FROM roles WHERE name=?", (name,)).fetchone()
        out = dict(r)
        out["permissions"] = json.loads(out.pop("permissions_json") or "[]")
        return {"ok": True, "role": out}
    finally:
        conn.close()

13-3-3) 권한 체크 헬퍼(강제/선택)
def _has_permission(conn, actor: str, project_id: int, perm: str) -> bool:
    u = conn.execute("SELECT user_id FROM users WHERE actor=? AND is_active=1", (actor,)).fetchone()
    if not u:
        return False
    user_id = int(u["user_id"])
    m = conn.execute(
        """
        SELECT r.permissions_json
        FROM project_memberships pm
        JOIN roles r ON r.role_id = pm.role_id
        WHERE pm.project_id=? AND pm.user_id=? AND pm.status='active'
        """,
        (int(project_id), user_id),
    ).fetchone()
    if not m:
        return False
    perms = set(json.loads(m["permissions_json"] or "[]"))
    return ("*" in perms) or (perm in perms)

def _require_permission(conn, project_id: int, perm: str):
    actor = _actor_from_env()
    # 개발 편의: unknown이면 강제 안 걸고 경고만(원하면 여기서 예외 던지도록 바꿔도 됨)
    if actor == "unknown":
        return
    if not _has_permission(conn, actor, project_id, perm):
        raise PermissionError(f"actor={actor} missing permission={perm} for project_id={project_id}")

13-4) 프로젝트 생성/모델 바인딩/멤버십 관리 도구
13-4-1) user upsert
@mcp.tool()
def structai_user_upsert(actor: str, display_name: str = "", email: str = "", is_active: bool = True) -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO users(actor, display_name, email, is_active)
            VALUES(?,?,?,?)
            ON CONFLICT(actor) DO UPDATE SET
              display_name=excluded.display_name,
              email=excluded.email,
              is_active=excluded.is_active
            """,
            (actor, display_name or None, email or None, 1 if is_active else 0),
        )
        conn.commit()
        row = conn.execute("SELECT user_id, actor, display_name, email, is_active FROM users WHERE actor=?", (actor,)).fetchone()
        return {"ok": True, "user": dict(row)}
    finally:
        conn.close()

13-4-2) project create + bind model + add member
@mcp.tool()
def structai_project_create(name: str, description: str = "") -> Dict[str, Any]:
    conn = _connect()
    try:
        conn.execute("INSERT OR IGNORE INTO projects(name, description) VALUES(?,?)", (name, description))
        conn.commit()
        p = conn.execute("SELECT project_id, name, description FROM projects WHERE name=?", (name,)).fetchone()
        return {"ok": True, "project": dict(p)}
    finally:
        conn.close()

@mcp.tool()
def structai_project_bind_model(project_id: int, model_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        _require_permission(conn, project_id, "project.manage")
        conn.execute(
            "INSERT OR REPLACE INTO project_models(project_id, model_id, bound_at) VALUES(?,?, datetime('now'))",
            (int(project_id), int(model_id)),
        )
        conn.commit()
        return {"ok": True, "project_id": int(project_id), "model_id": int(model_id)}
    finally:
        conn.close()

@mcp.tool()
def structai_project_add_member(project_id: int, actor: str, role_name: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        _require_permission(conn, project_id, "project.manage")

        u = conn.execute("SELECT user_id FROM users WHERE actor=?", (actor,)).fetchone()
        if not u:
            raise ValueError("user not found; run structai_user_upsert first")
        user_id = int(u["user_id"])

        r = conn.execute("SELECT role_id FROM roles WHERE name=?", (role_name,)).fetchone()
        if not r:
            raise ValueError("role not found; run structai_role_upsert first")
        role_id = int(r["role_id"])

        conn.execute(
            """
            INSERT INTO project_memberships(project_id, user_id, role_id, status)
            VALUES(?,?,?, 'active')
            ON CONFLICT(project_id, user_id) DO UPDATE SET
              role_id=excluded.role_id,
              status='active'
            """,
            (int(project_id), user_id, role_id),
        )
        conn.commit()
        return {"ok": True, "project_id": int(project_id), "actor": actor, "role": role_name}
    finally:
        conn.close()

13-5) 다단계 승인 워크플로우(1차/2차/최종)
13-5-1) workflow steps JSON 포맷(권장)

예: workflow_kds_default.json

{
  "name": "kds-default",
  "version": "1.0.0",
  "entity_types": ["report", "check_run", "compare_run", "regression_run"],
  "steps": [
    { "id": "peer",  "title": "1차(동료검토)", "required_role": "reviewer", "required_count": 1, "distinct_actors": true },
    { "id": "lead",  "title": "2차(팀장승인)", "required_role": "lead",     "required_count": 1, "distinct_actors": true },
    { "id": "final", "title": "최종(발행)",    "required_role": "admin",    "required_count": 1, "distinct_actors": true }
  ]
}


중요한 운영 원칙

distinct_actors=true: 같은 사람이 1차/2차/최종을 다 하지 못하게(감사/품질)

required_count를 2로 늘리면 “두 명 리뷰”도 바로 가능

13-5-2) workflow import / activate / bind to project
@mcp.tool()
def structai_workflow_import(path: str, activate: bool = False) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    name = data.get("name") or p.stem
    ver = data.get("version") or "0.0.0"
    entity_types = data.get("entity_types") or []
    steps = data.get("steps") or []
    if not isinstance(steps, list) or not steps:
        raise ValueError("workflow steps must be a non-empty list")

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO approval_workflow_defs(name, version, entity_types_json, steps_json, is_active)
            VALUES(?,?,?,?,0)
            ON CONFLICT(name, version) DO UPDATE SET
              entity_types_json=excluded.entity_types_json,
              steps_json=excluded.steps_json
            """,
            (name, ver, json.dumps(entity_types, ensure_ascii=False), json.dumps(steps, ensure_ascii=False)),
        )
        if activate:
            conn.execute("UPDATE approval_workflow_defs SET is_active=0")
            conn.execute("UPDATE approval_workflow_defs SET is_active=1 WHERE name=? AND version=?", (name, ver))
        conn.commit()
        row = conn.execute("SELECT workflow_def_id, name, version, is_active FROM approval_workflow_defs WHERE name=? AND version=?", (name, ver)).fetchone()
        return {"ok": True, "workflow": dict(row)}
    finally:
        conn.close()

@mcp.tool()
def structai_workflow_bind_project(project_id: int, name: str, version: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        _require_permission(conn, project_id, "workflow.manage")
        w = conn.execute("SELECT workflow_def_id FROM approval_workflow_defs WHERE name=? AND version=?", (name, version)).fetchone()
        if not w:
            raise ValueError("workflow not found")
        wid = int(w["workflow_def_id"])
        conn.execute(
            """
            INSERT INTO project_workflows(project_id, workflow_def_id, bound_at)
            VALUES(?,?, datetime('now'))
            ON CONFLICT(project_id) DO UPDATE SET
              workflow_def_id=excluded.workflow_def_id,
              bound_at=datetime('now')
            """,
            (int(project_id), wid),
        )
        conn.commit()
        return {"ok": True, "project_id": int(project_id), "workflow_def_id": wid}
    finally:
        conn.close()

13-5-3) approval request / vote / status
def _resolve_workflow_for_project(conn, project_id: int, entity_type: str) -> int:
    # project binding 우선
    row = conn.execute(
        """
        SELECT w.workflow_def_id, w.entity_types_json
        FROM project_workflows pw
        JOIN approval_workflow_defs w ON w.workflow_def_id = pw.workflow_def_id
        WHERE pw.project_id=?
        """,
        (int(project_id),),
    ).fetchone()

    if row:
        ets = json.loads(row["entity_types_json"] or "[]")
        if (not ets) or (entity_type in ets):
            return int(row["workflow_def_id"])

    # global active fallback
    row = conn.execute(
        "SELECT workflow_def_id, entity_types_json FROM approval_workflow_defs WHERE is_active=1 ORDER BY workflow_def_id DESC LIMIT 1"
    ).fetchone()
    if not row:
        raise ValueError("No workflow configured (bind project or set active workflow)")
    ets = json.loads(row["entity_types_json"] or "[]")
    if ets and (entity_type not in ets):
        raise ValueError("Active workflow does not support entity_type")
    return int(row["workflow_def_id"])

@mcp.tool()
def structai_approval_request_v2(project_id: int, entity_type: str, entity_id: int, comment: str = "") -> Dict[str, Any]:
    conn = _connect()
    try:
        _require_permission(conn, project_id, "approval.request")

        actor = _actor_from_env()
        wid = _resolve_workflow_for_project(conn, project_id, entity_type)

        # create instance
        conn.execute(
            """
            INSERT OR IGNORE INTO approval_instances(workflow_def_id, entity_type, entity_id, project_id, status, current_step_idx, requested_by)
            VALUES(?,?,?,?, 'in_progress', 0, ?)
            """,
            (wid, entity_type, int(entity_id), int(project_id), actor),
        )

        inst = conn.execute(
            "SELECT instance_id, status, current_step_idx FROM approval_instances WHERE workflow_def_id=? AND entity_type=? AND entity_id=?",
            (wid, entity_type, int(entity_id)),
        ).fetchone()

        # event
        conn.execute(
            "INSERT INTO project_events(project_id, event_type, message, payload_json, actor) VALUES(?,?,?,?,?)",
            (int(project_id), "approval_request", f"Approval requested for {entity_type}#{entity_id}", json.dumps({"entity_type":entity_type,"entity_id":int(entity_id),"comment":comment}, ensure_ascii=False), actor),
        )

        conn.commit()
        return {"ok": True, "instance_id": int(inst["instance_id"]), "status": inst["status"], "step": int(inst["current_step_idx"])}
    finally:
        conn.close()

def _step_requirements(steps: List[dict], idx: int) -> dict:
    if idx < 0 or idx >= len(steps):
        return {}
    return steps[idx]

def _role_of_actor_in_project(conn, project_id: int, actor: str) -> Optional[str]:
    u = conn.execute("SELECT user_id FROM users WHERE actor=? AND is_active=1", (actor,)).fetchone()
    if not u:
        return None
    row = conn.execute(
        """
        SELECT r.name
        FROM project_memberships pm
        JOIN roles r ON r.role_id = pm.role_id
        WHERE pm.project_id=? AND pm.user_id=? AND pm.status='active'
        """,
        (int(project_id), int(u["user_id"])),
    ).fetchone()
    return row["name"] if row else None

@mcp.tool()
def structai_approval_vote(instance_id: int, decision: str, comment: str = "") -> Dict[str, Any]:
    if decision not in ("approve","reject"):
        raise ValueError("decision must be approve|reject")

    conn = _connect()
    try:
        actor = _actor_from_env()

        inst = conn.execute(
            "SELECT instance_id, workflow_def_id, project_id, status, current_step_idx, entity_type, entity_id FROM approval_instances WHERE instance_id=?",
            (int(instance_id),),
        ).fetchone()
        if not inst:
            raise ValueError("instance not found")
        if inst["status"] != "in_progress":
            raise ValueError(f"instance not in progress: {inst['status']}")

        project_id = int(inst["project_id"]) if inst["project_id"] is not None else None
        if project_id is None:
            raise ValueError("instance has no project_id")

        _require_permission(conn, project_id, "approval.vote")

        w = conn.execute("SELECT steps_json FROM approval_workflow_defs WHERE workflow_def_id=?", (int(inst["workflow_def_id"]),)).fetchone()
        steps = json.loads(w["steps_json"] or "[]")
        step_idx = int(inst["current_step_idx"])
        step = _step_requirements(steps, step_idx)

        # role check
        required_role = step.get("required_role")
        actor_role = _role_of_actor_in_project(conn, project_id, actor)
        if required_role and actor_role != required_role and actor_role != "admin":
            raise PermissionError(f"actor role={actor_role} cannot vote on step requires={required_role}")

        # distinct actor constraint across steps
        if bool(step.get("distinct_actors", True)):
            prev = conn.execute(
                "SELECT 1 FROM approval_votes WHERE instance_id=? AND actor=?",
                (int(instance_id), actor),
            ).fetchone()
            if prev:
                raise ValueError("distinct_actors: same actor already voted in this workflow instance")

        # insert vote
        conn.execute(
            """
            INSERT INTO approval_votes(instance_id, step_idx, actor, decision, comment)
            VALUES(?,?,?,?,?)
            ON CONFLICT(instance_id, step_idx, actor) DO UPDATE SET
              decision=excluded.decision,
              comment=excluded.comment,
              created_at=datetime('now')
            """,
            (int(instance_id), step_idx, actor, decision, comment),
        )

        # decision rule:
        # - any reject at current step -> rejected
        # - else if approvals >= required_count -> advance step or approved
        votes = conn.execute(
            "SELECT decision FROM approval_votes WHERE instance_id=? AND step_idx=?",
            (int(instance_id), step_idx),
        ).fetchall()
        decisions = [v["decision"] for v in votes]
        if "reject" in decisions:
            conn.execute(
                "UPDATE approval_instances SET status='rejected', updated_at=datetime('now') WHERE instance_id=?",
                (int(instance_id),),
            )
            new_status = "rejected"
        else:
            req = int(step.get("required_count", 1))
            approves = sum(1 for d in decisions if d == "approve")
            if approves >= req:
                if step_idx + 1 >= len(steps):
                    conn.execute(
                        "UPDATE approval_instances SET status='approved', updated_at=datetime('now') WHERE instance_id=?",
                        (int(instance_id),),
                    )
                    new_status = "approved"
                else:
                    conn.execute(
                        "UPDATE approval_instances SET current_step_idx=?, updated_at=datetime('now') WHERE instance_id=?",
                        (step_idx + 1, int(instance_id)),
                    )
                    new_status = "in_progress"
            else:
                new_status = "in_progress"

        # project event
        conn.execute(
            "INSERT INTO project_events(project_id, event_type, message, payload_json, actor) VALUES(?,?,?,?,?)",
            (project_id, "approval_decision",
             f"{decision} on {inst['entity_type']}#{inst['entity_id']} step={step_idx}",
             json.dumps({"instance_id":int(instance_id),"decision":decision,"step_idx":step_idx,"status":new_status}, ensure_ascii=False),
             actor),
        )

        conn.commit()
        return {"ok": True, "instance_id": int(instance_id), "status": new_status}
    finally:
        conn.close()

@mcp.tool()
def structai_approval_read(entity_type: str, entity_id: int, project_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        wid = _resolve_workflow_for_project(conn, project_id, entity_type)
        inst = conn.execute(
            "SELECT * FROM approval_instances WHERE workflow_def_id=? AND entity_type=? AND entity_id=?",
            (wid, entity_type, int(entity_id)),
        ).fetchone()
        if not inst:
            return {"ok": True, "exists": False}

        w = conn.execute("SELECT name, version, steps_json FROM approval_workflow_defs WHERE workflow_def_id=?", (wid,)).fetchone()
        steps = json.loads(w["steps_json"] or "[]")

        votes = conn.execute(
            "SELECT step_idx, actor, decision, comment, created_at FROM approval_votes WHERE instance_id=? ORDER BY vote_id ASC",
            (int(inst["instance_id"]),),
        ).fetchall()

        return {
            "ok": True,
            "exists": True,
            "workflow": {"name": w["name"], "version": w["version"], "steps": steps},
            "instance": dict(inst),
            "votes": [dict(v) for v in votes],
        }
    finally:
        conn.close()

13-6) 변경관리(데이터셋/룰 변경 시 자동 공지 + 영향 분석)
13-6-1) “컨텍스트 스냅샷”을 반드시 남기기

Step12에서 결과 재현성을 위해 entity_contexts에 저장하는 걸 제안했지. Step13에서 그걸 “표준 운영 절차”로 확정해야 영향 분석이 가능해져.

도구: entity context capture / get
def _context_snapshot(conn, model_id: Optional[int] = None, project_id: Optional[int] = None) -> Dict[str, Any]:
    # active datasets snapshot
    active = conn.execute(
        """
        SELECT d.type, d.name, v.version, v.sha256, v.artifact_id
        FROM dataset_defs d
        JOIN dataset_versions v ON v.dataset_id=d.dataset_id
        WHERE v.is_active=1
        ORDER BY d.type, d.name
        """
    ).fetchall()
    datasets = [dict(r) for r in active]

    # effective qa profile
    qa = _get_effective_qa_profile(conn, model_id) if model_id is not None else None

    return {
        "actor": _actor_from_env(),
        "model_id": model_id,
        "project_id": project_id,
        "datasets_active": datasets,
        "qa_profile": qa,
        "env": {
            "engine_version": os.environ.get("STRUCTAI_ENGINE_VERSION"),
            "git_sha": os.environ.get("GIT_SHA"),
        },
        "captured_at": datetime.utcnow().isoformat() + "Z",
    }

@mcp.tool()
def structai_context_capture(entity_type: str, entity_id: int, model_id: Optional[int] = None, project_id: Optional[int] = None) -> Dict[str, Any]:
    conn = _connect()
    try:
        ctx = _context_snapshot(conn, model_id=model_id, project_id=project_id)
        conn.execute(
            """
            INSERT INTO entity_contexts(entity_type, entity_id, context_json, created_at, updated_at)
            VALUES(?,?,?, datetime('now'), datetime('now'))
            ON CONFLICT(entity_type, entity_id) DO UPDATE SET
              context_json=excluded.context_json,
              updated_at=datetime('now')
            """,
            (entity_type, int(entity_id), json.dumps(ctx, ensure_ascii=False)),
        )
        conn.commit()
        return {"ok": True, "entity_type": entity_type, "entity_id": int(entity_id), "context": ctx}
    finally:
        conn.close()

@mcp.tool()
def structai_context_get(entity_type: str, entity_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        r = conn.execute("SELECT context_json, updated_at FROM entity_contexts WHERE entity_type=? AND entity_id=?", (entity_type, int(entity_id))).fetchone()
        if not r:
            return {"ok": True, "exists": False}
        return {"ok": True, "exists": True, "context": json.loads(r["context_json"] or "{}"), "updated_at": r["updated_at"]}
    finally:
        conn.close()


운영 팁

structai_check_run 실행 직후 structai_context_capture(entity_type="check_run", entity_id=check_run_id, model_id, project_id) 호출을 자동화하면 베스트.

report 생성 시에도 artifact_id에 대해 context를 남겨두면 “리포트 승인/서명”이 프로젝트에 매핑된다.

13-6-2) dataset 활성 변경 + 영향 분석 + 프로젝트 공지

dataset 활성 변경은 Step12에서도 했지만, Step13에선 “영향분석/공지”까지 묶어야 운영이 된다.
따라서 새 도구를 하나로 묶는 걸 추천해.

def _active_version(conn, dataset_id: int) -> Optional[str]:
    r = conn.execute("SELECT version FROM dataset_versions WHERE dataset_id=? AND is_active=1", (dataset_id,)).fetchone()
    return r["version"] if r else None

@mcp.tool()
def structai_dataset_set_active_notify(
    type: str,
    name: str,
    version: str,
    reason: str = ""
) -> Dict[str, Any]:
    """
    1) dataset 활성 버전 변경
    2) activation event 기록
    3) 영향분석(어떤 프로젝트/모델의 최근 check_run이 구버전으로 계산되었는지)
    4) project_events에 공지 생성
    """
    conn = _connect()
    try:
        actor = _actor_from_env()
        d = conn.execute("SELECT dataset_id FROM dataset_defs WHERE type=? AND name=?", (type, name)).fetchone()
        if not d:
            raise ValueError("dataset not found")
        dataset_id = int(d["dataset_id"])

        from_ver = _active_version(conn, dataset_id)

        # activate
        conn.execute("UPDATE dataset_versions SET is_active=0 WHERE dataset_id=?", (dataset_id,))
        conn.execute("UPDATE dataset_versions SET is_active=1 WHERE dataset_id=? AND version=?", (dataset_id, version))
        conn.execute(
            "INSERT INTO dataset_activation_events(dataset_id, from_version, to_version, actor, reason) VALUES(?,?,?,?,?)",
            (dataset_id, from_ver, version, actor, reason),
        )
        conn.commit()

        # impact analysis: find models where latest check_run context used old version
        impacted = []

        # latest check_run per model (simple: max check_run_id)
        models = conn.execute("SELECT DISTINCT model_id FROM models").fetchall()
        for m in models:
            model_id = int(m["model_id"])
            cr = conn.execute("SELECT MAX(check_run_id) AS last_id FROM check_runs WHERE model_id=?", (model_id,)).fetchone()
            if not cr or not cr["last_id"]:
                continue
            last_id = int(cr["last_id"])

            ctx = conn.execute("SELECT context_json FROM entity_contexts WHERE entity_type='check_run' AND entity_id=?", (last_id,)).fetchone()
            if not ctx:
                continue
            c = json.loads(ctx["context_json"] or "{}")
            ds = c.get("datasets_active") or []
            used = None
            for item in ds:
                if item.get("type") == type and item.get("name") == name:
                    used = item.get("version")
                    break
            if used and used != version:
                impacted.append({"model_id": model_id, "check_run_id": last_id, "used_version": used})

        # map impacted models -> projects + create events
        proj_map = {}
        for it in impacted:
            rows = conn.execute("SELECT project_id FROM project_models WHERE model_id=?", (int(it["model_id"]),)).fetchall()
            for r in rows:
                pid = int(r["project_id"])
                proj_map.setdefault(pid, [])
                proj_map[pid].append(it)

        for pid, arr in proj_map.items():
            msg = f"Dataset changed: {type}:{name} {from_ver or '?'} -> {version}. Impacted models={len(set(x['model_id'] for x in arr))}"
            conn.execute(
                "INSERT INTO project_events(project_id, event_type, message, payload_json, actor) VALUES(?,?,?,?,?)",
                (pid, "dataset_change", msg, json.dumps({"dataset":{"type":type,"name":name,"from":from_ver,"to":version},"impacted":arr}, ensure_ascii=False), actor),
            )

        conn.commit()
        return {"ok": True, "dataset": {"type": type, "name": name, "from": from_ver, "to": version}, "impacted": impacted, "projects_notified": list(proj_map.keys())}
    finally:
        conn.close()


이게 “자동 공지 + 영향 분석”의 핵심이야.

중요: 영향 분석이 정확해지려면 check_run마다 entity_contexts가 있어야 함(그래서 13-6-1이 필수).

13-7) 프로젝트 상태 대시보드(진행률/누락/FAIL/승인대기)
13-7-1) dashboard 도구(핵심 지표만 MVP)
@mcp.tool()
def structai_project_dashboard(project_id: int, limit_events: int = 30) -> Dict[str, Any]:
    conn = _connect()
    try:
        p = conn.execute("SELECT project_id, name, description FROM projects WHERE project_id=?", (int(project_id),)).fetchone()
        if not p:
            raise ValueError("project not found")

        models = conn.execute(
            """
            SELECT m.model_id, m.name
            FROM project_models pm
            JOIN models m ON m.model_id = pm.model_id
            WHERE pm.project_id=?
            ORDER BY m.model_id ASC
            """,
            (int(project_id),),
        ).fetchall()

        model_cards = []
        for m in models:
            model_id = int(m["model_id"])

            total_members = conn.execute("SELECT COUNT(*) AS n FROM model_members WHERE model_id=?", (model_id,)).fetchone()["n"]

            # mapping coverage: distinct members with confirmed mapping
            mapped = conn.execute(
                """
                SELECT COUNT(DISTINCT model_member_id) AS n
                FROM member_mappings
                WHERE model_id=? AND status='confirmed'
                """,
                (model_id,),
            ).fetchone()["n"]

            # spec coverage: distinct members with confirmed spec link
            spec_cov = conn.execute(
                """
                SELECT COUNT(DISTINCT model_member_id) AS n
                FROM member_spec_links
                WHERE model_id=? AND status='confirmed'
                """,
                (model_id,),
            ).fetchone()["n"]

            # latest check run & FAIL counts
            last = conn.execute("SELECT MAX(check_run_id) AS last_id FROM check_runs WHERE model_id=?", (model_id,)).fetchone()
            last_id = int(last["last_id"]) if last and last["last_id"] else None

            check_summary = None
            if last_id:
                counts = conn.execute(
                    "SELECT status, COUNT(*) AS n FROM check_results WHERE check_run_id=? GROUP BY status",
                    (last_id,),
                ).fetchall()
                check_summary = {r["status"]: int(r["n"]) for r in counts}

            # approvals pending in this project for check_run/report etc
            pending = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM approval_instances
                WHERE project_id=? AND status='in_progress'
                """,
                (int(project_id),),
            ).fetchone()["n"]

            model_cards.append({
                "model_id": model_id,
                "model_name": m["name"],
                "members_total": int(total_members),
                "mapping_confirmed": int(mapped),
                "mapping_pct": (float(mapped)/float(total_members)*100.0) if total_members else 0.0,
                "spec_confirmed_members": int(spec_cov),
                "spec_pct": (float(spec_cov)/float(total_members)*100.0) if total_members else 0.0,
                "latest_check_run_id": last_id,
                "check_summary": check_summary,
                "approvals_in_progress": int(pending),
            })

        events = conn.execute(
            """
            SELECT event_type, message, payload_json, actor, created_at
            FROM project_events
            WHERE project_id=?
            ORDER BY project_event_id DESC
            LIMIT ?
            """,
            (int(project_id), int(limit_events)),
        ).fetchall()
        ev = []
        for e in events:
            it = dict(e)
            it["payload"] = json.loads(it.pop("payload_json") or "{}")
            ev.append(it)

        # members list (with roles)
        members = conn.execute(
            """
            SELECT u.actor, u.display_name, r.name as role_name, pm.status
            FROM project_memberships pm
            JOIN users u ON u.user_id = pm.user_id
            JOIN roles r ON r.role_id = pm.role_id
            WHERE pm.project_id=?
            ORDER BY r.name, u.actor
            """,
            (int(project_id),),
        ).fetchall()

        return {
            "ok": True,
            "project": dict(p),
            "memberships": [dict(x) for x in members],
            "models": model_cards,
            "events": ev,
        }
    finally:
        conn.close()


대시보드 MVP는 이것만 있어도 충분히 “운영”이 된다:

진행률(매핑/스펙)

최근 체크 결과(FAIL/WARN/NA)

승인 대기 개수

이벤트(변경/승인 요청/결정 로그)

13-8) VS Code UI 확장(Projects + Governance)

Step11의 QA 탭이 이미 있으니, Step13에서는 탭 2개만 추가하면 실무 운영이 가능해.

13-8-1) Projects View

프로젝트 목록

선택한 프로젝트 dashboard

이벤트 피드

멤버십/역할 보기

필요 MCP 호출:

structai_project_dashboard

(선택) structai_project_create, structai_project_bind_model, structai_project_add_member

13-8-2) Governance View

승인 요청 목록(진행중 인스턴스)

instance 상세(현재 단계/투표/다음 단계)

approve/reject 버튼

dataset 변경 이벤트 + 영향 분석 결과

추가로 “승인 UX”를 좋게 만들려면:

“현재 내가 투표 가능한 step인지”를 서버에서 함께 반환(권한 체크)

13-9) Step13 운영 순서(팀/조직 루틴)

role/permissions 등록

#structai_role_upsert { "name":"admin", "permissions":["*"], "description":"all permissions" }
#structai_role_upsert { "name":"lead", "permissions":["project.manage","dataset.activate","qa.bind","workflow.manage","approval.request","approval.vote","patch.rollback","report.sign"], "description":"team lead" }
#structai_role_upsert { "name":"reviewer", "permissions":["approval.vote","report.sign","mapping.confirm","spec.confirm"], "description":"peer reviewer" }
#structai_role_upsert { "name":"contributor", "permissions":["patch.apply","mapping.confirm","spec.confirm"], "description":"engineer" }
#structai_role_upsert { "name":"viewer", "permissions":[], "description":"read-only" }


프로젝트 생성/모델 바인딩/멤버 초대

#structai_project_create { "name":"PJ-001", "description":"A동 구조검토" }
#structai_user_upsert { "actor":"kim@company", "display_name":"Kim", "email":"kim@company" }
#structai_user_upsert { "actor":"lee@company", "display_name":"Lee", "email":"lee@company" }
#structai_project_add_member { "project_id": 1, "actor":"kim@company", "role_name":"lead" }
#structai_project_add_member { "project_id": 1, "actor":"lee@company", "role_name":"reviewer" }
#structai_project_bind_model { "project_id": 1, "model_id": 1 }


승인 워크플로우 등록 + 프로젝트 바인딩

#structai_workflow_import { "path":"workflow_kds_default.json", "activate": true }
#structai_workflow_bind_project { "project_id": 1, "name":"kds-default", "version":"1.0.0" }


체크 실행 후 context capture(재현성)

#structai_check_run { "model_id": 1, "analysis_run_id": 1 }
#structai_context_capture { "entity_type":"check_run", "entity_id": 1, "model_id": 1, "project_id": 1 }


리포트 생성 후 artifact context capture + 승인 요청

#structai_report_sign { "artifact_id": 123, "signer": "kim@company", "method":"sha256", "note":"검토 리포트" }
#structai_context_capture { "entity_type":"artifact", "entity_id": 123, "model_id": 1, "project_id": 1 }
#structai_approval_request_v2 { "project_id": 1, "entity_type":"report", "entity_id": 123, "comment":"1차 검토 요청" }


reviewer가 1차 승인 → lead가 2차 → admin이 최종

#structai_approval_vote { "instance_id": 1, "decision":"approve", "comment":"OK" }
#structai_approval_vote { "instance_id": 1, "decision":"approve", "comment":"승인" }
#structai_approval_vote { "instance_id": 1, "decision":"approve", "comment":"최종 발행" }


데이터셋 활성 버전 변경 시 자동 공지/영향 분석

#structai_dataset_set_active_notify {
  "type":"section_catalog",
  "name":"ks-steel",
  "version":"2025.02.0",
  "reason":"KS 단면 카탈로그 업데이트"
}

13단계 완료 체크리스트

 프로젝트 단위 관리(모델 묶기)

 사용자/역할/권한

 다단계 승인(1차/2차/최종) + 역할 제약

 컨텍스트 스냅샷으로 재현성 확보

 데이터셋 변경 이벤트 + 영향 분석 + 프로젝트 공지

 대시보드로 진행률/누락/FAIL/승인대기를 한 화면에서 확인

원하면 다음(14단계)은 “조직 확장” 단계로 갈 수 있어. 예를 들어:

사내 SSO/계정 연동(실제 인증)

중앙 서버(멀티클라이언트) + 작업 락/동시성

프로젝트 템플릿(프로파일/워크플로우/룰팩/데이터셋 번들)

“영향 분석 고도화”: 데이터셋 변경이 어떤 check_type/worst_ratio에 얼마나 영향을 주는지 자동 재계산(샘플 기반)

하지만 지금 Step13까지면, 이미 현장/감리 대응 가능한 운영체계로 완성된 수준이야.
