PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version','0.1.4');
UPDATE meta SET value='0.1.4' WHERE key='schema_version';

-- Artifacts
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  uri TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  title TEXT,
  source_path TEXT,
  sha256 TEXT,
  meta_json TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS doc_chunks (
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  page_start INTEGER,
  page_end INTEGER,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_artifact ON doc_chunks(artifact_id, chunk_index);

-- CAD entities
CREATE TABLE IF NOT EXISTS cad_entities (
  cad_entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  chunk_id INTEGER REFERENCES doc_chunks(chunk_id) ON DELETE SET NULL,
  handle TEXT,
  type TEXT NOT NULL,
  layer TEXT,
  layout TEXT,
  text TEXT,
  x REAL, y REAL, z REAL,
  geom_json TEXT,
  raw_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cad_entities_artifact ON cad_entities(artifact_id, type);
CREATE INDEX IF NOT EXISTS idx_cad_entities_layer ON cad_entities(artifact_id, layer);
CREATE INDEX IF NOT EXISTS idx_cad_entities_text ON cad_entities(artifact_id, type, text);

-- Models
CREATE TABLE IF NOT EXISTS models (
  model_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  source_path TEXT,
  units TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_members (
  model_member_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  member_uid TEXT NOT NULL,
  member_label TEXT,
  label_norm TEXT,
  type TEXT,
  x1 REAL, y1 REAL, z1 REAL,
  x2 REAL, y2 REAL, z2 REAL,
  section TEXT,
  story TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(model_id, member_uid)
);
CREATE INDEX IF NOT EXISTS idx_model_members_label ON model_members(model_id, label_norm);
CREATE INDEX IF NOT EXISTS idx_model_members_type ON model_members(model_id, type);
CREATE INDEX IF NOT EXISTS idx_model_members_story ON model_members(model_id, story);

CREATE TABLE IF NOT EXISTS member_mappings (
  mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  cad_token TEXT NOT NULL,
  cad_token_norm TEXT NOT NULL,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,
  confidence REAL NOT NULL,
  method TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'suggested',
  evidence_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(cad_artifact_id, cad_token_norm, model_id, model_member_id)
);
CREATE INDEX IF NOT EXISTS idx_member_mappings_lookup
ON member_mappings(cad_artifact_id, model_id, status, cad_token_norm);

-- Analysis results
CREATE TABLE IF NOT EXISTS analysis_runs (
  analysis_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  engine TEXT,
  units TEXT,
  source_path TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS member_results (
  result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(analysis_run_id) ON DELETE CASCADE,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,
  combo TEXT NOT NULL,
  envelope_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(analysis_run_id, model_member_id, combo)
);

CREATE INDEX IF NOT EXISTS idx_member_results_lookup
ON member_results(analysis_run_id, combo, model_member_id);

-- Design inputs
CREATE TABLE IF NOT EXISTS member_design_inputs (
  model_member_id INTEGER PRIMARY KEY REFERENCES model_members(model_member_id) ON DELETE CASCADE,
  design_json TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Rulepacks
CREATE TABLE IF NOT EXISTS rulepacks (
  rulepack_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  rulepack_json TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_rulepacks_namever ON rulepacks(name, version);

-- Check runs + results
CREATE TABLE IF NOT EXISTS check_runs (
  check_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  analysis_run_id INTEGER NOT NULL REFERENCES analysis_runs(analysis_run_id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  rulepack_name TEXT,
  rulepack_version TEXT,
  combos_json TEXT NOT NULL,
  check_types_json TEXT NOT NULL,
  scope_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS check_results (
  check_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  check_run_id INTEGER NOT NULL REFERENCES check_runs(check_run_id) ON DELETE CASCADE,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,
  combo TEXT NOT NULL,
  check_type TEXT NOT NULL,
  demand_value REAL,
  capacity_value REAL,
  ratio REAL,
  status TEXT NOT NULL,
  details_json TEXT NOT NULL DEFAULT '{}',
  citations_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_check_results_run ON check_results(check_run_id, status);
CREATE INDEX IF NOT EXISTS idx_check_results_member ON check_results(model_member_id);

-- Reports
CREATE TABLE IF NOT EXISTS reports (
  report_id INTEGER PRIMARY KEY AUTOINCREMENT,
  check_run_id INTEGER NOT NULL REFERENCES check_runs(check_run_id) ON DELETE CASCADE,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  format TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_reports_check_run ON reports(check_run_id, format);

-- CAD specs and codebooks
CREATE TABLE IF NOT EXISTS cad_specs (
  spec_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  cad_entity_id INTEGER REFERENCES cad_entities(cad_entity_id) ON DELETE SET NULL,
  spec_kind TEXT NOT NULL,
  spec_json TEXT NOT NULL,
  raw_text TEXT NOT NULL,
  x REAL, y REAL, z REAL,
  layer TEXT,
  confidence REAL NOT NULL DEFAULT 0.5,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_specs_artifact ON cad_specs(cad_artifact_id, spec_kind);
CREATE INDEX IF NOT EXISTS idx_cad_specs_entity ON cad_specs(cad_entity_id);

CREATE TABLE IF NOT EXISTS member_spec_links (
  link_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  spec_id INTEGER NOT NULL REFERENCES cad_specs(spec_id) ON DELETE CASCADE,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,
  cad_token_norm TEXT,
  distance REAL,
  method TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'suggested',
  evidence_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(cad_artifact_id, spec_id, model_member_id)
);

CREATE INDEX IF NOT EXISTS idx_member_spec_links_lookup
ON member_spec_links(cad_artifact_id, model_id, status, model_member_id);

CREATE TABLE IF NOT EXISTS codebooks (
  codebook_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  codebook_json TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_codebooks_namever ON codebooks(name, version);

-- Section catalog and tables
CREATE TABLE IF NOT EXISTS section_catalog (
  section_id INTEGER PRIMARY KEY AUTOINCREMENT,
  family TEXT NOT NULL,
  name TEXT NOT NULL,
  name_norm TEXT NOT NULL,
  dims_json TEXT NOT NULL DEFAULT '{}',
  props_json TEXT NOT NULL DEFAULT '{}',
  source TEXT,
  priority INTEGER NOT NULL DEFAULT 50,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_section_catalog_family_name
ON section_catalog(family, name_norm);

CREATE TABLE IF NOT EXISTS section_aliases (
  alias_norm TEXT PRIMARY KEY,
  section_id INTEGER NOT NULL REFERENCES section_catalog(section_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS member_section_resolutions (
  model_member_id INTEGER PRIMARY KEY REFERENCES model_members(model_member_id) ON DELETE CASCADE,
  section_id INTEGER REFERENCES section_catalog(section_id) ON DELETE SET NULL,
  resolved_name TEXT,
  confidence REAL NOT NULL DEFAULT 0.6,
  method TEXT NOT NULL DEFAULT 'parsed',
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cad_tables (
  table_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  method TEXT NOT NULL DEFAULT 'grid',
  bbox_json TEXT NOT NULL DEFAULT '{}',
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

CREATE TABLE IF NOT EXISTS cad_story_tags (
  tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  cad_entity_id INTEGER REFERENCES cad_entities(cad_entity_id) ON DELETE SET NULL,
  story_norm TEXT NOT NULL,
  raw_text TEXT NOT NULL,
  x REAL, y REAL, z REAL,
  layer TEXT,
  confidence REAL NOT NULL DEFAULT 0.6,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_story_tags_artifact ON cad_story_tags(cad_artifact_id, story_norm);

CREATE TABLE IF NOT EXISTS cad_table_schemas (
  schema_id INTEGER PRIMARY KEY AUTOINCREMENT,
  table_id INTEGER NOT NULL REFERENCES cad_tables(table_id) ON DELETE CASCADE,
  header_row_idx INTEGER,
  columns_json TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_table_schemas_table ON cad_table_schemas(table_id);

CREATE TABLE IF NOT EXISTS cad_table_row_parses (
  row_parse_id INTEGER PRIMARY KEY AUTOINCREMENT,
  table_id INTEGER NOT NULL REFERENCES cad_tables(table_id) ON DELETE CASCADE,
  row_idx INTEGER NOT NULL,
  token_norm TEXT,
  story_norm TEXT,
  fields_json TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cad_table_row_parses_table ON cad_table_row_parses(table_id, row_idx);

CREATE TABLE IF NOT EXISTS token_story_maps (
  map_id INTEGER PRIMARY KEY AUTOINCREMENT,
  cad_artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  model_id INTEGER NOT NULL REFERENCES models(model_id) ON DELETE CASCADE,
  cad_token_norm TEXT NOT NULL,
  story_norm TEXT NOT NULL,
  model_member_id INTEGER NOT NULL REFERENCES model_members(model_member_id) ON DELETE CASCADE,
  confidence REAL NOT NULL DEFAULT 0.7,
  method TEXT NOT NULL DEFAULT 'inferred',
  status TEXT NOT NULL DEFAULT 'suggested',
  evidence_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(cad_artifact_id, model_id, cad_token_norm, story_norm)
);

CREATE INDEX IF NOT EXISTS idx_token_story_maps_lookup
ON token_story_maps(cad_artifact_id, model_id, status, cad_token_norm, story_norm);

-- Regression harness
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
  fixture_json TEXT NOT NULL,
  golden_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(suite_id, name)
);

CREATE TABLE IF NOT EXISTS regression_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  suite_id INTEGER NOT NULL REFERENCES regression_suites(suite_id) ON DELETE CASCADE,
  started_at TEXT NOT NULL DEFAULT (datetime('now')),
  finished_at TEXT,
  status TEXT NOT NULL DEFAULT 'RUNNING',
  report_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS regression_case_results (
  case_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES regression_runs(run_id) ON DELETE CASCADE,
  case_id INTEGER NOT NULL REFERENCES regression_cases(case_id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  diff_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS regression_reports (
  regression_report_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES regression_runs(run_id) ON DELETE CASCADE,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  format TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_regression_reports_run
ON regression_reports(run_id, format);

-- Benchmark and compare
CREATE TABLE IF NOT EXISTS benchmarks (
  benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'commercial',
  source TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS benchmark_results (
  benchmark_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  benchmark_id INTEGER NOT NULL REFERENCES benchmarks(benchmark_id) ON DELETE CASCADE,
  member_uid TEXT NOT NULL,
  story_norm TEXT,
  check_type TEXT NOT NULL,
  combo TEXT NOT NULL,
  demand_value REAL,
  capacity_value REAL,
  ratio REAL,
  status TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(benchmark_id, member_uid, story_norm, check_type, combo)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_results_lookup
ON benchmark_results(benchmark_id, member_uid, story_norm, check_type, combo);

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
  severity TEXT NOT NULL,
  note TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_compare_items_sev
ON compare_items(compare_id, severity);

CREATE TABLE IF NOT EXISTS compare_reports (
  compare_report_id INTEGER PRIMARY KEY AUTOINCREMENT,
  compare_id INTEGER NOT NULL REFERENCES compare_runs(compare_id) ON DELETE CASCADE,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  format TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS check_template_sets (
  template_set_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  templates_json TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(name, version)
);

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

CREATE TABLE IF NOT EXISTS decision_logs (
  decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity_type TEXT NOT NULL,
  entity_id INTEGER NOT NULL,
  from_status TEXT,
  to_status TEXT,
  reason TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- QA profiles
CREATE TABLE IF NOT EXISTS qa_profiles (
  qa_profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
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

-- Dataset registry
CREATE TABLE IF NOT EXISTS dataset_defs (
  dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
  type TEXT NOT NULL,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(type, name)
);

CREATE TABLE IF NOT EXISTS dataset_versions (
  dataset_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id INTEGER NOT NULL REFERENCES dataset_defs(dataset_id) ON DELETE CASCADE,
  version TEXT NOT NULL,
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
  change_type TEXT NOT NULL DEFAULT 'change',
  summary TEXT NOT NULL,
  details TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Signatures and approvals
CREATE TABLE IF NOT EXISTS report_signatures (
  signature_id INTEGER PRIMARY KEY AUTOINCREMENT,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id) ON DELETE CASCADE,
  method TEXT NOT NULL DEFAULT 'sha256',
  digest_sha256 TEXT NOT NULL,
  signature_b64 TEXT,
  key_id TEXT,
  signer TEXT,
  note TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_report_signatures_artifact
ON report_signatures(artifact_id);

CREATE TABLE IF NOT EXISTS approvals (
  approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity_type TEXT NOT NULL,
  entity_id INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'requested',
  actor TEXT,
  comment TEXT,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_approvals_entity
ON approvals(entity_type, entity_id, status);

-- Users and roles
CREATE TABLE IF NOT EXISTS users (
  user_id INTEGER PRIMARY KEY AUTOINCREMENT,
  actor TEXT NOT NULL UNIQUE,
  display_name TEXT,
  email TEXT,
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS roles (
  role_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  permissions_json TEXT NOT NULL DEFAULT '[]'
);

-- Projects
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
  status TEXT NOT NULL DEFAULT 'active',
  joined_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY(project_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_project_memberships_role
ON project_memberships(project_id, role_id, status);

-- Approval workflows
CREATE TABLE IF NOT EXISTS approval_workflow_defs (
  workflow_def_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  entity_types_json TEXT NOT NULL DEFAULT '[]',
  steps_json TEXT NOT NULL,
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
  status TEXT NOT NULL DEFAULT 'in_progress',
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
  decision TEXT NOT NULL,
  comment TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(instance_id, step_idx, actor)
);

CREATE INDEX IF NOT EXISTS idx_approval_instances_project
ON approval_instances(project_id, status, updated_at);

-- Context snapshots and change events
CREATE TABLE IF NOT EXISTS entity_contexts (
  entity_type TEXT NOT NULL,
  entity_id INTEGER NOT NULL,
  context_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY(entity_type, entity_id)
);

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
  event_type TEXT NOT NULL,
  message TEXT,
  payload_json TEXT NOT NULL DEFAULT '{}',
  actor TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_project_events_project
ON project_events(project_id, created_at);
