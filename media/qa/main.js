(function () {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("app");

  let state = {
    suite: "core",
    tol: "0.001",
    modelId: "",
    entityType: "report",
    entityId: "",
    actor: "",
    comment: "",
    approvalId: "",
    approvalStatus: "approved",
    artifactId: "",
    signer: "",
    signMethod: "sha256",
    signNote: "",
    signatureId: "",
    runs: [],
    compares: [],
    runDetail: null,
    compareDetail: null,
    datasets: [],
    qaProfile: null,
    approvals: [],
    reports: []
  };

  function h(tag, attrs = {}, children = []) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") el.className = v;
      else if (k === "onclick") el.onclick = v;
      else if (k === "onchange") el.onchange = v;
      else if (v !== null && v !== undefined) el.setAttribute(k, v);
    }
    for (const c of children) {
      if (typeof c === "string") el.appendChild(document.createTextNode(c));
      else if (c) el.appendChild(c);
    }
    return el;
  }

  function captureInputs() {
    const ids = [
      "suite",
      "tol",
      "modelId",
      "entityType",
      "entityId",
      "actor",
      "comment",
      "approvalId",
      "approvalStatus",
      "artifactId",
      "signer",
      "signMethod",
      "signNote",
      "signatureId"
    ];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) state[id] = el.value;
    }
  }

  function render() {
    captureInputs();
    app.innerHTML = "";

    app.appendChild(h("h2", {}, ["StructAI QA"]));

    app.appendChild(
      h("div", { class: "row" }, [
        h("label", {}, ["Suite:"]),
        h("input", { id: "suite", value: state.suite }),
        h("label", { style: "margin-left:8px" }, ["RatioTol:"]),
        h("input", { id: "tol", type: "number", step: "0.0001", value: state.tol }),
        h("button", { onclick: refresh }, ["Refresh"]),
        h("button", { onclick: runSuite }, ["Run Suite"])
      ])
    );

    app.appendChild(h("h3", {}, ["Regression Runs"]));
    const runsList = h("ul", {}, []);
    for (const r of state.runs) {
      runsList.appendChild(
        h("li", {}, [
          h("button", { onclick: () => readRun(r.run_id) }, [`Open #${r.run_id}`]),
          document.createTextNode(` ${r.status} ${r.started_at || ""}`)
        ])
      );
    }
    app.appendChild(runsList);

    app.appendChild(h("h3", {}, ["Compare Runs"]));
    const compList = h("ul", {}, []);
    for (const c of state.compares) {
      compList.appendChild(
        h("li", {}, [
          h("button", { onclick: () => readCompare(c.compare_id) }, [`Open #${c.compare_id}`]),
          document.createTextNode(` (check:${c.check_run_id}, bench:${c.benchmark_id})`)
        ])
      );
    }
    app.appendChild(compList);

    app.appendChild(h("h3", {}, ["Detail"]));
    const detail = h("pre", { class: "detail" }, [
      state.runDetail ? JSON.stringify(state.runDetail, null, 2) : state.compareDetail ? JSON.stringify(state.compareDetail, null, 2) : ""
    ]);
    app.appendChild(detail);

    app.appendChild(h("h3", {}, ["Governance"]));
    app.appendChild(
      h("div", { class: "row" }, [
        h("label", {}, ["Model ID:"]),
        h("input", { id: "modelId", type: "number", value: state.modelId }),
        h("label", { style: "margin-left:8px" }, ["Entity Type:"]),
        h("select", { id: "entityType" }, [
          h("option", { value: "report", selected: state.entityType === "report" ? "selected" : null }, ["report"]),
          h("option", { value: "check_run", selected: state.entityType === "check_run" ? "selected" : null }, ["check_run"]),
          h("option", { value: "regression_run", selected: state.entityType === "regression_run" ? "selected" : null }, ["regression_run"]),
          h("option", { value: "compare_run", selected: state.entityType === "compare_run" ? "selected" : null }, ["compare_run"])
        ]),
        h("label", { style: "margin-left:8px" }, ["Entity ID:"]),
        h("input", { id: "entityId", type: "number", value: state.entityId }),
        h("button", { onclick: loadGovernance }, ["Load Governance"])
      ])
    );

    app.appendChild(h("div", { class: "section" }, [
      h("h4", {}, ["Active Datasets"]),
      renderDatasets()
    ]));

    app.appendChild(h("div", { class: "section" }, [
      h("h4", {}, ["QA Profile"]),
      h("pre", { class: "detail" }, [state.qaProfile ? JSON.stringify(state.qaProfile, null, 2) : ""])
    ]));

    app.appendChild(h("div", { class: "section" }, [
      h("h4", {}, ["Approvals"]),
      renderApprovals()
    ]));

    app.appendChild(h("div", { class: "section" }, [
      h("h4", {}, ["Approval Actions"]),
      h("div", { class: "row" }, [
        h("label", {}, ["Actor:"]),
        h("input", { id: "actor", value: state.actor }),
        h("label", { style: "margin-left:8px" }, ["Comment:"]),
        h("input", { id: "comment", value: state.comment })
      ]),
      h("div", { class: "row" }, [
        h("button", { onclick: requestApproval }, ["Request Approval"]),
        h("button", { onclick: listApprovals }, ["Refresh Approvals"])
      ]),
      h("div", { class: "row" }, [
        h("label", {}, ["Approval ID:"]),
        h("input", { id: "approvalId", type: "number", value: state.approvalId }),
        h("label", { style: "margin-left:8px" }, ["Status:"]),
        h("select", { id: "approvalStatus" }, [
          h("option", { value: "approved", selected: state.approvalStatus === "approved" ? "selected" : null }, ["approved"]),
          h("option", { value: "rejected", selected: state.approvalStatus === "rejected" ? "selected" : null }, ["rejected"])
        ]),
        h("button", { onclick: setApprovalStatus }, ["Set Status"])
      ])
    ]));

    app.appendChild(h("div", { class: "section" }, [
      h("h4", {}, ["Report Signatures"]),
      h("div", { class: "row" }, [
        h("label", {}, ["Artifact ID:"]),
        h("input", { id: "artifactId", type: "number", value: state.artifactId }),
        h("label", { style: "margin-left:8px" }, ["Signer:"]),
        h("input", { id: "signer", value: state.signer })
      ]),
      h("div", { class: "row" }, [
        h("label", {}, ["Method:"]),
        h("input", { id: "signMethod", value: state.signMethod }),
        h("label", { style: "margin-left:8px" }, ["Note:"]),
        h("input", { id: "signNote", value: state.signNote }),
        h("button", { onclick: signReport }, ["Sign"])
      ]),
      h("div", { class: "row" }, [
        h("label", {}, ["Signature ID:"]),
        h("input", { id: "signatureId", type: "number", value: state.signatureId }),
        h("button", { onclick: verifyReport }, ["Verify"])
      ]),
      h("div", { class: "muted" }, ["Recent reports (use artifact IDs from memory or report list)."]),
      renderReports()
    ]));
  }

  function renderDatasets() {
    if (!state.datasets || !state.datasets.length) {
      return h("div", { class: "muted" }, ["No active datasets."]);
    }
    const table = h("table", { class: "table" }, []);
    table.appendChild(
      h("tr", {}, [
        h("th", {}, ["Type"]),
        h("th", {}, ["Name"]),
        h("th", {}, ["Version"]),
        h("th", {}, ["Artifact"])
      ])
    );
    for (const d of state.datasets) {
      table.appendChild(
        h("tr", {}, [
          h("td", {}, [d.type || ""]),
          h("td", {}, [d.name || ""]),
          h("td", {}, [d.version || ""]),
          h("td", {}, [String(d.artifact_id || "")])
        ])
      );
    }
    return table;
  }

  function renderApprovals() {
    const items = state.approvals || [];
    if (!items.length) {
      return h("div", { class: "muted" }, ["No approvals loaded."]);
    }
    const table = h("table", { class: "table" }, []);
    table.appendChild(
      h("tr", {}, [
        h("th", {}, ["Approval ID"]),
        h("th", {}, ["Status"]),
        h("th", {}, ["Actor"]),
        h("th", {}, ["Comment"]),
        h("th", {}, ["Updated"])
      ])
    );
    for (const a of items) {
      table.appendChild(
        h("tr", {}, [
          h("td", {}, [String(a.approval_id)]),
          h("td", {}, [a.status || ""]),
          h("td", {}, [a.actor || ""]),
          h("td", {}, [a.comment || ""]),
          h("td", {}, [a.updated_at || ""])
        ])
      );
    }
    return table;
  }

  function renderReports() {
    const items = state.reports || [];
    if (!items.length) {
      return h("div", { class: "muted" }, ["No reports found."]);
    }
    const list = h("ul", {}, []);
    for (const r of items) {
      list.appendChild(h("li", {}, [`#${r.report_id} ${r.format} ${r.source_path || r.uri || ""}`]));
    }
    return list;
  }

  function refresh() {
    vscode.postMessage({ type: "refresh", suite_name: state.suite });
  }

  function runSuite() {
    const tol = parseFloat(state.tol || "0.001");
    vscode.postMessage({ type: "runSuite", suite_name: state.suite, ratio_tol: tol });
  }

  function readRun(runId) {
    vscode.postMessage({ type: "readRun", run_id: runId });
  }

  function readCompare(compareId) {
    vscode.postMessage({ type: "readCompare", compare_id: compareId });
  }

  function loadGovernance() {
    const modelId = state.modelId ? Number(state.modelId) : null;
    const entityId = state.entityId ? Number(state.entityId) : null;
    vscode.postMessage({
      type: "loadGovernance",
      model_id: modelId,
      entity_type: state.entityType,
      entity_id: entityId
    });
  }

  function requestApproval() {
    const entityId = Number(state.entityId || "0");
    if (!state.entityType || !entityId || !state.actor) {
      alert("Entity type, entity id, and actor are required.");
      return;
    }
    vscode.postMessage({
      type: "approvalRequest",
      entity_type: state.entityType,
      entity_id: entityId,
      actor: state.actor,
      comment: state.comment
    });
  }

  function listApprovals() {
    const entityId = Number(state.entityId || "0");
    if (!state.entityType || !entityId) {
      alert("Entity type and entity id are required.");
      return;
    }
    vscode.postMessage({ type: "listApprovals", entity_type: state.entityType, entity_id: entityId });
  }

  function setApprovalStatus() {
    const approvalId = Number(state.approvalId || "0");
    if (!approvalId || !state.actor) {
      alert("Approval ID and actor are required.");
      return;
    }
    vscode.postMessage({
      type: "approvalSetStatus",
      approval_id: approvalId,
      status: state.approvalStatus,
      actor: state.actor,
      comment: state.comment
    });
  }

  function signReport() {
    const artifactId = Number(state.artifactId || "0");
    if (!artifactId || !state.signer) {
      alert("Artifact ID and signer are required.");
      return;
    }
    vscode.postMessage({
      type: "reportSign",
      artifact_id: artifactId,
      signer: state.signer,
      method: state.signMethod,
      note: state.signNote
    });
  }

  function verifyReport() {
    const signatureId = Number(state.signatureId || "0");
    if (!signatureId) {
      alert("Signature ID is required.");
      return;
    }
    vscode.postMessage({ type: "reportVerify", signature_id: signatureId });
  }

  window.addEventListener("message", (ev) => {
    const msg = ev.data;
    if (msg.type === "data") {
      state.runs = msg.runs.items || [];
      state.compares = msg.compares.items || [];
      render();
    }
    if (msg.type === "runDetail") {
      state.runDetail = msg.run;
      state.compareDetail = null;
      render();
    }
    if (msg.type === "compareDetail") {
      state.compareDetail = msg.cmp;
      state.runDetail = null;
      render();
    }
    if (msg.type === "governance") {
      state.datasets = msg.datasets.items || [];
      state.qaProfile = msg.qaProfile.profile || null;
      state.approvals = msg.approvals.items || [];
      state.reports = msg.reports.items || [];
      render();
    }
    if (msg.type === "approvals") {
      state.approvals = msg.approvals.items || [];
      render();
    }
    if (msg.type === "toast") {
      if (!msg.ok) {
        alert(msg.text);
      }
    }
  });

  vscode.postMessage({ type: "refresh", suite_name: state.suite });
})();
