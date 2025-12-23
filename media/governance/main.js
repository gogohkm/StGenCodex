(function () {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("app");

  let state = {
    projectId: "",
    entityType: "report",
    entityId: "",
    comment: "",
    instanceId: "",
    decision: "approve",
    dashboard: null,
    approvalDetail: null
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
    const ids = ["projectId", "entityType", "entityId", "comment", "instanceId", "decision"];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) state[id] = el.value;
    }
  }

  function render() {
    captureInputs();
    app.innerHTML = "";
    app.appendChild(h("h2", {}, ["StructAI Governance"]));

    app.appendChild(
      h("div", { class: "row" }, [
        h("label", {}, ["Project ID:"]),
        h("input", { id: "projectId", type: "number", value: state.projectId }),
        h("button", { onclick: loadDashboard }, ["Load Dashboard"])
      ])
    );

    app.appendChild(renderDashboard());

    app.appendChild(h("h3", {}, ["Approval Request"]));
    app.appendChild(
      h("div", { class: "row" }, [
        h("label", {}, ["Entity Type:"]),
        h("select", { id: "entityType" }, [
          h("option", { value: "report", selected: state.entityType === "report" ? "selected" : null }, ["report"]),
          h("option", { value: "check_run", selected: state.entityType === "check_run" ? "selected" : null }, ["check_run"]),
          h("option", { value: "regression_run", selected: state.entityType === "regression_run" ? "selected" : null }, ["regression_run"]),
          h("option", { value: "compare_run", selected: state.entityType === "compare_run" ? "selected" : null }, ["compare_run"])
        ]),
        h("label", { style: "margin-left:8px" }, ["Entity ID:"]),
        h("input", { id: "entityId", type: "number", value: state.entityId }),
        h("label", { style: "margin-left:8px" }, ["Comment:"]),
        h("input", { id: "comment", value: state.comment }),
        h("button", { onclick: requestApproval }, ["Request"])
      ])
    );

    app.appendChild(h("h3", {}, ["Vote"]));
    app.appendChild(
      h("div", { class: "row" }, [
        h("label", {}, ["Instance ID:"]),
        h("input", { id: "instanceId", type: "number", value: state.instanceId }),
        h("label", { style: "margin-left:8px" }, ["Decision:"]),
        h("select", { id: "decision" }, [
          h("option", { value: "approve", selected: state.decision === "approve" ? "selected" : null }, ["approve"]),
          h("option", { value: "reject", selected: state.decision === "reject" ? "selected" : null }, ["reject"])
        ]),
        h("button", { onclick: voteApproval }, ["Vote"])
      ])
    );

    app.appendChild(h("h3", {}, ["Read Instance"]));
    app.appendChild(
      h("div", { class: "row" }, [
        h("button", { onclick: readApproval }, ["Read"])
      ])
    );

    app.appendChild(h("pre", { class: "detail" }, [state.approvalDetail ? JSON.stringify(state.approvalDetail, null, 2) : ""]));
  }

  function renderDashboard() {
    if (!state.dashboard) {
      return h("div", { class: "muted" }, ["No dashboard loaded."]);
    }
    const dash = state.dashboard;
    const container = h("div", { class: "section" }, [h("h3", {}, ["Dashboard"]) ]);

    container.appendChild(h("h4", {}, ["Approvals"]));
    const approvals = dash.approvals || [];
    if (!approvals.length) {
      container.appendChild(h("div", { class: "muted" }, ["No approvals."]));
    } else {
      const table = h("table", { class: "table" }, []);
      table.appendChild(
        h("tr", {}, [
          h("th", {}, ["Instance"]),
          h("th", {}, ["Status"]),
          h("th", {}, ["Entity"]),
          h("th", {}, ["Updated"])
        ])
      );
      for (const a of approvals) {
        table.appendChild(
          h("tr", {}, [
            h("td", {}, [String(a.instance_id)]),
            h("td", {}, [a.status || ""]),
            h("td", {}, [`${a.entity_type || ""} #${a.entity_id || ""}`]),
            h("td", {}, [a.updated_at || ""])
          ])
        );
      }
      container.appendChild(table);
    }

    container.appendChild(h("h4", {}, ["Events"]));
    const events = dash.events || [];
    if (!events.length) {
      container.appendChild(h("div", { class: "muted" }, ["No events."]));
    } else {
      const list = h("ul", {}, []);
      for (const e of events) {
        list.appendChild(h("li", {}, [`${e.event_type}: ${e.message || ""} (${e.actor || ""})` ]));
      }
      container.appendChild(list);
    }

    return container;
  }

  function loadDashboard() {
    const projectId = Number(state.projectId || "0");
    if (!projectId) {
      alert("Project ID required.");
      return;
    }
    vscode.postMessage({ type: "loadDashboard", project_id: projectId });
  }

  function requestApproval() {
    const projectId = Number(state.projectId || "0");
    const entityId = Number(state.entityId || "0");
    if (!projectId || !entityId) {
      alert("Project ID and entity ID are required.");
      return;
    }
    vscode.postMessage({
      type: "requestApproval",
      project_id: projectId,
      entity_type: state.entityType,
      entity_id: entityId,
      comment: state.comment
    });
  }

  function voteApproval() {
    const instanceId = Number(state.instanceId || "0");
    if (!instanceId) {
      alert("Instance ID required.");
      return;
    }
    vscode.postMessage({ type: "voteApproval", instance_id: instanceId, decision: state.decision, comment: state.comment });
  }

  function readApproval() {
    const projectId = Number(state.projectId || "0");
    const entityId = Number(state.entityId || "0");
    if (!projectId || !entityId) {
      alert("Project ID and entity ID are required.");
      return;
    }
    vscode.postMessage({
      type: "readApproval",
      project_id: projectId,
      entity_type: state.entityType,
      entity_id: entityId
    });
  }

  window.addEventListener("message", (ev) => {
    const msg = ev.data;
    if (msg.type === "dashboard") {
      state.dashboard = msg.dashboard;
      render();
    }
    if (msg.type === "approvalDetail") {
      state.approvalDetail = msg.detail;
      render();
    }
    if (msg.type === "toast") {
      if (!msg.ok) alert(msg.text);
    }
  });

  render();
})();
