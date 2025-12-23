(function () {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("app");

  let state = {
    projects: [],
    dashboard: null,
    projectId: "",
    newName: "",
    newDesc: "",
    bindModelId: "",
    memberActor: "",
    memberRole: ""
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
      "projectId",
      "newName",
      "newDesc",
      "bindModelId",
      "memberActor",
      "memberRole"
    ];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) state[id] = el.value;
    }
  }

  function render() {
    captureInputs();
    app.innerHTML = "";
    app.appendChild(h("h2", {}, ["StructAI Projects"]));

    app.appendChild(
      h("div", { class: "row" }, [
        h("button", { onclick: listProjects }, ["Refresh List"])
      ])
    );

    app.appendChild(renderProjectList());

    app.appendChild(
      h("div", { class: "row" }, [
        h("label", {}, ["Project ID:"]),
        h("input", { id: "projectId", type: "number", value: state.projectId }),
        h("button", { onclick: loadProject }, ["Load Dashboard"])
      ])
    );

    app.appendChild(renderDashboard());

    app.appendChild(h("h3", {}, ["Create Project"]));
    app.appendChild(
      h("div", { class: "row" }, [
        h("label", {}, ["Name:"]),
        h("input", { id: "newName", value: state.newName }),
        h("label", { style: "margin-left:8px" }, ["Description:"]),
        h("input", { id: "newDesc", value: state.newDesc }),
        h("button", { onclick: createProject }, ["Create"])
      ])
    );

    app.appendChild(h("h3", {}, ["Bind Model"]));
    app.appendChild(
      h("div", { class: "row" }, [
        h("label", {}, ["Model ID:"]),
        h("input", { id: "bindModelId", type: "number", value: state.bindModelId }),
        h("button", { onclick: bindModel }, ["Bind"])
      ])
    );

    app.appendChild(h("h3", {}, ["Add Member"]));
    app.appendChild(
      h("div", { class: "row" }, [
        h("label", {}, ["Actor:"]),
        h("input", { id: "memberActor", value: state.memberActor }),
        h("label", { style: "margin-left:8px" }, ["Role:"]),
        h("input", { id: "memberRole", value: state.memberRole }),
        h("button", { onclick: addMember }, ["Add"])
      ])
    );
  }

  function renderProjectList() {
    if (!state.projects.length) {
      return h("div", { class: "muted" }, ["No projects loaded."]);
    }
    const list = h("ul", {}, []);
    for (const p of state.projects) {
      list.appendChild(
        h("li", {}, [
          h("button", { onclick: () => selectProject(p.project_id) }, [`Open #${p.project_id}`]),
          document.createTextNode(` ${p.name || ""}`)
        ])
      );
    }
    return list;
  }

  function renderDashboard() {
    if (!state.dashboard) {
      return h("div", { class: "muted" }, ["No dashboard loaded."]);
    }
    const dash = state.dashboard;
    const container = h("div", { class: "section" }, [h("h3", {}, ["Dashboard"]) ]);

    const models = (dash.models || []).map((m) => String(m.model_id)).join(", ");
    container.appendChild(h("div", { class: "muted" }, [`Models: ${models || "-"}`]));

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

  function listProjects() {
    vscode.postMessage({ type: "listProjects" });
  }

  function selectProject(projectId) {
    state.projectId = String(projectId);
    loadProject();
  }

  function loadProject() {
    const projectId = Number(state.projectId || "0");
    if (!projectId) {
      alert("Project ID required.");
      return;
    }
    vscode.postMessage({ type: "loadProject", project_id: projectId });
  }

  function createProject() {
    if (!state.newName) {
      alert("Project name required.");
      return;
    }
    vscode.postMessage({ type: "createProject", name: state.newName, description: state.newDesc });
  }

  function bindModel() {
    const projectId = Number(state.projectId || "0");
    const modelId = Number(state.bindModelId || "0");
    if (!projectId || !modelId) {
      alert("Project ID and model ID are required.");
      return;
    }
    vscode.postMessage({ type: "bindModel", project_id: projectId, model_id: modelId });
  }

  function addMember() {
    const projectId = Number(state.projectId || "0");
    if (!projectId || !state.memberActor || !state.memberRole) {
      alert("Project ID, actor, and role are required.");
      return;
    }
    vscode.postMessage({
      type: "addMember",
      project_id: projectId,
      actor: state.memberActor,
      role_name: state.memberRole
    });
  }

  window.addEventListener("message", (ev) => {
    const msg = ev.data;
    if (msg.type === "projects") {
      state.projects = msg.projects.items || [];
      render();
    }
    if (msg.type === "dashboard") {
      state.dashboard = msg.dashboard;
      render();
    }
    if (msg.type === "toast") {
      if (!msg.ok) alert(msg.text);
    }
  });

  render();
  vscode.postMessage({ type: "listProjects" });
})();
