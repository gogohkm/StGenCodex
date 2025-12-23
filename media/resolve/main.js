(function () {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("app");

  let state = {
    cadId: "",
    modelId: "",
    analysisRunId: "",
    links: { items: [] },
    conflicts: { items: [] },
    quality: null,
    patches: { items: [] }
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
    const cad = document.getElementById("cadId");
    const model = document.getElementById("modelId");
    const run = document.getElementById("analysisRunId");
    const reason = document.getElementById("reason");
    if (cad) state.cadId = cad.value;
    if (model) state.modelId = model.value;
    if (run) state.analysisRunId = run.value;
    if (reason) state.reason = reason.value;
  }

  function render() {
    captureInputs();
    app.innerHTML = "";

    app.appendChild(h("h2", {}, ["StructAI Resolve"]));

    app.appendChild(
      h("div", { class: "row" }, [
        h("label", {}, ["CAD ID:"]),
        h("input", { id: "cadId", type: "number", value: state.cadId || "" }),
        h("label", { style: "margin-left:8px" }, ["Model ID:"]),
        h("input", { id: "modelId", type: "number", value: state.modelId || "" }),
        h("label", { style: "margin-left:8px" }, ["Analysis Run ID:"]),
        h("input", { id: "analysisRunId", type: "number", value: state.analysisRunId || "" })
      ])
    );

    app.appendChild(
      h("div", { class: "row" }, [
        h("label", {}, ["Reason:"]),
        h("input", { id: "reason", type: "text", value: state.reason || "" }),
        h("button", { onclick: refresh }, ["Refresh"]),
        h("button", { onclick: autoConfirm }, ["Auto-confirm table schema"]),
        h("button", { onclick: applySpecs }, ["Apply specs"])
      ])
    );

    app.appendChild(renderQuality());
    app.appendChild(renderConflicts());
    app.appendChild(renderLinks());
    app.appendChild(renderPatches());
  }

  function readIds() {
    const cad = Number(document.getElementById("cadId").value);
    const model = Number(document.getElementById("modelId").value);
    const runRaw = document.getElementById("analysisRunId").value;
    if (!cad || !model) {
      alert("CAD ID and Model ID are required.");
      return null;
    }
    return {
      cad_artifact_id: cad,
      model_id: model,
      analysis_run_id: runRaw ? Number(runRaw) : null
    };
  }

  function refresh() {
    const ids = readIds();
    if (!ids) return;
    vscode.postMessage({ type: "refresh", ...ids });
  }

  function autoConfirm() {
    const ids = readIds();
    if (!ids) return;
    vscode.postMessage({ type: "autoConfirmTableSchema", cad_artifact_id: ids.cad_artifact_id, model_id: ids.model_id });
  }

  function applySpecs() {
    const ids = readIds();
    if (!ids) return;
    vscode.postMessage({ type: "applySpecs", cad_artifact_id: ids.cad_artifact_id, model_id: ids.model_id });
  }

  function renderQuality() {
    const q = state.quality || {};
    const container = h("div", { class: "section" }, [h("h3", {}, ["Quality Summary"])]);
    if (!state.quality) {
      container.appendChild(h("div", { class: "muted" }, ["Run refresh to load summary."]));
      return container;
    }

    const summary = h("ul", {}, [
      h("li", {}, [`Steel missing: ${q.steel_missing_count || 0}`]),
      h("li", {}, [`RC missing: ${q.rc_missing_count || 0}`]),
      h("li", {}, [`Results missing: ${q.results_missing_count || 0}`])
    ]);

    container.appendChild(summary);

    if ((q.steel_missing_sample || []).length) {
      container.appendChild(h("div", { class: "muted" }, ["Steel sample (top 10)"]));
      container.appendChild(
        h("ul", {}, q.steel_missing_sample.slice(0, 10).map((s) => h("li", {}, [`${s.uid} ${s.label || ""} (${s.section || ""})`])) )
      );
    }
    if ((q.rc_missing_sample || []).length) {
      container.appendChild(h("div", { class: "muted" }, ["RC sample (top 10)"]));
      container.appendChild(
        h("ul", {}, q.rc_missing_sample.slice(0, 10).map((s) => h("li", {}, [`${s.uid} ${s.label || ""}`])) )
      );
    }
    if ((q.results_missing_sample || []).length) {
      container.appendChild(h("div", { class: "muted" }, ["Results missing (top 10)"]));
      container.appendChild(
        h("ul", {}, q.results_missing_sample.slice(0, 10).map((s) => h("li", {}, [`${s.uid} ${s.label || ""}`])) )
      );
    }

    return container;
  }

  function renderConflicts() {
    const items = (state.conflicts && state.conflicts.items) || [];
    const container = h("div", { class: "section" }, [h("h3", {}, ["Token-Story Conflicts"])]);
    if (!items.length) {
      container.appendChild(h("div", { class: "muted" }, ["No conflicts found."]));
      return container;
    }

    for (const c of items) {
      const candList = h(
        "ul",
        {},
        (c.candidates || []).map((m) =>
          h("li", {}, [`${m.member_label || ""} (${m.member_uid}) [${Number(m.confidence).toFixed(2)}]`])
        )
      );
      container.appendChild(h("div", { class: "card" }, [
        h("div", { class: "muted" }, [`${c.token} @ ${c.story}`]),
        candList
      ]));
    }

    return container;
  }

  function renderLinks() {
    const items = (state.links && state.links.items) || [];
    const container = h("div", { class: "section" }, [h("h3", {}, ["Suggested Spec Links"])]);
    if (!items.length) {
      container.appendChild(h("div", { class: "muted" }, ["No suggested links."]));
      return container;
    }

    const table = h("table", { class: "table" }, []);
    table.appendChild(
      h("tr", {}, [
        h("th", {}, ["Link ID"]),
        h("th", {}, ["Token"]),
        h("th", {}, ["Member"]),
        h("th", {}, ["Spec Kind"]),
        h("th", {}, ["Method"]),
        h("th", {}, ["Status"]),
        h("th", {}, ["Action"])
      ])
    );

    for (const it of items) {
      const actions = h("div", { class: "row" }, [
        h("button", { onclick: () => updateLink(it.link_id, "confirmed") }, ["Confirm"]),
        h("button", { onclick: () => updateLink(it.link_id, "rejected") }, ["Reject"])
      ]);

      table.appendChild(
        h("tr", {}, [
          h("td", {}, [String(it.link_id)]),
          h("td", {}, [it.cad_token_norm || ""]),
          h("td", {}, [`${it.member_label || ""} (${it.member_uid || ""})`]),
          h("td", {}, [it.spec_kind || ""]),
          h("td", {}, [it.method || ""]),
          h("td", {}, [it.status || ""]),
          h("td", {}, [actions])
        ])
      );
    }

    container.appendChild(table);
    return container;
  }

  function renderPatches() {
    const items = (state.patches && state.patches.items) || [];
    const container = h("div", { class: "section" }, [h("h3", {}, ["Patch Runs"])]);
    if (!items.length) {
      container.appendChild(h("div", { class: "muted" }, ["No patch runs."]));
      return container;
    }

    const table = h("table", { class: "table" }, []);
    table.appendChild(
      h("tr", {}, [
        h("th", {}, ["Patch ID"]),
        h("th", {}, ["CAD"]),
        h("th", {}, ["Note"]),
        h("th", {}, ["Created"]),
        h("th", {}, ["Action"])
      ])
    );

    for (const it of items) {
      table.appendChild(
        h("tr", {}, [
          h("td", {}, [String(it.patch_run_id)]),
          h("td", {}, [String(it.cad_artifact_id || "")]),
          h("td", {}, [it.note || ""]),
          h("td", {}, [it.created_at || ""]),
          h("td", {}, [
            h("button", { onclick: () => rollbackPatch(it.patch_run_id) }, ["Rollback"])
          ])
        ])
      );
    }

    container.appendChild(table);
    return container;
  }

  function updateLink(linkId, toStatus) {
    const reason = document.getElementById("reason").value || "";
    vscode.postMessage({ type: "setLinkStatus", link_id: linkId, to_status: toStatus, reason });
  }

  function rollbackPatch(patchRunId) {
    vscode.postMessage({ type: "rollbackPatch", patch_run_id: patchRunId, mode: "keys_only" });
  }

  window.addEventListener("message", (event) => {
    const msg = event.data;
    if (msg.type === "data") {
      state.links = msg.links || { items: [] };
      state.conflicts = msg.conflicts || { items: [] };
      state.quality = msg.quality || null;
      state.patches = msg.patches || { items: [] };
      render();
    } else if (msg.type === "toast") {
      if (!msg.ok) {
        alert(msg.text);
      }
    }
  });

  render();
})();
