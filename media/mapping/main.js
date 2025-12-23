(function () {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("app");

  let state = {
    artifacts: { items: [] },
    models: { items: [] },
    mappings: { items: [] }
  };

  let suggestions = null;

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

  function render() {
    app.innerHTML = "";

    const dxfArtifacts = (state.artifacts.items || []).filter((a) => a.kind === "dxf");
    const models = state.models.items || [];

    const cadSelect = h(
      "select",
      { id: "cadSelect" },
      dxfArtifacts.map((a) => h("option", { value: String(a.artifact_id) }, [`${a.title || a.uri}`]))
    );

    const modelSelect = h(
      "select",
      { id: "modelSelect" },
      models.map((m) => h("option", { value: String(m.model_id) }, [`${m.name}`]))
    );

    const thresholdInput = h("input", {
      id: "threshold",
      type: "number",
      step: "0.05",
      min: "0",
      max: "1",
      value: "0.85"
    });

    const btnRow = h("div", { class: "row" }, [
      h("button", { onclick: () => vscode.postMessage({ type: "refresh" }) }, ["Refresh"]),
      h("button", { onclick: () => vscode.postMessage({ type: "importModel" }) }, ["Import Model (CSV/JSON)"])
    ]);

    const actionRow = h("div", { class: "row" }, [
      h("label", {}, ["CAD(DXF): "]),
      cadSelect,
      h("label", { style: "margin-left:10px" }, ["Model: "]),
      modelSelect
    ]);

    const suggestRow = h("div", { class: "row" }, [
      h("label", {}, ["Auto-confirm threshold: "]),
      thresholdInput,
      h(
        "button",
        {
          style: "margin-left:10px",
          onclick: () => {
            const cadId = Number(document.getElementById("cadSelect").value);
            const modelId = Number(document.getElementById("modelSelect").value);
            if (!cadId || !modelId) {
              alert("Select CAD and model first.");
              return;
            }
            vscode.postMessage({
              type: "suggestMappings",
              input: {
                cad_artifact_id: cadId,
                model_id: modelId,
                max_tokens: 200,
                max_candidates_per_token: 5,
                spatial_tolerance: 5.0,
                enable_fuzzy: true
              }
            });
          }
        },
        ["Auto-map"]
      ),
      h(
        "button",
        {
          style: "margin-left:10px",
          onclick: () => saveSelected()
        },
        ["Confirm selected"]
      )
    ]);

    app.appendChild(h("h2", {}, ["StructAI Mapping"]));
    app.appendChild(btnRow);
    app.appendChild(actionRow);
    app.appendChild(suggestRow);

    app.appendChild(renderMappings());
    app.appendChild(renderSuggestions());
  }

  function renderMappings() {
    const items = state.mappings.items || [];
    const container = h("div", { class: "section" }, [h("h3", {}, [`Saved mappings (${items.length})`])]);

    if (items.length === 0) {
      container.appendChild(h("div", { class: "muted" }, ["No mappings yet."]));
      return container;
    }

    const table = h("table", { class: "table" }, []);
    table.appendChild(
      h("tr", {}, [
        h("th", {}, ["Token"]),
        h("th", {}, ["Member"]),
        h("th", {}, ["Type"]),
        h("th", {}, ["Confidence"]),
        h("th", {}, ["Status"]),
        h("th", {}, ["Method"])
      ])
    );

    for (const m of items.slice(0, 200)) {
      table.appendChild(
        h("tr", {}, [
          h("td", {}, [m.cad_token]),
          h("td", {}, [`${m.member_label || ""} (${m.member_uid})`]),
          h("td", {}, [m.member_type || ""]),
          h("td", {}, [String(Number(m.confidence).toFixed(2))]),
          h("td", {}, [m.status]),
          h("td", {}, [m.method])
        ])
      );
    }

    container.appendChild(table);
    return container;
  }

  function renderSuggestions() {
    const container = h("div", { class: "section" }, [h("h3", {}, ["Suggestions"])]);

    if (!suggestions) {
      container.appendChild(h("div", { class: "muted" }, ["Run Auto-map to see suggestions."]));
      return container;
    }

    const items = suggestions.items || [];
    const unmatched = suggestions.unmatched_tokens || [];

    container.appendChild(
      h("div", { class: "muted" }, [`Suggested tokens: ${items.length}, Unmatched: ${unmatched.length}`])
    );

    const table = h("table", { class: "table" }, []);
    table.appendChild(
      h("tr", {}, [
        h("th", {}, ["Confirm"]),
        h("th", {}, ["Token"]),
        h("th", {}, ["Type guess"]),
        h("th", {}, ["Occ."])
      , h("th", {}, ["Pick member"]),
        h("th", {}, ["Confidence"]),
        h("th", {}, ["Method"])
      ])
    );

    const threshold = Number(document.getElementById("threshold")?.value || "0.85");

    for (const it of items) {
      const token = it.token;
      const cand = it.candidates || [];
      const top = cand[0];

      const select = h(
        "select",
        { class: "candSelect", "data-token": token },
        cand.map((c) =>
          h("option", { value: String(c.model_member_id) }, [
            `${c.member_label || ""} (${c.member_uid}) [${Number(c.confidence).toFixed(2)}]`
          ])
        )
      );

      const chk = h("input", {
        type: "checkbox",
        class: "confirmChk",
        "data-token": token,
        checked: top && Number(top.confidence) >= threshold ? "checked" : null
      });

      table.appendChild(
        h("tr", {}, [
          h("td", {}, [chk]),
          h("td", {}, [token]),
          h("td", {}, [it.type_guess || ""]),
          h("td", {}, [String(it.occurrence_count || 0)]),
          h("td", {}, [select]),
          h("td", {}, [top ? String(Number(top.confidence).toFixed(2)) : ""]),
          h("td", {}, [top ? top.method : ""])
        ])
      );
    }

    container.appendChild(table);

    if (unmatched.length > 0) {
      container.appendChild(h("h4", {}, ["Unmatched tokens (top 20)"]));
      const ul = h(
        "ul",
        {},
        unmatched.slice(0, 20).map((u) => h("li", {}, [`${u.token} (occ:${u.occurrence_count})`]))
      );
      container.appendChild(ul);
    }

    return container;
  }

  function saveSelected() {
    if (!suggestions) return;

    const cadId = Number(document.getElementById("cadSelect").value);
    const modelId = Number(document.getElementById("modelSelect").value);

    if (!cadId || !modelId) {
      alert("Select CAD and model first.");
      return;
    }

    const rows = Array.from(document.querySelectorAll(".confirmChk"));
    const selects = new Map(
      Array.from(document.querySelectorAll(".candSelect")).map((s) => [s.getAttribute("data-token"), s])
    );

    const toSave = [];

    for (const chk of rows) {
      if (!chk.checked) continue;
      const token = chk.getAttribute("data-token");
      const sel = selects.get(token);
      const memberId = Number(sel.value);

      const item = (suggestions.items || []).find((i) => i.token === token);
      const cand = (item?.candidates || []).find((c) => Number(c.model_member_id) === memberId);
      toSave.push({
        cad_artifact_id: cadId,
        cad_token: token,
        model_id: modelId,
        model_member_id: memberId,
        confidence: cand?.confidence ?? 0.5,
        method: cand?.method ?? "manual",
        status: "confirmed",
        evidence: cand?.evidence ?? {}
      });
    }

    vscode.postMessage({ type: "saveMappings", mappings: toSave });
  }

  window.addEventListener("message", (event) => {
    const msg = event.data;
    if (msg.type === "state") {
      state = {
        artifacts: msg.artifacts || { items: [] },
        models: msg.models || { items: [] },
        mappings: msg.mappings || { items: [] }
      };
      render();
    } else if (msg.type === "suggestions") {
      suggestions = msg.result;
      render();
    } else if (msg.type === "saved") {
      vscode.postMessage({ type: "refresh" });
    } else if (msg.type === "error") {
      console.error(msg.message);
      alert(msg.message);
    }
  });

  vscode.postMessage({ type: "init" });
})();
