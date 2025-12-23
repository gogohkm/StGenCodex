(function () {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("app");

  let models = [];
  let runs = [];
  let checks = [];
  let checkResults = null;

  function h(tag, attrs = {}, children = []) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") el.className = v;
      else if (k === "onclick") el.onclick = v;
      else el.setAttribute(k, v);
    }
    for (const c of children) el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    return el;
  }

  function render() {
    app.innerHTML = "";
    app.appendChild(h("h2", {}, ["StructAI Results / Checks"]));

    const modelSelect = h(
      "select",
      { id: "modelSel" },
      models.map((m) => h("option", { value: String(m.model_id) }, [`${m.name} (#${m.model_id})`]))
    );

    const btnRow = h("div", { class: "row" }, [
      h("button", { onclick: () => vscode.postMessage({ type: "init" }) }, ["Refresh Models"]),
      h("button", { onclick: () => loadRuns() }, ["Load Runs/Checks"]),
      h("button", { onclick: () => importResults() }, ["Import Results"]),
      h("button", { onclick: () => importDesign() }, ["Import Design Inputs"]),
      h("button", { onclick: () => runChecks() }, ["Run Checks"])
    ]);

    app.appendChild(h("div", { class: "row" }, [h("label", {}, ["Model: "]), modelSelect]));
    app.appendChild(btnRow);

    app.appendChild(h("h3", {}, ["Analysis Runs"]));
    app.appendChild(h("div", { class: "muted" }, [runs.length ? `Runs: ${runs.length}` : "No runs loaded."]));
    app.appendChild(
      h("ul", {}, runs.map((r) => h("li", {}, [`#${r.analysis_run_id} ${r.name} (${r.engine || ""} ${r.units || ""})`])) )
    );

    app.appendChild(h("h3", {}, ["Check Runs"]));
    const ul = h("ul", {}, []);
    for (const c of checks) {
      ul.appendChild(
        h("li", {}, [
          h("button", { onclick: () => loadCheck(c.check_run_id) }, [`Open #${c.check_run_id}`]),
          document.createTextNode(` ${c.name} (rule: ${c.rulepack_name || "builtin"}/${c.rulepack_version || ""})`)
        ])
      );
    }
    app.appendChild(ul);

    app.appendChild(h("h3", {}, ["Check Results (top 200)"]));
    if (!checkResults) {
      app.appendChild(h("div", { class: "muted" }, ["No check results loaded."]));
    } else {
      const items = checkResults.items || [];
      const table = h("table", { class: "table" }, []);
      table.appendChild(
        h("tr", {}, [
          h("th", {}, ["Status"]),
          h("th", {}, ["UID"]),
          h("th", {}, ["Label"]),
          h("th", {}, ["Combo"]),
          h("th", {}, ["Type"]),
          h("th", {}, ["Ratio"]),
          h("th", {}, ["Citations"])
        ])
      );
      for (const it of items.slice(0, 200)) {
        const cites = (it.citations || [])
          .map((c) => (c.page ? `${c.title || ""} p.${c.page}` : c.title || ""))
          .join("; ");
        table.appendChild(
          h("tr", {}, [
            h("td", {}, [it.status]),
            h("td", {}, [it.member_uid]),
            h("td", {}, [it.member_label || ""]),
            h("td", {}, [it.combo]),
            h("td", {}, [it.check_type]),
            h("td", {}, [it.ratio != null ? Number(it.ratio).toFixed(3) : ""]),
            h("td", {}, [cites])
          ])
        );
      }
      app.appendChild(table);
    }
  }

  function getModelId() {
    const el = document.getElementById("modelSel");
    return el ? Number(el.value) : null;
  }

  function loadRuns() {
    const model_id = getModelId();
    if (!model_id) return;
    vscode.postMessage({ type: "loadRuns", model_id });
  }

  function importResults() {
    const model_id = getModelId();
    if (!model_id) return;
    vscode.postMessage({ type: "importResults", model_id });
  }

  function importDesign() {
    const model_id = getModelId();
    if (!model_id) return;
    vscode.postMessage({ type: "importDesignInputs", model_id });
  }

  function runChecks() {
    const model_id = getModelId();
    if (!model_id) return;

    const latest = runs[0];
    if (!latest) {
      alert("Import analysis results first.");
      return;
    }

    vscode.postMessage({
      type: "runChecks",
      input: {
        model_id,
        analysis_run_id: latest.analysis_run_id,
        name: `check_${Date.now()}`
      }
    });
  }

  function loadCheck(check_run_id) {
    vscode.postMessage({ type: "loadCheckResults", check_run_id, status: null });
  }

  window.addEventListener("message", (ev) => {
    const msg = ev.data;
    if (msg.type === "models") {
      models = msg.models.items || [];
      render();
    } else if (msg.type === "runs") {
      runs = msg.runs.items || [];
      checks = msg.checks.items || [];
      render();
    } else if (msg.type === "checkResults") {
      checkResults = msg.result;
      render();
    } else if (msg.type === "importResultsDone" || msg.type === "importDesignDone" || msg.type === "runChecksDone") {
      loadRuns();
    } else if (msg.type === "error") {
      console.error(msg.message);
      alert(msg.message);
    }
  });

  vscode.postMessage({ type: "init" });
})();
