(function () {
  const vscode = acquireVsCodeApi();
  const app = document.getElementById("app");

  let artifacts = [];
  let searchResults = null;

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
    app.appendChild(h("h2", {}, ["StructAI Memory"]));

    app.appendChild(
      h("div", { class: "row" }, [
        h("button", { onclick: () => vscode.postMessage({ type: "refresh" }) }, ["Refresh"]),
        h("input", { id: "query", placeholder: "Search query" }),
        h("button", { onclick: () => search() }, ["Search"])
      ])
    );

    app.appendChild(h("h3", {}, ["Artifacts"]));
    if (!artifacts.length) {
      app.appendChild(h("div", { class: "muted" }, ["No artifacts."]));
    } else {
      const list = h("ul", {}, []);
      for (const a of artifacts) {
        list.appendChild(h("li", {}, [`#${a.artifact_id} ${a.kind} ${a.title || a.uri}`]));
      }
      app.appendChild(list);
    }

    app.appendChild(h("h3", {}, ["Search Results"]));
    if (!searchResults) {
      app.appendChild(h("div", { class: "muted" }, ["No search results."]));
    } else {
      const list = h("ul", {}, []);
      for (const r of searchResults.results || []) {
        list.appendChild(h("li", {}, [`${r.title || r.uri} p.${r.page_start || ""} ${r.snippet || ""}`]));
      }
      app.appendChild(list);
    }
  }

  function search() {
    const q = document.getElementById("query").value || "";
    if (!q.trim()) {
      alert("Enter a query.");
      return;
    }
    vscode.postMessage({ type: "search", query: q });
  }

  window.addEventListener("message", (ev) => {
    const msg = ev.data;
    if (msg.type === "artifacts") {
      artifacts = msg.artifacts.items || [];
      render();
    }
    if (msg.type === "searchResults") {
      searchResults = msg.result;
      render();
    }
    if (msg.type === "toast") {
      if (!msg.ok) alert(msg.text);
    }
  });

  vscode.postMessage({ type: "init" });
})();
