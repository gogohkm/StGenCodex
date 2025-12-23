import * as vscode from "vscode";

import { invokeStructAi } from "../mcp";

export class ProjectsViewProvider implements vscode.WebviewViewProvider {
  constructor(private readonly context: vscode.ExtensionContext) {}

  resolveWebviewView(view: vscode.WebviewView) {
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")]
    };

    const jsUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "projects", "main.js"));
    const cssUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "projects", "styles.css"));

    view.webview.html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src ${view.webview.cspSource}; script-src ${view.webview.cspSource};" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="${cssUri}" rel="stylesheet" />
  <title>StructAI Projects</title>
</head>
<body>
  <div id="app"></div>
  <script src="${jsUri}"></script>
</body>
</html>`;

    view.webview.onDidReceiveMessage(async (msg) => {
      try {
        switch (msg.type) {
          case "listProjects": {
            const projects = await invokeStructAi("structai_project_list", { limit: 50 });
            view.webview.postMessage({ type: "projects", projects });
            break;
          }

          case "loadProject": {
            const dashboard = await invokeStructAi("structai_project_dashboard", {
              project_id: msg.project_id,
              limit_events: 30
            });
            view.webview.postMessage({ type: "dashboard", dashboard });
            break;
          }

          case "createProject": {
            const res = await invokeStructAi("structai_project_create", {
              name: msg.name,
              description: msg.description || ""
            });
            view.webview.postMessage({ type: "toast", ok: true, text: `project ${res.project_id} created` });
            break;
          }

          case "bindModel": {
            const res = await invokeStructAi("structai_project_bind_model", {
              project_id: msg.project_id,
              model_id: msg.model_id
            });
            view.webview.postMessage({ type: "toast", ok: true, text: `bound model ${res.model_id}` });
            break;
          }

          case "addMember": {
            const res = await invokeStructAi("structai_project_add_member", {
              project_id: msg.project_id,
              actor: msg.actor,
              role_name: msg.role_name
            });
            view.webview.postMessage({ type: "toast", ok: true, text: `member added: ${res.actor}` });
            break;
          }

          default:
            break;
        }
      } catch (e: any) {
        view.webview.postMessage({ type: "toast", ok: false, text: String(e?.message || e) });
      }
    });
  }
}
