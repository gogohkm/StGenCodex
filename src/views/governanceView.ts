import * as vscode from "vscode";

import { invokeStructAi } from "../mcp";

export class GovernanceViewProvider implements vscode.WebviewViewProvider {
  constructor(private readonly context: vscode.ExtensionContext) {}

  resolveWebviewView(view: vscode.WebviewView) {
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")]
    };

    const jsUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "governance", "main.js"));
    const cssUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "governance", "styles.css"));

    view.webview.html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src ${view.webview.cspSource}; script-src ${view.webview.cspSource};" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="${cssUri}" rel="stylesheet" />
  <title>StructAI Governance</title>
</head>
<body>
  <div id="app"></div>
  <script src="${jsUri}"></script>
</body>
</html>`;

    view.webview.onDidReceiveMessage(async (msg) => {
      try {
        switch (msg.type) {
          case "loadDashboard": {
            const dashboard = await invokeStructAi("structai_project_dashboard", {
              project_id: msg.project_id,
              limit_events: 30
            });
            view.webview.postMessage({ type: "dashboard", dashboard });
            break;
          }

          case "requestApproval": {
            const res = await invokeStructAi("structai_approval_request_v2", {
              project_id: msg.project_id,
              entity_type: msg.entity_type,
              entity_id: msg.entity_id,
              comment: msg.comment || ""
            });
            view.webview.postMessage({ type: "toast", ok: true, text: `request instance=${res.instance_id}` });
            break;
          }

          case "voteApproval": {
            const res = await invokeStructAi("structai_approval_vote", {
              instance_id: msg.instance_id,
              decision: msg.decision,
              comment: msg.comment || ""
            });
            view.webview.postMessage({ type: "toast", ok: true, text: `voted instance=${res.instance_id}` });
            break;
          }

          case "readApproval": {
            const res = await invokeStructAi("structai_approval_read", {
              entity_type: msg.entity_type,
              entity_id: msg.entity_id,
              project_id: msg.project_id
            });
            view.webview.postMessage({ type: "approvalDetail", detail: res });
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
