import * as vscode from "vscode";

import { invokeStructAi } from "../mcp";

export class MemoryViewProvider implements vscode.WebviewViewProvider {
  constructor(private readonly context: vscode.ExtensionContext) {}

  resolveWebviewView(view: vscode.WebviewView) {
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")]
    };

    const jsUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "memory", "main.js"));
    const cssUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "memory", "styles.css"));

    view.webview.html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src ${view.webview.cspSource}; script-src ${view.webview.cspSource};" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="${cssUri}" rel="stylesheet" />
  <title>StructAI Memory</title>
</head>
<body>
  <div id="app"></div>
  <script src="${jsUri}"></script>
</body>
</html>`;

    const sendArtifacts = async () => {
      const artifacts = await invokeStructAi("structai_list_artifacts", { limit: 200 });
      view.webview.postMessage({ type: "artifacts", artifacts });
    };

    view.webview.onDidReceiveMessage(async (msg) => {
      try {
        switch (msg.type) {
          case "init":
          case "refresh":
            await sendArtifacts();
            break;

          case "search": {
            const result = await invokeStructAi("structai_search", { query: msg.query, limit: 20 });
            view.webview.postMessage({ type: "searchResults", result });
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
