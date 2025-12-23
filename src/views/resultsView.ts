import * as vscode from "vscode";

import { invokeStructAi } from "../mcp";

export class ResultsViewProvider implements vscode.WebviewViewProvider {
  constructor(private readonly ctx: vscode.ExtensionContext) {}

  resolveWebviewView(view: vscode.WebviewView) {
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.ctx.extensionUri, "media")]
    };

    const jsUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.ctx.extensionUri, "media", "results", "main.js"));
    const cssUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.ctx.extensionUri, "media", "results", "styles.css"));

    view.webview.html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src ${view.webview.cspSource}; script-src ${view.webview.cspSource};" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="${cssUri}" rel="stylesheet" />
  <title>StructAI Results</title>
</head>
<body>
  <div id="app"></div>
  <script src="${jsUri}"></script>
</body>
</html>`;

    const sendState = async () => {
      const models = await invokeStructAi("structai_model_list", {});
      view.webview.postMessage({ type: "models", models });
    };

    view.webview.onDidReceiveMessage(async (msg) => {
      try {
        switch (msg.type) {
          case "init":
            await sendState();
            break;

          case "loadRuns": {
            const runs = await invokeStructAi("structai_results_list_runs", { model_id: msg.model_id });
            const checks = await invokeStructAi("structai_check_list_runs", { model_id: msg.model_id });
            view.webview.postMessage({ type: "runs", runs, checks });
            break;
          }

          case "importResults": {
            const pick = await vscode.window.showOpenDialog({
              canSelectMany: false,
              filters: { Results: ["csv", "json"] }
            });
            if (!pick?.length) return;
            const result = await invokeStructAi("structai_results_import", {
              model_id: msg.model_id,
              path: pick[0].fsPath
            });
            view.webview.postMessage({ type: "importResultsDone", result });
            break;
          }

          case "importDesignInputs": {
            const pick = await vscode.window.showOpenDialog({
              canSelectMany: false,
              filters: { "Design Inputs": ["csv", "json"] }
            });
            if (!pick?.length) return;
            const result = await invokeStructAi("structai_design_import_inputs", {
              model_id: msg.model_id,
              path: pick[0].fsPath
            });
            view.webview.postMessage({ type: "importDesignDone", result });
            break;
          }

          case "runChecks": {
            const result = await invokeStructAi("structai_check_run", msg.input);
            view.webview.postMessage({ type: "runChecksDone", result });
            break;
          }

          case "loadCheckResults": {
            const result = await invokeStructAi("structai_check_get_results", {
              check_run_id: msg.check_run_id,
              status: msg.status || null,
              limit: 500
            });
            view.webview.postMessage({ type: "checkResults", result });
            break;
          }

          default:
            break;
        }
      } catch (e: any) {
        view.webview.postMessage({ type: "error", message: String(e?.message ?? e) });
      }
    });
  }
}
