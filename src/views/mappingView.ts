import * as vscode from "vscode";

import { invokeStructAi } from "../mcp";

export class MappingViewProvider implements vscode.WebviewViewProvider {
  private view?: vscode.WebviewView;

  constructor(private readonly context: vscode.ExtensionContext) {}

  resolveWebviewView(view: vscode.WebviewView) {
    this.view = view;
    const webview = view.webview;

    webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")]
    };

    webview.html = this.getHtml(webview);

    webview.onDidReceiveMessage(async (msg) => {
      try {
        switch (msg.type) {
          case "init":
          case "refresh":
            await this.sendState();
            break;

          case "importModel": {
            const picked = await vscode.window.showOpenDialog({
              canSelectMany: false,
              openLabel: "Import model (CSV/JSON)",
              filters: { "Model Export": ["csv", "json"] }
            });
            if (!picked || picked.length === 0) return;
            const filePath = picked[0].fsPath;
            const result = await invokeStructAi("structai_model_import_members", { path: filePath });
            webview.postMessage({ type: "modelImported", result });
            await this.sendState();
            break;
          }

          case "suggestMappings": {
            const result = await invokeStructAi("structai_map_suggest_members", msg.input);
            webview.postMessage({ type: "suggestions", result });
            break;
          }

          case "saveMappings": {
            const result = await invokeStructAi("structai_map_save_mappings", { mappings: msg.mappings });
            webview.postMessage({ type: "saved", result });
            await this.sendState();
            break;
          }

          default:
            break;
        }
      } catch (e: any) {
        webview.postMessage({ type: "error", message: String(e?.message ?? e) });
      }
    });
  }

  private async sendState() {
    if (!this.view) return;
    const webview = this.view.webview;

    const artifacts = await invokeStructAi("structai_list_artifacts", {});
    const models = await invokeStructAi("structai_model_list", {});
    const mappings = await invokeStructAi("structai_map_list_mappings", { limit: 200 });

    webview.postMessage({
      type: "state",
      artifacts,
      models,
      mappings
    });
  }

  private getHtml(webview: vscode.Webview): string {
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.context.extensionUri, "media", "mapping", "main.js")
    );

    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.context.extensionUri, "media", "mapping", "styles.css")
    );

    return /* html */ `
      <!doctype html>
      <html>
        <head>
          <meta charset="UTF-8" />
          <meta http-equiv="Content-Security-Policy"
                content="default-src 'none'; style-src ${webview.cspSource}; script-src ${webview.cspSource};" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <link href="${styleUri}" rel="stylesheet" />
          <title>StructAI Mapping</title>
        </head>
        <body>
          <div id="app"></div>
          <script src="${scriptUri}"></script>
        </body>
      </html>
    `;
  }
}
