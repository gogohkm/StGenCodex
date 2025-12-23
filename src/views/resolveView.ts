import * as vscode from "vscode";

import { invokeStructAi } from "../mcp";

export class ResolveViewProvider implements vscode.WebviewViewProvider {
  constructor(private readonly context: vscode.ExtensionContext) {}

  resolveWebviewView(view: vscode.WebviewView) {
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")]
    };

    const jsUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "resolve", "main.js"));
    const cssUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "resolve", "styles.css"));

    view.webview.html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src ${view.webview.cspSource}; script-src ${view.webview.cspSource};" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="${cssUri}" rel="stylesheet" />
  <title>StructAI Resolve</title>
</head>
<body>
  <div id="app"></div>
  <script src="${jsUri}"></script>
</body>
</html>`;

    view.webview.onDidReceiveMessage(async (msg) => {
      try {
        switch (msg.type) {
          case "refresh": {
            const { cad_artifact_id, model_id, analysis_run_id } = msg;
            const [links, conflicts, quality, patches] = await Promise.all([
              invokeStructAi("structai_specs_list_links", {
                cad_artifact_id,
                model_id,
                status: "suggested",
                limit: 300
              }),
              invokeStructAi("structai_token_story_conflicts", { cad_artifact_id, model_id }),
              invokeStructAi("structai_quality_summary", { model_id, analysis_run_id }),
              invokeStructAi("structai_design_list_patch_runs", { model_id, limit: 30 })
            ]);
            view.webview.postMessage({ type: "data", links, conflicts, quality, patches });
            break;
          }

          case "setLinkStatus": {
            const res = await invokeStructAi("structai_specs_set_link_status", {
              link_id: msg.link_id,
              to_status: msg.to_status,
              reason: msg.reason || ""
            });
            view.webview.postMessage({ type: "toast", ok: true, text: `link ${res.link_id} -> ${res.status}` });
            break;
          }

          case "autoConfirmTableSchema": {
            const res = await invokeStructAi("structai_specs_auto_confirm_table_schema", {
              cad_artifact_id: msg.cad_artifact_id,
              model_id: msg.model_id
            });
            view.webview.postMessage({ type: "toast", ok: true, text: `confirmed=${res.confirmed}, kept=${res.kept_suggested}` });
            break;
          }

          case "applySpecs": {
            const res = await invokeStructAi("structai_design_apply_specs_to_inputs", {
              cad_artifact_id: msg.cad_artifact_id,
              model_id: msg.model_id,
              overwrite_keys: false
            });
            view.webview.postMessage({ type: "toast", ok: true, text: `applied. patch_run_id=${res.patch_run_id}` });
            break;
          }

          case "rollbackPatch": {
            const res = await invokeStructAi("structai_design_rollback_patch", {
              patch_run_id: msg.patch_run_id,
              mode: msg.mode || "keys_only"
            });
            view.webview.postMessage({ type: "toast", ok: true, text: `rollback done. restored=${res.restored_members}` });
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
