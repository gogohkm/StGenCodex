import * as vscode from "vscode";

import { invokeStructAi } from "../mcp";

export class QaViewProvider implements vscode.WebviewViewProvider {
  constructor(private readonly context: vscode.ExtensionContext) {}

  resolveWebviewView(view: vscode.WebviewView) {
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.context.extensionUri, "media")]
    };

    const jsUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "qa", "main.js"));
    const cssUri = view.webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "qa", "styles.css"));

    view.webview.html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src ${view.webview.cspSource}; script-src ${view.webview.cspSource};" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="${cssUri}" rel="stylesheet" />
  <title>StructAI QA</title>
</head>
<body>
  <div id="app"></div>
  <script src="${jsUri}"></script>
</body>
</html>`;

    view.webview.onDidReceiveMessage(async (msg) => {
      try {
        if (msg.type === "refresh") {
          const [runs, compares] = await Promise.all([
            invokeStructAi("structai_regression_list_runs", { suite_name: msg.suite_name, limit: 20 }),
            invokeStructAi("structai_compare_list_runs", { limit: 20 })
          ]);
          view.webview.postMessage({ type: "data", runs, compares });
          return;
        }

        if (msg.type === "runSuite") {
          const res = await invokeStructAi("structai_regression_run_suite_v2", {
            suite_name: msg.suite_name,
            isolated_db: true,
            ratio_tol: msg.ratio_tol || 0.001
          });
          await invokeStructAi("structai_regression_report_generate", { run_id: res.run_id, formats: ["md"] });
          view.webview.postMessage({ type: "toast", ok: true, text: `run=${res.run_id} status=${res.status}` });
          return;
        }

        if (msg.type === "readRun") {
          const run = await invokeStructAi("structai_regression_read_run", { run_id: msg.run_id });
          view.webview.postMessage({ type: "runDetail", run });
          return;
        }

        if (msg.type === "readCompare") {
          const cmp = await invokeStructAi("structai_compare_read_run", { compare_id: msg.compare_id });
          view.webview.postMessage({ type: "compareDetail", cmp });
          return;
        }

        if (msg.type === "loadGovernance") {
          const datasets = await invokeStructAi("structai_dataset_get_active_all", {});
          const qaProfile = msg.model_id
            ? await invokeStructAi("structai_qa_profile_get_effective", { model_id: msg.model_id })
            : { ok: true, profile: null };
          const reports = await invokeStructAi("structai_report_list", { limit: 50 });
          const approvals = msg.entity_type && msg.entity_id
            ? await invokeStructAi("structai_approval_list", { entity_type: msg.entity_type, entity_id: msg.entity_id })
            : { ok: true, items: [] };
          view.webview.postMessage({
            type: "governance",
            datasets,
            qaProfile,
            approvals,
            reports
          });
          return;
        }

        if (msg.type === "approvalRequest") {
          const res = await invokeStructAi("structai_approval_request", {
            entity_type: msg.entity_type,
            entity_id: msg.entity_id,
            actor: msg.actor,
            comment: msg.comment || ""
          });
          view.webview.postMessage({ type: "toast", ok: true, text: `approval requested: ${res.approval_id}` });
          return;
        }

        if (msg.type === "approvalSetStatus") {
          const res = await invokeStructAi("structai_approval_set_status", {
            approval_id: msg.approval_id,
            status: msg.status,
            actor: msg.actor,
            comment: msg.comment || ""
          });
          view.webview.postMessage({ type: "toast", ok: true, text: `approval ${res.approval_id} -> ${res.status}` });
          return;
        }

        if (msg.type === "listApprovals") {
          const approvals = await invokeStructAi("structai_approval_list", {
            entity_type: msg.entity_type,
            entity_id: msg.entity_id
          });
          view.webview.postMessage({ type: "approvals", approvals });
          return;
        }

        if (msg.type === "reportSign") {
          const res = await invokeStructAi("structai_report_sign", {
            artifact_id: msg.artifact_id,
            signer: msg.signer,
            method: msg.method || "sha256",
            note: msg.note || ""
          });
          view.webview.postMessage({ type: "toast", ok: true, text: `signed: signature_id=${res.signature_id}` });
          return;
        }

        if (msg.type === "reportVerify") {
          const res = await invokeStructAi("structai_report_verify", {
            signature_id: msg.signature_id
          });
          view.webview.postMessage({ type: "toast", ok: res.valid, text: `verify: ${res.valid ? "ok" : "invalid"}` });
          return;
        }
      } catch (e: any) {
        view.webview.postMessage({ type: "toast", ok: false, text: String(e?.message || e) });
      }
    });
  }
}
