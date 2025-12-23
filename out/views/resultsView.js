"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.ResultsViewProvider = void 0;
const vscode = __importStar(require("vscode"));
const mcp_1 = require("../mcp");
class ResultsViewProvider {
    ctx;
    constructor(ctx) {
        this.ctx = ctx;
    }
    resolveWebviewView(view) {
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
            const models = await (0, mcp_1.invokeStructAi)("structai_model_list", {});
            view.webview.postMessage({ type: "models", models });
        };
        view.webview.onDidReceiveMessage(async (msg) => {
            try {
                switch (msg.type) {
                    case "init":
                        await sendState();
                        break;
                    case "loadRuns": {
                        const runs = await (0, mcp_1.invokeStructAi)("structai_results_list_runs", { model_id: msg.model_id });
                        const checks = await (0, mcp_1.invokeStructAi)("structai_check_list_runs", { model_id: msg.model_id });
                        view.webview.postMessage({ type: "runs", runs, checks });
                        break;
                    }
                    case "importResults": {
                        const pick = await vscode.window.showOpenDialog({
                            canSelectMany: false,
                            filters: { Results: ["csv", "json"] }
                        });
                        if (!pick?.length)
                            return;
                        const result = await (0, mcp_1.invokeStructAi)("structai_results_import", {
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
                        if (!pick?.length)
                            return;
                        const result = await (0, mcp_1.invokeStructAi)("structai_design_import_inputs", {
                            model_id: msg.model_id,
                            path: pick[0].fsPath
                        });
                        view.webview.postMessage({ type: "importDesignDone", result });
                        break;
                    }
                    case "runChecks": {
                        const result = await (0, mcp_1.invokeStructAi)("structai_check_run", msg.input);
                        view.webview.postMessage({ type: "runChecksDone", result });
                        break;
                    }
                    case "loadCheckResults": {
                        const result = await (0, mcp_1.invokeStructAi)("structai_check_get_results", {
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
            }
            catch (e) {
                view.webview.postMessage({ type: "error", message: String(e?.message ?? e) });
            }
        });
    }
}
exports.ResultsViewProvider = ResultsViewProvider;
//# sourceMappingURL=resultsView.js.map