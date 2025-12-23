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
exports.GovernanceViewProvider = void 0;
const vscode = __importStar(require("vscode"));
const mcp_1 = require("../mcp");
class GovernanceViewProvider {
    context;
    constructor(context) {
        this.context = context;
    }
    resolveWebviewView(view) {
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
                        const dashboard = await (0, mcp_1.invokeStructAi)("structai_project_dashboard", {
                            project_id: msg.project_id,
                            limit_events: 30
                        });
                        view.webview.postMessage({ type: "dashboard", dashboard });
                        break;
                    }
                    case "requestApproval": {
                        const res = await (0, mcp_1.invokeStructAi)("structai_approval_request_v2", {
                            project_id: msg.project_id,
                            entity_type: msg.entity_type,
                            entity_id: msg.entity_id,
                            comment: msg.comment || ""
                        });
                        view.webview.postMessage({ type: "toast", ok: true, text: `request instance=${res.instance_id}` });
                        break;
                    }
                    case "voteApproval": {
                        const res = await (0, mcp_1.invokeStructAi)("structai_approval_vote", {
                            instance_id: msg.instance_id,
                            decision: msg.decision,
                            comment: msg.comment || ""
                        });
                        view.webview.postMessage({ type: "toast", ok: true, text: `voted instance=${res.instance_id}` });
                        break;
                    }
                    case "readApproval": {
                        const res = await (0, mcp_1.invokeStructAi)("structai_approval_read", {
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
            }
            catch (e) {
                view.webview.postMessage({ type: "toast", ok: false, text: String(e?.message || e) });
            }
        });
    }
}
exports.GovernanceViewProvider = GovernanceViewProvider;
//# sourceMappingURL=governanceView.js.map