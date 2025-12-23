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
exports.ProjectsViewProvider = void 0;
const vscode = __importStar(require("vscode"));
const mcp_1 = require("../mcp");
class ProjectsViewProvider {
    context;
    constructor(context) {
        this.context = context;
    }
    resolveWebviewView(view) {
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
                        const projects = await (0, mcp_1.invokeStructAi)("structai_project_list", { limit: 50 });
                        view.webview.postMessage({ type: "projects", projects });
                        break;
                    }
                    case "loadProject": {
                        const dashboard = await (0, mcp_1.invokeStructAi)("structai_project_dashboard", {
                            project_id: msg.project_id,
                            limit_events: 30
                        });
                        view.webview.postMessage({ type: "dashboard", dashboard });
                        break;
                    }
                    case "createProject": {
                        const res = await (0, mcp_1.invokeStructAi)("structai_project_create", {
                            name: msg.name,
                            description: msg.description || ""
                        });
                        view.webview.postMessage({ type: "toast", ok: true, text: `project ${res.project_id} created` });
                        break;
                    }
                    case "bindModel": {
                        const res = await (0, mcp_1.invokeStructAi)("structai_project_bind_model", {
                            project_id: msg.project_id,
                            model_id: msg.model_id
                        });
                        view.webview.postMessage({ type: "toast", ok: true, text: `bound model ${res.model_id}` });
                        break;
                    }
                    case "addMember": {
                        const res = await (0, mcp_1.invokeStructAi)("structai_project_add_member", {
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
            }
            catch (e) {
                view.webview.postMessage({ type: "toast", ok: false, text: String(e?.message || e) });
            }
        });
    }
}
exports.ProjectsViewProvider = ProjectsViewProvider;
//# sourceMappingURL=projectsView.js.map