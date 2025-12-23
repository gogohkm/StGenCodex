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
exports.MappingViewProvider = void 0;
const vscode = __importStar(require("vscode"));
const mcp_1 = require("../mcp");
class MappingViewProvider {
    context;
    view;
    constructor(context) {
        this.context = context;
    }
    resolveWebviewView(view) {
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
                        if (!picked || picked.length === 0)
                            return;
                        const filePath = picked[0].fsPath;
                        const result = await (0, mcp_1.invokeStructAi)("structai_model_import_members", { path: filePath });
                        webview.postMessage({ type: "modelImported", result });
                        await this.sendState();
                        break;
                    }
                    case "suggestMappings": {
                        const result = await (0, mcp_1.invokeStructAi)("structai_map_suggest_members", msg.input);
                        webview.postMessage({ type: "suggestions", result });
                        break;
                    }
                    case "saveMappings": {
                        const result = await (0, mcp_1.invokeStructAi)("structai_map_save_mappings", { mappings: msg.mappings });
                        webview.postMessage({ type: "saved", result });
                        await this.sendState();
                        break;
                    }
                    default:
                        break;
                }
            }
            catch (e) {
                webview.postMessage({ type: "error", message: String(e?.message ?? e) });
            }
        });
    }
    async sendState() {
        if (!this.view)
            return;
        const webview = this.view.webview;
        const artifacts = await (0, mcp_1.invokeStructAi)("structai_list_artifacts", {});
        const models = await (0, mcp_1.invokeStructAi)("structai_model_list", {});
        const mappings = await (0, mcp_1.invokeStructAi)("structai_map_list_mappings", { limit: 200 });
        webview.postMessage({
            type: "state",
            artifacts,
            models,
            mappings
        });
    }
    getHtml(webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "mapping", "main.js"));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, "media", "mapping", "styles.css"));
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
exports.MappingViewProvider = MappingViewProvider;
//# sourceMappingURL=mappingView.js.map