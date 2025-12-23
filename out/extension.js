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
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const participant_1 = require("./chat/participant");
const provider_1 = require("./mcp/provider");
const mappingView_1 = require("./views/mappingView");
const resultsView_1 = require("./views/resultsView");
const resolveView_1 = require("./views/resolveView");
const qaView_1 = require("./views/qaView");
const projectsView_1 = require("./views/projectsView");
const governanceView_1 = require("./views/governanceView");
const memoryView_1 = require("./views/memoryView");
function activate(context) {
    console.log("StructAI activate: start");
    context.subscriptions.push(vscode.window.registerWebviewViewProvider("structai-mapping", new mappingView_1.MappingViewProvider(context)));
    console.log("StructAI activate: mapping view registered");
    context.subscriptions.push(vscode.window.registerWebviewViewProvider("structai-results", new resultsView_1.ResultsViewProvider(context), {
        webviewOptions: { retainContextWhenHidden: true }
    }));
    console.log("StructAI activate: results view registered");
    context.subscriptions.push(vscode.window.registerWebviewViewProvider("structai-resolve", new resolveView_1.ResolveViewProvider(context), {
        webviewOptions: { retainContextWhenHidden: true }
    }));
    console.log("StructAI activate: resolve view registered");
    context.subscriptions.push(vscode.window.registerWebviewViewProvider("structai-qa", new qaView_1.QaViewProvider(context), {
        webviewOptions: { retainContextWhenHidden: true }
    }));
    console.log("StructAI activate: qa view registered");
    context.subscriptions.push(vscode.window.registerWebviewViewProvider("structai-projects", new projectsView_1.ProjectsViewProvider(context), {
        webviewOptions: { retainContextWhenHidden: true }
    }));
    console.log("StructAI activate: projects view registered");
    context.subscriptions.push(vscode.window.registerWebviewViewProvider("structai-governance", new governanceView_1.GovernanceViewProvider(context), {
        webviewOptions: { retainContextWhenHidden: true }
    }));
    console.log("StructAI activate: governance view registered");
    context.subscriptions.push(vscode.window.registerWebviewViewProvider("structai-memory", new memoryView_1.MemoryViewProvider(context)));
    console.log("StructAI activate: memory view registered");
    try {
        (0, participant_1.registerStructChatParticipant)(context);
    }
    catch (err) {
        console.error("StructAI chat participant registration failed", err);
    }
    try {
        (0, provider_1.registerStructMcpProvider)(context);
    }
    catch (err) {
        console.error("StructAI MCP provider registration failed", err);
    }
    const focusView = async (viewId) => {
        await vscode.commands.executeCommand("workbench.view.extension.structai-panel");
        await vscode.commands.executeCommand(`${viewId}.focus`);
    };
    context.subscriptions.push(vscode.commands.registerCommand("structai.openPanel", async () => {
        await vscode.commands.executeCommand("workbench.view.extension.structai-panel");
    }));
    context.subscriptions.push(vscode.commands.registerCommand("structai.openMapping", async () => {
        await focusView("structai-mapping");
    }));
    context.subscriptions.push(vscode.commands.registerCommand("structai.openResults", async () => {
        await focusView("structai-results");
    }));
    context.subscriptions.push(vscode.commands.registerCommand("structai.openResolve", async () => {
        await focusView("structai-resolve");
    }));
    context.subscriptions.push(vscode.commands.registerCommand("structai.openQa", async () => {
        await focusView("structai-qa");
    }));
    context.subscriptions.push(vscode.commands.registerCommand("structai.openProjects", async () => {
        await focusView("structai-projects");
    }));
    context.subscriptions.push(vscode.commands.registerCommand("structai.openGovernance", async () => {
        await focusView("structai-governance");
    }));
    context.subscriptions.push(vscode.commands.registerCommand("structai.openMemory", async () => {
        await focusView("structai-memory");
    }));
}
function deactivate() { }
//# sourceMappingURL=extension.js.map