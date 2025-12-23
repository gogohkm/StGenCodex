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
exports.registerStructChatParticipant = registerStructChatParticipant;
const vscode = __importStar(require("vscode"));
const chatUtils = __importStar(require("@vscode/chat-extension-utils"));
const PARTICIPANT_ID = "structai.struct";
function registerStructChatParticipant(context) {
    const handler = async (request, chatContext, stream, token) => {
        if (request.command === "map") {
            await vscode.commands.executeCommand("structai.openMapping");
            stream.markdown([
                "Opened the mapping view.",
                "",
                "- Select DXF + model",
                "- Auto-map to generate suggestions",
                "- Confirm to persist mappings"
            ].join("\n"));
            return { metadata: { command: "map" } };
        }
        if (request.command === "check") {
            await vscode.commands.executeCommand("structai.openResults");
            stream.markdown("Opened the results view. You can import analysis results and run checks.");
            return { metadata: { command: "check" } };
        }
        if (request.command === "report") {
            await vscode.commands.executeCommand("structai.openResults");
            stream.markdown("Opened the results view. You can generate and sign reports from there.");
            return { metadata: { command: "report" } };
        }
        if (request.command === "import") {
            stream.markdown([
                "You can import files via MCP tools:",
                "- `#structai_import_pdf { \"path\": \"...\" }`",
                "- `#structai_import_md { \"path\": \"...\" }`",
                "- `#structai_import_dxf { \"path\": \"...\" }`",
                "- `#structai_model_import_members { \"path\": \"...\" }`"
            ].join("\n"));
            return { metadata: { command: "import" } };
        }
        if (request.command === "reset") {
            stream.markdown([
                "To reset project memory, run the MCP tool:",
                "",
                "`#structai_reset_all`"
            ].join("\n"));
            return { metadata: { command: "reset" } };
        }
        const prompt = buildStructSystemPrompt();
        const tools = vscode.lm.tools.filter((t) => {
            const name = t.name;
            if (!name)
                return false;
            return name.includes("structai_") || name.includes("/structai_");
        });
        const libResult = chatUtils.sendChatParticipantRequest(request, chatContext, {
            prompt,
            tools,
            responseStreamOptions: {
                stream,
                references: true,
                responseText: true
            },
            extensionMode: context.extensionMode,
            requestJustification: "StructAI structural engineering context"
        }, token);
        return await libResult.result;
    };
    const participant = vscode.chat.createChatParticipant(PARTICIPANT_ID, handler);
    participant.iconPath = vscode.Uri.joinPath(context.extensionUri, "media", "icons", "structai.svg");
    participant.followupProvider = {
        provideFollowups() {
            return [
                { label: "Import project files", prompt: "@struct /import How do I import PDFs and DXF files?" },
                { label: "Open mapping view", prompt: "@struct /map" },
                { label: "Run checks", prompt: "@struct /check" }
            ];
        }
    };
    context.subscriptions.push(participant);
}
function buildStructSystemPrompt() {
    return [
        "You are StructAI, a structural engineering assistant.",
        "",
        "## Operating rules",
        "1) Evidence first: cite project documents/drawings/results when making judgments.",
        "2) If key inputs are missing (loads, units, materials, combos), list them instead of guessing.",
        "3) Treat imported CAD/PDF/MD/model/results as a single project memory.",
        "4) Use MCP tools actively, especially search and check tools.",
        "",
        "## Evidence",
        "Use structai_search to find relevant clauses and cite page references when possible.",
        "If evidence is missing, say so explicitly.",
        "",
        "Answer in Korean by default unless the user asks otherwise."
    ].join("\n");
}
//# sourceMappingURL=participant.js.map