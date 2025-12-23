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
exports.invokeStructAi = invokeStructAi;
exports.unwrapToolResult = unwrapToolResult;
const vscode = __importStar(require("vscode"));
async function invokeStructAi(toolSuffix, input) {
    const toolInfo = vscode.lm.tools.find((t) => {
        const name = t.name;
        return Boolean(name && name.includes(toolSuffix));
    });
    if (!toolInfo) {
        throw new Error(`StructAI MCP tool not found: ${toolSuffix}. ` +
            `Tip: open a StructAI chat once so MCP tool discovery completes.`);
    }
    const cts = new vscode.CancellationTokenSource();
    const result = await vscode.lm.invokeTool(toolInfo.name, {
        input,
        toolInvocationToken: cts.token
    });
    return unwrapToolResult(result);
}
function unwrapToolResult(result) {
    if (result && typeof result === "object" && !Array.isArray(result)) {
        if (Array.isArray(result.content)) {
            return unwrapToolResult(result.content);
        }
    }
    if (Array.isArray(result)) {
        const text = result
            .map((p) => {
            if (typeof p === "string")
                return p;
            if (p && typeof p === "object") {
                if (typeof p.value === "string")
                    return p.value;
                if (typeof p.text === "string")
                    return p.text;
                if (typeof p.content === "string")
                    return p.content;
                return JSON.stringify(p);
            }
            return String(p);
        })
            .join("");
        try {
            return JSON.parse(text);
        }
        catch {
            return { raw: text };
        }
    }
    if (typeof result === "string") {
        try {
            return JSON.parse(result);
        }
        catch {
            return { raw: result };
        }
    }
    return result;
}
//# sourceMappingURL=invoke.js.map