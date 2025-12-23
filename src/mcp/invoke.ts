import * as vscode from "vscode";

export async function invokeStructAi(toolSuffix: string, input: any): Promise<any> {
  const lm = (vscode as any).lm;
  if (!lm || !Array.isArray(lm.tools) || typeof lm.invokeTool !== "function") {
    throw new Error("Language model tools are not available in this VS Code build.");
  }

  const toolInfo = lm.tools.find((t: any) => {
    const name = (t as any).name as string | undefined;
    return Boolean(name && name.includes(toolSuffix));
  });

  if (!toolInfo) {
    throw new Error(
      `StructAI MCP tool not found: ${toolSuffix}. ` +
        `Tip: open a StructAI chat once so MCP tool discovery completes.`
    );
  }

  const cts = new vscode.CancellationTokenSource();
  const result: any = await lm.invokeTool((toolInfo as any).name, {
    input,
    toolInvocationToken: cts.token
  });
  return unwrapToolResult(result);
}

export function unwrapToolResult(result: any): any {
  if (result && typeof result === "object" && !Array.isArray(result)) {
    if (Array.isArray((result as any).content)) {
      return unwrapToolResult((result as any).content);
    }
  }

  if (Array.isArray(result)) {
    const text = result
      .map((p) => {
        if (typeof p === "string") return p;
        if (p && typeof p === "object") {
          if (typeof (p as any).value === "string") return (p as any).value;
          if (typeof (p as any).text === "string") return (p as any).text;
          if (typeof (p as any).content === "string") return (p as any).content;
          return JSON.stringify(p);
        }
        return String(p);
      })
      .join("");

    try {
      return JSON.parse(text);
    } catch {
      return { raw: text };
    }
  }

  if (typeof result === "string") {
    try {
      return JSON.parse(result);
    } catch {
      return { raw: result };
    }
  }

  return result;
}
