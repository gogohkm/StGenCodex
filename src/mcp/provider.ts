import * as path from "path";
import * as vscode from "vscode";

export function registerStructMcpProvider(context: vscode.ExtensionContext) {
  const lm = (vscode as any).lm;
  if (!lm || typeof lm.registerMcpServerDefinitionProvider !== "function") {
    return;
  }

  const didChange = new vscode.EventEmitter<void>();

  const provider: any = {
    onDidChangeMcpServerDefinitions: didChange.event,
    provideMcpServerDefinitions: async () => {
      const config = vscode.workspace.getConfiguration();
      const pythonPath = config.get<string>("structai.pythonPath") || "python";
      const configuredDbPath = config.get<string>("structai.dbPath") || "";

      const serverScript = vscode.Uri.joinPath(context.extensionUri, "mcp_server", "server.py");
      const cwd = vscode.workspace.workspaceFolders?.[0]?.uri ?? vscode.Uri.file(process.cwd());

      const rootPath = path.join(cwd.fsPath, ".structai");
      const dbPath = configuredDbPath || path.join(rootPath, "structai.sqlite");

      const env: Record<string, string> = {
        STRUCTAI_ROOT: rootPath,
        STRUCTAI_DB: dbPath
      };

      const def = new (vscode as any).McpStdioServerDefinition(
        "StructAI Local MCP",
        pythonPath,
        ["-u", serverScript.fsPath],
        env,
        "0.1.4"
      );
      def.cwd = cwd;

      return [def];
    },
    resolveMcpServerDefinition: async (server: any) => {
      return server;
    }
  };

  context.subscriptions.push(lm.registerMcpServerDefinitionProvider("structaiMcpProvider", provider));
  context.subscriptions.push(didChange);
}
