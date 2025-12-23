import * as vscode from "vscode";

import { registerStructChatParticipant } from "./chat/participant";
import { registerStructMcpProvider } from "./mcp/provider";
import { MappingViewProvider } from "./views/mappingView";
import { ResultsViewProvider } from "./views/resultsView";
import { ResolveViewProvider } from "./views/resolveView";
import { QaViewProvider } from "./views/qaView";
import { ProjectsViewProvider } from "./views/projectsView";
import { GovernanceViewProvider } from "./views/governanceView";
import { MemoryViewProvider } from "./views/memoryView";

export function activate(context: vscode.ExtensionContext) {
  registerStructMcpProvider(context);
  registerStructChatParticipant(context);

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("structai.mapping", new MappingViewProvider(context))
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("structai.results", new ResultsViewProvider(context), {
      webviewOptions: { retainContextWhenHidden: true }
    })
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("structai.resolve", new ResolveViewProvider(context), {
      webviewOptions: { retainContextWhenHidden: true }
    })
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("structai.qa", new QaViewProvider(context), {
      webviewOptions: { retainContextWhenHidden: true }
    })
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("structai.projects", new ProjectsViewProvider(context), {
      webviewOptions: { retainContextWhenHidden: true }
    })
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("structai.governance", new GovernanceViewProvider(context), {
      webviewOptions: { retainContextWhenHidden: true }
    })
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider("structai.memory", new MemoryViewProvider(context))
  );

  const focusView = async (viewId: string) => {
    await vscode.commands.executeCommand("workbench.view.extension.structai.panel");
    await vscode.commands.executeCommand(`${viewId}.focus`);
  };

  context.subscriptions.push(
    vscode.commands.registerCommand("structai.openPanel", async () => {
      await vscode.commands.executeCommand("workbench.view.extension.structai.panel");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("structai.openMapping", async () => {
      await focusView("structai.mapping");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("structai.openResults", async () => {
      await focusView("structai.results");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("structai.openResolve", async () => {
      await focusView("structai.resolve");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("structai.openQa", async () => {
      await focusView("structai.qa");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("structai.openProjects", async () => {
      await focusView("structai.projects");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("structai.openGovernance", async () => {
      await focusView("structai.governance");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("structai.openMemory", async () => {
      await focusView("structai.memory");
    })
  );
}

export function deactivate() {}
