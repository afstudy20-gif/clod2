const vscode = require("vscode");
const path = require("path");
const { spawn } = require("child_process");

let panel;
let backendProcess;

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand("clod.openChat", () => openChat(context)),
    vscode.commands.registerCommand("clod.sendSelection", () => sendSelection(context)),
    vscode.commands.registerCommand("clod.startBackend", () => startBackend(context))
  );
}

function deactivate() {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }
}

function getConfig() {
  return vscode.workspace.getConfiguration("clod");
}

function getServerUrl() {
  return getConfig().get("serverUrl", "http://127.0.0.1:8000").replace(/\/$/, "");
}

function getWorkspacePath() {
  const folder = vscode.workspace.workspaceFolders?.[0];
  return folder ? folder.uri.fsPath : undefined;
}

function getBackendPath(context) {
  const configured = getConfig().get("backendPath", "");
  if (configured && configured.trim()) return configured.trim();
  return path.resolve(context.extensionPath, "..");
}

async function startBackend(context) {
  if (backendProcess && !backendProcess.killed) {
    vscode.window.showInformationMessage("Clod backend is already running.");
    return;
  }

  const python = getConfig().get("pythonPath", "python3");
  const backendPath = getBackendPath(context);
  const terminalName = "Clod Backend";

  backendProcess = spawn(
    python,
    ["-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
    { cwd: backendPath, env: process.env }
  );

  const output = vscode.window.createOutputChannel(terminalName);
  output.show(true);
  output.appendLine(`Starting Clod backend in ${backendPath}`);

  backendProcess.stdout.on("data", chunk => output.append(chunk.toString()));
  backendProcess.stderr.on("data", chunk => output.append(chunk.toString()));
  backendProcess.on("exit", code => {
    output.appendLine(`\nClod backend exited with code ${code}`);
    backendProcess = undefined;
  });

  await waitForBackend();
  vscode.window.showInformationMessage("Clod backend is running.");
}

async function waitForBackend() {
  const deadline = Date.now() + 8000;
  while (Date.now() < deadline) {
    try {
      const resp = await fetch(`${getServerUrl()}/health`);
      if (resp.ok) return;
    } catch {
      // Keep waiting.
    }
    await new Promise(resolve => setTimeout(resolve, 400));
  }
}

function openChat(context) {
  if (panel) {
    panel.reveal(vscode.ViewColumn.Beside);
    return panel;
  }

  panel = vscode.window.createWebviewPanel(
    "clodAgent",
    "Clod Agent",
    vscode.ViewColumn.Beside,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
    }
  );

  panel.webview.html = getWebviewHtml(panel.webview);
  panel.onDidDispose(() => {
    panel = undefined;
  });

  panel.webview.onDidReceiveMessage(async message => {
    try {
      if (message.type === "send") {
        await streamChatToWebview(message.text || "", message.history || []);
      } else if (message.type === "selection") {
        panel.webview.postMessage({ type: "insert", text: getSelectionText() });
      } else if (message.type === "activeFile") {
        panel.webview.postMessage({ type: "insert", text: getActiveFileContext() });
      } else if (message.type === "startBackend") {
        await startBackend(context);
      }
    } catch (error) {
      panel?.webview.postMessage({
        type: "error",
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });

  return panel;
}

function sendSelection(context) {
  const target = openChat(context);
  target.webview.postMessage({ type: "insert", text: getSelectionText() });
}

function getSelectionText() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return "";
  const selection = editor.selection;
  const selected = editor.document.getText(selection);
  const file = editor.document.uri.fsPath;
  if (selected.trim()) {
    return `File: ${file}\n\nSelected code:\n\`\`\`\n${selected}\n\`\`\``;
  }
  return `Active file: ${file}`;
}

function getActiveFileContext() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return "No active file.";
  const file = editor.document.uri.fsPath;
  const text = editor.document.getText();
  return `Active file: ${file}\n\n\`\`\`\n${text.slice(0, 12000)}\n\`\`\``;
}

async function streamChatToWebview(text, history) {
  if (!panel) return;
  const config = getConfig();
  const payload = {
    messages: [...history, { role: "user", content: text }],
    provider: config.get("defaultProvider", "nvidia"),
    model: config.get("defaultModel", "nvidia/nemotron-3-super-120b-a12b"),
    mode: "build_debug",
    workspace: getWorkspacePath(),
  };

  panel.webview.postMessage({ type: "start", workspace: payload.workspace });

  const resp = await fetch(`${getServerUrl()}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${body}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6);
      if (raw === "[DONE]") {
        panel.webview.postMessage({ type: "done" });
        continue;
      }
      try {
        panel.webview.postMessage({ type: "event", event: JSON.parse(raw) });
      } catch {
        // Ignore malformed stream fragments.
      }
    }
  }

  panel.webview.postMessage({ type: "done" });
}

function getWebviewHtml(webview) {
  const nonce = String(Date.now());
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Clod Agent</title>
  <style>
    body { margin: 0; font: 13px var(--vscode-font-family); color: var(--vscode-foreground); background: var(--vscode-editor-background); }
    #app { display: grid; grid-template-rows: auto 1fr auto; height: 100vh; }
    header { display: flex; gap: 8px; align-items: center; padding: 10px; border-bottom: 1px solid var(--vscode-panel-border); }
    header strong { color: var(--vscode-charts-green); }
    button { color: var(--vscode-button-foreground); background: var(--vscode-button-background); border: 0; padding: 6px 10px; border-radius: 3px; cursor: pointer; }
    button.secondary { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); }
    #messages { overflow: auto; padding: 12px; }
    .msg { margin-bottom: 14px; padding: 10px; border: 1px solid var(--vscode-panel-border); border-radius: 5px; background: var(--vscode-editorWidget-background); white-space: pre-wrap; }
    .user { border-left: 3px solid var(--vscode-charts-blue); }
    .assistant { border-left: 3px solid var(--vscode-charts-green); }
    .tool { margin: 8px 0; padding: 8px; border-left: 3px solid var(--vscode-charts-purple); background: var(--vscode-textCodeBlock-background); }
    .meta { opacity: .7; font-size: 11px; margin-bottom: 6px; }
    textarea { width: 100%; min-height: 82px; resize: vertical; box-sizing: border-box; padding: 10px; color: var(--vscode-input-foreground); background: var(--vscode-input-background); border: 1px solid var(--vscode-input-border); }
    footer { display: grid; gap: 8px; padding: 10px; border-top: 1px solid var(--vscode-panel-border); }
    .row { display: flex; gap: 8px; align-items: center; }
    #status { opacity: .75; font-size: 11px; }
  </style>
</head>
<body>
  <div id="app">
    <header>
      <strong>Clod Agent</strong>
      <button class="secondary" id="backend">Start Backend</button>
      <button class="secondary" id="selection">Insert Selection</button>
      <button class="secondary" id="file">Insert Active File</button>
      <span id="status"></span>
    </header>
    <main id="messages"></main>
    <footer>
      <textarea id="input" placeholder="Ask Clod to build, debug, edit, run tests, or handle git in this VSCode workspace..."></textarea>
      <div class="row">
        <button id="send">Send</button>
        <span id="workspace"></span>
      </div>
    </footer>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const messages = [];
    let assistantText = "";
    let assistantTools = [];
    let assistantSaved = false;

    const els = {
      messages: document.getElementById("messages"),
      input: document.getElementById("input"),
      send: document.getElementById("send"),
      backend: document.getElementById("backend"),
      selection: document.getElementById("selection"),
      file: document.getElementById("file"),
      status: document.getElementById("status"),
      workspace: document.getElementById("workspace"),
    };

    els.backend.onclick = () => vscode.postMessage({ type: "startBackend" });
    els.selection.onclick = () => vscode.postMessage({ type: "selection" });
    els.file.onclick = () => vscode.postMessage({ type: "activeFile" });
    els.send.onclick = send;
    els.input.addEventListener("keydown", event => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) send();
    });

    function send() {
      const text = els.input.value.trim();
      if (!text) return;
      messages.push({ role: "user", content: text });
      renderMessage("user", text);
      els.input.value = "";
      assistantText = "";
      assistantTools = [];
      assistantSaved = false;
      renderMessage("assistant", "");
      vscode.postMessage({ type: "send", text, history: messages.slice(0, -1) });
    }

    window.addEventListener("message", event => {
      const msg = event.data;
      if (msg.type === "insert") {
        els.input.value = [els.input.value, msg.text].filter(Boolean).join("\\n\\n");
      } else if (msg.type === "start") {
        els.status.textContent = "running";
        els.workspace.textContent = msg.workspace ? "Workspace: " + msg.workspace : "No workspace folder";
      } else if (msg.type === "event") {
        handleStreamEvent(msg.event);
      } else if (msg.type === "done") {
        els.status.textContent = "done";
        if (!assistantSaved && (assistantText || assistantTools.length)) {
          messages.push({ role: "assistant", content: assistantText, toolEvents: assistantTools });
          assistantSaved = true;
        }
      } else if (msg.type === "error") {
        els.status.textContent = "error";
        appendToAssistant("\\nError: " + msg.error);
      }
    });

    function handleStreamEvent(event) {
      if (event.text) {
        appendToAssistant(event.text);
      } else if (event.tool_event) {
        assistantTools.push(event.tool_event);
        appendTool(event.tool_event);
      } else if (event.error) {
        appendToAssistant("\\nError: " + event.error);
      }
    }

    function appendToAssistant(text) {
      assistantText += text;
      const last = els.messages.querySelector(".assistant:last-child .content");
      if (last) last.textContent = assistantText;
      els.messages.scrollTop = els.messages.scrollHeight;
    }

    function appendTool(tool) {
      const last = els.messages.querySelector(".assistant:last-child .content");
      if (!last) return;
      const block = document.createElement("div");
      block.className = "tool";
      const args = tool.arguments ? JSON.stringify(tool.arguments, null, 2) : "";
      block.textContent = "* " + (tool.name || "tool") + " " + (tool.type || "") + "\\n" + (args || tool.result || "") + (tool.result ? "\\n" + tool.result : "");
      last.parentElement.appendChild(block);
      els.messages.scrollTop = els.messages.scrollHeight;
    }

    function renderMessage(role, text) {
      const wrap = document.createElement("div");
      wrap.className = "msg " + role;
      wrap.innerHTML = '<div class="meta">' + role + '</div><div class="content"></div>';
      wrap.querySelector(".content").textContent = text;
      els.messages.appendChild(wrap);
      els.messages.scrollTop = els.messages.scrollHeight;
    }
  </script>
</body>
</html>`;
}

module.exports = {
  activate,
  deactivate,
};
