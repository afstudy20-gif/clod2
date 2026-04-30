const { app, BrowserWindow, Menu, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const net = require("net");
const path = require("path");

const PRODUCT_NAME = "Clod";
const DEFAULT_PORT = Number(process.env.MACOS_APP_PORT || 8765);
const BACKEND_COMMAND = "python3 {project_dir}/api.py";
const START_URL_TEMPLATE = "http://localhost:{port}";
const HEALTH_URL_TEMPLATE = "http://localhost:{port}/health";

let backendProcess = null;
let mainWindow = null;

function appRoot() {
  if (app.isPackaged) return path.join(process.resourcesPath, "app");
  return path.resolve(__dirname, "..");
}

function renderTemplate(value, port) {
  return String(value || "").replace(/\{port\}/g, String(port));
}

function isPortOpen(port) {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host: "127.0.0.1", port });
    socket.once("connect", () => {
      socket.destroy();
      resolve(true);
    });
    socket.once("error", () => resolve(false));
    socket.setTimeout(500, () => {
      socket.destroy();
      resolve(false);
    });
  });
}

async function findPort(startPort) {
  for (let port = startPort; port < startPort + 50; port += 1) {
    if (!(await isPortOpen(port))) return port;
  }
  throw new Error(`No free localhost port found near ${startPort}`);
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForHealth(url, timeoutMs = 30000) {
  if (!url) {
    await sleep(1800);
    return;
  }
  const startedAt = Date.now();
  let lastError = "";
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
      lastError = `HTTP ${response.status}`;
    } catch (error) {
      lastError = error.message;
    }
    await sleep(500);
  }
  throw new Error(`Backend did not become healthy: ${lastError}`);
}

async function startBackend() {
  if (!BACKEND_COMMAND) return renderTemplate(START_URL_TEMPLATE, DEFAULT_PORT);

  const port = await findPort(DEFAULT_PORT);
  const root = appRoot();
  const command = renderTemplate(BACKEND_COMMAND, port);
  const startUrl = renderTemplate(START_URL_TEMPLATE, port);
  const healthUrl = renderTemplate(HEALTH_URL_TEMPLATE, port);
  const env = { ...process.env, PORT: String(port), PYTHONUNBUFFERED: "1" };

  backendProcess = spawn(command, {
    cwd: root,
    env,
    shell: true,
    stdio: ["ignore", "pipe", "pipe"]
  });

  backendProcess.stdout.on("data", (data) => process.stdout.write(`[app-backend] ${data}`));
  backendProcess.stderr.on("data", (data) => process.stderr.write(`[app-backend] ${data}`));
  await waitForHealth(healthUrl);
  return startUrl;
}

function createMenu() {
  const template = [
    {
      label: PRODUCT_NAME,
      submenu: [
        { role: "about" },
        { type: "separator" },
        { role: "quit" }
      ]
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
        { role: "pasteAndMatchStyle" },
        { role: "delete" },
        { type: "separator" },
        { role: "selectAll" }
      ]
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { type: "separator" },
        { role: "toggleDevTools" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" }
      ]
    },
    {
      label: "Help",
      submenu: [
        { label: "Open Project Folder", click: () => shell.openPath(appRoot()) }
      ]
    }
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function installContextMenu(win) {
  win.webContents.on("context-menu", (_event, params) => {
    const template = [];
    if (params.isEditable) {
      template.push(
        { role: "undo", enabled: params.editFlags.canUndo },
        { role: "redo", enabled: params.editFlags.canRedo },
        { type: "separator" },
        { role: "cut", enabled: params.editFlags.canCut },
        { role: "copy", enabled: params.editFlags.canCopy },
        { role: "paste", enabled: params.editFlags.canPaste },
        { role: "pasteAndMatchStyle", enabled: params.editFlags.canPaste },
        { role: "delete", enabled: params.editFlags.canDelete },
        { type: "separator" },
        { role: "selectAll", enabled: params.editFlags.canSelectAll }
      );
    } else {
      template.push(
        { role: "copy", enabled: params.selectionText.length > 0 },
        { role: "selectAll" }
      );
    }
    Menu.buildFromTemplate(template).popup({ window: win });
  });
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 640,
    title: PRODUCT_NAME,
    backgroundColor: "#101114",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  installContextMenu(mainWindow);

  try {
    const url = await startBackend();
    await mainWindow.loadURL(url);
  } catch (error) {
    await dialog.showMessageBox({
      type: "error",
      title: `${PRODUCT_NAME} could not start`,
      message: "The local app backend could not be started.",
      detail: error.stack || error.message
    });
    app.quit();
  }
}

function stopBackend() {
  if (backendProcess && !backendProcess.killed) backendProcess.kill("SIGTERM");
}

app.whenReady().then(() => {
  createMenu();
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("before-quit", stopBackend);
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
