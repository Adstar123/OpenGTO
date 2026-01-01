"use strict";
const electron = require("electron");
const child_process = require("child_process");
const path = require("path");
const fs = require("fs");
let mainWindow = null;
let pythonProcess = null;
const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
const isDev = !!VITE_DEV_SERVER_URL;
function getResourcesPath() {
  if (isDev) {
    return path.join(__dirname, "../..");
  }
  return path.join(process.resourcesPath, "backend");
}
function createWindow() {
  mainWindow = new electron.BrowserWindow({
    width: 1500,
    height: 950,
    minWidth: 1200,
    minHeight: 800,
    frame: false,
    transparent: false,
    backgroundColor: "#0a0a0f",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true
    },
    show: false
  });
  mainWindow.once("ready-to-show", () => {
    mainWindow == null ? void 0 : mainWindow.show();
  });
  if (VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(VITE_DEV_SERVER_URL);
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
  }
  mainWindow.on("closed", () => {
    mainWindow = null;
    killPythonBackend();
  });
}
function killPythonBackend() {
  if (pythonProcess) {
    try {
      if (process.platform === "win32") {
        child_process.spawn("taskkill", ["/pid", String(pythonProcess.pid), "/f", "/t"]);
      } else {
        pythonProcess.kill("SIGTERM");
      }
    } catch (e) {
      console.error("Error killing Python process:", e);
    }
    pythonProcess = null;
  }
}
function startPythonBackend() {
  var _a, _b;
  const resourcesPath = getResourcesPath();
  if (isDev) {
    const pythonPath = "python";
    const scriptPath = path.join(resourcesPath, "api_server.py");
    console.log("Starting Python backend in dev mode...");
    console.log("Script path:", scriptPath);
    console.log("CWD:", resourcesPath);
    pythonProcess = child_process.spawn(pythonPath, [scriptPath], {
      cwd: resourcesPath,
      stdio: ["pipe", "pipe", "pipe"]
    });
  } else {
    const exePath = path.join(resourcesPath, "opengto_backend.exe");
    console.log("Starting Python backend in production mode...");
    console.log("Executable path:", exePath);
    if (!fs.existsSync(exePath)) {
      console.error("Backend executable not found:", exePath);
      return;
    }
    pythonProcess = child_process.execFile(exePath, [], {
      cwd: resourcesPath
    });
  }
  (_a = pythonProcess.stdout) == null ? void 0 : _a.on("data", (data) => {
    console.log(`Backend: ${data}`);
  });
  (_b = pythonProcess.stderr) == null ? void 0 : _b.on("data", (data) => {
    console.error(`Backend Error: ${data}`);
  });
  pythonProcess.on("close", (code) => {
    console.log(`Backend process exited with code ${code}`);
  });
  pythonProcess.on("error", (err) => {
    console.error("Failed to start backend:", err);
  });
}
electron.app.whenReady().then(() => {
  startPythonBackend();
  setTimeout(createWindow, 1e3);
});
electron.app.on("window-all-closed", () => {
  killPythonBackend();
  if (process.platform !== "darwin") {
    electron.app.quit();
  }
});
electron.app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});
electron.ipcMain.on("window:minimize", () => mainWindow == null ? void 0 : mainWindow.minimize());
electron.ipcMain.on("window:maximize", () => {
  if (mainWindow == null ? void 0 : mainWindow.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow == null ? void 0 : mainWindow.maximize();
  }
});
electron.ipcMain.on("window:close", () => mainWindow == null ? void 0 : mainWindow.close());
