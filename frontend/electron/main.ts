import { app, BrowserWindow, ipcMain } from 'electron'
import { spawn, ChildProcess, execFile } from 'child_process'
import path from 'path'
import fs from 'fs'

let mainWindow: BrowserWindow | null = null
let pythonProcess: ChildProcess | null = null

const VITE_DEV_SERVER_URL = process.env['VITE_DEV_SERVER_URL']
const isDev = !!VITE_DEV_SERVER_URL

function getResourcesPath(): string {
  if (isDev) {
    return path.join(__dirname, '../..')
  }
  // In production, resources are in the app.asar.unpacked or extraResources
  return path.join(process.resourcesPath, 'backend')
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1500,
    height: 950,
    minWidth: 1200,
    minHeight: 800,
    frame: false,
    transparent: false,
    backgroundColor: '#0a0a0f',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
    show: false,
  })

  // Smooth window appearance
  mainWindow.once('ready-to-show', () => {
    mainWindow?.show()
  })

  if (VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(VITE_DEV_SERVER_URL)
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
    killPythonBackend()
  })
}

function killPythonBackend() {
  if (pythonProcess) {
    try {
      // On Windows, we need to kill the process tree
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', String(pythonProcess.pid), '/f', '/t'])
      } else {
        pythonProcess.kill('SIGTERM')
      }
    } catch (e) {
      console.error('Error killing Python process:', e)
    }
    pythonProcess = null
  }
}

function startPythonBackend() {
  const resourcesPath = getResourcesPath()

  if (isDev) {
    // Development mode: run Python directly
    const pythonPath = 'python'
    const scriptPath = path.join(resourcesPath, 'api_server.py')

    console.log('Starting Python backend in dev mode...')
    console.log('Script path:', scriptPath)
    console.log('CWD:', resourcesPath)

    pythonProcess = spawn(pythonPath, [scriptPath], {
      cwd: resourcesPath,
      stdio: ['pipe', 'pipe', 'pipe'],
    })
  } else {
    // Production mode: run the bundled executable
    const exePath = path.join(resourcesPath, 'opengto_backend.exe')

    console.log('Starting Python backend in production mode...')
    console.log('Executable path:', exePath)

    if (!fs.existsSync(exePath)) {
      console.error('Backend executable not found:', exePath)
      return
    }

    pythonProcess = execFile(exePath, [], {
      cwd: resourcesPath,
    })
  }

  pythonProcess.stdout?.on('data', (data) => {
    console.log(`Backend: ${data}`)
  })

  pythonProcess.stderr?.on('data', (data) => {
    console.error(`Backend Error: ${data}`)
  })

  pythonProcess.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`)
  })

  pythonProcess.on('error', (err) => {
    console.error('Failed to start backend:', err)
  })
}

app.whenReady().then(() => {
  startPythonBackend()
  setTimeout(createWindow, 1000)
})

app.on('window-all-closed', () => {
  killPythonBackend()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow()
  }
})

// Window control handlers
ipcMain.on('window:minimize', () => mainWindow?.minimize())
ipcMain.on('window:maximize', () => {
  if (mainWindow?.isMaximized()) {
    mainWindow.unmaximize()
  } else {
    mainWindow?.maximize()
  }
})
ipcMain.on('window:close', () => mainWindow?.close())
