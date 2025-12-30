/// <reference types="vite/client" />

interface Window {
  require: NodeRequire
  electronAPI: {
    minimize: () => void
    maximize: () => void
    close: () => void
  }
}
