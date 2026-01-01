/// <reference types="vite/client" />

interface Window {
  electronAPI?: {
    minimize: () => void
    maximize: () => void
    close: () => void
  }
}
