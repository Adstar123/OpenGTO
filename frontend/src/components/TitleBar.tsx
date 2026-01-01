import React from 'react'
import { motion } from 'framer-motion'
import Icon from '@mdi/react'
import { mdiCardsPlayingSpadeMultiple, mdiWindowMinimize, mdiWindowMaximize, mdiClose } from '@mdi/js'

const TitleBar: React.FC = () => {
  return (
    <div className="title-bar drag-region">
      <div className="title-left">
        <motion.div
          className="logo"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div className="logo-icon">
            <Icon path={mdiCardsPlayingSpadeMultiple} size={1} color="var(--accent-primary)" />
          </div>
          <span className="logo-text">OpenGTO</span>
        </motion.div>
      </div>

      <div className="title-center">
        <motion.span
          className="title-badge"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.3 }}
        >
          Preflop Trainer
        </motion.span>
      </div>

      <div className="title-right no-drag">
        <motion.button
          className="window-btn"
          onClick={() => window.electronAPI?.minimize()}
          whileHover={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }}
          whileTap={{ scale: 0.95 }}
        >
          <Icon path={mdiWindowMinimize} size={0.7} />
        </motion.button>
        <motion.button
          className="window-btn"
          onClick={() => window.electronAPI?.maximize()}
          whileHover={{ backgroundColor: 'rgba(255, 255, 255, 0.1)' }}
          whileTap={{ scale: 0.95 }}
        >
          <Icon path={mdiWindowMaximize} size={0.7} />
        </motion.button>
        <motion.button
          className="window-btn close"
          onClick={() => window.electronAPI?.close()}
          whileHover={{ backgroundColor: '#ff453a' }}
          whileTap={{ scale: 0.95 }}
        >
          <Icon path={mdiClose} size={0.7} />
        </motion.button>
      </div>

      <style>{`
        .title-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          height: 44px;
          padding: 0 16px;
          background: var(--bg-secondary);
          border-bottom: 1px solid var(--border-subtle);
          position: relative;
          z-index: 100;
        }

        .title-left {
          display: flex;
          align-items: center;
        }

        .logo {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .logo-icon {
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .logo-text {
          font-size: 15px;
          font-weight: 600;
          color: var(--text-primary);
          letter-spacing: -0.02em;
        }

        .title-center {
          position: absolute;
          left: 50%;
          transform: translateX(-50%);
        }

        .title-badge {
          font-size: 12px;
          font-weight: 500;
          color: var(--text-muted);
          letter-spacing: 0.02em;
        }

        .title-right {
          display: flex;
          gap: 4px;
        }

        .window-btn {
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: transparent;
          border: none;
          border-radius: 6px;
          color: var(--text-secondary);
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .window-btn:hover {
          color: var(--text-primary);
        }

        .window-btn.close:hover {
          color: white;
        }
      `}</style>
    </div>
  )
}

export default TitleBar
