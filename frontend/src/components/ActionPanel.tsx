import React from 'react'
import { motion } from 'framer-motion'
import Icon from '@mdi/react'
import {
  mdiCardsOutline,
  mdiCheck,
  mdiHandCoinOutline,
  mdiTrendingUp,
  mdiFlash
} from '@mdi/js'
import { Scenario, ActionType, TrainerResult } from '../types'

interface ActionPanelProps {
  scenario: Scenario | null
  onAction: (action: ActionType) => void
  disabled: boolean
  result: TrainerResult | null
}

const ACTION_CONFIG: Record<
  ActionType,
  { label: string; color: string; icon: string }
> = {
  fold: {
    label: 'Fold',
    color: 'var(--color-fold)',
    icon: mdiCardsOutline,
  },
  check: {
    label: 'Check',
    color: 'var(--color-check)',
    icon: mdiCheck,
  },
  call: {
    label: 'Call',
    color: 'var(--color-call)',
    icon: mdiHandCoinOutline,
  },
  raise: {
    label: 'Raise',
    color: 'var(--color-raise)',
    icon: mdiTrendingUp,
  },
  'all-in': {
    label: 'All-In',
    color: 'var(--color-allin)',
    icon: mdiFlash,
  },
}

const ActionPanel: React.FC<ActionPanelProps> = ({
  scenario,
  onAction,
  disabled,
  result: _result,
}) => {
  const legalActions = scenario?.legalActions || []

  return (
    <div className="action-panel">
      <div className="panel-header">
        <h2 className="panel-title">Your Action</h2>
        {scenario && (
          <div className="bet-info">
            <span className="bet-label">To Call</span>
            <span className="bet-value">{scenario.currentBet.toFixed(1)} BB</span>
          </div>
        )}
      </div>

      <div className="actions-grid">
        {(['fold', 'check', 'call', 'raise', 'all-in'] as ActionType[]).map(
          (action, index) => {
            const config = ACTION_CONFIG[action]
            const isLegal = legalActions.includes(action)
            const isDisabled = disabled || !isLegal

            return (
              <motion.button
                key={action}
                className={`action-btn ${isDisabled ? 'disabled' : ''}`}
                onClick={() => !isDisabled && onAction(action)}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  delay: index * 0.04,
                  type: 'spring',
                  stiffness: 300,
                  damping: 25,
                }}
                whileHover={!isDisabled ? { y: -2 } : {}}
                whileTap={!isDisabled ? { scale: 0.98 } : {}}
                style={{
                  '--action-color': config.color,
                } as React.CSSProperties}
              >
                <div className="btn-content">
                  <Icon path={config.icon} size={0.9} />
                  <span className="btn-label">{config.label}</span>
                </div>
              </motion.button>
            )
          }
        )}
      </div>

      {/* Keyboard shortcuts hint */}
      <div className="shortcuts-hint">
        <span className="shortcut-item"><kbd>F</kbd> Fold</span>
        <span className="shortcut-item"><kbd>X</kbd> Check</span>
        <span className="shortcut-item"><kbd>C</kbd> Call</span>
        <span className="shortcut-item"><kbd>R</kbd> Raise</span>
        <span className="shortcut-item"><kbd>A</kbd> All-In</span>
      </div>

      <style>{`
        .action-panel {
          background: var(--bg-card);
          border-radius: 16px;
          border: 1px solid var(--border-subtle);
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 16px;
          width: 100%;
        }

        .panel-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .panel-title {
          font-size: 15px;
          font-weight: 600;
          color: var(--text-primary);
          margin: 0;
        }

        .bet-info {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 6px 14px;
          background: var(--bg-elevated);
          border-radius: 10px;
          border: 1px solid var(--border-subtle);
        }

        .bet-label {
          font-size: 11px;
          color: var(--text-muted);
        }

        .bet-value {
          font-size: 14px;
          font-weight: 600;
          color: var(--text-primary);
          font-family: var(--font-mono);
        }

        .actions-grid {
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 10px;
        }

        .action-btn {
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 14px 10px;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-subtle);
          border-radius: 12px;
          cursor: pointer;
          transition: all var(--transition-base);
          color: var(--action-color);
        }

        .action-btn:not(.disabled):hover {
          border-color: var(--action-color);
          background: rgba(255, 255, 255, 0.03);
        }

        .action-btn.disabled {
          opacity: 0.25;
          cursor: not-allowed;
        }

        .btn-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 6px;
        }

        .btn-label {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-primary);
          letter-spacing: 0.01em;
        }

        .shortcuts-hint {
          display: flex;
          justify-content: center;
          gap: 14px;
          padding-top: 10px;
          border-top: 1px solid var(--border-subtle);
        }

        .shortcut-item {
          display: flex;
          align-items: center;
          gap: 5px;
          font-size: 10px;
          color: var(--text-muted);
        }

        .shortcut-item kbd {
          padding: 2px 5px;
          background: var(--bg-elevated);
          border: 1px solid var(--border-light);
          border-radius: 4px;
          font-family: var(--font-mono);
          font-size: 9px;
          font-weight: 600;
        }
      `}</style>
    </div>
  )
}

export default ActionPanel
