import React from 'react'
import { motion } from 'framer-motion'

interface ActionDistributionProps {
  distribution: {
    fold?: number
    check?: number
    call?: number
    raise?: number
    allIn?: number
  }
  legalActions: string[]
}

const ACTION_CONFIG = [
  { key: 'allIn', label: 'All-In', color: '#cc0000' },
  { key: 'raise', label: 'Raise', color: '#ff6b6b' },
  { key: 'call', label: 'Call', color: '#32d74b' },
  { key: 'check', label: 'Check', color: '#32d74b' },
  { key: 'fold', label: 'Fold', color: '#0066cc' },
]

const ActionDistribution: React.FC<ActionDistributionProps> = ({
  distribution,
  legalActions,
}) => {
  // Filter to only show relevant actions
  const relevantActions = ACTION_CONFIG.filter(action => {
    const value = distribution[action.key as keyof typeof distribution]
    return value !== undefined && value > 0.001
  })

  // Calculate total for percentage display
  const total = relevantActions.reduce((sum, action) => {
    return sum + (distribution[action.key as keyof typeof distribution] || 0)
  }, 0)

  return (
    <div className="action-distribution">
      <h3 className="distribution-title">Actions</h3>

      <div className="distribution-bars">
        {relevantActions.map((action, index) => {
          const value = distribution[action.key as keyof typeof distribution] || 0
          const percentage = total > 0 ? (value / total) * 100 : 0

          return (
            <motion.div
              key={action.key}
              className="distribution-row"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <div
                className="action-block"
                style={{ background: action.color }}
              >
                <span className="action-label">{action.label}</span>
                <span className="action-percent">{(value * 100).toFixed(1)}%</span>
              </div>
            </motion.div>
          )
        })}
      </div>

      {/* Stacked bar visualization */}
      <div className="stacked-bar">
        {relevantActions.map((action) => {
          const value = distribution[action.key as keyof typeof distribution] || 0
          const percentage = value * 100

          if (percentage < 0.1) return null

          return (
            <motion.div
              key={action.key}
              className="bar-segment"
              style={{
                background: action.color,
                width: `${percentage}%`,
              }}
              initial={{ width: 0 }}
              animate={{ width: `${percentage}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            />
          )
        })}
      </div>

      <style>{`
        .action-distribution {
          background: var(--bg-card);
          border-radius: 12px;
          padding: 16px;
          border: 1px solid var(--border-subtle);
        }

        .distribution-title {
          font-size: 14px;
          font-weight: 600;
          color: var(--text-primary);
          margin: 0 0 12px 0;
        }

        .distribution-bars {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 12px;
        }

        .distribution-row {
          flex: 1;
          min-width: 80px;
        }

        .action-block {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 12px 8px;
          border-radius: 8px;
          min-height: 60px;
        }

        .action-label {
          font-size: 12px;
          font-weight: 600;
          color: white;
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        }

        .action-percent {
          font-size: 18px;
          font-weight: 700;
          color: white;
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
        }

        .stacked-bar {
          display: flex;
          height: 8px;
          border-radius: 4px;
          overflow: hidden;
          background: var(--bg-tertiary);
        }

        .bar-segment {
          height: 100%;
          transition: width 0.3s ease;
        }
      `}</style>
    </div>
  )
}

export default ActionDistribution
