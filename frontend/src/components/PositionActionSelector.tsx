import React from 'react'
import { motion } from 'framer-motion'
import { Position } from '../types'

type PositionAction = 'none' | 'fold' | 'call' | 'raise' | 'all-in'

interface PositionConfig {
  position: Position
  action: PositionAction
  raiseAmount?: number
}

interface PositionActionSelectorProps {
  positions: PositionConfig[]
  heroPosition: Position
  onPositionChange: (position: Position, action: PositionAction, amount?: number) => void
  onHeroPositionChange: (position: Position) => void
}

const ALL_POSITIONS: Position[] = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']

const ACTION_OPTIONS: { value: PositionAction; label: string }[] = [
  { value: 'none', label: '-' },
  { value: 'fold', label: 'Fold' },
  { value: 'call', label: 'Call' },
  { value: 'raise', label: 'Raise 2.5' },
  { value: 'all-in', label: 'All-In' },
]

const PositionActionSelector: React.FC<PositionActionSelectorProps> = ({
  positions,
  heroPosition,
  onPositionChange,
  onHeroPositionChange,
}) => {
  const getPositionConfig = (pos: Position): PositionConfig | undefined => {
    return positions.find(p => p.position === pos)
  }

  const getHeroIndex = () => ALL_POSITIONS.indexOf(heroPosition)

  // Determine which positions can have actions (before hero)
  const canHaveAction = (pos: Position): boolean => {
    const posIndex = ALL_POSITIONS.indexOf(pos)
    const heroIndex = getHeroIndex()
    // Positions before hero can have actions, but not SB/BB in preflop (blinds already posted)
    return posIndex < heroIndex && pos !== 'SB' && pos !== 'BB'
  }

  return (
    <div className="position-action-selector">
      <div className="selector-header">
        <h3>Position Actions</h3>
        <div className="hero-selector">
          <label>Hero:</label>
          <select
            value={heroPosition}
            onChange={(e) => onHeroPositionChange(e.target.value as Position)}
          >
            {ALL_POSITIONS.map(pos => (
              <option key={pos} value={pos}>{pos}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="positions-grid">
        {ALL_POSITIONS.map((pos) => {
          const config = getPositionConfig(pos)
          const isHero = pos === heroPosition
          const canAct = canHaveAction(pos)

          return (
            <motion.div
              key={pos}
              className={`position-card ${isHero ? 'hero' : ''} ${!canAct && !isHero ? 'disabled' : ''}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: ALL_POSITIONS.indexOf(pos) * 0.05 }}
            >
              <div className="position-label">{pos}</div>
              {isHero ? (
                <div className="hero-badge">HERO</div>
              ) : canAct ? (
                <select
                  className="action-select"
                  value={config?.action || 'none'}
                  onChange={(e) => {
                    const action = e.target.value as PositionAction
                    onPositionChange(pos, action, action === 'raise' ? 2.5 : undefined)
                  }}
                >
                  {ACTION_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              ) : (
                <div className="no-action">-</div>
              )}
            </motion.div>
          )
        })}
      </div>

      <style>{`
        .position-action-selector {
          background: var(--bg-card);
          border-radius: 12px;
          padding: 16px;
          border: 1px solid var(--border-subtle);
        }

        .selector-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }

        .selector-header h3 {
          font-size: 14px;
          font-weight: 600;
          color: var(--text-primary);
          margin: 0;
        }

        .hero-selector {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .hero-selector label {
          font-size: 12px;
          color: var(--text-secondary);
        }

        .hero-selector select {
          background: var(--bg-tertiary);
          border: 1px solid var(--border-subtle);
          border-radius: 6px;
          padding: 4px 8px;
          font-size: 12px;
          color: var(--text-primary);
          cursor: pointer;
        }

        .positions-grid {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }

        .position-card {
          background: var(--bg-tertiary);
          border-radius: 8px;
          padding: 10px 12px;
          min-width: 80px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 6px;
          border: 1px solid transparent;
          transition: all 0.2s ease;
        }

        .position-card.hero {
          background: linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(212, 175, 55, 0.1));
          border-color: var(--accent-primary);
        }

        .position-card.disabled {
          opacity: 0.5;
        }

        .position-label {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .hero-badge {
          font-size: 10px;
          font-weight: 700;
          color: var(--accent-primary);
          background: rgba(212, 175, 55, 0.2);
          padding: 2px 8px;
          border-radius: 4px;
        }

        .action-select {
          background: var(--bg-secondary);
          border: 1px solid var(--border-subtle);
          border-radius: 4px;
          padding: 4px 6px;
          font-size: 11px;
          color: var(--text-primary);
          cursor: pointer;
          width: 100%;
        }

        .action-select:focus {
          outline: none;
          border-color: var(--accent-primary);
        }

        .no-action {
          font-size: 11px;
          color: var(--text-muted);
        }
      `}</style>
    </div>
  )
}

export default PositionActionSelector
