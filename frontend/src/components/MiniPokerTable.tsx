import React from 'react'
import { motion } from 'framer-motion'
import { Position } from '../types'

type PositionAction = 'none' | 'fold' | 'call' | 'raise' | 'all-in'

interface PositionConfig {
  position: Position
  action: PositionAction
  raiseAmount?: number
}

interface MiniPokerTableProps {
  positions: PositionConfig[]
  heroPosition: Position
  pot: number
  currentBet: number
}

const ALL_POSITIONS: Position[] = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']

// Position coordinates for 6-max table (relative percentages)
const POSITION_COORDS: Record<Position, { x: number; y: number }> = {
  UTG: { x: 20, y: 70 },
  HJ: { x: 20, y: 30 },
  CO: { x: 50, y: 10 },
  BTN: { x: 80, y: 30 },
  SB: { x: 80, y: 70 },
  BB: { x: 50, y: 90 },
}

const ACTION_COLORS: Record<PositionAction, string> = {
  none: 'var(--text-muted)',
  fold: '#6b7280',
  call: '#32d74b',
  raise: '#ff9f0a',
  all: '#ff453a',
  'all-in': '#ff453a',
}

const ACTION_LABELS: Record<PositionAction, string> = {
  none: '-',
  fold: 'F',
  call: 'C',
  raise: 'R',
  'all-in': 'A',
}

const MiniPokerTable: React.FC<MiniPokerTableProps> = ({
  positions,
  heroPosition,
  pot,
  currentBet,
}) => {
  const getPositionAction = (pos: Position): PositionAction => {
    const config = positions.find(p => p.position === pos)
    return config?.action || 'none'
  }

  const getRaiseAmount = (pos: Position): number | undefined => {
    const config = positions.find(p => p.position === pos)
    return config?.raiseAmount
  }

  return (
    <div className="mini-table-container">
      <div className="mini-table">
        {/* Table felt */}
        <div className="table-felt">
          {/* Pot display */}
          <div className="pot-display">
            <span className="pot-label">Pot</span>
            <span className="pot-amount">{pot.toFixed(1)}bb</span>
          </div>
        </div>

        {/* Position markers */}
        {ALL_POSITIONS.map((pos) => {
          const coords = POSITION_COORDS[pos]
          const action = getPositionAction(pos)
          const raiseAmount = getRaiseAmount(pos)
          const isHero = pos === heroPosition

          return (
            <motion.div
              key={pos}
              className={`position-marker ${isHero ? 'hero' : ''} ${action !== 'none' ? 'has-action' : ''}`}
              style={{
                left: `${coords.x}%`,
                top: `${coords.y}%`,
              }}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: ALL_POSITIONS.indexOf(pos) * 0.05, type: 'spring' }}
            >
              <div className="marker-label">{pos}</div>
              {isHero ? (
                <div className="hero-indicator">?</div>
              ) : action !== 'none' ? (
                <div
                  className="action-indicator"
                  style={{ background: ACTION_COLORS[action] }}
                >
                  {ACTION_LABELS[action]}
                  {action === 'raise' && raiseAmount && (
                    <span className="raise-amount">{raiseAmount}</span>
                  )}
                </div>
              ) : (
                <div className="stack-display">200</div>
              )}
            </motion.div>
          )
        })}
      </div>

      {/* Info panel */}
      <div className="table-info">
        <div className="info-row">
          <span className="info-label">Current Bet</span>
          <span className="info-value">{currentBet.toFixed(1)}bb</span>
        </div>
      </div>

      <style>{`
        .mini-table-container {
          background: var(--bg-card);
          border-radius: 12px;
          padding: 16px;
          border: 1px solid var(--border-subtle);
        }

        .mini-table {
          position: relative;
          width: 100%;
          aspect-ratio: 1.5;
          min-height: 180px;
        }

        .table-felt {
          position: absolute;
          inset: 15%;
          background: linear-gradient(135deg, #1a472a 0%, #0d2818 100%);
          border-radius: 50%;
          border: 4px solid #8b4513;
          box-shadow: inset 0 0 30px rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .pot-display {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 2px;
        }

        .pot-label {
          font-size: 10px;
          color: rgba(255, 255, 255, 0.6);
          text-transform: uppercase;
        }

        .pot-amount {
          font-size: 14px;
          font-weight: 700;
          color: white;
        }

        .position-marker {
          position: absolute;
          transform: translate(-50%, -50%);
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
        }

        .marker-label {
          font-size: 10px;
          font-weight: 600;
          color: var(--text-muted);
          background: var(--bg-tertiary);
          padding: 2px 6px;
          border-radius: 4px;
        }

        .position-marker.hero .marker-label {
          color: var(--accent-primary);
          background: rgba(212, 175, 55, 0.2);
        }

        .hero-indicator {
          width: 28px;
          height: 28px;
          border-radius: 50%;
          background: linear-gradient(135deg, var(--accent-primary), #c9a227);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 14px;
          font-weight: 700;
          color: #1a1a1a;
        }

        .action-indicator {
          min-width: 24px;
          height: 24px;
          padding: 0 6px;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 2px;
          font-size: 11px;
          font-weight: 700;
          color: white;
        }

        .raise-amount {
          font-size: 9px;
          opacity: 0.9;
        }

        .stack-display {
          font-size: 10px;
          color: var(--text-muted);
        }

        .table-info {
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid var(--border-subtle);
        }

        .info-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .info-label {
          font-size: 11px;
          color: var(--text-muted);
        }

        .info-value {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-primary);
        }
      `}</style>
    </div>
  )
}

export default MiniPokerTable
