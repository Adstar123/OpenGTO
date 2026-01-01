import React, { useState } from 'react'
import { motion } from 'framer-motion'

interface HandStrategy {
  fold?: number
  check?: number
  call?: number
  raise?: number
  allIn?: number
}

interface RangeMatrixProps {
  rangeData: Record<string, HandStrategy>
  selectedHand: string | null
  onHandSelect: (hand: string) => void
}

const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

const ACTION_COLORS: Record<string, string> = {
  fold: '#3b82f6',
  check: '#22c55e',
  call: '#22c55e',
  raise: '#f97316',
  allIn: '#ef4444',
}

function getHandType(row: number, col: number): string {
  const rank1 = RANKS[row]
  const rank2 = RANKS[col]

  if (row === col) {
    return `${rank1}${rank2}`
  } else if (row < col) {
    return `${rank1}${rank2}s`
  } else {
    return `${rank2}${rank1}o`
  }
}

function getDisplayLabel(row: number, col: number): string {
  const rank1 = RANKS[row]
  const rank2 = RANKS[col]

  if (row === col) {
    return `${rank1}${rank2}`
  } else if (row < col) {
    return `${rank1}${rank2}s`
  } else {
    return `${rank2}${rank1}o`
  }
}

function getStrategyGradient(strategy: HandStrategy): string {
  if (!strategy) return ACTION_COLORS.fold

  const actions = Object.entries(strategy)
    .filter(([_, v]) => v && v > 0.01)
    .sort((a, b) => (b[1] || 0) - (a[1] || 0))

  if (actions.length === 0) return ACTION_COLORS.fold

  if (actions.length === 1) {
    return ACTION_COLORS[actions[0][0]] || ACTION_COLORS.fold
  }

  // Create gradient stops for mixed strategies
  const stops: string[] = []
  let position = 0

  for (const [action, prob] of actions) {
    const color = ACTION_COLORS[action] || ACTION_COLORS.fold
    const percentage = (prob || 0) * 100
    stops.push(`${color} ${position}%`)
    position += percentage
    stops.push(`${color} ${position}%`)
  }

  return `linear-gradient(to right, ${stops.join(', ')})`
}

const RangeMatrix: React.FC<RangeMatrixProps> = ({
  rangeData,
  selectedHand,
  onHandSelect,
}) => {
  const [hoveredHand, setHoveredHand] = useState<string | null>(null)

  return (
    <div className="range-matrix-wrapper">
      <div className="range-matrix">
        {RANKS.map((_, rowIdx) => (
          <div key={rowIdx} className="matrix-row">
            {RANKS.map((_, colIdx) => {
              const handType = getHandType(rowIdx, colIdx)
              const displayLabel = getDisplayLabel(rowIdx, colIdx)
              const strategy = rangeData[handType]
              const isSelected = selectedHand === handType
              const isHovered = hoveredHand === handType
              const isPair = rowIdx === colIdx
              const isSuited = rowIdx < colIdx

              return (
                <motion.div
                  key={`${rowIdx}-${colIdx}`}
                  className={`matrix-cell ${isSelected ? 'selected' : ''} ${isHovered ? 'hovered' : ''} ${isPair ? 'pair' : ''} ${isSuited ? 'suited' : 'offsuit'}`}
                  style={{
                    background: getStrategyGradient(strategy),
                  }}
                  onClick={() => onHandSelect(handType)}
                  onMouseEnter={() => setHoveredHand(handType)}
                  onMouseLeave={() => setHoveredHand(null)}
                  whileHover={{ scale: 1.08, zIndex: 10 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span className="cell-label">{displayLabel}</span>
                </motion.div>
              )
            })}
          </div>
        ))}
      </div>

      <div className="matrix-legend">
        <div className="legend-item">
          <div className="legend-color" style={{ background: ACTION_COLORS.fold }} />
          <span>Fold</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: ACTION_COLORS.call }} />
          <span>Call/Check</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: ACTION_COLORS.raise }} />
          <span>Raise</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: ACTION_COLORS.allIn }} />
          <span>All-In</span>
        </div>
      </div>

      <style>{`
        .range-matrix-wrapper {
          display: flex;
          flex-direction: column;
          gap: 16px;
          flex: 1;
          min-height: 0;
        }

        .range-matrix {
          display: grid;
          grid-template-rows: repeat(13, 1fr);
          gap: 3px;
          background: var(--bg-card);
          padding: 12px;
          border-radius: 12px;
          border: 1px solid var(--border-subtle);
          flex: 1;
          min-height: 0;
          aspect-ratio: 1;
          max-height: calc(100vh - 280px);
          width: auto;
          align-self: flex-start;
        }

        .matrix-row {
          display: grid;
          grid-template-columns: repeat(13, 1fr);
          gap: 3px;
          min-height: 0;
        }

        .matrix-cell {
          aspect-ratio: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 4px;
          cursor: pointer;
          transition: box-shadow 0.15s ease;
          position: relative;
          min-width: 0;
          min-height: 0;
        }

        .matrix-cell.pair {
          border: 1px solid rgba(255, 255, 255, 0.15);
        }

        .matrix-cell.selected {
          box-shadow:
            0 0 0 2px var(--bg-card),
            0 0 0 4px var(--accent-primary);
        }

        .matrix-cell:hover {
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }

        .cell-label {
          font-size: clamp(8px, 1.2vw, 12px);
          font-weight: 700;
          color: white;
          text-shadow:
            0 1px 2px rgba(0, 0, 0, 0.6),
            0 0 4px rgba(0, 0, 0, 0.3);
          pointer-events: none;
          user-select: none;
        }

        .matrix-legend {
          display: flex;
          gap: 24px;
          justify-content: center;
          padding: 12px 16px;
          background: var(--bg-card);
          border-radius: 10px;
          border: 1px solid var(--border-subtle);
          flex-shrink: 0;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          font-weight: 500;
          color: var(--text-secondary);
        }

        .legend-color {
          width: 18px;
          height: 18px;
          border-radius: 4px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        }
      `}</style>
    </div>
  )
}

export default RangeMatrix
