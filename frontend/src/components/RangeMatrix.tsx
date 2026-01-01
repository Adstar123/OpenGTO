import React, { useState } from 'react'
import { motion } from 'framer-motion'

// Hand strategy type
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

// Ranks in order for the matrix (A high to 2 low)
const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']

// Action colors matching the reference image
const ACTION_COLORS = {
  fold: '#0066cc',      // Blue
  check: '#32d74b',     // Green
  call: '#32d74b',      // Green
  raise: '#ff6b6b',     // Light red/coral
  allIn: '#cc0000',     // Deep red
}

// Get hand type string for a matrix position
function getHandType(row: number, col: number): string {
  const rank1 = RANKS[row]
  const rank2 = RANKS[col]

  if (row === col) {
    // Pair (diagonal)
    return `${rank1}${rank2}`
  } else if (row < col) {
    // Suited (above diagonal) - higher rank first
    return `${rank1}${rank2}s`
  } else {
    // Offsuit (below diagonal) - higher rank first
    return `${rank2}${rank1}o`
  }
}

// Get display label for a hand cell
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

// Calculate color mix based on strategy probabilities
function getStrategyColor(strategy: HandStrategy): string {
  if (!strategy) return ACTION_COLORS.fold

  // Get the dominant action
  const actions = Object.entries(strategy).filter(([_, v]) => v && v > 0.01)

  if (actions.length === 0) return ACTION_COLORS.fold

  // If single dominant action (>90%), use solid color
  const maxAction = actions.reduce((a, b) => (b[1] || 0) > (a[1] || 0) ? b : a)
  if ((maxAction[1] || 0) > 0.90) {
    return ACTION_COLORS[maxAction[0] as keyof typeof ACTION_COLORS] || ACTION_COLORS.fold
  }

  // Mix colors based on probabilities
  return blendColors(strategy)
}

// Blend colors based on action probabilities
function blendColors(strategy: HandStrategy): string {
  let r = 0, g = 0, b = 0
  let totalWeight = 0

  const colorMap: Record<string, [number, number, number]> = {
    fold: [0, 102, 204],      // Blue
    check: [50, 215, 75],     // Green
    call: [50, 215, 75],      // Green
    raise: [255, 107, 107],   // Light red
    allIn: [204, 0, 0],       // Deep red
  }

  for (const [action, prob] of Object.entries(strategy)) {
    if (prob && prob > 0.01 && colorMap[action]) {
      const [cr, cg, cb] = colorMap[action]
      r += cr * prob
      g += cg * prob
      b += cb * prob
      totalWeight += prob
    }
  }

  if (totalWeight > 0) {
    r = Math.round(r / totalWeight)
    g = Math.round(g / totalWeight)
    b = Math.round(b / totalWeight)
  }

  return `rgb(${r}, ${g}, ${b})`
}

// Generate gradient for mixed strategies
function getStrategyGradient(strategy: HandStrategy): string {
  if (!strategy) return ACTION_COLORS.fold

  const actions = Object.entries(strategy)
    .filter(([_, v]) => v && v > 0.01)
    .sort((a, b) => (b[1] || 0) - (a[1] || 0))

  if (actions.length <= 1) {
    const action = actions[0]?.[0] || 'fold'
    return ACTION_COLORS[action as keyof typeof ACTION_COLORS] || ACTION_COLORS.fold
  }

  // Create gradient stops
  const stops: string[] = []
  let position = 0

  for (const [action, prob] of actions) {
    const color = ACTION_COLORS[action as keyof typeof ACTION_COLORS] || ACTION_COLORS.fold
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
    <div className="range-matrix-container">
      <div className="range-matrix">
        {RANKS.map((_, rowIdx) => (
          <div key={rowIdx} className="matrix-row">
            {RANKS.map((_, colIdx) => {
              const handType = getHandType(rowIdx, colIdx)
              const displayLabel = getDisplayLabel(rowIdx, colIdx)
              const strategy = rangeData[handType]
              const isSelected = selectedHand === handType
              const isHovered = hoveredHand === handType

              return (
                <motion.div
                  key={`${rowIdx}-${colIdx}`}
                  className={`matrix-cell ${isSelected ? 'selected' : ''} ${isHovered ? 'hovered' : ''}`}
                  style={{
                    background: getStrategyGradient(strategy),
                  }}
                  onClick={() => onHandSelect(handType)}
                  onMouseEnter={() => setHoveredHand(handType)}
                  onMouseLeave={() => setHoveredHand(null)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span className="cell-label">{displayLabel}</span>
                </motion.div>
              )
            })}
          </div>
        ))}
      </div>

      {/* Legend */}
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
        .range-matrix-container {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .range-matrix {
          display: flex;
          flex-direction: column;
          gap: 2px;
          background: var(--bg-tertiary);
          padding: 4px;
          border-radius: 8px;
        }

        .matrix-row {
          display: flex;
          gap: 2px;
        }

        .matrix-cell {
          width: 42px;
          height: 42px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 4px;
          cursor: pointer;
          transition: all 0.15s ease;
          position: relative;
        }

        .matrix-cell.selected {
          outline: 2px solid var(--accent-primary);
          outline-offset: 1px;
        }

        .matrix-cell.hovered {
          z-index: 10;
        }

        .cell-label {
          font-size: 10px;
          font-weight: 600;
          color: white;
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
          pointer-events: none;
        }

        .matrix-legend {
          display: flex;
          gap: 16px;
          justify-content: center;
          padding: 8px;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 11px;
          color: var(--text-secondary);
        }

        .legend-color {
          width: 16px;
          height: 16px;
          border-radius: 3px;
        }
      `}</style>
    </div>
  )
}

export default RangeMatrix
