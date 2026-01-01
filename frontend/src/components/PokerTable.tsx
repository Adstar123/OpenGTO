import React, { useEffect, useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Scenario, Position, Player } from '../types'
import PlayingCard from './PlayingCard'
import PlayerSeat from './PlayerSeat'

interface PokerTableProps {
  scenario: Scenario | null
  isAnimating: boolean
}

// Fixed visual positions around the table (hero always at bottom/center)
// These are percentage-based offsets from center of table
const VISUAL_POSITIONS = [
  { x: 0, y: 55 },      // 0 - Hero (bottom center)
  { x: -38, y: 30 },    // 1 - Bottom-left
  { x: -42, y: -8 },    // 2 - Left
  { x: -28, y: -42 },   // 3 - Top-left
  { x: 28, y: -42 },    // 4 - Top-right
  { x: 42, y: -8 },     // 5 - Right
]

// All poker positions in order
const ALL_POSITIONS: Position[] = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']

const PokerTable: React.FC<PokerTableProps> = ({ scenario, isAnimating }) => {
  const [revealedPositions, setRevealedPositions] = useState<Set<Position>>(new Set())

  // Calculate position mapping so hero is always at the bottom
  const positionMapping = useMemo(() => {
    if (!scenario) return {}

    const heroIndex = ALL_POSITIONS.indexOf(scenario.heroPosition)
    const mapping: Record<Position, number> = {} as Record<Position, number>

    for (let i = 0; i < ALL_POSITIONS.length; i++) {
      const pos = ALL_POSITIONS[i]
      const offset = (i - heroIndex + ALL_POSITIONS.length) % ALL_POSITIONS.length
      mapping[pos] = offset
    }

    return mapping
  }, [scenario?.heroPosition])

  // Animate actions sequentially
  useEffect(() => {
    if (!scenario) {
      setRevealedPositions(new Set())
      return
    }

    const animateActions = async () => {
      setRevealedPositions(new Set())

      for (let i = 0; i < scenario.actions.length; i++) {
        await new Promise((resolve) => setTimeout(resolve, 350))
        setRevealedPositions((prev) => new Set([...prev, scenario.actions[i].position]))
      }
    }

    animateActions()
  }, [scenario])

  const getPlayerAtPosition = (position: Position): Player | undefined => {
    return scenario?.players.find((p) => p.position === position)
  }

  const getActionForPosition = (position: Position) => {
    return scenario?.actions.find((a) => a.position === position)
  }

  return (
    <div className="poker-table-container">
      {/* Ambient background */}
      <div className="ambient-bg" />

      {/* Main table */}
      <motion.div
        className="poker-table"
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        {/* Table felt */}
        <div className="table-felt">
          <div className="felt-texture" />
          <div className="felt-highlight" />
        </div>

        {/* Center pot area - positioned relative to poker-table */}
        <div className="pot-area">
          <div className="pot-display">
            <span className="pot-label">POT</span>
            <span className="pot-amount">
              {scenario?.pot.toFixed(1) || '0.0'} BB
            </span>
          </div>
        </div>

        {/* Hero cards - wrapper handles positioning, motion handles animation */}
        <div className="hero-cards-wrapper">
          <AnimatePresence>
            {scenario?.heroCards && (
              <motion.div
                className="hero-cards"
                initial={{ opacity: 0, y: 20, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.95 }}
                transition={{
                  type: 'spring',
                  stiffness: 300,
                  damping: 25,
                  delay: 0.15
                }}
              >
                <PlayingCard
                  card={scenario.heroCards.card1}
                  index={0}
                  isHero
                />
                <PlayingCard
                  card={scenario.heroCards.card2}
                  index={1}
                  isHero
                />
                <div className="hand-type-badge">
                  {scenario.heroCards.handType}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Player positions - outside the felt, positioned relative to poker-table */}
        {ALL_POSITIONS.map((position) => {
          const player = getPlayerAtPosition(position)
          const action = getActionForPosition(position)
          const isRevealed = revealedPositions.has(position)
          const isHero = scenario?.heroPosition === position
          const visualSlot = positionMapping[position] ?? 0
          const visualPos = VISUAL_POSITIONS[visualSlot]

          return (
            <PlayerSeat
              key={position}
              position={position}
              player={player}
              action={action}
              isRevealed={isRevealed}
              isHero={isHero}
              offsetX={visualPos?.x ?? 0}
              offsetY={visualPos?.y ?? 0}
              delay={visualSlot * 0.08}
            />
          )
        })}
      </motion.div>

      <style>{`
        .poker-table-container {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          position: relative;
          min-height: 0;
          overflow: visible;
        }

        .ambient-bg {
          position: absolute;
          inset: 0;
          background:
            radial-gradient(ellipse 80% 50% at 50% 50%, rgba(26, 92, 58, 0.12) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 30% 30%, rgba(212, 165, 116, 0.04) 0%, transparent 50%),
            radial-gradient(ellipse 60% 40% at 70% 70%, rgba(212, 165, 116, 0.03) 0%, transparent 50%);
          pointer-events: none;
        }

        .poker-table {
          position: relative;
          width: 550px;
          height: 340px;
          flex-shrink: 0;
        }

        .table-felt {
          position: absolute;
          inset: 16px;
          background: linear-gradient(145deg, #1a5c3a 0%, #0f4028 50%, #0a3020 100%);
          border-radius: 50%;
          box-shadow:
            0 0 0 10px #3d2a1f,
            0 0 0 14px #261810,
            0 0 0 16px rgba(0, 0, 0, 0.4),
            0 20px 60px rgba(0, 0, 0, 0.5),
            inset 0 0 60px rgba(0, 0, 0, 0.3);
          overflow: visible;
        }

        .felt-texture {
          position: absolute;
          inset: 0;
          border-radius: 50%;
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%' height='100%' filter='url(%23noise)'/%3E%3C/svg%3E");
          opacity: 0.04;
          mix-blend-mode: overlay;
        }

        .felt-highlight {
          position: absolute;
          top: 10%;
          left: 20%;
          right: 20%;
          height: 30%;
          background: radial-gradient(ellipse at center, rgba(255, 255, 255, 0.05) 0%, transparent 70%);
          pointer-events: none;
        }

        .pot-area {
          position: absolute;
          top: 25%;
          left: 50%;
          transform: translate(-50%, -50%);
          z-index: 10;
        }

        .pot-display {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 12px 32px;
          background: rgba(0, 0, 0, 0.85);
          border-radius: 16px;
          border: 2px solid rgba(212, 175, 55, 0.4);
          box-shadow:
            0 4px 20px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        .pot-label {
          font-size: 10px;
          font-weight: 700;
          color: rgba(255, 255, 255, 0.6);
          letter-spacing: 0.15em;
          text-transform: uppercase;
        }

        .pot-amount {
          font-size: 24px;
          font-weight: 800;
          color: var(--accent-primary);
          font-family: var(--font-mono);
          text-shadow: 0 0 20px rgba(212, 175, 55, 0.3);
        }

        .hero-cards-wrapper {
          position: absolute;
          top: 55%;
          left: 50%;
          transform: translate(-50%, -50%);
          z-index: 20;
        }

        .hero-cards {
          display: flex;
          gap: 8px;
        }

        .hand-type-badge {
          position: absolute;
          bottom: -28px;
          left: 50%;
          transform: translateX(-50%);
          padding: 4px 14px;
          background: var(--bg-elevated);
          border: 1px solid var(--border-light);
          border-radius: 16px;
          font-size: 12px;
          font-weight: 600;
          color: var(--text-primary);
          white-space: nowrap;
        }
      `}</style>
    </div>
  )
}

export default PokerTable
