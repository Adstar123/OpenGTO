import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Position, Player, PlayerAction } from '../types'
import Icon from '@mdi/react'
import { mdiAccountCircle } from '@mdi/js'

interface PlayerSeatProps {
  position: Position
  player?: Player
  action?: PlayerAction
  isRevealed: boolean
  isHero: boolean
  offsetX: number  // percentage offset from center
  offsetY: number  // percentage offset from center
  delay: number
}

const ACTION_COLORS: Record<string, string> = {
  fold: 'var(--color-fold)',
  check: 'var(--color-check)',
  call: 'var(--color-call)',
  raise: 'var(--color-raise)',
  'all-in': 'var(--color-allin)',
}

const ACTION_LABELS: Record<string, string> = {
  fold: 'Fold',
  check: 'Check',
  call: 'Call',
  raise: 'Raise',
  'all-in': 'All-In',
}

const PlayerSeat: React.FC<PlayerSeatProps> = ({
  position,
  player,
  action,
  isRevealed,
  isHero,
  offsetX,
  offsetY,
  delay,
}) => {
  const actionColor = action ? ACTION_COLORS[action.action] : undefined

  return (
    <div
      className="player-seat-wrapper"
      style={{
        left: `calc(50% + ${offsetX}%)`,
        top: `calc(50% + ${offsetY}%)`,
      }}
    >
      <motion.div
        className={`player-seat ${isHero ? 'hero' : ''} ${action?.action === 'fold' ? 'folded' : ''}`}
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{
          type: 'spring',
          stiffness: 200,
          damping: 20,
          delay: delay,
        }}
      >
        <div className="seat-content">
        {/* Position badge */}
        <div className={`position-badge ${isHero ? 'hero' : ''}`}>
          <Icon path={mdiAccountCircle} size={0.6} color={isHero ? 'var(--accent-primary)' : 'var(--text-muted)'} />
          <span className="position-label">{position}</span>
        </div>

        {/* Action display */}
        <AnimatePresence mode="wait">
          {isRevealed && action && (
            <motion.div
              className="action-display"
              initial={{ opacity: 0, y: 6, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -6, scale: 0.95 }}
              transition={{
                type: 'spring',
                stiffness: 400,
                damping: 25,
              }}
              style={{
                '--action-color': actionColor,
              } as React.CSSProperties}
            >
              <span className="action-text">{ACTION_LABELS[action.action]}</span>
              {action.amount && action.action !== 'fold' && (
                <span className="action-amount">{action.amount.toFixed(1)}bb</span>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Hero indicator */}
        {isHero && (
          <motion.div
            className="hero-badge"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{
              type: 'spring',
              stiffness: 400,
              damping: 15,
              delay: delay + 0.15,
            }}
          >
            YOU
          </motion.div>
        )}
      </div>

      <style>{`
        .player-seat-wrapper {
          position: absolute;
          transform: translate(-50%, -50%);
          z-index: 30;
        }

        .player-seat-wrapper:has(.hero) {
          z-index: 40;
        }

        .player-seat {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
        }

        .player-seat.folded {
          opacity: 0.4;
        }

        .seat-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
        }

        .position-badge {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 12px;
          background: rgba(0, 0, 0, 0.75);
          border-radius: 18px;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .position-badge.hero {
          background: rgba(30, 25, 20, 0.9);
          border-color: rgba(212, 165, 116, 0.5);
        }

        .position-label {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-primary);
          letter-spacing: 0.02em;
        }

        .action-display {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 5px 12px;
          background: rgba(0, 0, 0, 0.85);
          border-radius: 8px;
          border: 1px solid var(--action-color);
          min-width: 55px;
        }

        .action-text {
          font-size: 10px;
          font-weight: 700;
          color: var(--action-color);
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }

        .action-amount {
          font-size: 11px;
          font-weight: 700;
          color: var(--text-primary);
          font-family: var(--font-mono);
        }

        .hero-badge {
          padding: 3px 10px;
          background: var(--gradient-primary);
          border-radius: 8px;
          font-size: 9px;
          font-weight: 700;
          color: #1a1a1a;
          letter-spacing: 0.06em;
        }
      `}</style>
      </motion.div>
    </div>
  )
}

export default PlayerSeat
