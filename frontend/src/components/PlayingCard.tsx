import React from 'react'
import { motion } from 'framer-motion'
import { Card } from '../types'

interface PlayingCardProps {
  card: Card
  index: number
  isHero?: boolean
  faceDown?: boolean
  small?: boolean
}

const SUIT_SYMBOLS: Record<string, string> = {
  spades: '\u2660',
  hearts: '\u2665',
  diamonds: '\u2666',
  clubs: '\u2663',
}

const SUIT_COLORS: Record<string, string> = {
  spades: '#1c1c1e',
  hearts: '#ff453a',
  diamonds: '#0a84ff',
  clubs: '#32d74b',
}

const PlayingCard: React.FC<PlayingCardProps> = ({
  card,
  index,
  isHero = false,
  faceDown = false,
  small = false,
}) => {
  const suitSymbol = SUIT_SYMBOLS[card.suit]
  const suitColor = SUIT_COLORS[card.suit]

  const cardVariants = {
    initial: {
      rotateY: 180,
      scale: 0.85,
      opacity: 0,
    },
    animate: {
      rotateY: 0,
      scale: 1,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 200,
        damping: 20,
        delay: index * 0.12,
      },
    },
    hover: isHero
      ? {
          y: -6,
          scale: 1.03,
        }
      : {},
  }

  return (
    <motion.div
      className={`playing-card ${isHero ? 'hero' : ''} ${small ? 'small' : ''}`}
      variants={cardVariants}
      initial="initial"
      animate="animate"
      whileHover="hover"
      style={{
        '--suit-color': suitColor,
      } as React.CSSProperties}
    >
      {faceDown ? (
        <div className="card-back">
          <div className="back-pattern" />
        </div>
      ) : (
        <div className="card-front">
          <div className="card-corner top-left">
            <span className="rank">{card.rank}</span>
            <span className="suit">{suitSymbol}</span>
          </div>
          <div className="card-center">
            <span className="suit-large">{suitSymbol}</span>
          </div>
          <div className="card-corner bottom-right">
            <span className="rank">{card.rank}</span>
            <span className="suit">{suitSymbol}</span>
          </div>
        </div>
      )}

      <style>{`
        .playing-card {
          width: 64px;
          height: 92px;
          border-radius: 8px;
          background: #ffffff;
          box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.25),
            0 1px 3px rgba(0, 0, 0, 0.15);
          position: relative;
          cursor: default;
          transform-style: preserve-3d;
        }

        .playing-card.hero {
          width: 72px;
          height: 104px;
          cursor: pointer;
        }

        .playing-card.small {
          width: 38px;
          height: 54px;
        }

        .card-front {
          position: absolute;
          inset: 0;
          border-radius: 8px;
          background: #ffffff;
          padding: 5px;
        }

        .card-corner {
          display: flex;
          flex-direction: column;
          align-items: center;
          line-height: 1;
        }

        .card-corner.top-left {
          position: absolute;
          top: 5px;
          left: 5px;
        }

        .card-corner.bottom-right {
          position: absolute;
          bottom: 5px;
          right: 5px;
          transform: rotate(180deg);
        }

        .rank {
          font-size: 15px;
          font-weight: 700;
          color: var(--suit-color);
          font-family: var(--font-primary);
        }

        .playing-card.hero .rank {
          font-size: 17px;
        }

        .playing-card.small .rank {
          font-size: 10px;
        }

        .suit {
          font-size: 13px;
          color: var(--suit-color);
          line-height: 1;
        }

        .playing-card.hero .suit {
          font-size: 15px;
        }

        .playing-card.small .suit {
          font-size: 9px;
        }

        .card-center {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
        }

        .suit-large {
          font-size: 28px;
          color: var(--suit-color);
          opacity: 0.85;
        }

        .playing-card.hero .suit-large {
          font-size: 34px;
        }

        .playing-card.small .suit-large {
          font-size: 16px;
        }

        .card-back {
          position: absolute;
          inset: 0;
          border-radius: 8px;
          background: linear-gradient(145deg, #2c3e50 0%, #1a252f 100%);
          overflow: hidden;
        }

        .back-pattern {
          position: absolute;
          inset: 3px;
          border: 1px solid rgba(255, 255, 255, 0.15);
          border-radius: 5px;
          background-image: repeating-linear-gradient(
            45deg,
            transparent,
            transparent 3px,
            rgba(255, 255, 255, 0.03) 3px,
            rgba(255, 255, 255, 0.03) 6px
          );
        }
      `}</style>
    </motion.div>
  )
}

export default PlayingCard
