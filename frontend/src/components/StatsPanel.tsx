import React from 'react'
import { motion } from 'framer-motion'
import Icon from '@mdi/react'
import { mdiChartLine, mdiCheck, mdiClose, mdiCardsPlayingOutline } from '@mdi/js'
import { UserStats } from '../types'

interface StatsPanelProps {
  stats: UserStats
}

const StatsPanel: React.FC<StatsPanelProps> = ({ stats }) => {
  const accuracyColor =
    stats.accuracy >= 70 ? '#32d74b' : stats.accuracy >= 50 ? '#ff9f0a' : '#ff453a'

  return (
    <motion.div
      className="stats-panel"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
    >
      {/* Header */}
      <div className="stats-header">
        <Icon path={mdiChartLine} size={0.85} color="var(--accent-primary)" />
        <h2 className="header-title">Session Stats</h2>
      </div>

      {/* Accuracy ring */}
      <div className="accuracy-ring-container">
        <svg className="accuracy-ring" viewBox="0 0 120 120">
          {/* Background circle */}
          <circle
            cx="60"
            cy="60"
            r="50"
            fill="none"
            stroke="var(--bg-elevated)"
            strokeWidth="8"
          />
          {/* Progress circle */}
          <motion.circle
            cx="60"
            cy="60"
            r="50"
            fill="none"
            stroke={accuracyColor}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${2 * Math.PI * 50}`}
            initial={{ strokeDashoffset: 2 * Math.PI * 50 }}
            animate={{
              strokeDashoffset: 2 * Math.PI * 50 * (1 - stats.accuracy / 100),
            }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            transform="rotate(-90 60 60)"
          />
        </svg>
        <div className="accuracy-value">
          <motion.span
            className="accuracy-number"
            key={stats.accuracy}
            initial={{ scale: 1.1, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            style={{ color: accuracyColor }}
          >
            {stats.accuracy.toFixed(0)}
          </motion.span>
          <span className="accuracy-percent">%</span>
        </div>
        <span className="accuracy-label">Accuracy</span>
      </div>

      {/* Stats grid */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon correct">
            <Icon path={mdiCheck} size={0.75} />
          </div>
          <div className="stat-info">
            <span className="stat-value">{stats.correctDecisions}</span>
            <span className="stat-label">Correct</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon total">
            <Icon path={mdiCardsPlayingOutline} size={0.75} />
          </div>
          <div className="stat-info">
            <span className="stat-value">{stats.totalHands}</span>
            <span className="stat-label">Total</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon incorrect">
            <Icon path={mdiClose} size={0.75} />
          </div>
          <div className="stat-info">
            <span className="stat-value">{stats.totalHands - stats.correctDecisions}</span>
            <span className="stat-label">Mistakes</span>
          </div>
        </div>
      </div>

      {/* Progress message */}
      <div className="progress-message">
        {stats.totalHands === 0 ? (
          <p>Start playing to track your progress</p>
        ) : stats.accuracy >= 70 ? (
          <p>Great job! Solid GTO play.</p>
        ) : stats.accuracy >= 50 ? (
          <p>Good progress. Keep practicing!</p>
        ) : (
          <p>Keep studying. GTO takes time.</p>
        )}
      </div>

      <style>{`
        .stats-panel {
          width: 260px;
          background: var(--bg-card);
          border-radius: 16px;
          border: 1px solid var(--border-subtle);
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 20px;
          flex-shrink: 0;
        }

        .stats-header {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .header-title {
          font-size: 14px;
          font-weight: 600;
          color: var(--text-primary);
          margin: 0;
        }

        .accuracy-ring-container {
          position: relative;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 10px 0;
        }

        .accuracy-ring {
          width: 120px;
          height: 120px;
        }

        .accuracy-value {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          display: flex;
          align-items: baseline;
          justify-content: center;
          margin-top: -8px;
        }

        .accuracy-number {
          font-size: 36px;
          font-weight: 700;
          font-family: var(--font-mono);
          line-height: 1;
        }

        .accuracy-percent {
          font-size: 18px;
          font-weight: 600;
          color: var(--text-muted);
          margin-left: 2px;
        }

        .accuracy-label {
          font-size: 11px;
          font-weight: 500;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.1em;
          margin-top: 4px;
        }

        .stats-grid {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .stat-card {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 10px 12px;
          background: var(--bg-tertiary);
          border-radius: 10px;
          border: 1px solid var(--border-subtle);
        }

        .stat-icon {
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 8px;
        }

        .stat-icon.correct {
          background: rgba(50, 215, 75, 0.12);
          color: #32d74b;
        }

        .stat-icon.incorrect {
          background: rgba(255, 69, 58, 0.12);
          color: #ff453a;
        }

        .stat-icon.total {
          background: rgba(212, 165, 116, 0.12);
          color: var(--accent-primary);
        }

        .stat-info {
          display: flex;
          flex-direction: column;
        }

        .stat-value {
          font-size: 18px;
          font-weight: 700;
          font-family: var(--font-mono);
          color: var(--text-primary);
        }

        .stat-label {
          font-size: 10px;
          color: var(--text-muted);
        }

        .progress-message {
          padding: 12px;
          background: var(--bg-tertiary);
          border-radius: 10px;
          text-align: center;
        }

        .progress-message p {
          font-size: 12px;
          color: var(--text-secondary);
          margin: 0;
          line-height: 1.4;
        }
      `}</style>
    </motion.div>
  )
}

export default StatsPanel
