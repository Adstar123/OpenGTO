import React from 'react'
import { motion } from 'framer-motion'
import Icon from '@mdi/react'
import { mdiCheck, mdiClose, mdiArrowRight } from '@mdi/js'
import { TrainerResult, Scenario } from '../types'

interface ResultModalProps {
  result: TrainerResult
  scenario: Scenario | null
  onNext: () => void
  onClose: () => void
}

const ACTION_LABELS: Record<string, string> = {
  fold: 'Fold',
  check: 'Check',
  call: 'Call',
  raise: 'Raise',
  allIn: 'All-In',
}

const ACTION_COLORS: Record<string, string> = {
  fold: '#ff453a',
  check: '#32d74b',
  call: '#0a84ff',
  raise: '#ff9f0a',
  allIn: '#ff375f',
}

const ResultModal: React.FC<ResultModalProps> = ({
  result,
  scenario: _scenario,
  onNext,
  onClose,
}) => {
  const strategyEntries = Object.entries(result.gtoStrategy)
    .filter(([_, value]) => value > 0.01)
    .sort((a, b) => b[1] - a[1])

  const maxValue = Math.max(...strategyEntries.map(([_, v]) => v))

  return (
    <motion.div
      className="modal-overlay"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.div
        className="modal-content"
        initial={{ opacity: 0, scale: 0.95, y: 16 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.98, y: 8 }}
        transition={{
          type: 'spring',
          stiffness: 300,
          damping: 25,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Result header */}
        <div className={`result-header ${result.isCorrect ? 'correct' : 'incorrect'}`}>
          <motion.div
            className="result-icon"
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{
              type: 'spring',
              stiffness: 400,
              damping: 15,
              delay: 0.1,
            }}
          >
            <Icon
              path={result.isCorrect ? mdiCheck : mdiClose}
              size={1.25}
              color="white"
            />
          </motion.div>
          <motion.h2
            className="result-title"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
          >
            {result.isCorrect ? 'Correct!' : 'Incorrect'}
          </motion.h2>
          <motion.p
            className="result-subtitle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.25 }}
          >
            {result.feedback}
          </motion.p>
        </div>

        {/* GTO Strategy breakdown */}
        <div className="strategy-section">
          <h3 className="section-title">GTO Strategy</h3>
          <div className="strategy-bars">
            {strategyEntries.map(([action, value], index) => (
              <motion.div
                key={action}
                className="strategy-bar-row"
                initial={{ opacity: 0, x: -16 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.25 + index * 0.08 }}
              >
                <span
                  className="bar-label"
                  style={{ color: ACTION_COLORS[action] }}
                >
                  {ACTION_LABELS[action]}
                </span>
                <div className="bar-container">
                  <motion.div
                    className="bar-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${(value / maxValue) * 100}%` }}
                    transition={{
                      delay: 0.35 + index * 0.08,
                      duration: 0.5,
                      ease: [0.22, 1, 0.36, 1],
                    }}
                    style={{ background: ACTION_COLORS[action] }}
                  />
                </div>
                <span className="bar-value">{(value * 100).toFixed(0)}%</span>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Your action vs recommended */}
        <div className="comparison-section">
          <div className="comparison-item">
            <span className="comparison-label">Your Action</span>
            <span
              className="comparison-value"
              style={{
                color: ACTION_COLORS[result.userAction] || 'var(--text-primary)',
              }}
            >
              {ACTION_LABELS[result.userAction] || result.userAction}
            </span>
          </div>
          <div className="comparison-divider">
            <Icon path={mdiArrowRight} size={0.9} color="var(--text-muted)" />
          </div>
          <div className="comparison-item">
            <span className="comparison-label">GTO Recommends</span>
            <span
              className="comparison-value"
              style={{
                color: ACTION_COLORS[strategyEntries[0]?.[0]] || 'var(--text-primary)',
              }}
            >
              {ACTION_LABELS[strategyEntries[0]?.[0]] || 'N/A'}
            </span>
          </div>
        </div>

        {/* Next button */}
        <motion.button
          className="next-btn"
          onClick={onNext}
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <span>Next Hand</span>
          <Icon path={mdiArrowRight} size={0.8} />
        </motion.button>
      </motion.div>

      <style>{`
        .modal-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.75);
          backdrop-filter: blur(8px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: 20px;
        }

        .modal-content {
          width: 100%;
          max-width: 420px;
          background: var(--bg-card);
          border-radius: 20px;
          border: 1px solid var(--border-subtle);
          overflow: hidden;
          box-shadow: var(--shadow-lg);
        }

        .result-header {
          padding: 28px;
          text-align: center;
          position: relative;
        }

        .result-header.correct {
          background: linear-gradient(180deg, rgba(50, 215, 75, 0.1) 0%, transparent 100%);
        }

        .result-header.incorrect {
          background: linear-gradient(180deg, rgba(255, 69, 58, 0.1) 0%, transparent 100%);
        }

        .result-icon {
          width: 56px;
          height: 56px;
          margin: 0 auto 14px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .result-header.correct .result-icon {
          background: linear-gradient(135deg, #32d74b 0%, #28a745 100%);
        }

        .result-header.incorrect .result-icon {
          background: linear-gradient(135deg, #ff453a 0%, #dc3545 100%);
        }

        .result-title {
          font-size: 24px;
          font-weight: 700;
          margin: 0 0 6px;
        }

        .result-header.correct .result-title {
          color: #32d74b;
        }

        .result-header.incorrect .result-title {
          color: #ff453a;
        }

        .result-subtitle {
          font-size: 13px;
          color: var(--text-secondary);
          margin: 0;
          max-width: 280px;
          margin: 0 auto;
        }

        .strategy-section {
          padding: 20px 28px;
          border-top: 1px solid var(--border-subtle);
        }

        .section-title {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin: 0 0 14px;
        }

        .strategy-bars {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .strategy-bar-row {
          display: grid;
          grid-template-columns: 60px 1fr 50px;
          align-items: center;
          gap: 12px;
        }

        .bar-label {
          font-size: 12px;
          font-weight: 600;
        }

        .bar-container {
          height: 20px;
          background: var(--bg-tertiary);
          border-radius: 10px;
          overflow: hidden;
        }

        .bar-fill {
          height: 100%;
          border-radius: 10px;
        }

        .bar-value {
          font-size: 13px;
          font-weight: 600;
          font-family: var(--font-mono);
          color: var(--text-primary);
          text-align: right;
        }

        .comparison-section {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 18px;
          padding: 20px 28px;
          border-top: 1px solid var(--border-subtle);
        }

        .comparison-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 3px;
        }

        .comparison-label {
          font-size: 10px;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .comparison-value {
          font-size: 16px;
          font-weight: 700;
        }

        .comparison-divider {
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .next-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          width: calc(100% - 56px);
          margin: 0 28px 28px;
          padding: 14px 20px;
          background: var(--gradient-primary);
          border: none;
          border-radius: 14px;
          font-size: 15px;
          font-weight: 600;
          color: #1a1a1a;
          cursor: pointer;
          transition: all var(--transition-base);
        }

        .next-btn:hover {
          opacity: 0.9;
        }
      `}</style>
    </motion.div>
  )
}

export default ResultModal
