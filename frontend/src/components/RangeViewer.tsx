import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Icon from '@mdi/react'
import {
  mdiCardsPlayingOutline,
  mdiChartBar,
  mdiRefresh,
  mdiArrowLeft,
} from '@mdi/js'
import RangeMatrix from './RangeMatrix'
import { Position } from '../types'

type PositionAction = 'fold' | 'call' | 'raise' | 'all-in'

interface PositionConfig {
  position: Position
  action: PositionAction
  raiseAmount?: number
}

interface HandStrategy {
  fold?: number
  check?: number
  call?: number
  raise?: number
  allIn?: number
}

interface RangeData {
  heroPosition: string
  rangeData: Record<string, HandStrategy>
  legalActions: string[]
  actionDistribution: Record<string, number>
  handCount: number
}

const API_BASE = 'http://localhost:5000'
const ALL_POSITIONS: Position[] = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']

// Visual positions for mini table (percentage based)
const TABLE_POSITIONS: Record<Position, { x: number; y: number }> = {
  UTG: { x: 15, y: 65 },
  HJ: { x: 15, y: 35 },
  CO: { x: 50, y: 15 },
  BTN: { x: 85, y: 35 },
  SB: { x: 85, y: 65 },
  BB: { x: 50, y: 85 },
}

const ACTION_COLORS: Record<string, string> = {
  fold: '#3b82f6',
  call: '#22c55e',
  raise: '#f97316',
  'all-in': '#ef4444',
  allIn: '#ef4444',
  check: '#22c55e',
}

type SidePanelView = 'overview' | 'hand'

const RangeViewer: React.FC = () => {
  const [heroPosition, setHeroPosition] = useState<Position>('CO')
  const [positionConfigs, setPositionConfigs] = useState<PositionConfig[]>([])
  const [rangeData, setRangeData] = useState<Record<string, HandStrategy>>({})
  const [actionDistribution, setActionDistribution] = useState<Record<string, number>>({})
  const [selectedHand, setSelectedHand] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pot, setPot] = useState(1.5)
  const [sidePanelView, setSidePanelView] = useState<SidePanelView>('overview')

  // Get positions that can act (before hero, excluding blinds)
  const activePositions = useMemo(() => {
    const heroIndex = ALL_POSITIONS.indexOf(heroPosition)
    return ALL_POSITIONS.filter((pos, idx) => idx < heroIndex && pos !== 'SB' && pos !== 'BB')
  }, [heroPosition])

  // Initialize position configs when hero position changes
  useEffect(() => {
    // When hero position changes, set all active positions to fold by default
    const newConfigs: PositionConfig[] = activePositions.map(pos => ({
      position: pos,
      action: 'fold' as PositionAction,
    }))
    setPositionConfigs(newConfigs)
  }, [activePositions])

  // Determine legal actions for a position based on game state
  // In preflop, there's always a bet (the big blind), so 'check' is never legal
  // All positions can: fold, call (or limp if no raise), raise, or go all-in
  const getLegalActionsForPosition = useCallback((_pos: Position): PositionAction[] => {
    // In preflop, all four actions are always legal:
    // - fold: always available
    // - call: matches current bet (or limp if just the BB)
    // - raise: increase the bet
    // - all-in: commit entire stack
    return ['fold', 'call', 'raise', 'all-in']
  }, [])

  // Fetch range data from API
  const fetchRangeData = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      const actions = positionConfigs.map(p => ({
        position: p.position,
        action: p.action,
        amount: p.raiseAmount,
      }))

      const response = await fetch(`${API_BASE}/range`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          heroPosition,
          actions,
          stack: 100,
        }),
      })

      if (!response.ok) throw new Error('Failed to fetch range data')

      const data: RangeData = await response.json()
      setRangeData(data.rangeData)
      setActionDistribution(data.actionDistribution)

      // Calculate pot and current bet
      let newPot = 1.5 // SB + BB
      let newBet = 1.0 // BB
      for (const action of actions) {
        if (action.action === 'raise' && action.amount) {
          newPot += action.amount
          newBet = action.amount
        } else if (action.action === 'all-in') {
          newPot += 100
          newBet = 100
        } else if (action.action === 'call') {
          newPot += newBet
        }
      }
      setPot(newPot)
    } catch (err) {
      console.error('Error fetching range:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }, [heroPosition, positionConfigs])

  useEffect(() => {
    if (positionConfigs.length > 0 || activePositions.length === 0) {
      fetchRangeData()
    }
  }, [fetchRangeData, positionConfigs.length, activePositions.length])

  // Handle hand selection - switch to hand view
  const handleHandSelect = (hand: string) => {
    setSelectedHand(hand)
    setSidePanelView('hand')
  }

  // Position action handling - always requires a valid action (no deselection)
  const handlePositionAction = (pos: Position, action: PositionAction) => {
    if (pos === heroPosition) return

    setPositionConfigs(prev => {
      return prev.map(p => {
        if (p.position === pos) {
          return {
            ...p,
            action,
            raiseAmount: action === 'raise' ? 2.5 : undefined,
          }
        }
        return p
      })
    })
  }

  const getPositionAction = (pos: Position): PositionAction => {
    return positionConfigs.find(p => p.position === pos)?.action || 'fold'
  }

  const selectedHandStrategy = selectedHand ? rangeData[selectedHand] : null

  return (
    <div className="range-viewer">
      {/* Top Action Bar - Full Width */}
      <div className="action-bar">
        <div className="action-bar-content">
          {/* Hero Position Selector */}
          <div className="hero-section">
            <span className="section-label">Your Position</span>
            <div className="hero-buttons">
              {ALL_POSITIONS.map(pos => (
                <motion.button
                  key={pos}
                  className={`hero-btn ${heroPosition === pos ? 'active' : ''}`}
                  onClick={() => setHeroPosition(pos)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {pos}
                </motion.button>
              ))}
            </div>
          </div>

          <div className="divider" />

          {/* Opponent Actions */}
          <div className="positions-section">
            <span className="section-label">Opponent Actions</span>
            <div className="position-actions">
              {activePositions.length === 0 ? (
                <span className="no-opponents">No opponents act before you in this position</span>
              ) : (
                activePositions.map(pos => {
                  const currentAction = getPositionAction(pos)
                  const legalActions = getLegalActionsForPosition(pos)

                  return (
                    <div key={pos} className="position-group">
                      <span className="position-label">{pos}</span>
                      <div className="action-buttons">
                        {(['fold', 'call', 'raise', 'all-in'] as PositionAction[]).map(action => {
                          const isLegal = legalActions.includes(action)
                          const isActive = currentAction === action

                          return (
                            <motion.button
                              key={action}
                              className={`action-btn ${isActive ? 'active' : ''} ${!isLegal ? 'disabled' : ''}`}
                              style={{
                                '--action-color': ACTION_COLORS[action],
                              } as React.CSSProperties}
                              onClick={() => isLegal && handlePositionAction(pos, action)}
                              disabled={!isLegal}
                              whileHover={isLegal ? { scale: 1.05 } : {}}
                              whileTap={isLegal ? { scale: 0.95 } : {}}
                              title={!isLegal ? 'Not a legal action in this situation' : undefined}
                            >
                              {action === 'all-in' ? 'A' : action.charAt(0).toUpperCase()}
                            </motion.button>
                          )
                        })}
                      </div>
                    </div>
                  )
                })
              )}
            </div>
          </div>

          {/* Refresh Button */}
          <motion.button
            className="refresh-btn"
            onClick={fetchRangeData}
            disabled={isLoading}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Icon path={mdiRefresh} size={0.8} className={isLoading ? 'spinning' : ''} />
          </motion.button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="main-area">
        {/* Range Matrix - Left Side */}
        <div className="matrix-container">
          {error && (
            <div className="error-banner">
              {error}
              <button onClick={fetchRangeData}>Retry</button>
            </div>
          )}

          <RangeMatrix
            rangeData={rangeData}
            selectedHand={selectedHand}
            onHandSelect={handleHandSelect}
          />
        </div>

        {/* Side Panel - Right Side */}
        <div className="side-panel">
          <AnimatePresence mode="wait">
            {sidePanelView === 'hand' && selectedHand ? (
              <motion.div
                key="hand-view"
                className="panel-content"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
              >
                {/* Back Button */}
                <motion.button
                  className="back-btn"
                  onClick={() => setSidePanelView('overview')}
                  whileHover={{ x: -2 }}
                >
                  <Icon path={mdiArrowLeft} size={0.7} />
                  <span>Back to Overview</span>
                </motion.button>

                {/* Hand Details */}
                <div className="hand-header">
                  <Icon path={mdiCardsPlayingOutline} size={1} />
                  <h2>{selectedHand}</h2>
                </div>

                <div className="strategy-details">
                  <h3>GTO Strategy</h3>
                  <div className="strategy-bars">
                    {selectedHandStrategy && Object.entries(selectedHandStrategy)
                      .filter(([_, v]) => v && v > 0.005)
                      .sort((a, b) => (b[1] || 0) - (a[1] || 0))
                      .map(([action, prob]) => (
                        <div key={action} className="strategy-row">
                          <span className="action-label" style={{ color: ACTION_COLORS[action] }}>
                            {action === 'allIn' ? 'All-In' : action.charAt(0).toUpperCase() + action.slice(1)}
                          </span>
                          <div className="bar-track">
                            <motion.div
                              className="bar-fill"
                              style={{ background: ACTION_COLORS[action] }}
                              initial={{ width: 0 }}
                              animate={{ width: `${(prob || 0) * 100}%` }}
                              transition={{ duration: 0.4, ease: 'easeOut' }}
                            />
                          </div>
                          <span className="prob-value">{((prob || 0) * 100).toFixed(1)}%</span>
                        </div>
                      ))
                    }
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="overview"
                className="panel-content"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
              >
                {/* Mini Table */}
                <div className="mini-table-section">
                  <h3>Table View</h3>
                  <div className="mini-table">
                    <div className="table-felt">
                      <div className="pot-display">
                        <span className="pot-label">POT</span>
                        <span className="pot-amount">{pot.toFixed(1)}</span>
                        <span className="pot-unit">BB</span>
                      </div>
                    </div>
                    {ALL_POSITIONS.map(pos => {
                      const coords = TABLE_POSITIONS[pos]
                      const action = getPositionAction(pos)
                      const isHero = pos === heroPosition
                      const isActiveOpponent = activePositions.includes(pos)

                      return (
                        <div
                          key={pos}
                          className={`table-seat ${isHero ? 'hero' : ''}`}
                          style={{
                            left: `${coords.x}%`,
                            top: `${coords.y}%`,
                          }}
                        >
                          <div className="seat-label">{pos}</div>
                          {isHero ? (
                            <div className="hero-marker">?</div>
                          ) : isActiveOpponent ? (
                            <div
                              className="action-marker"
                              style={{ background: ACTION_COLORS[action] }}
                            >
                              {action === 'all-in' ? 'A' : action.charAt(0).toUpperCase()}
                            </div>
                          ) : null}
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Action Distribution */}
                <div className="distribution-section">
                  <h3>
                    <Icon path={mdiChartBar} size={0.7} />
                    Action Distribution
                  </h3>
                  <div className="distribution-grid">
                    {Object.entries(actionDistribution)
                      .filter(([_, v]) => v > 0.001)
                      .sort((a, b) => b[1] - a[1])
                      .map(([action, prob]) => (
                        <div
                          key={action}
                          className="dist-card"
                          style={{ '--card-color': ACTION_COLORS[action] } as React.CSSProperties}
                        >
                          <span className="dist-label">
                            {action === 'allIn' ? 'All-In' : action.charAt(0).toUpperCase() + action.slice(1)}
                          </span>
                          <span className="dist-value">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      ))
                    }
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <style>{`
        .range-viewer {
          flex: 1;
          display: flex;
          flex-direction: column;
          min-height: 0;
          overflow: hidden;
        }

        /* Action Bar */
        .action-bar {
          background: var(--bg-card);
          border-bottom: 1px solid var(--border-subtle);
          padding: 12px 20px;
          flex-shrink: 0;
        }

        .action-bar-content {
          display: flex;
          align-items: center;
          gap: 20px;
        }

        .section-label {
          font-size: 10px;
          font-weight: 600;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-bottom: 8px;
          display: block;
        }

        .hero-section {
          flex-shrink: 0;
        }

        .hero-buttons {
          display: flex;
          gap: 4px;
        }

        .hero-btn {
          padding: 8px 14px;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-subtle);
          border-radius: 6px;
          font-size: 12px;
          font-weight: 600;
          color: var(--text-secondary);
          cursor: pointer;
          transition: all 0.15s ease;
        }

        .hero-btn.active {
          background: linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(212, 175, 55, 0.1));
          border-color: var(--accent-primary);
          color: var(--accent-primary);
        }

        .divider {
          width: 1px;
          height: 50px;
          background: var(--border-subtle);
        }

        .positions-section {
          flex: 1;
          min-width: 0;
        }

        .position-actions {
          display: flex;
          gap: 16px;
          flex-wrap: wrap;
          align-items: center;
        }

        .no-opponents {
          font-size: 12px;
          color: var(--text-muted);
          font-style: italic;
        }

        .position-group {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .position-label {
          font-size: 11px;
          font-weight: 600;
          color: var(--text-secondary);
          width: 28px;
        }

        .action-buttons {
          display: flex;
          gap: 3px;
        }

        .action-btn {
          width: 28px;
          height: 28px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-subtle);
          border-radius: 4px;
          font-size: 11px;
          font-weight: 700;
          color: var(--text-muted);
          cursor: pointer;
          transition: all 0.15s ease;
        }

        .action-btn:not(.disabled):hover {
          border-color: var(--action-color);
          color: var(--action-color);
        }

        .action-btn.active {
          background: var(--action-color);
          border-color: var(--action-color);
          color: white;
        }

        .action-btn.disabled {
          opacity: 0.3;
          cursor: not-allowed;
        }

        .refresh-btn {
          width: 40px;
          height: 40px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-subtle);
          border-radius: 8px;
          color: var(--text-secondary);
          cursor: pointer;
          transition: all 0.15s ease;
          flex-shrink: 0;
        }

        .refresh-btn:hover {
          border-color: var(--accent-primary);
          color: var(--accent-primary);
        }

        .refresh-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .spinning {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        /* Main Area */
        .main-area {
          flex: 1;
          display: flex;
          gap: 20px;
          padding: 20px;
          min-height: 0;
          overflow: hidden;
        }

        .matrix-container {
          flex: 1;
          display: flex;
          flex-direction: column;
          min-width: 0;
          min-height: 0;
          overflow: auto;
        }

        .error-banner {
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          padding: 12px 16px;
          margin-bottom: 16px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-size: 13px;
          color: #ef4444;
        }

        .error-banner button {
          background: rgba(239, 68, 68, 0.2);
          border: none;
          border-radius: 4px;
          padding: 4px 12px;
          font-size: 12px;
          color: #ef4444;
          cursor: pointer;
        }

        /* Side Panel */
        .side-panel {
          width: 320px;
          flex-shrink: 0;
          display: flex;
          flex-direction: column;
          min-height: 0;
          overflow: hidden;
        }

        .panel-content {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 16px;
          overflow-y: auto;
        }

        .back-btn {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 12px;
          background: var(--bg-tertiary);
          border: 1px solid var(--border-subtle);
          border-radius: 8px;
          font-size: 12px;
          color: var(--text-secondary);
          cursor: pointer;
          transition: all 0.15s ease;
          align-self: flex-start;
        }

        .back-btn:hover {
          color: var(--text-primary);
          border-color: var(--border-light);
        }

        .hand-header {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 16px;
          background: var(--bg-card);
          border-radius: 12px;
          border: 1px solid var(--border-subtle);
          color: var(--accent-primary);
        }

        .hand-header h2 {
          font-size: 24px;
          font-weight: 700;
          color: var(--text-primary);
          margin: 0;
        }

        .strategy-details {
          background: var(--bg-card);
          border-radius: 12px;
          border: 1px solid var(--border-subtle);
          padding: 16px;
          flex: 1;
        }

        .strategy-details h3 {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin: 0 0 16px 0;
        }

        .strategy-bars {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .strategy-row {
          display: grid;
          grid-template-columns: 60px 1fr 50px;
          align-items: center;
          gap: 12px;
        }

        .action-label {
          font-size: 12px;
          font-weight: 600;
        }

        .bar-track {
          height: 24px;
          background: var(--bg-tertiary);
          border-radius: 12px;
          overflow: hidden;
        }

        .bar-fill {
          height: 100%;
          border-radius: 12px;
        }

        .prob-value {
          font-size: 13px;
          font-weight: 600;
          color: var(--text-primary);
          text-align: right;
          font-family: var(--font-mono);
        }

        /* Mini Table */
        .mini-table-section {
          background: var(--bg-card);
          border-radius: 12px;
          border: 1px solid var(--border-subtle);
          padding: 16px;
        }

        .mini-table-section h3 {
          font-size: 12px;
          font-weight: 600;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin: 0 0 12px 0;
        }

        .mini-table {
          position: relative;
          width: 100%;
          aspect-ratio: 1.4;
        }

        .table-felt {
          position: absolute;
          inset: 20%;
          background: linear-gradient(145deg, #1a5c3a 0%, #0f4028 100%);
          border-radius: 50%;
          border: 4px solid #5d4a3a;
          box-shadow:
            0 0 0 2px #3d2a1f,
            inset 0 0 30px rgba(0, 0, 0, 0.5),
            0 4px 16px rgba(0, 0, 0, 0.3);
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .pot-display {
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 10px 20px;
          background: rgba(0, 0, 0, 0.85);
          border-radius: 12px;
          border: 2px solid rgba(212, 175, 55, 0.5);
          box-shadow:
            0 4px 16px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        .pot-label {
          font-size: 9px;
          font-weight: 700;
          color: rgba(255, 255, 255, 0.7);
          text-transform: uppercase;
          letter-spacing: 0.12em;
        }

        .pot-amount {
          font-size: 22px;
          font-weight: 800;
          color: var(--accent-primary);
          font-family: var(--font-mono);
          line-height: 1.1;
          text-shadow: 0 0 15px rgba(212, 175, 55, 0.3);
        }

        .pot-unit {
          font-size: 11px;
          font-weight: 700;
          color: var(--accent-primary);
          opacity: 0.9;
        }

        .table-seat {
          position: absolute;
          transform: translate(-50%, -50%);
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 4px;
        }

        .seat-label {
          font-size: 9px;
          font-weight: 600;
          color: var(--text-muted);
          background: var(--bg-tertiary);
          padding: 2px 6px;
          border-radius: 4px;
        }

        .table-seat.hero .seat-label {
          color: var(--accent-primary);
          background: rgba(212, 175, 55, 0.2);
        }

        .hero-marker {
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: var(--gradient-primary);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          font-weight: 700;
          color: #1a1a1a;
        }

        .action-marker {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 10px;
          font-weight: 700;
          color: white;
        }

        /* Distribution Section */
        .distribution-section {
          background: var(--bg-card);
          border-radius: 12px;
          border: 1px solid var(--border-subtle);
          padding: 16px;
          flex: 1;
        }

        .distribution-section h3 {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          font-weight: 600;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin: 0 0 12px 0;
        }

        .distribution-grid {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .dist-card {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 16px;
          background: var(--bg-tertiary);
          border-radius: 8px;
          border-left: 3px solid var(--card-color);
        }

        .dist-label {
          font-size: 13px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .dist-value {
          font-size: 16px;
          font-weight: 700;
          color: var(--card-color);
          font-family: var(--font-mono);
        }
      `}</style>
    </div>
  )
}

export default RangeViewer
