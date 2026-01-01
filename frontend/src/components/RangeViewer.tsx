import React, { useState, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import RangeMatrix from './RangeMatrix'
import PositionActionSelector from './PositionActionSelector'
import MiniPokerTable from './MiniPokerTable'
import ActionDistribution from './ActionDistribution'
import { Position } from '../types'

type PositionAction = 'none' | 'fold' | 'call' | 'raise' | 'all-in'

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

const RangeViewer: React.FC = () => {
  const [heroPosition, setHeroPosition] = useState<Position>('CO')
  const [positionConfigs, setPositionConfigs] = useState<PositionConfig[]>([])
  const [rangeData, setRangeData] = useState<Record<string, HandStrategy>>({})
  const [actionDistribution, setActionDistribution] = useState<Record<string, number>>({})
  const [legalActions, setLegalActions] = useState<string[]>([])
  const [selectedHand, setSelectedHand] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pot, setPot] = useState(1.5)
  const [currentBet, setCurrentBet] = useState(1.0)

  // Fetch range data from API
  const fetchRangeData = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      // Build actions array from position configs
      const actions = positionConfigs
        .filter(p => p.action !== 'none')
        .map(p => ({
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

      if (!response.ok) {
        throw new Error('Failed to fetch range data')
      }

      const data: RangeData = await response.json()
      setRangeData(data.rangeData)
      setActionDistribution(data.actionDistribution)
      setLegalActions(data.legalActions)

      // Calculate pot based on actions
      let newPot = 1.5 // blinds
      let newBet = 1.0
      for (const action of actions) {
        if (action.action === 'raise' && action.amount) {
          newPot += action.amount
          newBet = action.amount
        } else if (action.action === 'call') {
          newPot += newBet
        }
      }
      setPot(newPot)
      setCurrentBet(newBet)

    } catch (err) {
      console.error('Error fetching range:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }, [heroPosition, positionConfigs])

  // Fetch on mount and when config changes
  useEffect(() => {
    fetchRangeData()
  }, [fetchRangeData])

  // Handle position action change
  const handlePositionChange = (position: Position, action: PositionAction, amount?: number) => {
    setPositionConfigs(prev => {
      const existing = prev.find(p => p.position === position)
      if (existing) {
        if (action === 'none') {
          return prev.filter(p => p.position !== position)
        }
        return prev.map(p =>
          p.position === position
            ? { ...p, action, raiseAmount: amount }
            : p
        )
      }
      if (action !== 'none') {
        return [...prev, { position, action, raiseAmount: amount }]
      }
      return prev
    })
  }

  // Handle hero position change
  const handleHeroPositionChange = (newPosition: Position) => {
    setHeroPosition(newPosition)
    // Clear actions for positions after new hero
    const ALL_POSITIONS: Position[] = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']
    const heroIndex = ALL_POSITIONS.indexOf(newPosition)
    setPositionConfigs(prev =>
      prev.filter(p => ALL_POSITIONS.indexOf(p.position) < heroIndex)
    )
  }

  // Get selected hand strategy
  const selectedHandStrategy = selectedHand ? rangeData[selectedHand] : null

  return (
    <div className="range-viewer">
      <div className="viewer-main">
        {/* Left side - Range Matrix */}
        <div className="matrix-section">
          <div className="section-header">
            <h2>Range Viewer</h2>
            {isLoading && <span className="loading-indicator">Loading...</span>}
          </div>

          {error && (
            <div className="error-message">
              {error}
              <button onClick={fetchRangeData}>Retry</button>
            </div>
          )}

          <RangeMatrix
            rangeData={rangeData}
            selectedHand={selectedHand}
            onHandSelect={setSelectedHand}
          />

          {/* Selected hand details */}
          {selectedHand && selectedHandStrategy && (
            <motion.div
              className="hand-details"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h3>{selectedHand}</h3>
              <div className="strategy-breakdown">
                {Object.entries(selectedHandStrategy)
                  .filter(([_, v]) => v && v > 0.01)
                  .sort((a, b) => (b[1] || 0) - (a[1] || 0))
                  .map(([action, prob]) => (
                    <div key={action} className="strategy-item">
                      <span className="action-name">{action}</span>
                      <span className="action-prob">{((prob || 0) * 100).toFixed(1)}%</span>
                    </div>
                  ))
                }
              </div>
            </motion.div>
          )}
        </div>

        {/* Right side - Controls and Info */}
        <div className="controls-section">
          <PositionActionSelector
            positions={positionConfigs}
            heroPosition={heroPosition}
            onPositionChange={handlePositionChange}
            onHeroPositionChange={handleHeroPositionChange}
          />

          <MiniPokerTable
            positions={positionConfigs}
            heroPosition={heroPosition}
            pot={pot}
            currentBet={currentBet}
          />

          <ActionDistribution
            distribution={actionDistribution}
            legalActions={legalActions}
          />
        </div>
      </div>

      <style>{`
        .range-viewer {
          flex: 1;
          display: flex;
          flex-direction: column;
          padding: 16px 20px;
          overflow: hidden;
        }

        .viewer-main {
          flex: 1;
          display: flex;
          gap: 20px;
          overflow: hidden;
        }

        .matrix-section {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 16px;
          min-width: 0;
          overflow-y: auto;
        }

        .section-header {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .section-header h2 {
          font-size: 20px;
          font-weight: 700;
          color: var(--text-primary);
          margin: 0;
        }

        .loading-indicator {
          font-size: 12px;
          color: var(--accent-primary);
          animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .error-message {
          background: rgba(255, 69, 58, 0.1);
          border: 1px solid rgba(255, 69, 58, 0.3);
          border-radius: 8px;
          padding: 12px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-size: 13px;
          color: #ff453a;
        }

        .error-message button {
          background: rgba(255, 69, 58, 0.2);
          border: none;
          border-radius: 4px;
          padding: 4px 12px;
          font-size: 12px;
          color: #ff453a;
          cursor: pointer;
        }

        .hand-details {
          background: var(--bg-card);
          border-radius: 12px;
          padding: 16px;
          border: 1px solid var(--border-subtle);
        }

        .hand-details h3 {
          font-size: 18px;
          font-weight: 700;
          color: var(--text-primary);
          margin: 0 0 12px 0;
        }

        .strategy-breakdown {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
        }

        .strategy-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          background: var(--bg-tertiary);
          padding: 8px 16px;
          border-radius: 8px;
          min-width: 60px;
        }

        .action-name {
          font-size: 11px;
          color: var(--text-muted);
          text-transform: capitalize;
        }

        .action-prob {
          font-size: 16px;
          font-weight: 700;
          color: var(--text-primary);
        }

        .controls-section {
          width: 320px;
          flex-shrink: 0;
          display: flex;
          flex-direction: column;
          gap: 16px;
          overflow-y: auto;
        }
      `}</style>
    </div>
  )
}

export default RangeViewer
