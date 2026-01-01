import React, { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import TitleBar from './components/TitleBar'
import PokerTable from './components/PokerTable'
import ActionPanel from './components/ActionPanel'
import ResultModal from './components/ResultModal'
import StatsPanel from './components/StatsPanel'
import RangeViewer from './components/RangeViewer'
import { Scenario, TrainerResult, ActionType, UserStats } from './types'
import { fetchScenario, submitAction } from './hooks/useTrainer'

type AppTab = 'trainer' | 'range'

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<AppTab>('trainer')
  const [scenario, setScenario] = useState<Scenario | null>(null)
  const [result, setResult] = useState<TrainerResult | null>(null)
  const [isAnimating, setIsAnimating] = useState(false)
  const [showResult, setShowResult] = useState(false)
  const [stats, setStats] = useState<UserStats>({
    totalHands: 0,
    correctDecisions: 0,
    accuracy: 0,
    sessionStart: new Date().toISOString(),
  })

  const startNewHand = async () => {
    setResult(null)
    setShowResult(false)
    setIsAnimating(true)

    await new Promise((resolve) => setTimeout(resolve, 250))

    const newScenario = await fetchScenario()
    setScenario(newScenario)
    setIsAnimating(false)
  }

  const handleAction = async (action: ActionType) => {
    if (!scenario || isAnimating) return

    setIsAnimating(true)

    const evalResult = await submitAction(scenario, action)
    setResult(evalResult)

    setStats((prev) => {
      const newTotal = prev.totalHands + 1
      const newCorrect = prev.correctDecisions + (evalResult.isCorrect ? 1 : 0)
      return {
        ...prev,
        totalHands: newTotal,
        correctDecisions: newCorrect,
        accuracy: (newCorrect / newTotal) * 100,
      }
    })

    await new Promise((resolve) => setTimeout(resolve, 400))
    setShowResult(true)
    setIsAnimating(false)
  }

  const handleNextHand = () => {
    startNewHand()
  }

  React.useEffect(() => {
    if (activeTab === 'trainer' && !scenario) {
      startNewHand()
    }
  }, [activeTab])

  return (
    <div className="app">
      <TitleBar />

      {/* Navigation Tabs */}
      <div className="nav-tabs">
        <motion.button
          className={`nav-tab ${activeTab === 'trainer' ? 'active' : ''}`}
          onClick={() => setActiveTab('trainer')}
          whileHover={{ backgroundColor: 'rgba(255, 255, 255, 0.05)' }}
          whileTap={{ scale: 0.98 }}
        >
          <span className="tab-icon">🎯</span>
          <span className="tab-label">Trainer</span>
        </motion.button>
        <motion.button
          className={`nav-tab ${activeTab === 'range' ? 'active' : ''}`}
          onClick={() => setActiveTab('range')}
          whileHover={{ backgroundColor: 'rgba(255, 255, 255, 0.05)' }}
          whileTap={{ scale: 0.98 }}
        >
          <span className="tab-icon">📊</span>
          <span className="tab-label">Range Viewer</span>
        </motion.button>
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'trainer' ? (
          <motion.main
            key="trainer"
            className="main-content"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.2 }}
          >
            <div className="game-area">
              <PokerTable scenario={scenario} isAnimating={isAnimating} />

              <ActionPanel
                scenario={scenario}
                onAction={handleAction}
                disabled={isAnimating || showResult}
                result={result}
              />
            </div>

            <StatsPanel stats={stats} />
          </motion.main>
        ) : (
          <motion.div
            key="range"
            className="range-content"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
          >
            <RangeViewer />
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showResult && result && activeTab === 'trainer' && (
          <ResultModal
            result={result}
            scenario={scenario}
            onNext={handleNextHand}
            onClose={() => setShowResult(false)}
          />
        )}
      </AnimatePresence>

      <style>{`
        .app {
          display: flex;
          flex-direction: column;
          height: 100vh;
          background: var(--bg-primary);
          overflow: hidden;
        }

        .nav-tabs {
          display: flex;
          gap: 4px;
          padding: 8px 20px;
          background: var(--bg-secondary);
          border-bottom: 1px solid var(--border-subtle);
        }

        .nav-tab {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 20px;
          background: transparent;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .nav-tab .tab-icon {
          font-size: 16px;
        }

        .nav-tab .tab-label {
          font-size: 13px;
          font-weight: 500;
          color: var(--text-secondary);
        }

        .nav-tab.active {
          background: var(--bg-tertiary);
        }

        .nav-tab.active .tab-label {
          color: var(--text-primary);
          font-weight: 600;
        }

        .main-content {
          flex: 1;
          display: flex;
          padding: 16px 20px;
          gap: 20px;
          overflow: hidden;
          min-height: 0;
        }

        .range-content {
          flex: 1;
          display: flex;
          overflow: hidden;
          min-height: 0;
        }

        .game-area {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 16px;
          min-width: 0;
        }
      `}</style>
    </div>
  )
}

export default App
