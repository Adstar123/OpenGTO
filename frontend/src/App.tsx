import React, { useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import TitleBar from './components/TitleBar'
import PokerTable from './components/PokerTable'
import ActionPanel from './components/ActionPanel'
import ResultModal from './components/ResultModal'
import StatsPanel from './components/StatsPanel'
import { Scenario, TrainerResult, ActionType, UserStats } from './types'
import { generateMockScenario, evaluateAction } from './hooks/useTrainer'

const App: React.FC = () => {
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

    const newScenario = generateMockScenario()
    setScenario(newScenario)
    setIsAnimating(false)
  }

  const handleAction = async (action: ActionType) => {
    if (!scenario || isAnimating) return

    setIsAnimating(true)

    const evalResult = evaluateAction(scenario, action)
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
    startNewHand()
  }, [])

  return (
    <div className="app">
      <TitleBar />

      <main className="main-content">
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
      </main>

      <AnimatePresence>
        {showResult && result && (
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

        .main-content {
          flex: 1;
          display: flex;
          padding: 16px 20px;
          gap: 20px;
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
