import { Scenario, Player, Position, ActionType, TrainerResult, GTOStrategy, Card, HoleCards, PlayerAction } from '../types'

const POSITIONS: Position[] = ['UTG', 'HJ', 'CO', 'BTN', 'SB']
const RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
const SUITS: Card['suit'][] = ['spades', 'hearts', 'diamonds', 'clubs']

function getRandomElement<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

function generateRandomCard(excludeCards: Card[] = []): Card {
  let card: Card
  do {
    card = {
      rank: getRandomElement(RANKS),
      suit: getRandomElement(SUITS),
    }
  } while (
    excludeCards.some((c) => c.rank === card.rank && c.suit === card.suit)
  )
  return card
}

function getHandType(card1: Card, card2: Card): string {
  const rank1 = RANKS.indexOf(card1.rank)
  const rank2 = RANKS.indexOf(card2.rank)

  const highRank = rank1 < rank2 ? card1.rank : card2.rank
  const lowRank = rank1 < rank2 ? card2.rank : card1.rank

  if (card1.rank === card2.rank) {
    return `${card1.rank}${card2.rank}`
  }

  const suited = card1.suit === card2.suit ? 's' : 'o'
  return `${highRank}${lowRank}${suited}`
}

export function generateMockScenario(): Scenario {
  // Pick random hero position (excluding BB as per earlier fix)
  const heroPosition = getRandomElement(POSITIONS)

  // Generate hero cards
  const card1 = generateRandomCard()
  const card2 = generateRandomCard([card1])

  const heroCards: HoleCards = {
    card1,
    card2,
    handType: getHandType(card1, card2),
  }

  // Create players
  const allPositions: Position[] = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']
  const players: Player[] = allPositions.map((pos) => ({
    position: pos,
    stack: 100,
    isActive: true,
    isHero: pos === heroPosition,
    cards: pos === heroPosition ? heroCards : undefined,
  }))

  // Generate action history before hero
  const actions: PlayerAction[] = []
  const heroIdx = allPositions.indexOf(heroPosition)
  let raiseHappened = false
  let currentBet = 1.0 // BB

  for (let i = 0; i < heroIdx; i++) {
    const pos = allPositions[i]

    // Skip blinds
    if (pos === 'SB' || pos === 'BB') continue

    if (raiseHappened) {
      // After raise, fold
      actions.push({
        position: pos,
        action: 'fold',
        timestamp: Date.now() + i * 100,
      })
    } else {
      // 70% fold, 30% raise
      if (Math.random() < 0.7) {
        actions.push({
          position: pos,
          action: 'fold',
          timestamp: Date.now() + i * 100,
        })
      } else {
        raiseHappened = true
        currentBet = 2.5
        actions.push({
          position: pos,
          action: 'raise',
          amount: 2.5,
          timestamp: Date.now() + i * 100,
        })
      }
    }
  }

  // Determine legal actions
  const legalActions: ActionType[] = ['fold']
  if (!raiseHappened && heroPosition !== 'BB') {
    legalActions.push('check', 'raise', 'all-in')
  } else if (raiseHappened) {
    legalActions.push('call', 'raise', 'all-in')
  } else {
    legalActions.push('check', 'raise', 'all-in')
  }

  // Calculate pot
  const pot = 1.5 + actions.reduce((sum, a) => sum + (a.amount || 0), 0)

  return {
    players,
    heroPosition,
    heroCards,
    pot,
    currentBet,
    actions,
    legalActions,
  }
}

export function evaluateAction(
  scenario: Scenario,
  userAction: ActionType
): TrainerResult {
  // Generate mock GTO strategy based on hand strength and position
  // In real app, this would call the Python backend
  const handType = scenario.heroCards.handType

  // Simple mock strategy generation
  let gtoStrategy: GTOStrategy = {
    fold: 0.25,
    check: 0.05,
    call: 0.35,
    raise: 0.30,
    allIn: 0.05,
  }

  // Adjust based on hand type (very simplified)
  const isPremium = ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo'].some((h) =>
    handType.startsWith(h.slice(0, 2))
  )
  const isMedium = ['TT', '99', '88', 'AQs', 'AQo', 'AJs', 'KQs'].some((h) =>
    handType.startsWith(h.slice(0, 2))
  )

  if (isPremium) {
    gtoStrategy = {
      fold: 0,
      check: 0,
      call: 0.1,
      raise: 0.75,
      allIn: 0.15,
    }
  } else if (isMedium) {
    gtoStrategy = {
      fold: 0.1,
      check: 0.05,
      call: 0.45,
      raise: 0.35,
      allIn: 0.05,
    }
  }

  // Filter to only legal actions and renormalize
  const legalActions = scenario.legalActions
  let filteredStrategy: GTOStrategy = { ...gtoStrategy }

  const actionKeys: (keyof GTOStrategy)[] = ['fold', 'check', 'call', 'raise', 'allIn']
  let sum = 0

  for (const key of actionKeys) {
    const actionType = key === 'allIn' ? 'all-in' : key
    if (!legalActions.includes(actionType as ActionType)) {
      filteredStrategy[key] = 0
    } else {
      sum += filteredStrategy[key]
    }
  }

  // Renormalize
  if (sum > 0) {
    for (const key of actionKeys) {
      filteredStrategy[key] /= sum
    }
  }

  // Find best action
  const bestAction = actionKeys.reduce((best, key) =>
    filteredStrategy[key] > filteredStrategy[best] ? key : best
  )

  // Convert userAction to match strategy keys
  const userActionKey = userAction === 'all-in' ? 'allIn' : userAction

  // Check if correct (user chose action with >25% frequency)
  const userProb = filteredStrategy[userActionKey as keyof GTOStrategy] || 0
  const isCorrect = userProb >= 0.25

  // Generate feedback
  let feedback: string
  if (isCorrect) {
    if (userActionKey === bestAction) {
      feedback = 'Perfect! This is the optimal GTO play.'
    } else {
      feedback = `Good choice! ${userAction} is acceptable in this spot.`
    }
  } else {
    const bestActionLabel = bestAction === 'allIn' ? 'All-In' : bestAction.charAt(0).toUpperCase() + bestAction.slice(1)
    feedback = `GTO prefers ${bestActionLabel} here (${(filteredStrategy[bestAction] * 100).toFixed(0)}%).`
  }

  return {
    userAction,
    gtoStrategy: filteredStrategy,
    isCorrect,
    feedback,
  }
}

// API connection for production
const API_BASE = 'http://localhost:5000'

export async function fetchScenario(): Promise<Scenario> {
  try {
    const response = await fetch(`${API_BASE}/scenario`)
    if (!response.ok) throw new Error('Failed to fetch scenario')
    return await response.json()
  } catch (error) {
    console.warn('API not available, using mock data:', error)
    return generateMockScenario()
  }
}

export async function submitAction(
  scenario: Scenario,
  action: ActionType
): Promise<TrainerResult> {
  try {
    const response = await fetch(`${API_BASE}/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenarioId: scenario.scenarioId, action }),
    })
    if (!response.ok) throw new Error('Failed to evaluate action')
    return await response.json()
  } catch (error) {
    console.warn('API not available, using mock evaluation:', error)
    return evaluateAction(scenario, action)
  }
}
