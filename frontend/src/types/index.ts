export type Position = 'UTG' | 'HJ' | 'CO' | 'BTN' | 'SB' | 'BB'

export type ActionType = 'fold' | 'check' | 'call' | 'raise' | 'all-in'

export interface PlayerAction {
  position: Position
  action: ActionType
  amount?: number
  timestamp: number
}

export interface Card {
  rank: string
  suit: 'spades' | 'hearts' | 'diamonds' | 'clubs'
}

export interface HoleCards {
  card1: Card
  card2: Card
  handType: string // e.g., "AKs", "QQ", "72o"
}

export interface Player {
  position: Position
  stack: number
  isActive: boolean
  isHero: boolean
  action?: PlayerAction
  cards?: HoleCards
}

export interface GTOStrategy {
  fold: number
  check: number
  call: number
  raise: number
  allIn: number
}

export interface Scenario {
  scenarioId?: string
  players: Player[]
  heroPosition: Position
  heroCards: HoleCards
  pot: number
  currentBet: number
  actions: PlayerAction[]
  legalActions: ActionType[]
}

export interface TrainerResult {
  userAction: ActionType
  gtoStrategy: GTOStrategy
  isCorrect: boolean
  feedback: string
}

export interface UserStats {
  totalHands: number
  correctDecisions: number
  accuracy: number
  sessionStart: string
}
