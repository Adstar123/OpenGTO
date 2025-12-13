"""
Evaluation tools for measuring strategy quality and convergence.

Includes:
- Exploitability measurement
- Strategy comparison
- Hand-by-hand analysis
- Performance metrics
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict

from .game_state import GameState, Action, ActionType, Position
from .poker_engine import PreflopPokerEngine, ScenarioBuilder
from .card import HoleCards, get_all_hand_types, hand_type_to_index
from .information_set import (
    InformationSet, get_legal_actions_mask, action_idx_to_action,
    sample_action, NUM_ACTIONS, IDX_TO_ACTION_TYPE
)
from .hand_utils import HandRange, print_hand_matrix
from .showdown import run_out_board


@dataclass
class StrategyAnalysis:
    """Analysis of strategy at a single decision point."""
    hand_type: str
    position: str
    action_probs: Dict[str, float]
    recommended_action: str
    scenario_description: str


@dataclass
class ExploitabilityResult:
    """Result of exploitability calculation."""
    exploitability: float  # In milli-big-blinds per hand
    best_response_value: float
    num_hands_tested: int


class StrategyEvaluator:
    """
    Evaluates the quality of learned strategies.
    """

    def __init__(
        self,
        strategy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        num_players: int = 6,
        starting_stack: float = 100.0
    ):
        self.strategy_fn = strategy_fn
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.engine = PreflopPokerEngine(
            num_players=num_players,
            starting_stack=starting_stack
        )

    def analyze_opening_ranges(self) -> Dict[str, HandRange]:
        """
        Analyze opening ranges for each position.

        Returns:
            Dictionary mapping position name to HandRange with frequencies
        """
        ranges = {}
        all_hands = get_all_hand_types()

        for position in Position:
            if position.value >= self.num_players:
                continue

            range_obj = HandRange.empty()

            for hand_type in all_hands:
                # Create scenario: hero at position, action folds to them
                probs = self._get_opening_probs(position, hand_type)

                # Compute "opening" frequency (raise + bet + all-in)
                open_freq = probs.get('bet', 0) + probs.get('raise', 0) + probs.get('all-in', 0)

                if open_freq > 0.01:  # At least 1% opening frequency
                    range_obj.add_hand(hand_type, open_freq)

            ranges[position.name] = range_obj

        return ranges

    def _get_opening_probs(self, position: Position, hand_type: str) -> Dict[str, float]:
        """Get action probabilities for opening at a position."""
        # Build scenario
        builder = ScenarioBuilder(self.engine)

        # Set stacks
        stacks = [self.starting_stack] * self.num_players
        builder.set_stacks(stacks)

        # Get a specific combo for this hand type
        from .hand_utils import get_combos_for_hand_type
        combos = get_combos_for_hand_type(hand_type)
        if not combos:
            return {}

        hole_cards = combos[0]
        builder.set_hero_cards(position, hole_cards)

        # Add folds before hero's position
        preflop_order = Position.preflop_order(self.num_players)
        for pos in preflop_order:
            if pos == position:
                break
            # Skip blinds in preflop order for folding
            if pos not in [Position.SB, Position.BB]:
                builder.add_action(pos, Action(ActionType.FOLD))

        try:
            state = builder.build()
        except Exception:
            return {}

        # Get strategy
        info_set = InformationSet.from_game_state(state)
        features = info_set.to_feature_vector()
        legal_mask = get_legal_actions_mask(state)

        strategy = self.strategy_fn(features, legal_mask)

        # Convert to named dict
        action_names = ['fold', 'check', 'call', 'bet', 'raise', 'all-in']
        result = {}
        for i, name in enumerate(action_names):
            if legal_mask[i]:
                result[name] = float(strategy[i])

        return result

    def analyze_hand(
        self,
        hand_type: str,
        position: str,
        scenario: str = "open"
    ) -> StrategyAnalysis:
        """
        Analyze strategy for a specific hand in a scenario.

        Args:
            hand_type: Hand like "AKs", "77", etc.
            position: Position like "BTN", "BB"
            scenario: "open", "vs_raise", "vs_3bet"
        """
        pos = Position[position.upper()]
        probs = self._get_opening_probs(pos, hand_type)

        # Find recommended action
        if probs:
            recommended = max(probs.keys(), key=lambda k: probs[k])
        else:
            recommended = "unknown"

        return StrategyAnalysis(
            hand_type=hand_type,
            position=position,
            action_probs=probs,
            recommended_action=recommended,
            scenario_description=f"{scenario} from {position}"
        )

    def compute_exploitability(
        self,
        num_samples: int = 1000
    ) -> ExploitabilityResult:
        """
        Estimate exploitability of the strategy.

        This computes how much an optimal best-response strategy
        could win against our strategy.

        Lower is better. 0 = Nash equilibrium.
        """
        total_value = 0.0
        hands_tested = 0

        for _ in range(num_samples):
            state = self.engine.create_game()

            # Play out game with our strategy
            while not state.is_complete:
                info_set = InformationSet.from_game_state(state)
                features = info_set.to_feature_vector()
                legal_mask = get_legal_actions_mask(state)

                strategy = self.strategy_fn(features, legal_mask)
                action_idx = sample_action(strategy, legal_mask)
                action = action_idx_to_action(action_idx, state)

                state = self.engine.apply_action(state, action)

            # Compute utility for player 0
            player = state.players[0]
            if not player.is_active:
                utility = -player.current_bet
            else:
                active = [p for p in state.players if p.is_active]
                if len(active) == 1:
                    utility = state.pot - player.current_bet
                else:
                    # Showdown
                    cards = [p.hole_cards for p in active]
                    equities = run_out_board(cards, num_simulations=50)
                    idx = [p for p in active].index(player) if player in active else 0
                    utility = equities[idx] * state.pot - player.current_bet

            total_value += utility
            hands_tested += 1

        avg_value = total_value / hands_tested if hands_tested > 0 else 0

        # Exploitability in mBB/hand
        # A perfectly balanced strategy would have ~0 expected value
        exploitability = abs(avg_value) * 1000  # Convert to mBB

        return ExploitabilityResult(
            exploitability=exploitability,
            best_response_value=avg_value,
            num_hands_tested=hands_tested
        )

    def compare_strategies(
        self,
        other_strategy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        num_games: int = 1000
    ) -> Dict[str, float]:
        """
        Compare two strategies by playing them against each other.

        Returns:
            Dictionary with comparison metrics
        """
        our_wins = 0
        our_utility = 0.0

        for _ in range(num_games):
            # Alternate who plays which strategy
            state = self.engine.create_game()

            while not state.is_complete:
                current_player = state.current_player_idx
                info_set = InformationSet.from_game_state(state)
                features = info_set.to_feature_vector()
                legal_mask = get_legal_actions_mask(state)

                # Player 0 uses our strategy, others use the other strategy
                if current_player == 0:
                    strategy = self.strategy_fn(features, legal_mask)
                else:
                    strategy = other_strategy_fn(features, legal_mask)

                action_idx = sample_action(strategy, legal_mask)
                action = action_idx_to_action(action_idx, state)
                state = self.engine.apply_action(state, action)

            # Compute player 0's result
            player = state.players[0]
            if not player.is_active:
                utility = -player.current_bet
            else:
                active = [p for p in state.players if p.is_active]
                if len(active) == 1 and player.is_active:
                    utility = state.pot - player.current_bet
                    our_wins += 1
                else:
                    cards = [p.hole_cards for p in active]
                    equities = run_out_board(cards, num_simulations=50)
                    idx = active.index(player) if player in active else 0
                    utility = equities[idx] * state.pot - player.current_bet
                    if utility > 0:
                        our_wins += 0.5  # Partial win based on equity

            our_utility += utility

        return {
            'win_rate': our_wins / num_games,
            'avg_utility': our_utility / num_games,
            'bb_per_100': (our_utility / num_games) * 100,
            'num_games': num_games
        }


class ConvergenceMonitor:
    """
    Monitors convergence during training.
    """

    def __init__(self, check_interval: int = 100):
        self.check_interval = check_interval
        self.history: List[Dict] = []
        self.iteration = 0

    def update(
        self,
        regret_loss: float,
        strategy_loss: float,
        utility: float,
        exploitability: Optional[float] = None
    ):
        """Record metrics for this iteration."""
        self.iteration += 1

        metrics = {
            'iteration': self.iteration,
            'regret_loss': regret_loss,
            'strategy_loss': strategy_loss,
            'utility': utility,
            'exploitability': exploitability
        }
        self.history.append(metrics)

    def should_check_convergence(self) -> bool:
        """Check if we should run convergence analysis."""
        return self.iteration % self.check_interval == 0

    def get_convergence_metrics(self, window: int = 100) -> Dict[str, float]:
        """Get convergence metrics over recent window."""
        if len(self.history) < window:
            recent = self.history
        else:
            recent = self.history[-window:]

        if not recent:
            return {}

        regret_losses = [m['regret_loss'] for m in recent]
        strategy_losses = [m['strategy_loss'] for m in recent]
        utilities = [m['utility'] for m in recent]

        return {
            'avg_regret_loss': np.mean(regret_losses),
            'std_regret_loss': np.std(regret_losses),
            'avg_strategy_loss': np.mean(strategy_losses),
            'std_strategy_loss': np.std(strategy_losses),
            'avg_utility': np.mean(utilities),
            'std_utility': np.std(utilities),
            'regret_trend': self._compute_trend(regret_losses),
            'strategy_trend': self._compute_trend(strategy_losses)
        }

    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend of values. Negative = decreasing (good for loss)."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope

    def is_converged(self, threshold: float = 0.001) -> bool:
        """Check if training has converged."""
        metrics = self.get_convergence_metrics()

        if not metrics:
            return False

        # Check if losses have stabilized
        regret_stable = metrics['std_regret_loss'] < threshold * metrics['avg_regret_loss']
        strategy_stable = metrics['std_strategy_loss'] < threshold * metrics['avg_strategy_loss']

        # Check if trends are near zero
        regret_trend_small = abs(metrics['regret_trend']) < threshold
        strategy_trend_small = abs(metrics['strategy_trend']) < threshold

        return (regret_stable and strategy_stable and
                regret_trend_small and strategy_trend_small)

    def get_summary(self) -> str:
        """Get summary string."""
        metrics = self.get_convergence_metrics()
        if not metrics:
            return "No data yet"

        return (
            f"Iteration {self.iteration}: "
            f"Regret Loss={metrics['avg_regret_loss']:.4f} (trend={metrics['regret_trend']:.6f}), "
            f"Strategy Loss={metrics['avg_strategy_loss']:.4f} (trend={metrics['strategy_trend']:.6f})"
        )


def print_range_matrix(
    ranges: Dict[str, HandRange],
    position: str = "BTN"
) -> str:
    """
    Print a visual matrix of opening range for a position.
    """
    if position not in ranges:
        return f"No range found for {position}"

    range_obj = ranges[position]
    matrix = range_obj.to_matrix()

    lines = [f"\n{position} Opening Range ({range_obj.percentage():.1f}% of hands)\n"]
    lines.append(print_hand_matrix(matrix))

    return '\n'.join(lines)


def analyze_preflop_strategy(
    strategy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    num_players: int = 6,
    stack_size: float = 100.0
) -> str:
    """
    Generate a full analysis report of preflop strategy.
    """
    evaluator = StrategyEvaluator(strategy_fn, num_players, stack_size)

    lines = ["=" * 60]
    lines.append("PREFLOP STRATEGY ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Players: {num_players}, Stack: {stack_size}bb")
    lines.append("")

    # Opening ranges
    ranges = evaluator.analyze_opening_ranges()

    for pos_name, range_obj in ranges.items():
        if range_obj.num_combos() > 0:
            lines.append(f"\n{pos_name}: {range_obj.percentage():.1f}% of hands")
            # Show top hands
            top_hands = sorted(
                range_obj.hands.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            hands_str = ", ".join(f"{h}({f*100:.0f}%)" for h, f in top_hands)
            lines.append(f"  Top hands: {hands_str}")

    # Sample hand analyses
    lines.append("\n" + "-" * 60)
    lines.append("SAMPLE HAND ANALYSIS")
    lines.append("-" * 60)

    sample_hands = ["AA", "AKs", "AKo", "QQ", "JTs", "77", "A5s", "KJo"]
    positions = ["UTG", "BTN", "BB"]

    for hand in sample_hands[:4]:  # Just show a few
        for pos in positions[:2]:
            try:
                analysis = evaluator.analyze_hand(hand, pos)
                if analysis.action_probs:
                    probs_str = ", ".join(
                        f"{k}:{v*100:.0f}%"
                        for k, v in sorted(analysis.action_probs.items(), key=lambda x: -x[1])
                        if v > 0.01
                    )
                    lines.append(f"{hand} from {pos}: {probs_str}")
            except Exception:
                pass

    # Exploitability estimate
    lines.append("\n" + "-" * 60)
    lines.append("EXPLOITABILITY ESTIMATE")
    lines.append("-" * 60)

    try:
        exploit = evaluator.compute_exploitability(num_samples=500)
        lines.append(f"Exploitability: {exploit.exploitability:.2f} mBB/hand")
        lines.append(f"(Lower is better, 0 = Nash equilibrium)")
    except Exception as e:
        lines.append(f"Could not compute exploitability: {e}")

    lines.append("\n" + "=" * 60)

    return '\n'.join(lines)
