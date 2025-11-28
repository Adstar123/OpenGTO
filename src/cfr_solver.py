"""
Counterfactual Regret Minimization (CFR) solver for preflop poker.
Implements vanilla CFR algorithm to find Nash equilibrium strategies.
"""
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from src.card import Hand, create_deck
from src.game_tree import PreflopGameTree, GameState, Action, Position
from src.equity import PreflopEquityCalculator
from itertools import combinations


class CFRSolver:
    """
    CFR solver for computing Nash equilibrium strategies in preflop poker.
    """

    def __init__(self, stack_size: float = 100.0, big_blind: float = 1.0):
        """
        Initialize CFR solver.

        Args:
            stack_size: Stack size in big blinds
            big_blind: Big blind size
        """
        self.game_tree = PreflopGameTree(stack_size, big_blind)
        self.equity_calc = PreflopEquityCalculator()

        # Information set -> Action -> cumulative regret
        self.regret_sum: Dict[str, Dict[Action, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Information set -> Action -> cumulative strategy
        self.strategy_sum: Dict[str, Dict[Action, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Current iteration
        self.iteration = 0

    def get_information_set(self, hand: Hand, state: GameState, position: Position) -> str:
        """
        Create information set string for a given hand and game state.

        Information set includes:
        - Player's hand
        - Action history
        - Position
        """
        hand_str = hand.to_string()
        history_str = "_".join([a.value for a in state.action_history])
        return f"{position.value}:{hand_str}:{history_str}"

    def get_strategy(self, info_set: str, legal_actions: List[Action]) -> Dict[Action, float]:
        """
        Get current strategy for an information set using regret matching.

        Args:
            info_set: Information set string
            legal_actions: List of legal actions

        Returns:
            Dictionary mapping actions to probabilities
        """
        if not legal_actions:
            return {}

        # Get regrets for this info set
        regrets = {action: max(0, self.regret_sum[info_set][action])
                  for action in legal_actions}

        normalizing_sum = sum(regrets.values())

        if normalizing_sum > 0:
            # Regret matching: probability proportional to positive regret
            strategy = {action: regrets[action] / normalizing_sum
                       for action in legal_actions}
        else:
            # Uniform random if no positive regrets
            strategy = {action: 1.0 / len(legal_actions)
                       for action in legal_actions}

        return strategy

    def get_average_strategy(self, info_set: str) -> Dict[Action, float]:
        """
        Get the average strategy for an information set across all iterations.
        This is the Nash equilibrium approximation.

        Args:
            info_set: Information set string

        Returns:
            Dictionary mapping actions to probabilities
        """
        strategy = {}
        normalizing_sum = sum(self.strategy_sum[info_set].values())

        if normalizing_sum > 0:
            strategy = {action: self.strategy_sum[info_set][action] / normalizing_sum
                       for action in self.strategy_sum[info_set].keys()}
        else:
            # Return uniform if no data
            actions = list(self.strategy_sum[info_set].keys())
            if actions:
                strategy = {action: 1.0 / len(actions) for action in actions}

        return strategy

    def cfr(self, state: GameState, hand1: Hand, hand2: Hand,
            reach_prob_btn: float, reach_prob_bb: float,
            position: Position) -> float:
        """
        Recursive CFR algorithm.

        Args:
            state: Current game state
            hand1: BTN's hand
            hand2: BB's hand
            reach_prob_btn: Probability of reaching this state for BTN
            reach_prob_bb: Probability of reaching this state for BB
            position: Which player's perspective (for updating regrets)

        Returns:
            Expected value for the position player
        """
        # Terminal state
        if state.is_terminal:
            if state.payoff is None:
                raise ValueError("Terminal state must have a payoff")

            # Return payoff from the perspective of the position player
            if position == Position.BTN:
                return state.payoff
            else:
                return -state.payoff

        # Get information set for current player
        current_player = state.position_to_act
        current_hand = hand1 if current_player == Position.BTN else hand2

        info_set = self.get_information_set(current_hand, state, current_player)
        legal_actions = self.game_tree.get_legal_actions(state)

        if not legal_actions:
            return 0.0

        # Get current strategy
        strategy = self.get_strategy(info_set, legal_actions)

        # Track action values
        action_values = {}
        node_value = 0.0

        # Recursively calculate value for each action
        for action in legal_actions:
            new_state = self.game_tree.apply_action(state, action, hand1, hand2,
                                                    self.equity_calc)

            # Update reach probabilities
            if current_player == Position.BTN:
                action_value = self.cfr(
                    new_state, hand1, hand2,
                    reach_prob_btn * strategy[action],
                    reach_prob_bb,
                    position
                )
            else:
                action_value = self.cfr(
                    new_state, hand1, hand2,
                    reach_prob_btn,
                    reach_prob_bb * strategy[action],
                    position
                )

            action_values[action] = action_value
            node_value += strategy[action] * action_value

        # Update regrets and strategy sum (only for the updating player)
        if current_player == position:
            # Calculate counterfactual reach probability
            cf_reach_prob = reach_prob_bb if position == Position.BTN else reach_prob_btn

            for action in legal_actions:
                regret = action_values[action] - node_value
                self.regret_sum[info_set][action] += cf_reach_prob * regret

            # Update strategy sum
            reach_prob = reach_prob_btn if position == Position.BTN else reach_prob_bb
            for action in legal_actions:
                self.strategy_sum[info_set][action] += reach_prob * strategy[action]

        return node_value

    def train(self, iterations: int = 10000, sample_hands: int = 100):
        """
        Train the CFR solver for a number of iterations.

        Args:
            iterations: Number of CFR iterations to run
            sample_hands: Number of random hand matchups to sample per iteration
        """
        print(f"Training CFR solver for {iterations} iterations...")

        deck = create_deck()

        for i in range(iterations):
            self.iteration = i

            # Sample random hands
            for _ in range(sample_hands):
                # Sample two random distinct cards for each player
                sampled_cards = np.random.choice(deck, size=4, replace=False)

                hand1 = Hand(sampled_cards[0], sampled_cards[1])
                hand2 = Hand(sampled_cards[2], sampled_cards[3])

                # Get initial state
                initial_state = self.game_tree.get_initial_state()

                # Run CFR from both perspectives
                self.cfr(initial_state, hand1, hand2, 1.0, 1.0, Position.BTN)
                self.cfr(initial_state, hand1, hand2, 1.0, 1.0, Position.BB)

            if (i + 1) % 100 == 0:
                print(f"Completed iteration {i + 1}/{iterations}")

        print("Training complete!")

    def get_all_strategies(self) -> Dict[str, Dict[Action, float]]:
        """
        Get average strategies for all information sets.

        Returns:
            Dictionary mapping info sets to average strategies
        """
        strategies = {}
        for info_set in self.strategy_sum.keys():
            strategies[info_set] = self.get_average_strategy(info_set)
        return strategies

    def print_sample_strategies(self, num_samples: int = 20):
        """Print some sample strategies for inspection."""
        print("\n" + "="*80)
        print("Sample Strategies (Nash Equilibrium Approximation)")
        print("="*80)

        strategies = self.get_all_strategies()
        sample_info_sets = list(strategies.keys())[:num_samples]

        for info_set in sample_info_sets:
            strategy = strategies[info_set]
            print(f"\nInfo Set: {info_set}")
            for action, prob in sorted(strategy.items(), key=lambda x: -x[1]):
                print(f"  {action.value:15s}: {prob:6.2%}")
