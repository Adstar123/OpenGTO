"""
Counterfactual Regret Minimization (CFR) Engine.

This implements the core CFR algorithm that the neural network learns from.
The algorithm works by:
1. Traversing the game tree
2. Computing counterfactual values for each action
3. Computing regrets (how much better each action would have been)
4. Updating strategy via regret matching

Over many iterations, the average strategy converges to Nash equilibrium.
"""
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

from .game_state import GameState, Action, ActionType, Position
from .poker_engine import PreflopPokerEngine
from .information_set import (
    InformationSet, get_legal_actions_mask, action_idx_to_action,
    sample_action, CFRTraversalState, NUM_ACTIONS, ACTION_TYPE_TO_IDX
)
from .showdown import run_out_board
from .card import HoleCards


@dataclass
class RegretStore:
    """
    Stores cumulative regrets and strategy sums for tabular CFR.
    Used for small games or as a reference implementation.
    """
    # Cumulative regrets: info_set_key -> regrets[action]
    cumulative_regrets: Dict[str, np.ndarray] = field(
        default_factory=lambda: defaultdict(lambda: np.zeros(NUM_ACTIONS))
    )

    # Strategy sum for computing average strategy
    strategy_sum: Dict[str, np.ndarray] = field(
        default_factory=lambda: defaultdict(lambda: np.zeros(NUM_ACTIONS))
    )

    def get_strategy(self, info_set_key: str, legal_mask: np.ndarray) -> np.ndarray:
        """
        Get current strategy via regret matching.
        """
        regrets = self.cumulative_regrets[info_set_key]
        return self._regret_matching(regrets, legal_mask)

    def get_average_strategy(self, info_set_key: str, legal_mask: np.ndarray) -> np.ndarray:
        """
        Get average strategy (converges to Nash equilibrium).
        """
        strategy_sum = self.strategy_sum[info_set_key]

        # Normalize
        total = (strategy_sum * legal_mask).sum()
        if total > 0:
            avg_strategy = strategy_sum / total
            avg_strategy = avg_strategy * legal_mask
            return avg_strategy / avg_strategy.sum()
        else:
            # Uniform
            num_legal = legal_mask.sum()
            return legal_mask.astype(float) / num_legal

    def update_regrets(
        self,
        info_set_key: str,
        regrets: np.ndarray,
        weight: float = 1.0
    ):
        """Update cumulative regrets."""
        self.cumulative_regrets[info_set_key] += regrets * weight

    def update_strategy_sum(
        self,
        info_set_key: str,
        strategy: np.ndarray,
        weight: float = 1.0
    ):
        """Update strategy sum for average strategy computation."""
        self.strategy_sum[info_set_key] += strategy * weight

    def _regret_matching(self, regrets: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        """Convert regrets to strategy via regret matching."""
        # Mask illegal actions
        masked_regrets = np.where(legal_mask, regrets, -np.inf)

        # Take positive part
        positive_regrets = np.maximum(masked_regrets, 0)

        # Normalize
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        else:
            # Uniform over legal actions
            num_legal = legal_mask.sum()
            return legal_mask.astype(float) / num_legal


class CFREngine:
    """
    Core CFR engine for preflop poker.

    This implements External Sampling Monte Carlo CFR (MCCFR),
    which is more efficient than vanilla CFR for large games.
    """

    def __init__(
        self,
        poker_engine: PreflopPokerEngine,
        regret_store: Optional[RegretStore] = None
    ):
        self.poker_engine = poker_engine
        self.regret_store = regret_store or RegretStore()

        # Statistics
        self.num_iterations = 0
        self.total_traversals = 0

    def cfr_iteration(self) -> float:
        """
        Run one iteration of CFR.

        Returns:
            Average utility for the iteration
        """
        self.num_iterations += 1
        total_utility = 0.0

        # Create initial game state
        state = self.poker_engine.create_game()

        # Traverse for each player
        for player_idx in range(self.poker_engine.num_players):
            traversal_state = CFRTraversalState(
                reach_probs={i: 1.0 for i in range(self.poker_engine.num_players)},
                iteration=self.num_iterations,
                traversing_player=player_idx
            )

            utility = self._cfr_traverse(state, traversal_state, player_idx)
            total_utility += utility

        return total_utility / self.poker_engine.num_players

    def _cfr_traverse(
        self,
        state: GameState,
        traversal_state: CFRTraversalState,
        traversing_player: int
    ) -> float:
        """
        Recursive CFR traversal.

        Args:
            state: Current game state
            traversal_state: Reach probabilities and iteration info
            traversing_player: Which player's regrets we're computing

        Returns:
            Expected utility for the traversing player
        """
        self.total_traversals += 1

        # Terminal state - return utility
        if state.is_complete:
            return self._get_terminal_utility(state, traversing_player)

        current_player_idx = state.current_player_idx
        info_set = InformationSet.from_game_state(state)
        info_set_key = info_set.to_key()
        legal_mask = get_legal_actions_mask(state)

        # Get current strategy
        strategy = self.regret_store.get_strategy(info_set_key, legal_mask)

        if current_player_idx == traversing_player:
            # Traversing player - compute regrets
            action_values = np.zeros(NUM_ACTIONS)
            node_value = 0.0

            for action_idx in range(NUM_ACTIONS):
                if not legal_mask[action_idx]:
                    continue

                # Apply action
                action = action_idx_to_action(action_idx, state)
                next_state = self.poker_engine.apply_action(state, action)

                # Update reach probability
                new_traversal = traversal_state.update_reach_prob(
                    current_player_idx, strategy[action_idx]
                )

                # Recurse
                action_values[action_idx] = self._cfr_traverse(
                    next_state, new_traversal, traversing_player
                )

                node_value += strategy[action_idx] * action_values[action_idx]

            # Compute regrets
            regrets = action_values - node_value

            # Weight by opponent reach probability
            opponent_reach = traversal_state.opponent_reach_prob(traversing_player)
            self.regret_store.update_regrets(info_set_key, regrets, opponent_reach)

            # Update strategy sum
            player_reach = traversal_state.player_reach_prob(traversing_player)
            self.regret_store.update_strategy_sum(info_set_key, strategy, player_reach)

            return node_value

        else:
            # Opponent - sample action according to strategy
            action_idx = sample_action(strategy, legal_mask)
            action = action_idx_to_action(action_idx, state)
            next_state = self.poker_engine.apply_action(state, action)

            # Update reach probability
            new_traversal = traversal_state.update_reach_prob(
                current_player_idx, strategy[action_idx]
            )

            return self._cfr_traverse(next_state, new_traversal, traversing_player)

    def _get_terminal_utility(self, state: GameState, player_idx: int) -> float:
        """
        Compute utility at terminal state for a player.

        For preflop, if multiple players are all-in, we run out the board
        to determine equity. Otherwise, the player who won the pot gets it.
        """
        player = state.players[player_idx]

        # If player folded, they lose what they put in
        if not player.is_active:
            return -player.current_bet

        active_players = [p for p in state.players if p.is_active]

        if len(active_players) == 1:
            # Player won by everyone else folding
            if active_players[0] == player:
                # Won the pot minus what they put in
                return state.pot - player.current_bet
            else:
                return -player.current_bet

        # Multiple players active - need showdown
        # For preflop all-in, use equity calculation
        hole_cards_list = [p.hole_cards for p in active_players]
        equities = run_out_board(hole_cards_list, num_simulations=100)

        # Find this player's index among active players
        active_idx = None
        for i, p in enumerate(active_players):
            if p == player:
                active_idx = i
                break

        if active_idx is None:
            return -player.current_bet

        # Expected utility = equity * pot - amount invested
        expected_winnings = equities[active_idx] * state.pot
        return expected_winnings - player.current_bet

    def get_strategy(self, state: GameState) -> np.ndarray:
        """Get current strategy for a game state."""
        info_set = InformationSet.from_game_state(state)
        info_set_key = info_set.to_key()
        legal_mask = get_legal_actions_mask(state)
        return self.regret_store.get_strategy(info_set_key, legal_mask)

    def get_average_strategy(self, state: GameState) -> np.ndarray:
        """Get average strategy (GTO approximation) for a game state."""
        info_set = InformationSet.from_game_state(state)
        info_set_key = info_set.to_key()
        legal_mask = get_legal_actions_mask(state)
        return self.regret_store.get_average_strategy(info_set_key, legal_mask)


class ExternalSamplingCFR(CFREngine):
    """
    External Sampling MCCFR variant.

    More efficient than vanilla CFR because it only samples one
    action for the opponent instead of traversing all actions.
    """

    def __init__(
        self,
        poker_engine: PreflopPokerEngine,
        regret_store: Optional[RegretStore] = None,
        exploration: float = 0.6
    ):
        super().__init__(poker_engine, regret_store)
        self.exploration = exploration  # Epsilon for epsilon-greedy exploration

    def _cfr_traverse(
        self,
        state: GameState,
        traversal_state: CFRTraversalState,
        traversing_player: int
    ) -> float:
        """
        External sampling CFR traverse with exploration.
        """
        self.total_traversals += 1

        if state.is_complete:
            return self._get_terminal_utility(state, traversing_player)

        current_player_idx = state.current_player_idx
        info_set = InformationSet.from_game_state(state)
        info_set_key = info_set.to_key()
        legal_mask = get_legal_actions_mask(state)

        strategy = self.regret_store.get_strategy(info_set_key, legal_mask)

        if current_player_idx == traversing_player:
            # Traverse all actions for the traversing player
            action_values = np.zeros(NUM_ACTIONS)

            for action_idx in range(NUM_ACTIONS):
                if not legal_mask[action_idx]:
                    action_values[action_idx] = 0.0
                    continue

                action = action_idx_to_action(action_idx, state)
                next_state = self.poker_engine.apply_action(state, action)

                new_traversal = traversal_state.update_reach_prob(
                    current_player_idx, strategy[action_idx]
                )

                action_values[action_idx] = self._cfr_traverse(
                    next_state, new_traversal, traversing_player
                )

            # Node value under current strategy
            node_value = (strategy * action_values).sum()

            # Regrets
            regrets = action_values - node_value

            # Update regrets (weighted by opponent reach)
            opponent_reach = traversal_state.opponent_reach_prob(traversing_player)
            self.regret_store.update_regrets(info_set_key, regrets, opponent_reach)

            # Update strategy sum
            player_reach = traversal_state.player_reach_prob(traversing_player)
            self.regret_store.update_strategy_sum(info_set_key, strategy, player_reach)

            return node_value

        else:
            # Sample action for opponent (external sampling)
            # Use epsilon-greedy for exploration
            if np.random.random() < self.exploration:
                # Explore: uniform random among legal actions
                legal_indices = np.where(legal_mask)[0]
                action_idx = np.random.choice(legal_indices)
            else:
                # Exploit: sample from strategy
                action_idx = sample_action(strategy, legal_mask)

            action = action_idx_to_action(action_idx, state)
            next_state = self.poker_engine.apply_action(state, action)

            new_traversal = traversal_state.update_reach_prob(
                current_player_idx, strategy[action_idx]
            )

            return self._cfr_traverse(next_state, new_traversal, traversing_player)


def run_cfr_training(
    num_iterations: int,
    num_players: int = 6,
    starting_stack: float = 100.0,
    exploration: float = 0.6,
    print_every: int = 100
) -> ExternalSamplingCFR:
    """
    Run tabular CFR training.

    Args:
        num_iterations: Number of CFR iterations
        num_players: Number of players
        starting_stack: Starting stack in big blinds
        exploration: Exploration rate for external sampling
        print_every: Print progress every N iterations

    Returns:
        Trained CFR engine
    """
    engine = PreflopPokerEngine(
        num_players=num_players,
        starting_stack=starting_stack
    )

    cfr = ExternalSamplingCFR(
        poker_engine=engine,
        exploration=exploration
    )

    print(f"Starting CFR training for {num_iterations} iterations...")
    print(f"Players: {num_players}, Stack: {starting_stack}bb")

    for i in range(num_iterations):
        utility = cfr.cfr_iteration()

        if (i + 1) % print_every == 0:
            print(f"Iteration {i + 1}/{num_iterations}, "
                  f"Traversals: {cfr.total_traversals}, "
                  f"Info sets: {len(cfr.regret_store.cumulative_regrets)}")

    print(f"Training complete! Total traversals: {cfr.total_traversals}")
    return cfr
