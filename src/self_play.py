"""
Self-Play Game Simulation for Training.

This module handles efficient generation of training data through
self-play games. Supports parallel game simulation for faster training.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict
import time

from .game_state import GameState, Action, ActionType, Position
from .poker_engine import PreflopPokerEngine
from .card import HoleCards, Deck
from .information_set import (
    InformationSet, get_legal_actions_mask, action_idx_to_action,
    sample_action, NUM_ACTIONS
)
from .showdown import run_out_board


@dataclass
class GameResult:
    """Result of a single game."""
    states: List[GameState]
    actions: List[Action]
    utilities: Dict[int, float]  # player_idx -> utility
    winner_idx: Optional[int]
    went_to_showdown: bool


@dataclass
class TraversalResult:
    """Result of a CFR traversal."""
    regret_samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]  # (features, regrets, mask, iter)
    strategy_samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]]  # (features, strategy, mask, iter, reach)
    utility: float
    num_nodes: int


class ParallelGameSimulator:
    """
    Simulates multiple games in parallel for efficient data collection.
    """

    def __init__(
        self,
        num_players: int = 6,
        starting_stack: float = 100.0,
        num_workers: int = 4
    ):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.num_workers = num_workers

        # Create engine for each worker
        self.engines = [
            PreflopPokerEngine(num_players=num_players, starting_stack=starting_stack)
            for _ in range(num_workers)
        ]

    def simulate_random_games(self, num_games: int) -> List[GameResult]:
        """
        Simulate multiple games with random actions.
        Useful for initial exploration.
        """
        results = []

        for i in range(num_games):
            engine = self.engines[i % self.num_workers]
            result = self._simulate_single_random_game(engine)
            results.append(result)

        return results

    def _simulate_single_random_game(self, engine: PreflopPokerEngine) -> GameResult:
        """Simulate one game with random actions."""
        state = engine.create_game()
        states = [state]
        actions = []

        while not state.is_complete:
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                break

            action = np.random.choice(legal_actions)
            actions.append(action)
            state = engine.apply_action(state, action)
            states.append(state)

        # Compute utilities
        utilities = {}
        active_players = [p for p in state.players if p.is_active]

        for i, player in enumerate(state.players):
            if not player.is_active:
                utilities[i] = -player.current_bet
            elif len(active_players) == 1:
                if player.is_active:
                    utilities[i] = state.pot - player.current_bet
                else:
                    utilities[i] = -player.current_bet
            else:
                # Would need showdown
                utilities[i] = 0.0  # Placeholder

        winner_idx = None
        if len(active_players) == 1:
            winner_idx = state.players.index(active_players[0])

        return GameResult(
            states=states,
            actions=actions,
            utilities=utilities,
            winner_idx=winner_idx,
            went_to_showdown=len(active_players) > 1
        )

    def simulate_with_strategy(
        self,
        strategy_fn: Callable[[GameState], np.ndarray],
        num_games: int
    ) -> List[GameResult]:
        """
        Simulate games using a strategy function.

        Args:
            strategy_fn: Function that takes GameState and returns action probabilities
            num_games: Number of games to simulate

        Returns:
            List of game results
        """
        results = []

        for i in range(num_games):
            engine = self.engines[i % self.num_workers]
            result = self._simulate_with_strategy(engine, strategy_fn)
            results.append(result)

        return results

    def _simulate_with_strategy(
        self,
        engine: PreflopPokerEngine,
        strategy_fn: Callable[[GameState], np.ndarray]
    ) -> GameResult:
        """Simulate one game using strategy function."""
        state = engine.create_game()
        states = [state]
        actions = []

        while not state.is_complete:
            legal_mask = get_legal_actions_mask(state)

            # Get strategy
            strategy = strategy_fn(state)

            # Sample action
            action_idx = sample_action(strategy, legal_mask)
            action = action_idx_to_action(action_idx, state)

            actions.append(action)
            state = engine.apply_action(state, action)
            states.append(state)

        # Compute utilities
        utilities = self._compute_utilities(state)

        active_players = [p for p in state.players if p.is_active]
        winner_idx = None
        if len(active_players) == 1:
            winner_idx = state.players.index(active_players[0])

        return GameResult(
            states=states,
            actions=actions,
            utilities=utilities,
            winner_idx=winner_idx,
            went_to_showdown=len(active_players) > 1
        )

    def _compute_utilities(self, state: GameState) -> Dict[int, float]:
        """Compute utilities for all players at terminal state."""
        utilities = {}
        active_players = [p for p in state.players if p.is_active]

        if len(active_players) == 1:
            # Single winner
            for i, player in enumerate(state.players):
                if player.is_active:
                    utilities[i] = state.pot - player.current_bet
                else:
                    utilities[i] = -player.current_bet
        else:
            # Multiple players - compute equity
            hole_cards_list = [p.hole_cards for p in active_players]
            equities = run_out_board(hole_cards_list, num_simulations=100)

            active_idx = 0
            for i, player in enumerate(state.players):
                if player.is_active:
                    utilities[i] = equities[active_idx] * state.pot - player.current_bet
                    active_idx += 1
                else:
                    utilities[i] = -player.current_bet

        return utilities


class BatchedCFRTraversal:
    """
    Batched CFR traversal for more efficient training.

    Instead of traversing one game at a time, this collects samples
    from multiple games to create larger batches for neural network training.
    """

    def __init__(
        self,
        poker_engine: PreflopPokerEngine,
        strategy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        exploration: float = 0.6
    ):
        self.poker_engine = poker_engine
        self.strategy_fn = strategy_fn
        self.exploration = exploration

    def collect_batch(
        self,
        num_traversals: int,
        iteration: int
    ) -> Tuple[List, List]:
        """
        Collect a batch of samples through CFR traversals.

        Args:
            num_traversals: Number of game traversals
            iteration: Current CFR iteration

        Returns:
            Tuple of (regret_samples, strategy_samples)
        """
        regret_samples = []
        strategy_samples = []

        for _ in range(num_traversals):
            # Create new game
            state = self.poker_engine.create_game()

            # Random traversing player
            traversing_player = np.random.randint(0, self.poker_engine.num_players)

            # Initialize reach probabilities
            reach_probs = {i: 1.0 for i in range(self.poker_engine.num_players)}

            # Traverse
            r_samples, s_samples, _ = self._traverse(
                state, reach_probs, traversing_player, iteration
            )

            regret_samples.extend(r_samples)
            strategy_samples.extend(s_samples)

        return regret_samples, strategy_samples

    def _traverse(
        self,
        state: GameState,
        reach_probs: Dict[int, float],
        traversing_player: int,
        iteration: int
    ) -> Tuple[List, List, float]:
        """
        Single CFR traverse returning samples and utility.
        """
        if state.is_complete:
            utility = self._terminal_utility(state, traversing_player)
            return [], [], utility

        current_player = state.current_player_idx
        info_set = InformationSet.from_game_state(state)
        features = info_set.to_feature_vector()
        legal_mask = get_legal_actions_mask(state)

        # Get strategy
        strategy = self.strategy_fn(features, legal_mask)

        regret_samples = []
        strategy_samples = []

        if current_player == traversing_player:
            # Compute action values
            action_values = np.zeros(NUM_ACTIONS)

            for action_idx in range(NUM_ACTIONS):
                if not legal_mask[action_idx]:
                    continue

                action = action_idx_to_action(action_idx, state)
                next_state = self.poker_engine.apply_action(state, action)

                new_reach = reach_probs.copy()
                new_reach[current_player] *= strategy[action_idx]

                r_samp, s_samp, value = self._traverse(
                    next_state, new_reach, traversing_player, iteration
                )

                action_values[action_idx] = value
                regret_samples.extend(r_samp)
                strategy_samples.extend(s_samp)

            # Node value
            node_value = (strategy * action_values).sum()

            # Regrets
            regrets = action_values - node_value

            # Compute opponent reach probability
            opponent_reach = 1.0
            for p, rp in reach_probs.items():
                if p != traversing_player:
                    opponent_reach *= rp

            # Store samples
            if opponent_reach > 1e-6:
                regret_samples.append((features, regrets, legal_mask, iteration))

            player_reach = reach_probs.get(traversing_player, 1.0)
            if player_reach > 1e-6:
                strategy_samples.append((features, strategy, legal_mask, iteration, player_reach))

            return regret_samples, strategy_samples, node_value

        else:
            # Opponent - sample action
            if np.random.random() < self.exploration:
                legal_indices = np.where(legal_mask)[0]
                action_idx = np.random.choice(legal_indices)
            else:
                action_idx = sample_action(strategy, legal_mask)

            action = action_idx_to_action(action_idx, state)
            next_state = self.poker_engine.apply_action(state, action)

            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[action_idx]

            return self._traverse(next_state, new_reach, traversing_player, iteration)

    def _terminal_utility(self, state: GameState, player_idx: int) -> float:
        """Compute utility at terminal state."""
        player = state.players[player_idx]

        if not player.is_active:
            return -player.current_bet

        active_players = [p for p in state.players if p.is_active]

        if len(active_players) == 1:
            return state.pot - player.current_bet

        # Multiple players - compute equity
        hole_cards_list = [p.hole_cards for p in active_players]
        equities = run_out_board(hole_cards_list, num_simulations=50)

        active_idx = None
        for i, p in enumerate(active_players):
            if p == player:
                active_idx = i
                break

        if active_idx is None:
            return -player.current_bet

        return equities[active_idx] * state.pot - player.current_bet


class SelfPlayDataGenerator:
    """
    Generates training data through self-play.

    This is the main interface for collecting training samples
    using the current strategy networks.
    """

    def __init__(
        self,
        poker_engine: PreflopPokerEngine,
        exploration: float = 0.6
    ):
        self.poker_engine = poker_engine
        self.exploration = exploration
        self.games_played = 0
        self.total_samples = 0

    def generate_samples(
        self,
        strategy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        num_traversals: int,
        iteration: int
    ) -> Tuple[List, List]:
        """
        Generate training samples through CFR traversals.

        Args:
            strategy_fn: Function (features, legal_mask) -> action_probs
            num_traversals: Number of game traversals
            iteration: Current CFR iteration

        Returns:
            Tuple of (regret_samples, strategy_samples)
        """
        traverser = BatchedCFRTraversal(
            poker_engine=self.poker_engine,
            strategy_fn=strategy_fn,
            exploration=self.exploration
        )

        regret_samples, strategy_samples = traverser.collect_batch(
            num_traversals=num_traversals,
            iteration=iteration
        )

        self.games_played += num_traversals
        self.total_samples += len(regret_samples) + len(strategy_samples)

        return regret_samples, strategy_samples

    def get_stats(self) -> Dict[str, int]:
        """Get data generation statistics."""
        return {
            'games_played': self.games_played,
            'total_samples': self.total_samples
        }
