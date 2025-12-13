"""
Deep Counterfactual Regret Minimization (Deep CFR) implementation.

Deep CFR uses neural networks to approximate:
1. Cumulative regrets (regret network)
2. Average strategy (strategy network)

This allows CFR to scale to games with large state spaces by
generalizing across similar situations rather than storing
values for every information set.

Reference: "Deep Counterfactual Regret Minimization" (Brown et al., 2019)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time
import os

from .game_state import GameState, Action, ActionType
from .poker_engine import PreflopPokerEngine
from .information_set import (
    InformationSet, get_legal_actions_mask, action_idx_to_action,
    sample_action, CFRTraversalState, NUM_ACTIONS
)
from .neural_network import DeepCFRNetworks, regret_matching
from .memory_buffer import DeepCFRMemory, TrainingStats
from .showdown import run_out_board


class DeepCFRTrainer:
    """
    Deep CFR trainer that learns GTO strategy through self-play.

    The training process:
    1. Run CFR traversals collecting regret and strategy samples
    2. Train regret network on collected regrets
    3. Train average strategy network on collected strategies
    4. Repeat

    The average strategy network converges to Nash equilibrium.
    """

    def __init__(
        self,
        num_players: int = 6,
        starting_stack: float = 100.0,
        device: str = 'cpu',
        regret_buffer_size: int = 2_000_000,
        strategy_buffer_size: int = 2_000_000,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        exploration: float = 0.6
    ):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.device = device
        self.exploration = exploration

        # Poker engine
        self.poker_engine = PreflopPokerEngine(
            num_players=num_players,
            starting_stack=starting_stack
        )

        # Neural networks
        input_size = InformationSet.feature_size()
        self.networks = DeepCFRNetworks(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_actions=NUM_ACTIONS,
            device=device
        )

        # Memory buffers
        self.memory = DeepCFRMemory(
            regret_buffer_size=regret_buffer_size,
            strategy_buffer_size=strategy_buffer_size
        )

        # Training statistics
        self.stats = TrainingStats()

        # Iteration counter
        self.iteration = 0
        self.total_traversals = 0

    def train(
        self,
        num_iterations: int,
        traversals_per_iter: int = 1000,
        batch_size: int = 2048,
        train_steps_per_iter: int = 100,
        print_every: int = 10,
        save_every: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Run Deep CFR training.

        Args:
            num_iterations: Number of CFR iterations
            traversals_per_iter: Number of game traversals per iteration
            batch_size: Batch size for neural network training
            train_steps_per_iter: Number of training steps per iteration
            print_every: Print progress every N iterations
            save_every: Save checkpoint every N iterations
            save_path: Path to save checkpoints
        """
        print(f"Starting Deep CFR training")
        print(f"  Players: {self.num_players}")
        print(f"  Stack: {self.starting_stack}bb")
        print(f"  Device: {self.device}")
        print(f"  Iterations: {num_iterations}")
        print()

        start_time = time.time()

        for i in range(num_iterations):
            self.iteration += 1
            iter_start = time.time()

            # Collect samples through CFR traversals
            traversals_before = self.total_traversals
            utilities = self._collect_samples(traversals_per_iter)
            traversals_this_iter = self.total_traversals - traversals_before

            self.stats.add_traversals(traversals_this_iter)
            for u in utilities:
                self.stats.add_utility(u)

            # Train neural networks
            if len(self.memory.regret_buffer) >= batch_size:
                for _ in range(train_steps_per_iter):
                    regret_loss = self._train_regret_network(batch_size)
                    strategy_loss = self._train_strategy_network(batch_size)

                    self.stats.add_regret_loss(regret_loss)
                    self.stats.add_strategy_loss(strategy_loss)

            # Print progress
            if (i + 1) % print_every == 0:
                elapsed = time.time() - start_time
                iter_time = time.time() - iter_start

                print(f"Iteration {i + 1}/{num_iterations}")
                print(f"  Time: {elapsed:.1f}s total, {iter_time:.2f}s/iter")
                print(f"  Traversals: {self.total_traversals:,}")
                print(f"  Regret samples: {self.memory.num_regret_samples:,}")
                print(f"  Strategy samples: {self.memory.num_strategy_samples:,}")
                print(f"  {self.stats.get_summary()}")
                print()

            # Save checkpoint
            if save_path and (i + 1) % save_every == 0:
                self.save(f"{save_path}_iter{i + 1}.pt")

        print(f"Training complete!")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Total traversals: {self.total_traversals:,}")

        if save_path:
            self.save(f"{save_path}_final.pt")

    def _collect_samples(self, num_traversals: int) -> List[float]:
        """
        Run CFR traversals and collect samples.

        Returns:
            List of utilities from each traversal
        """
        utilities = []

        for _ in range(num_traversals):
            # Create new game
            state = self.poker_engine.create_game()

            # Randomly select traversing player
            traversing_player = np.random.randint(0, self.num_players)

            # Run traversal
            traversal_state = CFRTraversalState(
                reach_probs={i: 1.0 for i in range(self.num_players)},
                iteration=self.iteration,
                traversing_player=traversing_player
            )

            utility = self._cfr_traverse(state, traversal_state, traversing_player)
            utilities.append(utility)

        return utilities

    def _cfr_traverse(
        self,
        state: GameState,
        traversal_state: CFRTraversalState,
        traversing_player: int
    ) -> float:
        """
        Deep CFR traversal.

        Uses neural network to get strategy, collects regret and strategy
        samples for training.
        """
        self.total_traversals += 1

        # Terminal state
        if state.is_complete:
            return self._get_terminal_utility(state, traversing_player)

        current_player_idx = state.current_player_idx
        info_set = InformationSet.from_game_state(state)
        features = info_set.to_feature_vector()
        legal_mask = get_legal_actions_mask(state)

        # Get strategy from regret network
        strategy = self.networks.get_strategy(features, legal_mask)

        if current_player_idx == traversing_player:
            # Compute counterfactual values for each action
            action_values = np.zeros(NUM_ACTIONS)

            for action_idx in range(NUM_ACTIONS):
                if not legal_mask[action_idx]:
                    continue

                action = action_idx_to_action(action_idx, state)
                next_state = self.poker_engine.apply_action(state, action)

                new_traversal = traversal_state.update_reach_prob(
                    current_player_idx, strategy[action_idx]
                )

                action_values[action_idx] = self._cfr_traverse(
                    next_state, new_traversal, traversing_player
                )

            # Node value
            node_value = (strategy * action_values).sum()

            # Regrets
            regrets = action_values - node_value

            # Store regret sample
            opponent_reach = traversal_state.opponent_reach_prob(traversing_player)
            if opponent_reach > 1e-6:
                self.memory.add_regret_sample(
                    state_features=features,
                    regrets=regrets,
                    legal_mask=legal_mask,
                    iteration=self.iteration
                )

            # Store strategy sample
            player_reach = traversal_state.player_reach_prob(traversing_player)
            if player_reach > 1e-6:
                self.memory.add_strategy_sample(
                    state_features=features,
                    strategy=strategy,
                    legal_mask=legal_mask,
                    iteration=self.iteration,
                    reach_prob=player_reach
                )

            return node_value

        else:
            # Opponent - sample action with exploration
            if np.random.random() < self.exploration:
                # Explore: uniform random
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

    def _get_terminal_utility(self, state: GameState, player_idx: int) -> float:
        """Compute utility at terminal state."""
        player = state.players[player_idx]

        if not player.is_active:
            return -player.current_bet

        active_players = [p for p in state.players if p.is_active]

        if len(active_players) == 1:
            if active_players[0] == player:
                return state.pot - player.current_bet
            return -player.current_bet

        # Multiple players - compute equity
        hole_cards_list = [p.hole_cards for p in active_players]
        equities = run_out_board(hole_cards_list, num_simulations=50)

        # Find player's index among active players
        active_idx = None
        for i, p in enumerate(active_players):
            if p == player:
                active_idx = i
                break

        if active_idx is None:
            return -player.current_bet

        expected_winnings = equities[active_idx] * state.pot
        return expected_winnings - player.current_bet

    def _train_regret_network(self, batch_size: int) -> float:
        """Train regret network on a batch of samples."""
        states, regrets, legal_masks, iterations = \
            self.memory.sample_regret_batch(batch_size)

        if len(states) == 0:
            return 0.0

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        regrets_t = torch.FloatTensor(regrets).to(self.device)
        iterations_t = torch.FloatTensor(iterations).to(self.device)

        loss = self.networks.train_regret_net(states_t, regrets_t, iterations_t)
        return loss

    def _train_strategy_network(self, batch_size: int) -> float:
        """Train average strategy network on a batch of samples."""
        states, strategies, legal_masks, iterations = \
            self.memory.sample_strategy_batch(batch_size)

        if len(states) == 0:
            return 0.0

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        strategies_t = torch.FloatTensor(strategies).to(self.device)
        legal_masks_t = torch.BoolTensor(legal_masks).to(self.device)
        iterations_t = torch.FloatTensor(iterations).to(self.device)

        loss = self.networks.train_avg_strategy_net(
            states_t, strategies_t, legal_masks_t, iterations_t
        )
        return loss

    def get_strategy(self, state: GameState) -> np.ndarray:
        """
        Get GTO strategy for a game state.

        This uses the average strategy network which converges to Nash equilibrium.
        """
        info_set = InformationSet.from_game_state(state)
        features = info_set.to_feature_vector()
        legal_mask = get_legal_actions_mask(state)

        return self.networks.get_average_strategy(features, legal_mask)

    def get_action_probs(
        self,
        hole_cards_str: str,
        position: str,
        action_history: List[Tuple[str, str, float]],
        stack_size: float = 100.0
    ) -> Dict[str, float]:
        """
        Get action probabilities for a specific scenario.

        Convenience method for the trainer interface.

        Args:
            hole_cards_str: Hole cards like "AsKs"
            position: Position like "BTN", "BB", etc.
            action_history: List of (position, action, amount) tuples
            stack_size: Stack size in big blinds

        Returns:
            Dictionary mapping action names to probabilities
        """
        from .card import HoleCards
        from .game_state import Position, ActionType

        # Build scenario
        from .poker_engine import ScenarioBuilder

        builder = ScenarioBuilder(self.poker_engine)

        # Parse position
        pos = Position[position.upper()]

        # Set stacks
        stacks = [stack_size] * self.num_players
        builder.set_stacks(stacks)

        # Set hero cards
        hole_cards = HoleCards.from_string(hole_cards_str)
        builder.set_hero_cards(pos, hole_cards)

        # Add action history
        for pos_str, action_str, amount in action_history:
            action_pos = Position[pos_str.upper()]
            action_type = ActionType[action_str.upper()]
            action = Action(action_type, amount=amount)
            builder.add_action(action_pos, action)

        # Build state
        state = builder.build()

        # Get strategy
        strategy = self.get_strategy(state)
        legal_mask = get_legal_actions_mask(state)

        # Convert to named dict
        action_names = ['fold', 'check', 'call', 'bet', 'raise', 'all-in']
        result = {}
        for i, name in enumerate(action_names):
            if legal_mask[i]:
                result[name] = float(strategy[i])

        return result

    def save(self, path: str):
        """Save trainer state."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'networks': {
                'regret_net': self.networks.regret_net.state_dict(),
                'avg_strategy_net': self.networks.avg_strategy_net.state_dict(),
            },
            'iteration': self.iteration,
            'total_traversals': self.total_traversals,
            'config': {
                'num_players': self.num_players,
                'starting_stack': self.starting_stack,
            }
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """Load trainer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.networks.regret_net.load_state_dict(checkpoint['networks']['regret_net'])
        self.networks.avg_strategy_net.load_state_dict(checkpoint['networks']['avg_strategy_net'])
        self.iteration = checkpoint['iteration']
        self.total_traversals = checkpoint['total_traversals']
        print(f"Loaded checkpoint from {path}")
        print(f"  Iteration: {self.iteration}")
        print(f"  Traversals: {self.total_traversals}")


def train_deep_cfr(
    num_iterations: int = 1000,
    num_players: int = 6,
    starting_stack: float = 100.0,
    traversals_per_iter: int = 1000,
    batch_size: int = 2048,
    train_steps: int = 100,
    save_path: str = "checkpoints/deep_cfr",
    device: str = 'cpu'
) -> DeepCFRTrainer:
    """
    Convenience function to train Deep CFR.

    Args:
        num_iterations: Number of CFR iterations
        num_players: Number of players
        starting_stack: Starting stack in big blinds
        traversals_per_iter: Traversals per iteration
        batch_size: Training batch size
        train_steps: Training steps per iteration
        save_path: Path to save checkpoints
        device: 'cpu' or 'cuda'

    Returns:
        Trained DeepCFRTrainer
    """
    trainer = DeepCFRTrainer(
        num_players=num_players,
        starting_stack=starting_stack,
        device=device
    )

    trainer.train(
        num_iterations=num_iterations,
        traversals_per_iter=traversals_per_iter,
        batch_size=batch_size,
        train_steps_per_iter=train_steps,
        save_path=save_path
    )

    return trainer
