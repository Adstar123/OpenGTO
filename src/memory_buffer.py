"""
Memory buffers for Deep CFR training.

Stores experience tuples collected during CFR traversals for training
the neural networks. Uses reservoir sampling to maintain a fixed-size
buffer while giving appropriate weight to samples from later iterations.
"""
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
import numpy as np
import random
from collections import deque


class RegretSample(NamedTuple):
    """Sample for training the regret network."""
    state_features: np.ndarray  # Information set features
    regrets: np.ndarray  # Regret values for each action
    legal_mask: np.ndarray  # Legal actions mask
    iteration: int  # CFR iteration when collected


class StrategySample(NamedTuple):
    """Sample for training the average strategy network."""
    state_features: np.ndarray  # Information set features
    strategy: np.ndarray  # Strategy probabilities
    legal_mask: np.ndarray  # Legal actions mask
    iteration: int  # CFR iteration when collected
    reach_prob: float  # Player's reach probability


class ReservoirBuffer:
    """
    Reservoir sampling buffer for maintaining a fixed-size sample
    of experiences with linear CFR weighting.

    Linear CFR weights later iterations more heavily, which helps
    convergence. The reservoir maintains this weighting property.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List = []
        self.total_weight = 0.0
        self.num_added = 0

    def add(self, sample, weight: float = 1.0):
        """
        Add a sample to the buffer with given weight.

        Uses weighted reservoir sampling to maintain proper distribution.
        """
        self.num_added += 1
        self.total_weight += weight

        if len(self.buffer) < self.capacity:
            self.buffer.append((sample, weight))
        else:
            # Weighted reservoir sampling
            # Probability of keeping new sample
            p = weight * self.capacity / self.total_weight

            if random.random() < p:
                # Replace a random existing sample
                idx = random.randint(0, self.capacity - 1)
                self.buffer[idx] = (sample, weight)

    def sample_batch(self, batch_size: int) -> Tuple[List, np.ndarray]:
        """
        Sample a batch from the buffer.

        Returns:
            Tuple of (samples, weights) where weights are normalized
        """
        if len(self.buffer) == 0:
            return [], np.array([])

        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)

        samples = [self.buffer[i][0] for i in indices]
        weights = np.array([self.buffer[i][1] for i in indices])

        # Normalize weights
        weights = weights / weights.sum()

        return samples, weights

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.total_weight = 0.0
        self.num_added = 0


class DeepCFRMemory:
    """
    Memory management for Deep CFR training.

    Maintains separate buffers for:
    - Regret samples (for training regret network)
    - Strategy samples (for training average strategy network)
    """

    def __init__(
        self,
        regret_buffer_size: int = 2_000_000,
        strategy_buffer_size: int = 2_000_000
    ):
        self.regret_buffer = ReservoirBuffer(regret_buffer_size)
        self.strategy_buffer = ReservoirBuffer(strategy_buffer_size)

    def add_regret_sample(
        self,
        state_features: np.ndarray,
        regrets: np.ndarray,
        legal_mask: np.ndarray,
        iteration: int
    ):
        """Add a regret sample with linear CFR weighting."""
        sample = RegretSample(
            state_features=state_features.copy(),
            regrets=regrets.copy(),
            legal_mask=legal_mask.copy(),
            iteration=iteration
        )
        # Linear weighting: weight = iteration
        self.regret_buffer.add(sample, weight=float(iteration))

    def add_strategy_sample(
        self,
        state_features: np.ndarray,
        strategy: np.ndarray,
        legal_mask: np.ndarray,
        iteration: int,
        reach_prob: float
    ):
        """Add a strategy sample with linear CFR weighting."""
        sample = StrategySample(
            state_features=state_features.copy(),
            strategy=strategy.copy(),
            legal_mask=legal_mask.copy(),
            iteration=iteration,
            reach_prob=reach_prob
        )
        # Weight by iteration * reach probability
        weight = float(iteration) * reach_prob
        self.strategy_buffer.add(sample, weight=max(weight, 1e-6))

    def sample_regret_batch(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of regret samples.

        Returns:
            Tuple of (state_features, regrets, legal_masks, iterations)
        """
        samples, weights = self.regret_buffer.sample_batch(batch_size)

        if len(samples) == 0:
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([])
            )

        state_features = np.stack([s.state_features for s in samples])
        regrets = np.stack([s.regrets for s in samples])
        legal_masks = np.stack([s.legal_mask for s in samples])
        iterations = np.array([s.iteration for s in samples])

        return state_features, regrets, legal_masks, iterations

    def sample_strategy_batch(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of strategy samples.

        Returns:
            Tuple of (state_features, strategies, legal_masks, iterations)
        """
        samples, weights = self.strategy_buffer.sample_batch(batch_size)

        if len(samples) == 0:
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([])
            )

        state_features = np.stack([s.state_features for s in samples])
        strategies = np.stack([s.strategy for s in samples])
        legal_masks = np.stack([s.legal_mask for s in samples])
        iterations = np.array([s.iteration for s in samples])

        return state_features, strategies, legal_masks, iterations

    @property
    def num_regret_samples(self) -> int:
        return len(self.regret_buffer)

    @property
    def num_strategy_samples(self) -> int:
        return len(self.strategy_buffer)

    def clear(self):
        """Clear all buffers."""
        self.regret_buffer.clear()
        self.strategy_buffer.clear()


class TrainingStats:
    """Track training statistics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.regret_losses = deque(maxlen=window_size)
        self.strategy_losses = deque(maxlen=window_size)
        self.utilities = deque(maxlen=window_size)
        self.traversals_per_iter = deque(maxlen=window_size)

    def add_regret_loss(self, loss: float):
        self.regret_losses.append(loss)

    def add_strategy_loss(self, loss: float):
        self.strategy_losses.append(loss)

    def add_utility(self, utility: float):
        self.utilities.append(utility)

    def add_traversals(self, count: int):
        self.traversals_per_iter.append(count)

    @property
    def avg_regret_loss(self) -> float:
        if len(self.regret_losses) == 0:
            return 0.0
        return sum(self.regret_losses) / len(self.regret_losses)

    @property
    def avg_strategy_loss(self) -> float:
        if len(self.strategy_losses) == 0:
            return 0.0
        return sum(self.strategy_losses) / len(self.strategy_losses)

    @property
    def avg_utility(self) -> float:
        if len(self.utilities) == 0:
            return 0.0
        return sum(self.utilities) / len(self.utilities)

    def get_summary(self) -> str:
        return (
            f"Regret Loss: {self.avg_regret_loss:.4f}, "
            f"Strategy Loss: {self.avg_strategy_loss:.4f}, "
            f"Avg Utility: {self.avg_utility:.4f}"
        )
