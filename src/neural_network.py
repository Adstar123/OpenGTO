"""
Neural Network architectures for Deep CFR.

Two networks:
1. Strategy Network - Outputs action probabilities given game state
2. Value Network - Outputs expected values for regret estimation

These networks learn EVERYTHING from self-play. No poker knowledge is encoded here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class StrategyNetwork(nn.Module):
    """
    Neural network that outputs action probabilities.

    Input: Game state feature vector
    Output: Probability distribution over actions

    The network learns which actions are good in which situations
    purely from CFR regret updates during self-play.
    """

    def __init__(
        self,
        input_size: int = 317,  # From GameState.feature_size()
        hidden_sizes: Tuple[int, ...] = (512, 512, 256),
        num_actions: int = 6,  # fold, check, call, bet, raise, all-in
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions

        # Build hidden layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        self.hidden = nn.Sequential(*layers)

        # Output layer - logits for each action
        self.output = nn.Linear(prev_size, num_actions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        legal_actions_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_size]
            legal_actions_mask: Boolean mask of legal actions [batch_size, num_actions]
                               True = legal, False = illegal

        Returns:
            Action probabilities [batch_size, num_actions]
        """
        # Pass through hidden layers
        hidden = self.hidden(x)

        # Get logits
        logits = self.output(hidden)

        # Mask illegal actions with large negative value
        if legal_actions_mask is not None:
            # Set illegal actions to -inf so they get 0 probability after softmax
            logits = logits.masked_fill(~legal_actions_mask, float('-inf'))

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        return probs

    def get_action_probs(
        self,
        state_features: np.ndarray,
        legal_actions_mask: np.ndarray
    ) -> np.ndarray:
        """
        Convenience method to get action probabilities from numpy arrays.
        """
        self.eval()
        with torch.no_grad():
            # Get device from network parameters
            device = next(self.parameters()).device
            x = torch.FloatTensor(state_features).unsqueeze(0).to(device)
            mask = torch.BoolTensor(legal_actions_mask).unsqueeze(0).to(device)
            probs = self.forward(x, mask)
            return probs.squeeze(0).cpu().numpy()


class ValueNetwork(nn.Module):
    """
    Neural network that estimates expected values / counterfactual values.

    Input: Game state feature vector
    Output: Expected value for each action (used for regret calculation)

    This is used in Deep CFR to approximate the cumulative regrets
    without having to store them for every information set.
    """

    def __init__(
        self,
        input_size: int = 317,
        hidden_sizes: Tuple[int, ...] = (512, 512, 256),
        num_actions: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions

        # Build hidden layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        self.hidden = nn.Sequential(*layers)

        # Output layer - value for each action
        self.output = nn.Linear(prev_size, num_actions)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_size]

        Returns:
            Action values [batch_size, num_actions]
        """
        hidden = self.hidden(x)
        values = self.output(hidden)
        return values

    def get_action_values(self, state_features: np.ndarray) -> np.ndarray:
        """
        Convenience method to get action values from numpy arrays.
        """
        self.eval()
        with torch.no_grad():
            # Get device from network parameters
            device = next(self.parameters()).device
            x = torch.FloatTensor(state_features).unsqueeze(0).to(device)
            values = self.forward(x)
            return values.squeeze(0).cpu().numpy()


class RegretNetwork(nn.Module):
    """
    Neural network for approximating cumulative regrets in Deep CFR.

    This network is trained to predict the cumulative regret for each action
    given a game state. The regrets are then used via regret matching to
    compute the strategy.
    """

    def __init__(
        self,
        input_size: int = 317,
        hidden_sizes: Tuple[int, ...] = (512, 512, 256),
        num_actions: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, num_actions)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_size]

        Returns:
            Regret predictions [batch_size, num_actions]
        """
        hidden = self.hidden(x)
        regrets = self.output(hidden)
        return regrets

    def get_strategy(
        self,
        state_features: np.ndarray,
        legal_actions_mask: np.ndarray
    ) -> np.ndarray:
        """
        Get strategy via regret matching from predicted regrets.

        Args:
            state_features: Game state features
            legal_actions_mask: Boolean mask of legal actions

        Returns:
            Action probabilities via regret matching
        """
        self.eval()
        with torch.no_grad():
            # Get device from network parameters
            device = next(self.parameters()).device
            x = torch.FloatTensor(state_features).unsqueeze(0).to(device)
            regrets = self.forward(x).squeeze(0).cpu().numpy()

            # Apply regret matching
            return regret_matching(regrets, legal_actions_mask)


class AverageStrategyNetwork(nn.Module):
    """
    Neural network for approximating the average strategy in Deep CFR.

    The average strategy converges to Nash equilibrium as training progresses.
    This network is trained on the weighted average of strategies played
    throughout training.
    """

    def __init__(
        self,
        input_size: int = 317,
        hidden_sizes: Tuple[int, ...] = (512, 512, 256),
        num_actions: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, num_actions)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        legal_actions_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, input_size]
            legal_actions_mask: Boolean mask of legal actions

        Returns:
            Action probabilities [batch_size, num_actions]
        """
        hidden = self.hidden(x)
        logits = self.output(hidden)

        if legal_actions_mask is not None:
            logits = logits.masked_fill(~legal_actions_mask, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        return probs

    def get_action_probs(
        self,
        state_features: np.ndarray,
        legal_actions_mask: np.ndarray
    ) -> np.ndarray:
        """Get action probabilities from numpy arrays."""
        self.eval()
        with torch.no_grad():
            # Get device from network parameters
            device = next(self.parameters()).device
            x = torch.FloatTensor(state_features).unsqueeze(0).to(device)
            mask = torch.BoolTensor(legal_actions_mask).unsqueeze(0).to(device)
            probs = self.forward(x, mask)
            return probs.squeeze(0).cpu().numpy()


def regret_matching(regrets: np.ndarray, legal_actions_mask: np.ndarray) -> np.ndarray:
    """
    Convert regrets to strategy via regret matching.

    Regret matching:
    - Take positive part of regrets
    - Normalize to get probabilities
    - If all regrets are non-positive, use uniform distribution over legal actions

    Args:
        regrets: Array of regret values for each action
        legal_actions_mask: Boolean mask of legal actions

    Returns:
        Probability distribution over actions
    """
    # Apply legal actions mask
    masked_regrets = np.where(legal_actions_mask, regrets, -np.inf)

    # Take positive part
    positive_regrets = np.maximum(masked_regrets, 0)

    # Sum of positive regrets
    regret_sum = positive_regrets.sum()

    if regret_sum > 0:
        # Normalize by sum of positive regrets
        strategy = positive_regrets / regret_sum
    else:
        # Uniform over legal actions
        num_legal = legal_actions_mask.sum()
        strategy = np.where(legal_actions_mask, 1.0 / num_legal, 0.0)

    return strategy


class DeepCFRNetworks:
    """
    Container for all networks used in Deep CFR.

    This manages the regret network (for current strategy) and
    average strategy network (for final GTO strategy).
    """

    def __init__(
        self,
        input_size: int = 317,
        hidden_sizes: Tuple[int, ...] = (512, 512, 256),
        num_actions: int = 6,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.input_size = input_size
        self.num_actions = num_actions

        # Network for predicting regrets (used during traversal)
        self.regret_net = RegretNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_actions=num_actions
        ).to(self.device)

        # Network for average strategy (converges to Nash)
        self.avg_strategy_net = AverageStrategyNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_actions=num_actions
        ).to(self.device)

        # Optimizers
        self.regret_optimizer = torch.optim.Adam(
            self.regret_net.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        self.avg_strategy_optimizer = torch.optim.Adam(
            self.avg_strategy_net.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )

    def get_strategy(
        self,
        state_features: np.ndarray,
        legal_actions_mask: np.ndarray
    ) -> np.ndarray:
        """Get current strategy via regret matching on predicted regrets."""
        return self.regret_net.get_strategy(state_features, legal_actions_mask)

    def get_average_strategy(
        self,
        state_features: np.ndarray,
        legal_actions_mask: np.ndarray
    ) -> np.ndarray:
        """Get average strategy (the GTO approximation)."""
        return self.avg_strategy_net.get_action_probs(state_features, legal_actions_mask)

    def train_regret_net(
        self,
        states: torch.Tensor,
        target_regrets: torch.Tensor,
        iterations: torch.Tensor
    ) -> float:
        """
        Train regret network on collected samples.

        Args:
            states: Batch of state features [batch_size, input_size]
            target_regrets: Target regret values [batch_size, num_actions]
            iterations: CFR iteration when sample was collected [batch_size]

        Returns:
            Training loss
        """
        self.regret_net.train()

        # Weight samples by iteration (linear CFR weighting)
        weights = iterations.float()
        weights = weights / weights.sum()

        # Forward pass
        predicted_regrets = self.regret_net(states)

        # MSE loss weighted by iteration
        loss = (weights.unsqueeze(1) * (predicted_regrets - target_regrets) ** 2).sum()

        # Backward pass
        self.regret_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.regret_net.parameters(), 1.0)
        self.regret_optimizer.step()

        return loss.item()

    def train_avg_strategy_net(
        self,
        states: torch.Tensor,
        target_strategies: torch.Tensor,
        legal_masks: torch.Tensor,
        iterations: torch.Tensor
    ) -> float:
        """
        Train average strategy network on collected samples.

        Args:
            states: Batch of state features [batch_size, input_size]
            target_strategies: Target strategy probabilities [batch_size, num_actions]
            legal_masks: Legal action masks [batch_size, num_actions]
            iterations: CFR iteration when sample was collected [batch_size]

        Returns:
            Training loss
        """
        self.avg_strategy_net.train()

        # Weight by iteration
        weights = iterations.float()
        weights = weights / weights.sum()

        # Forward pass
        predicted_probs = self.avg_strategy_net(states, legal_masks)

        # Cross-entropy loss weighted by iteration
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        loss = -(weights.unsqueeze(1) * target_strategies * torch.log(predicted_probs + eps)).sum()

        # Backward pass
        self.avg_strategy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.avg_strategy_net.parameters(), 1.0)
        self.avg_strategy_optimizer.step()

        return loss.item()

    def save(self, path: str):
        """Save networks to file."""
        torch.save({
            'regret_net': self.regret_net.state_dict(),
            'avg_strategy_net': self.avg_strategy_net.state_dict(),
            'regret_optimizer': self.regret_optimizer.state_dict(),
            'avg_strategy_optimizer': self.avg_strategy_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load networks from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.regret_net.load_state_dict(checkpoint['regret_net'])
        self.avg_strategy_net.load_state_dict(checkpoint['avg_strategy_net'])
        self.regret_optimizer.load_state_dict(checkpoint['regret_optimizer'])
        self.avg_strategy_optimizer.load_state_dict(checkpoint['avg_strategy_optimizer'])
