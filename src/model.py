"""
Neural network model for GTO preflop strategy learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreflopStrategyNet(nn.Module):
    """
    Neural network for learning GTO preflop poker strategies.

    Architecture:
    - Input: 15 features (position, hand encoding, stack, action history)
    - Hidden layers: 256 -> 128 -> 64
    - Output: 7 action probabilities (softmax)
    """

    def __init__(self, input_size: int = 15, hidden_sizes: list = None,
                 output_size: int = 7, dropout_rate: float = 0.2):
        """
        Initialize the network.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output actions
            dropout_rate: Dropout probability for regularization
        """
        super(PreflopStrategyNet, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        self.input_size = input_size
        self.output_size = output_size

        # Build layers dynamically
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size) with action probabilities
        """
        logits = self.network(x)
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        return probs


class PreflopStrategyNetV2(nn.Module):
    """
    Enhanced version with separate embeddings for categorical features.

    Architecture:
    - Position embedding: 2 -> 8
    - Rank embeddings: 13 -> 16 (for each rank)
    - Action history embedding: 8 -> 16 (for each action in history)
    - Concatenate all embeddings + continuous features
    - Feed through deep network
    """

    def __init__(self, output_size: int = 7, dropout_rate: float = 0.3):
        """
        Initialize the enhanced network.

        Args:
            output_size: Number of output actions
            dropout_rate: Dropout probability
        """
        super(PreflopStrategyNetV2, self).__init__()

        # Embeddings
        self.position_embed = nn.Embedding(2, 8)  # 2 positions -> 8 dims
        self.rank_embed = nn.Embedding(13, 16)  # 13 ranks -> 16 dims
        self.action_embed = nn.Embedding(8, 16)  # 7 actions + padding -> 16 dims

        # Calculate total embedding size
        # position: 8, rank1: 16, rank2: 16, is_suited: 1, stack: 1
        # action_history (10 actions * 16): 160
        # Total: 8 + 16 + 16 + 1 + 1 + 160 = 202
        embedding_size = 8 + 16 + 16 + 1 + 1 + 160

        # Deep network
        self.fc1 = nn.Linear(embedding_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with embeddings.

        Args:
            x: Input tensor of shape (batch_size, 15)
               [position, rank1, rank2, is_suited, stack, action_history (10)]

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Extract features
        position = x[:, 0].long()
        rank1 = x[:, 1].long()
        rank2 = x[:, 2].long()
        is_suited = x[:, 3:4]
        stack = x[:, 4:5]
        action_history = x[:, 5:15].long()

        # Handle negative values in action history (padding)
        action_history = torch.clamp(action_history, min=0)

        # Embed categorical features
        pos_emb = self.position_embed(position)  # (batch, 8)
        rank1_emb = self.rank_embed(rank1)  # (batch, 16)
        rank2_emb = self.rank_embed(rank2)  # (batch, 16)

        # Embed action history
        action_emb = self.action_embed(action_history)  # (batch, 10, 16)
        action_emb = action_emb.view(action_emb.size(0), -1)  # (batch, 160)

        # Concatenate all features
        features = torch.cat([
            pos_emb, rank1_emb, rank2_emb, is_suited, stack, action_emb
        ], dim=1)

        # Deep network
        x = F.relu(self.bn1(self.fc1(features)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        logits = self.fc_out(x)
        probs = F.softmax(logits, dim=1)

        return probs
