"""
Improved preflop model with better architecture and feature handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np

class ImprovedPreflopGTOModel(nn.Module):
    """Enhanced neural network for preflop GTO decisions with class balancing."""
    
    def __init__(self, input_size: int = 23, hidden_sizes: List[int] = [256, 128, 64, 32]):
        super(ImprovedPreflopGTOModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Feature extraction layers with residual connections
        layers = []
        current_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Action probability head with more capacity
        self.action_head = nn.Sequential(
            nn.Linear(current_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # fold, call, raise
            # Note: No softmax here, will be applied in forward or loss function
        )
        
        # Raise sizing head
        self.sizing_head = nn.Sequential(
            nn.Linear(current_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 0-1 range, will map to actual sizes
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using proper techniques."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and raise sizing."""
        # Extract features
        features = self.feature_layers(x)
        
        # Get action logits (not probabilities)
        action_logits = self.action_head(features)
        
        # Get raise sizing
        raise_sizing = self.sizing_head(features)
        
        return action_logits, raise_sizing
    
    def predict_action(self, game_features: Dict) -> Tuple[str, float]:
        """Predict action from game features with proper device handling."""
        self.eval()
        
        # Convert features to tensor
        feature_vector = self._features_to_tensor(game_features)
        
        # Move to same device as model
        device = next(self.parameters()).device
        feature_vector = feature_vector.to(device)
        
        with torch.no_grad():
            action_logits, raise_sizing = self.forward(feature_vector.unsqueeze(0))
            
            # Apply softmax to get probabilities
            action_probs = F.softmax(action_logits, dim=1)
            
            # Get most likely action
            action_idx = torch.argmax(action_probs, dim=1).item()
            actions = ['fold', 'call', 'raise']
            predicted_action = actions[action_idx]
            
            # Get confidence
            confidence = action_probs[0, action_idx].item()
            
            # If raise, get sizing
            if predicted_action == 'raise':
                size_multiplier = raise_sizing.item()
                # Map 0-1 to reasonable raise sizes (2.2BB to 4.5BB)
                raise_size = 2.2 + (size_multiplier * 2.3)
                return predicted_action, raise_size
            
            return predicted_action, 0.0
    
    def get_action_probabilities(self, game_features: Dict) -> Dict[str, float]:
        """Get probabilities for all actions."""
        self.eval()
        
        feature_vector = self._features_to_tensor(game_features)
        device = next(self.parameters()).device
        feature_vector = feature_vector.to(device)
        
        with torch.no_grad():
            action_logits, _ = self.forward(feature_vector.unsqueeze(0))
            action_probs = F.softmax(action_logits, dim=1)
            
            return {
                'fold': action_probs[0, 0].item(),
                'call': action_probs[0, 1].item(),
                'raise': action_probs[0, 2].item()
            }
    
    def _features_to_tensor(self, features: Dict) -> torch.Tensor:
        """Convert feature dictionary to tensor with proper ordering."""
        
        # Define expected feature order (must match training data)
        feature_names = [
            # Position features (6)
            'position_UTG', 'position_MP', 'position_CO', 
            'position_BTN', 'position_SB', 'position_BB',
            
            # Hand features (6)
            'is_pocket_pair', 'is_suited', 'hand_strength', 
            'is_premium', 'is_strong', 'is_playable',
            
            # Action context features (6)
            'facing_raise', 'raises_before_me', 'calls_before_me', 
            'folds_before_me', 'last_raiser_position', 'bet_size_ratio',
            
            # Pot and stack context (5)
            'pot_size_bb', 'current_bet_bb', 'pot_odds', 
            'effective_stack', 'stack_to_pot'
        ]
        
        # Extract values in correct order, with defaults for missing
        values = []
        for name in feature_names:
            value = features.get(name, 0.0)
            # Ensure value is numeric and handle any edge cases
            if isinstance(value, (int, float)):
                values.append(float(value))
            else:
                values.append(0.0)
        
        # Verify we have the right number of features
        if len(values) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(values)}")
        
        return torch.tensor(values, dtype=torch.float32)