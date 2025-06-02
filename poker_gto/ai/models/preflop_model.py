import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np

class PreflopGTOModel(nn.Module):
    """Neural network for preflop GTO decision making."""
    
    def __init__(self, input_size: int = 23, hidden_sizes: List[int] = [128, 64, 32]):
        super(PreflopGTOModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Build neural network layers WITHOUT BatchNorm
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            current_size = hidden_size
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Action probability head (fold, call, raise)
        self.action_head = nn.Sequential(
            nn.Linear(current_size, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # fold, call, raise probabilities
            nn.Softmax(dim=1)
        )
        
        # Raise sizing head (when action is raise)
        self.sizing_head = nn.Sequential(
            nn.Linear(current_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # raise size multiplier
            nn.Sigmoid()  # Scale to 0-1, will be mapped to actual bet sizes
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action probabilities and raise sizing."""
        features = self.feature_layers(x)
        
        action_probs = self.action_head(features)
        raise_sizing = self.sizing_head(features)
        
        return action_probs, raise_sizing
    
    def predict_action(self, game_features: Dict) -> Tuple[str, float]:
        """Predict action from game features with proper device handling."""
        self.eval()
        
        # Convert features to tensor
        feature_vector = self._features_to_tensor(game_features)
        
        # Move tensor to same device as model (GPU/CPU)
        device = next(self.parameters()).device
        feature_vector = feature_vector.to(device)
        
        with torch.no_grad():
            action_probs, raise_sizing = self.forward(feature_vector.unsqueeze(0))
            
            # Get most likely action
            action_idx = torch.argmax(action_probs, dim=1).item()
            actions = ['fold', 'call', 'raise']
            predicted_action = actions[action_idx]
            
            # If raise, get sizing
            if predicted_action == 'raise':
                size_multiplier = raise_sizing.item()
                # Map to reasonable raise sizes (2BB to 5BB for preflop)
                raise_size = 2.0 + (size_multiplier * 3.0)
                return predicted_action, raise_size
            
            return predicted_action, 0.0
    
    def _features_to_tensor(self, features: Dict) -> torch.Tensor:
        """Convert feature dictionary to tensor - updated for realistic generator."""
        # NEW: 23-feature format from realistic generator
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
        
        # Extract values in order, defaulting to 0.0 for missing features
        values = [features.get(name, 0.0) for name in feature_names]
        return torch.tensor(values, dtype=torch.float32)
