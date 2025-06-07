"""Preflop GTO model implementation."""

from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np


@dataclass
class ModelConfig:
    """Configuration for preflop model."""
    input_size: int = 20
    hidden_sizes: List[int] = None
    dropout_rate: float = 0.3
    output_size: int = 4  # fold, call, raise, check
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64, 32]


class PreflopGTOModel(nn.Module):
    """Neural network for preflop poker decisions.
    
    This model takes poker game state features and outputs
    action probabilities (fold, call, raise, check).
    
    Attributes:
        config: Model configuration
        feature_names: List of expected feature names
        network: The neural network layers
    """
    
    # Define expected features for validation and documentation
    FEATURE_NAMES = [
        # Position features (6)
        'position_UTG', 'position_MP', 'position_CO', 
        'position_BTN', 'position_SB', 'position_BB',
        # Hand features (4)
        'is_pocket_pair', 'is_suited', 'hand_strength', 'high_card',
        # Context features (10)
        'facing_raise', 'pot_size', 'bet_to_call', 'pot_odds',
        'stack_ratio', 'position_strength', 'num_players',
        'premium_hand', 'strong_hand', 'playable_hand'
    ]
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the model.
        
        Args:
            config: Model configuration. If None, uses default config.
        """
        super().__init__()
        self.config = config or ModelConfig()
        
        # Validate configuration
        if self.config.input_size != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Input size {self.config.input_size} doesn't match "
                f"expected features {len(self.FEATURE_NAMES)}"
            )
        
        # Build network layers
        layers = []
        input_dim = self.config.input_size
        
        for hidden_dim in self.config.hidden_sizes:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Output layer (no dropout on final layer)
        layers.append(nn.Linear(input_dim, self.config.output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 4) with action logits
        """
        return self.network(x)
    
    def predict_action(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Predict action from feature dictionary.
        
        Args:
            features: Dictionary mapping feature names to values
            
        Returns:
            Tuple of (action_name, raise_size)
            where action_name is 'fold', 'call', 'raise', or 'check'
            and raise_size is bet size in BB (0.0 for non-raise actions)
        """
        self.eval()
        
        # Convert features to tensor
        feature_tensor = self.features_to_tensor(features)
        
        # Move to same device as model
        device = next(self.parameters()).device
        feature_tensor = feature_tensor.to(device)
        
        with torch.no_grad():
            # Get action probabilities
            logits = self.forward(feature_tensor.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            
            # Get predicted action
            action_idx = torch.argmax(probs, dim=1).item()
            action_names = ['fold', 'call', 'raise', 'check']
            predicted_action = action_names[action_idx]
            
            # Determine raise size if raising
            raise_size = 0.0
            if predicted_action == 'raise':
                # Simple raise sizing based on position and context
                position_strength = features.get('position_strength', 0.5)
                facing_raise = features.get('facing_raise', 0.0)
                
                if facing_raise > 0.5:
                    # 3-bet sizing
                    raise_size = np.random.uniform(3.0, 4.0)
                else:
                    # Open raise sizing
                    raise_size = 2.2 + (1 - position_strength) * 0.8
            
            return predicted_action, raise_size
    
    def features_to_tensor(self, features: Dict[str, float]) -> torch.Tensor:
        """Convert feature dictionary to tensor.
        
        Args:
            features: Dictionary mapping feature names to values
            
        Returns:
            Tensor of shape (input_size,) with feature values
            
        Raises:
            ValueError: If required features are missing
        """
        # Validate all required features are present
        missing_features = set(self.FEATURE_NAMES) - set(features.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract features in correct order
        values = [features[name] for name in self.FEATURE_NAMES]
        return torch.tensor(values, dtype=torch.float32)
    
    def get_action_probabilities(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get probability distribution over actions.
        
        Args:
            features: Dictionary mapping feature names to values
            
        Returns:
            Dictionary mapping action names to probabilities
        """
        self.eval()
        
        feature_tensor = self.features_to_tensor(features)
        device = next(self.parameters()).device
        feature_tensor = feature_tensor.to(device)
        
        with torch.no_grad():
            logits = self.forward(feature_tensor.unsqueeze(0))
            probs = F.softmax(logits, dim=1).squeeze(0)
            
            return {
                'fold': probs[0].item(),
                'call': probs[1].item(),
                'raise': probs[2].item(),
                'check': probs[3].item()
            }
    
    def save(self, filepath: str, metadata: Optional[Dict] = None):
        """Save model to file with metadata.
        
        Args:
            filepath: Path to save the model
            metadata: Optional metadata to save with model
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config.__dict__,
            'feature_names': self.FEATURE_NAMES,
            'metadata': metadata or {}
        }
        torch.save(save_dict, filepath)
    
    @classmethod
    def load(cls, filepath: str, map_location=None) -> 'PreflopGTOModel':
        """Load model from file.
        
        Args:
            filepath: Path to load the model from
            map_location: Device mapping for loading (e.g., 'cpu')
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Reconstruct config
        config_dict = checkpoint.get('model_config', {})
        # Handle old models with 3 outputs
        if 'output_size' not in config_dict or config_dict['output_size'] == 3:
            config_dict['output_size'] = 4
            
        config = ModelConfig(**config_dict)
        
        # Create model and load weights
        model = cls(config)
        
        # Handle old models with different output sizes
        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()
        
        # Check if we need to adapt the final layer
        for key in state_dict:
            if 'weight' in key and key in model_state:
                if state_dict[key].shape != model_state[key].shape:
                    # Old model with 3 outputs, new model with 4
                    if state_dict[key].shape[0] == 3 and model_state[key].shape[0] == 4:
                        # Initialize new weights preserving old ones
                        new_weight = model_state[key].clone()
                        new_weight[:3] = state_dict[key]
                        state_dict[key] = new_weight
                    elif 'bias' not in key:
                        # For other layers, sizes should match
                        state_dict[key] = model_state[key]
            elif 'bias' in key and key in model_state:
                if state_dict[key].shape != model_state[key].shape:
                    if state_dict[key].shape[0] == 3 and model_state[key].shape[0] == 4:
                        new_bias = model_state[key].clone()
                        new_bias[:3] = state_dict[key]
                        state_dict[key] = new_bias
        
        model.load_state_dict(state_dict, strict=False)
        
        return model