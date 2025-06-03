"""Abstract base classes for ML components.

These provide a consistent interface for extending the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for all poker models.
    
    This ensures consistent interface across different model types.
    """
    
    @abstractmethod
    def predict_action(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Predict action from features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tuple of (action, bet_size)
        """
        pass
    
    @abstractmethod
    def get_action_probabilities(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get probability distribution over actions.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        pass
    
    @abstractmethod
    def features_to_tensor(self, features: Dict[str, float]) -> torch.Tensor:
        """Convert features to tensor.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Feature tensor
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str, metadata: Optional[Dict] = None):
        """Save model to file.
        
        Args:
            filepath: Path to save model
            metadata: Optional metadata to include
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str, map_location=None) -> 'BaseModel':
        """Load model from file.
        
        Args:
            filepath: Path to load from
            map_location: Device mapping
            
        Returns:
            Loaded model instance
        """
        pass


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors.
    
    This allows easy addition of new feature extraction strategies.
    """
    
    @abstractmethod
    def extract_features(self, *args, **kwargs) -> Dict[str, float]:
        """Extract features from game state or scenario.
        
        Returns:
            Dictionary of feature names to values
        """
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces.
        
        Returns:
            List of feature names
        """
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Get dimensionality of features.
        
        Returns:
            Number of features
        """
        pass


class BaseScenarioGenerator(ABC):
    """Abstract base class for scenario generators.
    
    This allows different strategies for generating training data.
    """
    
    @abstractmethod
    def generate_scenario(self) -> Optional[Dict[str, Any]]:
        """Generate a single scenario.
        
        Returns:
            Scenario dictionary or None if generation failed
        """
        pass
    
    @abstractmethod
    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Generate a batch of scenarios.
        
        Args:
            batch_size: Number of scenarios to generate
            
        Returns:
            List of scenario dictionaries
        """
        pass
    
    @abstractmethod
    def generate_balanced_dataset(self, total_size: int) -> List[Dict[str, Any]]:
        """Generate a balanced dataset.
        
        Args:
            total_size: Total number of scenarios
            
        Returns:
            List of balanced scenarios
        """
        pass


class BaseTrainer(ABC):
    """Abstract base class for model trainers.
    
    This provides a consistent training interface.
    """
    
    @abstractmethod
    def train(self, data: List[Dict], **kwargs) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            data: Training data
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        pass
    
    @abstractmethod
    def validate(self, data: List[Dict]) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            data: Validation data
            
        Returns:
            Validation metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            metrics: Current metrics
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint
            
        Returns:
            Checkpoint data
        """
        pass