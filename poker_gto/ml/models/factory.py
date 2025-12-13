"""Factory pattern for creating models.

This makes it easy to add new model types without changing existing code.
"""

from typing import Dict, Type, Optional, Any
from pathlib import Path
import torch

from poker_gto.ml.base import BaseModel
from poker_gto.ml.models.preflop_model import PreflopGTOModel, ModelConfig


class ModelFactory:
    """Factory for creating and managing poker models."""
    
    # Registry of available models
    _models: Dict[str, Type[BaseModel]] = {
        'preflop': PreflopGTOModel,
        # Future models can be added here:
        # 'postflop': PostflopGTOModel,
        # 'multistreet': MultiStreetGTOModel,
    }
    
    # Default configurations for each model type
    _default_configs: Dict[str, Dict[str, Any]] = {
        'preflop': {
            'input_size': 20,
            'hidden_sizes': [128, 64, 32],
            'dropout_rate': 0.3,
            'output_size': 4,  # fold, call, raise, check
        }
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseModel:
        """Create a model of the specified type.
        
        Args:
            model_type: Type of model to create ('preflop', etc.)
            config: Model configuration dictionary
            **kwargs: Additional arguments passed to model constructor
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type is not recognized
        """
        if model_type not in cls._models:
            available = ', '.join(cls._models.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available types: {available}"
            )
        
        # Get model class
        model_class = cls._models[model_type]
        
        # Merge configurations
        default_config = cls._default_configs.get(model_type, {})
        if config:
            merged_config = {**default_config, **config}
        else:
            merged_config = default_config
        
        # Create appropriate config object
        if model_type == 'preflop':
            config_obj = ModelConfig(**merged_config)
            return model_class(config_obj, **kwargs)
        else:
            # For future model types
            return model_class(**merged_config, **kwargs)
    
    @classmethod
    def load_model(
        cls,
        filepath: str,
        model_type: Optional[str] = None,
        map_location=None
    ) -> BaseModel:
        """Load a model from file.
        
        Args:
            filepath: Path to model file
            model_type: Type of model (if known)
            map_location: Device mapping for loading
            
        Returns:
            Loaded model instance
            
        Raises:
            ValueError: If model type cannot be determined
        """
        # Load checkpoint to inspect
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Try to determine model type
        if model_type is None:
            # Look for type hint in metadata
            metadata = checkpoint.get('metadata', {})
            model_type = metadata.get('model_type')
            
            # If not found, try to infer from architecture
            if model_type is None:
                state_dict = checkpoint.get('model_state_dict', {})
                if any('network' in key for key in state_dict.keys()):
                    model_type = 'preflop'  # Default assumption
        
        if model_type is None:
            raise ValueError(
                "Could not determine model type from file. "
                "Please specify model_type parameter."
            )
        
        # Get model class
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._models[model_type]
        
        # Load using model's load method
        return model_class.load(filepath, map_location=map_location)
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]):
        """Register a new model type.
        
        This allows extending the factory without modifying this file.
        
        Args:
            name: Name for the model type
            model_class: Model class (must inherit from BaseModel)
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"{model_class} must inherit from BaseModel")
        
        cls._models[name] = model_class
    
    @classmethod
    def list_available_models(cls) -> list[str]:
        """Get list of available model types.
        
        Returns:
            List of model type names
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """Get information about a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with model information
        """
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._models[model_type]
        default_config = cls._default_configs.get(model_type, {})
        
        return {
            'name': model_type,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'default_config': default_config,
            'docstring': model_class.__doc__,
        }