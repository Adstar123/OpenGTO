"""Configuration management for OpenGTO."""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import json
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Data generation
    num_scenarios: int = 15000
    player_counts: list = None
    stack_sizes: list = None
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    val_split: float = 0.2
    patience: int = 20
    
    # Model parameters
    input_size: int = 20
    hidden_sizes: list = None
    dropout_rate: float = 0.3
    
    # System
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    seed: int = 42
    
    def __post_init__(self):
        if self.player_counts is None:
            self.player_counts = [6]
        if self.stack_sizes is None:
            self.stack_sizes = [100.0]
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64, 32]
        
        # Auto-detect device
        if self.device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class TestingConfig:
    """Configuration for model testing."""
    num_test_scenarios: int = 100
    show_examples: int = 20
    interactive_mode: bool = True


class ConfigManager:
    """Manages configuration loading and saving."""
    
    @staticmethod
    def load_config(config_path: Path, config_class=TrainingConfig) -> Any:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            config_class: Dataclass to load into
            
        Returns:
            Configuration instance
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load based on extension
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return config_class(**config_dict)
    
    @staticmethod
    def save_config(config: Any, config_path: Path):
        """Save configuration to file.
        
        Args:
            config: Configuration instance (dataclass)
            config_path: Path to save configuration
        """
        config_dict = asdict(config)
        
        # Save based on extension
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    @staticmethod
    def get_default_config_path() -> Path:
        """Get default configuration path."""
        return Path("configs/training_config.yaml")
    
    @staticmethod
    def create_default_configs(base_dir: Path = Path("configs")):
        """Create default configuration files.
        
        Args:
            base_dir: Directory to create configs in
        """
        base_dir.mkdir(exist_ok=True)
        
        # Training config
        training_config = TrainingConfig()
        ConfigManager.save_config(
            training_config,
            base_dir / "training_config.yaml"
        )
        
        # Testing config
        testing_config = TestingConfig()
        ConfigManager.save_config(
            testing_config,
            base_dir / "testing_config.yaml"
        )
        
        # Model config
        model_config = {
            "input_size": 20,
            "hidden_sizes": [128, 64, 32],
            "dropout_rate": 0.3,
            "output_size": 3
        }
        
        with open(base_dir / "model_config.yaml", 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)