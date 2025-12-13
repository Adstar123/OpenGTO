"""Logging utilities for OpenGTO."""

import logging
from pathlib import Path
from datetime import datetime
import sys


def setup_logging(
    name: str = "opengto",
    log_dir: Path = Path("logs"),
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to log to console
        file: Whether to log to file
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """Specialized logger for training progress."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize training logger.
        
        Args:
            logger: Base logger to use
        """
        self.logger = logger
        self.start_time = None
    
    def start_training(self):
        """Log training start."""
        self.start_time = datetime.now()
        self.logger.info("="*60)
        self.logger.info("Training Started")
        self.logger.info("="*60)
    
    def end_training(self, success: bool = True):
        """Log training end.
        
        Args:
            success: Whether training was successful
        """
        if self.start_time:
            duration = datetime.now() - self.start_time
            self.logger.info("="*60)
            if success:
                self.logger.info(f"Training Completed Successfully")
            else:
                self.logger.info(f"Training Failed")
            self.logger.info(f"Total Duration: {duration}")
            self.logger.info("="*60)
    
    def log_phase(self, phase: str, description: str):
        """Log a training phase.
        
        Args:
            phase: Phase name
            description: Phase description
        """
        self.logger.info("")
        self.logger.info(f"{'='*10} {phase} {'='*10}")
        self.logger.info(description)
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        action_acc: dict = None
    ):
        """Log epoch summary in a formatted way.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            action_acc: Per-action accuracies
        """
        self.logger.info(f"\nEpoch {epoch} Summary:")
        self.logger.info(f"  Training   | Loss: {train_loss:6.4f} | Acc: {train_acc:6.2%}")
        self.logger.info(f"  Validation | Loss: {val_loss:6.4f} | Acc: {val_acc:6.2%}")
        
        if action_acc:
            acc_str = " | ".join([f"{k}: {v:.2%}" for k, v in action_acc.items()])
            self.logger.info(f"  Actions    | {acc_str}")