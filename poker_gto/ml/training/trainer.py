"""Training utilities for poker models."""

from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from poker_gto.ml.models.preflop_model import PreflopGTOModel, ModelConfig
from poker_gto.ml.data.scenario_generator import PreflopScenarioGenerator


class PreflopTrainer:
    """Trainer for preflop GTO models.
    
    Handles training loop, validation, and model checkpointing.
    """
    
    def __init__(
        self,
        model: Optional[PreflopGTOModel] = None,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train. If None, creates default model.
            device: Device to train on. If None, uses CUDA if available.
            logger: Logger for training messages. If None, creates one.
        """
        self.model = model or PreflopGTOModel()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Training state
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'action_accuracies': []
        }
    
    def train(
        self,
        scenarios: List[Dict],
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        val_split: float = 0.2,
        patience: int = 20,
        save_dir: Optional[Path] = None
    ) -> Dict:
        """Train the model on scenarios.
        
        Args:
            scenarios: List of training scenarios
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            val_split: Validation split ratio
            patience: Early stopping patience
            save_dir: Directory to save best model
            
        Returns:
            Dictionary with training results
        """
        # Split data
        train_data, val_data = train_test_split(
            scenarios, test_size=val_split, random_state=42
        )
        self.logger.info(f"Training: {len(train_data)}, Validation: {len(val_data)}")
        
        # Create data loaders
        train_loader = self._create_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_data, batch_size, shuffle=False)
        
        # Setup training
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        criterion = self._get_criterion(train_data)
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train epoch
            train_metrics = self._train_epoch(
                train_loader, optimizer, criterion
            )
            
            # Validate
            val_metrics = self._validate_epoch(
                val_loader, criterion
            )
            
            # Update learning rate
            scheduler.step()
            
            # Track history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['action_accuracies'].append(val_metrics['action_accuracies'])
            
            # Check if all actions are being learned
            min_action_acc = min(val_metrics['action_accuracies'].values())
            all_actions_learned = all(
                acc > 0.1 for acc in val_metrics['action_accuracies'].values()
            )
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc and all_actions_learned:
                self.best_val_acc = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                if save_dir:
                    self._save_checkpoint(save_dir, epoch, val_metrics)
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 10 == 0 or epoch < 5:
                self._log_epoch_results(epoch, epochs, train_metrics, val_metrics)
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(f"Loaded best model with validation accuracy: {self.best_val_acc:.4f}")
        
        return {
            'best_val_acc': self.best_val_acc,
            'final_epoch': epoch,
            'training_history': self.training_history
        }
    
    def _create_dataloader(
        self,
        data: List[Dict],
        batch_size: int,
        shuffle: bool
    ) -> DataLoader:
        """Create DataLoader from scenarios."""
        features = []
        labels = []
        action_to_idx = {'fold': 0, 'call': 1, 'raise': 2, 'check': 3}
        
        for sample in data:
            feature_tensor = self.model.features_to_tensor(sample['features'])
            features.append(feature_tensor)
            labels.append(action_to_idx[sample['optimal_action']['action']])
        
        features_tensor = torch.stack(features)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        dataset = TensorDataset(features_tensor, labels_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _get_criterion(self, train_data: List[Dict]) -> nn.Module:
        """Get loss criterion with class balancing."""
        # Count actions
        action_to_idx = {'fold': 0, 'call': 1, 'raise': 2, 'check': 3}
        action_counts = [0, 0, 0, 0]
        
        for sample in train_data:
            action_idx = action_to_idx[sample['optimal_action']['action']]
            action_counts[action_idx] += 1
        
        # Calculate weights
        total_samples = sum(action_counts)
        class_weights = [
            total_samples / (4 * count) if count > 0 else 1.0
            for count in action_counts
        ]
        
        weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        return nn.CrossEntropyLoss(weight=weights_tensor)
    
    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': correct / total
        }
    
    def _validate_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module
    ) -> Dict:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Track per-action accuracy
        action_correct = {'fold': 0, 'call': 0, 'raise': 0, 'check': 0}
        action_total = {'fold': 0, 'call': 0, 'raise': 0, 'check': 0}
        action_names = ['fold', 'call', 'raise', 'check']
        
        with torch.no_grad():
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                # Per-action accuracy
                for true_label, pred_label in zip(batch_labels.cpu(), predicted.cpu()):
                    true_action = action_names[true_label.item()]
                    action_total[true_action] += 1
                    if true_label == pred_label:
                        action_correct[true_action] += 1
        
        # Calculate per-action accuracies
        action_accuracies = {}
        for action in action_correct:
            if action_total[action] > 0:
                action_accuracies[action] = action_correct[action] / action_total[action]
            else:
                action_accuracies[action] = 0.0
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': correct / total,
            'action_accuracies': action_accuracies
        }
    
    def _log_epoch_results(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict,
        val_metrics: Dict
    ):
        """Log results for an epoch."""
        self.logger.info(f"Epoch {epoch}/{total_epochs}")
        self.logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        self.logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        self.logger.info(f"  Action Acc: {val_metrics['action_accuracies']}")
        
        min_action_acc = min(val_metrics['action_accuracies'].values())
        self.logger.info(f"  Min Action Acc: {min_action_acc:.4f}")
    
    def _save_checkpoint(self, save_dir: Path, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint_path = save_dir / f"checkpoint_epoch{epoch}_{timestamp}.pth"
        
        self.model.save(
            str(checkpoint_path),
            metadata={
                'epoch': epoch,
                'val_accuracy': metrics['accuracy'],
                'val_loss': metrics['loss'],
                'action_accuracies': metrics['action_accuracies'],
                'timestamp': timestamp
            }
        )