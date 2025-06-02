"""
Comprehensive trainer with advanced techniques for class imbalance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn.functional as F
import logging
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            ce_loss = ce_loss * at
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """Label smoothing for better generalization."""
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))

class ComprehensiveGTOTrainer:
    """Advanced trainer with multiple techniques for balanced learning."""
    
    def __init__(self, model, device=None, config=None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Default config
        self.config = config or {
            'learning_rate': 0.002,
            'weight_decay': 1e-4,
            'focal_gamma': 2.0,
            'label_smoothing': 0.05,
            'use_focal_loss': True,
            'use_weighted_sampling': True,
            'use_class_weights': True
        }
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # Loss functions will be set up in train method
        self.action_loss_fn = None
        self.sizing_loss_fn = nn.SmoothL1Loss()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_action_acc': [], 'val_action_acc': []
        }
        
        self.logger = logging.getLogger(__name__)
    
    def train(self, train_data: List[Dict], epochs: int = 200, 
              batch_size: int = 128, validation_split: float = 0.15) -> Dict:
        """Train with comprehensive techniques."""
        
        # Split data
        if validation_split > 0:
            train_scenarios, val_scenarios = train_test_split(
                train_data, test_size=validation_split, random_state=42,
                stratify=[s['optimal_action']['action'] for s in train_data]
            )
        else:
            train_scenarios = train_data
            val_scenarios = []
        
        self.logger.info(f"Training samples: {len(train_scenarios)}")
        self.logger.info(f"Validation samples: {len(val_scenarios)}")
        
        # Analyze class distribution
        train_actions = [s['optimal_action']['action'] for s in train_scenarios]
        action_counts = {action: train_actions.count(action) for action in ['fold', 'call', 'raise']}
        self.logger.info(f"Training action distribution: {action_counts}")
        
        # Setup loss functions with class balancing
        self._setup_loss_functions(train_actions)
        
        # Create data loaders
        train_loader = self._create_dataloader(train_scenarios, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_scenarios, batch_size, shuffle=False) if val_scenarios else None
        
        # Training loop
        best_balanced_score = 0.0
        best_model_state = None
        patience_counter = 0
        patience = 30
        
        self.logger.info(f"Starting training on {self.device}")
        
        for epoch in range(epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader) if val_loader else {'loss': 0, 'acc': 0, 'action_acc': {'fold': 0, 'call': 0, 'raise': 0}}
            
            # Update scheduler
            self.scheduler.step()
            
            # Calculate balanced score (geometric mean of action accuracies)
            val_action_accs = list(val_metrics['action_acc'].values())
            balanced_score = np.prod([acc for acc in val_action_accs if acc > 0]) ** (1/3) if all(acc > 0 for acc in val_action_accs) else 0
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['train_action_acc'].append(train_metrics['action_acc'])
            self.history['val_action_acc'].append(val_metrics['action_acc'])
            
            # Save best model based on balanced score
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 10 == 0 or epoch < 10:
                self.logger.info(f"Epoch {epoch}/{epochs}")
                self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
                self.logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}")
                self.logger.info(f"Action Accuracies - Train: {train_metrics['action_acc']}")
                self.logger.info(f"Action Accuracies - Val: {val_metrics['action_acc']}")
                self.logger.info(f"Balanced Score: {balanced_score:.4f}")
                self.logger.info(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Loaded best model with balanced score: {best_balanced_score:.4f}")
        
        return {
            'best_balanced_score': best_balanced_score,
            'history': self.history,
            'final_train_acc': self.history['train_acc'][-1],
            'final_val_acc': self.history['val_acc'][-1]
        }
    
    def _setup_loss_functions(self, train_actions: List[str]):
        """Setup loss functions with class balancing."""
        
        # Convert actions to labels
        action_to_label = {'fold': 0, 'call': 1, 'raise': 2}
        labels = [action_to_label[action] for action in train_actions]
        
        if self.config['use_class_weights']:
            # Compute class weights
            class_weights = compute_class_weight(
                'balanced', classes=np.unique(labels), y=labels
            )
            class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
            self.logger.info(f"Class weights: {dict(zip(['fold', 'call', 'raise'], class_weights))}")
        else:
            class_weights_tensor = None
        
        if self.config['use_focal_loss']:
            self.action_loss_fn = FocalLoss(
                alpha=class_weights_tensor,
                gamma=self.config['focal_gamma']
            )
            self.logger.info("Using Focal Loss")
        else:
            if self.config['label_smoothing'] > 0:
                self.action_loss_fn = LabelSmoothingLoss(
                    num_classes=3,
                    smoothing=self.config['label_smoothing']
                )
                self.logger.info("Using Label Smoothing Loss")
            else:
                self.action_loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
                self.logger.info("Using Weighted Cross Entropy Loss")
    
    def _create_dataloader(self, data: List[Dict], batch_size: int, shuffle: bool) -> DataLoader:
        """Create DataLoader with optional weighted sampling."""
        
        features, actions, sizes = [], [], []
        action_to_label = {'fold': 0, 'call': 1, 'raise': 2}
        
        for sample in data:
            # Features
            feature_vector = self.model._features_to_tensor(sample['features'])
            features.append(feature_vector)
            
            # Action label
            action_label = action_to_label[sample['optimal_action']['action']]
            actions.append(action_label)
            
            # Raise size (normalized)
            raise_size = sample['optimal_action'].get('size', 0.0)
            normalized_size = max(0.0, min(1.0, (raise_size - 2.2) / 2.3))
            sizes.append(normalized_size)
        
        # Convert to tensors
        features_tensor = torch.stack(features)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        sizes_tensor = torch.tensor(sizes, dtype=torch.float32)
        
        dataset = TensorDataset(features_tensor, actions_tensor, sizes_tensor)
        
        # Use weighted sampling if enabled
        if shuffle and self.config['use_weighted_sampling']:
            # Calculate sample weights
            class_counts = torch.bincount(actions_tensor)
            class_weights = 1.0 / class_counts.float()
            sample_weights = class_weights[actions_tensor]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True
            )
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train for one epoch with detailed metrics."""
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Per-action tracking
        action_correct = {'fold': 0, 'call': 0, 'raise': 0}
        action_total = {'fold': 0, 'call': 0, 'raise': 0}
        action_map = {0: 'fold', 1: 'call', 2: 'raise'}
        
        for batch_features, batch_actions, batch_sizes in dataloader:
            batch_features = batch_features.to(self.device)
            batch_actions = batch_actions.to(self.device)
            batch_sizes = batch_sizes.to(self.device)
            
            # Forward pass
            action_logits, raise_sizes = self.model(batch_features)
            
            # Calculate losses
            action_loss = self.action_loss_fn(action_logits, batch_actions)
            sizing_loss = self.sizing_loss_fn(raise_sizes.squeeze(), batch_sizes)
            
            # Combined loss (heavily weight action prediction)
            total_loss_batch = action_loss + 0.05 * sizing_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += total_loss_batch.item()
            predictions = torch.argmax(action_logits, dim=1)
            correct = (predictions == batch_actions).sum().item()
            correct_predictions += correct
            total_samples += batch_actions.size(0)
            
            # Per-action accuracy
            for true_label, pred_label in zip(batch_actions.cpu(), predictions.cpu()):
                true_action = action_map[true_label.item()]
                action_total[true_action] += 1
                if true_label == pred_label:
                    action_correct[true_action] += 1
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        action_accuracies = {}
        for action in action_correct:
            if action_total[action] > 0:
                action_accuracies[action] = action_correct[action] / action_total[action]
            else:
                action_accuracies[action] = 0.0
        
        return {
            'loss': avg_loss,
            'acc': accuracy,
            'action_acc': action_accuracies
        }
    
    def _validate_epoch(self, dataloader: DataLoader) -> Dict:
        """Validate for one epoch."""
        if not dataloader:
            return {'loss': 0, 'acc': 0, 'action_acc': {'fold': 0, 'call': 0, 'raise': 0}}
        
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        action_correct = {'fold': 0, 'call': 0, 'raise': 0}
        action_total = {'fold': 0, 'call': 0, 'raise': 0}
        action_map = {0: 'fold', 1: 'call', 2: 'raise'}
        
        with torch.no_grad():
            for batch_features, batch_actions, batch_sizes in dataloader:
                batch_features = batch_features.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_sizes = batch_sizes.to(self.device)
                
                action_logits, raise_sizes = self.model(batch_features)
                
                action_loss = self.action_loss_fn(action_logits, batch_actions)
                sizing_loss = self.sizing_loss_fn(raise_sizes.squeeze(), batch_sizes)
                total_loss_batch = action_loss + 0.05 * sizing_loss
                
                total_loss += total_loss_batch.item()
                predictions = torch.argmax(action_logits, dim=1)
                correct_predictions += (predictions == batch_actions).sum().item()
                total_samples += batch_actions.size(0)
                
                for true_label, pred_label in zip(batch_actions.cpu(), predictions.cpu()):
                    true_action = action_map[true_label.item()]
                    action_total[true_action] += 1
                    if true_label == pred_label:
                        action_correct[true_action] += 1
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        action_accuracies = {}
        for action in action_correct:
            if action_total[action] > 0:
                action_accuracies[action] = action_correct[action] / action_total[action]
            else:
                action_accuracies[action] = 0.0
        
        return {
            'loss': avg_loss,
            'acc': accuracy,
            'action_acc': action_accuracies
        }
    
    def save_model(self, filepath: str):
        """Save model and training state."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_sizes': self.model.hidden_sizes
            },
            'training_config': self.config,
            'training_history': self.history
        }
        
        torch.save(save_dict, filepath)
        self.logger.info(f"Model saved to {filepath}")