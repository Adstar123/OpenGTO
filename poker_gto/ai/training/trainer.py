import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from poker_gto.ai.models.preflop_model import PreflopGTOModel

class PreflopTrainer:
    """Trainer for the preflop GTO model."""
    
    def __init__(self, model: PreflopGTOModel, learning_rate: float = 0.001, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizers and loss functions
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        
        # Loss functions
        self.action_loss_fn = nn.CrossEntropyLoss()
        self.sizing_loss_fn = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self, train_data: List[Dict], val_data: List[Dict], 
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train the model on training data."""
        
        # Prepare data loaders
        train_loader = self._prepare_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._prepare_dataloader(val_data, batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_accuracy'].append(val_acc)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            # Logging
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}/{epochs}")
                self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                self.logger.info(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return {
            'best_val_loss': best_val_loss,
            'training_history': self.training_history,
            'final_train_acc': self.training_history['train_accuracy'][-1],
            'final_val_acc': self.training_history['val_accuracy'][-1]
        }
    
    def _train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_features, batch_actions, batch_sizes in dataloader:
            batch_features = batch_features.to(self.device)
            batch_actions = batch_actions.to(self.device)
            batch_sizes = batch_sizes.to(self.device)
            
            # Forward pass
            action_probs, raise_sizes = self.model(batch_features)
            
            # Calculate losses
            action_loss = self.action_loss_fn(action_probs, batch_actions)
            sizing_loss = self.sizing_loss_fn(raise_sizes.squeeze(), batch_sizes)
            
            # Combined loss (weight sizing loss less since it's only for raise actions)
            total_loss_batch = action_loss + 0.3 * sizing_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += total_loss_batch.item()
            predictions = torch.argmax(action_probs, dim=1)
            correct_predictions += (predictions == batch_actions).sum().item()
            total_samples += batch_actions.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_features, batch_actions, batch_sizes in dataloader:
                batch_features = batch_features.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_sizes = batch_sizes.to(self.device)
                
                # Forward pass
                action_probs, raise_sizes = self.model(batch_features)
                
                # Calculate losses
                action_loss = self.action_loss_fn(action_probs, batch_actions)
                sizing_loss = self.sizing_loss_fn(raise_sizes.squeeze(), batch_sizes)
                total_loss_batch = action_loss + 0.3 * sizing_loss
                
                # Track metrics
                total_loss += total_loss_batch.item()
                predictions = torch.argmax(action_probs, dim=1)
                correct_predictions += (predictions == batch_actions).sum().item()
                total_samples += batch_actions.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _prepare_dataloader(self, data: List[Dict], batch_size: int, shuffle: bool) -> DataLoader:
        """Prepare PyTorch DataLoader from training data."""
        features = []
        actions = []
        sizes = []
        
        action_map = {'fold': 0, 'call': 1, 'raise': 2}
        
        for sample in data:
            # Extract features
            feature_vector = self.model._features_to_tensor(sample['features'])
            features.append(feature_vector)
            
            # Extract target action
            action_label = action_map[sample['optimal_action']['action']]
            actions.append(action_label)
            
            # Extract raise size (normalized to 0-1)
            raise_size = sample['optimal_action'].get('size', 0.0)
            # Normalize raise size: 2BB-5BB -> 0-1
            normalized_size = max(0.0, min(1.0, (raise_size - 2.0) / 3.0))
            sizes.append(normalized_size)
        
        # Convert to tensors
        features_tensor = torch.stack(features)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        sizes_tensor = torch.tensor(sizes, dtype=torch.float32)
        
        dataset = TensorDataset(features_tensor, actions_tensor, sizes_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def save_model(self, filepath: str, metadata: Dict = None):
        """Save model and training metadata."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_sizes': self.model.hidden_sizes
            },
            'training_history': self.training_history,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Dict:
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        self.logger.info(f"Model loaded from {filepath}")
        return checkpoint.get('metadata', {})