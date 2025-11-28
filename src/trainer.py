"""
Training utilities for the GTO strategy neural network.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple
import os


class GTOTrainer:
    """
    Trainer class for GTO strategy neural network.
    """

    def __init__(self, model: nn.Module, learning_rate: float = 0.001,
                 device: str = None):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Use KL divergence loss since we're learning probability distributions
        self.criterion = nn.KLDivLoss(reduction='batchmean')

        # Adam optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Track training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for features, targets in train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)

            # KL divergence requires log probabilities as input
            loss = self.criterion(torch.log(outputs + 1e-10), targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * features.size(0)

            # Calculate accuracy (top action matches)
            pred_actions = torch.argmax(outputs, dim=1)
            target_actions = torch.argmax(targets, dim=1)
            correct_predictions += (pred_actions == target_actions).sum().item()
            total_predictions += features.size(0)

        avg_loss = total_loss / total_predictions
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(torch.log(outputs + 1e-10), targets)

                total_loss += loss.item() * features.size(0)

                pred_actions = torch.argmax(outputs, dim=1)
                target_actions = torch.argmax(targets, dim=1)
                correct_predictions += (pred_actions == target_actions).sum().item()
                total_predictions += features.size(0)

        avg_loss = total_loss / total_predictions
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, early_stopping_patience: int = 10,
              checkpoint_dir: str = 'models') -> Dict:
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save model checkpoints

        Returns:
            Training history dictionary
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0

        print("="*80)
        print("Starting Training")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print("="*80)

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.6f}, Val Acc:   {val_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                }, checkpoint_path)
                print(f"  Saved best model (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

            print()

        print("="*80)
        print("Training Complete")
        print("="*80)
        print(f"Best validation loss: {best_val_loss:.6f}")

        return self.history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the model on test data.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dictionary with evaluation metrics
        """
        test_loss, test_acc = self.validate(test_loader)

        # Calculate KL divergence statistics
        self.model.eval()
        kl_divs = []

        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(features)

                # Calculate KL divergence for each sample
                batch_kl = nn.functional.kl_div(
                    torch.log(outputs + 1e-10),
                    targets,
                    reduction='none'
                ).sum(dim=1)

                kl_divs.extend(batch_kl.cpu().numpy())

        kl_divs = np.array(kl_divs)

        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'mean_kl_divergence': np.mean(kl_divs),
            'median_kl_divergence': np.median(kl_divs),
            'max_kl_divergence': np.max(kl_divs)
        }

        return results
