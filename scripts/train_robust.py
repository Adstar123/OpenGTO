"""
Simplified, robust training script that focuses on getting basic learning working.
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Adds the  project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split


# Simple generator and model imports
from poker_gto.core.game_state import GameState, GameConfig
from poker_gto.core.position import Position, PositionManager


def setup_logging() -> logging.Logger:
    """Setup simple logging without emojis."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"robust_training_{timestamp}.log"
    
    # Remove emojis and fancy characters
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class SimpleGTOModel(nn.Module):
    """Simple preflop model."""
    
    def __init__(self, input_size: int = 20):
        super(SimpleGTOModel, self).__init__()
        self.input_size = input_size
        
        # Simple but effective architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # fold, call, raise
        )
        
        # Initialise weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
    def predict_action(self, features: Dict) -> Tuple[str, float]:
        """Simple prediction function."""
        self.eval()
        feature_tensor = self._features_to_tensor(features)
        
        with torch.no_grad():
            logits = self.forward(feature_tensor.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            action_idx = torch.argmax(probs, dim=1).item()
            
            actions = ['fold', 'call', 'raise']
            predicted_action = actions[action_idx]
            
            # Simple raise sizing
            raise_size = random.uniform(2.2, 3.5) if predicted_action == 'raise' else 0.0
            
            return predicted_action, raise_size
    
    
    def _features_to_tensor(self, features: Dict) -> torch.Tensor:
        """Convert features to tensor."""
        # Simple feature order
        feature_names = [
            # Position (6)
            'position_UTG', 'position_MP', 'position_CO', 
            'position_BTN', 'position_SB', 'position_BB',
            # Hand (4)
            'is_pocket_pair', 'is_suited', 'hand_strength', 'high_card',
            # Context (10)
            'facing_raise', 'pot_size', 'bet_to_call', 'pot_odds',
            'stack_ratio', 'position_strength', 'num_players',
            'premium_hand', 'strong_hand', 'playable_hand'
        ]
        
        values = [features.get(name, 0.0) for name in feature_names]
        return torch.tensor(values, dtype=torch.float32)


def generate_simple_scenario() -> Dict:
    """Generate a single, simple poker scenario."""
    
    # Create game
    config = GameConfig(player_count=6, starting_stack=100.0)
    game_state = GameState(config)
    game_state.deal_hole_cards()
    
    # Pick random position
    positions = PositionManager.get_positions_for_player_count(6)
    position = random.choice(positions)
    player = game_state.get_player_by_position(position)
    
    if not player or not player.hole_cards:
        return None
    
    # Calculate hand strength
    high_rank = player.hole_cards.high_card_rank.numeric_value
    low_rank = player.hole_cards.low_card_rank.numeric_value
    
    if player.hole_cards.is_pocket_pair:
        hand_strength = 0.5 + (high_rank / 28.0)
    else:
        rank_sum = high_rank + low_rank
        hand_strength = (rank_sum - 5) / (27 - 5)
        if player.hole_cards.is_suited:
            hand_strength += 0.1
        gap = high_rank - low_rank - 1
        hand_strength -= gap * 0.02
    
    hand_strength = max(0.05, min(0.95, hand_strength))
    
    # Situation
    facing_raise = random.choice([True, False])
    
    # Simple decision logic based on hand strength and position
    position_strength = {
        Position.UNDER_THE_GUN: 0.1,
        Position.MIDDLE_POSITION: 0.3,
        Position.CUTOFF: 0.6,
        Position.BUTTON: 1.0,
        Position.SMALL_BLIND: 0.2,
        Position.BIG_BLIND: 0.4
    }.get(position, 0.5)
    
    # Decision logic
    if facing_raise:
        # Facing a raise
        if hand_strength > 0.8:  # Premium hands
            action = 'raise' if random.random() < 0.6 else 'call'
        elif hand_strength > 0.5:  # Medium hands
            action = 'call' if random.random() < 0.7 else 'fold'
        else:  # Weak hands
            action = 'fold'
    else:
        # First to act
        threshold = 0.7 - (position_strength * 0.4)  # Looser in later position
        if hand_strength > threshold:
            action = 'raise'
        elif hand_strength > 0.3 and position in [Position.BIG_BLIND, Position.SMALL_BLIND]:
            action = 'call'
        else:
            action = 'fold'
    
    # Extract features
    features = {}
    
    # Position features
    for pos in positions:
        features[f'position_{pos.abbreviation}'] = 1.0 if pos == position else 0.0
    
    # Hand features
    features.update({
        'is_pocket_pair': 1.0 if player.hole_cards.is_pocket_pair else 0.0,
        'is_suited': 1.0 if player.hole_cards.is_suited else 0.0,
        'hand_strength': hand_strength,
        'high_card': high_rank / 14.0,
        'premium_hand': 1.0 if hand_strength > 0.8 else 0.0,
        'strong_hand': 1.0 if hand_strength > 0.6 else 0.0,
        'playable_hand': 1.0 if hand_strength > 0.3 else 0.0,
    })
    
    # Context features
    pot_size = 3.5 if facing_raise else 1.5
    bet_to_call = 3.0 if facing_raise else 0.0
    pot_odds = bet_to_call / (pot_size + bet_to_call) if facing_raise else 0.0
    
    features.update({
        'facing_raise': 1.0 if facing_raise else 0.0,
        'pot_size': pot_size / 10.0,  # Normalise
        'bet_to_call': bet_to_call / 5.0,  # Normalise
        'pot_odds': pot_odds,
        'stack_ratio': 1.0,  # Simplified
        'position_strength': position_strength,
        'num_players': 6.0 / 6.0,  # Normalised
    })
    
    return {
        'features': features,
        'optimal_action': {'action': action, 'size': 0.0},
        'context': {
            'position': position.abbreviation,
            'hole_cards': player.hole_cards.to_string_notation(),
            'hand_strength': hand_strength,
            'facing_raise': facing_raise
        }
    }

def generate_balanced_data(num_scenarios: int) -> List[Dict]:
    """Generate balanced training data with forced distribution."""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {num_scenarios} balanced scenarios...")
    
    scenarios_per_action = num_scenarios // 3
    target_counts = {'fold': scenarios_per_action, 'call': scenarios_per_action, 'raise': scenarios_per_action}
    generated_counts = {'fold': 0, 'call': 0, 'raise': 0}
    
    scenarios = []
    attempts = 0
    max_attempts = num_scenarios * 50
    
    while sum(generated_counts.values()) < num_scenarios and attempts < max_attempts:
        attempts += 1
        scenario = generate_simple_scenario()
        if not scenario:
            continue

        action = scenario['optimal_action']['action']
        
        # Only accept if we need more of this action
        if generated_counts[action] < target_counts[action]:
            scenarios.append(scenario)
            generated_counts[action] += 1 
            if len(scenarios) % 1000 == 0:
                logger.info(f"Generated {len(scenarios)}/{num_scenarios} - {generated_counts}")
    logger.info(f"Final distribution: {generated_counts}")
    return scenarios


def train_simple_model(scenarios: List[Dict], epochs: int = 100) -> SimpleGTOModel:
    """Train the simple model with robust techniques."""
    
    logger = logging.getLogger(__name__)
    
    # Split data
    train_data, val_data = train_test_split(scenarios, test_size=0.2, random_state=42)
    logger.info(f"Training: {len(train_data)}, Validation: {len(val_data)}")
    
    # Check action distribution
    train_actions = [s['optimal_action']['action'] for s in train_data]
    action_counts = {action: train_actions.count(action) for action in ['fold', 'call', 'raise']}
    logger.info(f"Training action distribution: {action_counts}")
    
    # Create model
    model = SimpleGTOModel(input_size=20)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters on {device}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Class weights for imbalance
    action_to_idx = {'fold': 0, 'call': 1, 'raise': 2}
    class_counts = [action_counts[action] for action in ['fold', 'call', 'raise']]
    total_samples = sum(class_counts)
    class_weights = [total_samples / (3 * count) for count in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    logger.info(f"Class weights: {dict(zip(['fold', 'call', 'raise'], class_weights))}")
    
    # Prepare data
    def create_dataloader(data, batch_size=64, shuffle=True):
        features, labels = [], []
        for sample in data:
            feature_tensor = model._features_to_tensor(sample['features'])
            features.append(feature_tensor)
            labels.append(action_to_idx[sample['optimal_action']['action']])
        
        features_tensor = torch.stack(features)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(features_tensor, labels_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    train_loader = create_dataloader(train_data, batch_size=128, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=128, shuffle=False)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    patience = 20
    patience_counter = 0
    logger.info("Starting training...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        action_correct = {'fold': 0, 'call': 0, 'raise': 0}
        action_total = {'fold': 0, 'call': 0, 'raise': 0}
        action_names = ['fold', 'call', 'raise']
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
                
                # Per-action accuracy
                for true_label, pred_label in zip(batch_labels.cpu(), predicted.cpu()):
                    true_action = action_names[true_label.item()]
                    action_total[true_action] += 1
                    if true_label == pred_label:
                        action_correct[true_action] += 1
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        action_accuracies = {}
        for action in action_correct:
            if action_total[action] > 0:
                action_accuracies[action] = action_correct[action] / action_total[action]
            else:
                action_accuracies[action] = 0.0
        
        # Check if all actions are being learned
        min_action_acc = min(action_accuracies.values())
        all_actions_learned = all(acc > 0.1 for acc in action_accuracies.values())

        # Save best model based on validation accuracy
        if val_acc > best_val_acc and all_actions_learned:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update learning rate
        scheduler.step()
        
        # Logging
        if epoch % 10 == 0 or epoch < 5:
            logger.info(f"Epoch {epoch}/{epochs}")
            logger.info(f"  Train - Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")
            logger.info(f"  Val   - Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}")
            logger.info(f"  Action Acc: {action_accuracies}")
            logger.info(f"  Min Action Acc: {min_action_acc:.4f}")
            logger.info(f"  All Actions Learned: {all_actions_learned}")
        

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
    return model


def test_model_quickly(model: SimpleGTOModel, num_tests: int = 50):
    """Quick model test."""
    logger = logging.getLogger(__name__)
    logger.info(f"Testing model on {num_tests} scenarios...")
    
    # Move model to CPU for testing to avoid device issues
    model.to('cpu')
    model.eval()
    
    predictions = {'fold': 0, 'call': 0, 'raise': 0}
    correct = 0

    for _ in range(num_tests):
        scenario = generate_simple_scenario()
        if not scenario:
            continue

        predicted_action, _ = model.predict_action(scenario['features'])
        expected_action = scenario['optimal_action']['action']
        
        predictions[predicted_action] += 1
        if predicted_action == expected_action:
            correct += 1

    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {correct}/{num_tests} ({100*correct/num_tests:.1f}%)")
    logger.info(f"  Predictions: {predictions}")

    # Check diversity
    min_predictions = min(predictions.values())
    if min_predictions == 0:
        logger.warning("Model not making diverse predictions!")
        return False
    else:
        logger.info("Model making diverse predictions!")
        return True

def main():
    """Main training function."""
    logger = setup_logging()
    logger.info("Starting Robust GTO Poker Training")

    # Configuration
    config = {
        'num_scenarios': 15000,
        'epochs': 80,
        'test_scenarios': 100
    }

    logger.info(f"Configuration: {config}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    try:
        # Generate data
        logger.info("PHASE 1: Generating balanced training data...")
        scenarios = generate_balanced_data(config['num_scenarios'])

        if len(scenarios) < config['num_scenarios'] * 0.8:
            logger.warning(f"Only generated {len(scenarios)} scenarios")

        # Train model
        logger.info("PHASE 2: Training model...")
        model = train_simple_model(scenarios, epochs=config['epochs'])

        # Test model
        logger.info("PHASE 3: Testing model...")
        is_diverse = test_model_quickly(model, config['test_scenarios'])

        # Save model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = models_dir / f"robust_gto_model_{timestamp}.pth"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {'input_size': 20},
            'timestamp': timestamp
        }, model_file)

        logger.info(f"Model saved to: {model_file}")
        if is_diverse:
            logger.info("SUCCESS! Model is making diverse predictions.")
        else:
            logger.warning("Model needs improvement but shows progress.")
        return str(model_file)

        

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    model_path = main()
    if model_path:
        print(f"\nModel saved to: {model_path}")
        print("Try testing with: python scripts/test_simple_model.py")
    else:
        print("Training failed.")
        sys.exit(1)