"""

Simple testing script for the robust model.

"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from pathlib import Path
from typing import Dict, Tuple

# Add the project root to path again
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from poker_gto.core.game_state import GameState, GameConfig
from poker_gto.core.position import Position, PositionManager

# Import generate_simple_scenario from train_robust.py
from scripts.train_robust import generate_simple_scenario

class SimpleGTOModel(nn.Module):
    """Simplified preflop model architecture."""
    
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
        
        # Move to same device as model
        device = next(self.parameters()).device
        feature_tensor = feature_tensor.to(device)
        
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


def load_simple_model(model_path: str) -> SimpleGTOModel:
    """Load the simple model."""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint.get('model_config', {'input_size': 20})
    
    model = SimpleGTOModel(input_size=model_config['input_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def test_diverse_scenarios(model: SimpleGTOModel, num_tests: int = 100):
    """Test model on diverse scenarios."""
    print(f"\nTesting model on {num_tests} scenarios...")
    
    predictions = {'fold': 0, 'call': 0, 'raise': 0}
    correct_predictions = 0
    
    print("\nSample predictions:")
    print("-" * 70)
    print(f"{'Position':<8} {'Hand':<6} {'Situation':<12} {'Expected':<8} {'Predicted':<10} {'Match'}")
    print("-" * 70)
    
    for i in range(num_tests):
        scenario = generate_simple_scenario()
        if not scenario:
            continue
        
        features = scenario['features']
        expected_action = scenario['optimal_action']['action']
        context = scenario['context']
        
        # Get model prediction
        predicted_action, predicted_size = model.predict_action(features)
        predictions[predicted_action] += 1
        
        is_correct = predicted_action == expected_action
        if is_correct:
            correct_predictions += 1
        
        # Show first 20 examples
        if i < 20:
            position = context['position']
            hand = context['hole_cards']
            situation = "vs Raise" if context['facing_raise'] else "First In"
            
            match_str = "YES" if is_correct else "NO"
            
            print(f"{position:<8} {hand:<6} {situation:<12} {expected_action:<8} {predicted_action:<10} {match_str}")
    
    print("-" * 70)
    print(f"\nResults:")
    print(f"  Accuracy: {correct_predictions}/{num_tests} ({100*correct_predictions/num_tests:.1f}%)")
    print(f"  Prediction distribution: {predictions}")
    
    # Check diversity
    min_predictions = min(predictions.values())
    total_predictions = sum(predictions.values())
    
    if min_predictions == 0:
        print("  Status: Model is NOT making diverse predictions!")
        return False
    elif min_predictions < total_predictions * 0.05:  # Less than 5%
        print("  Status: Model predictions are imbalanced but functional.")
        return True
    else:
        print("  Status: Model is making diverse predictions!")
        return True


def interactive_test(model: SimpleGTOModel):
    """Interactive testing."""
    print("\nInteractive Testing Mode")
    print("Enter poker scenarios to test the model!")
    print("Type 'quit' to exit.\n")
    
    # Simple hand strengths for testing
    hand_strengths = {
        'AA': 0.95, 'KK': 0.90, 'QQ': 0.85, 'JJ': 0.80, 'TT': 0.75,
        'AKs': 0.78, 'AKo': 0.73, 'AQs': 0.70, 'AQo': 0.65,
        '99': 0.70, '88': 0.65, '77': 0.60, '66': 0.55,
        'AJs': 0.65, 'AJo': 0.58, 'KQs': 0.62, 'KQo': 0.55,
        '55': 0.50, '44': 0.45, '33': 0.40, '22': 0.35,
        '72o': 0.15, '83o': 0.20, '94o': 0.25
    }
    
    while True:
        try:
            print("\n" + "="*40)
      
            # Get inputs
            position_input = input("Position (UTG/MP/CO/BTN/SB/BB): ").strip().upper()
            if position_input.lower() == 'quit':
                break
            
            if position_input not in ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']:
                print("Invalid position!")
                continue
            
            hand = input("Hand (AA, AKs, 72o, etc.): ").strip()
            if hand.lower() == 'quit':
                break
            
            situation = input("Situation (open/vs_raise): ").strip().lower()
            if situation == 'quit':
                break
            
            facing_raise = situation == 'vs_raise'
            
            # Create features
            features = {}
            
            # Position features
            for pos in ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']:
                features[f'position_{pos}'] = 1.0 if pos == position_input else 0.0
            
            # Hand features
            hand_strength = hand_strengths.get(hand, 0.40)
            is_pair = len(set(hand[:2])) == 1
            is_suited = 's' in hand.lower()
            
            features.update({
                'is_pocket_pair': 1.0 if is_pair else 0.0,
                'is_suited': 1.0 if is_suited else 0.0,
                'hand_strength': hand_strength,
                'high_card': max([hand_strengths.get(hand[:2], 0.5), 0.5]),
                'premium_hand': 1.0 if hand_strength > 0.8 else 0.0,
                'strong_hand': 1.0 if hand_strength > 0.6 else 0.0,
                'playable_hand': 1.0 if hand_strength > 0.3 else 0.0,
            })
            
            # Context features
            position_strength = {
                'UTG': 0.1, 'MP': 0.3, 'CO': 0.6, 'BTN': 1.0, 'SB': 0.2, 'BB': 0.4
            }.get(position_input, 0.5)
            
            pot_size = 3.5 if facing_raise else 1.5
            bet_to_call = 3.0 if facing_raise else 0.0
            pot_odds = bet_to_call / (pot_size + bet_to_call) if facing_raise else 0.0
            
            features.update({
                'facing_raise': 1.0 if facing_raise else 0.0,
                'pot_size': pot_size / 10.0,
                'bet_to_call': bet_to_call / 5.0,
                'pot_odds': pot_odds,
                'stack_ratio': 1.0,
                'position_strength': position_strength,
                'num_players': 1.0,
            })
            
            # Get prediction
            action, size = model.predict_action(features)
            
            # Display results
            print(f"\nModel Recommendation:")
            print(f"  Position: {position_input}")
            print(f"  Hand: {hand} (strength: {hand_strength:.2f})")
            print(f"  Situation: {'Facing raise' if facing_raise else 'First to act'}")
            print(f"  Recommended Action: {action.upper()}")
            if action == 'raise':
                print(f"  Raise Size: {size:.1f} BB")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main testing function."""
    
    # Try to find the most recent model if no path provided
    if len(sys.argv) == 1:
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("robust_gto_model_*.pth"))
            if model_files:
                # Sort by modification time, get most recent
                model_path = max(model_files, key=lambda p: p.stat().st_mtime)
                print(f"No model path provided. Using most recent: {model_path}")
            else:
                print("No robust models found. Run train_robust.py first.")
                return
        else:
            print("No models directory found. Run train_robust.py first.")
            return
    else:
        model_path = sys.argv[1]
        if not Path(model_path).exists():
            print(f"Model file not found: {model_path}")
            return
    
    try:
        # Load and test model
        model = load_simple_model(str(model_path))
        
        # Run diversity test
        is_diverse = test_diverse_scenarios(model, num_tests=100)
        
        if is_diverse:
            print("\nModel passed diversity test!")
            
            # Offer interactive testing
            response = input("\nTry interactive testing? (y/n): ").strip().lower()
            if response == 'y':
                interactive_test(model)
            else:
                print("Testing complete!")
        else:
            print("\nModel failed diversity test but may still be learning.")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()