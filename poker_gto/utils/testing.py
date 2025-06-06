"""Model testing utilities."""

import logging
from typing import Dict, Optional
import numpy as np

from poker_gto.ml.models.preflop_model import PreflopGTOModel
from poker_gto.ml.data.scenario_generator import PreflopScenarioGenerator
from poker_gto.ml.features.feature_extractors import PreflopFeatureExtractor


class ModelTester:
    """Test and evaluate poker models."""
    
    def __init__(self, model: PreflopGTOModel, logger: logging.Logger):
        """Initialize tester.
        
        Args:
            model: Model to test
            logger: Logger for output
        """
        self.model = model
        self.logger = logger
        self.generator = PreflopScenarioGenerator()
        self.extractor = PreflopFeatureExtractor()
    
    def test_diverse_scenarios(self, num_tests: int = 100, show_examples: int = 20):
        """Test model on diverse scenarios.
        
        Args:
            num_tests: Number of test scenarios
            show_examples: Number of examples to display
        """
        self.logger.info(f"\nTesting model on {num_tests} scenarios...")
        
        predictions = {'fold': 0, 'call': 0, 'raise': 0, 'check': 0}
        correct_predictions = 0
        
        print("\nSample predictions:")
        print("-" * 80)
        print(f"{'Position':<8} {'Hand':<6} {'Situation':<15} {'Expected':<8} {'Predicted':<10} {'Size':<6} {'Match'}")
        print("-" * 80)
        
        for i in range(num_tests):
            scenario = self.generator.generate_single_scenario()
            if not scenario:
                continue
            
            features = scenario['features']
            expected_action = scenario['optimal_action']['action']
            context = scenario['context']
            
            # Get model prediction
            predicted_action, predicted_size = self.model.predict_action(features)
            predictions[predicted_action] += 1
            
            is_correct = predicted_action == expected_action
            if is_correct:
                correct_predictions += 1
            
            # Show examples
            if i < show_examples:
                position = context['position']
                hand = context['hole_cards']
                
                # Determine situation string
                if context['position'] == 'BB' and context['bet_to_call'] == 0.0 and not context['facing_raise']:
                    situation = "vs Limp"
                elif context['facing_raise']:
                    situation = "Facing Raise"
                else:
                    situation = "Opening"
                
                match_str = "✓" if is_correct else "✗"
                size_str = f"{predicted_size:.1f}" if predicted_action == 'raise' else "-"
                
                print(f"{position:<8} {hand:<6} {situation:<15} {expected_action:<8} "
                      f"{predicted_action:<10} {size_str:<6} {match_str}")
        
        print("-" * 80)
        print(f"\nResults:")
        print(f"  Accuracy: {correct_predictions}/{num_tests} ({100*correct_predictions/num_tests:.1f}%)")
        print(f"  Prediction distribution: {predictions}")
        
        # Check diversity
        min_predictions = min(predictions.values())
        total_predictions = sum(predictions.values())
        
        if min_predictions == 0:
            # Check is often rare, so don't require it
            non_check_predictions = {k: v for k, v in predictions.items() if k != 'check'}
            min_non_check = min(non_check_predictions.values())
            if min_non_check == 0:
                print("  ⚠️  WARNING: Model is NOT making diverse predictions!")
                return False
            else:
                print("  ✅ Model is making reasonably diverse predictions!")
                return True
        elif min_predictions < total_predictions * 0.02:  # Lower threshold for check
            print("  ⚠️  Model predictions are imbalanced but functional.")
            return True
        else:
            print("  ✅ Model is making well-balanced predictions!")
            return True
    
    def interactive_test(self):
        """Interactive testing mode."""
        print("\n" + "="*60)
        print("Interactive Testing Mode")
        print("="*60)
        print("\nEnter poker scenarios to test the model")
        print("Type 'quit' or 'q' to exit\n")
        
        while True:
            try:
                print("\n" + "-"*40)
                
                # Get position
                position_input = input("Position (UTG/MP/CO/BTN/SB/BB): ").strip().upper()
                if position_input.lower() in ['quit', 'q']:
                    break
                
                if position_input not in ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']:
                    print("❌ Invalid position! Please use: UTG, MP, CO, BTN, SB, or BB")
                    continue
                
                # Get hand
                hand = input("Hand (e.g., AA, AKs, 72o): ").strip()
                if hand.lower() in ['quit', 'q']:
                    break
                
                # Get situation
                situation = input("Situation (open/vs_raise/vs_limp) [default: open]: ").strip().lower()
                if situation in ['quit', 'q']:
                    break
                if situation == '':
                    situation = 'open'
                
                # Set up scenario based on situation
                if situation == 'vs_raise':
                    facing_raise = True
                    pot_size = 3.5
                    bet_to_call = 3.0
                elif situation == 'vs_limp' and position_input == 'BB':
                    facing_raise = False
                    pot_size = 2.0  # SB completed
                    bet_to_call = 0.0
                else:  # open
                    facing_raise = False
                    pot_size = 1.5
                    bet_to_call = 0.0
                
                # Create scenario
                scenario_data = {
                    'position': position_input,
                    'hole_cards': hand,
                    'player_count': 6,
                    'facing_raise': facing_raise,
                    'pot_size': pot_size,
                    'bet_to_call': bet_to_call,
                    'stack_ratio': 1.0,
                    'num_players': 6,
                }
                
                # Extract features
                features = self.extractor.extract_from_scenario(scenario_data)
                
                # Get prediction
                action, size = self.model.predict_action(features)
                
                # Get action probabilities
                probs = self.model.get_action_probabilities(features)
                
                # Display results
                print(f"\n{'='*40}")
                print(f"Model Analysis:")
                print(f"{'='*40}")
                print(f"Position: {position_input}")
                print(f"Hand: {hand}")
                
                # Format situation description
                if situation == 'vs_limp' and position_input == 'BB':
                    situation_desc = 'Facing a limp (can check or raise)'
                elif facing_raise:
                    situation_desc = 'Facing a raise'
                else:
                    situation_desc = 'First to act'
                
                print(f"Situation: {situation_desc}")
                print(f"\nRecommended Action: {action.upper()}")
                if action == 'raise':
                    print(f"Raise Size: {size:.1f} BB")
                
                print(f"\nAction Probabilities:")
                print(f"  Fold:  {probs['fold']:.1%}")
                print(f"  Call:  {probs['call']:.1%}")
                print(f"  Raise: {probs['raise']:.1%}")
                print(f"  Check: {probs['check']:.1%}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")