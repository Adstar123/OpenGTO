"""
Generate training data from CFR solver's converged strategies.
"""
import json
import csv
import os
from typing import List, Dict
import numpy as np
from src.cfr_solver import CFRSolver
from src.game_tree import Action, Position


class TrainingDataGenerator:
    """
    Generates training data from a trained CFR solver.
    """

    def __init__(self, cfr_solver: CFRSolver):
        """
        Initialize data generator.

        Args:
            cfr_solver: Trained CFR solver
        """
        self.solver = cfr_solver
        self.strategies = cfr_solver.get_all_strategies()

    def parse_info_set(self, info_set: str) -> Dict:
        """
        Parse an information set string into components.

        Format: "position:hand:action_history"
        Example: "btn:AKs:raise_3bb_call"

        Returns:
            Dictionary with parsed components
        """
        parts = info_set.split(':')
        if len(parts) < 2:
            return None

        position = parts[0]
        hand = parts[1]
        action_history = parts[2] if len(parts) > 2 else ""

        return {
            'position': position,
            'hand': hand,
            'action_history': action_history
        }

    def generate_training_examples(self) -> List[Dict]:
        """
        Generate training examples from converged strategies.

        Each example contains:
        - position: 'btn' or 'bb'
        - hand: hand string like 'AKs', 'QQ', etc.
        - action_history: sequence of actions leading to this decision point
        - stack_bb: stack size in big blinds
        - action_probabilities: dict of action -> probability

        Returns:
            List of training examples
        """
        examples = []

        for info_set, strategy in self.strategies.items():
            parsed = self.parse_info_set(info_set)
            if not parsed:
                continue

            # Create training example
            example = {
                'position': parsed['position'],
                'hand': parsed['hand'],
                'action_history': parsed['action_history'],
                'stack_bb': self.solver.game_tree.stack_size,
            }

            # Add action probabilities
            for action in Action:
                prob = strategy.get(action, 0.0)
                example[f'prob_{action.value}'] = prob

            examples.append(example)

        return examples

    def save_to_csv(self, output_path: str):
        """
        Save training data to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        examples = self.generate_training_examples()

        if not examples:
            print("No training examples generated!")
            return

        # Define CSV columns
        fieldnames = [
            'position', 'hand', 'action_history', 'stack_bb',
            'prob_fold', 'prob_check', 'prob_call',
            'prob_raise_2bb', 'prob_raise_3bb', 'prob_raise_4bb', 'prob_all_in'
        ]

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(examples)

        print(f"Saved {len(examples)} training examples to {output_path}")

    def save_to_json(self, output_path: str):
        """
        Save training data to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        examples = self.generate_training_examples()

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)

        print(f"Saved {len(examples)} training examples to {output_path}")

    def print_statistics(self):
        """Print statistics about the generated training data."""
        examples = self.generate_training_examples()

        if not examples:
            print("No training examples to analyze!")
            return

        print("\n" + "="*80)
        print("Training Data Statistics")
        print("="*80)

        print(f"Total examples: {len(examples)}")

        # Count by position
        btn_count = sum(1 for e in examples if e['position'] == 'btn')
        bb_count = sum(1 for e in examples if e['position'] == 'bb')
        print(f"BTN examples: {btn_count}")
        print(f"BB examples: {bb_count}")

        # Count unique hands
        unique_hands = set(e['hand'] for e in examples)
        print(f"Unique hands: {len(unique_hands)}")

        # Action distribution
        print("\nAction probability distribution (average):")
        action_cols = ['prob_fold', 'prob_check', 'prob_call', 'prob_raise_2bb',
                      'prob_raise_3bb', 'prob_raise_4bb', 'prob_all_in']

        for col in action_cols:
            avg_prob = np.mean([e[col] for e in examples])
            if avg_prob > 0.001:
                print(f"  {col:20s}: {avg_prob:6.2%}")
