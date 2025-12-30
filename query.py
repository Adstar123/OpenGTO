#!/usr/bin/env python3
"""
Query a specific poker scenario for GTO strategy.

Usage:
    python query.py --position HJ --hand 62s --actions "UTG:fold"
    python query.py --position BTN --hand AKs --actions "UTG:fold,HJ:raise 2.5,CO:fold"
    python query.py --position BB --hand 99 --actions "UTG:raise 2.5,HJ:fold,CO:fold,BTN:fold,SB:fold"
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trainer_interface import GTOTrainerInterface, ACTION_NAMES


def parse_actions(actions_str: str):
    """Parse action string like 'UTG:fold,HJ:raise 2.5' into list of tuples."""
    if not actions_str:
        return []

    actions = []
    for part in actions_str.split(','):
        part = part.strip()
        if ':' not in part:
            continue
        pos, action = part.split(':', 1)
        actions.append((pos.strip(), action.strip()))
    return actions


def main():
    parser = argparse.ArgumentParser(description='Query GTO strategy for a specific scenario')
    parser.add_argument('--checkpoint', '-c', default='checkpoints_improved/gto_trainer_final.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--position', '-p', required=True,
                        help='Your position (UTG, HJ, CO, BTN, SB, BB)')
    parser.add_argument('--hand', '-H', required=True,
                        help='Your hand (e.g., AKs, 62s, AsKh)')
    parser.add_argument('--actions', '-a', default='',
                        help='Actions before you (e.g., "UTG:fold,HJ:raise 2.5")')
    parser.add_argument('--stack', '-s', type=float, default=100.0,
                        help='Stack size in BB (default: 100)')

    args = parser.parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        # Try alternate paths
        alt_paths = [
            'checkpoints/gto_trainer_final.pt',
            'checkpoints_improved/gto_trainer_final.pt',
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                args.checkpoint = alt
                break
        else:
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    # Initialize trainer
    trainer = GTOTrainerInterface(
        checkpoint_path=args.checkpoint,
        num_players=6,
        device='cpu'
    )
    trainer.load_model()

    # Parse actions
    action_history = parse_actions(args.actions)

    # Generate scenario
    try:
        scenario = trainer.generate_scenario(
            position=args.position,
            hand=args.hand,
            stack=args.stack,
            action_history=action_history if action_history else None
        )
    except Exception as e:
        print(f"Error creating scenario: {e}")
        sys.exit(1)

    # Display scenario
    print(trainer.display_scenario(scenario))

    # Get GTO strategy
    strategy = trainer.get_gto_strategy(scenario.state)

    print("\n" + "=" * 50)
    print("GTO STRATEGY")
    print("=" * 50)

    # Display strategy for legal actions
    probs = []
    for i in scenario.legal_action_indices:
        probs.append((ACTION_NAMES[i], float(strategy[i])))

    probs.sort(key=lambda x: -x[1])

    for action, prob in probs:
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"  {action:8} [{bar}] {prob*100:5.1f}%")

    print("=" * 50)

    # Recommendation
    best_action = probs[0][0]
    best_prob = probs[0][1]
    print(f"\nRecommendation: {best_action} ({best_prob*100:.0f}%)")


if __name__ == '__main__':
    main()
