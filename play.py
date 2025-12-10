#!/usr/bin/env python3
"""
Quick launcher for the GTO Poker Trainer.

Usage:
    python play.py                                    # Use default checkpoint
    python play.py checkpoints/gto_trainer_final.pt  # Specify checkpoint
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trainer_interface import GTOTrainerInterface


def main():
    # Default checkpoint path
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
    else:
        # Try to find a checkpoint
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if 'gto_trainer_final.pt' in files:
                checkpoint = os.path.join(checkpoint_dir, 'gto_trainer_final.pt')
            elif files:
                # Use the latest
                files.sort()
                checkpoint = os.path.join(checkpoint_dir, files[-1])
            else:
                print("No checkpoint found in checkpoints/")
                print("Usage: python play.py <checkpoint_path>")
                sys.exit(1)
        else:
            print("No checkpoints directory found.")
            print("Usage: python play.py <checkpoint_path>")
            sys.exit(1)

    print(f"Using checkpoint: {checkpoint}")

    trainer = GTOTrainerInterface(
        checkpoint_path=checkpoint,
        num_players=6,
        device='cpu'
    )
    trainer.run_interactive_session()


if __name__ == '__main__':
    main()
