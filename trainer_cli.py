#!/usr/bin/env python3
"""
GTO Poker Trainer - Command Line Interface

Usage:
    python trainer_cli.py train [--quick] [--iterations N] [--device cpu|cuda]
    python trainer_cli.py evaluate <checkpoint>
    python trainer_cli.py analyze <checkpoint>
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trainer import GTOTrainer, TrainingConfig, train_gto
from src.evaluation import analyze_preflop_strategy


def cmd_train(args):
    """Run training."""
    print("=" * 60)
    print("GTO POKER TRAINER")
    print("=" * 60)
    print()

    config = TrainingConfig(
        quick_mode=args.quick,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        print_every=args.print_every,
        save_every=args.save_every,
        eval_every=args.eval_every
    )

    trainer = GTOTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(num_iterations=args.iterations)

    # Print final analysis
    print("\n" + "=" * 60)
    print("FINAL STRATEGY ANALYSIS")
    print("=" * 60)
    print(trainer.analyze_strategy())


def cmd_evaluate(args):
    """Evaluate a trained model."""
    print("Loading checkpoint...")

    config = TrainingConfig(device=args.device)
    trainer = GTOTrainer(config)
    trainer.load_checkpoint(args.checkpoint)

    print("\nRunning evaluation...")
    print(trainer.analyze_strategy())


def cmd_analyze(args):
    """Analyze strategy from checkpoint."""
    print("Loading checkpoint...")

    config = TrainingConfig(device=args.device)
    trainer = GTOTrainer(config)
    trainer.load_checkpoint(args.checkpoint)

    print("\n" + trainer.analyze_strategy())


def cmd_quick_test(args):
    """Run a quick test to verify everything works."""
    print("Running quick training test...")
    print("This will train for just a few iterations to verify setup.\n")

    config = TrainingConfig(
        quick_mode=True,
        device=args.device,
        print_every=5,
        save_every=20,
        eval_every=10
    )

    trainer = GTOTrainer(config)
    trainer.train(num_iterations=args.iterations or 20)

    print("\nQuick test complete!")
    print("If you see strategy analysis above, everything is working.")


def main():
    parser = argparse.ArgumentParser(
        description="GTO Poker Trainer - Train neural networks to play GTO poker"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a GTO model')
    train_parser.add_argument('--quick', action='store_true',
                              help='Use quick curriculum for testing')
    train_parser.add_argument('--iterations', type=int, default=None,
                              help='Override number of iterations')
    train_parser.add_argument('--device', type=str, default='cpu',
                              choices=['cpu', 'cuda'],
                              help='Device to train on')
    train_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                              help='Directory to save checkpoints')
    train_parser.add_argument('--resume', type=str, default=None,
                              help='Resume from checkpoint')
    train_parser.add_argument('--print-every', type=int, default=10,
                              help='Print progress every N iterations')
    train_parser.add_argument('--save-every', type=int, default=100,
                              help='Save checkpoint every N iterations')
    train_parser.add_argument('--eval-every', type=int, default=50,
                              help='Run evaluation every N iterations')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    eval_parser.add_argument('--device', type=str, default='cpu')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze strategy')
    analyze_parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    analyze_parser.add_argument('--device', type=str, default='cpu')

    # Quick test command
    test_parser = subparsers.add_parser('test', help='Run a quick test')
    test_parser.add_argument('--device', type=str, default='cpu')
    test_parser.add_argument('--iterations', type=int, default=20)

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'test':
        cmd_quick_test(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
