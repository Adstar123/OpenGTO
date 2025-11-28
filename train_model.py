"""
Main script to train the GTO preflop strategy neural network.
"""
import argparse
import torch
from torch.utils.data import DataLoader
from src.preprocessing import PreflopDataPreprocessor
from src.dataset import PreflopDataset
from src.model import PreflopStrategyNet, PreflopStrategyNetV2
from src.trainer import GTOTrainer
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description='Train neural network for GTO preflop poker strategy'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/training_data.csv',
        help='Path to training data CSV (default: data/training_data.csv)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['simple', 'advanced'],
        default='simple',
        help='Model architecture to use (default: simple)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate (default: 0.2)'
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=15,
        help='Early stopping patience (default: 15)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models',
        help='Directory to save model checkpoints (default: models)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set (default: 0.2)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Proportion of training data for validation (default: 0.1)'
    )

    args = parser.parse_args()

    print("="*80)
    print("GTO Preflop Trainer - Neural Network Training")
    print("="*80)
    print(f"Configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Model type: {args.model_type}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Early stopping patience: {args.early_stopping}")
    print("="*80)

    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = PreflopDataPreprocessor()
    data_splits = preprocessor.load_and_split(
        args.data_path,
        test_size=args.test_size,
        val_size=args.val_size
    )

    # Create datasets
    train_dataset = PreflopDataset(*data_splits['train'])
    val_dataset = PreflopDataset(*data_splits['val'])
    test_dataset = PreflopDataset(*data_splits['test'])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Create model
    print(f"\nInitializing {args.model_type} model...")
    if args.model_type == 'simple':
        model = PreflopStrategyNet(
            input_size=15,
            hidden_sizes=[256, 128, 64],
            output_size=7,
            dropout_rate=args.dropout
        )
    else:
        model = PreflopStrategyNetV2(
            output_size=7,
            dropout_rate=args.dropout
        )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # Create trainer
    trainer = GTOTrainer(
        model=model,
        learning_rate=args.learning_rate
    )

    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        checkpoint_dir=args.checkpoint_dir
    )

    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on Test Set")
    print("="*80)

    # Load best model
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = trainer.evaluate(test_loader)

    print(f"Test Loss: {test_results['test_loss']:.6f}")
    print(f"Test Accuracy: {test_results['test_accuracy']:.4f}")
    print(f"Mean KL Divergence: {test_results['mean_kl_divergence']:.6f}")
    print(f"Median KL Divergence: {test_results['median_kl_divergence']:.6f}")
    print(f"Max KL Divergence: {test_results['max_kl_divergence']:.6f}")

    # Save results (convert numpy types to Python types)
    def convert_to_python_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        return obj

    results = {
        'config': vars(args),
        'num_parameters': num_params,
        'history': convert_to_python_types(history),
        'test_results': convert_to_python_types(test_results)
    }

    results_path = os.path.join(args.checkpoint_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
