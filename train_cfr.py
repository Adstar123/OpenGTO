"""
Main script to train CFR solver and generate training data.
"""
import argparse
from src.cfr_solver import CFRSolver
from src.data_generator import TrainingDataGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Train CFR solver for preflop poker and generate training data'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='Number of CFR iterations (default: 1000)'
    )
    parser.add_argument(
        '--sample-hands',
        type=int,
        default=50,
        help='Number of random hand matchups per iteration (default: 50)'
    )
    parser.add_argument(
        '--stack-size',
        type=float,
        default=100.0,
        help='Stack size in big blinds (default: 100.0)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default='data/training_data.csv',
        help='Output CSV file path (default: data/training_data.csv)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='data/training_data.json',
        help='Output JSON file path (default: data/training_data.json)'
    )
    parser.add_argument(
        '--show-samples',
        type=int,
        default=20,
        help='Number of sample strategies to display (default: 20)'
    )

    args = parser.parse_args()

    print("="*80)
    print("GTO Preflop Trainer - CFR Training")
    print("="*80)
    print(f"Configuration:")
    print(f"  Iterations: {args.iterations}")
    print(f"  Sample hands per iteration: {args.sample_hands}")
    print(f"  Stack size: {args.stack_size} BB")
    print("="*80)

    # Initialize and train CFR solver
    solver = CFRSolver(stack_size=args.stack_size, big_blind=1.0)
    solver.train(iterations=args.iterations, sample_hands=args.sample_hands)

    # Display sample strategies
    solver.print_sample_strategies(num_samples=args.show_samples)

    # Generate training data
    print("\n" + "="*80)
    print("Generating Training Data")
    print("="*80)

    data_gen = TrainingDataGenerator(solver)
    data_gen.print_statistics()

    # Save to files
    data_gen.save_to_csv(args.output_csv)
    data_gen.save_to_json(args.output_json)

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Training data saved to:")
    print(f"  - {args.output_csv}")
    print(f"  - {args.output_json}")


if __name__ == '__main__':
    main()
