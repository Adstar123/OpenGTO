"""Command-line interface for OpenGTO.

This provides a user-friendly interface for all major operations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

from poker_gto.config.config import ConfigManager, TrainingConfig, TestingConfig
from poker_gto.ml.models.factory import ModelFactory
from poker_gto.ml.data.scenario_generator import PreflopScenarioGenerator
from poker_gto.ml.training.trainer import PreflopTrainer
from poker_gto.utils.logging_utils import setup_logging
from poker_gto.utils.validation import DataValidator
from poker_gto.utils.performance import PerformanceMonitor, ModelPerformanceTracker
from poker_gto.utils.testing import ModelTester


class OpenGTOCLI:
    """Command-line interface for OpenGTO."""
    
    def __init__(self):
        """Initialize CLI."""
        self.parser = self._create_parser()
        self.logger = None
        self.performance_monitor = None
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with subcommands."""
        parser = argparse.ArgumentParser(
            prog='opengto',
            description='OpenGTO - Neural Network GTO Poker Trainer',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands'
        )
        
        # Train command
        train_parser = subparsers.add_parser(
            'train',
            help='Train a new model'
        )
        train_parser.add_argument(
            '--config', '-c',
            type=Path,
            default=Path('configs/training_config.yaml'),
            help='Training configuration file'
        )
        train_parser.add_argument(
            '--output-dir', '-o',
            type=Path,
            default=Path('models'),
            help='Output directory for models'
        )
        train_parser.add_argument(
            '--validate-data',
            action='store_true',
            help='Validate training data before training'
        )
        
        # Test command
        test_parser = subparsers.add_parser(
            'test',
            help='Test a trained model'
        )
        test_parser.add_argument(
            'model',
            type=Path,
            nargs='?',
            help='Path to model file'
        )
        test_parser.add_argument(
            '--config', '-c',
            type=Path,
            default=Path('configs/testing_config.yaml'),
            help='Testing configuration file'
        )
        test_parser.add_argument(
            '--interactive', '-i',
            action='store_true',
            help='Run interactive testing'
        )
        
        # Benchmark command
        bench_parser = subparsers.add_parser(
            'benchmark',
            help='Benchmark model performance'
        )
        bench_parser.add_argument(
            'model',
            type=Path,
            help='Path to model file'
        )
        bench_parser.add_argument(
            '--batch-size', '-b',
            type=int,
            default=1,
            help='Batch size for benchmarking'
        )
        bench_parser.add_argument(
            '--iterations', '-n',
            type=int,
            default=100,
            help='Number of iterations'
        )
        
        # Generate command
        gen_parser = subparsers.add_parser(
            'generate',
            help='Generate training data'
        )
        gen_parser.add_argument(
            'output',
            type=Path,
            help='Output file for scenarios'
        )
        gen_parser.add_argument(
            '--count', '-n',
            type=int,
            default=10000,
            help='Number of scenarios to generate'
        )
        gen_parser.add_argument(
            '--validate',
            action='store_true',
            help='Validate generated data'
        )
        
        # Info command
        info_parser = subparsers.add_parser(
            'info',
            help='Show model information'
        )
        info_parser.add_argument(
            'model',
            type=Path,
            help='Path to model file'
        )
        
        # Config command
        config_parser = subparsers.add_parser(
            'config',
            help='Configuration management'
        )
        config_parser.add_argument(
            '--create-defaults',
            action='store_true',
            help='Create default configuration files'
        )
        config_parser.add_argument(
            '--show',
            type=Path,
            help='Show configuration from file'
        )
        
        return parser
    
    def run(self, args: Optional[list] = None):
        """Run the CLI.
        
        Args:
            args: Command line arguments (defaults to sys.argv)
        """
        parsed_args = self.parser.parse_args(args)
        
        # Setup logging
        log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
        self.logger = setup_logging(level=log_level)
        self.performance_monitor = PerformanceMonitor(self.logger)
        
        # Dispatch to command handler
        if not parsed_args.command:
            self.parser.print_help()
            return
        
        command_map = {
            'train': self._handle_train,
            'test': self._handle_test,
            'benchmark': self._handle_benchmark,
            'generate': self._handle_generate,
            'info': self._handle_info,
            'config': self._handle_config,
        }
        
        handler = command_map.get(parsed_args.command)
        if handler:
            try:
                handler(parsed_args)
            except Exception as e:
                self.logger.error(f"Error: {e}", exc_info=parsed_args.verbose)
                sys.exit(1)
        else:
            self.parser.print_help()
    
    def _handle_train(self, args):
        """Handle train command."""
        self.logger.info("Starting training...")
        
        # Load configuration
        config = TrainingConfig()
        if args.config.exists():
            config = ConfigManager.load_config(args.config, TrainingConfig)
            self.logger.info(f"Loaded configuration from {args.config}")
        
        with self.performance_monitor.measure("training"):
            # Generate data
            self.logger.info(f"Generating {config.num_scenarios} training scenarios...")
            generator = PreflopScenarioGenerator()
            
            with self.performance_monitor.measure("data_generation"):
                scenarios = generator.generate_balanced_scenarios(config.num_scenarios)
            
            # Validate data if requested
            if args.validate_data:
                self.logger.info("Validating training data...")
                validator = DataValidator(self.logger)
                report = validator.validate_dataset(scenarios)
                
                if report['invalid_scenarios'] > 0:
                    self.logger.warning(
                        f"Found {report['invalid_scenarios']} invalid scenarios"
                    )
            
            # Create model
            self.logger.info("Creating model...")
            # Extract only model-related config parameters
            model_config = {
                'input_size': config.input_size,
                'hidden_sizes': config.hidden_sizes,
                'dropout_rate': config.dropout_rate,
            }
            model = ModelFactory.create_model('preflop', model_config)
            
            # Train
            trainer = PreflopTrainer(model=model, device=config.device, logger=self.logger)
            
            with self.performance_monitor.measure("model_training"):
                results = trainer.train(
                    scenarios=scenarios,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    val_split=config.val_split,
                    patience=config.patience,
                    save_dir=args.output_dir
                )
            
            # Save final model
            args.output_dir.mkdir(exist_ok=True)
            model_path = args.output_dir / f"opengto_model_final.pth"
            
            model.save(
                str(model_path),
                metadata={
                    'model_type': 'preflop',
                    'training_config': config.__dict__,
                    'training_results': results,
                }
            )
            
            self.logger.info(f"Model saved to {model_path}")
            self.logger.info(f"Best validation accuracy: {results['best_val_acc']:.2%}")
        
        # Show performance summary
        summary = self.performance_monitor.get_summary()
        self.logger.info(f"Total training time: {summary['total_duration']:.2f}s")
    
    def _handle_test(self, args):
        """Handle test command."""
        # Find model if not specified
        if not args.model:
            models_dir = Path('models')
            if models_dir.exists():
                model_files = list(models_dir.glob('*.pth'))
                if model_files:
                    args.model = max(model_files, key=lambda p: p.stat().st_mtime)
                    self.logger.info(f"Using most recent model: {args.model}")
                else:
                    self.logger.error("No models found")
                    return
        
        # Load model
        self.logger.info(f"Loading model from {args.model}")
        model = ModelFactory.load_model(str(args.model))
        
        # Run tests
        tester = ModelTester(model, self.logger)
        
        # Load config
        config = TestingConfig()
        if args.config.exists():
            config = ConfigManager.load_config(args.config, TestingConfig)
        
        # Test
        tester.test_diverse_scenarios(
            num_tests=config.num_test_scenarios,
            show_examples=config.show_examples
        )
        
        # Interactive mode
        if args.interactive or config.interactive_mode:
            tester.interactive_test()
    
    def _handle_benchmark(self, args):
        """Handle benchmark command."""
        self.logger.info(f"Benchmarking model: {args.model}")
        
        # Load model
        model = ModelFactory.load_model(str(args.model))
        tracker = ModelPerformanceTracker(model)
        
        # Create sample input
        import torch
        input_size = 20  # For preflop model
        input_tensor = torch.randn(args.batch_size, input_size)
        
        # Benchmark
        self.logger.info("Running benchmark...")
        results = tracker.measure_inference_speed(
            input_tensor,
            num_iterations=args.iterations
        )
        
        # Display results
        print("\nBenchmark Results:")
        print("-" * 40)
        print(f"Device: {results['device']}")
        print(f"Batch size: {results['batch_size']}")
        print(f"Mean inference time: {results['mean_inference_time']*1000:.2f}ms")
        print(f"Throughput: {results['throughput']:.0f} samples/sec")
        print(f"Model parameters: {results['num_parameters']:,}")
        
        # Profile model
        profile = tracker.profile_model()
        print(f"\nModel size: {profile['model_size_mb']:.2f}MB")
        print(f"Trainable parameters: {profile['trainable_parameters']:,}")
    
    def _handle_generate(self, args):
        """Handle generate command."""
        self.logger.info(f"Generating {args.count} scenarios...")
        
        generator = PreflopScenarioGenerator()
        
        with self.performance_monitor.measure("scenario_generation"):
            scenarios = generator.generate_balanced_scenarios(args.count)
        
        # Validate if requested
        if args.validate:
            validator = DataValidator(self.logger)
            report = validator.validate_dataset(scenarios)
            
            if report['warnings']:
                self.logger.warning("Validation warnings found")
        
        # Save scenarios
        import json
        with open(args.output, 'w') as f:
            json.dump(scenarios, f, indent=2)
        
        self.logger.info(f"Saved {len(scenarios)} scenarios to {args.output}")
    
    def _handle_info(self, args):
        """Handle info command."""
        import torch
        
        # Load checkpoint
        checkpoint = torch.load(args.model, map_location='cpu')
        
        print(f"\nModel Information: {args.model}")
        print("=" * 50)
        
        # Model type
        metadata = checkpoint.get('metadata', {})
        model_type = metadata.get('model_type', 'unknown')
        print(f"Model type: {model_type}")
        
        # Training info
        if 'training_config' in metadata:
            config = metadata['training_config']
            print(f"\nTraining Configuration:")
            print(f"  Epochs: {config.get('epochs', 'N/A')}")
            print(f"  Batch size: {config.get('batch_size', 'N/A')}")
            print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
        
        # Results
        if 'training_results' in metadata:
            results = metadata['training_results']
            print(f"\nTraining Results:")
            print(f"  Best validation accuracy: {results.get('best_val_acc', 0):.2%}")
            print(f"  Final epoch: {results.get('final_epoch', 'N/A')}")
        
        # Model architecture
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            print(f"\nModel Architecture:")
            print(f"  Input size: {model_config.get('input_size', 'N/A')}")
            print(f"  Hidden sizes: {model_config.get('hidden_sizes', 'N/A')}")
            print(f"  Dropout rate: {model_config.get('dropout_rate', 'N/A')}")
    
    def _handle_config(self, args):
        """Handle config command."""
        if args.create_defaults:
            self.logger.info("Creating default configuration files...")
            ConfigManager.create_default_configs()
            self.logger.info("Default configurations created in configs/")
        
        elif args.show:
            # Load and display config
            import yaml
            
            with open(args.show, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"\nConfiguration from {args.show}:")
            print("-" * 40)
            print(yaml.dump(config, default_flow_style=False))
        
        else:
            print("Use --create-defaults or --show <file>")


def main():
    """Main entry point for CLI."""
    cli = OpenGTOCLI()
    cli.run()


if __name__ == '__main__':
    main()