"""
Main Training Orchestration for GTO Poker Trainer.

This is the main entry point for training the neural network.
Combines all components: Deep CFR, curriculum learning, evaluation.
"""
import torch
import numpy as np
import os
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from .game_state import GameState
from .poker_engine import PreflopPokerEngine
from .information_set import InformationSet, get_legal_actions_mask, NUM_ACTIONS
from .neural_network import DeepCFRNetworks
from .memory_buffer import DeepCFRMemory, TrainingStats
from .self_play import SelfPlayDataGenerator
from .curriculum import CurriculumScheduler, CurriculumStage, create_full_curriculum, create_quick_curriculum
from .evaluation import StrategyEvaluator, ConvergenceMonitor, analyze_preflop_strategy


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Curriculum
    use_curriculum: bool = True
    quick_mode: bool = False  # Use quick curriculum for testing

    # Network architecture
    hidden_sizes: Tuple[int, ...] = (256, 256, 128)

    # Training parameters
    batch_size: int = 2048
    train_steps_per_iter: int = 100
    traversals_per_iter: int = 500
    exploration: float = 0.6

    # Memory
    regret_buffer_size: int = 2_000_000
    strategy_buffer_size: int = 2_000_000

    # Checkpointing
    save_every: int = 100
    eval_every: int = 50
    print_every: int = 10

    # Device
    device: str = 'cpu'

    # Paths
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingConfig':
        return cls(**d)


class GTOTrainer:
    """
    Main GTO Poker Trainer.

    Orchestrates the training process using Deep CFR with curriculum learning.
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()

        # Set up directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        # Initialize components (will be set up when training starts)
        self.networks: Optional[DeepCFRNetworks] = None
        self.memory: Optional[DeepCFRMemory] = None
        self.curriculum: Optional[CurriculumScheduler] = None
        self.engine: Optional[PreflopPokerEngine] = None
        self.data_generator: Optional[SelfPlayDataGenerator] = None
        self.convergence_monitor: Optional[ConvergenceMonitor] = None
        self.stats: Optional[TrainingStats] = None

        # Training state
        self.iteration = 0
        self.total_traversals = 0
        self.training_start_time: Optional[float] = None

    def setup(self):
        """Initialize all training components."""
        print("Setting up trainer...")

        # Curriculum
        if self.config.use_curriculum:
            if self.config.quick_mode:
                self.curriculum = create_quick_curriculum()
            else:
                self.curriculum = create_full_curriculum()
            initial_stage = self.curriculum.current_stage
            num_players = initial_stage.num_players
            starting_stack = (initial_stage.min_stack + initial_stage.max_stack) / 2
        else:
            num_players = 6
            starting_stack = 100.0

        # Poker engine
        self.engine = PreflopPokerEngine(
            num_players=num_players,
            starting_stack=starting_stack
        )

        # Neural networks
        input_size = InformationSet.feature_size()
        self.networks = DeepCFRNetworks(
            input_size=input_size,
            hidden_sizes=self.config.hidden_sizes,
            num_actions=NUM_ACTIONS,
            device=self.config.device
        )

        # Memory buffers
        self.memory = DeepCFRMemory(
            regret_buffer_size=self.config.regret_buffer_size,
            strategy_buffer_size=self.config.strategy_buffer_size
        )

        # Data generator
        self.data_generator = SelfPlayDataGenerator(
            poker_engine=self.engine,
            exploration=self.config.exploration
        )

        # Monitoring
        self.convergence_monitor = ConvergenceMonitor(
            check_interval=self.config.eval_every
        )
        self.stats = TrainingStats()

        print(f"  Device: {self.config.device}")
        print(f"  Network: {self.config.hidden_sizes}")
        print(f"  Initial game: {num_players}p, {starting_stack}bb")
        if self.curriculum:
            print(f"  Curriculum: {len(self.curriculum.stages)} stages")
        print("Setup complete!\n")

    def train(self, num_iterations: Optional[int] = None):
        """
        Run training.

        Args:
            num_iterations: Override number of iterations (otherwise uses curriculum)
        """
        if self.networks is None:
            self.setup()

        self.training_start_time = time.time()

        # Determine total iterations
        if num_iterations is not None:
            total_iters = num_iterations
        elif self.curriculum:
            total_iters = sum(s.iterations for s in self.curriculum.stages)
        else:
            total_iters = 1000

        print(f"Starting training for {total_iters} iterations...")
        print("=" * 60)

        try:
            while self.iteration < total_iters:
                # Check curriculum advancement
                if self.curriculum and not self.curriculum.is_complete:
                    if self.curriculum.step():
                        # Advanced to new stage - update engine
                        self._update_engine_for_curriculum()

                self.iteration += 1

                # Training step
                metrics = self._training_step()

                # Update monitoring
                self.convergence_monitor.update(
                    regret_loss=metrics['regret_loss'],
                    strategy_loss=metrics['strategy_loss'],
                    utility=metrics['utility']
                )

                # Print progress
                if self.iteration % self.config.print_every == 0:
                    self._print_progress(metrics)

                # Evaluation
                if self.iteration % self.config.eval_every == 0:
                    self._run_evaluation()

                # Save checkpoint
                if self.iteration % self.config.save_every == 0:
                    self._save_checkpoint()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        # Final save
        self._save_checkpoint(final=True)
        print("\nTraining complete!")
        self._print_final_summary()

    def _training_step(self) -> Dict[str, float]:
        """Run one training iteration."""
        # Get strategy function
        def strategy_fn(features, legal_mask):
            return self.networks.get_strategy(features, legal_mask)

        # Collect samples
        regret_samples, strategy_samples = self.data_generator.generate_samples(
            strategy_fn=strategy_fn,
            num_traversals=self.config.traversals_per_iter,
            iteration=self.iteration
        )

        self.total_traversals += self.config.traversals_per_iter

        # Add to memory
        for features, regrets, mask, iter_num in regret_samples:
            self.memory.add_regret_sample(features, regrets, mask, iter_num)

        for features, strategy, mask, iter_num, reach in strategy_samples:
            self.memory.add_strategy_sample(features, strategy, mask, iter_num, reach)

        # Train networks
        regret_loss = 0.0
        strategy_loss = 0.0

        if len(self.memory.regret_buffer) >= self.config.batch_size:
            for _ in range(self.config.train_steps_per_iter):
                rl = self._train_regret_network()
                sl = self._train_strategy_network()
                regret_loss += rl
                strategy_loss += sl

            regret_loss /= self.config.train_steps_per_iter
            strategy_loss /= self.config.train_steps_per_iter

        self.stats.add_regret_loss(regret_loss)
        self.stats.add_strategy_loss(strategy_loss)

        # Compute average utility from samples
        utility = np.mean([s[1].mean() for s in regret_samples]) if regret_samples else 0.0

        return {
            'regret_loss': regret_loss,
            'strategy_loss': strategy_loss,
            'utility': utility,
            'num_regret_samples': len(regret_samples),
            'num_strategy_samples': len(strategy_samples)
        }

    def _train_regret_network(self) -> float:
        """Train regret network on a batch."""
        states, regrets, masks, iterations = self.memory.sample_regret_batch(
            self.config.batch_size
        )

        if len(states) == 0:
            return 0.0

        states_t = torch.FloatTensor(states).to(self.config.device)
        regrets_t = torch.FloatTensor(regrets).to(self.config.device)
        iterations_t = torch.FloatTensor(iterations).to(self.config.device)

        return self.networks.train_regret_net(states_t, regrets_t, iterations_t)

    def _train_strategy_network(self) -> float:
        """Train strategy network on a batch."""
        states, strategies, masks, iterations = self.memory.sample_strategy_batch(
            self.config.batch_size
        )

        if len(states) == 0:
            return 0.0

        states_t = torch.FloatTensor(states).to(self.config.device)
        strategies_t = torch.FloatTensor(strategies).to(self.config.device)
        masks_t = torch.BoolTensor(masks).to(self.config.device)
        iterations_t = torch.FloatTensor(iterations).to(self.config.device)

        return self.networks.train_avg_strategy_net(
            states_t, strategies_t, masks_t, iterations_t
        )

    def _update_engine_for_curriculum(self):
        """Update poker engine when curriculum advances."""
        stage = self.curriculum.current_stage

        self.engine = PreflopPokerEngine(
            num_players=stage.num_players,
            starting_stack=(stage.min_stack + stage.max_stack) / 2
        )

        self.data_generator = SelfPlayDataGenerator(
            poker_engine=self.engine,
            exploration=self.config.exploration
        )

    def _print_progress(self, metrics: Dict):
        """Print training progress."""
        elapsed = time.time() - self.training_start_time
        iters_per_sec = self.iteration / elapsed if elapsed > 0 else 0

        print(f"\nIteration {self.iteration}")
        print(f"  Time: {elapsed:.1f}s ({iters_per_sec:.2f} it/s)")
        print(f"  Traversals: {self.total_traversals:,}")
        print(f"  Regret samples: {self.memory.num_regret_samples:,}")
        print(f"  Strategy samples: {self.memory.num_strategy_samples:,}")
        print(f"  Regret Loss: {metrics['regret_loss']:.4f}")
        print(f"  Strategy Loss: {metrics['strategy_loss']:.4f}")

        if self.curriculum:
            print(f"  Curriculum: {self.curriculum.get_summary()}")

    def _run_evaluation(self):
        """Run strategy evaluation."""
        print("\n  Running evaluation...")

        def strategy_fn(features, legal_mask):
            return self.networks.get_average_strategy(features, legal_mask)

        evaluator = StrategyEvaluator(
            strategy_fn=strategy_fn,
            num_players=self.engine.num_players,
            starting_stack=self.engine.starting_stack
        )

        try:
            exploit = evaluator.compute_exploitability(num_samples=200)
            print(f"  Exploitability: {exploit.exploitability:.2f} mBB/hand")
        except Exception as e:
            print(f"  Evaluation error: {e}")

    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        suffix = "final" if final else f"iter{self.iteration}"
        path = os.path.join(self.config.checkpoint_dir, f"gto_trainer_{suffix}.pt")

        torch.save({
            'networks': {
                'regret_net': self.networks.regret_net.state_dict(),
                'avg_strategy_net': self.networks.avg_strategy_net.state_dict(),
            },
            'iteration': self.iteration,
            'total_traversals': self.total_traversals,
            'config': self.config.to_dict(),
            'curriculum_stage': self.curriculum.current_stage_idx if self.curriculum else None,
        }, path)

        print(f"  Saved checkpoint: {path}")

    def _print_final_summary(self):
        """Print final training summary."""
        elapsed = time.time() - self.training_start_time

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total iterations: {self.iteration}")
        print(f"Total traversals: {self.total_traversals:,}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Regret samples collected: {self.memory.num_regret_samples:,}")
        print(f"Strategy samples collected: {self.memory.num_strategy_samples:,}")
        print(f"Final regret loss: {self.stats.avg_regret_loss:.4f}")
        print(f"Final strategy loss: {self.stats.avg_strategy_loss:.4f}")

    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        if self.networks is None:
            self.setup()

        checkpoint = torch.load(path, map_location=self.config.device)

        self.networks.regret_net.load_state_dict(checkpoint['networks']['regret_net'])
        self.networks.avg_strategy_net.load_state_dict(checkpoint['networks']['avg_strategy_net'])
        self.iteration = checkpoint['iteration']
        self.total_traversals = checkpoint['total_traversals']

        if 'curriculum_stage' in checkpoint and checkpoint['curriculum_stage'] is not None:
            if self.curriculum:
                self.curriculum.current_stage_idx = checkpoint['curriculum_stage']

        print(f"Loaded checkpoint from {path}")
        print(f"  Iteration: {self.iteration}")
        print(f"  Traversals: {self.total_traversals}")

    def get_strategy(self, state: GameState) -> np.ndarray:
        """Get GTO strategy for a game state."""
        info_set = InformationSet.from_game_state(state)
        features = info_set.to_feature_vector()
        legal_mask = get_legal_actions_mask(state)
        return self.networks.get_average_strategy(features, legal_mask)

    def analyze_strategy(self) -> str:
        """Generate analysis report of current strategy."""
        def strategy_fn(features, legal_mask):
            return self.networks.get_average_strategy(features, legal_mask)

        return analyze_preflop_strategy(
            strategy_fn=strategy_fn,
            num_players=self.engine.num_players,
            stack_size=self.engine.starting_stack
        )


def train_gto(
    quick: bool = False,
    num_iterations: Optional[int] = None,
    device: str = 'cpu',
    checkpoint_dir: str = 'checkpoints'
) -> GTOTrainer:
    """
    Convenience function to train a GTO model.

    Args:
        quick: Use quick curriculum for testing
        num_iterations: Override number of iterations
        device: 'cpu' or 'cuda'
        checkpoint_dir: Directory for checkpoints

    Returns:
        Trained GTOTrainer
    """
    config = TrainingConfig(
        quick_mode=quick,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    trainer = GTOTrainer(config)
    trainer.train(num_iterations=num_iterations)

    return trainer
