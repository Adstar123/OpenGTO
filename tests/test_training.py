"""
Tests for the complete training pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tempfile

from src.poker_engine import PreflopPokerEngine
from src.self_play import ParallelGameSimulator, SelfPlayDataGenerator, BatchedCFRTraversal
from src.curriculum import (
    CurriculumScheduler, CurriculumStage, StackVariationGenerator,
    create_quick_curriculum, create_full_curriculum
)
from src.evaluation import StrategyEvaluator, ConvergenceMonitor, ExploitabilityResult
from src.trainer import GTOTrainer, TrainingConfig
from src.information_set import InformationSet, get_legal_actions_mask, NUM_ACTIONS


def test_parallel_game_simulator():
    """Test parallel game simulation."""
    print("Testing parallel game simulator...")

    simulator = ParallelGameSimulator(
        num_players=2,
        starting_stack=20.0,
        num_workers=2
    )

    # Test random games
    results = simulator.simulate_random_games(10)
    assert len(results) == 10

    for result in results:
        assert len(result.states) > 0
        assert result.states[-1].is_complete

    print("  Parallel simulator: PASSED")


def test_self_play_data_generator():
    """Test self-play data generation."""
    print("Testing self-play data generator...")

    engine = PreflopPokerEngine(num_players=2, starting_stack=15.0)
    generator = SelfPlayDataGenerator(engine, exploration=0.5)

    # Dummy strategy function (uniform)
    def uniform_strategy(features, legal_mask):
        probs = np.zeros(NUM_ACTIONS)
        num_legal = legal_mask.sum()
        if num_legal > 0:
            probs[legal_mask] = 1.0 / num_legal
        return probs

    regret_samples, strategy_samples = generator.generate_samples(
        strategy_fn=uniform_strategy,
        num_traversals=20,
        iteration=1
    )

    assert len(regret_samples) > 0
    assert len(strategy_samples) > 0

    # Check sample format
    features, regrets, mask, iter_num = regret_samples[0]
    assert len(features) == InformationSet.feature_size()
    assert len(regrets) == NUM_ACTIONS
    assert len(mask) == NUM_ACTIONS

    print(f"  Self-play generator: PASSED ({len(regret_samples)} regret, {len(strategy_samples)} strategy samples)")


def test_curriculum_stages():
    """Test curriculum stage functionality."""
    print("Testing curriculum stages...")

    stage = CurriculumStage(
        name="Test",
        num_players=2,
        min_stack=10.0,
        max_stack=20.0,
        iterations=100,
        traversals_per_iter=50,
        description="Test stage"
    )

    # Test random stack generation
    stacks = [stage.get_random_stack() for _ in range(100)]
    assert all(10.0 <= s <= 20.0 for s in stacks)

    # Test stack list
    stack_list = stage.get_stack_list()
    assert len(stack_list) == 2

    print("  Curriculum stages: PASSED")


def test_curriculum_scheduler():
    """Test curriculum scheduler."""
    print("Testing curriculum scheduler...")

    scheduler = create_quick_curriculum()

    assert not scheduler.is_complete
    assert scheduler.current_stage_idx == 0

    # Step through iterations
    initial_stage = scheduler.current_stage.name
    for _ in range(scheduler.current_stage.iterations - 1):
        scheduler.step()

    # Should still be on first stage
    assert scheduler.current_stage.name == initial_stage

    # One more step should advance
    scheduler.step()

    # Check progress
    progress = scheduler.get_progress()
    assert progress['stage_num'] == 2

    print("  Curriculum scheduler: PASSED")


def test_stack_variation_generator():
    """Test stack variation generator."""
    print("Testing stack variation generator...")

    gen = StackVariationGenerator(num_players=6, min_stack=20.0, max_stack=100.0)

    # Test uniform
    stacks = gen.uniform_stacks()
    assert len(stacks) == 6
    assert len(set(stacks)) == 1  # All same

    # Test varied
    stacks = gen.varied_stacks()
    assert len(stacks) == 6

    # Test one short
    stacks = gen.one_short_stack()
    assert len(stacks) == 6
    assert min(stacks) < max(stacks)

    # Test tournament
    stacks = gen.tournament_style()
    assert len(stacks) == 6

    print("  Stack variation: PASSED")


def test_convergence_monitor():
    """Test convergence monitoring."""
    print("Testing convergence monitor...")

    monitor = ConvergenceMonitor(check_interval=10)

    # Add some metrics
    for i in range(50):
        # Simulate decreasing loss
        regret_loss = 100.0 / (i + 1) + np.random.randn() * 0.1
        strategy_loss = 50.0 / (i + 1) + np.random.randn() * 0.1
        utility = np.random.randn()

        monitor.update(regret_loss, strategy_loss, utility)

    # Check metrics
    metrics = monitor.get_convergence_metrics()
    assert 'avg_regret_loss' in metrics
    assert 'regret_trend' in metrics
    assert metrics['regret_trend'] < 0  # Should be decreasing

    # Check summary
    summary = monitor.get_summary()
    assert 'Regret Loss' in summary

    print("  Convergence monitor: PASSED")


def test_strategy_evaluator():
    """Test strategy evaluation."""
    print("Testing strategy evaluator...")

    # Uniform strategy
    def uniform_strategy(features, legal_mask):
        probs = np.zeros(NUM_ACTIONS)
        num_legal = legal_mask.sum()
        if num_legal > 0:
            probs[legal_mask] = 1.0 / num_legal
        return probs

    evaluator = StrategyEvaluator(
        strategy_fn=uniform_strategy,
        num_players=2,
        starting_stack=20.0
    )

    # Test exploitability
    result = evaluator.compute_exploitability(num_samples=50)
    assert isinstance(result, ExploitabilityResult)
    assert result.num_hands_tested == 50

    print(f"  Strategy evaluator: PASSED (exploitability: {result.exploitability:.2f} mBB)")


def test_gto_trainer_init():
    """Test GTOTrainer initialization."""
    print("Testing GTOTrainer initialization...")

    config = TrainingConfig(
        quick_mode=True,
        device='cpu',
        checkpoint_dir=tempfile.mkdtemp()
    )

    trainer = GTOTrainer(config)
    trainer.setup()

    assert trainer.networks is not None
    assert trainer.memory is not None
    assert trainer.curriculum is not None
    assert trainer.engine is not None

    print("  GTOTrainer init: PASSED")


def test_gto_trainer_short_run():
    """Test GTOTrainer with very short training."""
    print("Testing GTOTrainer short training run...")
    print("(This may take a minute)")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainingConfig(
            quick_mode=True,
            device='cpu',
            checkpoint_dir=tmpdir,
            print_every=5,
            save_every=10,
            eval_every=10,
            batch_size=64,
            train_steps_per_iter=10,
            traversals_per_iter=50
        )

        trainer = GTOTrainer(config)
        trainer.train(num_iterations=15)

        assert trainer.iteration == 15
        assert trainer.total_traversals > 0

        # Test getting strategy
        state = trainer.engine.create_game()
        strategy = trainer.get_strategy(state)

        assert len(strategy) == NUM_ACTIONS
        assert np.isclose(strategy.sum(), 1.0)

        # Test checkpoint save/load
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
        trainer._save_checkpoint()

        # Load into new trainer
        trainer2 = GTOTrainer(config)
        trainer2.setup()

        # Find saved checkpoint
        saved_files = os.listdir(tmpdir)
        checkpoint_file = [f for f in saved_files if f.endswith('.pt')][0]
        trainer2.load_checkpoint(os.path.join(tmpdir, checkpoint_file))

        assert trainer2.iteration == trainer.iteration

    print("  GTOTrainer short run: PASSED")


def run_all_tests():
    """Run all training pipeline tests."""
    print("=" * 60)
    print("Training Pipeline Tests")
    print("=" * 60)
    print()

    test_parallel_game_simulator()
    test_self_play_data_generator()
    test_curriculum_stages()
    test_curriculum_scheduler()
    test_stack_variation_generator()
    test_convergence_monitor()
    test_strategy_evaluator()
    test_gto_trainer_init()

    print()
    print("Running short training test...")
    print("(This tests the full training loop)")
    print()

    test_gto_trainer_short_run()

    print()
    print("=" * 60)
    print("All training pipeline tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
