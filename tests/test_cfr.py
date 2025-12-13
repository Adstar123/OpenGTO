"""
Tests for CFR and neural network components.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.card import Card, HoleCards
from src.game_state import GameState, Position, ActionType, Action
from src.poker_engine import PreflopPokerEngine
from src.information_set import (
    InformationSet, get_legal_actions_mask, action_idx_to_action,
    sample_action, NUM_ACTIONS
)
from src.neural_network import (
    StrategyNetwork, ValueNetwork, RegretNetwork,
    AverageStrategyNetwork, DeepCFRNetworks, regret_matching
)
from src.memory_buffer import DeepCFRMemory, ReservoirBuffer, TrainingStats
from src.cfr_engine import CFREngine, ExternalSamplingCFR, RegretStore
from src.deep_cfr import DeepCFRTrainer


def test_information_set():
    """Test information set creation and encoding."""
    print("Testing information set...")

    engine = PreflopPokerEngine(num_players=6, starting_stack=100.0)
    state = engine.create_game()

    # Create information set
    info_set = InformationSet.from_game_state(state)

    # Check basic properties
    assert info_set.position == state.current_player.position
    assert info_set.pot_size == state.pot
    assert info_set.num_players == 6

    # Test feature vector
    features = info_set.to_feature_vector()
    assert len(features) == InformationSet.feature_size()
    assert features.dtype == np.float32

    # Test key generation
    key = info_set.to_key()
    assert isinstance(key, str)
    assert len(key) > 0

    print("  Information set: PASSED")


def test_legal_actions_mask():
    """Test legal actions mask generation."""
    print("Testing legal actions mask...")

    engine = PreflopPokerEngine(num_players=6, starting_stack=100.0)
    state = engine.create_game()

    mask = get_legal_actions_mask(state)

    assert len(mask) == NUM_ACTIONS
    assert mask.dtype == bool

    # At the start, UTG should be able to fold, call, raise
    assert mask[0]  # fold
    assert mask[2]  # call
    # Should have some raise option
    assert mask[4] or mask[5]  # raise or all-in

    print("  Legal actions mask: PASSED")


def test_action_conversion():
    """Test action index to Action conversion."""
    print("Testing action conversion...")

    engine = PreflopPokerEngine(num_players=6, starting_stack=100.0)
    state = engine.create_game()

    # Test fold
    action = action_idx_to_action(0, state)
    assert action.action_type == ActionType.FOLD

    # Test call
    action = action_idx_to_action(2, state)
    assert action.action_type == ActionType.CALL

    # Test all-in
    action = action_idx_to_action(5, state)
    assert action.action_type == ActionType.ALL_IN

    print("  Action conversion: PASSED")


def test_regret_matching():
    """Test regret matching algorithm."""
    print("Testing regret matching...")

    # All positive regrets
    regrets = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    legal_mask = np.array([True, True, True, False, False, False])

    strategy = regret_matching(regrets, legal_mask)

    assert np.isclose(strategy.sum(), 1.0)
    assert strategy[2] > strategy[1] > strategy[0]  # Higher regret = higher prob
    assert strategy[3] == 0.0  # Illegal action

    # All non-positive regrets (should be uniform)
    regrets = np.array([-1.0, -2.0, 0.0, 0.0, 0.0, 0.0])
    strategy = regret_matching(regrets, legal_mask)

    assert np.isclose(strategy.sum(), 1.0)
    # Should be uniform over legal actions
    assert np.isclose(strategy[0], 1/3)
    assert np.isclose(strategy[1], 1/3)
    assert np.isclose(strategy[2], 1/3)

    print("  Regret matching: PASSED")


def test_strategy_network():
    """Test strategy network forward pass."""
    print("Testing strategy network...")

    input_size = InformationSet.feature_size()
    network = StrategyNetwork(
        input_size=input_size,
        hidden_sizes=(64, 64),
        num_actions=NUM_ACTIONS
    )

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, input_size)
    legal_mask = torch.ones(batch_size, NUM_ACTIONS, dtype=torch.bool)
    legal_mask[:, 3] = False  # Make bet illegal

    # Forward pass
    probs = network(x, legal_mask)

    assert probs.shape == (batch_size, NUM_ACTIONS)
    # Check probabilities sum to 1
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    # Check illegal action has 0 probability
    assert torch.allclose(probs[:, 3], torch.zeros(batch_size), atol=1e-5)

    print("  Strategy network: PASSED")


def test_regret_network():
    """Test regret network forward pass."""
    print("Testing regret network...")

    input_size = InformationSet.feature_size()
    network = RegretNetwork(
        input_size=input_size,
        hidden_sizes=(64, 64),
        num_actions=NUM_ACTIONS
    )

    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, input_size)

    # Forward pass
    regrets = network(x)

    assert regrets.shape == (batch_size, NUM_ACTIONS)

    # Test get_strategy method
    features = np.random.randn(input_size).astype(np.float32)
    legal_mask = np.array([True, True, True, False, True, True])

    strategy = network.get_strategy(features, legal_mask)

    assert len(strategy) == NUM_ACTIONS
    assert np.isclose(strategy.sum(), 1.0)
    assert strategy[3] == 0.0  # Illegal action

    print("  Regret network: PASSED")


def test_deep_cfr_networks():
    """Test DeepCFRNetworks container."""
    print("Testing DeepCFRNetworks...")

    input_size = InformationSet.feature_size()
    networks = DeepCFRNetworks(
        input_size=input_size,
        hidden_sizes=(64, 64),
        num_actions=NUM_ACTIONS,
        device='cpu'
    )

    # Test strategy retrieval
    features = np.random.randn(input_size).astype(np.float32)
    legal_mask = np.array([True, True, True, False, True, True])

    strategy = networks.get_strategy(features, legal_mask)
    assert len(strategy) == NUM_ACTIONS
    assert np.isclose(strategy.sum(), 1.0)

    avg_strategy = networks.get_average_strategy(features, legal_mask)
    assert len(avg_strategy) == NUM_ACTIONS
    assert np.isclose(avg_strategy.sum(), 1.0)

    print("  DeepCFRNetworks: PASSED")


def test_reservoir_buffer():
    """Test reservoir sampling buffer."""
    print("Testing reservoir buffer...")

    buffer = ReservoirBuffer(capacity=100)

    # Add samples
    for i in range(200):
        buffer.add(f"sample_{i}", weight=float(i + 1))

    assert len(buffer) == 100

    # Sample batch
    samples, weights = buffer.sample_batch(10)
    assert len(samples) == 10
    assert len(weights) == 10
    assert np.isclose(weights.sum(), 1.0)

    print("  Reservoir buffer: PASSED")


def test_deep_cfr_memory():
    """Test Deep CFR memory buffers."""
    print("Testing Deep CFR memory...")

    memory = DeepCFRMemory(
        regret_buffer_size=1000,
        strategy_buffer_size=1000
    )

    input_size = InformationSet.feature_size()

    # Add samples
    for i in range(100):
        features = np.random.randn(input_size).astype(np.float32)
        regrets = np.random.randn(NUM_ACTIONS).astype(np.float32)
        strategy = np.abs(np.random.randn(NUM_ACTIONS))
        strategy = strategy / strategy.sum()
        legal_mask = np.ones(NUM_ACTIONS, dtype=bool)

        memory.add_regret_sample(features, regrets, legal_mask, iteration=i+1)
        memory.add_strategy_sample(features, strategy, legal_mask, iteration=i+1, reach_prob=0.5)

    assert memory.num_regret_samples == 100
    assert memory.num_strategy_samples == 100

    # Sample batches
    states, regrets, masks, iters = memory.sample_regret_batch(32)
    assert states.shape[0] == 32
    assert regrets.shape == (32, NUM_ACTIONS)

    states, strategies, masks, iters = memory.sample_strategy_batch(32)
    assert states.shape[0] == 32
    assert strategies.shape == (32, NUM_ACTIONS)

    print("  Deep CFR memory: PASSED")


def test_tabular_cfr():
    """Test tabular CFR (small number of iterations)."""
    print("Testing tabular CFR...")

    engine = PreflopPokerEngine(num_players=2, starting_stack=10.0)
    cfr = ExternalSamplingCFR(poker_engine=engine, exploration=0.3)

    # Run a few iterations
    for _ in range(10):
        cfr.cfr_iteration()

    assert cfr.num_iterations == 10
    assert cfr.total_traversals > 0
    assert len(cfr.regret_store.cumulative_regrets) > 0

    # Get strategy for a state
    state = engine.create_game()
    strategy = cfr.get_strategy(state)

    assert len(strategy) == NUM_ACTIONS
    assert np.isclose(strategy.sum(), 1.0)

    print(f"  Tabular CFR: PASSED (traversals: {cfr.total_traversals})")


def test_deep_cfr_trainer_init():
    """Test Deep CFR trainer initialization."""
    print("Testing Deep CFR trainer initialization...")

    trainer = DeepCFRTrainer(
        num_players=2,
        starting_stack=20.0,
        device='cpu',
        regret_buffer_size=10000,
        strategy_buffer_size=10000,
        hidden_sizes=(64, 64)
    )

    assert trainer.num_players == 2
    assert trainer.starting_stack == 20.0
    assert trainer.iteration == 0

    print("  Deep CFR trainer init: PASSED")


def test_deep_cfr_short_training():
    """Test Deep CFR with very short training."""
    print("Testing Deep CFR short training...")

    trainer = DeepCFRTrainer(
        num_players=2,
        starting_stack=10.0,
        device='cpu',
        regret_buffer_size=10000,
        strategy_buffer_size=10000,
        hidden_sizes=(32, 32),
        exploration=0.5
    )

    # Run very short training
    trainer.train(
        num_iterations=3,
        traversals_per_iter=50,
        batch_size=32,
        train_steps_per_iter=5,
        print_every=1
    )

    assert trainer.iteration == 3
    assert trainer.total_traversals > 0

    # Test getting strategy
    state = trainer.poker_engine.create_game()
    strategy = trainer.get_strategy(state)

    assert len(strategy) == NUM_ACTIONS
    assert np.isclose(strategy.sum(), 1.0)

    print("  Deep CFR short training: PASSED")


def test_sample_action():
    """Test action sampling."""
    print("Testing action sampling...")

    strategy = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
    legal_mask = np.array([True, True, True, False, False, False])

    # Sample many times and check distribution
    samples = [sample_action(strategy, legal_mask) for _ in range(1000)]

    # Check only legal actions are sampled
    assert all(s < 3 for s in samples)

    # Check rough distribution
    counts = [samples.count(i) for i in range(3)]
    # Action 0 should be most common (50%)
    assert counts[0] > counts[1] > counts[2]

    print("  Action sampling: PASSED")


def run_all_tests():
    """Run all CFR tests."""
    print("=" * 50)
    print("CFR and Neural Network Tests")
    print("=" * 50)
    print()

    test_information_set()
    test_legal_actions_mask()
    test_action_conversion()
    test_regret_matching()
    test_strategy_network()
    test_regret_network()
    test_deep_cfr_networks()
    test_reservoir_buffer()
    test_deep_cfr_memory()
    test_tabular_cfr()
    test_deep_cfr_trainer_init()
    test_sample_action()

    print()
    print("Running short Deep CFR training test...")
    print("(This may take a minute)")
    print()
    test_deep_cfr_short_training()

    print()
    print("=" * 50)
    print("All CFR tests PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
