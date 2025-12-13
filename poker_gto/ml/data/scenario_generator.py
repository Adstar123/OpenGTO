"""Scenario generation for training data."""

from typing import Dict, List, Optional
import random
from dataclasses import dataclass

from poker_gto.core.game_state import GameState, GameConfig
from poker_gto.core.position import Position, PositionManager
from poker_gto.ml.features.feature_extractors import PreflopFeatureExtractor


@dataclass
class ScenarioConfig:
    """Configuration for scenario generation."""
    player_counts: List[int] = None
    stack_sizes: List[float] = None
    
    def __post_init__(self):
        if self.player_counts is None:
            self.player_counts = [6]
        if self.stack_sizes is None:
            self.stack_sizes = [100.0]


class PreflopScenarioGenerator:
    """Generates preflop training scenarios with balanced actions."""
    
    def __init__(self, config: Optional[ScenarioConfig] = None):
        """Initialize generator with configuration.
        
        Args:
            config: Scenario generation configuration
        """
        self.config = config or ScenarioConfig()
        self.feature_extractor = PreflopFeatureExtractor()
    
    def generate_balanced_scenarios(self, num_scenarios: int) -> List[Dict]:
        """Generate balanced training scenarios.
        
        Ensures reasonable distribution of fold/call/raise/check actions.
        
        Args:
            num_scenarios: Total number of scenarios to generate
            
        Returns:
            List of scenario dictionaries
        """
        # Adjust target distribution to include check
        scenarios_per_action = num_scenarios // 4
        target_counts = {
            'fold': scenarios_per_action,
            'call': scenarios_per_action,
            'raise': scenarios_per_action,
            'check': scenarios_per_action
        }
        generated_counts = {'fold': 0, 'call': 0, 'raise': 0, 'check': 0}
        
        scenarios = []
        max_attempts = num_scenarios * 50
        attempts = 0
        
        while sum(generated_counts.values()) < num_scenarios and attempts < max_attempts:
            attempts += 1
            
            # Generate a scenario
            scenario = self.generate_single_scenario()
            if not scenario:
                continue
            
            action = scenario['optimal_action']['action']
            
            # Only accept if we need more of this action
            if generated_counts[action] < target_counts[action]:
                scenarios.append(scenario)
                generated_counts[action] += 1
        
        # If we couldn't generate enough check scenarios, fill with others
        remaining = num_scenarios - len(scenarios)
        if remaining > 0:
            for _ in range(remaining):
                scenario = self.generate_single_scenario()
                if scenario:
                    scenarios.append(scenario)
        
        return scenarios
    
    def generate_single_scenario(self) -> Optional[Dict]:
        """Generate a single poker scenario.
        
        Returns:
            Scenario dictionary or None if generation failed
        """
        # Random configuration
        player_count = random.choice(self.config.player_counts)
        stack_size = random.choice(self.config.stack_sizes)
        
        # Create game
        config = GameConfig(
            player_count=player_count,
            starting_stack=stack_size
        )
        game_state = GameState(config)
        game_state.deal_hole_cards()
        
        # Pick random position to evaluate
        positions = PositionManager.get_positions_for_player_count(player_count)
        position = random.choice(positions)
        player = game_state.get_player_by_position(position)
        
        if not player or not player.hole_cards:
            return None
        
        # Generate different scenarios
        scenario_type = random.choice(['open', 'vs_raise', 'vs_limp'])
        
        if scenario_type == 'open':
            facing_raise = False
            pot_size = 1.5
            bet_to_call = 0.0
        elif scenario_type == 'vs_raise':
            facing_raise = True
            pot_size = 3.5
            bet_to_call = 3.0
        else:  # vs_limp - only valid for BB
            if position != Position.BIG_BLIND:
                # If not BB, generate a different scenario
                facing_raise = random.choice([True, False])
                pot_size = 3.5 if facing_raise else 1.5
                bet_to_call = 3.0 if facing_raise else 0.0
            else:
                # BB facing a limp
                facing_raise = False
                pot_size = 2.0  # SB completed
                bet_to_call = 0.0  # No additional bet to call
        
        # Create scenario data
        scenario_data = {
            'position': position.abbreviation,
            'hole_cards': player.hole_cards.to_string_notation(),
            'player_count': player_count,
            'facing_raise': facing_raise,
            'pot_size': pot_size,
            'bet_to_call': bet_to_call,
            'pot_odds': bet_to_call / (pot_size + bet_to_call) if bet_to_call > 0 else 0.0,
            'stack_ratio': 1.0,
            'num_players': player_count,
        }
        
        # Extract features
        features = self.feature_extractor.extract_from_scenario(scenario_data)
        
        # Determine optimal action
        optimal_action = self._determine_optimal_action(
            features,
            position,
            facing_raise,
            bet_to_call,
            pot_size
        )
        
        return {
            'features': features,
            'optimal_action': optimal_action,
            'context': scenario_data
        }
    
    def _determine_optimal_action(
        self,
        features: Dict[str, float],
        position: Position,
        facing_raise: bool,
        bet_to_call: float,
        pot_size: float
    ) -> Dict:
        """Determine optimal action based on simplified GTO logic.
        
        Args:
            features: Extracted features
            position: Player position
            facing_raise: Whether facing a raise
            bet_to_call: Amount to call
            pot_size: Current pot size
            
        Returns:
            Dictionary with 'action' and 'size'
        """
        hand_strength = features.get('hand_strength', 0.5)
        position_strength = features.get('position_strength', 0.5)
        
        # Special case: BB facing a limp (can check)
        if position == Position.BIG_BLIND and not facing_raise and bet_to_call == 0.0:
            # BB can check with weak hands or raise with strong hands
            if hand_strength < 0.3:
                action = 'check'
            elif hand_strength > 0.7:
                action = 'raise'
            else:
                # Mixed strategy for medium hands
                action = 'check' if random.random() < 0.6 else 'raise'
        elif facing_raise:
            # Facing a raise - tighter ranges
            if hand_strength > 0.8:  # Premium hands
                action = 'raise' if random.random() < 0.6 else 'call'
            elif hand_strength > 0.5:  # Medium hands
                action = 'call' if random.random() < 0.7 else 'fold'
            else:  # Weak hands
                action = 'fold'
        else:
            # Opening - position-dependent
            threshold = 0.7 - (position_strength * 0.4)
            if hand_strength > threshold:
                action = 'raise'
            else:
                action = 'fold'
        
        # Determine raise size
        raise_size = 0.0
        if action == 'raise':
            if facing_raise:
                raise_size = random.uniform(3.0, 4.0)  # 3-bet
            else:
                raise_size = random.uniform(2.2, 3.0)  # Open
        
        return {
            'action': action,
            'size': raise_size
        }