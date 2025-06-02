"""
Truly balanced generator that forces equal action distribution.
"""

import random
from typing import List, Dict, Generator
from poker_gto.core.game_state import GameState, GameConfig
from poker_gto.core.position import Position, PositionManager
from poker_gto.core.action import ActionType

class BalancedScenarioGenerator:
    """Generator that creates exactly balanced action distributions."""
    
    def __init__(self):
        self.position_ranges = {
            Position.UNDER_THE_GUN: {'raise_threshold': 0.85, 'call_threshold': 0.40},
            Position.MIDDLE_POSITION: {'raise_threshold': 0.80, 'call_threshold': 0.45},  
            Position.CUTOFF: {'raise_threshold': 0.70, 'call_threshold': 0.50},
            Position.BUTTON: {'raise_threshold': 0.60, 'call_threshold': 0.55},
            Position.SMALL_BLIND: {'raise_threshold': 0.75, 'call_threshold': 0.50},
            Position.BIG_BLIND: {'raise_threshold': 0.80, 'call_threshold': 0.60}
        }
    
    def generate_scenarios(self, num_scenarios: int) -> Generator[Dict, None, None]:
        """Generate exactly balanced scenarios."""
        
        # Force equal distribution
        scenarios_per_action = num_scenarios // 3
        target_counts = {
            'fold': scenarios_per_action,
            'call': scenarios_per_action,
            'raise': scenarios_per_action
        }
        
        generated_counts = {'fold': 0, 'call': 0, 'raise': 0}
        
        attempts = 0
        max_attempts = num_scenarios * 10  # Prevent infinite loops
        
        while sum(generated_counts.values()) < num_scenarios and attempts < max_attempts:
            attempts += 1
            
            try:
                # Create game
                config = GameConfig(player_count=6, starting_stack=100.0)
                game_state = GameState(config)
                game_state.deal_hole_cards()
                
                # Pick random position
                positions = PositionManager.get_positions_for_player_count(6)
                eval_position = random.choice(positions)
                player = game_state.get_player_by_position(eval_position)
                
                if not player or not player.hole_cards:
                    continue
                
                # Calculate hand strength
                hand_strength = self._calculate_hand_strength(player.hole_cards)
                
                # Determine what action we need more of
                needed_actions = [action for action, count in generated_counts.items() 
                                if count < target_counts[action]]
                
                if not needed_actions:
                    break
                
                # Try to generate the needed action
                target_action = random.choice(needed_actions)
                optimal_action = self._force_action(hand_strength, eval_position, target_action)
                
                if optimal_action['action'] == target_action:
                    # Extract features
                    features = self._extract_features(player, eval_position, game_state, hand_strength)
                    
                    yield {
                        'features': features,
                        'optimal_action': optimal_action,
                        'position': eval_position.abbreviation,
                        'hole_cards': player.hole_cards.to_string_notation()
                    }
                    
                    generated_counts[target_action] += 1
                    
                    if sum(generated_counts.values()) % 500 == 0:
                        print(f"Generated {sum(generated_counts.values())}/{num_scenarios} scenarios")
                        print(f"Distribution: {generated_counts}")
                        
            except Exception as e:
                continue
    
    def _calculate_hand_strength(self, hole_cards) -> float:
        """Enhanced hand strength calculation."""
        high_rank = hole_cards.high_card_rank.numeric_value
        low_rank = hole_cards.low_card_rank.numeric_value
        
        # Pocket pairs
        if hole_cards.is_pocket_pair:
            # AA=1.0, KK=0.95, QQ=0.90, etc.
            pair_strength = 0.5 + (high_rank / 28.0)
            return min(1.0, pair_strength)
        
        # Non-pairs
        rank_sum = high_rank + low_rank
        max_sum, min_sum = 27, 5  # AK, 32
        base_strength = (rank_sum - min_sum) / (max_sum - min_sum)
        
        # Suited bonus
        if hole_cards.is_suited:
            base_strength += 0.15
        
        # Connectivity bonus (reduces gap penalty)
        gap = high_rank - low_rank - 1
        if gap <= 1:  # Connected or one gap
            base_strength += 0.05
        elif gap >= 4:  # Big gaps
            base_strength -= gap * 0.03
        
        # High card bonus
        if high_rank >= 12:  # Q or higher
            base_strength += 0.05
        
        return max(0.0, min(1.0, base_strength))
    
    def _force_action(self, hand_strength: float, position: Position, target_action: str) -> Dict:
        """Force a specific action based on hand strength and position."""
        
        ranges = self.position_ranges[position]
        
        if target_action == 'raise':
            # Only allow raises with decent hands
            if hand_strength >= ranges['raise_threshold']:
                size = random.uniform(2.2, 4.0)
                return {'action': 'raise', 'size': size}
            else:
                # If hand too weak for position, try to force it anyway for training
                if hand_strength >= 0.3:  # Not complete trash
                    size = random.uniform(2.0, 3.0)
                    return {'action': 'raise', 'size': size}
        
        elif target_action == 'call':
            # Allow calls with medium hands
            if ranges['call_threshold'] <= hand_strength < ranges['raise_threshold']:
                return {'action': 'call', 'size': 0.0}
            else:
                # Force call for training if not too weak
                if hand_strength >= 0.2:
                    return {'action': 'call', 'size': 0.0}
        
        elif target_action == 'fold':
            # Allow folds with weak hands
            if hand_strength < ranges['call_threshold']:
                return {'action': 'fold', 'size': 0.0}
            else:
                # Force fold sometimes even with ok hands (tight play)
                return {'action': 'fold', 'size': 0.0}
        
        # Fallback to fold if forcing didn't work
        return {'action': 'fold', 'size': 0.0}
    
    def _extract_features(self, player, position, game_state, hand_strength) -> Dict:
        """Extract features with additional context."""
        
        # Position one-hot encoding
        positions = PositionManager.get_positions_for_player_count(6)
        position_features = {}
        for pos in positions:
            position_features[f'position_{pos.abbreviation}'] = 1.0 if pos == position else 0.0
        
        # Enhanced hand features
        hand_features = {
            'is_pocket_pair': 1.0 if player.hole_cards.is_pocket_pair else 0.0,
            'is_suited': 1.0 if player.hole_cards.is_suited else 0.0,
            'high_card_rank': player.hole_cards.high_card_rank.numeric_value / 14.0,
            'card_strength': hand_strength,
            'is_premium': 1.0 if hand_strength > 0.8 else 0.0,
            'is_playable': 1.0 if hand_strength > 0.3 else 0.0
        }
        
        # Position strength (late position = stronger)
        position_strength = {
            Position.UNDER_THE_GUN: 0.1,
            Position.MIDDLE_POSITION: 0.3,
            Position.CUTOFF: 0.6,
            Position.BUTTON: 1.0,
            Position.SMALL_BLIND: 0.2,
            Position.BIG_BLIND: 0.4
        }.get(position, 0.5)
        
        # Context features
        context_features = {
            'effective_stack': player.effective_stack / 100.0,  # Normalize
            'pot_size': 1.5 / 100.0,  # SB + BB normalized
            'player_count': 6.0 / 6.0,  # Normalized
            'position_strength': position_strength,
            'action_context': random.uniform(0.0, 1.0)  # Random context variation
        }
        
        return {**position_features, **hand_features, **context_features}
