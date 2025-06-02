"""
Ultra-simple scenario generator for testing.
"""

import random
from typing import List, Dict, Generator
from poker_gto.core.game_state import GameState, GameConfig
from poker_gto.core.position import Position, PositionManager
from poker_gto.core.action import ActionType

class SimpleScenarioGenerator:
    """Simple, fast scenario generator."""
    
    def __init__(self):
        self.position_open_rates = {
            Position.UNDER_THE_GUN: 0.10,
            Position.MIDDLE_POSITION: 0.15,  
            Position.CUTOFF: 0.25,
            Position.BUTTON: 0.45,
            Position.SMALL_BLIND: 0.30,
            Position.BIG_BLIND: 0.20
        }
    
    def generate_scenarios(self, num_scenarios: int) -> Generator[Dict, None, None]:
        """Generate simple training scenarios quickly."""
        
        for i in range(num_scenarios):
            try:
                # Create game
                config = GameConfig(player_count=6, starting_stack=100.0)
                game_state = GameState(config)
                game_state.deal_hole_cards()
                
                # Pick random position to evaluate
                positions = PositionManager.get_positions_for_player_count(6)
                eval_position = random.choice(positions)
                player = game_state.get_player_by_position(eval_position)
                
                if not player or not player.hole_cards:
                    continue
                
                # Simple hand strength
                hand_strength = self._calculate_hand_strength(player.hole_cards)
                
                # Simple decision logic
                open_rate = self.position_open_rates.get(eval_position, 0.15)
                
                if hand_strength > (1.0 - open_rate):
                    optimal_action = {'action': 'raise', 'size': 2.5}
                elif hand_strength > 0.3 and eval_position in [Position.BIG_BLIND, Position.SMALL_BLIND]:
                    optimal_action = {'action': 'call', 'size': 0.0}
                else:
                    optimal_action = {'action': 'fold', 'size': 0.0}
                
                # Extract simple features
                features = self._extract_simple_features(player, eval_position, game_state)
                
                yield {
                    'features': features,
                    'optimal_action': optimal_action,
                    'position': eval_position.abbreviation,
                    'hole_cards': player.hole_cards.to_string_notation()
                }
                
                if (i + 1) % 1000 == 0:
                    print(f"Generated {i + 1}/{num_scenarios} scenarios")
                    
            except Exception as e:
                # Skip problematic scenarios
                continue
    
    def _calculate_hand_strength(self, hole_cards) -> float:
        """Simple hand strength calculation."""
        high_rank = hole_cards.high_card_rank.numeric_value
        low_rank = hole_cards.low_card_rank.numeric_value
        
        if hole_cards.is_pocket_pair:
            return 0.5 + (high_rank / 28.0)
        
        rank_sum = high_rank + low_rank
        base_strength = (rank_sum - 5) / (27 - 5)  # Normalize 32 to AK
        
        if hole_cards.is_suited:
            base_strength += 0.1
        
        gap = high_rank - low_rank - 1
        base_strength -= gap * 0.02
        
        return max(0.0, min(1.0, base_strength))
    
    def _extract_simple_features(self, player, position, game_state) -> Dict:
        """Extract simple features for neural network."""
        
        # Position one-hot encoding
        positions = PositionManager.get_positions_for_player_count(6)
        position_features = {}
        for pos in positions:
            position_features[f'position_{pos.abbreviation}'] = 1.0 if pos == position else 0.0
        
        # Hand features
        hand_features = {
            'is_pocket_pair': 1.0 if player.hole_cards.is_pocket_pair else 0.0,
            'is_suited': 1.0 if player.hole_cards.is_suited else 0.0,
            'high_card_rank': player.hole_cards.high_card_rank.numeric_value / 14.0,
            'card_strength': self._calculate_hand_strength(player.hole_cards)
        }
        
        # Simple context
        context_features = {
            'effective_stack': player.effective_stack,
            'pot_size': 1.5,  # SB + BB
            'player_count': 6
        }
        
        return {**position_features, **hand_features, **context_features}
