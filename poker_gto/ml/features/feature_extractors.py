"""Feature extraction for poker game states."""

from typing import Dict, Optional
from poker_gto.core.game_state import GameState
from poker_gto.core.player import Player
from poker_gto.core.position import Position, PositionManager
from poker_gto.core.card import HoleCards


class PreflopFeatureExtractor:
    """Extracts features for preflop poker decisions.
    
    This class is responsible for converting game state into
    numerical features suitable for ML models.
    """
    
    # Hand strength lookup table (can be expanded)
    HAND_STRENGTHS = {
        'AA': 0.95, 'KK': 0.90, 'QQ': 0.85, 'JJ': 0.80, 'TT': 0.75,
        '99': 0.70, '88': 0.65, '77': 0.60, '66': 0.55, '55': 0.50,
        '44': 0.45, '33': 0.40, '22': 0.35,
        'AKs': 0.78, 'AKo': 0.73, 'AQs': 0.70, 'AQo': 0.65,
        'AJs': 0.65, 'AJo': 0.58, 'KQs': 0.62, 'KQo': 0.55,
        'KJs': 0.58, 'KJo': 0.52, 'QJs': 0.55, 'QJo': 0.48,
        'JTs': 0.52, 'JTo': 0.45, 'T9s': 0.48, 'T9o': 0.40,
        '98s': 0.45, '98o': 0.38, '87s': 0.42, '87o': 0.35,
        '76s': 0.40, '76o': 0.33, '65s': 0.38, '65o': 0.30,
        '72o': 0.15, '83o': 0.20, '94o': 0.25
    }
    
    @classmethod
    def extract_from_game_state(cls, game_state: GameState, player: Player) -> Dict[str, float]:
        """Extract features from game state for a specific player.
        
        Args:
            game_state: Current game state
            player: Player to extract features for
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Position features
        features.update(cls._extract_position_features(player.position, game_state.config.player_count))
        
        # Hole card features
        features.update(cls._extract_hole_card_features(player.hole_cards))
        
        # Game context features
        features.update(cls._extract_context_features(game_state, player))
        
        return features
    
    @classmethod
    def extract_from_scenario(cls, scenario: Dict) -> Dict[str, float]:
        """Extract features from a scenario dictionary.
        
        Args:
            scenario: Scenario dictionary with position, cards, etc.
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Get position from scenario
        position_str = scenario.get('position', 'BTN')
        position_map = {
            'UTG': Position.UNDER_THE_GUN,
            'MP': Position.MIDDLE_POSITION,
            'CO': Position.CUTOFF,
            'BTN': Position.BUTTON,
            'SB': Position.SMALL_BLIND,
            'BB': Position.BIG_BLIND
        }
        position = position_map.get(position_str, Position.BUTTON)
        
        # Position features
        player_count = scenario.get('player_count', 6)
        features.update(cls._extract_position_features(position, player_count))
        
        # Hole card features from string notation
        hole_cards_str = scenario.get('hole_cards', 'AKo')
        features.update(cls._extract_hole_card_features_from_string(hole_cards_str))
        
        # Context features from scenario
        features.update({
            'facing_raise': float(scenario.get('facing_raise', False)),
            'pot_size': scenario.get('pot_size', 1.5) / 10.0,
            'bet_to_call': scenario.get('bet_to_call', 0.0) / 5.0,
            'pot_odds': scenario.get('pot_odds', 0.0),
            'stack_ratio': scenario.get('stack_ratio', 1.0),
            'num_players': scenario.get('num_players', 6) / 6.0,
        })
        
        # Derive position strength
        position_strength = cls._get_position_strength(position)
        features['position_strength'] = position_strength
        
        # Derive hand categories
        hand_strength = features.get('hand_strength', 0.5)
        features['premium_hand'] = 1.0 if hand_strength > 0.8 else 0.0
        features['strong_hand'] = 1.0 if hand_strength > 0.6 else 0.0
        features['playable_hand'] = 1.0 if hand_strength > 0.3 else 0.0
        
        return features
    
    @classmethod
    def _extract_position_features(cls, position: Position, player_count: int) -> Dict[str, float]:
        """Extract position-related features."""
        features = {}
        
        # One-hot encode position
        all_positions = PositionManager.get_positions_for_player_count(6)  # Always use 6 positions
        for pos in all_positions:
            features[f'position_{pos.abbreviation}'] = 1.0 if pos == position else 0.0
        
        # Add position strength
        features['position_strength'] = cls._get_position_strength(position)
        
        return features
    
    @classmethod
    def _extract_hole_card_features(cls, hole_cards: Optional[HoleCards]) -> Dict[str, float]:
        """Extract features from hole cards object."""
        if not hole_cards:
            return {
                'is_pocket_pair': 0.0,
                'is_suited': 0.0,
                'hand_strength': 0.0,
                'high_card': 0.0,
                'premium_hand': 0.0,
                'strong_hand': 0.0,
                'playable_hand': 0.0,
            }
        
        # Basic features
        features = {
            'is_pocket_pair': 1.0 if hole_cards.is_pocket_pair else 0.0,
            'is_suited': 1.0 if hole_cards.is_suited else 0.0,
            'high_card': hole_cards.high_card_rank.numeric_value / 14.0,
        }
        
        # Calculate hand strength
        hand_notation = hole_cards.to_string_notation()
        hand_strength = cls._calculate_hand_strength(hole_cards, hand_notation)
        features['hand_strength'] = hand_strength
        
        # Categorize hand
        features['premium_hand'] = 1.0 if hand_strength > 0.8 else 0.0
        features['strong_hand'] = 1.0 if hand_strength > 0.6 else 0.0
        features['playable_hand'] = 1.0 if hand_strength > 0.3 else 0.0
        
        return features
    
    @classmethod
    def _extract_hole_card_features_from_string(cls, hole_cards_str: str) -> Dict[str, float]:
        """Extract features from hole cards string notation."""
        # Look up in strength table
        hand_strength = cls.HAND_STRENGTHS.get(hole_cards_str, 0.40)
        
        # Determine properties from string
        is_pair = len(set(hole_cards_str[:2])) == 1
        is_suited = 's' in hole_cards_str.lower()
        
        # Extract high card rank
        rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
        for i in range(2, 10):
            rank_values[str(i)] = i
        
        high_card_char = hole_cards_str[0]
        high_card_value = rank_values.get(high_card_char, 10) / 14.0
        
        return {
            'is_pocket_pair': 1.0 if is_pair else 0.0,
            'is_suited': 1.0 if is_suited else 0.0,
            'hand_strength': hand_strength,
            'high_card': high_card_value,
            'premium_hand': 1.0 if hand_strength > 0.8 else 0.0,
            'strong_hand': 1.0 if hand_strength > 0.6 else 0.0,
            'playable_hand': 1.0 if hand_strength > 0.3 else 0.0,
        }
    
    @classmethod
    def _extract_context_features(cls, game_state: GameState, player: Player) -> Dict[str, float]:
        """Extract game context features."""
        # Current betting context
        current_bet = game_state.action_history.current_bet_amount
        bet_to_call = current_bet - player.current_bet
        facing_raise = 1.0 if bet_to_call > 0 else 0.0
        
        # Pot odds calculation
        pot_odds = 0.0
        if bet_to_call > 0:
            pot_odds = bet_to_call / (game_state.pot_size + bet_to_call)
        
        # Number of active players
        active_players = sum(1 for p in game_state.players if p.is_active)
        
        return {
            'facing_raise': facing_raise,
            'pot_size': game_state.pot_size / 10.0,  # Normalize
            'bet_to_call': bet_to_call / 5.0,  # Normalize
            'pot_odds': pot_odds,
            'stack_ratio': player.effective_stack / game_state.config.starting_stack,
            'num_players': active_players / 6.0,  # Normalize to 6-max
        }
    
    @classmethod
    def _calculate_hand_strength(cls, hole_cards: HoleCards, hand_notation: str) -> float:
        """Calculate hand strength from hole cards."""
        # First check lookup table
        if hand_notation in cls.HAND_STRENGTHS:
            return cls.HAND_STRENGTHS[hand_notation]
        
        # Otherwise calculate algorithmically
        high_rank = hole_cards.high_card_rank.numeric_value
        low_rank = hole_cards.low_card_rank.numeric_value
        
        if hole_cards.is_pocket_pair:
            # Pocket pairs: scale from 0.35 (22) to 0.95 (AA)
            return 0.35 + (high_rank - 2) * 0.05
        
        # Non-pairs: consider connectivity and suitedness
        rank_sum = high_rank + low_rank
        base_strength = (rank_sum - 5) / (27 - 5)  # Normalize
        
        # Bonus for suited
        if hole_cards.is_suited:
            base_strength += 0.08
        
        # Penalty for gaps
        gap = high_rank - low_rank - 1
        gap_penalty = gap * 0.02
        
        return max(0.15, min(0.85, base_strength - gap_penalty))
    
    @classmethod
    def _get_position_strength(cls, position: Position) -> float:
        """Get position strength value (0-1, higher is better position)."""
        position_strengths = {
            Position.UNDER_THE_GUN: 0.1,
            Position.MIDDLE_POSITION: 0.3,
            Position.CUTOFF: 0.6,
            Position.BUTTON: 1.0,
            Position.SMALL_BLIND: 0.2,
            Position.BIG_BLIND: 0.4
        }
        return position_strengths.get(position, 0.5)