"""
Realistic poker scenario generator based on actual GTO principles.
"""

import random
from typing import List, Dict, Generator, Tuple
from poker_gto.core.game_state import GameState, GameConfig
from poker_gto.core.position import Position, PositionManager
from poker_gto.core.action import ActionType, Action
from poker_gto.core.card import Card, Rank, Suit, HoleCards

class RealisticPokerGenerator:
    """Generator that creates realistic poker scenarios with proper GTO logic."""
    
    def __init__(self):
        # Real preflop ranges based on actual GTO solutions
        self.preflop_ranges = {
            Position.UNDER_THE_GUN: {
                'open_raise': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo'],
                'call_vs_raise': ['88', '77', 'ATs', 'KQs', 'KJs', 'QJs'],
                'three_bet': ['AA', 'KK', 'QQ', 'AKs', 'AKo']
            },
            Position.MIDDLE_POSITION: {
                'open_raise': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo', 'ATs', 'KQs', 'KJs'],
                'call_vs_raise': ['77', '66', 'A9s', 'KTs', 'QTs', 'JTs'],
                'three_bet': ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo']
            },
            Position.CUTOFF: {
                'open_raise': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', 'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo', 'ATs', 'A9s', 'KQs', 'KJs', 'KTs', 'QJs', 'JTs'],
                'call_vs_raise': ['55', '44', 'A8s', 'A7s', 'KQo', 'QTs', 'J9s'],
                'three_bet': ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AKo', 'AQs']
            },
            Position.BUTTON: {
                'open_raise': ['22+', 'A2s+', 'K2s+', 'Q2s+', 'J2s+', 'T2s+', '92s+', '82s+', '72s+', '62s+', '52s+', 'A2o+', 'K2o+', 'Q5o+', 'J8o+', 'T8o+', '98o'],  # Very wide
                'call_vs_raise': ['22+', 'A2s+', 'K5s+', 'Q8s+', 'J8s+', 'T8s+', '98s', 'A8o+', 'KTo+'],
                'three_bet': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AKo', 'AQs', 'AJs', 'KQs']
            },
            Position.SMALL_BLIND: {
                'open_raise': ['22+', 'A2s+', 'K2s+', 'Q2s+', 'J6s+', 'T7s+', '97s+', '87s', '76s', '65s', 'A2o+', 'K8o+', 'Q9o+', 'JTo'],
                'call_vs_raise': ['22+', 'A2s+', 'K2s+', 'Q6s+', 'J8s+', 'T8s+', '98s', '87s', 'A2o+', 'K9o+', 'QTo+'],
                'three_bet': ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo', 'AQs', 'KQs']
            },
            Position.BIG_BLIND: {
                'call_vs_raise': ['22+', 'A2s+', 'K2s+', 'Q2s+', 'J2s+', 'T2s+', '92s+', '82s+', '72s+', '62s+', '52s+', '42s+', '32s', 'A2o+', 'K2o+', 'Q2o+', 'J6o+', 'T8o+', '98o'],  # Very wide in BB
                'three_bet': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AKs', 'AKo', 'AQs', 'AJs', 'KQs']
            }
        }
    
    def generate_realistic_scenarios(self, num_scenarios: int) -> Generator[Dict, None, None]:
        """Generate realistic poker scenarios with proper action sequences."""
        
        scenarios_generated = 0
        target_actions = {'fold': 0, 'call': 0, 'raise': 0}
        max_per_action = num_scenarios // 3
        
        while scenarios_generated < num_scenarios:
            try:
                # Create game
                config = GameConfig(player_count=6, starting_stack=100.0)
                game_state = GameState(config)
                game_state.deal_hole_cards()
                
                # Simulate realistic action sequence
                action_sequence, decision_spot = self._simulate_realistic_hand(game_state)
                
                if not decision_spot:
                    continue
                
                # Extract the decision scenario
                scenario = self._create_realistic_scenario(decision_spot, action_sequence, game_state)
                
                if scenario:
                    action = scenario['optimal_action']['action']
                    
                    # Ensure balanced distribution
                    if target_actions[action] < max_per_action:
                        yield scenario
                        target_actions[action] += 1
                        scenarios_generated += 1
                        
                        if scenarios_generated % 1000 == 0:
                            print(f"Generated {scenarios_generated}/{num_scenarios}")
                            print(f"Distribution: {target_actions}")
                            
            except Exception as e:
                continue
    
    def _simulate_realistic_hand(self, game_state: GameState) -> Tuple[List, Dict]:
        """Simulate a realistic preflop sequence."""
        
        positions = PositionManager.get_positions_for_player_count(6)
        action_sequence = []
        
        # Post blinds
        action_sequence.append(('SB', 'bet', 0.5))
        action_sequence.append(('BB', 'bet', 1.0))
        
        current_bet = 1.0
        opened = False
        active_players = positions[2:]  # Skip blinds for now
        
        # Action goes around the table
        for position in active_players:
            player = game_state.get_player_by_position(position)
            if not player or not player.hole_cards:
                continue
            
            hand_notation = player.hole_cards.to_string_notation()
            
            if not opened:
                # First to act - can open or fold
                if self._should_open_raise(hand_notation, position):
                    size = random.uniform(2.2, 3.0)
                    action_sequence.append((position.abbreviation, 'raise', size))
                    current_bet = size
                    opened = True
                else:
                    action_sequence.append((position.abbreviation, 'fold', 0.0))
            else:
                # Facing a raise
                action = self._decide_vs_raise(hand_notation, position, current_bet)
                action_sequence.append((position.abbreviation, action[0], action[1]))
                
                if action[0] == 'raise':
                    current_bet = action[1]
        
        # Blinds get to act if there was action
        if opened:
            for blind_pos in [Position.SMALL_BLIND, Position.BIG_BLIND]:
                player = game_state.get_player_by_position(blind_pos)
                if player and player.hole_cards:
                    hand_notation = player.hole_cards.to_string_notation()
                    action = self._decide_vs_raise(hand_notation, blind_pos, current_bet)
                    
                    # Create decision spot for this player
                    decision_spot = {
                        'position': blind_pos,
                        'player': player,
                        'hand_notation': hand_notation,
                        'current_bet': current_bet,
                        'pot_size': self._calculate_pot_size(action_sequence)
                    }
                    
                    return action_sequence, decision_spot
        
        return action_sequence, None
    
    def _should_open_raise(self, hand_notation: str, position: Position) -> bool:
        """Determine if hand should open raise from position."""
        ranges = self.preflop_ranges.get(position, {})
        open_hands = ranges.get('open_raise', [])
        
        # Check if hand is in opening range
        for hand_range in open_hands:
            if self._hand_in_range(hand_notation, hand_range):
                return True
        
        return False
    
    def _decide_vs_raise(self, hand_notation: str, position: Position, bet_size: float) -> Tuple[str, float]:
        """Decide action when facing a raise."""
        ranges = self.preflop_ranges.get(position, {})
        
        # Check for 3-bet hands
        three_bet_hands = ranges.get('three_bet', [])
        for hand_range in three_bet_hands:
            if self._hand_in_range(hand_notation, hand_range):
                # 3-bet with premium hands
                if random.random() < 0.7:  # Sometimes flat call even with premiums
                    return ('raise', bet_size * random.uniform(2.2, 3.0))
        
        # Check for calling hands
        call_hands = ranges.get('call_vs_raise', [])
        for hand_range in call_hands:
            if self._hand_in_range(hand_notation, hand_range):
                return ('call', 0.0)
        
        # Special case: Big blind gets great odds
        if position == Position.BIG_BLIND and bet_size <= 3.0:
            # Call with wider range in BB due to pot odds
            if self._hand_strength(hand_notation) > 0.15:
                return ('call', 0.0)
        
        return ('fold', 0.0)
    
    def _hand_in_range(self, hand_notation: str, hand_range: str) -> bool:
        """Check if hand is in a given range (simplified)."""
        if hand_range == hand_notation:
            return True
        
        # Handle ranges like "22+" for pocket pairs
        if '+' in hand_range and len(hand_range) == 3:
            if hand_notation[0] == hand_notation[1]:  # Pocket pair
                range_rank = hand_range[0]
                hand_rank = hand_notation[0]
                rank_order = 'AKQJT98765432'
                return rank_order.index(hand_rank) <= rank_order.index(range_rank)
        
        # Handle suited ranges like "A2s+"
        if '+' in hand_range and 's' in hand_range:
            if 's' in hand_notation.lower():
                # Simplified: just check if it's suited and reasonable
                return True
        
        # Handle specific hands
        simplified_hand = hand_notation.replace('o', '').replace('s', '')
        simplified_range = hand_range.replace('o', '').replace('s', '')
        
        return simplified_hand == simplified_range
    
    def _hand_strength(self, hand_notation: str) -> float:
        """Calculate realistic hand strength."""
        # Pocket pairs
        if hand_notation[0] == hand_notation[1]:
            rank = hand_notation[0]
            rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
            return 0.5 + (rank_values.get(rank, 2) / 28.0)
        
        # Non-pairs
        high_rank = hand_notation[0]
        low_rank = hand_notation[1]
        suited = 's' in hand_notation.lower()
        
        rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
        
        high_val = rank_values.get(high_rank, 2)
        low_val = rank_values.get(low_rank, 2)
        
        base = (high_val + low_val) / 28.0
        
        if suited:
            base += 0.1
        
        # Gap penalty
        gap = high_val - low_val - 1
        base -= gap * 0.02
        
        return max(0.0, min(1.0, base))
    
    def _calculate_pot_size(self, action_sequence: List) -> float:
        """Calculate current pot size from action sequence."""
        pot = 0.0
        for action in action_sequence:
            if action[1] in ['bet', 'raise', 'call']:
                pot += action[2]
        return pot
    
    def _create_realistic_scenario(self, decision_spot: Dict, action_sequence: List, game_state: GameState) -> Dict:
        """Create a realistic training scenario."""
        
        position = decision_spot['position']
        player = decision_spot['player']
        hand_notation = decision_spot['hand_notation']
        current_bet = decision_spot['current_bet']
        pot_size = decision_spot['pot_size']
        
        # Determine optimal action using realistic logic
        optimal_action = self._decide_vs_raise(hand_notation, position, current_bet)
        
        # Create rich features including action context
        features = self._extract_realistic_features(
            player, position, hand_notation, action_sequence, pot_size, current_bet, game_state
        )
        
        return {
            'features': features,
            'optimal_action': {
                'action': optimal_action[0],
                'size': optimal_action[1] if optimal_action[1] > 0 else 0.0
            },
            'context': {
                'position': position.abbreviation,
                'hole_cards': hand_notation,
                'action_sequence': action_sequence,
                'pot_size': pot_size,
                'current_bet': current_bet,
                'pot_odds': current_bet / (pot_size + current_bet) if current_bet > 0 else 0.0
            }
        }
    
    def _extract_realistic_features(self, player, position, hand_notation, action_sequence, pot_size, current_bet, game_state) -> Dict:
        """Extract comprehensive features including action context."""
        
        # Position encoding
        positions = PositionManager.get_positions_for_player_count(6)
        position_features = {}
        for pos in positions:
            position_features[f'position_{pos.abbreviation}'] = 1.0 if pos == position else 0.0
        
        # Hand strength features
        hand_strength = self._hand_strength(hand_notation)
        hand_features = {
            'is_pocket_pair': 1.0 if hand_notation[0] == hand_notation[1] else 0.0,
            'is_suited': 1.0 if 's' in hand_notation.lower() else 0.0,
            'hand_strength': hand_strength,
            'is_premium': 1.0 if hand_strength > 0.8 else 0.0,
            'is_strong': 1.0 if hand_strength > 0.6 else 0.0,
            'is_playable': 1.0 if hand_strength > 0.3 else 0.0
        }
        
        # Action context features (CRITICAL FOR POKER)
        raises_before = sum(1 for action in action_sequence if action[1] == 'raise')
        calls_before = sum(1 for action in action_sequence if action[1] == 'call')
        folds_before = sum(1 for action in action_sequence if action[1] == 'fold')
        
        # Who was the last raiser?
        last_raiser_position = 0.0
        for i, action in enumerate(reversed(action_sequence)):
            if action[1] == 'raise':
                last_raiser_position = (len(action_sequence) - i) / len(action_sequence)
                break
        
        action_features = {
            'facing_raise': 1.0 if current_bet > 1.0 else 0.0,
            'raises_before_me': raises_before,
            'calls_before_me': calls_before,
            'folds_before_me': folds_before,
            'last_raiser_position': last_raiser_position,
            'bet_size_ratio': current_bet / 3.0 if current_bet > 0 else 0.0,  # Normalize to typical 3BB open
        }
        
        # Pot and stack context
        pot_odds = current_bet / (pot_size + current_bet) if current_bet > 0 else 0.0
        context_features = {
            'pot_size_bb': pot_size,
            'current_bet_bb': current_bet,
            'pot_odds': pot_odds,
            'effective_stack': player.effective_stack,
            'stack_to_pot': player.effective_stack / max(pot_size, 1.0)
        }
        
        return {
            **position_features,
            **hand_features,
            **action_features,
            **context_features
        }
