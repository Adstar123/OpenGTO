"""
Improved scenario generator that creates realistic poker situations
with proper action sequences and betting history.
"""

import random
import itertools
from typing import List, Dict, Generator, Tuple
from collections import namedtuple

from poker_gto.core.game_state import GameState, GameConfig
from poker_gto.core.card import Card, Rank, Suit, HoleCards
from poker_gto.core.position import Position, PositionManager
from poker_gto.core.action import ActionType, Action

# Define a betting action for simulation
BettingAction = namedtuple('BettingAction', ['position', 'action_type', 'amount'])

class RealisticScenarioGenerator:
    """Generates realistic poker scenarios with proper action sequences."""
    
    def __init__(self, player_counts: List[int] = [6]):
        self.player_counts = player_counts
        self.position_ranges = self._load_position_ranges()
    
    def _load_position_ranges(self) -> Dict:
        """Load realistic preflop ranges by position."""
        # These are approximate GTO ranges - much more realistic than before
        return {
            Position.UNDER_THE_GUN: {
                'open_raise': 0.12,  # 12% of hands
                'strong_hands': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AQs', 'AJs', 'AKo', 'AQo'],
                'medium_hands': ['88', '77', 'ATs', 'A9s', 'KQs', 'KJs', 'QJs', 'JTs'],
                'sizing': [2.5, 3.0]
            },
            Position.MIDDLE_POSITION: {
                'open_raise': 0.15,
                'strong_hands': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AKs', 'AQs', 'AJs', 'ATs', 'AKo', 'AQo'],
                'medium_hands': ['77', '66', 'A9s', 'A8s', 'KQs', 'KJs', 'KTs', 'QJs', 'JTs', 'T9s'],
                'sizing': [2.5, 3.0, 3.5]
            },
            Position.CUTOFF: {
                'open_raise': 0.25,
                'strong_hands': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'AKo', 'AQo', 'AJo'],
                'medium_hands': ['66', '55', 'A8s', 'A7s', 'A6s', 'A5s', 'KQs', 'KJs', 'KTs', 'K9s', 'QJs', 'QTs', 'JTs', 'T9s'],
                'sizing': [2.2, 2.5, 3.0]
            },
            Position.BUTTON: {
                'open_raise': 0.45,  # Very wide on button
                'call_vs_raise': 0.15,
                'sizing': [2.2, 2.5, 3.0]
            },
            Position.SMALL_BLIND: {
                'open_raise': 0.35,  # Vs unopened pot
                'call_vs_raise': 0.25,
                'sizing': [3.0, 3.5, 4.0]
            },
            Position.BIG_BLIND: {
                'call_vs_raise': 0.35,  # Defending range
                '3bet_vs_raise': 0.08,
                'sizing': [3.0, 3.5, 4.0]
            }
        }
    
    def generate_realistic_scenarios(self, num_scenarios: int) -> Generator[Dict, None, None]:
        """Generate realistic training scenarios with proper betting sequences."""
        
        scenarios_generated = 0
        
        while scenarios_generated < num_scenarios:
            try:
                # Create random game configuration
                player_count = random.choice(self.player_counts)
                config = GameConfig(player_count=player_count)
                game_state = GameState(config)
                
                # Deal cards to all players
                game_state.deal_hole_cards()
                
                # Simulate realistic preflop action sequence
                action_sequence = self._simulate_preflop_action(game_state)
                
                if not action_sequence:
                    continue
                
                # Choose a random decision point in the action sequence
                decision_point = self._choose_decision_point(action_sequence, game_state)
                
                if decision_point:
                    scenario = self._create_scenario_from_decision_point(
                        decision_point, action_sequence, game_state
                    )
                    
                    if scenario:
                        yield scenario
                        scenarios_generated += 1
                        
                        if scenarios_generated % 1000 == 0:
                            print(f"Generated {scenarios_generated}/{num_scenarios} realistic scenarios")
                        
            except Exception as e:
                # Skip problematic scenarios
                continue
    
    def _simulate_preflop_action(self, game_state: GameState) -> List[BettingAction]:
        """Simulate a realistic preflop betting sequence."""
        
        positions = PositionManager.get_positions_for_player_count(game_state.config.player_count)
        action_sequence = []
        
        # Start with blinds
        action_sequence.append(BettingAction(Position.SMALL_BLIND, ActionType.BET, 0.5))
        action_sequence.append(BettingAction(Position.BIG_BLIND, ActionType.BET, 1.0))
        
        current_bet = 1.0
        players_to_act = [p for p in positions if p not in [Position.SMALL_BLIND, Position.BIG_BLIND]]
        
        # Simulate action going around the table
        opened = False
        
        for position in players_to_act:
            player = game_state.get_player_by_position(position)
            if not player or not player.hole_cards:
                continue
            
            hand_strength = self._evaluate_hand_strength(player.hole_cards)
            
            if not opened:
                # First to act decision
                open_prob = self.position_ranges.get(position, {}).get('open_raise', 0.1)
                
                if hand_strength > (1.0 - open_prob):
                    # Open raise
                    sizing = random.choice(self.position_ranges.get(position, {}).get('sizing', [2.5, 3.0]))
                    action_sequence.append(BettingAction(position, ActionType.RAISE, sizing))
                    current_bet = sizing
                    opened = True
                else:
                    # Fold
                    action_sequence.append(BettingAction(position, ActionType.FOLD, 0.0))
            else:
                # Facing a raise
                if hand_strength > 0.85:  # Very strong hands
                    if random.random() < 0.3:  # Sometimes 3-bet
                        sizing = current_bet * random.uniform(2.2, 3.0)
                        action_sequence.append(BettingAction(position, ActionType.RAISE, sizing))
                        current_bet = sizing
                    else:  # Call
                        action_sequence.append(BettingAction(position, ActionType.CALL, current_bet))
                elif hand_strength > 0.6:  # Medium hands
                    if random.random() < 0.2:  # Sometimes call
                        action_sequence.append(BettingAction(position, ActionType.CALL, current_bet))
                    else:
                        action_sequence.append(BettingAction(position, ActionType.FOLD, 0.0))
                else:  # Weak hands
                    action_sequence.append(BettingAction(position, ActionType.FOLD, 0.0))
        
        # Blinds get to act if there was a raise
        if opened and current_bet > 1.0:
            # SB action
            sb_player = game_state.get_player_by_position(Position.SMALL_BLIND)
            if sb_player and sb_player.hole_cards:
                hand_strength = self._evaluate_hand_strength(sb_player.hole_cards)
                call_prob = self.position_ranges[Position.SMALL_BLIND].get('call_vs_raise', 0.2)
                
                if hand_strength > (1.0 - call_prob):
                    action_sequence.append(BettingAction(Position.SMALL_BLIND, ActionType.CALL, current_bet))
                else:
                    action_sequence.append(BettingAction(Position.SMALL_BLIND, ActionType.FOLD, 0.0))
            
            # BB action
            bb_player = game_state.get_player_by_position(Position.BIG_BLIND)
            if bb_player and bb_player.hole_cards:
                hand_strength = self._evaluate_hand_strength(bb_player.hole_cards)
                call_prob = self.position_ranges[Position.BIG_BLIND].get('call_vs_raise', 0.3)
                threebet_prob = self.position_ranges[Position.BIG_BLIND].get('3bet_vs_raise', 0.08)
                
                if hand_strength > 0.9:  # Very strong - sometimes 3-bet
                    if random.random() < threebet_prob * 3:  # Higher chance with strong hands
                        sizing = current_bet * random.uniform(2.5, 3.5)
                        action_sequence.append(BettingAction(Position.BIG_BLIND, ActionType.RAISE, sizing))
                    else:
                        action_sequence.append(BettingAction(Position.BIG_BLIND, ActionType.CALL, current_bet))
                elif hand_strength > (1.0 - call_prob):
                    action_sequence.append(BettingAction(Position.BIG_BLIND, ActionType.CALL, current_bet))
                else:
                    action_sequence.append(BettingAction(Position.BIG_BLIND, ActionType.FOLD, 0.0))
        
        return action_sequence
    
    def _evaluate_hand_strength(self, hole_cards: HoleCards) -> float:
        """Quick hand strength evaluation (0-1)."""
        # This is the same as before but could be improved
        high_rank = hole_cards.high_card_rank.numeric_value
        low_rank = hole_cards.low_card_rank.numeric_value
        
        if hole_cards.is_pocket_pair:
            return 0.5 + (high_rank / 28.0)
        
        rank_sum = high_rank + low_rank
        max_sum, min_sum = 27, 5  # AK, 32
        base_strength = (rank_sum - min_sum) / (max_sum - min_sum)
        
        if hole_cards.is_suited:
            base_strength += 0.1
        
        gap = high_rank - low_rank - 1
        gap_penalty = gap * 0.02
        
        return max(0.0, min(1.0, base_strength - gap_penalty))
    
    def _choose_decision_point(self, action_sequence: List[BettingAction], game_state: GameState) -> Dict:
        """Choose a meaningful decision point for training."""
        
        # Find positions that haven't acted yet or need to make interesting decisions
        positions = PositionManager.get_positions_for_player_count(game_state.config.player_count)
        
        # Look for interesting spots:
        # 1. Facing a raise
        # 2. First to act in certain positions
        # 3. Defending blinds
        
        acted_positions = {action.position for action in action_sequence}
        
        for position in positions:
            if position not in acted_positions:
                player = game_state.get_player_by_position(position)
                if player and player.hole_cards:
                    # Create decision point
                    return {
                        'position': position,
                        'player': player,
                        'action_sequence': action_sequence,
                        'game_state': game_state
                    }
        
        return None
    
    def _create_scenario_from_decision_point(self, decision_point: Dict, 
                                           action_sequence: List[BettingAction], 
                                           game_state: GameState) -> Dict:
        """Create training scenario from decision point."""
        
        position = decision_point['position']
        player = decision_point['player']
        
        # Calculate pot size from action sequence
        pot_size = sum(action.amount for action in action_sequence)
        
        # Determine current bet to call
        current_bet = 0.0
        for action in reversed(action_sequence):
            if action.action_type in [ActionType.BET, ActionType.RAISE]:
                current_bet = action.amount
                break
        
        # Extract features
        features = self._extract_features(player, position, action_sequence, pot_size, current_bet, game_state)
        
        # Determine optimal action based on hand strength and situation
        optimal_action = self._determine_optimal_action_realistic(
            player, position, action_sequence, pot_size, current_bet
        )
        
        return {
            'features': features,
            'optimal_action': optimal_action,
            'position': position.abbreviation,
            'hole_cards': player.hole_cards.to_string_notation(),
            'action_sequence': [{'pos': a.position.abbreviation, 'action': a.action_type.value, 'amount': a.amount} 
                              for a in action_sequence],
            'pot_size': pot_size,
            'current_bet': current_bet
        }
    
    def _extract_features(self, player, position, action_sequence, pot_size, current_bet, game_state) -> Dict:
        """Extract features for neural network."""
        
        # Position encoding
        position_features = {}
        all_positions = PositionManager.get_positions_for_player_count(game_state.config.player_count)
        for pos in all_positions:
            position_features[f'position_{pos.abbreviation}'] = 1.0 if pos == position else 0.0
        
        # Hole cards features
        hole_cards_features = {
            'is_pocket_pair': 1.0 if player.hole_cards.is_pocket_pair else 0.0,
            'is_suited': 1.0 if player.hole_cards.is_suited else 0.0,
            'high_card_rank': player.hole_cards.high_card_rank.numeric_value / 14.0,
            'card_strength': self._evaluate_hand_strength(player.hole_cards)
        }
        
        # Action history features (CRITICAL IMPROVEMENT)
        raises_count = sum(1 for a in action_sequence if a.action_type == ActionType.RAISE)
        calls_count = sum(1 for a in action_sequence if a.action_type == ActionType.CALL)
        folds_count = sum(1 for a in action_sequence if a.action_type == ActionType.FOLD)
        
        # Position of last raiser (important context)
        last_raiser_pos = 0.0
        for action in reversed(action_sequence):
            if action.action_type == ActionType.RAISE:
                pos_index = PositionManager.get_position_index(action.position, game_state.config.player_count)
                last_raiser_pos = pos_index / (game_state.config.player_count - 1)
                break
        
        action_features = {
            'raises_count': raises_count,
            'calls_count': calls_count, 
            'folds_count': folds_count,
            'total_actions': len(action_sequence),
            'last_raiser_position': last_raiser_pos,
            'facing_raise': 1.0 if current_bet > 1.0 else 0.0
        }
        
        # Pot and stack features
        effective_stack = player.effective_stack
        pot_odds = current_bet / (pot_size + current_bet) if current_bet > 0 else 0.0
        
        stack_features = {
            'effective_stack': effective_stack / game_state.config.big_blind,
            'pot_size': pot_size / game_state.config.big_blind,
            'current_bet_to_call': current_bet / game_state.config.big_blind,
            'pot_odds': pot_odds,
            'stack_to_pot_ratio': effective_stack / max(pot_size, 1.0)
        }
        
        return {
            **position_features,
            **hole_cards_features,
            **action_features,
            **stack_features,
            'player_count': game_state.config.player_count
        }
    
    def _determine_optimal_action_realistic(self, player, position, action_sequence, 
                                          pot_size, current_bet) -> Dict:
        """Determine realistic optimal action based on situation."""
        
        hand_strength = self._evaluate_hand_strength(player.hole_cards)
        
        # If no one has raised (current_bet <= 1.0), we can open or check/fold
        if current_bet <= 1.0:
            # First to act decisions
            position_ranges = self.position_ranges.get(position, {})
            open_threshold = 1.0 - position_ranges.get('open_raise', 0.1)
            
            if hand_strength > open_threshold:
                sizing = random.choice(position_ranges.get('sizing', [2.5, 3.0]))
                return {'action': 'raise', 'size': sizing}
            else:
                return {'action': 'fold', 'size': 0.0}
        
        # Facing a raise - much more nuanced decision
        else:
            pot_odds = current_bet / (pot_size + current_bet)
            
            # Strong hands (top 15%) - mostly call or raise
            if hand_strength > 0.85:
                if random.random() < 0.3:  # Sometimes 3-bet
                    size = current_bet * random.uniform(2.2, 3.0)
                    return {'action': 'raise', 'size': size}
                else:
                    return {'action': 'call', 'size': 0.0}
            
            # Medium-strong hands (15-40%) - depends on pot odds and position
            elif hand_strength > 0.6:
                call_threshold = pot_odds * 1.5  # Adjust based on pot odds
                if hand_strength > (0.6 + call_threshold):
                    return {'action': 'call', 'size': 0.0}
                else:
                    return {'action': 'fold', 'size': 0.0}
            
            # Weak hands - mostly fold, occasional bluff
            else:
                if position in [Position.BUTTON, Position.CUTOFF] and random.random() < 0.05:
                    # Rare bluff 3-bet from position
                    size = current_bet * random.uniform(2.2, 2.8)
                    return {'action': 'raise', 'size': size}
                else:
                    return {'action': 'fold', 'size': 0.0}
