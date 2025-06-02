import random
import itertools
from typing import List, Dict, Generator
from poker_gto.core.game_state import GameState, GameConfig
from poker_gto.core.card import Card, Rank, Suit, HoleCards
from poker_gto.core.position import Position, PositionManager
from poker_gto.core.action import ActionType

class PreflopScenarioGenerator:
    """Generates training scenarios for preflop play."""
    
    def __init__(self, player_counts: List[int] = [6], stack_sizes: List[float] = [100.0]):
        self.player_counts = player_counts
        self.stack_sizes = stack_sizes
        
        # Precompute all possible hole card combinations
        self.all_hole_cards = self._generate_all_hole_cards()
        
        # Basic preflop ranges (simplified for initial training)
        self.basic_ranges = self._load_basic_preflop_ranges()
    
    def _generate_all_hole_cards(self) -> List[HoleCards]:
        """Generate all possible hole card combinations."""
        all_cards = [Card(rank, suit) for rank, suit in itertools.product(Rank, Suit)]
        hole_cards = []
        
        for i, card1 in enumerate(all_cards):
            for card2 in all_cards[i+1:]:
                hole_cards.append(HoleCards(card1, card2))
        
        return hole_cards
    
    def _load_basic_preflop_ranges(self) -> Dict:
        """Load basic preflop ranges for different positions."""
        # Simplified ranges for initial training
        # In a real implementation, these would come from GTO solvers
        
        ranges = {
            Position.UNDER_THE_GUN: {
                'raise': ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AQs', 'AJs', 'AKo', 'AQo'],
                'call': [],
                'fold': 'everything_else'
            },
            Position.MIDDLE_POSITION: {
                'raise': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AQs', 'AJs', 'ATs', 'AKo', 'AQo', 'KQs'],
                'call': [],
                'fold': 'everything_else'
            },
            Position.CUTOFF: {
                'raise': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'AKo', 'AQo', 'AJo', 'KQs', 'KJs'],
                'call': [],
                'fold': 'everything_else'
            },
            Position.BUTTON: {
                'raise': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', 'AKs', 'AQs', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'AKo', 'AQo', 'AJo', 'ATo', 'KQs', 'KJs', 'KTs', 'QJs', 'JTs'],
                'call': [],
                'fold': 'everything_else'
            },
            Position.SMALL_BLIND: {
                'raise': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AKs', 'AQs', 'AJs', 'ATs', 'AKo', 'AQo'],
                'call': ['77', '66', '55', '44', '33', '22', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'AJo', 'ATo', 'KQs', 'KJs', 'KTs', 'QJs', 'JTs'],
                'fold': 'everything_else'
            },
            Position.BIG_BLIND: {
                'raise': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AQs', 'AJs', 'AKo'],
                'call': 'most_other_hands',  # BB can call with wide range
                'fold': 'weak_hands'
            }
        }
        
        return ranges
    
    def generate_scenarios(self, num_scenarios: int) -> Generator[Dict, None, None]:
        """Generate training scenarios."""
        
        for _ in range(num_scenarios):
            # Random game configuration
            player_count = random.choice(self.player_counts)
            stack_size = random.choice(self.stack_sizes)
            
            config = GameConfig(
                player_count=player_count,
                starting_stack=stack_size
            )
            
            # Create game state
            game_state = GameState(config)
            
            # Deal random hole cards
            game_state.deal_hole_cards()
            
            # Choose random position to evaluate
            positions = PositionManager.get_positions_for_player_count(player_count)
            eval_position = random.choice(positions)
            eval_player = game_state.get_player_by_position(eval_position)
            
            if not eval_player or not eval_player.hole_cards:
                continue
            
            # Generate some random action history (preflop only)
            self._generate_random_preflop_history(game_state, eval_position)
            
            # Extract features
            features = game_state.get_state_features()
            if not features:
                continue
            
            # Determine optimal action based on simplified ranges
            optimal_action = self._determine_optimal_action(
                eval_player.hole_cards, 
                eval_position, 
                game_state
            )
            
            yield {
                'features': features,
                'optimal_action': optimal_action,
                'position': eval_position.abbreviation,
                'hole_cards': eval_player.hole_cards.to_string_notation(),
                'player_count': player_count
            }
    
    def _generate_random_preflop_history(self, game_state: GameState, eval_position: Position):
        """Generate random but realistic preflop action history."""
        positions = PositionManager.get_positions_for_player_count(game_state.config.player_count)
        
        # Skip blinds (already posted)
        action_positions = [p for p in positions if p not in [Position.SMALL_BLIND, Position.BIG_BLIND]]
        
        # Stop before the evaluation position
        eval_index = positions.index(eval_position)
        
        for pos in action_positions:
            pos_index = positions.index(pos)
            if pos_index >= eval_index:
                break
            
            player = game_state.get_player_by_position(pos)
            if not player:
                continue
            
            # Random action based on position tendencies
            action_prob = random.random()
            
            if pos in [Position.UNDER_THE_GUN, Position.UNDER_THE_GUN_PLUS_ONE]:
                # Tight positions
                if action_prob < 0.7:
                    action_type = ActionType.FOLD
                elif action_prob < 0.95:
                    action_type = ActionType.CALL
                else:
                    action_type = ActionType.RAISE
            else:
                # Looser positions
                if action_prob < 0.5:
                    action_type = ActionType.FOLD
                elif action_prob < 0.8:
                    action_type = ActionType.CALL
                else:
                    action_type = ActionType.RAISE
            
            # Execute action (simplified)
            if action_type == ActionType.FOLD:
                player.fold()
            elif action_type == ActionType.CALL:
                amount = game_state.action_history.current_bet_amount - player.current_bet
                if amount > 0:
                    player.commit_chips(amount)
                    game_state.pot_size += amount
            elif action_type == ActionType.RAISE:
                raise_size = random.choice([2.5, 3.0, 3.5, 4.0]) * game_state.config.big_blind
                amount = player.commit_chips(raise_size)
                game_state.pot_size += amount
    
    def _determine_optimal_action(self, hole_cards: HoleCards, position: Position, game_state: GameState) -> Dict:
        """Determine optimal action based on simplified ranges."""
        
        hand_notation = hole_cards.to_string_notation()
        ranges = self.basic_ranges.get(position, {})
        
        # Check if hand is in raise range
        if hand_notation in ranges.get('raise', []):
            return {
                'action': 'raise',
                'size': random.choice([2.5, 3.0, 3.5])  # Random raise size
            }
        
        # Check if hand is in call range
        if hand_notation in ranges.get('call', []):
            return {'action': 'call', 'size': 0.0}
        
        # Special case for big blind
        if position == Position.BIG_BLIND:
            current_bet = game_state.action_history.current_bet_amount
            if current_bet <= game_state.config.big_blind * 3:  # Small raise
                # Call with medium strength hands
                card_strength = game_state._calculate_preflop_strength(hole_cards)
                if card_strength > 0.3:
                    return {'action': 'call', 'size': 0.0}
        
        # Default to fold
        return {'action': 'fold', 'size': 0.0}