from typing import Optional
from dataclasses import dataclass
from .card import HoleCards
from .position import Position
from .action import Action, ActionHistory

@dataclass
class Player:
    """Represents a poker player with stack, position, and actions."""
    
    position: Position
    stack_size: float  # in big blinds
    hole_cards: Optional[HoleCards] = None
    has_acted: bool = False
    is_active: bool = True
    current_bet: float = 0.0  # amount committed to current betting round
    
    def __post_init__(self):
        if self.stack_size <= 0:
            raise ValueError("Stack size must be positive")
    
    @property
    def is_all_in(self) -> bool:
        """Check if player is all-in."""
        return self.stack_size == 0 and self.is_active
    
    @property
    def effective_stack(self) -> float:
        """Get effective stack size (remaining stack)."""
        return max(0, self.stack_size)
    
    def can_bet(self, amount: float) -> bool:
        """Check if player can make a bet of given amount."""
        return self.is_active and self.effective_stack >= amount
    
    def commit_chips(self, amount: float) -> float:
        """Commit chips to pot, returns actual amount committed."""
        if not self.is_active:
            return 0.0
        
        actual_amount = min(amount, self.effective_stack)
        self.stack_size -= actual_amount
        self.current_bet += actual_amount
        
        if self.stack_size == 0:
            # Player is now all-in but still active
            pass
        
        return actual_amount
    
    def fold(self):
        """Player folds and becomes inactive."""
        self.is_active = False
        self.has_acted = True
    
    def reset_for_new_round(self):
        """Reset player state for new betting round."""
        self.current_bet = 0.0
        self.has_acted = False

# core/game_state.py
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .player import Player
from .position import Position, PositionManager
from .action import Action, ActionType, ActionHistory
from .card import Deck, HoleCards

@dataclass
class GameConfig:
    """Configuration for a poker game."""
    player_count: int
    small_blind: float = 0.5
    big_blind: float = 1.0
    starting_stack: float = 100.0  # in big blinds
    
    def __post_init__(self):
        if not 2 <= self.player_count <= 6:
            raise ValueError("Player count must be between 2 and 6")
        if self.small_blind >= self.big_blind:
            raise ValueError("Small blind must be less than big blind")

class GameState:
    """Manages the complete state of a poker game."""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.deck = Deck()
        self.players: List[Player] = []
        self.action_history = ActionHistory()
        self.pot_size = 0.0
        self.current_player_index = 0
        self.game_phase = "preflop"  # preflop, flop, turn, river
        self.is_hand_complete = False
        
        self._initialize_players()
        self._post_blinds()
    
    def _initialize_players(self):
        """Initialize players with positions and starting stacks."""
        positions = PositionManager.get_positions_for_player_count(self.config.player_count)
        
        for position in positions:
            player = Player(
                position=position,
                stack_size=self.config.starting_stack
            )
            self.players.append(player)
    
    def _post_blinds(self):
        """Post small and big blinds."""
        sb_player = self.get_player_by_position(Position.SMALL_BLIND)
        bb_player = self.get_player_by_position(Position.BIG_BLIND)
        
        if sb_player:
            sb_amount = sb_player.commit_chips(self.config.small_blind)
            self.pot_size += sb_amount
            sb_action = Action(ActionType.BET, sb_amount, Position.SMALL_BLIND)
            self.action_history.add_action(sb_action)
        
        if bb_player:
            bb_amount = bb_player.commit_chips(self.config.big_blind)
            self.pot_size += bb_amount
            bb_action = Action(ActionType.BET, bb_amount, Position.BIG_BLIND)
            self.action_history.add_action(bb_action)
    
    def deal_hole_cards(self):
        """Deal hole cards to all active players."""
        self.deck.reset()
        for player in self.players:
            if player.is_active:
                player.hole_cards = self.deck.deal_hole_cards()
    
    def get_player_by_position(self, position: Position) -> Optional[Player]:
        """Get player by their position."""
        for player in self.players:
            if player.position == position:
                return player
        return None
    
    def get_current_player(self) -> Optional[Player]:
        """Get the player whose turn it is to act."""
        if self.is_hand_complete:
            return None
        
        active_players = [p for p in self.players if p.is_active]
        if len(active_players) <= 1:
            self.is_hand_complete = True
            return None
        
        # Find next player to act (simplified for preflop)
        for i in range(len(self.players)):
            player_idx = (self.current_player_index + i) % len(self.players)
            player = self.players[player_idx]
            if player.is_active and not player.has_acted:
                return player
        
        return None
    
    def get_valid_actions(self, player: Player) -> List[ActionType]:
        """Get list of valid actions for a player."""
        if not player.is_active or player.has_acted:
            return []
        
        valid_actions = [ActionType.FOLD]
        current_bet = self.action_history.current_bet_amount
        
        # Can always call if there's a bet to call
        if current_bet > player.current_bet:
            valid_actions.append(ActionType.CALL)
        else:
            valid_actions.append(ActionType.CHECK)
        
        # Can raise if not all-in
        if player.effective_stack > 0:
            valid_actions.append(ActionType.RAISE)
        
        return valid_actions
    
    def execute_action(self, player: Player, action: Action) -> bool:
        """Execute a player action and update game state."""
        if action.action_type not in self.get_valid_actions(player):
            return False
        
        if action.action_type == ActionType.FOLD:
            player.fold()
        
        elif action.action_type == ActionType.CALL:
            call_amount = self.action_history.current_bet_amount - player.current_bet
            actual_amount = player.commit_chips(call_amount)
            self.pot_size += actual_amount
            action.amount = actual_amount
        
        elif action.action_type == ActionType.RAISE:
            # Ensure minimum raise
            current_bet = self.action_history.current_bet_amount
            min_raise = current_bet * 2 if current_bet > 0 else self.config.big_blind
            raise_amount = max(action.amount, min_raise)
            
            total_amount = player.commit_chips(raise_amount)
            self.pot_size += total_amount
            action.amount = total_amount
        
        elif action.action_type == ActionType.CHECK:
            # No chips committed for check
            pass
        
        player.has_acted = True
        self.action_history.add_action(action)
        self._advance_to_next_player()
        
        return True
    
    def _advance_to_next_player(self):
        """Move to the next player in turn order."""
        for i in range(1, len(self.players)):
            next_idx = (self.current_player_index + i) % len(self.players)
            next_player = self.players[next_idx]
            if next_player.is_active and not next_player.has_acted:
                self.current_player_index = next_idx
                return
        
        # All players have acted, round is complete
        self._complete_betting_round()
    
    def _complete_betting_round(self):
        """Complete the current betting round."""
        active_players = [p for p in self.players if p.is_active]
        
        if len(active_players) <= 1:
            self.is_hand_complete = True
        else:
            # For now, just mark hand as complete (Phase 1 is preflop only)
            self.is_hand_complete = True
    
    def get_state_features(self) -> Dict:
        """Extract features for neural network input."""
        current_player = self.get_current_player()
        if not current_player:
            return {}
        
        # Position encoding
        position_features = self._encode_position(current_player.position)
        
        # Hole cards encoding (simplified for now)
        hole_cards_features = self._encode_hole_cards(current_player.hole_cards)
        
        # Action history features
        action_features = self._encode_action_history()
        
        # Stack and pot features
        stack_features = {
            'effective_stack': current_player.effective_stack / self.config.big_blind,
            'pot_size': self.pot_size / self.config.big_blind,
            'current_bet_to_call': (self.action_history.current_bet_amount - current_player.current_bet) / self.config.big_blind
        }
        
        return {
            **position_features,
            **hole_cards_features,
            **action_features,
            **stack_features,
            'player_count': self.config.player_count
        }
    
    def _encode_position(self, position: Position) -> Dict:
        """One-hot encode position."""
        all_positions = PositionManager.get_positions_for_player_count(self.config.player_count)
        encoding = {}
        
        for pos in all_positions:
            encoding[f'position_{pos.abbreviation}'] = 1.0 if pos == position else 0.0
        
        # Also include relative position
        encoding['relative_position'] = PositionManager.get_relative_position(position, self.config.player_count)
        
        return encoding
    
    def _encode_hole_cards(self, hole_cards: Optional[HoleCards]) -> Dict:
        """Encode hole cards features."""
        if not hole_cards:
            return {
                'is_pocket_pair': 0.0,
                'is_suited': 0.0,
                'high_card_rank': 0.0,
                'card_strength': 0.0
            }
        
        # Simplified encoding for Phase 1
        high_rank_value = hole_cards.high_card_rank.value / 14.0  # Normalize to 0-1
        card_strength = self._calculate_preflop_strength(hole_cards)
        
        return {
            'is_pocket_pair': 1.0 if hole_cards.is_pocket_pair else 0.0,
            'is_suited': 1.0 if hole_cards.is_suited else 0.0,
            'high_card_rank': high_rank_value,
            'card_strength': card_strength
        }
    
    def _calculate_preflop_strength(self, hole_cards: HoleCards) -> float:
        """Calculate simplified preflop hand strength (0-1)."""
        # Very basic hand strength calculation
        # This would be replaced with more sophisticated analysis later
        
        high_rank = hole_cards.high_card_rank.value
        low_rank = hole_cards.low_card_rank.value
        
        if hole_cards.is_pocket_pair:
            # Pocket pairs are strong, with higher pairs being stronger
            return 0.5 + (high_rank / 28.0)  # Scale to 0.5-1.0
        
        # Non-pairs: consider rank gap and suitedness
        rank_sum = high_rank + low_rank
        max_sum = 14 + 13  # AA (non-pair impossible, but AK)
        min_sum = 3 + 2    # 32
        
        base_strength = (rank_sum - min_sum) / (max_sum - min_sum)
        
        # Bonus for suited cards
        if hole_cards.is_suited:
            base_strength += 0.1
        
        # Penalty for large gaps
        gap = high_rank - low_rank - 1
        gap_penalty = gap * 0.02
        
        return max(0.0, min(1.0, base_strength - gap_penalty))
    
    def _encode_action_history(self) -> Dict:
        """Encode action history features."""
        actions = self.action_history.actions
        
        return {
            'raises_count': len(self.action_history.get_actions_by_type(ActionType.RAISE)),
            'calls_count': len(self.action_history.get_actions_by_type(ActionType.CALL)),
            'folds_count': len(self.action_history.get_actions_by_type(ActionType.FOLD)),
            'total_actions': len(actions),
            'pot_odds': self._calculate_pot_odds()
        }
    
    def _calculate_pot_odds(self) -> float:
        """Calculate current pot odds."""
        current_player = self.get_current_player()
        if not current_player:
            return 0.0
        
        amount_to_call = self.action_history.current_bet_amount - current_player.current_bet
        if amount_to_call <= 0:
            return 0.0
        
        return amount_to_call / (self.pot_size + amount_to_call)