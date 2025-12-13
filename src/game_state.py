"""
Game State representation for poker preflop scenarios.
Encodes all information needed for the neural network.
"""
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np

from .card import HoleCards, hand_type_to_index


class Position(IntEnum):
    """
    Player positions for 6-max poker.
    Can be extended for other formats.
    """
    # 6-max positions (in action order preflop)
    UTG = 0      # Under the Gun (first to act preflop)
    HJ = 1       # Hijack
    CO = 2       # Cutoff
    BTN = 3      # Button (dealer)
    SB = 4       # Small Blind
    BB = 5       # Big Blind

    def __str__(self) -> str:
        return self.name

    @classmethod
    def get_positions(cls, num_players: int) -> List['Position']:
        """Get positions for a given number of players."""
        if num_players == 6:
            return list(cls)
        elif num_players == 2:
            # Heads up: BTN/SB acts first preflop, BB acts second
            return [cls.BTN, cls.BB]
        elif num_players == 3:
            return [cls.BTN, cls.SB, cls.BB]
        elif num_players == 9:
            # 9-max would need additional positions
            raise NotImplementedError("9-max not yet implemented")
        else:
            raise ValueError(f"Unsupported number of players: {num_players}")

    @classmethod
    def preflop_order(cls, num_players: int) -> List['Position']:
        """Get positions in preflop action order."""
        if num_players == 6:
            # UTG acts first, then HJ, CO, BTN, SB, BB
            return [cls.UTG, cls.HJ, cls.CO, cls.BTN, cls.SB, cls.BB]
        elif num_players == 2:
            # Heads up: BTN/SB acts first, BB second
            return [cls.BTN, cls.BB]
        elif num_players == 3:
            # BTN acts first, then SB, then BB
            return [cls.BTN, cls.SB, cls.BB]
        else:
            raise ValueError(f"Unsupported number of players: {num_players}")


class ActionType(IntEnum):
    """Types of actions a player can take."""
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3      # Opening bet (no prior bet)
    RAISE = 4    # Raising a previous bet
    ALL_IN = 5   # Special case for all-in

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class Action:
    """
    Represents a single action taken by a player.
    """
    action_type: ActionType
    amount: float = 0.0  # In big blinds
    player_position: Optional[Position] = None

    def __str__(self) -> str:
        if self.action_type in [ActionType.FOLD, ActionType.CHECK]:
            return f"{self.action_type}"
        elif self.action_type == ActionType.CALL:
            return f"call {self.amount:.1f}bb"
        elif self.action_type == ActionType.BET:
            return f"bet {self.amount:.1f}bb"
        elif self.action_type == ActionType.RAISE:
            return f"raise to {self.amount:.1f}bb"
        elif self.action_type == ActionType.ALL_IN:
            return f"all-in {self.amount:.1f}bb"
        return str(self.action_type)


@dataclass
class PlayerState:
    """
    State of a single player in the game.
    """
    position: Position
    stack: float  # Stack size in big blinds
    hole_cards: Optional[HoleCards] = None
    is_active: bool = True  # Still in the hand
    has_acted: bool = False
    current_bet: float = 0.0  # Amount already committed this round
    is_all_in: bool = False

    def effective_stack(self) -> float:
        """Return remaining stack that can be bet."""
        return self.stack - self.current_bet


@dataclass
class GameState:
    """
    Complete state of a poker game at any point.
    This is the main class used to represent game states for the neural network.
    """
    # Game configuration
    num_players: int
    small_blind: float = 0.5  # In big blinds (so SB = 0.5, BB = 1.0)
    big_blind: float = 1.0

    # Player states
    players: List[PlayerState] = field(default_factory=list)

    # Current game state
    pot: float = 0.0
    current_bet: float = 0.0  # Current bet to call
    min_raise: float = 1.0    # Minimum raise size
    action_history: List[Action] = field(default_factory=list)

    # Tracking
    current_player_idx: int = 0
    num_actions_this_round: int = 0
    is_complete: bool = False
    winners: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.players:
            self._initialize_players()

    def _initialize_players(self):
        """Initialize players with default stacks."""
        positions = Position.get_positions(self.num_players)
        self.players = [
            PlayerState(position=pos, stack=100.0)  # Default 100bb
            for pos in positions
        ]

    def set_stacks(self, stacks: List[float]):
        """Set stack sizes for all players."""
        for i, stack in enumerate(stacks):
            if i < len(self.players):
                self.players[i].stack = stack

    def get_player_by_position(self, position: Position) -> Optional[PlayerState]:
        """Get player at a specific position."""
        for player in self.players:
            if player.position == position:
                return player
        return None

    @property
    def current_player(self) -> PlayerState:
        """Get the current player to act."""
        return self.players[self.current_player_idx]

    @property
    def active_players(self) -> List[PlayerState]:
        """Get list of players still in the hand."""
        return [p for p in self.players if p.is_active]

    @property
    def players_to_act(self) -> int:
        """Count players who still need to act."""
        return sum(1 for p in self.players if p.is_active and not p.is_all_in
                   and (not p.has_acted or p.current_bet < self.current_bet))

    def get_pot_odds(self) -> float:
        """Calculate pot odds for calling."""
        call_amount = self.current_bet - self.current_player.current_bet
        if call_amount <= 0:
            return 0.0
        return call_amount / (self.pot + call_amount)

    def get_stack_to_pot_ratio(self) -> float:
        """Calculate stack to pot ratio for current player."""
        if self.pot == 0:
            return float('inf')
        return self.current_player.effective_stack() / self.pot

    def get_legal_actions(self) -> List[Action]:
        """
        Get all legal actions for the current player.
        Returns list of possible actions with amounts.
        """
        actions = []
        player = self.current_player

        if not player.is_active or player.is_all_in:
            return actions

        call_amount = self.current_bet - player.current_bet
        effective_stack = player.effective_stack()

        # Can always fold if there's a bet to face
        if call_amount > 0:
            actions.append(Action(ActionType.FOLD))

        # Check is only available if no bet to call
        if call_amount == 0:
            actions.append(Action(ActionType.CHECK))

        # Call if there's a bet and we have chips
        if call_amount > 0 and effective_stack > 0:
            if effective_stack <= call_amount:
                # All-in call
                actions.append(Action(ActionType.ALL_IN, amount=player.stack))
            else:
                actions.append(Action(ActionType.CALL, amount=self.current_bet))

        # Bet/Raise options
        if effective_stack > call_amount:
            remaining_after_call = effective_stack - call_amount

            if self.current_bet == 0:
                # Opening bet (no prior bet)
                # Standard bet sizes: 2x, 2.5x, 3x, 4x BB or all-in
                bet_sizes = [2.0, 2.5, 3.0, 4.0]
                for size in bet_sizes:
                    if size <= effective_stack:
                        actions.append(Action(ActionType.BET, amount=size))
            else:
                # Raise
                min_raise_to = self.current_bet + self.min_raise
                if min_raise_to <= player.stack:
                    # Standard raise sizes: 2.2x, 2.5x, 3x, 4x the current bet
                    raise_multipliers = [2.2, 2.5, 3.0, 4.0]
                    for mult in raise_multipliers:
                        raise_to = self.current_bet * mult
                        if min_raise_to <= raise_to <= player.stack:
                            actions.append(Action(ActionType.RAISE, amount=raise_to))

            # All-in is always an option if we have more than the call
            if effective_stack > call_amount:
                actions.append(Action(ActionType.ALL_IN, amount=player.stack))

        return actions

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert game state to feature vector for neural network input.

        Features:
        - Hand type encoding (169 one-hot or index)
        - Position encoding (one-hot)
        - Stack depths (normalized)
        - Pot size (normalized)
        - Current bet to call (normalized)
        - Action history encoding
        - Number of active players
        - Pot odds
        """
        features = []

        # 1. Hand type (169-dim one-hot)
        hand_features = np.zeros(169)
        if self.current_player.hole_cards:
            hand_idx = self.current_player.hole_cards.hand_type_index()
            hand_features[hand_idx] = 1.0
        features.extend(hand_features)

        # 2. Position (6-dim one-hot for 6-max)
        position_features = np.zeros(6)
        position_features[self.current_player.position.value] = 1.0
        features.extend(position_features)

        # 3. Stack sizes (normalized by 100bb)
        for player in self.players:
            features.append(player.stack / 100.0)
        # Pad to 6 players if needed
        while len(features) < 169 + 6 + 6:
            features.append(0.0)

        # 4. Pot size (normalized by 100bb)
        features.append(self.pot / 100.0)

        # 5. Current bet to call (normalized)
        call_amount = self.current_bet - self.current_player.current_bet
        features.append(call_amount / 100.0)

        # 6. Pot odds
        features.append(self.get_pot_odds())

        # 7. Stack to pot ratio (capped at 10)
        spr = min(self.get_stack_to_pot_ratio(), 10.0) / 10.0
        features.append(spr)

        # 8. Number of active players (normalized)
        features.append(len(self.active_players) / self.num_players)

        # 9. Number of players who have raised (counts aggression)
        num_raises = sum(1 for a in self.action_history
                         if a.action_type in [ActionType.RAISE, ActionType.BET])
        features.append(min(num_raises / 4.0, 1.0))

        # 10. Action history encoding (last 10 actions)
        # Each action: [position_one_hot(6), action_type_one_hot(6), amount_normalized(1)]
        action_encoding_size = 13  # 6 + 6 + 1
        max_actions = 10
        action_features = np.zeros(action_encoding_size * max_actions)

        for i, action in enumerate(self.action_history[-max_actions:]):
            base_idx = i * action_encoding_size
            # Position
            if action.player_position is not None:
                action_features[base_idx + action.player_position.value] = 1.0
            # Action type
            action_features[base_idx + 6 + action.action_type.value] = 1.0
            # Amount
            action_features[base_idx + 12] = action.amount / 100.0

        features.extend(action_features)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def feature_size() -> int:
        """Return the size of the feature vector."""
        # 169 (hand) + 6 (position) + 6 (stacks) + 1 (pot) + 1 (call) +
        # 1 (pot odds) + 1 (spr) + 1 (active players) + 1 (num raises) +
        # 130 (action history: 10 actions * 13 features)
        return 169 + 6 + 6 + 1 + 1 + 1 + 1 + 1 + 1 + 130

    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        import copy
        return copy.deepcopy(self)

    def __str__(self) -> str:
        lines = [
            f"Pot: {self.pot:.1f}bb | Current bet: {self.current_bet:.1f}bb",
            f"Players: {len(self.active_players)}/{self.num_players} active",
            f"To act: {self.current_player.position}"
        ]
        if self.current_player.hole_cards:
            lines.append(f"Hand: {self.current_player.hole_cards.pretty()}")
        if self.action_history:
            lines.append(f"Actions: {' -> '.join(str(a) for a in self.action_history)}")
        return '\n'.join(lines)
