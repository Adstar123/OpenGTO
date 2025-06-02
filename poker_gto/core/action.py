from enum import Enum
from typing import Optional, List
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from poker_gto.core.position import Position

class ActionType(Enum):
    FOLD = "fold"
    CALL = "call"
    RAISE = "raise"
    CHECK = "check"
    BET = "bet"

class Action:
    """Represents a poker action with amount and validation."""
    
    def __init__(self, action_type: ActionType, amount: float = 0.0, player_position: 'Position' = None):
        self.action_type = action_type
        self.amount = amount  # Amount in big blinds
        self.player_position = player_position
        self._validate()
    
    def _validate(self):
        """Validate action consistency."""
        if self.action_type in [ActionType.FOLD, ActionType.CHECK]:
            if self.amount != 0.0:
                raise ValueError(f"{self.action_type.value} action cannot have non-zero amount")
        
        elif self.action_type in [ActionType.CALL, ActionType.BET, ActionType.RAISE]:
            if self.amount <= 0.0:
                raise ValueError(f"{self.action_type.value} action must have positive amount")
    
    def __str__(self) -> str:
        if self.amount > 0:
            return f"{self.action_type.value}({self.amount}BB)"
        return self.action_type.value
    
    def __repr__(self) -> str:
        return f"Action({self.action_type.value}, {self.amount})"

class ActionHistory:
    """Tracks the sequence of actions in a hand."""
    
    def __init__(self):
        self.actions: List[Action] = []
    
    def add_action(self, action: Action):
        """Add an action to the history."""
        self.actions.append(action)
    
    @property
    def last_action(self) -> Optional[Action]:
        """Get the most recent action."""
        return self.actions[-1] if self.actions else None
    
    @property
    def current_bet_amount(self) -> float:
        """Get the current bet amount to call."""
        for action in reversed(self.actions):
            if action.action_type in [ActionType.BET, ActionType.RAISE]:
                return action.amount
        return 0.0
    
    @property
    def total_pot_contribution(self) -> float:
        """Calculate total pot contribution from all actions."""
        return sum(action.amount for action in self.actions 
                  if action.action_type in [ActionType.CALL, ActionType.BET, ActionType.RAISE])
    
    def get_actions_by_type(self, action_type: ActionType) -> List[Action]:
        """Get all actions of a specific type."""
        return [action for action in self.actions if action.action_type == action_type]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage/serialization."""
        return {
            'actions': [
                {
                    'type': action.action_type.value,
                    'amount': action.amount,
                    'position': action.player_position.abbreviation if action.player_position else None
                }
                for action in self.actions
            ]
        }