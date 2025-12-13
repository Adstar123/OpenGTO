from enum import Enum
from typing import List, Dict

class Position(Enum):
    """Poker positions for different player counts."""
    # Universal positions
    BIG_BLIND = ("BB", "Big Blind")
    SMALL_BLIND = ("SB", "Small Blind")
    BUTTON = ("BTN", "Button")
    
    # Additional positions for more players
    CUTOFF = ("CO", "Cutoff")
    MIDDLE_POSITION = ("MP", "Middle Position")
    UNDER_THE_GUN_PLUS_ONE = ("UTG+1", "Under the Gun +1")
    UNDER_THE_GUN = ("UTG", "Under the Gun")
    
    def __init__(self, abbreviation: str, full_name: str):
        self.abbreviation = abbreviation
        self.full_name = full_name

class PositionManager:
    """Manages position assignments based on player count."""
    
    # Position orders for different player counts (from first to act preflop)
    POSITION_ORDERS = {
        2: [Position.SMALL_BLIND, Position.BIG_BLIND],
        3: [Position.SMALL_BLIND, Position.BIG_BLIND, Position.BUTTON],
        4: [Position.SMALL_BLIND, Position.BIG_BLIND, Position.CUTOFF, Position.BUTTON],
        5: [Position.SMALL_BLIND, Position.BIG_BLIND, Position.MIDDLE_POSITION, Position.CUTOFF, Position.BUTTON],
        6: [Position.SMALL_BLIND, Position.BIG_BLIND, Position.UNDER_THE_GUN, Position.MIDDLE_POSITION, Position.CUTOFF, Position.BUTTON]
    }
    
    @classmethod
    def get_positions_for_player_count(cls, player_count: int) -> List[Position]:
        """Get ordered list of positions for given player count."""
        if player_count not in cls.POSITION_ORDERS:
            raise ValueError(f"Unsupported player count: {player_count}")
        return cls.POSITION_ORDERS[player_count].copy()
    
    @classmethod
    def get_position_index(cls, position: Position, player_count: int) -> int:
        """Get the index of a position in the action order."""
        positions = cls.get_positions_for_player_count(player_count)
        return positions.index(position)
    
    @classmethod
    def get_relative_position(cls, position: Position, player_count: int) -> float:
        """Get relative position (0.0 = earliest to act, 1.0 = latest)."""
        index = cls.get_position_index(position, player_count)
        return index / (player_count - 1) if player_count > 1 else 0.0