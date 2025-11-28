"""
Game tree representation for preflop poker.
Defines the structure of betting sequences and game states.
"""
from enum import Enum
from typing import List, Optional, Dict
from dataclasses import dataclass
from src.card import Hand


class Action(Enum):
    """Possible actions in preflop poker."""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE_2BB = "raise_2bb"
    RAISE_3BB = "raise_3bb"
    RAISE_4BB = "raise_4bb"
    ALL_IN = "all_in"


class Position(Enum):
    """Player positions."""
    BTN = "btn"  # Button (dealer)
    BB = "bb"    # Big blind


@dataclass
class GameState:
    """Represents a state in the game tree."""
    pot: float
    player_stacks: Dict[Position, float]
    current_bet: float
    position_to_act: Position
    action_history: List[Action]
    is_terminal: bool = False
    payoff: Optional[float] = None

    def __hash__(self):
        """Make GameState hashable for use as dictionary key."""
        return hash((
            self.pot,
            tuple(sorted(self.player_stacks.items())),
            self.current_bet,
            self.position_to_act,
            tuple(self.action_history)
        ))

    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        return GameState(
            pot=self.pot,
            player_stacks=self.player_stacks.copy(),
            current_bet=self.current_bet,
            position_to_act=self.position_to_act,
            action_history=self.action_history.copy(),
            is_terminal=self.is_terminal,
            payoff=self.payoff
        )


class PreflopGameTree:
    """
    Manages the preflop game tree for heads-up poker.
    Handles action generation and state transitions.
    """

    def __init__(self, stack_size: float = 100.0, big_blind: float = 1.0):
        """
        Initialize the game tree.

        Args:
            stack_size: Starting stack size in big blinds
            big_blind: Size of the big blind
        """
        self.stack_size = stack_size
        self.bb = big_blind
        self.sb = big_blind / 2

    def get_initial_state(self) -> GameState:
        """Get the initial game state (BTN has SB, BB has BB posted)."""
        return GameState(
            pot=self.sb + self.bb,
            player_stacks={
                Position.BTN: self.stack_size - self.sb,
                Position.BB: self.stack_size - self.bb
            },
            current_bet=self.bb,
            position_to_act=Position.BTN,
            action_history=[],
            is_terminal=False
        )

    def get_legal_actions(self, state: GameState) -> List[Action]:
        """Get list of legal actions for the current state."""
        if state.is_terminal:
            return []

        actions = []
        player = state.position_to_act
        player_stack = state.player_stacks[player]
        amount_to_call = state.current_bet - self._get_player_bet(state, player)

        # Can always fold if there's a bet to call
        if amount_to_call > 0:
            actions.append(Action.FOLD)

        # Check if no bet, call if there is a bet
        if amount_to_call == 0:
            actions.append(Action.CHECK)
        elif amount_to_call <= player_stack:
            actions.append(Action.CALL)

        # Raising options
        if player_stack > amount_to_call:
            remaining = player_stack - amount_to_call

            # Raise to 2bb, 3bb, 4bb if possible
            for raise_action, raise_size in [
                (Action.RAISE_2BB, 2 * self.bb),
                (Action.RAISE_3BB, 3 * self.bb),
                (Action.RAISE_4BB, 4 * self.bb)
            ]:
                # Can only raise if this raise size is less than all-in
                # and greater than current bet
                if raise_size > state.current_bet and (raise_size - state.current_bet) <= remaining:
                    actions.append(raise_action)

            # All-in is always an option if we have chips
            if player_stack > 0:
                actions.append(Action.ALL_IN)

        return actions

    def apply_action(self, state: GameState, action: Action, hand1: Hand, hand2: Hand,
                     equity_calc) -> GameState:
        """
        Apply an action to a state and return the new state.

        Args:
            state: Current game state
            action: Action to apply
            hand1: BTN's hand
            hand2: BB's hand
            equity_calc: Equity calculator for terminal node evaluation
        """
        new_state = state.copy()
        player = state.position_to_act
        opponent = Position.BB if player == Position.BTN else Position.BTN

        new_state.action_history.append(action)

        if action == Action.FOLD:
            # Opponent wins the pot
            new_state.is_terminal = True
            new_state.payoff = -state.pot / 2 if player == Position.BTN else state.pot / 2

        elif action == Action.CHECK:
            # Check
            if len(state.action_history) > 0 and state.action_history[-1] == Action.CHECK:
                # Both players checked, go to showdown
                new_state.is_terminal = True
                new_state.payoff = self._evaluate_showdown(hand1, hand2, state.pot, equity_calc)
            else:
                # Pass action to opponent
                new_state.position_to_act = opponent

        elif action == Action.CALL:
            # Call the current bet
            amount_to_call = state.current_bet - self._get_player_bet(state, player)
            new_state.player_stacks[player] -= amount_to_call
            new_state.pot += amount_to_call

            # After call, go to showdown
            new_state.is_terminal = True
            new_state.payoff = self._evaluate_showdown(hand1, hand2, new_state.pot, equity_calc)

        elif action in [Action.RAISE_2BB, Action.RAISE_3BB, Action.RAISE_4BB]:
            # Raise to specific amount
            raise_sizes = {
                Action.RAISE_2BB: 2 * self.bb,
                Action.RAISE_3BB: 3 * self.bb,
                Action.RAISE_4BB: 4 * self.bb
            }
            new_bet = raise_sizes[action]
            amount_to_raise = new_bet - self._get_player_bet(state, player)

            new_state.player_stacks[player] -= amount_to_raise
            new_state.pot += amount_to_raise
            new_state.current_bet = new_bet
            new_state.position_to_act = opponent

        elif action == Action.ALL_IN:
            # All-in
            amount = new_state.player_stacks[player]
            new_state.player_stacks[player] = 0
            new_state.pot += amount
            new_state.current_bet = state.current_bet + amount
            new_state.position_to_act = opponent

        return new_state

    def _get_player_bet(self, state: GameState, player: Position) -> float:
        """Calculate how much a player has bet in the current round."""
        if player == Position.BTN:
            return self.sb + (self.stack_size - self.sb - state.player_stacks[player])
        else:
            return self.bb + (self.stack_size - self.bb - state.player_stacks[player])

    def _evaluate_showdown(self, hand1: Hand, hand2: Hand, pot: float,
                          equity_calc) -> float:
        """
        Evaluate showdown and return payoff for BTN.

        Args:
            hand1: BTN's hand
            hand2: BB's hand
            pot: Total pot size
            equity_calc: Equity calculator

        Returns:
            Expected payoff for BTN (positive = BTN wins, negative = BB wins)
        """
        equity = equity_calc.get_equity(hand1, hand2)

        # BTN's expected value
        # equity = probability BTN wins
        # (1 - equity) = probability BB wins
        btn_ev = equity * pot - (1 - equity) * pot
        return btn_ev - pot / 2  # Subtract initial investment
