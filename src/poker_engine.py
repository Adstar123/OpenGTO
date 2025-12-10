"""
Poker Engine for preflop game simulation.
Handles all game mechanics, action validation, and state transitions.
"""
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import random

from .card import Card, Deck, HoleCards
from .game_state import (
    GameState, PlayerState, Action, ActionType, Position
)


class PreflopPokerEngine:
    """
    Engine for simulating preflop poker scenarios.
    Handles action flow, validation, and state transitions.
    """

    def __init__(
        self,
        num_players: int = 6,
        starting_stack: float = 100.0,
        small_blind: float = 0.5,
        big_blind: float = 1.0
    ):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = Deck()

    def create_game(
        self,
        stacks: Optional[List[float]] = None,
        hole_cards: Optional[Dict[Position, HoleCards]] = None
    ) -> GameState:
        """
        Create a new game state with optional custom stacks and hole cards.

        Args:
            stacks: List of stack sizes for each position (in BB)
            hole_cards: Dict mapping positions to specific hole cards

        Returns:
            Initialized GameState ready for play
        """
        # Create game state
        state = GameState(
            num_players=self.num_players,
            small_blind=self.small_blind,
            big_blind=self.big_blind
        )

        # Set up players
        positions = Position.get_positions(self.num_players)
        state.players = []

        for i, pos in enumerate(positions):
            stack = stacks[i] if stacks and i < len(stacks) else self.starting_stack
            player = PlayerState(
                position=pos,
                stack=stack,
                is_active=True,
                has_acted=False,
                current_bet=0.0
            )
            state.players.append(player)

        # Post blinds
        self._post_blinds(state)

        # Deal cards
        self._deal_cards(state, hole_cards)

        # Set first player to act (UTG in 6-max, BTN in HU)
        state.current_player_idx = self._get_first_to_act_idx(state)

        return state

    def _post_blinds(self, state: GameState):
        """Post small and big blinds."""
        for player in state.players:
            if player.position == Position.SB or (
                self.num_players == 2 and player.position == Position.BTN
            ):
                # Small blind
                blind_amount = min(self.small_blind, player.stack)
                player.current_bet = blind_amount
                player.stack -= blind_amount
                state.pot += blind_amount
            elif player.position == Position.BB:
                # Big blind
                blind_amount = min(self.big_blind, player.stack)
                player.current_bet = blind_amount
                player.stack -= blind_amount
                state.pot += blind_amount
                state.current_bet = blind_amount
                state.min_raise = blind_amount  # Min raise is 1 BB

    def _deal_cards(
        self,
        state: GameState,
        preset_cards: Optional[Dict[Position, HoleCards]] = None
    ):
        """Deal hole cards to all players."""
        self.deck.reset()
        self.deck.shuffle()

        # Remove preset cards from deck
        if preset_cards:
            for hole_cards in preset_cards.values():
                self.deck.remove(hole_cards.card1)
                self.deck.remove(hole_cards.card2)

        # Deal cards
        for player in state.players:
            if preset_cards and player.position in preset_cards:
                player.hole_cards = preset_cards[player.position]
            else:
                player.hole_cards = self.deck.deal_hole_cards()

    def _get_first_to_act_idx(self, state: GameState) -> int:
        """Get index of first player to act preflop."""
        preflop_order = Position.preflop_order(self.num_players)

        for pos in preflop_order:
            for i, player in enumerate(state.players):
                if player.position == pos and player.is_active and not player.is_all_in:
                    return i

        return 0

    def _get_next_player_idx(self, state: GameState) -> Optional[int]:
        """Get index of next player to act."""
        preflop_order = Position.preflop_order(self.num_players)
        current_pos = state.current_player.position

        # Find current position in order
        try:
            current_order_idx = preflop_order.index(current_pos)
        except ValueError:
            current_order_idx = 0

        # Look for next active player
        for i in range(1, len(preflop_order) + 1):
            next_order_idx = (current_order_idx + i) % len(preflop_order)
            next_pos = preflop_order[next_order_idx]

            for j, player in enumerate(state.players):
                if (player.position == next_pos and
                    player.is_active and
                    not player.is_all_in and
                    (not player.has_acted or player.current_bet < state.current_bet)):
                    return j

        return None

    def apply_action(self, state: GameState, action: Action) -> GameState:
        """
        Apply an action to the game state.
        Returns a new game state (does not modify original).

        Args:
            state: Current game state
            action: Action to apply

        Returns:
            New game state after action
        """
        new_state = state.copy()
        player = new_state.current_player

        # Record action with player position
        action.player_position = player.position
        new_state.action_history.append(action)
        new_state.num_actions_this_round += 1

        if action.action_type == ActionType.FOLD:
            self._handle_fold(new_state, player)

        elif action.action_type == ActionType.CHECK:
            self._handle_check(new_state, player)

        elif action.action_type == ActionType.CALL:
            self._handle_call(new_state, player)

        elif action.action_type == ActionType.BET:
            self._handle_bet(new_state, player, action.amount)

        elif action.action_type == ActionType.RAISE:
            self._handle_raise(new_state, player, action.amount)

        elif action.action_type == ActionType.ALL_IN:
            self._handle_all_in(new_state, player)

        player.has_acted = True

        # Check if hand is complete
        if self._is_hand_complete(new_state):
            new_state.is_complete = True
            self._determine_winners(new_state)
        else:
            # Move to next player
            next_idx = self._get_next_player_idx(new_state)
            if next_idx is not None:
                new_state.current_player_idx = next_idx
            else:
                new_state.is_complete = True
                self._determine_winners(new_state)

        return new_state

    def _handle_fold(self, state: GameState, player: PlayerState):
        """Handle fold action."""
        player.is_active = False

    def _handle_check(self, state: GameState, player: PlayerState):
        """Handle check action."""
        pass  # Nothing changes on check

    def _handle_call(self, state: GameState, player: PlayerState):
        """Handle call action."""
        call_amount = state.current_bet - player.current_bet
        actual_call = min(call_amount, player.stack)

        player.stack -= actual_call
        state.pot += actual_call
        player.current_bet += actual_call

        if player.stack == 0:
            player.is_all_in = True

    def _handle_bet(self, state: GameState, player: PlayerState, amount: float):
        """Handle bet action."""
        bet_amount = min(amount, player.stack)

        player.stack -= bet_amount
        state.pot += bet_amount
        player.current_bet = bet_amount
        state.current_bet = bet_amount
        state.min_raise = bet_amount  # Min raise = bet size

        if player.stack == 0:
            player.is_all_in = True

        # Reset has_acted for other players (they need to respond to the bet)
        for p in state.players:
            if p != player and p.is_active:
                p.has_acted = False

    def _handle_raise(self, state: GameState, player: PlayerState, raise_to: float):
        """Handle raise action."""
        raise_amount = raise_to - player.current_bet
        actual_raise = min(raise_amount, player.stack)

        player.stack -= actual_raise
        state.pot += actual_raise
        old_bet = state.current_bet
        player.current_bet += actual_raise
        state.current_bet = player.current_bet

        # Update min raise (raise size must be at least the previous raise)
        state.min_raise = state.current_bet - old_bet

        if player.stack == 0:
            player.is_all_in = True

        # Reset has_acted for other players
        for p in state.players:
            if p != player and p.is_active:
                p.has_acted = False

    def _handle_all_in(self, state: GameState, player: PlayerState):
        """Handle all-in action."""
        all_in_amount = player.stack
        player.current_bet += all_in_amount
        state.pot += all_in_amount
        player.stack = 0
        player.is_all_in = True

        if player.current_bet > state.current_bet:
            # This is a raise
            old_bet = state.current_bet
            state.current_bet = player.current_bet
            state.min_raise = max(state.min_raise, state.current_bet - old_bet)

            # Reset has_acted for other players
            for p in state.players:
                if p != player and p.is_active:
                    p.has_acted = False

    def _is_hand_complete(self, state: GameState) -> bool:
        """Check if the hand is complete."""
        active_players = [p for p in state.players if p.is_active]

        # Only one player left
        if len(active_players) == 1:
            return True

        # All active players have acted and bets are matched
        all_acted = all(p.has_acted for p in active_players if not p.is_all_in)
        bets_matched = all(
            p.current_bet == state.current_bet or p.is_all_in
            for p in active_players
        )

        if all_acted and bets_matched:
            return True

        # All players all-in
        if all(p.is_all_in for p in active_players):
            return True

        return False

    def _determine_winners(self, state: GameState):
        """
        Determine winners of the hand.
        For preflop all-in situations, we'd need to run out the board.
        For now, just track who can win.
        """
        active_players = [p for p in state.players if p.is_active]

        if len(active_players) == 1:
            # Everyone else folded
            winner = active_players[0]
            state.winners = [state.players.index(winner)]
        else:
            # Multiple players - would need showdown
            # For preflop trainer, we'll handle equity calculation separately
            state.winners = [state.players.index(p) for p in active_players]

    def is_valid_action(self, state: GameState, action: Action) -> bool:
        """Check if an action is valid in the current state."""
        legal_actions = state.get_legal_actions()

        for legal in legal_actions:
            if legal.action_type == action.action_type:
                if action.action_type in [ActionType.FOLD, ActionType.CHECK]:
                    return True
                elif action.action_type == ActionType.CALL:
                    return True
                elif action.action_type in [ActionType.BET, ActionType.RAISE, ActionType.ALL_IN]:
                    # Check if amount is reasonable
                    return True

        return False

    def get_available_action_types(self, state: GameState) -> List[ActionType]:
        """Get simplified list of available action types."""
        legal_actions = state.get_legal_actions()
        action_types = set()

        for action in legal_actions:
            action_types.add(action.action_type)

        return list(action_types)


class GameSimulator:
    """
    Simulates complete poker games for training data generation.
    """

    def __init__(self, engine: PreflopPokerEngine):
        self.engine = engine

    def simulate_random_game(self) -> Tuple[List[GameState], List[Action]]:
        """
        Simulate a game with random actions.
        Returns sequence of states and actions taken.
        """
        state = self.engine.create_game()
        states = [state]
        actions = []

        while not state.is_complete:
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                break

            # Random action selection
            action = random.choice(legal_actions)
            actions.append(action)

            state = self.engine.apply_action(state, action)
            states.append(state)

        return states, actions

    def simulate_with_policy(
        self,
        policy_fn: Callable[[GameState], Action]
    ) -> Tuple[List[GameState], List[Action]]:
        """
        Simulate a game using a policy function.

        Args:
            policy_fn: Function that takes GameState and returns Action

        Returns:
            Tuple of (states, actions) for the game
        """
        state = self.engine.create_game()
        states = [state]
        actions = []

        while not state.is_complete:
            action = policy_fn(state)
            actions.append(action)

            state = self.engine.apply_action(state, action)
            states.append(state)

        return states, actions


class ScenarioBuilder:
    """
    Helper class for building specific poker scenarios.
    Useful for the trainer interface.
    """

    def __init__(self, engine: PreflopPokerEngine):
        self.engine = engine
        self.stacks: Optional[List[float]] = None
        self.hole_cards: Dict[Position, HoleCards] = {}
        self.actions: List[Tuple[Position, Action]] = []

    def set_stacks(self, stacks: List[float]) -> 'ScenarioBuilder':
        """Set stack sizes for all players."""
        self.stacks = stacks
        return self

    def set_hero_cards(self, position: Position, cards: HoleCards) -> 'ScenarioBuilder':
        """Set specific hole cards for hero."""
        self.hole_cards[position] = cards
        return self

    def add_action(self, position: Position, action: Action) -> 'ScenarioBuilder':
        """Add a predefined action to the scenario."""
        self.actions.append((position, action))
        return self

    def build(self) -> GameState:
        """Build the scenario and return the game state."""
        state = self.engine.create_game(
            stacks=self.stacks,
            hole_cards=self.hole_cards
        )

        # Apply predefined actions
        for position, action in self.actions:
            # Find player at this position
            for i, player in enumerate(state.players):
                if player.position == position:
                    state.current_player_idx = i
                    break

            state = self.engine.apply_action(state, action)

        return state

    def reset(self) -> 'ScenarioBuilder':
        """Reset the builder for a new scenario."""
        self.stacks = None
        self.hole_cards = {}
        self.actions = []
        return self
