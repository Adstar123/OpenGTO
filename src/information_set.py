"""
Information Set encoding for CFR.

An information set represents what a player knows at a decision point:
- Their hole cards
- Their position
- The action history
- Stack sizes
- Pot size

This module handles encoding game states into information sets
and converting them to neural network input features.
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

from .game_state import GameState, Action, ActionType, Position
from .card import HoleCards


# Action type mapping for consistent encoding
ACTION_TYPE_TO_IDX = {
    ActionType.FOLD: 0,
    ActionType.CHECK: 1,
    ActionType.CALL: 2,
    ActionType.BET: 3,
    ActionType.RAISE: 4,
    ActionType.ALL_IN: 5,
}

IDX_TO_ACTION_TYPE = {v: k for k, v in ACTION_TYPE_TO_IDX.items()}

NUM_ACTIONS = len(ACTION_TYPE_TO_IDX)


@dataclass
class InformationSet:
    """
    Represents an information set - what a player knows at a decision point.

    This is the key abstraction for CFR. Players with the same information set
    should play the same strategy.
    """
    # Player's private information
    hole_cards: HoleCards
    position: Position

    # Public information
    pot_size: float  # In big blinds
    current_bet: float  # Current bet to call (in BB)
    stack_sizes: Tuple[float, ...]  # All players' stacks (in BB)
    action_history: Tuple[Tuple[int, int, float], ...]  # (position, action_type, amount)

    # Game configuration
    num_players: int

    def to_key(self) -> str:
        """
        Convert to a string key for indexing.
        Used for tabular CFR (not deep CFR).
        """
        hand_key = self.hole_cards.hand_type_string()
        pos_key = self.position.name
        history_key = '|'.join(
            f"{p}:{a}:{amt:.1f}"
            for p, a, amt in self.action_history
        )
        return f"{hand_key}_{pos_key}_{history_key}"

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to feature vector for neural network input.

        Feature layout:
        - [0:169] Hand type one-hot encoding
        - [169:175] Position one-hot encoding (6 positions)
        - [175:181] Stack sizes (normalized by 100bb)
        - [181] Pot size (normalized by 100bb)
        - [182] Current bet to call (normalized by 100bb)
        - [183] Pot odds
        - [184] Stack-to-pot ratio (capped at 10)
        - [185] Number of active players (normalized)
        - [186] Number of raises so far (normalized)
        - [187:317] Action history encoding (10 actions * 13 features each)
        """
        features = []

        # 1. Hand type one-hot (169 dimensions)
        hand_features = np.zeros(169)
        hand_idx = self.hole_cards.hand_type_index()
        hand_features[hand_idx] = 1.0
        features.extend(hand_features)

        # 2. Position one-hot (6 dimensions)
        position_features = np.zeros(6)
        position_features[self.position.value] = 1.0
        features.extend(position_features)

        # 3. Stack sizes (6 dimensions, normalized)
        for i, stack in enumerate(self.stack_sizes[:6]):
            features.append(stack / 100.0)
        # Pad if fewer than 6 players
        while len(features) < 169 + 6 + 6:
            features.append(0.0)

        # 4. Pot size (normalized)
        features.append(self.pot_size / 100.0)

        # 5. Current bet to call (normalized)
        my_stack_idx = self.position.value
        my_current_bet = 0.0  # Would need to track this
        call_amount = max(0, self.current_bet - my_current_bet)
        features.append(call_amount / 100.0)

        # 6. Pot odds
        if call_amount > 0 and self.pot_size > 0:
            pot_odds = call_amount / (self.pot_size + call_amount)
        else:
            pot_odds = 0.0
        features.append(pot_odds)

        # 7. Stack-to-pot ratio (capped at 10, normalized)
        my_stack = self.stack_sizes[my_stack_idx] if my_stack_idx < len(self.stack_sizes) else 100.0
        if self.pot_size > 0:
            spr = min(my_stack / self.pot_size, 10.0) / 10.0
        else:
            spr = 1.0
        features.append(spr)

        # 8. Number of active players (normalized)
        # Estimate from action history - players who haven't folded
        num_active = self.num_players  # Simplified
        features.append(num_active / self.num_players)

        # 9. Number of raises (normalized by 4)
        num_raises = sum(1 for _, action, _ in self.action_history
                         if action in [ACTION_TYPE_TO_IDX[ActionType.RAISE],
                                       ACTION_TYPE_TO_IDX[ActionType.BET]])
        features.append(min(num_raises / 4.0, 1.0))

        # 10. Action history encoding (10 actions * 13 features = 130 dimensions)
        action_encoding_size = 13  # 6 position + 6 action type + 1 amount
        max_actions = 10
        action_features = np.zeros(action_encoding_size * max_actions)

        for i, (pos, action_type, amount) in enumerate(self.action_history[-max_actions:]):
            base_idx = i * action_encoding_size
            # Position one-hot
            if 0 <= pos < 6:
                action_features[base_idx + pos] = 1.0
            # Action type one-hot
            if 0 <= action_type < 6:
                action_features[base_idx + 6 + action_type] = 1.0
            # Amount (normalized)
            action_features[base_idx + 12] = amount / 100.0

        features.extend(action_features)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def feature_size() -> int:
        """Return the size of the feature vector."""
        return 169 + 6 + 6 + 1 + 1 + 1 + 1 + 1 + 1 + 130  # = 317

    @classmethod
    def from_game_state(cls, state: GameState) -> 'InformationSet':
        """
        Create an InformationSet from a GameState.

        Args:
            state: Current game state

        Returns:
            InformationSet for the current player
        """
        player = state.current_player

        # Extract action history
        action_history = []
        for action in state.action_history:
            pos_val = action.player_position.value if action.player_position else 0
            action_type_idx = ACTION_TYPE_TO_IDX.get(action.action_type, 0)
            action_history.append((pos_val, action_type_idx, action.amount))

        # Extract stack sizes
        stack_sizes = tuple(p.stack for p in state.players)

        return cls(
            hole_cards=player.hole_cards,
            position=player.position,
            pot_size=state.pot,
            current_bet=state.current_bet,
            stack_sizes=stack_sizes,
            action_history=tuple(action_history),
            num_players=state.num_players
        )


def get_legal_actions_mask(state: GameState) -> np.ndarray:
    """
    Get boolean mask of legal actions for the current state.

    Returns:
        Boolean array of shape (NUM_ACTIONS,) where True = legal
    """
    legal_actions = state.get_legal_actions()
    mask = np.zeros(NUM_ACTIONS, dtype=bool)

    for action in legal_actions:
        idx = ACTION_TYPE_TO_IDX.get(action.action_type)
        if idx is not None:
            mask[idx] = True

    return mask


def action_idx_to_action(
    idx: int,
    state: GameState
) -> Action:
    """
    Convert action index to Action object with appropriate amount.

    Args:
        idx: Action index (0-5)
        state: Current game state (to determine amounts)

    Returns:
        Action object
    """
    action_type = IDX_TO_ACTION_TYPE[idx]

    if action_type == ActionType.FOLD:
        return Action(ActionType.FOLD)

    elif action_type == ActionType.CHECK:
        return Action(ActionType.CHECK)

    elif action_type == ActionType.CALL:
        return Action(ActionType.CALL, amount=state.current_bet)

    elif action_type == ActionType.BET:
        # Default bet size: 2.5x BB
        return Action(ActionType.BET, amount=2.5)

    elif action_type == ActionType.RAISE:
        # Default raise: 3x current bet
        return Action(ActionType.RAISE, amount=state.current_bet * 3)

    elif action_type == ActionType.ALL_IN:
        return Action(ActionType.ALL_IN, amount=state.current_player.stack)

    raise ValueError(f"Unknown action index: {idx}")


def sample_action(
    strategy: np.ndarray,
    legal_mask: np.ndarray
) -> int:
    """
    Sample an action from a strategy distribution.

    Args:
        strategy: Probability distribution over actions
        legal_mask: Boolean mask of legal actions

    Returns:
        Sampled action index
    """
    # Ensure strategy is properly normalized over legal actions
    masked_strategy = strategy * legal_mask
    total = masked_strategy.sum()

    if total > 0:
        probs = masked_strategy / total
    else:
        # Uniform over legal actions
        num_legal = legal_mask.sum()
        probs = legal_mask.astype(float) / num_legal

    return np.random.choice(len(strategy), p=probs)


@dataclass
class CFRTraversalState:
    """
    State maintained during CFR traversal.
    Tracks reach probabilities and other traversal information.
    """
    # Reach probabilities for each player
    reach_probs: Dict[int, float] = field(default_factory=dict)

    # Current CFR iteration
    iteration: int = 0

    # Player whose regrets we're computing (traversing player)
    traversing_player: int = 0

    def player_reach_prob(self, player_idx: int) -> float:
        """Get reach probability for a player."""
        return self.reach_probs.get(player_idx, 1.0)

    def opponent_reach_prob(self, player_idx: int) -> float:
        """Get product of opponent reach probabilities."""
        prob = 1.0
        for p, rp in self.reach_probs.items():
            if p != player_idx:
                prob *= rp
        return prob

    def update_reach_prob(
        self,
        player_idx: int,
        action_prob: float
    ) -> 'CFRTraversalState':
        """Create new state with updated reach probability."""
        new_probs = self.reach_probs.copy()
        new_probs[player_idx] = new_probs.get(player_idx, 1.0) * action_prob
        return CFRTraversalState(
            reach_probs=new_probs,
            iteration=self.iteration,
            traversing_player=self.traversing_player
        )
