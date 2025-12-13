"""
Hand utilities for poker hand representation and manipulation.
Includes the 169 hand type matrix and range utilities.

NOTE: This file intentionally does NOT contain any hand strength rankings,
tier systems, or equity approximations. The neural network must learn
all hand values purely through self-play. The only "knowledge" we encode
here is structural (how to enumerate hands, display matrices, etc.)
"""
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np

from .card import (
    Card, Rank, Suit, Deck, HoleCards,
    get_all_hand_types, hand_type_to_index, index_to_hand_type, get_hand_combos
)


@dataclass
class HandRange:
    """
    Represents a range of hands with frequencies.
    Used for defining ranges, visualizing learned strategies, etc.

    This is a data structure only - it does NOT encode any poker knowledge
    about which hands are "good" or "bad".
    """
    # Dict mapping hand type to frequency (0.0 to 1.0)
    hands: Dict[str, float]

    def __init__(self, hands: Optional[Dict[str, float]] = None):
        self.hands = hands if hands else {}

    @classmethod
    def empty(cls) -> 'HandRange':
        """Create empty range."""
        return cls({})

    @classmethod
    def full(cls) -> 'HandRange':
        """Create range with all hands at 100%."""
        return cls({hand: 1.0 for hand in get_all_hand_types()})

    @classmethod
    def from_string(cls, range_str: str) -> 'HandRange':
        """
        Parse range string notation.
        Examples:
        - "AA,KK,QQ" -> specific hands
        - "AKs,AQs" -> suited hands
        """
        hands = {}
        parts = range_str.replace(' ', '').split(',')

        for part in parts:
            if not part:
                continue
            # Single hand type
            hands[part] = 1.0

        return cls(hands)

    def add_hand(self, hand_type: str, frequency: float = 1.0):
        """Add a hand to the range."""
        self.hands[hand_type] = frequency

    def remove_hand(self, hand_type: str):
        """Remove a hand from the range."""
        if hand_type in self.hands:
            del self.hands[hand_type]

    def contains(self, hand: HoleCards) -> Tuple[bool, float]:
        """Check if range contains a hand and return frequency."""
        hand_type = hand.hand_type_string()
        if hand_type in self.hands:
            return True, self.hands[hand_type]
        return False, 0.0

    def to_matrix(self) -> np.ndarray:
        """Convert range to 13x13 matrix for visualization."""
        matrix = np.zeros((13, 13))
        ranks = 'AKQJT98765432'

        for hand_type, freq in self.hands.items():
            if len(hand_type) == 2:
                # Pair
                idx = ranks.index(hand_type[0])
                matrix[idx, idx] = freq
            elif hand_type.endswith('s'):
                # Suited
                r1_idx = ranks.index(hand_type[0])
                r2_idx = ranks.index(hand_type[1])
                matrix[min(r1_idx, r2_idx), max(r1_idx, r2_idx)] = freq
            elif hand_type.endswith('o'):
                # Offsuit
                r1_idx = ranks.index(hand_type[0])
                r2_idx = ranks.index(hand_type[1])
                matrix[max(r1_idx, r2_idx), min(r1_idx, r2_idx)] = freq

        return matrix

    def num_combos(self) -> float:
        """Calculate total number of combos in range (weighted by frequency)."""
        total = 0.0
        for hand_type, freq in self.hands.items():
            total += get_hand_combos(hand_type) * freq
        return total

    def percentage(self) -> float:
        """Calculate percentage of all hands in range."""
        return self.num_combos() / 1326.0 * 100.0

    def __str__(self) -> str:
        hands = [f"{h}:{f:.0%}" if f < 1.0 else h
                 for h, f in sorted(self.hands.items())]
        return ', '.join(hands[:10]) + ('...' if len(hands) > 10 else '')


def print_hand_matrix(
    matrix: np.ndarray,
    show_values: bool = True
) -> str:
    """
    Create string representation of a 13x13 hand matrix.
    Diagonal = pairs, upper = suited, lower = offsuit.

    This is purely for visualization - no poker knowledge encoded.
    """
    ranks = 'AKQJT98765432'
    lines = []

    # Header
    header = '    ' + '  '.join(ranks)
    lines.append(header)
    lines.append('  ' + '-' * 40)

    for i, r1 in enumerate(ranks):
        row = f"{r1} |"
        for j, r2 in enumerate(ranks):
            val = matrix[i, j]
            if show_values and val > 0:
                if val >= 0.995:
                    row += ' ██'
                elif val >= 0.5:
                    row += ' ▓▓'
                elif val > 0:
                    row += ' ░░'
                else:
                    row += ' · '
            else:
                row += ' · '
        lines.append(row)

    return '\n'.join(lines)


def enumerate_all_hole_cards() -> List[HoleCards]:
    """Generate all 1326 possible hole card combinations."""
    deck = Deck()
    cards = deck.cards.copy()
    hole_cards = []

    for i, card1 in enumerate(cards):
        for card2 in cards[i + 1:]:
            hole_cards.append(HoleCards(card1, card2))

    return hole_cards


def get_combos_for_hand_type(hand_type: str) -> List[HoleCards]:
    """Get all specific card combinations for a hand type."""
    combos = []

    if len(hand_type) == 2:
        # Pair
        rank = Rank.from_char(hand_type[0])
        suits = list(Suit)
        for i, s1 in enumerate(suits):
            for s2 in suits[i + 1:]:
                combos.append(HoleCards(Card(rank, s1), Card(rank, s2)))

    elif hand_type.endswith('s'):
        # Suited
        r1 = Rank.from_char(hand_type[0])
        r2 = Rank.from_char(hand_type[1])
        for suit in Suit:
            combos.append(HoleCards(Card(r1, suit), Card(r2, suit)))

    elif hand_type.endswith('o'):
        # Offsuit
        r1 = Rank.from_char(hand_type[0])
        r2 = Rank.from_char(hand_type[1])
        for s1 in Suit:
            for s2 in Suit:
                if s1 != s2:
                    combos.append(HoleCards(Card(r1, s1), Card(r2, s2)))

    return combos


def hand_type_from_index(index: int) -> str:
    """Convert index (0-168) back to hand type string."""
    return index_to_hand_type(index)
