"""
Card and Deck classes for poker game representation.
"""
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random


class Rank(IntEnum):
    """Card ranks from 2 to Ace."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        if self.value <= 9:
            return str(self.value)
        return {10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}[self.value]

    @classmethod
    def from_char(cls, char: str) -> 'Rank':
        """Create Rank from character representation."""
        char = char.upper()
        if char.isdigit():
            return cls(int(char))
        if char == 'T':
            return cls.TEN
        mapping = {'J': cls.JACK, 'Q': cls.QUEEN, 'K': cls.KING, 'A': cls.ACE}
        if char in mapping:
            return mapping[char]
        raise ValueError(f"Invalid rank character: {char}")


class Suit(IntEnum):
    """Card suits."""
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    def __str__(self) -> str:
        return {0: 'c', 1: 'd', 2: 'h', 3: 's'}[self.value]

    def symbol(self) -> str:
        """Return unicode symbol for suit."""
        return {0: '♣', 1: '♦', 2: '♥', 3: '♠'}[self.value]

    @classmethod
    def from_char(cls, char: str) -> 'Suit':
        """Create Suit from character representation."""
        char = char.lower()
        mapping = {'c': cls.CLUBS, 'd': cls.DIAMONDS, 'h': cls.HEARTS, 's': cls.SPADES}
        if char in mapping:
            return mapping[char]
        raise ValueError(f"Invalid suit character: {char}")


@dataclass(frozen=True)
class Card:
    """
    Represents a single playing card.
    Immutable and hashable for use in sets and as dict keys.
    """
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def pretty(self) -> str:
        """Return pretty string with unicode suit symbol."""
        return f"{self.rank}{self.suit.symbol()}"

    def __lt__(self, other: 'Card') -> bool:
        """Compare cards by rank first, then suit."""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.suit < other.suit

    @classmethod
    def from_string(cls, s: str) -> 'Card':
        """
        Create a Card from string representation.
        Examples: 'As' -> Ace of spades, 'Th' -> Ten of hearts, '2c' -> Two of clubs
        """
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s}")
        rank = Rank.from_char(s[0])
        suit = Suit.from_char(s[1])
        return cls(rank, suit)


@dataclass
class HoleCards:
    """
    Represents a player's two hole cards.
    Provides methods for hand type classification (pair, suited, offsuit).
    """
    card1: Card
    card2: Card

    def __post_init__(self):
        # Always store higher rank first for consistency
        if self.card1.rank < self.card2.rank:
            # Swap cards - need to save reference first to avoid overwriting
            temp = self.card1
            object.__setattr__(self, 'card1', self.card2)
            object.__setattr__(self, 'card2', temp)
        elif self.card1.rank == self.card2.rank and self.card1.suit > self.card2.suit:
            temp = self.card1
            object.__setattr__(self, 'card1', self.card2)
            object.__setattr__(self, 'card2', temp)

    @property
    def is_pair(self) -> bool:
        """Check if hole cards are a pocket pair."""
        return self.card1.rank == self.card2.rank

    @property
    def is_suited(self) -> bool:
        """Check if hole cards are suited."""
        return self.card1.suit == self.card2.suit

    @property
    def high_rank(self) -> Rank:
        """Get the higher rank."""
        return max(self.card1.rank, self.card2.rank)

    @property
    def low_rank(self) -> Rank:
        """Get the lower rank."""
        return min(self.card1.rank, self.card2.rank)

    def hand_type_string(self) -> str:
        """
        Get string representation of hand type.
        Examples: 'AA', 'AKs', 'AKo'
        """
        high = str(self.high_rank)
        low = str(self.low_rank)

        if self.is_pair:
            return f"{high}{low}"
        elif self.is_suited:
            return f"{high}{low}s"
        else:
            return f"{high}{low}o"

    def hand_type_index(self) -> int:
        """
        Get unique index (0-168) for this hand type.
        This maps all 1326 specific hands to 169 strategic hand types.

        Matrix layout (13x13):
        - Diagonal: pairs (AA, KK, ..., 22)
        - Upper triangle: suited hands
        - Lower triangle: offsuit hands
        """
        high_idx = self.high_rank.value - 2  # 0-12 (2=0, A=12)
        low_idx = self.low_rank.value - 2

        if self.is_pair:
            # Pairs on diagonal: index = high_idx * 13 + high_idx
            return high_idx * 13 + high_idx
        elif self.is_suited:
            # Suited in upper triangle
            return high_idx * 13 + low_idx
        else:
            # Offsuit in lower triangle
            return low_idx * 13 + high_idx

    def __str__(self) -> str:
        return f"{self.card1}{self.card2}"

    def pretty(self) -> str:
        """Return pretty string with unicode suit symbols."""
        return f"{self.card1.pretty()}{self.card2.pretty()}"

    @classmethod
    def from_string(cls, s: str) -> 'HoleCards':
        """
        Create HoleCards from string representation.
        Examples: 'AsKs', 'AhKd', '2c2d'
        """
        if len(s) != 4:
            raise ValueError(f"Invalid hole cards string: {s}")
        card1 = Card.from_string(s[:2])
        card2 = Card.from_string(s[2:])
        return cls(card1, card2)


class Deck:
    """
    Standard 52-card deck with shuffle and deal operations.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset deck to full 52 cards."""
        self.cards: List[Card] = [
            Card(rank, suit)
            for suit in Suit
            for rank in Rank
        ]
        self._dealt: List[Card] = []

    def shuffle(self):
        """Shuffle remaining cards in deck."""
        random.shuffle(self.cards)

    def deal(self) -> Card:
        """Deal one card from the deck."""
        if not self.cards:
            raise ValueError("No cards left in deck")
        card = self.cards.pop()
        self._dealt.append(card)
        return card

    def deal_hole_cards(self) -> HoleCards:
        """Deal two cards as hole cards."""
        return HoleCards(self.deal(), self.deal())

    def remove(self, card: Card) -> bool:
        """Remove a specific card from the deck (for setting up scenarios)."""
        if card in self.cards:
            self.cards.remove(card)
            return True
        return False

    def remove_cards(self, cards: List[Card]) -> int:
        """Remove multiple cards from deck. Returns count of cards removed."""
        count = 0
        for card in cards:
            if self.remove(card):
                count += 1
        return count

    def remaining(self) -> int:
        """Return number of cards remaining in deck."""
        return len(self.cards)

    def __len__(self) -> int:
        return len(self.cards)


def get_all_hand_types() -> List[str]:
    """
    Get list of all 169 strategic hand types.
    Returns in order: AA, AKs, AQs, ..., AKo, KK, KQs, ..., 32o, 22
    """
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    hand_types = []

    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if i == j:
                # Pair
                hand_types.append(f"{r1}{r2}")
            elif i < j:
                # Suited (higher rank first)
                hand_types.append(f"{r1}{r2}s")
            else:
                # Offsuit (higher rank first)
                hand_types.append(f"{r2}{r1}o")

    return hand_types


def hand_type_to_index(hand_type: str) -> int:
    """
    Convert hand type string to index (0-168).
    Examples: 'AA' -> 0, 'AKs' -> 1, 'AKo' -> 13, '22' -> 168
    """
    ranks = 'AKQJT98765432'

    if len(hand_type) == 2:
        # Pair
        r = hand_type[0]
        idx = ranks.index(r)
        return idx * 13 + idx
    elif len(hand_type) == 3:
        r1, r2, suited = hand_type[0], hand_type[1], hand_type[2]
        idx1 = ranks.index(r1)
        idx2 = ranks.index(r2)

        if suited.lower() == 's':
            # Suited: upper triangle
            return min(idx1, idx2) * 13 + max(idx1, idx2)
        else:
            # Offsuit: lower triangle
            return max(idx1, idx2) * 13 + min(idx1, idx2)
    else:
        raise ValueError(f"Invalid hand type: {hand_type}")


def index_to_hand_type(index: int) -> str:
    """
    Convert index (0-168) to hand type string.
    """
    ranks = 'AKQJT98765432'
    row = index // 13
    col = index % 13

    if row == col:
        # Pair
        return f"{ranks[row]}{ranks[col]}"
    elif row < col:
        # Suited (upper triangle)
        return f"{ranks[row]}{ranks[col]}s"
    else:
        # Offsuit (lower triangle)
        return f"{ranks[col]}{ranks[row]}o"


def get_hand_combos(hand_type: str) -> int:
    """
    Get number of specific card combinations for a hand type.
    Pairs: 6 combos
    Suited: 4 combos
    Offsuit: 12 combos
    """
    if len(hand_type) == 2:
        return 6  # Pairs
    elif hand_type[-1].lower() == 's':
        return 4  # Suited
    else:
        return 12  # Offsuit
