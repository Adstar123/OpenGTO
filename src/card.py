"""
Card representation and hand utilities for poker.
"""
from enum import IntEnum
from typing import Tuple, List
from dataclasses import dataclass


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


class Suit(IntEnum):
    """Card suits."""
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


@dataclass(frozen=True)
class Card:
    """Immutable card representation."""
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        rank_str = {
            Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5',
            Rank.SIX: '6', Rank.SEVEN: '7', Rank.EIGHT: '8', Rank.NINE: '9',
            Rank.TEN: 'T', Rank.JACK: 'J', Rank.QUEEN: 'Q', Rank.KING: 'K',
            Rank.ACE: 'A'
        }
        suit_str = {
            Suit.CLUBS: 'c', Suit.DIAMONDS: 'd',
            Suit.HEARTS: 'h', Suit.SPADES: 's'
        }
        return f"{rank_str[self.rank]}{suit_str[self.suit]}"

    def __lt__(self, other: 'Card') -> bool:
        return self.rank < other.rank


@dataclass(frozen=True)
class Hand:
    """Represents a poker hand (two hole cards)."""
    card1: Card
    card2: Card

    def __post_init__(self):
        """Ensure cards are sorted by rank (higher first)."""
        if self.card1.rank < self.card2.rank:
            object.__setattr__(self, 'card1', self.card2)
            object.__setattr__(self, 'card2', self.card1)

    def is_suited(self) -> bool:
        """Check if the hand is suited."""
        return self.card1.suit == self.card2.suit

    def is_pair(self) -> bool:
        """Check if the hand is a pocket pair."""
        return self.card1.rank == self.card2.rank

    def to_string(self) -> str:
        """
        Convert hand to standard poker notation.
        Examples: 'AKs', 'AKo', 'AA', 'T9s'
        """
        rank_str = {
            Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5',
            Rank.SIX: '6', Rank.SEVEN: '7', Rank.EIGHT: '8', Rank.NINE: '9',
            Rank.TEN: 'T', Rank.JACK: 'J', Rank.QUEEN: 'Q', Rank.KING: 'K',
            Rank.ACE: 'A'
        }

        r1 = rank_str[self.card1.rank]
        r2 = rank_str[self.card2.rank]

        if self.is_pair():
            return f"{r1}{r2}"
        elif self.is_suited():
            return f"{r1}{r2}s"
        else:
            return f"{r1}{r2}o"

    def __str__(self) -> str:
        return f"{self.card1}{self.card2}"


def create_deck() -> List[Card]:
    """Create a standard 52-card deck."""
    deck = []
    for suit in Suit:
        for rank in Rank:
            deck.append(Card(rank, suit))
    return deck


def parse_hand(hand_str: str) -> Hand:
    """
    Parse a hand string like 'AKs', 'AKo', or 'AA' into a Hand object.
    Returns the first matching hand (doesn't specify exact suits for non-pairs).
    """
    rank_map = {
        '2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR, '5': Rank.FIVE,
        '6': Rank.SIX, '7': Rank.SEVEN, '8': Rank.EIGHT, '9': Rank.NINE,
        'T': Rank.TEN, 'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING,
        'A': Rank.ACE
    }

    if len(hand_str) < 2:
        raise ValueError(f"Invalid hand string: {hand_str}")

    rank1 = rank_map[hand_str[0]]
    rank2 = rank_map[hand_str[1]]

    if len(hand_str) == 2:
        # Pocket pair like 'AA'
        if rank1 != rank2:
            raise ValueError(f"Two-character hand must be a pair: {hand_str}")
        card1 = Card(rank1, Suit.SPADES)
        card2 = Card(rank2, Suit.HEARTS)
    elif len(hand_str) == 3:
        # Suited or offsuit
        suited = hand_str[2] == 's'
        if suited:
            card1 = Card(rank1, Suit.SPADES)
            card2 = Card(rank2, Suit.SPADES)
        else:
            card1 = Card(rank1, Suit.SPADES)
            card2 = Card(rank2, Suit.HEARTS)
    else:
        raise ValueError(f"Invalid hand string: {hand_str}")

    return Hand(card1, card2)
