from enum import Enum
from typing import List, Tuple, Set
import itertools
import random

class Suit(Enum):
    HEARTS = "h"
    DIAMONDS = "d"
    CLUBS = "c"
    SPADES = "s"

class Rank(Enum):
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
    
    @property
    def numeric_value(self):
        """Get numeric value of rank."""
        return self.value
    
    @property
    def symbol(self):
        """Get symbol representation of rank."""
        symbols = {
            2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
            10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"
        }
        return symbols[self.value]

class Card:
    """Represents a playing card with rank and suit."""
    
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit
        # Pre-compute hash for efficiency
        self._hash = hash((rank.value, suit.value))
    
    def __str__(self) -> str:
        return f"{self.rank.symbol}{self.suit.value}"
    
    def __repr__(self) -> str:
        return f"Card({self.rank.name}, {self.suit.name})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        return self._hash

class HoleCards:
    """Represents a player's hole cards with utility methods."""
    
    def __init__(self, card1: Card, card2: Card):
        self.card1 = card1
        self.card2 = card2
    
    @property
    def is_pocket_pair(self) -> bool:
        return self.card1.rank == self.card2.rank
    
    @property
    def is_suited(self) -> bool:
        return self.card1.suit == self.card2.suit
    
    @property
    def high_card_rank(self) -> Rank:
        return max(self.card1.rank, self.card2.rank, key=lambda r: r.numeric_value)
    
    @property
    def low_card_rank(self) -> Rank:
        return min(self.card1.rank, self.card2.rank, key=lambda r: r.numeric_value)
    
    def __str__(self) -> str:
        return f"{self.card1}{self.card2}"
    
    def to_string_notation(self) -> str:
        """Convert to standard poker notation (e.g., 'AKs', 'QQ', '72o')"""
        if self.is_pocket_pair:
            return f"{self.high_card_rank.symbol}{self.low_card_rank.symbol}"
        
        high = self.high_card_rank.symbol
        low = self.low_card_rank.symbol
        suited = "s" if self.is_suited else "o"
        return f"{high}{low}{suited}"

class Deck:
    """FIXED: Standard 52-card deck with efficient deal functionality."""
    
    def __init__(self):
        # Create all cards once and store as list
        self.all_cards = []
        for rank in Rank:
            for suit in Suit:
                self.all_cards.append(Card(rank, suit))
        
        # Use a simple list to track available cards (much faster)
        self.reset()
    
    def reset(self):
        """Reset deck to full 52 cards."""
        # Create indices list for efficient random selection
        self.available_indices = list(range(52))
        random.shuffle(self.available_indices)  # Pre-shuffle for efficiency
    
    def cards_remaining(self) -> int:
        """Get number of cards remaining."""
        return len(self.available_indices)
    
    def deal_card(self) -> Card:
        """Deal a random available card efficiently."""
        if not self.available_indices:
            raise ValueError("No more cards to deal")
        
        # Pop from end for O(1) operation
        card_index = self.available_indices.pop()
        return self.all_cards[card_index]
    
    def deal_hole_cards(self) -> HoleCards:
        """Deal two cards for hole cards."""
        if len(self.available_indices) < 2:
            raise ValueError("Not enough cards to deal hole cards")
        
        return HoleCards(self.deal_card(), self.deal_card())
    
    def deal_multiple_hole_cards(self, num_players: int) -> List[HoleCards]:
        """Deal hole cards to multiple players efficiently."""
        if len(self.available_indices) < num_players * 2:
            raise ValueError(f"Not enough cards for {num_players} players")
        
        hole_cards = []
        for _ in range(num_players):
            hole_cards.append(self.deal_hole_cards())
        
        return hole_cards