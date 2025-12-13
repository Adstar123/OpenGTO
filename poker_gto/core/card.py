"""Optimized card and deck implementation with better time complexity."""

from enum import Enum
from typing import List, Tuple, Set, Optional
import random


class Suit(Enum):
    """Card suits."""
    HEARTS = "h"
    DIAMONDS = "d"
    CLUBS = "c"
    SPADES = "s"


class Rank(Enum):
    """Card ranks with numeric values."""
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
    def numeric_value(self) -> int:
        """Get numeric value of rank."""
        return self.value
    
    @property
    def symbol(self) -> str:
        """Get symbol representation of rank."""
        _symbols = {
            2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
            10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"
        }
        return _symbols[self.value]
    
    @classmethod
    def from_symbol(cls, symbol: str) -> 'Rank':
        """Create Rank from symbol.
        
        Args:
            symbol: Single character rank symbol
            
        Returns:
            Rank enum value
            
        Raises:
            ValueError: If symbol is invalid
        """
        symbol_map = {
            '2': cls.TWO, '3': cls.THREE, '4': cls.FOUR, '5': cls.FIVE,
            '6': cls.SIX, '7': cls.SEVEN, '8': cls.EIGHT, '9': cls.NINE,
            'T': cls.TEN, 'J': cls.JACK, 'Q': cls.QUEEN, 'K': cls.KING, 'A': cls.ACE
        }
        
        if symbol not in symbol_map:
            raise ValueError(f"Invalid rank symbol: {symbol}")
        
        return symbol_map[symbol]


class Card:
    """Represents a playing card with rank and suit.
    
    Optimized with __slots__ for memory efficiency.
    """
    
    __slots__ = ('rank', 'suit', '_hash', '_str')
    
    def __init__(self, rank: Rank, suit: Suit):
        """Initialize card.
        
        Args:
            rank: Card rank
            suit: Card suit
        """
        self.rank = rank
        self.suit = suit
        # Pre-compute for efficiency
        self._hash = hash((rank.value, suit.value))
        self._str = f"{rank.symbol}{suit.value}"
    
    def __str__(self) -> str:
        """String representation (e.g., 'As', 'Kh')."""
        return self._str
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"Card({self.rank.name}, {self.suit.name})"
    
    def __eq__(self, other) -> bool:
        """Check equality with another card."""
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        """Hash for use in sets and dicts."""
        return self._hash
    
    @property
    def index(self) -> int:
        """Get unique index 0-51 for this card."""
        return (self.rank.value - 2) * 4 + list(Suit).index(self.suit)
    
    @classmethod
    def from_string(cls, card_str: str) -> 'Card':
        """Create card from string notation.
        
        Args:
            card_str: String like 'As', 'Kh', etc.
            
        Returns:
            Card instance
            
        Raises:
            ValueError: If string format is invalid
        """
        if len(card_str) != 2:
            raise ValueError(f"Invalid card string: {card_str}")
        
        rank_symbol = card_str[0].upper()
        suit_symbol = card_str[1].lower()
        
        rank = Rank.from_symbol(rank_symbol)
        
        suit_map = {'h': Suit.HEARTS, 'd': Suit.DIAMONDS, 'c': Suit.CLUBS, 's': Suit.SPADES}
        if suit_symbol not in suit_map:
            raise ValueError(f"Invalid suit symbol: {suit_symbol}")
        
        suit = suit_map[suit_symbol]
        
        return cls(rank, suit)


class HoleCards:
    """Represents a player's hole cards with utility methods."""
    
    __slots__ = ('card1', 'card2', '_is_pocket_pair', '_is_suited', '_high_card', '_low_card')
    
    def __init__(self, card1: Card, card2: Card):
        """Initialize hole cards.
        
        Args:
            card1: First card
            card2: Second card
        """
        self.card1 = card1
        self.card2 = card2
        
        # Pre-compute properties
        self._is_pocket_pair = card1.rank == card2.rank
        self._is_suited = card1.suit == card2.suit
        
        if card1.rank.numeric_value >= card2.rank.numeric_value:
            self._high_card = card1
            self._low_card = card2
        else:
            self._high_card = card2
            self._low_card = card1
    
    @property
    def is_pocket_pair(self) -> bool:
        """Check if cards form a pocket pair."""
        return self._is_pocket_pair
    
    @property
    def is_suited(self) -> bool:
        """Check if cards are suited."""
        return self._is_suited
    
    @property
    def high_card_rank(self) -> Rank:
        """Get rank of higher card."""
        return self._high_card.rank
    
    @property
    def low_card_rank(self) -> Rank:
        """Get rank of lower card."""
        return self._low_card.rank
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.card1}{self.card2}"
    
    def to_string_notation(self) -> str:
        """Convert to standard poker notation (e.g., 'AKs', 'QQ', '72o')."""
        if self.is_pocket_pair:
            return f"{self.high_card_rank.symbol}{self.low_card_rank.symbol}"
        
        high = self.high_card_rank.symbol
        low = self.low_card_rank.symbol
        suited = "s" if self.is_suited else "o"
        return f"{high}{low}{suited}"
    
    @classmethod
    def from_string(cls, notation: str) -> 'HoleCards':
        """Create hole cards from notation.
        
        Args:
            notation: String like 'AKs', 'QQ', '72o'
            
        Returns:
            HoleCards instance
        """
        if len(notation) == 2:
            # Pocket pair
            rank = Rank.from_symbol(notation[0])
            card1 = Card(rank, Suit.HEARTS)
            card2 = Card(rank, Suit.SPADES)
        elif len(notation) == 3:
            # Non-pair
            rank1 = Rank.from_symbol(notation[0])
            rank2 = Rank.from_symbol(notation[1])
            
            if notation[2] == 's':
                # Suited
                card1 = Card(rank1, Suit.HEARTS)
                card2 = Card(rank2, Suit.HEARTS)
            else:
                # Offsuit
                card1 = Card(rank1, Suit.HEARTS)
                card2 = Card(rank2, Suit.SPADES)
        else:
            raise ValueError(f"Invalid hole cards notation: {notation}")
        
        return cls(card1, card2)


class Deck:
    """Optimized deck implementation with O(1) deal operations."""
    
    # Class-level card pool (created once)
    _ALL_CARDS = None
    
    @classmethod
    def _initialize_cards(cls):
        """Initialize the global card pool."""
        if cls._ALL_CARDS is None:
            cls._ALL_CARDS = []
            for rank in Rank:
                for suit in Suit:
                    cls._ALL_CARDS.append(Card(rank, suit))
    
    def __init__(self):
        """Initialize a new deck."""
        # Ensure cards are initialized
        self._initialize_cards()
        
        # Use indices for O(1) operations
        self._available_indices = list(range(52))
        self._rng = random.Random()  # Own RNG for thread safety
    
    def reset(self):
        """Reset deck to full 52 cards."""
        self._available_indices = list(range(52))
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck."""
        self._rng.shuffle(self._available_indices)
    
    def cards_remaining(self) -> int:
        """Get number of cards remaining."""
        return len(self._available_indices)
    
    def deal_card(self) -> Card:
        """Deal a single card in O(1) time.
        
        Returns:
            A card from the deck
            
        Raises:
            ValueError: If deck is empty
        """
        if not self._available_indices:
            raise ValueError("No more cards to deal")
        
        # Pop from end for O(1)
        card_index = self._available_indices.pop()
        return self._ALL_CARDS[card_index]
    
    def deal_cards(self, num_cards: int) -> List[Card]:
        """Deal multiple cards efficiently.
        
        Args:
            num_cards: Number of cards to deal
            
        Returns:
            List of cards
            
        Raises:
            ValueError: If not enough cards
        """
        if len(self._available_indices) < num_cards:
            raise ValueError(f"Not enough cards to deal {num_cards}")
        
        cards = []
        for _ in range(num_cards):
            cards.append(self.deal_card())
        
        return cards
    
    def deal_hole_cards(self) -> HoleCards:
        """Deal two cards as hole cards.
        
        Returns:
            HoleCards instance
            
        Raises:
            ValueError: If less than 2 cards remain
        """
        if len(self._available_indices) < 2:
            raise ValueError("Not enough cards to deal hole cards")
        
        return HoleCards(self.deal_card(), self.deal_card())
    
    def deal_multiple_hole_cards(self, num_players: int) -> List[HoleCards]:
        """Deal hole cards to multiple players efficiently.
        
        Args:
            num_players: Number of players
            
        Returns:
            List of HoleCards
            
        Raises:
            ValueError: If not enough cards
        """
        if len(self._available_indices) < num_players * 2:
            raise ValueError(f"Not enough cards for {num_players} players")
        
        return [self.deal_hole_cards() for _ in range(num_players)]
    
    def remove_cards(self, cards: List[Card]):
        """Remove specific cards from deck.
        
        Useful for dealing known cards.
        
        Args:
            cards: Cards to remove
            
        Raises:
            ValueError: If card not in deck
        """
        for card in cards:
            card_idx = card.index
            if card_idx in self._available_indices:
                self._available_indices.remove(card_idx)
            else:
                raise ValueError(f"Card {card} not in deck")


# Initialize the card pool at import time
Deck._initialize_cards()