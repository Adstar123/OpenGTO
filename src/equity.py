"""
Preflop equity calculator for Texas Hold'em.
Uses lookup tables and hand strength calculations for performance.
"""
from typing import Dict, Tuple
from itertools import combinations
from src.card import Hand, Card, Rank, create_deck
import pickle
import os


class PreflopEquityCalculator:
    """
    Calculates preflop equity between hands.
    Uses precomputed lookup tables for performance.
    """

    def __init__(self):
        self.equity_cache: Dict[Tuple[str, str], float] = {}
        self._load_or_compute_equity_table()

    def _load_or_compute_equity_table(self):
        """Load precomputed equity table or compute if not available."""
        cache_file = "data/preflop_equity_cache.pkl"

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.equity_cache = pickle.load(f)
        else:
            print("Computing preflop equity table (this may take a while)...")
            self._compute_equity_table()
            os.makedirs("data", exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.equity_cache, f)
            print(f"Equity table cached to {cache_file}")

    def _compute_equity_table(self):
        """
        Compute all preflop head-to-head equities.
        This is a simplified version that uses hand strength approximations.
        For production, you would want Monte Carlo simulation or exact enumeration.
        """
        # For now, use simplified hand strength model
        # In a production system, you'd run millions of simulations
        all_hands = self._generate_all_hand_types()

        for hand1_type in all_hands:
            for hand2_type in all_hands:
                if hand1_type != hand2_type:
                    equity = self._estimate_equity(hand1_type, hand2_type)
                    key = tuple(sorted([hand1_type, hand2_type]))
                    if key not in self.equity_cache:
                        self.equity_cache[key] = equity

    def _generate_all_hand_types(self) -> list:
        """Generate all unique hand types (169 distinct starting hands)."""
        hands = []
        ranks = list(Rank)

        # Pairs
        for rank in ranks:
            hands.append(f"{self._rank_to_str(rank)}{self._rank_to_str(rank)}")

        # Suited and offsuit non-pairs
        for i, rank1 in enumerate(ranks):
            for rank2 in ranks[i+1:]:
                r1_str = self._rank_to_str(rank1)
                r2_str = self._rank_to_str(rank2)
                # Higher rank first
                if rank1 > rank2:
                    hands.append(f"{r1_str}{r2_str}s")
                    hands.append(f"{r1_str}{r2_str}o")
                else:
                    hands.append(f"{r2_str}{r1_str}s")
                    hands.append(f"{r2_str}{r1_str}o")

        return list(set(hands))

    def _rank_to_str(self, rank: Rank) -> str:
        """Convert rank to string."""
        rank_map = {
            Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5',
            Rank.SIX: '6', Rank.SEVEN: '7', Rank.EIGHT: '8', Rank.NINE: '9',
            Rank.TEN: 'T', Rank.JACK: 'J', Rank.QUEEN: 'Q', Rank.KING: 'K',
            Rank.ACE: 'A'
        }
        return rank_map[rank]

    def _estimate_equity(self, hand1_str: str, hand2_str: str) -> float:
        """
        Simplified equity estimation based on hand strength.
        This is a placeholder - in production, use Monte Carlo simulation.
        """
        strength1 = self._hand_strength(hand1_str)
        strength2 = self._hand_strength(hand2_str)

        # Simple logistic model for equity
        diff = strength1 - strength2
        equity = 1.0 / (1.0 + pow(10, -diff / 400.0))
        return equity

    def _hand_strength(self, hand_str: str) -> float:
        """
        Estimate raw hand strength (0-100 scale).
        Based on Chen formula and adjustments.
        """
        if len(hand_str) < 2:
            return 0.0

        rank_values = {
            'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10,
            '9': 9, '8': 8, '7': 7, '6': 6, '5': 5,
            '4': 4, '3': 3, '2': 2
        }

        rank1 = rank_values[hand_str[0]]
        rank2 = rank_values[hand_str[1]]

        # Ensure rank1 >= rank2
        if rank1 < rank2:
            rank1, rank2 = rank2, rank1

        is_pair = (rank1 == rank2)
        is_suited = len(hand_str) == 3 and hand_str[2] == 's'

        # Base strength from high card
        strength = rank1 * 10

        # Pair bonus
        if is_pair:
            strength += rank1 * 8

        # High card and kicker
        strength += rank2 * 2

        # Suited bonus
        if is_suited:
            strength += 20

        # Gap penalty (connectedness bonus)
        gap = rank1 - rank2
        if gap == 0:  # Pair
            pass
        elif gap == 1:  # Connected
            strength += 15
        elif gap == 2:
            strength += 10
        elif gap == 3:
            strength += 5
        else:
            strength -= (gap - 3) * 3

        return strength

    def get_equity(self, hand1: Hand, hand2: Hand) -> float:
        """
        Get the equity of hand1 vs hand2.
        Returns a value between 0 and 1.
        """
        hand1_str = hand1.to_string()
        hand2_str = hand2.to_string()

        key = tuple(sorted([hand1_str, hand2_str]))

        if key in self.equity_cache:
            equity = self.equity_cache[key]
            # Return correct equity based on which hand is which
            if hand1_str == key[0]:
                return equity
            else:
                return 1.0 - equity

        # If not in cache, estimate
        return self._estimate_equity(hand1_str, hand2_str)
