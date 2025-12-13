"""
Showdown evaluation for determining winners.

This implements the RULES of poker hand rankings - not strategy.
The neural network needs to know who wins at showdown to learn from outcomes.

Hand rankings (strongest to weakest):
1. Royal Flush
2. Straight Flush
3. Four of a Kind
4. Full House
5. Flush
6. Straight
7. Three of a Kind
8. Two Pair
9. One Pair
10. High Card
"""
from typing import List, Tuple, Optional
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum

from .card import Card, Rank, Suit, HoleCards


class HandRank(IntEnum):
    """Poker hand rankings. Higher value = stronger hand."""
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


@dataclass
class HandEvaluation:
    """Result of evaluating a poker hand."""
    rank: HandRank
    # Kickers for tie-breaking, in order of importance (highest first)
    kickers: Tuple[int, ...]

    def __lt__(self, other: 'HandEvaluation') -> bool:
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.kickers < other.kickers

    def __eq__(self, other: 'HandEvaluation') -> bool:
        return self.rank == other.rank and self.kickers == other.kickers

    def __le__(self, other: 'HandEvaluation') -> bool:
        return self < other or self == other


def evaluate_hand(cards: List[Card]) -> HandEvaluation:
    """
    Evaluate a poker hand (5-7 cards) and return its ranking.
    For 6-7 cards, finds the best 5-card hand.
    """
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")

    if len(cards) == 5:
        return _evaluate_five_cards(cards)

    # For 6-7 cards, try all 5-card combinations
    from itertools import combinations
    best = None
    for combo in combinations(cards, 5):
        eval_result = _evaluate_five_cards(list(combo))
        if best is None or eval_result > best:
            best = eval_result
    return best


def _evaluate_five_cards(cards: List[Card]) -> HandEvaluation:
    """Evaluate exactly 5 cards."""
    ranks = sorted([c.rank.value for c in cards], reverse=True)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1
    is_straight, straight_high = _check_straight(ranks)

    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    # Check for each hand type from strongest to weakest
    if is_straight and is_flush:
        if straight_high == 14:  # Ace high
            return HandEvaluation(HandRank.ROYAL_FLUSH, (14,))
        return HandEvaluation(HandRank.STRAIGHT_FLUSH, (straight_high,))

    if counts == [4, 1]:
        quad_rank = _get_rank_with_count(rank_counts, 4)
        kicker = _get_rank_with_count(rank_counts, 1)
        return HandEvaluation(HandRank.FOUR_OF_A_KIND, (quad_rank, kicker))

    if counts == [3, 2]:
        trips_rank = _get_rank_with_count(rank_counts, 3)
        pair_rank = _get_rank_with_count(rank_counts, 2)
        return HandEvaluation(HandRank.FULL_HOUSE, (trips_rank, pair_rank))

    if is_flush:
        return HandEvaluation(HandRank.FLUSH, tuple(ranks))

    if is_straight:
        return HandEvaluation(HandRank.STRAIGHT, (straight_high,))

    if counts == [3, 1, 1]:
        trips_rank = _get_rank_with_count(rank_counts, 3)
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return HandEvaluation(HandRank.THREE_OF_A_KIND, (trips_rank,) + tuple(kickers))

    if counts == [2, 2, 1]:
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kicker = _get_rank_with_count(rank_counts, 1)
        return HandEvaluation(HandRank.TWO_PAIR, (pairs[0], pairs[1], kicker))

    if counts == [2, 1, 1, 1]:
        pair_rank = _get_rank_with_count(rank_counts, 2)
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return HandEvaluation(HandRank.ONE_PAIR, (pair_rank,) + tuple(kickers))

    # High card
    return HandEvaluation(HandRank.HIGH_CARD, tuple(ranks))


def _check_straight(ranks: List[int]) -> Tuple[bool, int]:
    """Check if sorted ranks form a straight. Returns (is_straight, high_card)."""
    unique_ranks = sorted(set(ranks), reverse=True)

    if len(unique_ranks) < 5:
        return False, 0

    # Check for regular straight
    for i in range(len(unique_ranks) - 4):
        if unique_ranks[i] - unique_ranks[i + 4] == 4:
            return True, unique_ranks[i]

    # Check for wheel (A-2-3-4-5)
    if set([14, 5, 4, 3, 2]).issubset(set(unique_ranks)):
        return True, 5  # 5-high straight

    return False, 0


def _get_rank_with_count(rank_counts: Counter, count: int) -> int:
    """Get the rank that appears exactly 'count' times."""
    for rank, c in rank_counts.items():
        if c == count:
            return rank
    return 0


def compare_hands(
    hole_cards1: HoleCards,
    hole_cards2: HoleCards,
    board: List[Card]
) -> int:
    """
    Compare two hands given a board.

    Returns:
        1 if hand1 wins
        -1 if hand2 wins
        0 if tie
    """
    cards1 = [hole_cards1.card1, hole_cards1.card2] + board
    cards2 = [hole_cards2.card1, hole_cards2.card2] + board

    eval1 = evaluate_hand(cards1)
    eval2 = evaluate_hand(cards2)

    if eval1 > eval2:
        return 1
    elif eval1 < eval2:
        return -1
    return 0


def run_out_board(
    hole_cards_list: List[HoleCards],
    dead_cards: Optional[List[Card]] = None,
    num_simulations: int = 1000
) -> List[float]:
    """
    Run Monte Carlo simulation to estimate equity for each hand.

    This is used for training reward calculation when hands go all-in preflop.
    The neural network learns from these outcomes.

    Args:
        hole_cards_list: List of hole cards for each player
        dead_cards: Cards that cannot appear on board
        num_simulations: Number of random boards to simulate

    Returns:
        List of equity (win probability) for each hand
    """
    from .card import Deck
    import random

    num_players = len(hole_cards_list)
    wins = [0.0] * num_players
    ties = [0.0] * num_players

    # Build set of unavailable cards
    unavailable = set()
    for hc in hole_cards_list:
        unavailable.add(hc.card1)
        unavailable.add(hc.card2)
    if dead_cards:
        unavailable.update(dead_cards)

    # Get available cards for board
    deck = Deck()
    available = [c for c in deck.cards if c not in unavailable]

    for _ in range(num_simulations):
        # Deal random board
        board = random.sample(available, 5)

        # Evaluate each hand
        evaluations = []
        for hc in hole_cards_list:
            cards = [hc.card1, hc.card2] + board
            evaluations.append(evaluate_hand(cards))

        # Find winner(s)
        best_eval = max(evaluations)
        winners = [i for i, e in enumerate(evaluations) if e == best_eval]

        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            # Split pot
            for w in winners:
                ties[w] += 1

    # Calculate equity
    equities = []
    for i in range(num_players):
        equity = (wins[i] + ties[i] / 2) / num_simulations
        equities.append(equity)

    return equities
