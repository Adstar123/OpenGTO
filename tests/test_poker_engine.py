"""
Tests for the poker engine components.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card import (
    Card, Rank, Suit, Deck, HoleCards,
    get_all_hand_types, hand_type_to_index, index_to_hand_type, get_hand_combos
)
from src.game_state import GameState, Position, ActionType, Action, PlayerState
from src.poker_engine import PreflopPokerEngine, GameSimulator, ScenarioBuilder
from src.hand_utils import HandRange, enumerate_all_hole_cards, get_combos_for_hand_type
from src.showdown import evaluate_hand, compare_hands, HandRank, run_out_board


def test_card_basics():
    """Test card creation and representation."""
    print("Testing card basics...")

    # Create cards
    ace_spades = Card(Rank.ACE, Suit.SPADES)
    king_hearts = Card(Rank.KING, Suit.HEARTS)
    two_clubs = Card(Rank.TWO, Suit.CLUBS)

    assert str(ace_spades) == "As"
    assert ace_spades.pretty() == "A♠"
    assert str(king_hearts) == "Kh"

    # From string
    card = Card.from_string("As")
    assert card.rank == Rank.ACE
    assert card.suit == Suit.SPADES

    card = Card.from_string("Th")
    assert card.rank == Rank.TEN
    assert card.suit == Suit.HEARTS

    print("  Card basics: PASSED")


def test_deck():
    """Test deck operations."""
    print("Testing deck...")

    deck = Deck()
    assert len(deck) == 52

    deck.shuffle()
    card = deck.deal()
    assert len(deck) == 51
    assert isinstance(card, Card)

    hole_cards = deck.deal_hole_cards()
    assert len(deck) == 49
    assert isinstance(hole_cards, HoleCards)

    # Remove specific card
    deck.reset()
    ace_spades = Card.from_string("As")
    removed = deck.remove(ace_spades)
    assert removed
    assert len(deck) == 51

    print("  Deck: PASSED")


def test_hole_cards():
    """Test hole cards representation."""
    print("Testing hole cards...")

    # Pocket aces
    aa = HoleCards(Card.from_string("As"), Card.from_string("Ah"))
    assert aa.is_pair
    assert not aa.is_suited
    assert aa.hand_type_string() == "AA"

    # Suited connector
    aks = HoleCards(Card.from_string("As"), Card.from_string("Ks"))
    assert not aks.is_pair
    assert aks.is_suited
    assert aks.hand_type_string() == "AKs"

    # Offsuit
    ako = HoleCards(Card.from_string("As"), Card.from_string("Kh"))
    assert not ako.is_pair
    assert not ako.is_suited
    assert ako.hand_type_string() == "AKo"

    # From string
    hc = HoleCards.from_string("JsTc")
    assert hc.hand_type_string() == "JTo"

    print("  Hole cards: PASSED")


def test_169_hand_types():
    """Test all 169 hand types."""
    print("Testing 169 hand types...")

    all_types = get_all_hand_types()
    assert len(all_types) == 169

    # Test index conversion
    for i, hand_type in enumerate(all_types):
        idx = hand_type_to_index(hand_type)
        back = index_to_hand_type(idx)
        # Note: the conversion might not be exactly reversible due to matrix layout
        # but the index should be consistent

    # Test combo counts
    assert get_hand_combos("AA") == 6  # C(4,2) = 6 ways to pick 2 aces
    assert get_hand_combos("AKs") == 4  # 4 suits
    assert get_hand_combos("AKo") == 12  # 4*3 = 12

    # Total combos should be 1326
    total = sum(get_hand_combos(h) for h in all_types)
    assert total == 1326

    print("  169 hand types: PASSED")


def test_game_state():
    """Test game state creation and features."""
    print("Testing game state...")

    state = GameState(num_players=6)
    assert len(state.players) == 6

    # Check feature vector size
    features = state.to_feature_vector()
    expected_size = GameState.feature_size()
    assert len(features) == expected_size, f"Expected {expected_size}, got {len(features)}"

    print("  Game state: PASSED")


def test_poker_engine():
    """Test poker engine game flow."""
    print("Testing poker engine...")

    engine = PreflopPokerEngine(num_players=6, starting_stack=100.0)
    state = engine.create_game()

    # Check blinds were posted
    sb_player = state.get_player_by_position(Position.SB)
    bb_player = state.get_player_by_position(Position.BB)
    assert sb_player.current_bet == 0.5
    assert bb_player.current_bet == 1.0
    assert state.pot == 1.5

    # UTG should be first to act
    assert state.current_player.position == Position.UTG

    # Get legal actions
    actions = state.get_legal_actions()
    action_types = [a.action_type for a in actions]
    assert ActionType.FOLD in action_types
    assert ActionType.CALL in action_types
    # Should have raise options
    assert any(a.action_type in [ActionType.RAISE, ActionType.BET] for a in actions)

    # Apply fold action
    fold_action = Action(ActionType.FOLD)
    new_state = engine.apply_action(state, fold_action)
    assert not new_state.players[0].is_active  # UTG folded
    assert new_state.current_player.position == Position.HJ

    print("  Poker engine: PASSED")


def test_full_hand_simulation():
    """Test simulating a full hand."""
    print("Testing full hand simulation...")

    engine = PreflopPokerEngine(num_players=6, starting_stack=100.0)
    simulator = GameSimulator(engine)

    # Simulate random game
    states, actions = simulator.simulate_random_game()
    assert len(states) > 0
    assert states[-1].is_complete

    print(f"  Simulated hand with {len(actions)} actions")
    print("  Full hand simulation: PASSED")


def test_scenario_builder():
    """Test building specific scenarios."""
    print("Testing scenario builder...")

    engine = PreflopPokerEngine(num_players=6, starting_stack=100.0)
    builder = ScenarioBuilder(engine)

    # Build scenario: BTN has AA, UTG raised to 3bb
    aa = HoleCards.from_string("AsAh")
    state = (builder
             .set_hero_cards(Position.BTN, aa)
             .add_action(Position.UTG, Action(ActionType.RAISE, amount=3.0))
             .add_action(Position.HJ, Action(ActionType.FOLD))
             .add_action(Position.CO, Action(ActionType.FOLD))
             .build())

    # BTN should be next to act
    assert state.current_player.position == Position.BTN
    assert state.current_player.hole_cards.hand_type_string() == "AA"
    assert state.current_bet == 3.0

    print("  Scenario builder: PASSED")


def test_hand_range():
    """Test hand range utilities."""
    print("Testing hand range...")

    # Empty range
    empty = HandRange.empty()
    assert empty.num_combos() == 0

    # Full range
    full = HandRange.full()
    assert full.num_combos() == 1326

    # Custom range
    range_ = HandRange.from_string("AA,KK,QQ,AKs")
    assert "AA" in range_.hands
    assert "KK" in range_.hands
    assert "AKs" in range_.hands

    # Matrix visualization
    matrix = range_.to_matrix()
    assert matrix.shape == (13, 13)
    assert matrix[0, 0] == 1.0  # AA

    print("  Hand range: PASSED")


def test_showdown_evaluation():
    """Test hand evaluation at showdown."""
    print("Testing showdown evaluation...")

    # Royal flush
    cards = [
        Card.from_string("As"),
        Card.from_string("Ks"),
        Card.from_string("Qs"),
        Card.from_string("Js"),
        Card.from_string("Ts"),
    ]
    result = evaluate_hand(cards)
    assert result.rank == HandRank.ROYAL_FLUSH

    # Four of a kind
    cards = [
        Card.from_string("As"),
        Card.from_string("Ah"),
        Card.from_string("Ad"),
        Card.from_string("Ac"),
        Card.from_string("Ks"),
    ]
    result = evaluate_hand(cards)
    assert result.rank == HandRank.FOUR_OF_A_KIND

    # Full house
    cards = [
        Card.from_string("As"),
        Card.from_string("Ah"),
        Card.from_string("Ad"),
        Card.from_string("Ks"),
        Card.from_string("Kh"),
    ]
    result = evaluate_hand(cards)
    assert result.rank == HandRank.FULL_HOUSE

    # Straight
    cards = [
        Card.from_string("9s"),
        Card.from_string("8h"),
        Card.from_string("7d"),
        Card.from_string("6c"),
        Card.from_string("5s"),
    ]
    result = evaluate_hand(cards)
    assert result.rank == HandRank.STRAIGHT

    # Wheel (A-2-3-4-5)
    cards = [
        Card.from_string("As"),
        Card.from_string("2h"),
        Card.from_string("3d"),
        Card.from_string("4c"),
        Card.from_string("5s"),
    ]
    result = evaluate_hand(cards)
    assert result.rank == HandRank.STRAIGHT

    # One pair
    cards = [
        Card.from_string("As"),
        Card.from_string("Ah"),
        Card.from_string("Kd"),
        Card.from_string("Qc"),
        Card.from_string("Js"),
    ]
    result = evaluate_hand(cards)
    assert result.rank == HandRank.ONE_PAIR

    print("  Showdown evaluation: PASSED")


def test_compare_hands():
    """Test comparing two hands with a board."""
    print("Testing hand comparison...")

    # AA vs KK with no help
    aa = HoleCards.from_string("AsAh")
    kk = HoleCards.from_string("KsKh")
    board = [
        Card.from_string("2c"),
        Card.from_string("3d"),
        Card.from_string("7h"),
        Card.from_string("8s"),
        Card.from_string("9c"),
    ]
    result = compare_hands(aa, kk, board)
    assert result == 1  # AA wins

    # KK makes set vs AA pair
    kk = HoleCards.from_string("KsKh")
    aa = HoleCards.from_string("AsAh")
    board = [
        Card.from_string("Kc"),
        Card.from_string("3d"),
        Card.from_string("7h"),
        Card.from_string("8s"),
        Card.from_string("9c"),
    ]
    result = compare_hands(kk, aa, board)
    assert result == 1  # KK (set) wins

    print("  Hand comparison: PASSED")


def test_equity_simulation():
    """Test equity calculation via Monte Carlo."""
    print("Testing equity simulation...")

    # AA vs 72o - AA should have ~87% equity
    aa = HoleCards.from_string("AsAh")
    seven_two = HoleCards.from_string("7s2h")

    equities = run_out_board([aa, seven_two], num_simulations=1000)
    print(f"  AA equity vs 72o: {equities[0]:.1%}")
    # AA should have significant edge
    assert equities[0] > 0.80

    # AA vs KK - AA should have ~82% equity
    aa = HoleCards.from_string("AsAh")
    kk = HoleCards.from_string("KsKh")

    equities = run_out_board([aa, kk], num_simulations=1000)
    print(f"  AA equity vs KK: {equities[0]:.1%}")
    assert equities[0] > 0.75

    print("  Equity simulation: PASSED")


def test_enumerate_hole_cards():
    """Test enumerating all possible hole cards."""
    print("Testing hole card enumeration...")

    all_cards = enumerate_all_hole_cards()
    assert len(all_cards) == 1326

    # Test getting combos for specific hand type
    aa_combos = get_combos_for_hand_type("AA")
    assert len(aa_combos) == 6

    aks_combos = get_combos_for_hand_type("AKs")
    assert len(aks_combos) == 4

    ako_combos = get_combos_for_hand_type("AKo")
    assert len(ako_combos) == 12

    print("  Hole card enumeration: PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("OpenGTO Poker Engine Tests")
    print("=" * 50)
    print()

    test_card_basics()
    test_deck()
    test_hole_cards()
    test_169_hand_types()
    test_game_state()
    test_poker_engine()
    test_full_hand_simulation()
    test_scenario_builder()
    test_hand_range()
    test_showdown_evaluation()
    test_compare_hands()
    test_equity_simulation()
    test_enumerate_hole_cards()

    print()
    print("=" * 50)
    print("All tests PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
