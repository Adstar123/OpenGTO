"""
Interactive GTO Poker Trainer Interface.

This is the user-facing trainer where you can:
- Configure scenarios (position, cards, stack sizes, opponent actions)
- Make decisions and get GTO feedback
- Track your accuracy over time
- Practice specific situations
"""
import numpy as np
import random
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .game_state import GameState, Action, ActionType, Position
from .poker_engine import PreflopPokerEngine, ScenarioBuilder
from .card import Card, HoleCards, Deck, get_all_hand_types
from .hand_utils import get_combos_for_hand_type, HandRange
from .information_set import (
    InformationSet, get_legal_actions_mask, NUM_ACTIONS,
    ACTION_TYPE_TO_IDX, IDX_TO_ACTION_TYPE
)
from .neural_network import DeepCFRNetworks


# Action names for display
ACTION_NAMES = ['Fold', 'Check', 'Call', 'Bet', 'Raise', 'All-In']
ACTION_SHORTCUTS = {'f': 0, 'x': 1, 'c': 2, 'b': 3, 'r': 4, 'a': 5}


@dataclass
class Decision:
    """Records a single decision made by the user."""
    hand: str
    position: str
    scenario: str
    user_action: str
    gto_action: str
    gto_probs: Dict[str, float]
    was_correct: bool
    ev_loss: float  # Estimated EV loss from suboptimal play
    timestamp: str


@dataclass
class UserStats:
    """Tracks user statistics over time."""
    total_hands: int = 0
    correct_decisions: int = 0
    decisions_by_position: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # pos -> (correct, total)
    decisions_by_hand_type: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # hand -> (correct, total)
    recent_decisions: List[Decision] = field(default_factory=list)
    session_start: str = ""
    total_ev_loss: float = 0.0

    def record_decision(self, decision: Decision):
        """Record a decision."""
        self.total_hands += 1
        if decision.was_correct:
            self.correct_decisions += 1

        # By position
        pos = decision.position
        if pos not in self.decisions_by_position:
            self.decisions_by_position[pos] = (0, 0)
        correct, total = self.decisions_by_position[pos]
        self.decisions_by_position[pos] = (
            correct + (1 if decision.was_correct else 0),
            total + 1
        )

        # By hand type
        hand = decision.hand
        if hand not in self.decisions_by_hand_type:
            self.decisions_by_hand_type[hand] = (0, 0)
        correct, total = self.decisions_by_hand_type[hand]
        self.decisions_by_hand_type[hand] = (
            correct + (1 if decision.was_correct else 0),
            total + 1
        )

        self.total_ev_loss += decision.ev_loss
        self.recent_decisions.append(decision)

        # Keep only last 100 decisions in memory
        if len(self.recent_decisions) > 100:
            self.recent_decisions = self.recent_decisions[-100:]

    @property
    def accuracy(self) -> float:
        if self.total_hands == 0:
            return 0.0
        return self.correct_decisions / self.total_hands

    def get_weakest_positions(self, min_hands: int = 5) -> List[Tuple[str, float]]:
        """Get positions with lowest accuracy."""
        results = []
        for pos, (correct, total) in self.decisions_by_position.items():
            if total >= min_hands:
                results.append((pos, correct / total))
        return sorted(results, key=lambda x: x[1])

    def get_weakest_hands(self, min_hands: int = 3) -> List[Tuple[str, float]]:
        """Get hand types with lowest accuracy."""
        results = []
        for hand, (correct, total) in self.decisions_by_hand_type.items():
            if total >= min_hands:
                results.append((hand, correct / total))
        return sorted(results, key=lambda x: x[1])[:10]

    def save(self, path: str):
        """Save stats to file."""
        data = {
            'total_hands': self.total_hands,
            'correct_decisions': self.correct_decisions,
            'decisions_by_position': self.decisions_by_position,
            'decisions_by_hand_type': self.decisions_by_hand_type,
            'total_ev_loss': float(self.total_ev_loss),  # Convert numpy float32 to Python float
            'session_start': self.session_start
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'UserStats':
        """Load stats from file."""
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            stats = cls()
            stats.total_hands = data.get('total_hands', 0)
            stats.correct_decisions = data.get('correct_decisions', 0)
            stats.decisions_by_position = data.get('decisions_by_position', {})
            stats.decisions_by_hand_type = data.get('decisions_by_hand_type', {})
            stats.total_ev_loss = data.get('total_ev_loss', 0.0)
            return stats
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Could not load stats file ({e}). Starting fresh.")
            return cls()


@dataclass
class TrainerScenario:
    """A scenario for the user to solve."""
    state: GameState
    hero_position: Position
    hero_hand: HoleCards
    villain_actions: List[str]  # Description of what happened before
    legal_actions: List[str]
    legal_action_indices: List[int]
    description: str
    stack_size: float


class GTOTrainerInterface:
    """
    Interactive GTO Poker Trainer.

    Load a trained model and practice preflop decisions.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_players: int = 6,
        device: str = 'cpu'
    ):
        self.checkpoint_path = checkpoint_path
        self.num_players = num_players
        self.device = device

        # Load model
        self.networks: Optional[DeepCFRNetworks] = None
        self.engine: Optional[PreflopPokerEngine] = None

        # User stats
        self.stats = UserStats()
        self.stats_path = "user_stats.json"

        # Settings
        self.stack_size = 100.0
        self.positions_to_train: List[Position] = list(Position)[:num_players]
        self.hand_filter: Optional[List[str]] = None  # None = all hands

    def load_model(self):
        """Load the trained model."""
        import torch

        print(f"Loading model from {self.checkpoint_path}...")

        # First load checkpoint to get config
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Get config from checkpoint if available
        config = checkpoint.get('config', {})
        hidden_sizes = config.get('hidden_sizes', (256, 256, 128))
        self.num_players = config.get('num_players', self.num_players)

        # Convert list to tuple if needed (JSON doesn't preserve tuples)
        if isinstance(hidden_sizes, list):
            hidden_sizes = tuple(hidden_sizes)

        print(f"  Network architecture: {hidden_sizes}")

        input_size = InformationSet.feature_size()
        self.networks = DeepCFRNetworks(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_actions=NUM_ACTIONS,
            device=self.device
        )

        self.networks.regret_net.load_state_dict(checkpoint['networks']['regret_net'])
        self.networks.avg_strategy_net.load_state_dict(checkpoint['networks']['avg_strategy_net'])

        self.engine = PreflopPokerEngine(
            num_players=self.num_players,
            starting_stack=self.stack_size
        )

        print(f"Model loaded! ({self.num_players} players)")

        # Load user stats
        if os.path.exists(self.stats_path):
            self.stats = UserStats.load(self.stats_path)
            print(f"Loaded stats: {self.stats.total_hands} hands, {self.stats.accuracy*100:.1f}% accuracy")

    def get_gto_strategy(self, state: GameState) -> np.ndarray:
        """Get GTO strategy for a state."""
        info_set = InformationSet.from_game_state(state)
        features = info_set.to_feature_vector()
        legal_mask = get_legal_actions_mask(state)
        return self.networks.get_average_strategy(features, legal_mask)

    def generate_random_scenario(self) -> TrainerScenario:
        """Generate a random training scenario."""
        # Random position
        hero_position = random.choice(self.positions_to_train)

        # Random hand (or from filter)
        if self.hand_filter:
            hand_type = random.choice(self.hand_filter)
        else:
            hand_type = random.choice(get_all_hand_types())

        # Get a specific combo
        combos = get_combos_for_hand_type(hand_type)
        hero_hand = random.choice(combos)

        # Random stack (80-120bb for variety)
        stack = random.uniform(80, 120)

        return self._build_scenario(hero_position, hero_hand, stack)

    def generate_scenario(
        self,
        position: str,
        hand: str,
        stack: float = 100.0,
        action_history: Optional[List[Tuple[str, str]]] = None
    ) -> TrainerScenario:
        """
        Generate a specific scenario.

        Args:
            position: Hero's position (e.g., "BTN", "BB")
            hand: Hand like "AKs" or specific like "AsKs"
            stack: Stack size in BB
            action_history: List of (position, action) tuples before hero
        """
        hero_position = Position[position.upper()]

        # Parse hand
        if len(hand) <= 3:
            # Hand type like "AKs"
            combos = get_combos_for_hand_type(hand)
            hero_hand = random.choice(combos) if combos else None
        else:
            # Specific cards like "AsKs"
            hero_hand = HoleCards.from_string(hand)

        if hero_hand is None:
            raise ValueError(f"Invalid hand: {hand}")

        return self._build_scenario(hero_position, hero_hand, stack, action_history)

    def _build_scenario(
        self,
        hero_position: Position,
        hero_hand: HoleCards,
        stack: float,
        action_history: Optional[List[Tuple[str, str]]] = None
    ) -> TrainerScenario:
        """Build a scenario from components."""
        self.engine = PreflopPokerEngine(
            num_players=self.num_players,
            starting_stack=stack
        )

        builder = ScenarioBuilder(self.engine)
        stacks = [stack] * self.num_players
        builder.set_stacks(stacks)
        builder.set_hero_cards(hero_position, hero_hand)

        villain_actions = []
        preflop_order = Position.preflop_order(self.num_players)

        if action_history:
            # Use provided action history
            for pos_str, action_str in action_history:
                pos = Position[pos_str.upper()]
                if action_str.lower() == 'fold':
                    builder.add_action(pos, Action(ActionType.FOLD))
                    villain_actions.append(f"{pos.name} folds")
                elif action_str.lower() == 'call':
                    builder.add_action(pos, Action(ActionType.CALL, amount=1.0))
                    villain_actions.append(f"{pos.name} calls")
                elif action_str.lower().startswith('raise'):
                    # Parse raise amount if provided
                    parts = action_str.split()
                    amount = float(parts[1]) if len(parts) > 1 else 3.0
                    builder.add_action(pos, Action(ActionType.RAISE, amount=amount))
                    villain_actions.append(f"{pos.name} raises to {amount}bb")
        else:
            # Generate random action history up to hero
            # First, decide if there will be a raise and from which position
            raise_happened = False

            for pos in preflop_order:
                if pos == hero_position:
                    break
                if pos in [Position.SB, Position.BB]:
                    continue  # Blinds already posted

                if raise_happened:
                    # After a raise, everyone else folds until hero
                    builder.add_action(pos, Action(ActionType.FOLD))
                    villain_actions.append(f"{pos.name} folds")
                else:
                    # Random action: 70% fold, 30% raise (standardized to 2.5bb)
                    r = random.random()
                    if r < 0.70:
                        builder.add_action(pos, Action(ActionType.FOLD))
                        villain_actions.append(f"{pos.name} folds")
                    else:
                        # Standard open raise is 2.5bb
                        builder.add_action(pos, Action(ActionType.RAISE, amount=2.5))
                        villain_actions.append(f"{pos.name} raises to 2.5bb")
                        raise_happened = True

        state = builder.build()

        # Get legal actions
        legal_mask = get_legal_actions_mask(state)
        legal_indices = [i for i in range(NUM_ACTIONS) if legal_mask[i]]
        legal_names = [ACTION_NAMES[i] for i in legal_indices]

        # Build description
        if villain_actions:
            desc = " -> ".join(villain_actions) + f" -> {hero_position.name} to act"
        else:
            desc = f"{hero_position.name} to act (folded to you)"

        return TrainerScenario(
            state=state,
            hero_position=hero_position,
            hero_hand=hero_hand,
            villain_actions=villain_actions,
            legal_actions=legal_names,
            legal_action_indices=legal_indices,
            description=desc,
            stack_size=stack
        )

    def evaluate_decision(
        self,
        scenario: TrainerScenario,
        user_action_idx: int
    ) -> Tuple[Dict[str, float], bool, float, str]:
        """
        Evaluate a user's decision against GTO.

        Returns:
            Tuple of (gto_probs, was_correct, ev_loss, feedback_message)
        """
        # Get GTO strategy
        strategy = self.get_gto_strategy(scenario.state)
        legal_mask = get_legal_actions_mask(scenario.state)

        # Build probability dict for legal actions
        gto_probs = {}
        for i in range(NUM_ACTIONS):
            if legal_mask[i]:
                gto_probs[ACTION_NAMES[i]] = float(strategy[i])

        # Find GTO recommended action (highest probability)
        gto_action_idx = max(
            scenario.legal_action_indices,
            key=lambda i: strategy[i]
        )
        gto_action = ACTION_NAMES[gto_action_idx]

        user_action = ACTION_NAMES[user_action_idx]

        # Check if correct (within threshold of GTO)
        user_prob = strategy[user_action_idx]
        gto_prob = strategy[gto_action_idx]

        # Consider "correct" if user chose an action with >25% frequency
        # OR if their action is within 10% of the best action
        was_correct = user_prob >= 0.25 or (gto_prob - user_prob) < 0.10

        # Estimate EV loss (simplified)
        ev_loss = max(0, gto_prob - user_prob) * 10  # Scale by arbitrary factor

        # Generate feedback
        if was_correct:
            if user_action == gto_action:
                feedback = f"Perfect! {user_action} is the GTO play here."
            else:
                feedback = f"Good! {user_action} is acceptable. GTO prefers {gto_action} ({gto_prob*100:.0f}%) but {user_action} ({user_prob*100:.0f}%) is fine."
        else:
            feedback = f"Suboptimal. GTO recommends {gto_action} ({gto_prob*100:.0f}%). You chose {user_action} ({user_prob*100:.0f}%)."

        return gto_probs, was_correct, ev_loss, feedback

    def display_scenario(self, scenario: TrainerScenario) -> str:
        """Format scenario for display."""
        lines = []
        lines.append("")
        lines.append("=" * 50)
        lines.append(f"STACK: {scenario.stack_size:.0f}bb | POSITION: {scenario.hero_position.name}")
        lines.append("=" * 50)
        lines.append("")
        # Use ASCII-friendly format (avoid Unicode suit symbols for Windows compatibility)
        lines.append(f"Your hand: [{scenario.hero_hand.card1}] [{scenario.hero_hand.card2}] ({scenario.hero_hand.hand_type_string()})")
        lines.append("")

        if scenario.villain_actions:
            lines.append("Action:")
            for action in scenario.villain_actions:
                lines.append(f"  {action}")
        else:
            lines.append("Action folds to you.")

        lines.append("")
        lines.append(f"Pot: {scenario.state.pot:.1f}bb | To call: {scenario.state.current_bet - scenario.state.current_player.current_bet:.1f}bb")
        lines.append("")
        lines.append("Your options:")

        for i, (name, idx) in enumerate(zip(scenario.legal_actions, scenario.legal_action_indices)):
            shortcut = [k for k, v in ACTION_SHORTCUTS.items() if v == idx]
            shortcut_str = f"({shortcut[0]})" if shortcut else ""
            lines.append(f"  {i+1}. {name} {shortcut_str}")

        return "\n".join(lines)

    def display_result(
        self,
        gto_probs: Dict[str, float],
        was_correct: bool,
        feedback: str
    ) -> str:
        """Format result for display."""
        lines = []
        lines.append("")
        lines.append("-" * 50)

        if was_correct:
            lines.append("CORRECT!")
        else:
            lines.append("INCORRECT")

        lines.append("")
        lines.append(feedback)
        lines.append("")
        lines.append("GTO Strategy:")

        # Sort by probability
        sorted_probs = sorted(gto_probs.items(), key=lambda x: -x[1])
        for action, prob in sorted_probs:
            bar_len = int(prob * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            lines.append(f"  {action:8} [{bar}] {prob*100:5.1f}%")

        lines.append("-" * 50)

        return "\n".join(lines)

    def display_stats(self) -> str:
        """Format user stats for display."""
        lines = []
        lines.append("")
        lines.append("=" * 50)
        lines.append("YOUR STATISTICS")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"Total hands: {self.stats.total_hands}")
        lines.append(f"Accuracy: {self.stats.accuracy*100:.1f}%")
        lines.append(f"Estimated EV loss: {self.stats.total_ev_loss:.1f}bb")
        lines.append("")

        # By position
        if self.stats.decisions_by_position:
            lines.append("By Position:")
            for pos, (correct, total) in sorted(self.stats.decisions_by_position.items()):
                acc = correct / total * 100 if total > 0 else 0
                lines.append(f"  {pos:4} {correct}/{total} ({acc:.0f}%)")
            lines.append("")

        # Weakest hands
        weak_hands = self.stats.get_weakest_hands()
        if weak_hands:
            lines.append("Hands to work on:")
            for hand, acc in weak_hands[:5]:
                lines.append(f"  {hand}: {acc*100:.0f}%")

        lines.append("=" * 50)

        return "\n".join(lines)

    def run_interactive_session(self):
        """Run an interactive training session."""
        if self.networks is None:
            self.load_model()

        self.stats.session_start = datetime.now().isoformat()

        print("\n" + "=" * 50)
        print("GTO POKER TRAINER")
        print("=" * 50)
        print(f"\nModel: {self.checkpoint_path}")
        print(f"Players: {self.num_players}")
        print(f"Stack: {self.stack_size}bb")
        print("\nCommands:")
        print("  [number] or [f/x/c/b/r/a] - Make action")
        print("  'stats' - View your statistics")
        print("  'set stack [N]' - Set stack size")
        print("  'set pos [POS]' - Train specific position")
        print("  'set hand [HAND]' - Train specific hand")
        print("  'quit' - Exit")
        print("\n" + "=" * 50)

        try:
            while True:
                # Generate scenario
                scenario = self.generate_random_scenario()

                # Display
                print(self.display_scenario(scenario))

                # Get user input
                while True:
                    user_input = input("\nYour action: ").strip().lower()

                    if user_input == 'quit':
                        raise KeyboardInterrupt

                    if user_input == 'stats':
                        print(self.display_stats())
                        continue

                    if user_input.startswith('set '):
                        self._handle_set_command(user_input)
                        continue

                    # Parse action
                    action_idx = self._parse_action(user_input, scenario)
                    if action_idx is not None:
                        break
                    print("Invalid action. Try again.")

                # Evaluate
                gto_probs, was_correct, ev_loss, feedback = self.evaluate_decision(
                    scenario, action_idx
                )

                # Record decision
                decision = Decision(
                    hand=scenario.hero_hand.hand_type_string(),
                    position=scenario.hero_position.name,
                    scenario=scenario.description,
                    user_action=ACTION_NAMES[action_idx],
                    gto_action=max(gto_probs.keys(), key=lambda k: gto_probs[k]),
                    gto_probs=gto_probs,
                    was_correct=was_correct,
                    ev_loss=ev_loss,
                    timestamp=datetime.now().isoformat()
                )
                self.stats.record_decision(decision)

                # Display result
                print(self.display_result(gto_probs, was_correct, feedback))

                # Save stats periodically
                if self.stats.total_hands % 10 == 0:
                    self.stats.save(self.stats_path)

                input("\nPress Enter for next hand...")

        except KeyboardInterrupt:
            print("\n\nSession ended.")
            print(self.display_stats())
            self.stats.save(self.stats_path)
            print(f"\nStats saved to {self.stats_path}")

    def _parse_action(self, user_input: str, scenario: TrainerScenario) -> Optional[int]:
        """Parse user action input."""
        # Check shortcut
        if user_input in ACTION_SHORTCUTS:
            idx = ACTION_SHORTCUTS[user_input]
            if idx in scenario.legal_action_indices:
                return idx
            return None

        # Check number
        try:
            num = int(user_input)
            if 1 <= num <= len(scenario.legal_action_indices):
                return scenario.legal_action_indices[num - 1]
        except ValueError:
            pass

        # Check action name
        for i, name in enumerate(scenario.legal_actions):
            if name.lower().startswith(user_input):
                return scenario.legal_action_indices[i]

        return None

    def _handle_set_command(self, command: str):
        """Handle set commands."""
        parts = command.split()
        if len(parts) < 3:
            print("Usage: set [stack/pos/hand] [value]")
            return

        setting = parts[1]
        value = parts[2]

        if setting == 'stack':
            try:
                self.stack_size = float(value)
                print(f"Stack set to {self.stack_size}bb")
            except ValueError:
                print("Invalid stack size")

        elif setting == 'pos':
            try:
                pos = Position[value.upper()]
                self.positions_to_train = [pos]
                print(f"Training position: {pos.name}")
            except KeyError:
                print(f"Invalid position. Options: {[p.name for p in Position]}")

        elif setting == 'hand':
            if value.upper() == 'ALL':
                self.hand_filter = None
                print("Training all hands")
            else:
                self.hand_filter = [value.upper()]
                print(f"Training hand: {value.upper()}")


def run_trainer(checkpoint_path: str, num_players: int = 6):
    """Run the interactive trainer."""
    trainer = GTOTrainerInterface(
        checkpoint_path=checkpoint_path,
        num_players=num_players
    )
    trainer.run_interactive_session()
