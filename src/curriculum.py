"""
Curriculum Learning for GTO Poker Training.

Starts with simple scenarios and progressively increases complexity:
1. Heads-up, short stacks (push/fold)
2. Heads-up, medium stacks
3. Heads-up, deep stacks
4. 3-player, various stacks
5. 6-player, various stacks

This helps the neural network learn fundamental concepts before
tackling more complex multi-way scenarios.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

from .poker_engine import PreflopPokerEngine


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    name: str
    num_players: int
    min_stack: float  # Minimum stack in BB
    max_stack: float  # Maximum stack in BB
    iterations: int  # Training iterations for this stage
    traversals_per_iter: int
    description: str

    def get_random_stack(self) -> float:
        """Get a random stack size within the range."""
        return np.random.uniform(self.min_stack, self.max_stack)

    def get_stack_list(self) -> List[float]:
        """Get stack sizes for all players."""
        return [self.get_random_stack() for _ in range(self.num_players)]


# Default curriculum stages
DEFAULT_CURRICULUM = [
    # Stage 1: Heads-up push/fold (very short stacks)
    CurriculumStage(
        name="HU_PushFold",
        num_players=2,
        min_stack=5.0,
        max_stack=15.0,
        iterations=500,
        traversals_per_iter=500,
        description="Heads-up push/fold with 5-15bb stacks"
    ),

    # Stage 2: Heads-up short stacks
    CurriculumStage(
        name="HU_Short",
        num_players=2,
        min_stack=15.0,
        max_stack=30.0,
        iterations=500,
        traversals_per_iter=500,
        description="Heads-up with 15-30bb stacks"
    ),

    # Stage 3: Heads-up medium stacks
    CurriculumStage(
        name="HU_Medium",
        num_players=2,
        min_stack=30.0,
        max_stack=60.0,
        iterations=500,
        traversals_per_iter=500,
        description="Heads-up with 30-60bb stacks"
    ),

    # Stage 4: Heads-up deep stacks
    CurriculumStage(
        name="HU_Deep",
        num_players=2,
        min_stack=60.0,
        max_stack=100.0,
        iterations=500,
        traversals_per_iter=500,
        description="Heads-up with 60-100bb stacks"
    ),

    # Stage 5: 3-way short stacks
    CurriculumStage(
        name="3Way_Short",
        num_players=3,
        min_stack=15.0,
        max_stack=40.0,
        iterations=500,
        traversals_per_iter=500,
        description="3-way with 15-40bb stacks"
    ),

    # Stage 6: 3-way deep stacks
    CurriculumStage(
        name="3Way_Deep",
        num_players=3,
        min_stack=40.0,
        max_stack=100.0,
        iterations=500,
        traversals_per_iter=500,
        description="3-way with 40-100bb stacks"
    ),

    # Stage 7: 6-max short stacks
    CurriculumStage(
        name="6Max_Short",
        num_players=6,
        min_stack=15.0,
        max_stack=40.0,
        iterations=1000,
        traversals_per_iter=500,
        description="6-max with 15-40bb stacks"
    ),

    # Stage 8: 6-max deep stacks
    CurriculumStage(
        name="6Max_Deep",
        num_players=6,
        min_stack=40.0,
        max_stack=100.0,
        iterations=2000,
        traversals_per_iter=500,
        description="6-max with 40-100bb stacks"
    ),
]


# Quick curriculum for testing
QUICK_CURRICULUM = [
    CurriculumStage(
        name="HU_PushFold",
        num_players=2,
        min_stack=8.0,
        max_stack=12.0,
        iterations=50,
        traversals_per_iter=100,
        description="Quick HU push/fold test"
    ),
    CurriculumStage(
        name="HU_Medium",
        num_players=2,
        min_stack=20.0,
        max_stack=40.0,
        iterations=50,
        traversals_per_iter=100,
        description="Quick HU medium test"
    ),
]


class CurriculumScheduler:
    """
    Manages curriculum progression during training.
    """

    def __init__(
        self,
        stages: List[CurriculumStage] = None,
        auto_advance: bool = True,
        convergence_threshold: float = 0.01
    ):
        self.stages = stages or DEFAULT_CURRICULUM
        self.current_stage_idx = 0
        self.auto_advance = auto_advance
        self.convergence_threshold = convergence_threshold

        # Progress tracking
        self.stage_iterations = 0
        self.total_iterations = 0
        self.stage_history: List[Dict] = []

    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]

    @property
    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return self.current_stage_idx >= len(self.stages)

    def get_engine(self) -> PreflopPokerEngine:
        """Get poker engine configured for current stage."""
        stage = self.current_stage
        return PreflopPokerEngine(
            num_players=stage.num_players,
            starting_stack=stage.get_random_stack()
        )

    def get_engine_with_stacks(self, stacks: List[float]) -> PreflopPokerEngine:
        """Get poker engine with specific stacks."""
        stage = self.current_stage
        engine = PreflopPokerEngine(
            num_players=stage.num_players,
            starting_stack=stacks[0] if stacks else stage.get_random_stack()
        )
        return engine

    def step(self, metrics: Optional[Dict] = None) -> bool:
        """
        Advance one iteration.

        Args:
            metrics: Optional metrics from the iteration (loss, utility, etc.)

        Returns:
            True if advanced to next stage
        """
        self.stage_iterations += 1
        self.total_iterations += 1

        # Check if should advance to next stage
        if self.stage_iterations >= self.current_stage.iterations:
            return self.advance_stage(metrics)

        return False

    def advance_stage(self, metrics: Optional[Dict] = None) -> bool:
        """
        Advance to the next curriculum stage.

        Returns:
            True if successfully advanced, False if curriculum complete
        """
        # Record stage completion
        self.stage_history.append({
            'stage': self.current_stage.name,
            'iterations': self.stage_iterations,
            'metrics': metrics
        })

        # Reset stage counter
        self.stage_iterations = 0
        self.current_stage_idx += 1

        if self.is_complete:
            return False

        print(f"\n{'='*50}")
        print(f"Advancing to stage: {self.current_stage.name}")
        print(f"  {self.current_stage.description}")
        print(f"  Players: {self.current_stage.num_players}")
        print(f"  Stacks: {self.current_stage.min_stack}-{self.current_stage.max_stack}bb")
        print(f"{'='*50}\n")

        return True

    def get_progress(self) -> Dict:
        """Get curriculum progress information."""
        total_planned_iters = sum(s.iterations for s in self.stages)
        completed_iters = sum(s.iterations for s in self.stages[:self.current_stage_idx])
        completed_iters += self.stage_iterations

        return {
            'current_stage': self.current_stage.name if not self.is_complete else 'Complete',
            'stage_num': self.current_stage_idx + 1,
            'total_stages': len(self.stages),
            'stage_progress': self.stage_iterations / self.current_stage.iterations if not self.is_complete else 1.0,
            'total_progress': completed_iters / total_planned_iters,
            'total_iterations': self.total_iterations
        }

    def get_summary(self) -> str:
        """Get a summary string of progress."""
        progress = self.get_progress()
        return (
            f"Stage {progress['stage_num']}/{progress['total_stages']} "
            f"({progress['current_stage']}): "
            f"{progress['stage_progress']*100:.1f}% | "
            f"Total: {progress['total_progress']*100:.1f}%"
        )


class StackVariationGenerator:
    """
    Generates varied stack configurations for training diversity.
    """

    def __init__(self, num_players: int, min_stack: float, max_stack: float):
        self.num_players = num_players
        self.min_stack = min_stack
        self.max_stack = max_stack

    def uniform_stacks(self) -> List[float]:
        """All players have the same random stack."""
        stack = np.random.uniform(self.min_stack, self.max_stack)
        return [stack] * self.num_players

    def varied_stacks(self) -> List[float]:
        """Each player has a different random stack."""
        return [
            np.random.uniform(self.min_stack, self.max_stack)
            for _ in range(self.num_players)
        ]

    def one_short_stack(self) -> List[float]:
        """One player has a short stack, others have deep stacks."""
        stacks = [self.max_stack * 0.9] * self.num_players
        short_idx = np.random.randint(0, self.num_players)
        stacks[short_idx] = self.min_stack * 1.1
        return stacks

    def tournament_style(self) -> List[float]:
        """Varied stacks simulating tournament play."""
        # Average stack
        avg = (self.min_stack + self.max_stack) / 2

        stacks = []
        for _ in range(self.num_players):
            # Log-normal distribution for more realistic tournament stacks
            stack = np.random.lognormal(np.log(avg), 0.5)
            stack = np.clip(stack, self.min_stack, self.max_stack)
            stacks.append(stack)

        return stacks

    def random_configuration(self) -> List[float]:
        """Randomly choose a configuration type."""
        configs = [
            self.uniform_stacks,
            self.varied_stacks,
            self.one_short_stack,
            self.tournament_style
        ]
        return np.random.choice(configs)()


def create_quick_curriculum() -> CurriculumScheduler:
    """Create a quick curriculum for testing."""
    return CurriculumScheduler(stages=QUICK_CURRICULUM)


def create_full_curriculum() -> CurriculumScheduler:
    """Create the full training curriculum."""
    return CurriculumScheduler(stages=DEFAULT_CURRICULUM)


def create_custom_curriculum(
    stages: List[Tuple[int, float, float, int]]
) -> CurriculumScheduler:
    """
    Create a custom curriculum.

    Args:
        stages: List of (num_players, min_stack, max_stack, iterations) tuples
    """
    custom_stages = []
    for i, (num_players, min_stack, max_stack, iterations) in enumerate(stages):
        custom_stages.append(CurriculumStage(
            name=f"Stage_{i+1}",
            num_players=num_players,
            min_stack=min_stack,
            max_stack=max_stack,
            iterations=iterations,
            traversals_per_iter=500,
            description=f"{num_players}p {min_stack}-{max_stack}bb"
        ))

    return CurriculumScheduler(stages=custom_stages)
