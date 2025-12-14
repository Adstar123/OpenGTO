#!/usr/bin/env python3
"""
Improved GTO Training Script with GPU Support.

Key improvements over default training:
1. Uses GPU (CUDA) for faster training
2. Lower exploration rate for more stable learning
3. Larger network for 6-max complexity
4. Better curriculum progression
5. More training iterations per stage

Usage:
    python train_improved.py                    # Start fresh
    python train_improved.py --resume <path>   # Resume from checkpoint
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.trainer import GTOTrainer, TrainingConfig
from src.curriculum import CurriculumStage, CurriculumScheduler


def create_improved_curriculum():
    """
    Improved curriculum with more iterations and better progression.
    """
    stages = [
        # Stage 1: Heads-up push/fold - learn basic hand values
        CurriculumStage(
            name="HU_PushFold",
            num_players=2,
            min_stack=8.0,
            max_stack=15.0,
            iterations=1000,
            traversals_per_iter=1000,
            description="Heads-up push/fold (8-15bb) - Learn basic hand values"
        ),

        # Stage 2: Heads-up short stacks
        CurriculumStage(
            name="HU_Short",
            num_players=2,
            min_stack=15.0,
            max_stack=30.0,
            iterations=1000,
            traversals_per_iter=1000,
            description="Heads-up short (15-30bb) - Learn raise sizing"
        ),

        # Stage 3: Heads-up medium stacks
        CurriculumStage(
            name="HU_Medium",
            num_players=2,
            min_stack=30.0,
            max_stack=60.0,
            iterations=1000,
            traversals_per_iter=1000,
            description="Heads-up medium (30-60bb) - Positional play"
        ),

        # Stage 4: Heads-up deep
        CurriculumStage(
            name="HU_Deep",
            num_players=2,
            min_stack=60.0,
            max_stack=100.0,
            iterations=1000,
            traversals_per_iter=1000,
            description="Heads-up deep (60-100bb)"
        ),

        # Stage 5: 3-way - introduce multiway dynamics
        CurriculumStage(
            name="3Way_Mixed",
            num_players=3,
            min_stack=20.0,
            max_stack=100.0,
            iterations=2000,
            traversals_per_iter=1000,
            description="3-way (20-100bb) - Multiway dynamics"
        ),

        # Stage 6: 6-max short stacks
        CurriculumStage(
            name="6Max_Short",
            num_players=6,
            min_stack=20.0,
            max_stack=50.0,
            iterations=3000,
            traversals_per_iter=1000,
            description="6-max short (20-50bb) - Full table dynamics"
        ),

        # Stage 7: 6-max deep - the main game
        CurriculumStage(
            name="6Max_Deep",
            num_players=6,
            min_stack=50.0,
            max_stack=100.0,
            iterations=5000,
            traversals_per_iter=1000,
            description="6-max deep (50-100bb) - Standard cash game"
        ),

        # Stage 8: 6-max full range - include very deep stacks
        CurriculumStage(
            name="6Max_Full",
            num_players=6,
            min_stack=80.0,
            max_stack=150.0,
            iterations=3000,
            traversals_per_iter=1000,
            description="6-max deep (80-150bb) - Deep stack play"
        ),
    ]

    return CurriculumScheduler(stages=stages)


def main():
    parser = argparse.ArgumentParser(description="Improved GTO Training with GPU")
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--iterations', type=int, default=None, help='Override total iterations')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, or cpu')
    args = parser.parse_args()

    # Detect device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("IMPROVED GTO POKER TRAINER")
    print("=" * 60)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Improved training configuration
    config = TrainingConfig(
        # Use curriculum
        use_curriculum=True,
        quick_mode=False,

        # Larger network for 6-max complexity
        hidden_sizes=(512, 512, 256, 128),

        # Training parameters - key improvements
        batch_size=4096,           # Larger batches for GPU efficiency
        train_steps_per_iter=200,  # More training per iteration
        traversals_per_iter=1000,  # More game simulations
        exploration=0.3,           # LOWER exploration for more stable learning

        # Larger memory for diverse training
        regret_buffer_size=5_000_000,
        strategy_buffer_size=5_000_000,

        # Checkpointing
        save_every=500,
        eval_every=250,
        print_every=50,

        # GPU
        device=device,

        # Paths
        checkpoint_dir='checkpoints_improved',
    )

    # Create trainer
    trainer = GTOTrainer(config)

    # Set the improved curriculum BEFORE loading checkpoint
    trainer.curriculum = create_improved_curriculum()

    if args.resume:
        trainer.load_checkpoint(args.resume)

        # Re-apply the improved curriculum (load_checkpoint may have used default)
        # Keep the iteration count but recalculate which stage we should be in
        saved_iteration = trainer.iteration
        trainer.curriculum = create_improved_curriculum()

        # Calculate which stage we should be in based on iteration count
        cumulative = 0
        for idx, stage in enumerate(trainer.curriculum.stages):
            if cumulative + stage.iterations > saved_iteration:
                trainer.curriculum.current_stage_idx = idx
                trainer.curriculum.stage_iterations = saved_iteration - cumulative
                break
            cumulative += stage.iterations
        else:
            # Past all stages
            trainer.curriculum.current_stage_idx = len(trainer.curriculum.stages) - 1
            trainer.curriculum.stage_iterations = trainer.curriculum.stages[-1].iterations

        trainer.curriculum.total_iterations = saved_iteration
        trainer._update_engine_for_curriculum()

        print(f"  Applied improved curriculum: {trainer.curriculum.get_summary()}")

    # Calculate total iterations (from curriculum)
    total_curriculum_iters = sum(s.iterations for s in trainer.curriculum.stages)

    if args.iterations:
        total_iters = args.iterations
    else:
        # Train for the full curriculum length, regardless of current position
        total_iters = total_curriculum_iters

    print(f"Total curriculum iterations: {total_curriculum_iters}")
    if args.resume:
        remaining = total_iters - trainer.iteration
        print(f"Already completed: {trainer.iteration}")
        print(f"Remaining iterations: {remaining}")
    print(f"Curriculum stages: {len(trainer.curriculum.stages)}")
    print()

    for i, stage in enumerate(trainer.curriculum.stages):
        marker = " <-- current" if i == trainer.curriculum.current_stage_idx else ""
        print(f"  Stage {i+1}: {stage.name} ({stage.iterations} iters){marker}")
    print()

    # Start training
    trainer.train(num_iterations=total_iters)

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {config.checkpoint_dir}")


if __name__ == '__main__':
    main()
