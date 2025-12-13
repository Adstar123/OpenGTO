OpenGTO

1. to train: python train_improved.py
2. to play: python play.py
3. to play based off a checkpoint: python play.py checkpoints_improved/gto_trainer_final.pt
4. to query a specific hand: python trainer_cli.py query checkpoints/gto_trainer_final.pt --hand AKs --position BTN

How to play:

When you run python play.py, you'll see:

STACK: 100bb | POSITION: BTN


Your hand: [Ac] [Kc] (AKs)

Action:
  UTG folds
  CO folds

Pot: 1.5bb | To call: 1.0bb

Your options:
  1. Fold (f)
  2. Call (c)
  3. Raise (r)
  4. All-In (a)

Your action: _
Commands:
Type a number (1-4) or shortcut letter (f/x/c/b/r/a) to make your action
stats - View your accuracy statistics
set stack 50 - Change stack size to 50bb
set pos BTN - Only train from Button
set hand AKs - Only train AKs hands
quit - Exit and save stats
After each decision, you'll see GTO feedback:


CORRECT!

Good! Raise is acceptable. GTO prefers All-In (79%) but Raise (18%) is fine.

GTO Strategy:
  All-In   [#######################-------]  79.0%
  Raise    [#####-------------------------]  17.5%
  Call     [------------------------------]   3.1%
  Fold     [------------------------------]   0.4%


Your statistics are automatically saved to user_stats.json and tracked across sessions!
