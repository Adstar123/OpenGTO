## GTO Preflop Poker Trainer

A neural network-based poker trainer for learning Game Theory Optimal (GTO) preflop strategies in Texas Hold'em.

### Project Structure

```
OpenGTO/
├── src/
│   ├── card.py              # Card and hand representations
│   ├── equity.py            # Preflop equity calculator
│   ├── game_tree.py         # Game tree and state management
│   ├── cfr_solver.py        # CFR algorithm implementation
│   └── data_generator.py    # Training data extraction
├── data/                    # Generated training data
├── models/                  # Trained neural network models
├── tests/                   # Unit tests
├── train_cfr.py            # CFR training script
├── requirements.txt        # Python dependencies
└── plan.md                 # Development plan
```

### Step 1: Data Generation (Completed)

We have implemented a custom Counterfactual Regret Minimization (CFR) solver for preflop poker:

#### Components:

1. **Card Representation** ([src/card.py](src/card.py))
   - Immutable Card and Hand classes
   - Hand parsing and string representation
   - Support for suited/offsuit notation (e.g., "AKs", "AKo", "QQ")

2. **Equity Calculator** ([src/equity.py](src/equity.py))
   - Preflop equity estimation between hands
   - Cached lookup tables for performance
   - Hand strength approximation

3. **Game Tree** ([src/game_tree.py](src/game_tree.py))
   - Preflop betting sequences
   - Legal action generation
   - State transitions
   - Showdown evaluation

4. **CFR Solver** ([src/cfr_solver.py](src/cfr_solver.py))
   - Vanilla CFR algorithm
   - Regret matching
   - Strategy averaging for Nash equilibrium approximation
   - Information set tracking

5. **Data Generator** ([src/data_generator.py](src/data_generator.py))
   - Extracts strategies from trained CFR solver
   - Exports to CSV and JSON formats
   - Generates training examples with action probability distributions

### Usage

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Train CFR Solver and Generate Data

```bash
python train_cfr.py --iterations 10000 --sample-hands 100
```

Options:
- `--iterations`: Number of CFR iterations (default: 1000)
- `--sample-hands`: Random hand matchups per iteration (default: 50)
- `--stack-size`: Stack size in big blinds (default: 100.0)
- `--output-csv`: Output CSV file path
- `--output-json`: Output JSON file path
- `--show-samples`: Number of sample strategies to display

#### Quick Test Run

```bash
python train_cfr.py --iterations 100 --sample-hands 20
```

### Training Data Format

Each training example contains:

| Field | Description |
|-------|-------------|
| `position` | Player position ("btn" or "bb") |
| `hand` | Hand notation (e.g., "AKs", "QQ", "72o") |
| `action_history` | Sequence of previous actions |
| `stack_bb` | Stack size in big blinds |
| `prob_fold` | Probability of folding |
| `prob_check` | Probability of checking |
| `prob_call` | Probability of calling |
| `prob_raise_2bb` | Probability of raising to 2BB |
| `prob_raise_3bb` | Probability of raising to 3BB |
| `prob_raise_4bb` | Probability of raising to 4BB |
| `prob_all_in` | Probability of going all-in |

### Next Steps

- **Step 2**: Build and train PyTorch neural network
- **Step 3**: Create interactive CLI trainer
- **Step 4**: Build GUI application

### Implementation Notes

- Code follows PEP 8 style guidelines
- Type hints used throughout
- Modular design with clear separation of concerns
- No emojis in code or output
- Well-documented with docstrings
