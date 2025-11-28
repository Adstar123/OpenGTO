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

### Step 2: Neural Network Training (Completed)

We have implemented and trained a PyTorch neural network to learn GTO strategies:

#### Components:

1. **Data Preprocessing** ([src/preprocessing.py](src/preprocessing.py))
   - Feature encoding for hands, positions, and action history
   - Train/validation/test split
   - Normalization and padding

2. **PyTorch Dataset** ([src/dataset.py](src/dataset.py))
   - Custom dataset class for efficient data loading
   - Compatible with DataLoader for batching

3. **Neural Network Models** ([src/model.py](src/model.py))
   - **PreflopStrategyNet**: Simple feedforward network (256->128->64)
   - **PreflopStrategyNetV2**: Advanced model with embeddings and batch normalization
   - Both models output probability distributions over 7 actions

4. **Trainer** ([src/trainer.py](src/trainer.py))
   - KL divergence loss for probability distribution learning
   - Adam optimizer with learning rate scheduling
   - Early stopping based on validation loss
   - Model checkpointing

#### Training Results:

- **Test Accuracy**: 58-88% (varies by run due to limited data)
- **Mean KL Divergence**: 0.0004-0.001 (very low - close to GTO)
- **Model Size**: 45,703 parameters (simple model)

#### Train a Model:

```bash
python train_model.py --epochs 50 --batch-size 32 --model-type simple
```

Options:
- `--model-type`: 'simple' or 'advanced'
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--learning-rate`: Learning rate
- `--dropout`: Dropout rate for regularization
- `--early-stopping`: Patience for early stopping

### Next Steps

- **Step 3**: Create interactive CLI trainer
- **Step 4**: Build GUI application


