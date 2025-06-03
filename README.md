# OpenGTO - Neural Network GTO Poker Trainer

**Version 0.2.0** - Fully refactored with modular architecture

A machine learning-based poker training system that learns Game Theory Optimal (GTO) strategies for Texas Hold'em. Currently focused on preflop play with plans to expand to full game coverage.

## Current State

### What's Working
- **Complete modular architecture** - Clean separation of concerns
- **Preflop neural network** - ~90% accuracy on GTO decisions
- **CLI interface** - Professional command-line tool
- **Configuration system** - YAML-based configs
- **Testing suite** - Comprehensive unit tests
- **Performance monitoring** - Built-in benchmarking
- **Data validation** - Automatic quality checks

### Prerequisites
- Python 3.8+ (tested on 3.13)
- PyTorch 2.0+ with CUDA support (for GPU training)
- NVIDIA GPU recommended (tested on RTX 4080)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/OpenGTO.git
cd OpenGTO

# Install dependencies
pip install -r requirements.txt

# Optional: Install as package
pip install -e .
```

### Basic Usage

#### 1. Train a Model
```bash
# Using CLI
python -m poker_gto.cli train

# With custom config
python -m poker_gto.cli train --config configs/training_config.yaml

# Using script
python scripts/train.py
```

#### 2. Test a Model
```bash
# Test latest model
python -m poker_gto.cli test

# Interactive testing
python -m poker_gto.cli test --interactive

# Test specific model
python -m poker_gto.cli test models/your_model.pth
```

#### 3. Benchmark Performance
```bash
python -m poker_gto.cli benchmark models/your_model.pth
```

## Project Structure

```
OpenGTO/
├── poker_gto/              # Main package
│   ├── core/              # Game logic (poker rules, game state)
│   ├── ml/                # Machine learning components
│   │   ├── models/        # Neural network architectures
│   │   ├── features/      # Feature extraction
│   │   ├── data/          # Data generation
│   │   └── training/      # Training utilities
│   ├── config/            # Configuration management
│   ├── utils/             # Utilities (logging, validation, testing)
│   └── cli.py             # Command-line interface
├── configs/               # YAML configuration files
├── scripts/               # Entry point scripts
├── tests/                 # Unit tests
├── models/                # Saved models
├── logs/                  # Training logs
└── data/                  # Data storage (future use)
```

## Technical Architecture

### Neural Network
- **Architecture**: Feed-forward network (128→64→32→3)
- **Input**: 20 features (position, cards, game context)
- **Output**: Action probabilities (fold/call/raise)
- **Training**: Balanced dataset of 15,000 scenarios

### Key Components
1. **PreflopGTOModel**: Main neural network model
2. **PreflopFeatureExtractor**: Converts game state to features
3. **PreflopScenarioGenerator**: Creates balanced training data
4. **PreflopTrainer**: Handles training loop and validation
5. **ModelFactory**: Creates models with consistent interface
6. **ConfigManager**: Handles YAML configurations

### Design Patterns
- **Factory Pattern**: For model creation
- **Abstract Base Classes**: Define interfaces
- **Dependency Injection**: For testing and flexibility
- **Single Responsibility**: Each module has one job

## Config

Training configuration example (`configs/training_config.yaml`):
```yaml
# Data generation
num_scenarios: 15000
player_counts: [6]
stack_sizes: [100.0]

# Training parameters
epochs: 100
batch_size: 128
learning_rate: 0.001
patience: 20

# Model architecture
input_size: 20
hidden_sizes: [128, 64, 32]
dropout_rate: 0.3
```

## Performance

- **Training Time**: ~3 seconds on RTX 4080
- **Inference Speed**: <1ms per decision
- **Model Size**: ~100KB
- **Memory Usage**: <500MB during training

## Roadmap

### Phase 1: Preflop (mostly complete)
- Neural network for preflop decisions
- 90% accuracy on GTO strategy
- CLI interface and testing suite

### Phase 2: Simple Postflop (Next)
- Flop texture analysis
- Continuation betting strategies
- Board representation features

### Phase 3: Full Game (Future)
- Turn and river play
- Multi-street planning
- Opponent modeling

### Phase 4: Desktop App (Future)
- Electron + React frontend
- Real-time training interface
- Hand history analysis

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run formatter
black poker_gto/

# Run linter
flake8 poker_gto/

# Run type checker
mypy poker_gto/
```
