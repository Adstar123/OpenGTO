# OpenGTO - Neural Network GTO Poker Trainer

A machine learning-based poker training system that learns Game Theory Optimal (GTO) strategies for Texas Hold'em preflop play. (future iterations will have GTO past preflop).

## 🎯 Project Overview

OpenGTO uses deep learning to master preflop poker decisions. The system trains neural networks on thousands of poker scenarios, learning optimal play across all positions, stack sizes, and betting situations.

**Current Status: Phase 1 - Preflop Training**
- ✅ Complete preflop decision engine
- ✅ 15,000 balanced training scenarios
- ✅ Stack depth and bet sizing considerations  
- ✅ Position-aware GTO strategy
- ✅ ~90% accuracy achieved on preflop decisions
- 🎯 Expandable to postflop play (future phases)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ with CUDA support (for GPU training)
- NVIDIA GPU recommended (tested on RTX 4080)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/OpenGTO.git
cd OpenGTO

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 1. Training a Model

Train a GTO model on 15,000 balanced scenarios:

```bash
python scripts/train_robust.py
```

This will:
- Generate perfectly balanced poker scenarios (equal fold/call/raise distribution)
- Train for up to 80 epochs with early stopping
- Save the trained model to `models/` directory
- Log training progress to `logs/`

Training typically completes in 1-3 seconds on an RTX 4080.

### 2. Testing Your Model

Test the trained model comprehensively:

```bash
python scripts/test_simple_model.py
```

Features:
- Tests diverse poker scenarios
- Interactive testing mode for custom situations
- Validates model is making balanced decisions

## 📊 Technical Details

### Neural Network Architecture
- **Input**: 20 comprehensive features including:
  - Position encoding (6 features for 6-max games)
  - Hand features (pocket pair, suited, strength, high card)
  - Context features (facing raise, pot size, pot odds, etc.)
- **Architecture**: 128→64→32→3 neurons with ReLU activation and Dropout(0.3)
- **Output**: Fold/Call/Raise probabilities
- **Training**: AdamW optimiser, CrossEntropyLoss with class balancing

### Key Features
- **Position-Aware**: Different strategies for UTG, MP, CO, BTN, SB, BB
- **Context-Sensitive**: Adapts to facing raises vs opening pots
- **Balanced Data**: Enforced equal distribution of fold/call/raise in training
- **Simplified GTO Logic**: Based on position and hand strength heuristics

### Training Process
1. **Data Generation**: Creates scenarios with realistic hand distributions and action contexts
2. **Balancing**: Ensures exactly 5,000 scenarios each for fold/call/raise decisions
3. **Validation**: 80/20 train/validation split with per-action accuracy tracking
4. **Early Stopping**: Prevents overfitting with patience of 20 epochs

## 🎮 Using the Trained Model

### Interactive Testing
The test script includes an interactive mode where you can input:
- Position (UTG/MP/CO/BTN/SB/BB)
- Hand (e.g., AA, AKs, 72o)
- Situation (opening or facing a raise)

The model will recommend an action based on its training.


## 🎯 Roadmap

### ✅ Phase 1: Preflop Mastery (50% done)
- Preflop decision engine with 90% accuracy
- Balanced training data generation
- Position and context awareness
- Interactive testing interface

### 🔜 Phase 2: Simple Postflop (Future)
- Add flop play for heads-up pots
- Board texture analysis
- Continuation betting strategies
- Transfer learning from preflop model

### 🔮 Phase 3: Multi-Street Play (Future)
- Turn and river decisions
- Multi-way pot dynamics
- Advanced betting strategies
- Tournament considerations (ICM)

### 🎨 Phase 4: Desktop Application (Future)
- Electron + React frontend
- Real-time training interface
- Hand history analysis
- Progress tracking and visualisation
