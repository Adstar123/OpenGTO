# OpenGTO - Neural Network GTO Poker Trainer

A machine learning-based poker training system that learns Game Theory Optimal (GTO) strategies for Texas Hold'em preflop play (for now) (hoping to exapnd to post flop).

## 🎯 Project Overview

OpenGTO uses deep learning to master preflop poker decisions with near-perfect accuracy. The system trains neural networks on hundreds of thousands of poker scenarios, learning optimal play across all positions, stack sizes, and betting situations.

**Current Status: Phase 1 - Preflop Mastery**
- ✅ Complete preflop decision engine
- ✅ 500k+ training scenarios for RTX 4080
- ✅ Stack depth and bet sizing considerations  
- ✅ Position-aware GTO strategy
- 🎯 Target: 98.5%+ accuracy on preflop decisions
- Acc currently at around 90% on robust model.

## 🚀 Quick Start

### 1. Training (RTX 4080 Optimised)

Train a massive GTO model on 500,000 scenarios:

```bash
python scripts/train_robust.py
```

This will:
- Generate balanced poker scenarios
- Train epochs 
- Save the trained model automatically

### 2. Testing Your Model

Test the trained model comprehensively:

```bash
python scripts/test_simple_model.py
```

Features:
- Tests critical GTO scenarios (premium hands, stack sizes, position play)
- Interactive testing mode


## 📊 System Architecture
### Neural Network
- **Input**: 32 comprehensive features including position, hand strength, action context, and stack dynamics
- **Architecture**: 512→256→128→64→32→3 neurons with BatchNorm and Dropout
- **Output**: Fold/Call/Raise probabilities with GTO bet sizing
- **Training**: 500k scenarios, class-balanced, AdamW optimiser

### Key Features
- **Position-Aware**: Different strategies for UTG, MP, CO, BTN, SB, BB
- **Stack-Sensitive**: Adapts play for 20BB short stacks to 200BB deep stacks  
- **Action-Contextual**: Considers facing raises, 3-bets, number of callers
- **GTO Bet Sizing**: Proper 2.2x-3.5x raise sizes based on situation

### Training Data
Each scenario includes:
- Position and hole cards
- Stack sizes in big blinds
- Previous betting action
- Pot odds and implied odds
- Optimal GTO decision with reasoning

## 🧠 How It Works

### 1. Data Generation
The system generates realistic poker scenarios by:
- Creating valid 6-max game states
- Dealing random hole cards to all positions
- Simulating realistic betting sequences
- Applying GTO decision-making logic
- Balancing fold/call/raise distributions

### 2. Neural Network Training
- **Input Processing**: 32 features covering all GTO-relevant factors
- **Architecture**: Deep network optimized for poker decision-making
- **Loss Function**: Weighted cross-entropy for balanced learning
- **Optimization**: AdamW with cosine annealing for stable convergence
- **Validation**: Comprehensive testing on unseen scenarios

### 3. GTO Decision Making
The model considers:
- **Position Strength**: UTG tight, BTN loose
- **Stack Depth**: Short stack push/fold, deep stack postflop
- **Action Context**: First to act vs facing raises/3-bets
- **Pot Odds**: Mathematical calling requirements
- **Hand Strength**: Precise evaluation including suitedness, connectivity


## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- scikit-learn, numpy

### Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn numpy
```

## 🎯 Roadmap

### ✅ Phase 1: Preflop Mastery (Current)
- Complete preflop GTO engine
- 500k+ training scenarios
- 98.5%+ accuracy target
- All stack depths (20BB-200BB)

### 🔜 Phase 2: Postflop Foundation (Next)
- Add flop play for heads-up pots
- Board texture analysis
- Continuation betting strategies
- Hand reading improvements

### 🔮 Phase 3: Complete Postflop (Future)
- Turn and river play
- Multi-way pots
- Advanced betting strategies
- Tournament considerations (ICM)

### 🎨 Phase 4: User Interface (Future)
- Electron desktop application
- Interactive training modes
- Hand history analysis
- Progress tracking