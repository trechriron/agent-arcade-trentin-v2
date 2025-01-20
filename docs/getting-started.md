# Getting Started with Agent Arcade

Welcome to Agent Arcade! This guide will help you get started with training and evaluating AI agents for classic arcade games on the NEAR blockchain.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- NEAR CLI (for blockchain interactions)

## Installation

### **Clone the Repository**

```bash
git clone https://github.com/your-username/agent-arcade.git
cd agent-arcade
```

### **Set Up Python Environment**

```bash
# Create and activate virtual environment
python -m venv drl-env
source drl-env/bin/activate  # On Windows: drl-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Atari ROMs
pip install "gymnasium[accept-rom-license,atari]"
```

### **Install NEAR CLI**

```bash
npm install -g near-cli
```

## Quick Start

### Training Your First Agent

```bash
# Train a new Pong agent with visualization
python scripts/train_pong.py --render

# Train without visualization (faster)
python scripts/train_pong.py
```

### Monitor Training Progress

```bash
# Start TensorBoard
tensorboard --logdir ./tensorboard

# Access at http://localhost:6006
```

### Using Pre-trained Models

```bash
# Evaluate a pre-trained model
python scripts/evaluate_pong.py --model models/pong_dqn_1000000_steps.zip
```

## What's Next?

- Learn about [NEAR Integration](near-integration.md)
- Explore [Training Configuration](training-config.md)
- Join the [Competition](competition.md)

## Need Help?

- Submit issues on GitHub
