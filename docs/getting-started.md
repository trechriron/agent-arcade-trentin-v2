# Getting Started

Welcome to Agent Arcade! This guide will help you get started with training and competing with AI agents.

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv drl-env
source drl-env/bin/activate  # On Windows: drl-env\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt

# Install Atari ROMs
pip install "gymnasium[accept-rom-license,atari]"
```

## Available Games

Currently, Agent Arcade supports:

### Pong

- Classic paddle-based game
- Simple action space (UP/DOWN)
- Clear scoring system
- Great for learning RL basics

### Space Invaders

- Shoot-em-up arcade classic
- More complex action space (LEFT/RIGHT/FIRE)
- Progressive difficulty
- Multiple game modes

## Quick Start

### Training an Agent

```bash
# Train Pong agent
agent-arcade pong train --render

# Train Space Invaders agent
agent-arcade space-invaders train --render
```

### Evaluating an Agent

```bash
# Evaluate Pong agent
agent-arcade pong evaluate --model models/pong_dqn_1000000_steps.zip

# Evaluate Space Invaders agent
agent-arcade space-invaders evaluate --model models/space_invaders_dqn_1000000_steps.zip
```

### Staking and Competition

```bash
# Stake on Pong performance
agent-arcade pong stake --amount 10 --model models/my_pong_agent.zip --target-score 15

# Stake on Space Invaders performance
agent-arcade space-invaders stake --amount 10 --model models/my_space_invaders_agent.zip --target-score 1000
```

## Training Tips

### General Tips

- Start with default configurations
- Monitor training through TensorBoard
- Use video recording to track progress
- Save checkpoints regularly

### Game-Specific Tips

#### Pong

- Focus on paddle positioning
- Watch for ball trajectory patterns
- Start with shorter training runs

#### Space Invaders

- Start with difficulty 0
- Focus on survival in early training
- Use mode 0 for standard gameplay
- Balance shooting and dodging strategies

## Next Steps

1. Read the [Training Configuration Guide](training-config.md) for detailed parameter tuning
2. Explore the [Competition Guide](competition.md) for staking and rewards
3. Check [Understanding Metrics](understanding-metrics.md) for performance analysis
4. See [NEAR Integration](near-integration.md) for blockchain features

## Need Help?

- Submit issues on GitHub
