# Getting Started

Welcome to Agent Arcade! This guide will help you get started with training and competing with AI agents.

## Quick Install

```bash
# One-line install (recommended)
curl -sSL https://raw.githubusercontent.com/jbarnes850/agent-arcade/main/install.sh | bash
```

## Manual Installation

```bash
# Clone repository
git clone https://github.com/jbarnes850/agent-arcade.git
cd agent-arcade

# Run install script
./install.sh
```

## Verify Installation

```bash
# Check CLI is working
agent-arcade --version

# List available games
agent-arcade list-games
```

## Training an Agent

```bash
# Train with visualization
agent-arcade train pong --render

# Train with custom config
agent-arcade train space-invaders --config configs/space_invaders_sb3_config.yaml
```

## Evaluating an Agent

```bash
# Basic evaluation
agent-arcade evaluate pong --model models/pong_final.zip --episodes 10

# Record evaluation video
agent-arcade evaluate space-invaders --model models/space_invaders_final.zip --episodes 5
```

## NEAR Integration

```bash
# Login to NEAR wallet
agent-arcade login

# Stake on performance
agent-arcade stake pong --model models/pong_final.zip --amount 10 --target-score 15
```

## Next Steps

1. [Training Configuration Guide](training-config.md)
2. [Competition Guide](competition.md)
3. [Understanding Metrics](understanding-metrics.md)
4. [NEAR Integration](near-integration.md)

## Need Help?

- Submit issues on GitHub: [https://github.com/jbarnes850/agent-arcade/issues]
