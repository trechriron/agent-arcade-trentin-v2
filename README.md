# Agent Arcade: AI Game Agents on NEAR

A platform for training and competing with AI agents in classic arcade games using Stable Baselines3, Gymnasium, and the Arcade Learning Environment with optional staking powered by NEAR.

## 🎬 Demo: Trained Agents in Action

<p align="center">
    <img src="videos/Atari%20Environments%20Hands-on.gif" width="400" alt="Atari Environments Demo">
</p>

> Our agents learn to play classic Atari games from scratch through deep reinforcement learning. Train your own agents and compete for rewards!

## 🧠 Core Concepts

### Deep Q-Learning

Our agents use Deep Q-Learning (DQN), a reinforcement learning algorithm that learns to play games by:

- Observing game frames as input (what the agent "sees")
- Learning which actions lead to higher rewards through trial and error
- Using a neural network to approximate the optimal action-value function
- Storing and learning from past experiences (replay buffer)

### Training Process

1. **Exploration Phase**: Agent tries random actions to discover the game mechanics
2. **Experience Collection**: Stores (state, action, reward, next_state) in memory
3. **Learning Phase**: Updates its strategy by learning from past experiences
4. **Exploitation**: Gradually shifts from random actions to learned optimal actions

### Key Components

- **Environment**: Atari games (via Gymnasium/ALE) - provides game state and handles actions
- **Agent**: DQN with custom CNN - processes game frames and decides actions
- **Memory**: Replay buffer - stores experiences for learning
- **Training Loop**: Alternates between playing and learning from experiences

## 🎮 Current Games

- **Pong**: Classic paddle vs paddle game (recommended for beginners)
- **Space Invaders**: Defend Earth from alien invasion
- **River Raid**: Control a jet, manage fuel, and destroy enemies

> **Interested in adding a new game? See the [Adding New Games](docs/adding-games.md) guide.**

## 🚀 Quick Start

### Prerequisites

Core Requirements:
- **Python**: Version 3.8 - 3.12 (3.13 not yet supported)
- **Operating System**: Linux, macOS, or WSL2 on Windows
- **Storage**: At least 2GB free space
- **Memory**: At least 4GB RAM recommended

Optional Requirements (for staking):
- **Node.js & npm**: Required for NEAR CLI (v14 or higher)
- **NEAR Account**: Required for staking and competitions
- **GPU**: Optional for faster training

### Installation

```bash
# Clone the repository
git clone https://github.com/jbarnes850/agent-arcade.git
cd agent-arcade

# Run the installation script
chmod +x ./install.sh
./install.sh

# Optional: Install NEAR integration for staking
pip install -e ".[staking]"
```

### Installation Troubleshooting

If you encounter issues during installation:

1. **Dependency Conflicts**

   ```bash
   # Clean existing installations
   pip uninstall -y ale-py shimmy gymnasium
   
   # Install dependencies in correct order
   pip install "ale-py==0.10.2"
   pip install "shimmy[atari]==0.2.1"
   pip install "gymnasium[atari]==0.28.1"
   ```

2. **ROM Installation Issues**

   ```bash
   # Verify ROM installation
   python3 -c "import ale_py; print(ale_py.get_roms_path())"
   ```

3. **Python Version Issues**

   ```bash
   # Check Python version
   python3 --version  # Should be between 3.8 and 3.12
   ```

For detailed troubleshooting steps, see [Installation Guide](docs/installation.md).

### Verify Installation

```bash
# Check CLI is working
agent-arcade --version

# List available games
agent-arcade list-games
```

### Training an Agent

```bash
# Train Pong agent
agent-arcade train pong --render  # With visualization
agent-arcade train pong           # Without visualization (faster)

# Train Space Invaders agent
agent-arcade train space-invaders --render
agent-arcade train space-invaders --config configs/space_invaders_optimized_sb3_config.yaml

# Monitor training progress
tensorboard --logdir ./tensorboard/DQN_[game]_[timestamp]
```

### Evaluating Agents

```bash
# Evaluate Pong agent
agent-arcade evaluate pong --model models/pong_final.zip --episodes 10 --render

# Evaluate Space Invaders agent
agent-arcade evaluate space-invaders --model models/space_invaders_optimized/final_model.zip --episodes 5 --render --record

# View evaluation metrics and competition recommendations
agent-arcade stats [game] --model [model_path]
```

### Competition and Staking

```bash
# Check your wallet status
agent-arcade wallet-cmd status

# Stake on agent performance
agent-arcade stake place pong --model models/pong_final.zip --amount 10 --target-score 15
agent-arcade stake place space-invaders --model models/space_invaders_optimized/final_model.zip --amount 5 --target-score 300

# View competition leaderboard
agent-arcade leaderboard top pong

# View recent games
agent-arcade leaderboard recent pong

# View player stats
agent-arcade leaderboard player pong

# View global stats
agent-arcade leaderboard stats
```

## 🛠 Implementation Details

### DQN Architecture

- Custom CNN feature extractor (3 convolutional layers)
- Dual 512-unit fully connected layers
- Frame stacking (4 frames) for temporal information
- Optimized for Apple Silicon (MPS) and CPU performance

### Training Parameters

```yaml
total_timesteps: 1000000
learning_rate: 0.00025
buffer_size: 250000
learning_starts: 50000
batch_size: 256
exploration_fraction: 0.2
target_update_interval: 2000
frame_stack: 4
```

### Performance Optimizations

- Reward scaling for stable learning
- Frame normalization (0-255 to 0-1)
- Terminal on life loss for better exploration
- Gradient accumulation with optimized batch sizes

## 📊 Monitoring & Visualization

### TensorBoard Integration

- Real-time training metrics
- Episode rewards
- Learning rate progression
- Loss curves
- Exploration rate
- Training FPS

### Video Recording

- Automatic recording of milestone performances during the training run (stored in `videos/` directory)
- Progress visualization for workshops
- Performance comparison tools

## 🔄 Development Workflow

1. Train baseline model (15 min to 4 hours on M1/M2 Macs depending on the game and the number of training steps)
2. Evaluate and record performance
3. Iterate on hyperparameters if needed
4. Save best model for competition baseline

## 💎 NEAR Integration (Optional)

The NEAR integration allows you to stake tokens on your agent's performance and compete for rewards. This is an optional feature that requires:

1. **Prerequisites**:
   - Node.js >= 14.0.0 and npm
   - NEAR account (create at https://wallet.near.org/)
   - NEAR CLI (installed via npm)

2. **Installation**:
```bash
# Install NEAR CLI
npm install -g near-cli

# Install Agent Arcade with staking support
pip install -e ".[staking]"
```

3. **Login**:
```bash
# Simple login (opens web browser)
agent-arcade wallet-cmd login

# Specify network and account
agent-arcade wallet-cmd login --network testnet --account-id your-account.testnet
```

### Technical Implementation

Agent Arcade uses:
- NEAR CLI for wallet operations
- Direct JSON RPC API calls for contract interactions
- Secure key management via system keychain
- Asynchronous contract calls for better performance

### Staking System

- Stake NEAR on your agent's performance
- Tiered reward structure based on achieved scores:
  - Score ≥ 15: 3x stake
  - Score ≥ 10: 2x stake
  - Score ≥ 5: 1.5x stake
  - Score < 5: Stake goes to pool

### Example Usage

```bash
# Check your balance
agent-arcade wallet-cmd status

# Place a stake
agent-arcade stake place pong --model models/pong_final.zip --amount 10 --target-score 15

# View leaderboard
agent-arcade leaderboard top pong

# View your stats
agent-arcade leaderboard player pong
```

For detailed documentation, see [NEAR Integration Guide](docs/near-integration.md).

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

[MIT LICENSE](LICENSE)

# Login to NEAR Wallet
agent-arcade wallet-cmd login

# Check your wallet status
agent-arcade wallet-cmd status

# Logout from wallet
agent-arcade wallet-cmd logout
