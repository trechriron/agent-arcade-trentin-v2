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

- Observing game frames as input (what the agent "sees" to play the game).
- Learning which actions lead to higher rewards through trial and error.
- Using a neural network to approximate the optimal action-value function and make decisions based on the processed game state.
- Storing and learning from past experiences (replay buffer) and improving through experience.

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

- **Node.js & npm**: Required for NEAR CLI (v16 or higher)
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

# Optional: Install NEAR integration for staking (if not selected during installation)
pip install -e ".[staking]"
```

The installation script will:

- Set up a Python virtual environment
- Install all required dependencies
- Download and configure Atari ROMs
- Install NEAR CLI (optional)
- Create necessary directories
- Verify the installation

For GPU support, an additional script is available:

```bash
chmod +x ./gpu_install.sh
./gpu_install.sh
```

### Installation Troubleshooting

If you encounter issues during installation:

1. **Dependency Conflicts**

   ```bash
   # Clean existing installations
   pip uninstall -y ale-py shimmy gymnasium
   
   # Install dependencies in correct order
   pip install "ale-py==0.10.1"
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

# Train with custom config
agent-arcade train space-invaders --config models/space_invaders/config.yaml --output-dir models/space_invaders

# Train with checkpointing
agent-arcade train river-raid --checkpoint-freq 50000 --output-dir models/river_raid

# Monitor training progress
tensorboard --logdir ./tensorboard/DQN_[game]_[timestamp]
```

### Evaluating Agents

> **Important**: You must be logged in with your NEAR wallet before running evaluations:

```bash
agent-arcade wallet-cmd login
```

```bash
# Basic evaluation
agent-arcade evaluate pong models/pong/final_model.zip --episodes 100

# Evaluation with rendering
agent-arcade evaluate space-invaders models/space_invaders/final_model.zip --render

# Evaluation with video recording
agent-arcade evaluate river-raid models/river_raid/final_model.zip --render --record
```

### Competition and Staking

```bash
# Check wallet status
agent-arcade wallet-cmd status

# Place stake
agent-arcade stake place pong \
  --model models/pong/final_model.zip \
  --amount 10 \
  --target-score 15

# View leaderboard
agent-arcade leaderboard top pong

# View recent games
agent-arcade leaderboard recent pong --limit 5

# View player stats
agent-arcade leaderboard player pong

# View global stats
agent-arcade leaderboard stats

# View pool balance
agent-arcade pool balance
```

### Training Parameters

```yaml
total_timesteps: 1000000
learning_rate: 0.00025
buffer_size: 250000
learning_starts: 50000
batch_size: 256
exploration_fraction: 0.2
target_update_interval: 2000
frame_stack: 16
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

Training locally? See the [Training Guide](docs/training-guide.md) for more details.

1. Train baseline model:
   - Pong: ~2 hours on M-series Mac
   - Space Invaders: ~6 hours on M-series Mac
   - River Raid: Custom reward shaping, training time varies
2. Evaluate and record performance
3. Iterate on hyperparameters if needed
4. Save best model for competition baseline

Each game directory contains:

- `config.yaml`: Game-specific training configuration
- `checkpoints/`: Saved model checkpoints during training
- `final_model.zip`: Best performing model

## 💎 NEAR Integration (Optional)

The NEAR integration allows you to stake tokens on your agent's performance and compete for rewards. Our staking contract is written in Rust and is already deployed on testnet. To keep the repository lightweight, compiled artifacts (including the generated WASM) are ignored from version control.

### Building the Staking Contract Locally

If you want to build the contract locally (for testing or to deploy your own version), follow these steps:

1. **Navigate to the Contract Directory:**

   ```bash
   cd contract
   ```

2. **Build the Contract for the WASM Target:**

   ```bash
   cargo build --target wasm32-unknown-unknown --release
   ```

   This will compile the contract, and the output (e.g., a `.wasm` file) will be generated under the `target/` directory.

3. **Note:**  
   The contract's compiled artifacts (including the WASM file) are not tracked in the repository to keep our codebase lightweight. We recommend rebuilding the contract locally as needed.

### Using the Deployed Contract

Since our staking contract is already deployed on NEAR testnet, you can use the provided NEAR CLI commands to interact with it:

```bash
# Check your wallet status
agent-arcade wallet-cmd status

# Place a stake on your agent
agent-arcade stake place pong --model models/pong/final_model.zip --amount 10 --target-score 15
```

See the [Competition Guide](docs/competition-guide.md) for more details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

[MIT LICENSE](LICENSE)
