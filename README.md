# Agent Arcade: AI Game Agents on NEAR

A platform for training and competing with AI agents in classic arcade games using Stable Baselines3, Gymnasium, and the Arcade Learning Environment with staking powered by NEAR.

## 🎬 Demo: Trained Agent in Action

Watch our trained DQN agent play Pong after completing its training run:

[Watch Demo Video](docs/Pong%20-%20Reinforcement%20Learning%20Demo.mov)

> This agent learned to play Pong from scratch through trial and error through deep reinforcement learning. Train your own agent and compete for rewards!

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

- **Environment**: Atari Pong (via Gymnasium/ALE) - provides game state and handles actions
- **Agent**: DQN with custom CNN - processes game frames and decides actions
- **Memory**: Replay buffer - stores experiences for learning
- **Training Loop**: Alternates between playing and learning from experiences

## 🎮 Current Games

- **Pong**: Deep Q-Network (DQN) implementation with optimized training parameters

> More games will be added soon.

## 🚀 Quick Start

### Prerequisites

```bash
# Create and activate virtual environment
python -m venv drl-env
source drl-env/bin/activate  # On Windows: drl-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Atari ROMs
pip install "gymnasium[accept-rom-license,atari]"
```

### Training an Agent

```bash
# Train from scratch with visualization
python scripts/train_pong.py --render

# Train without visualization (faster)
python scripts/train_pong.py

# Monitor training progress
tensorboard --logdir ./tensorboard/DQN_pong_[timestamp]
```

### Using Pre-trained Models

```bash
# Evaluate the pre-trained model
python scripts/evaluate_pong.py --model models/pong_dqn_1000000_steps.zip
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

## 💎 NEAR Integration

Train your agent, stake NEAR tokens on its performance, and compete for rewards!

### Staking System

- Stake NEAR on your agent's performance
- Tiered reward structure based on achieved scores:
  - Score ≥ 15: 3x stake
  - Score ≥ 10: 2x stake
  - Score ≥ 5: 1.5x stake
  - Score < 5: Stake goes to pool

### Smart Contract

- Manages stakes and rewards
- Maintains leaderboard
- Handles pool distribution
- Automatic reward calculation

### Getting Started with NEAR

**Install NEAR CLI**:

```bash
npm install -g near-cli
```

**Login to Your NEAR Account**:

```bash
# First-time login (opens web wallet)
agent-arcade login

# Or specify your account
agent-arcade login --account-id alice.near
```

**Check Your Balance**:

```bash
agent-arcade balance
```

### Staking and Rewards

**Stake on Performance**:

```bash
# Stake 10 NEAR on achieving score ≥ 15
pong-arcade stake --model-path models/my_agent.zip --amount 10 --target-score 15 ## make sure to replace the model path with the path to your trained model
```

**Evaluate Your Agent**:

```bash
# Automatically evaluates and claims rewards if successful
agent-arcade pong evaluate
```

**View Rewards**:

```bash
# Check your earnings and statistics
agent-arcade pong stats
```

### Competition Features

**Global Leaderboard**:

```bash
# View top players
agent-arcade pong leaderboard

# View recent games
agent-arcade pong recent
```

**Reward Structure**:

- Score ≥ 15: 3x stake
- Score ≥ 10: 2x stake
- Score ≥ 5: 1.5x stake
- Score < 5: Stake goes to pool

**Pool Statistics**:

```bash
# View current pool balance
agent-arcade pool
```

### Pool Statistics

- Initial pool: 1000 NEAR
- Minimum stake: 1 NEAR
- Maximum reward: 5x stake

For detailed documentation, see [NEAR Integration Guide](docs/near-integration.md).

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

[MIT LICENSE](LICENSE)
