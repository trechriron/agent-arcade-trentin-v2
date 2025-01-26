# Training Configuration Guide

This guide explains the configuration options for training agents in Agent Arcade.

## Common Parameters

These parameters are common across all games:

```yaml
algo: "DQN"                    # Algorithm to use (currently only DQN supported)
total_timesteps: 1000000       # Total number of training steps
learning_rate: 0.00025         # Learning rate for the optimizer
buffer_size: 500000            # Size of the replay buffer
learning_starts: 100000        # Number of steps before learning starts
batch_size: 256               # Batch size for training
exploration_fraction: 0.4      # Fraction of total timesteps for exploration
exploration_final_eps: 0.01    # Final exploration rate
train_log_interval: 100       # Interval for logging training metrics

# Network optimization
gamma: 0.99                   # Discount factor
target_update_interval: 1000   # Steps between target network updates
gradient_steps: 4             # Number of gradient steps per update
train_freq: 4                 # Number of steps between updates
frame_stack: 4                # Number of frames to stack

# Preprocessing
scale_rewards: true           # Whether to scale rewards
normalize_frames: true        # Whether to normalize frames
terminal_on_life_loss: true   # End episode on life loss
```

## Game-Specific Parameters

### Pong

```yaml
env: "ALE/Pong-v5"            # Environment ID for Pong
```

### Space Invaders

```yaml
env: "ALE/SpaceInvaders-v5"   # Environment ID for Space Invaders
difficulty: 0                 # Game difficulty (0-1)
mode: 0                      # Game mode (0-15)
```

Space Invaders specific notes:

- `difficulty`: Controls game difficulty (0: normal, 1: hard)
- `mode`: Different game variations (0-15) affecting gameplay mechanics

## Workshop and Visualization

```yaml
viz_interval: 25000           # Steps between visualization updates
video_interval: 100000        # Steps between video recordings
video_length: 400            # Length of recorded videos
checkpoint_interval: 100000   # Steps between model checkpoints
demo_mode: false             # Whether to run in demo mode
```

## Example Configurations

### Pong Configuration

```yaml
algo: "DQN"
env: "ALE/Pong-v5"
total_timesteps: 1000000
learning_rate: 0.00025
buffer_size: 500000
learning_starts: 100000
batch_size: 256
exploration_fraction: 0.4
exploration_final_eps: 0.01
```

### Space Invaders Configuration

```yaml
algo: "DQN"
env: "ALE/SpaceInvaders-v5"
total_timesteps: 1000000
learning_rate: 0.00025
buffer_size: 500000
learning_starts: 100000
batch_size: 256
exploration_fraction: 0.4
exploration_final_eps: 0.01
difficulty: 0
mode: 0
```

## Using Configurations

You can use these configurations through the CLI:

```bash
# Train Pong
agent-arcade pong train --config configs/pong_sb3_config.yaml

# Train Space Invaders
agent-arcade space-invaders train --config configs/space_invaders_sb3_config.yaml
```

Or modify parameters directly:

```bash
# Train with visualization
agent-arcade space-invaders train --config configs/space_invaders_sb3_config.yaml --render

# Train with video recording
agent-arcade space-invaders train --config configs/space_invaders_sb3_config.yaml --record
```

## Parameter Explanations

### Core Parameters

- **total_timesteps**: Total number of game steps for training
  - Higher values = more training time but potentially better performance
  - Recommended: 500K-1M steps for Pong

- **exploration_fraction**: Balance between exploration and exploitation
  - Higher values = more exploration
  - Recommended: 0.3-0.4 for discovering winning strategies

- **buffer_size**: Memory for storing game experiences
  - Larger buffer = more diverse learning experiences
  - Recommended: At least 50% of total timesteps

### Network Parameters

- **frame_stack**: Number of consecutive frames fed to the network
  - Higher values = better motion detection
  - Recommended: 4 frames for Pong

- **gradient_steps**: Learning iterations per update
  - Higher values = more thorough learning
  - Recommended: 2-4 steps

## Optimization Tips

1. **For Better Exploration**

   ```yaml
   exploration_fraction: 0.4
   exploration_final_eps: 0.01
   ```

2. **For Stable Learning**

   ```yaml
   learning_rate: 0.00025
   buffer_size: 500000
   batch_size: 256
   ```

3. **For Quick Adaptation**

   ```yaml
   target_update_interval: 1000
   gradient_steps: 4
   ```

## Performance Monitoring

Monitor these metrics in TensorBoard:

- Episode rewards (target: consistently > 15)
- Loss values (should stabilize over time)
- Exploration rate (should decrease smoothly)
- Training FPS (higher is better)

## Common Issues

1. **Poor Performance**
   - Increase exploration_fraction
   - Increase buffer_size
   - Check reward scaling

2. **Unstable Learning**
   - Decrease learning_rate
   - Increase batch_size
   - Adjust target_update_interval

3. **Slow Training**
   - Decrease gradient_steps
   - Adjust frame_stack
   - Optimize hardware usage

## Hardware Recommendations

- CPU: 4+ cores
- RAM: 8GB minimum
- GPU: Optional but recommended
- Storage: 1GB for models
