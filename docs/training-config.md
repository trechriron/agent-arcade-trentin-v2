# Training Configuration Guide

Learn how to configure and optimize your AI agent's training process for maximum performance.

## Configuration File

The training configuration is managed through `configs/pong_sb3_config.yaml`. Here's a detailed breakdown of key parameters:

```yaml
# Core Training Parameters
total_timesteps: 1000000    # Total training steps
learning_rate: 0.00025      # Learning rate for neural network
buffer_size: 500000         # Size of replay buffer
learning_starts: 100000     # Steps before learning begins
batch_size: 256            # Batch size for training
exploration_fraction: 0.4   # Fraction of total steps for exploration
exploration_final_eps: 0.01 # Final exploration rate

# Network Optimization
gamma: 0.99                # Discount factor
target_update_interval: 1000 # Steps between target network updates
gradient_steps: 4          # Gradient steps per update
train_freq: 4             # Training frequency
frame_stack: 4            # Number of frames to stack
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
