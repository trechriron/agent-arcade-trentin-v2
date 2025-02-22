# Training Guide

This guide covers everything you need to know about training agents in Agent Arcade, from configuration to performance metrics.

## Training Configuration

### Basic Configuration

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

### Advanced Parameters

- **total_timesteps**: Total number of environment steps for training
- **learning_rate**: Rate at which the model updates its parameters
- **buffer_size**: Size of the replay buffer for experience replay
- **learning_starts**: Number of steps before starting gradient updates
- **batch_size**: Number of samples per gradient update
- **exploration_fraction**: Fraction of training spent exploring
- **target_update_interval**: Steps between target network updates
- **frame_stack**: Number of frames to stack for temporal information

## Performance Metrics

### Core Metrics

1. **Mean Episode Reward**
   - Primary indicator of agent performance
   - Higher values indicate better gameplay
   - Compare against human benchmarks

2. **Success Rate**
   - Percentage of episodes meeting target score
   - Key metric for competition evaluation
   - Used for reward multiplier calculation

3. **Episode Length**
   - Number of steps per episode
   - Indicates efficiency and survival ability
   - Game-specific optimal ranges

### Training Progress Metrics

1. **Learning Curves**
   - Episode rewards over time
   - Loss function trends
   - Exploration rate decay

2. **Policy Statistics**
   - Action distribution
   - Value function estimates
   - Policy entropy

3. **Resource Usage**
   - Training FPS
   - Memory utilization
   - GPU/CPU usage

## Monitoring Tools

### TensorBoard Integration

```bash
# Launch TensorBoard
tensorboard --logdir ./tensorboard/DQN_[game]_[timestamp]
```

Available metrics:

- Episode rewards
- Learning rate
- Loss curves
- Training FPS
- Network gradients

### Video Recording

```bash
# Enable recording during training
agent-arcade train pong --record

# Record evaluation episodes
agent-arcade evaluate pong models/pong/final_model.zip --record
```

## Optimization Tips

### Performance Tuning

1. **Frame Processing**
   - Normalize pixel values (0-1)
   - Stack frames for temporal information
   - Apply frame skipping for efficiency

2. **Network Architecture**
   - Custom CNN feature extractor
   - Dual 512-unit fully connected layers
   - Apple Silicon (MPS) optimizations

3. **Training Stability**
   - Gradient clipping
   - Learning rate scheduling
   - Reward scaling

### Common Issues

1. **Poor Learning**
   - Check learning rate
   - Verify reward scaling
   - Inspect network gradients

2. **Unstable Performance**
   - Increase buffer size
   - Adjust batch size
   - Modify update frequency

3. **Resource Constraints**
   - Enable frame skipping
   - Reduce batch size
   - Optimize replay buffer

## Game-Specific Guidelines

### Pong

- **Recommended Settings**

  ```yaml
  total_timesteps: 500000
  learning_rate: 0.0001
  frame_stack: 4
  ```

- Target Score: 21
- Success Threshold: 15

### Space Invaders

- **Recommended Settings**

  ```yaml
  total_timesteps: 1000000
  learning_rate: 0.00025
  frame_stack: 16
  ```

- Target Score: 1000
- Success Threshold: 500

### River Raid

- **Recommended Settings**

  ```yaml
  total_timesteps: 2000000
  learning_rate: 0.0001
  frame_stack: 16
  ```

- Target Score: 15000
- Success Threshold: 10000

## Best Practices

1. **Start Simple**
   - Use default configurations
   - Train for short periods
   - Validate basic functionality

2. **Iterate Carefully**
   - Change one parameter at a time
   - Document changes and results
   - Use version control for configs

3. **Monitor Progress**
   - Regular evaluation episodes
   - Track key metrics
   - Save checkpoints frequently

4. **Prepare for Competition**
   - Validate against test episodes
   - Record demonstration videos
   - Document performance characteristics
