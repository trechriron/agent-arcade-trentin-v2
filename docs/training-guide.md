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

## Model Management

### Saved Model Structure

After training, models are organized as follows:

```bash
models/
├── pong/
│   ├── pong_final.zip           # Final trained model
│   ├── checkpoints/             # Periodic checkpoints
│   │   ├── checkpoint_100000.zip
│   │   ├── checkpoint_200000.zip
│   │   └── ...
│   └── videos/                  # Recorded gameplay
├── space_invaders/
│   └── ...
└── river_raid/
    └── ...
```

### Transferring Models

#### From Remote GPU to Local Machine

1. **Using SCP**:

```bash
# Create local directory
mkdir -p ~/agent-arcade-models

# Copy all models and checkpoints
scp -r ubuntu@your-lambda-ip:~/agent-arcade/models/* ~/agent-arcade-models/
```

2. **Using Rsync (Recommended)**:

```bash
# Copy models with progress indication
rsync -avz --progress ubuntu@your-lambda-ip:~/agent-arcade/models/ ~/agent-arcade-models/

# Copy TensorBoard logs for local analysis
rsync -avz --progress ubuntu@your-lambda-ip:~/agent-arcade/tensorboard/ ~/agent-arcade-tensorboard/
```

### Local Evaluation

After transferring, you can evaluate models locally:

```bash
# Basic evaluation
agent-arcade evaluate pong ~/agent-arcade-models/pong/pong_final.zip --episodes 10

# Evaluation with rendering
agent-arcade evaluate pong ~/agent-arcade-models/pong/pong_final.zip --episodes 10 --render

# Record evaluation videos
agent-arcade evaluate pong ~/agent-arcade-models/pong/pong_final.zip --episodes 5 --record
```

### Best Practices for Model Management

1. **Version Control**
   - Keep checkpoints for different training runs
   - Document hyperparameters used
   - Track evaluation metrics

2. **Backup Strategy**
   - Regular transfers from GPU instances
   - Keep multiple checkpoints
   - Document performance at each checkpoint

3. **Performance Documentation**
   - Record final evaluation metrics
   - Save TensorBoard logs
   - Document hardware specifications used

4. **Model Sharing**
   - Include configuration files
   - Document environment versions
   - Provide example evaluation scripts
