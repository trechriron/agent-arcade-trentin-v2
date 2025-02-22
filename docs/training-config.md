# Training Configuration Guide

This guide explains the optimized configuration options for training agents in Agent Arcade, based on research from DeepMind's DQN papers and subsequent improvements.

## Core DQN Improvements

Our configurations implement several key DQN enhancements:

1. **Double Q-Learning**

```yaml
double_q: true  # Reduces value overestimation
```

2. **Prioritized Experience Replay (PER)**

```yaml
prioritized_replay: true
prioritized_replay_alpha: 0.6  # How much prioritization to use
prioritized_replay_beta0: 0.4  # Initial importance sampling weight
```

3. **Dueling Networks** (for complex games)

```yaml
dueling: true  # Separate value and advantage streams
```

4. **Noisy Networks** (for exploration-heavy games)

```yaml
noisy_nets: true  # Parameter space noise for exploration
```

## Common Parameters

These parameters are optimized across all games:

```yaml
algo: "DQN"                    # Algorithm (DQN with improvements)
total_timesteps: 2000000+      # Game-specific training length
learning_rate: 0.00025         # From Nature DQN paper
buffer_size: 1000000           # 1M transitions (Nature paper)
learning_starts: 50000         # Pre-fill replay buffer
batch_size: 32                 # Standard for Atari
exploration_fraction: 0.1      # Linear annealing over 10% of steps
exploration_final_eps: 0.01    # Final exploration rate

# Network optimization
gamma: 0.99                    # Discount factor
target_update_interval: 1000   # Target network update frequency
train_freq: 4                  # Update every 4 steps
gradient_steps: 1              # Gradient steps per update
frame_stack: 4                 # Standard Atari frame stacking
```

## Game-Specific Configurations

### Pong (Simpler Dynamics)

```yaml
total_timesteps: 2000000       # Shorter training for simpler dynamics
success_threshold: 15.0        # Winning score threshold
frame_stack: 4                 # Standard frame stacking
# No dueling/noisy nets needed
```

### Space Invaders (Complex Patterns)

```yaml
total_timesteps: 4000000       # Longer for pattern learning
dueling: true                  # Better value estimation
fire_reset: true              # Game-specific requirement
reward_clipping: [-1, 1]      # Stability improvement
success_threshold: 500.0       # Competitive score
```

### River Raid (Navigation + Resource)

```yaml
total_timesteps: 5000000       # Longest for fuel management
dueling: true                  # Value/advantage separation
noisy_nets: true              # Better exploration
reward_shaping:               # Custom rewards
  fuel_bonus: 2.0
  progress_bonus: 0.1
success_threshold: 1000.0
```

## Evaluation Settings

Standardized evaluation across games:

```yaml
eval_episodes: 100            # Statistically significant sample
eval_freq: 25000             # Regular evaluation intervals
eval_deterministic: true     # No exploration during eval
render_eval: false          # Headless evaluation
```

## Checkpointing and Monitoring

```yaml
save_freq: 100000            # Save every 100K steps
checkpoint_path: "models/{game}/checkpoints"
keep_checkpoints: 5          # Keep last 5 checkpoints

# Visualization
viz_interval: 25000          # TensorBoard updates
video_interval: 100000       # Record gameplay videos
video_length: 400           # Video duration
```

## Hardware Optimization

```yaml
n_envs: 4                    # Parallel environments
n_steps: 4                   # Steps per env before update
device: "auto"               # Auto GPU/CPU selection
```

## Training Duration Guidelines

Based on complexity:

- **Pong**: 2M steps (~4-5 hours)
- **Space Invaders**: 4M steps (~8-10 hours)
- **River Raid**: 5M steps (~10-12 hours)

## Performance Monitoring

Monitor in TensorBoard:

- Episode rewards (game-specific thresholds)
- Loss values (should stabilize <0.5)
- Exploration rate (smooth decay)
- Training FPS (hardware efficiency)

## Hardware Recommendations

- CPU: 4+ cores recommended
- RAM: 16GB recommended (8GB minimum)
- GPU: Optional but recommended for faster training
- Storage: 2GB for models and checkpoints

## Common Issues and Solutions

1. **Unstable Training**

   ```yaml
   max_grad_norm: 10         # Gradient clipping
   reward_clipping: [-1, 1]  # Reward stability
   ```

2. **Poor Exploration**

   ```yaml
   noisy_nets: true         # Parameter space noise
   # or
   exploration_fraction: 0.2 # More exploration time
   ```

3. **Slow Learning**

   ```yaml
   prioritized_replay: true  # Focus on important transitions
   n_envs: 4                # Parallel environments
   ```

## Pre-trained Models

We provide pre-trained models for each game:

```bash
models/
├── pong/
│   ├── final_model.zip        # 15+ average score
├── space_invaders/
│   ├── final_model.zip        # 500+ average score
└── river_raid/
    ├── final_model.zip        # 1000+ average score
```
