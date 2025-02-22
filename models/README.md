# Agent Arcade Models

This directory contains model configurations and saved models for different Atari games supported by Agent Arcade.

## Directory Structure

```bash
models/
├── README.md
├── pong/
│   ├── config.yaml           # Training configuration
│   ├── checkpoints/          # Training checkpoints
│   └── final_model.zip       # Best performing model
├── space_invaders/
│   ├── config.yaml           # Training configuration
│   ├── checkpoints/          # Training checkpoints
│   └── final_model.zip       # Best performing model
└── river_raid/
    ├── config.yaml           # Training configuration
    ├── checkpoints/          # Training checkpoints
    └── final_model.zip       # Best performing model
```

## Game Configurations

### Pong

- Environment: `ALE/Pong-v5`
- Training Steps: 1M
- Expected Score: 15+ points
- Training Time: ~2 hours on M-series MacBook

### Space Invaders

- Environment: `ALE/SpaceInvaders-v5`
- Training Steps: 3M
- Expected Score: 270+ points
- Training Time: ~6 hours on M-series MacBook
- Special Features: Prioritized replay, dueling networks

### River Raid

- Environment: `ALE/RiverRaid-v5`
- Training Steps: 2M
- Custom reward shaping for fuel management
- Advanced training features enabled

## Usage

1. **Training a Model**:

   ```bash
   agent-arcade train pong --config models/pong/config.yaml
   ```

2. **Evaluating a Model**:

   ```bash
   agent-arcade evaluate pong --model models/pong/final_model.zip
   ```

3. **Competition Entry**:

   ```bash
   agent-arcade stake place pong --model models/pong/final_model.zip --amount 10 --target-score 15
   ```

## Model Checkpoints

Each game directory contains a `checkpoints` folder that stores intermediate models during training. Checkpoints are saved every 100,000 steps and can be used to:

- Resume training from a specific point
- Compare performance across training stages
- Select the best performing model

## Configuration Details

Each `config.yaml` file contains:

- Training hyperparameters
- Network architecture settings
- Preprocessing options
- Game-specific configurations
- Visualization settings

## Performance Metrics

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir ./tensorboard
```

Key metrics to watch:

- Episode rewards
- Loss values
- Exploration rate
- Training FPS

## Best Practices

1. **Start with Default Configs**:
   - Use provided configurations as starting points
   - They're optimized for M-series MacBooks
   - Adjust based on your hardware

2. **Incremental Training**:
   - Start with 250K steps to verify setup
   - Increase to full training once stable
   - Save checkpoints frequently

3. **Model Selection**:
   - Evaluate models over multiple episodes
   - Consider both mean score and consistency
   - Test against competition requirements

4. **Hardware Optimization**:
   - CPU: Use 4+ cores
   - RAM: 8GB minimum
   - GPU: Optional but recommended
   - Storage: 1GB for models
