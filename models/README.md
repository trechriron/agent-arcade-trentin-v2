# Agent Arcade Models

This directory contains model configurations and saved models for different Atari games supported by Agent Arcade.

## Model Versioning

Each model follows semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Breaking changes in model architecture or input requirements
- MINOR: Performance improvements, non-breaking feature additions
- PATCH: Bug fixes, hyperparameter tuning

Example: `pong_v2.1.0.zip` indicates version 2.1.0 of the Pong model.

## Directory Structure

```bash
models/
├── README.md
├── pong/
│   ├── config.yaml           # Training configuration
│   ├── checkpoints/          # Training checkpoints
│   ├── versions/             # Version history
│   │   ├── v1.0.0/          # Initial release
│   │   └── v2.0.0/          # Current version
│   └── metadata.json         # Model metadata and version info
├── space_invaders/
│   ├── config.yaml
│   ├── checkpoints/
│   ├── versions/
│   └── metadata.json
└── riverraid/
    ├── config.yaml
    ├── checkpoints/
    ├── versions/
    └── metadata.json
```

## Hardware-Specific Configurations

### GH200 GPU Optimizations

- Large batch sizes (1024-2048)
- Extended frame stacking (16 frames)
- Multiple parallel environments (16)
- Gradient accumulation
- Large replay buffers

### Macbook M-SeriesOptimizations

- Reduced batch sizes (32-128)
- Standard frame stacking (4 frames)
- Memory-efficient buffers
- MPS-specific settings

## Game-Specific Configurations

### Pong (v2.0.0)

- Environment: `ALE/Pong-v5`
- Training Steps: 2M
- Expected Score: >19 points (95% win rate)
- Architecture: CNN with [512, 512] hidden layers
- Hardware Target: GH200 GPU
- Training Time: ~1 hour

### Space Invaders (v2.0.0)

- Environment: `ALE/SpaceInvaders-v5`
- Training Steps: 5M
- Expected Score: >1000 points
- Architecture: CNN with [1024, 512] hidden layers
- Hardware Target: GH200 GPU
- Training Time: ~2 hours
- Special Features: Prioritized replay, dueling networks

### River Raid (v2.0.0)

- Environment: `ALE/RiverRaid-v5`
- Training Steps: 10M
- Expected Score: >12000 points
- Architecture: CNN with [2048, 1024] hidden layers
- Hardware Target: GH200 GPU
- Training Time: ~3 hours
- Special Features: Custom reward shaping, extended exploration

## Model Metadata

Each model's `metadata.json` includes:

```json
{
  "version": "2.0.0",
  "training_hardware": "GH200",
  "framework_versions": {
    "pytorch": "2.2.0",
    "gymnasium": "0.29.1",
    "ale_py": "0.8.1"
  },
  "training_date": "2024-02-23",
  "performance_metrics": {
    "avg_score": 0.0,
    "success_rate": 0.0,
    "training_time": "0h0m",
    "convergence_step": 0
  },
  "hyperparameters": {
    "learning_rate": 0.00025,
    "batch_size": 2048,
    "buffer_size": 2000000
  }
}
```

## Usage

1. **Training a Model**:

   ```bash
   agent-arcade train pong --config models/pong/config.yaml
   ```

2. **Evaluating a Model**:

   ```bash
   agent-arcade evaluate pong --model models/pong/versions/v2.0.0/model.zip
   ```

3. **Competition Entry**:

   ```bash
   agent-arcade stake place pong --model models/pong/versions/v2.0.0/model.zip --amount 10
   ```

## Model Selection Guidelines

1. **Version Selection**:
   - Use latest version for best performance
   - Check hardware compatibility
   - Review performance metrics

2. **Hardware Requirements**:
   - GH200 Models:
     - GPU: NVIDIA GH200 or equivalent
     - RAM: 32GB+ recommended
     - Storage: 10GB+ for full version history

   - M-Series Models:
     - Apple Silicon M-Series
     - RAM: 32GB recommended
     - Storage: 5GB+ for full version history

3. **Performance Verification**:
   - Run evaluation over 200+ episodes
   - Check for consistent performance
   - Verify hardware utilization

## Contributing Models

1. **Testing Requirements**:
   - 200+ evaluation episodes
   - Performance metrics documented
   - Hardware specifications listed

2. **Documentation**:
   - Update metadata.json
   - Document training process
   - Include performance graphs

3. **Version Control**:
   - Create new version directory
   - Update README.md
   - Tag release in repository

## Monitoring and Metrics

Access training metrics:

```bash
tensorboard --logdir ./tensorboard
```

Key metrics:

- Episode rewards
- Learning curves
- Resource utilization
- Training speed (FPS)
- Model convergence

## Support

For issues or questions:

- GitHub Issues: [Agent Arcade Issues](https://github.com/jbarnes850/agent-arcade/issues)
- Documentation: [Agent Arcade Docs](https://github.com/jbarnes850/agent-arcade/tree/main/docs)
