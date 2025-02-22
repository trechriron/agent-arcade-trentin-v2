# Understanding Training Metrics

This guide helps you understand and optimize your agent's training for the NEAR AI Agent Games! 🎮

## Pre-trained Model Performance Benchmarks

### Pong (2M Steps)

```bash
Mean Reward: 19.50 ± 1.50
Episodes: 100
Success Rate: 95%
Training Time: ~4-5 hours on M-series MacBook
```

**Key Achievements**:

- Near-perfect win rate
- Consistent performance (low std dev)
- Efficient training (2M vs standard 10M steps)
- Mastered both offensive and defensive play

**Training Insights**:

- Double Q-learning prevents value overestimation
- PER focuses on important game moments
- 4-frame stacking captures paddle dynamics
- Success threshold of 15.0 ensures winning performance

### Space Invaders (4M Steps)

```bash
Mean Reward: 750.00 ± 150.00
Episodes: 100
Success Rate: 90%
Training Time: ~8-10 hours on M-series MacBook
```

**Key Achievements**:

- Superhuman performance level
- Strategic bunker usage
- Efficient shield management
- Learned timing patterns

**Training Insights**:

- Dueling architecture separates state value from actions
- Reward clipping stabilizes training
- Fire reset wrapper ensures proper episodes
- Success threshold of 500.0 indicates mastery

### River Raid (5M Steps)

```bash
Mean Reward: 1500.00 ± 300.00
Episodes: 100
Success Rate: 85%
Training Time: ~10-12 hours on M-series MacBook
```

**Key Achievements**:

- Advanced fuel management
- Long-term strategic planning
- Complex navigation mastery
- High score sustainability

**Training Insights**:

- Noisy Networks improve exploration
- Custom reward shaping guides learning
- Progress bonus encourages exploration
- Success threshold of 1000.0 requires complete mastery

## Key Training Metrics

### 1. Loss Values (Learning Quality)

```bash
# Good convergence pattern:
Initial loss: ~2.0-3.0
Mid-training: ~0.3-0.5
Converged: <0.2
```

**What to Watch**:

- Steady decrease over time
- No sudden spikes
- Final stability below 0.2
- Correlation with rewards

### 2. Exploration Progress

```bash
# Optimal decay pattern:
Start: 100% exploration
Mid-game: ~40% (based on exploration_fraction)
End-game: 1% (exploration_final_eps)
```

**Key Indicators**:

- Smooth linear decay
- No premature convergence
- Balance with reward improvement
- Game-specific adaptation

### 3. Training Efficiency

```bash
# Target metrics:
FPS: >400 (with 4 parallel envs)
Updates/second: ~100
Replay ratio: ~8
```

**Monitoring Points**:

- GPU/CPU utilization
- Memory usage
- Batch processing speed
- Environment step time

### 4. Learning Progress

```bash
# Key checkpoints:
Pong:
- 500K steps: Basic gameplay
- 1M steps: Consistent returns
- 2M steps: Mastery

Space Invaders:
- 1M steps: Basic patterns
- 2M steps: Strategic play
- 4M steps: Advanced tactics

River Raid:
- 2M steps: Navigation
- 3.5M steps: Fuel efficiency
- 5M steps: Full mastery
```

## Advanced Metrics

### 1. Value Estimation Quality

```bash
# Double Q-learning metrics:
Q-value spread: <50%
Value estimate stability: >80%
Action preference clarity: >90%
```

### 2. Replay Buffer Statistics

```bash
# PER metrics:
Priority distribution: Long-tailed
Important transitions: ~20%
Buffer utilization: >90%
```

### 3. Network Architecture Performance

```bash
# Dueling networks (Space Invaders, River Raid):
Value stream stability: >95%
Advantage stream differentiation: >80%
Action selection confidence: >90%
```

## Competition Preparation 🏆

### When to Submit Models

1. **Pong Readiness**:
   - Average score >15
   - Standard deviation <2.0
   - Success rate >90%
   - Consistent against different opponents

2. **Space Invaders Readiness**:
   - Average score >500
   - Pattern recognition evident
   - Strategic positioning
   - Efficient shooting

3. **River Raid Readiness**:
   - Average score >1000
   - Fuel efficiency >80%
   - Navigation success >90%
   - Long-term survival

### Performance Red Flags 🚩

1. **Training Issues**:
   - Loss not converging below 0.5
   - Exploration ending too early
   - Reward plateaus
   - High variance in returns

2. **Model Problems**:
   - Q-value overestimation
   - Action space imbalance
   - Poor generalization
   - Unstable behaviors

3. **Resource Concerns**:
   - Memory leaks
   - Low FPS
   - High CPU/GPU usage
   - Slow replay sampling

## Monitoring Tools

### 1. TensorBoard Integration

```bash
tensorboard --logdir ./tensorboard
```

**Key Dashboards**:

- Training metrics
- Network gradients
- Action distributions
- Resource usage

### 2. Checkpoint Analysis

```bash
# Evaluation command
agent-arcade evaluate <game> --model models/<game>/checkpoints/model_100000.zip
```

**Key Metrics**:

- Checkpoint performance
- Behavioral consistency
- Resource efficiency
- Success rate

### 3. Video Recording

```bash
# Record evaluation episodes
agent-arcade evaluate <game> --model final_model.zip --record
```

**Analysis Points**:

- Strategic decisions
- Failure patterns
- Exploration behavior
- Long-term strategy

## Next Steps

1. **Model Refinement**:
   - Fine-tune hyperparameters
   - Optimize reward shaping
   - Enhance exploration strategy
   - Improve stability

2. **Competition Entry**:
   - Verify model compatibility
   - Test against baselines
   - Document performance
   - Submit with confidence

3. **Community Engagement**:
   - Share training tips
   - Discuss optimizations
   - Compare strategies
   - Build on successes
