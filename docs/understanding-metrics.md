# Understanding Training Metrics

This guide helps you understand what's happening while your agent is training, so you can build winning agents for the NEAR AI Agent Games! 🎮

## Pre-trained Model Results

### Space Invaders (3M Steps)

```bash
Mean Reward: 270.00 ± 0.00
Episodes: 20
Success Rate: Perfect consistency
Training Time: ~6 hours on Macbook M3 Max (local, no GPU)
```

- **Key Achievements**:
  - Perfect consistency (0.00 standard deviation)
  - Above average human performance
  - Efficient training (3M vs industry standard 10M-40M steps)
  - Learned optimal defensive strategy

- **Training Insights**:
  - Longer training (3M steps) significantly improves stability
  - Model shows perfect reproducibility across episodes
  - Demonstrates successful exploration-exploitation balance

### Pong (500K Steps)

```bash
Mean Reward: 15.00 ± 4.71
Episodes: 20
Success Rate: ~70%
Training Time: ~2 hours on Macbook M3 Max (local, no GPU)
```

- **Key Achievements**:
  - Competitive performance against AI opponents
  - Good balance of aggression and defense
  - Efficient training for the complexity level
  - Demonstrates successful paddle control

- **Training Insights**:
  - 500K steps provides good performance/training time trade-off
  - Higher variance than Space Invaders but still competitive
  - Shows effective learning of game physics and timing

### Training Duration Guidelines

Based on our model results:

- **250K steps**: ~30 minutes (basic learning)
- **500K steps**: ~2 hours (competitive performance)
- **3M steps**: ~6 hours (optimal performance)
- **Training Speed**: ~1.4K steps/minute on M-series MacBook

## Key Metrics to Watch

### 1. Exploration Rate (Your Agent's Learning Journey)

```bash
exploration_rate: 0.591 (59.1%)
```

- **What it means**: How often your agent tries new moves vs using what it knows
- **Good signs**:
  - Starts high (around 100%) and gradually decreases
  - For Pong, should be around 60% at 150K steps
- **Why it matters**: Too low too early = agent might miss winning strategies

### 2. Loss Values (Learning Quality)

```bash
loss: 0.0496
```

- **What it means**: How "surprised" your agent is by game outcomes
- **Good signs**:
  - Values between 0.03-0.06
  - Relatively stable, not wildly jumping
- **Why it matters**: Stable loss = agent is learning consistently

### 3. Training Progress

```bash
total_timesteps: 165,601/1,000,000 (17%)
```

- **What it means**: How many game actions your agent has experienced
- **Good signs**:
  - Steady progress
  - FPS (frames per second) staying consistent
- **Why it matters**: More experience = better gameplay

### 4. Episodes (Complete Games)

```bash
episodes: 776
```

- **What it means**: Number of full games played
- **Good signs**:
  - Regular increase in episode count
  - For Pong, aim for 700+ episodes by 150K steps
- **Why it matters**: More complete games = more learning opportunities

## Competition Tips 🏆

### When to Stop Training

1. **Early Signs** (around 200K steps):
   - Loss consistently below 0.05
   - Agent scoring points occasionally
   - Keep training!

2. **Good Progress** (around 500K steps):
   - Loss stable around 0.04
   - Agent scoring 5+ points regularly
   - Consider saving a backup model

3. **Competition Ready** (around 800K-1M steps):
   - Loss stable below 0.04
   - Agent consistently scoring 15+ points
   - Ready to compete for rewards!

### Red Flags 🚩

- Loss consistently above 0.08
- Exploration rate dropping too quickly
- FPS dropping significantly
- No improvement in game scores

## Monitoring Tips

1. **Use TensorBoard**

   ```bash
   tensorboard --logdir ./tensorboard
   ```

   Visit <http://localhost:6006> to see:
   - Score trends
   - Learning progress
   - Performance graphs

2. **Save Your Best Models**
   - Models are auto-saved every 100K steps
   - Keep models that show good performance
   - Test different models before competing

## Next Steps

- Once your agent consistently scores 15+ points, you're ready for [NEAR Integration](near-integration.md)
- Learn about [Competition Rules](competition.md)
- Join our community to share training tips!
