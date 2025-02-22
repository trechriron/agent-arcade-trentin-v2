# Competition Guide

This guide covers everything you need to know about participating in Agent Arcade competitions, including NEAR blockchain integration and competition rules.

## NEAR Integration

### Prerequisites

1. **NEAR Account**
   - Create testnet account at [NEAR Wallet](https://wallet.testnet.near.org)
   - Install [NEAR CLI](https://docs.near.org/tools/near-cli)
   - Configure local environment

2. **CLI Setup**

   ```bash
   # Login to your NEAR account
   agent-arcade wallet-cmd login
   
   # Verify connection
   agent-arcade wallet-cmd status
   ```

### Smart Contract Interaction

1. **View Contract**

   ```bash
   # Check contract status
   agent-arcade contract view
   
   # View pool balance
   agent-arcade pool balance
   ```

2. **Game Registration**

   ```bash
   # Register game configuration
   agent-arcade contract register-game pong \
     --min-stake 0.1 \
     --max-stake 10 \
     --min-score 0 \
     --max-score 21
   ```

## Competition Rules

### Participation Requirements

1. **Eligibility**
   - Valid NEAR testnet account
   - Minimum stake requirement
   - Compliant agent implementation

2. **Technical Requirements**
   - Use provided training environment
   - No external data/pre-training
   - Pass automated verification

### Staking System

1. **Basic Rules**
   - Minimum stake: 0.1 NEAR
   - Maximum stake: 10 NEAR
   - 24-hour evaluation period
   - Performance-based rewards

2. **Reward Calculation**

   ```bash
   reward = stake_amount * (achieved_score / target_score) * multiplier
   ```

   - Multiplier ranges: 1.0x - 3.0x
   - Score thresholds vary by game

### Game-Specific Rules

1. **Pong**
   - Score range: 0-21
   - Minimum success: 15 points (71% win rate)
   - Maximum reward: 3x stake
   - ALE Benchmarks:
     - Random Agent: -20.7 points
     - Human Average: 9.3 points
     - Human Expert: 14.6 points
     - DQN Baseline: 18.9 points
   - Performance Tiers:
     - Bronze (1x): 8-11 points
     - Silver (1.5x): 12-14 points
     - Gold (2x): 15-17 points
     - Platinum (3x): 18+ points

2. **Space Invaders**
   - Score range: 0-1000
   - Minimum success: 500 points
   - Maximum reward: 3x stake
   - ALE Benchmarks:
     - Random Agent: 148.0 points
     - Human Average: 671.0 points
     - Human Expert: 1652.0 points
     - DQN Baseline: 1076.0 points
   - Performance Tiers:
     - Bronze (1x): 300-499 points
     - Silver (1.5x): 500-749 points
     - Gold (2x): 750-999 points
     - Platinum (3x): 1000+ points
   - Special Considerations:
     - Consistent shield usage
     - Efficient bunker management
     - Strategic enemy prioritization

3. **River Raid**
   - Score range: 0-15000
   - Minimum success: 10000 points
   - Maximum reward: 3x stake
   - ALE Benchmarks:
     - Random Agent: 1338.5 points
     - Human Average: 5981.9 points
     - Human Expert: 13513.3 points
     - DQN Baseline: 8627.5 points
   - Performance Tiers:
     - Bronze (1x): 5000-7499 points
     - Silver (1.5x): 7500-9999 points
     - Gold (2x): 10000-12499 points
     - Platinum (3x): 12500+ points
   - Special Considerations:
     - Fuel management efficiency
     - Bridge navigation success rate
     - Enemy avoidance patterns

### Performance Metrics

1. **Core Metrics**
   - Episode Score: Raw game score
   - Success Rate: % of episodes above minimum success threshold
   - Stability: Standard deviation across episodes
   - Completion Rate: % of episodes reaching target length

2. **Advanced Metrics**
   - Action Efficiency: Ratio of successful to total actions
   - Resource Management: Game-specific resource usage
   - Strategic Patterns: Recognized gameplay patterns
   - Learning Progress: Score improvement over episodes

3. **Evaluation Requirements**
   - Minimum 100 evaluation episodes
   - Fixed frame skip (4 frames)
   - No stochastic frame skipping
   - Deterministic evaluation
   - Standardized preprocessing:
     - Grayscale conversion
     - Frame stacking (4 frames)
     - 84x84 resolution
     - Reward clipping [-1, 1]

### Evaluation Process

1. **Initial Evaluation**

   ```bash
   # Run comprehensive evaluation
   agent-arcade evaluate <game> <model-path> --episodes 100 --record
   ```

2. **Performance Verification**
   - Multiple evaluation runs
   - Statistical significance checks
   - Video recording for verification
   - Automated pattern analysis

3. **Submission Requirements**
   - Model checksum verification
   - Environment version matching
   - Configuration validation
   - Performance consistency check

## Participation Workflow

### 1. Training Phase

```bash
# Train your agent
agent-arcade train pong --config models/pong/config.yaml

# Evaluate performance
agent-arcade evaluate pong models/pong/final_model.zip --episodes 100
```

### 2. Staking Phase

```bash
# Place stake
agent-arcade stake place pong \
  --model models/pong/final_model.zip \
  --amount 1 \
  --target-score 18

# Monitor stake status
agent-arcade stake view
```

### 3. Evaluation Phase

```bash
# Submit evaluation score
agent-arcade stake submit pong 18

# View leaderboard
agent-arcade leaderboard top pong
```

## Reward Distribution

### Calculation Example

1. **Basic Scenario**
   - Stake: 1 NEAR
   - Target: 18 points
   - Achieved: 15 points
   - Reward: 1 *(15/18)* 1.5 = 1.25 NEAR

2. **Maximum Reward**
   - Stake: 10 NEAR
   - Perfect score
   - 3x multiplier
   - Reward: 30 NEAR

### Claiming Rewards

```bash
# View available rewards
agent-arcade rewards view

# Claim rewards
agent-arcade rewards claim
```

## Best Practices

1. **Risk Management**
   - Start with small stakes
   - Test thoroughly before staking
   - Monitor performance regularly

2. **Performance Optimization**
   - Use evaluation metrics
   - Record validation episodes
   - Document training process

3. **Fair Play**
   - Follow provided guidelines
   - Report bugs responsibly
   - Maintain code integrity

## Troubleshooting

### Common Issues

1. **Stake Placement**
   - Insufficient funds
   - Invalid target score
   - Model validation errors

2. **Score Submission**
   - Network connectivity
   - Transaction failures
   - Validation timeouts

3. **Reward Claims**
   - Pending evaluations
   - Contract errors
   - Account issues

### Support Channels

1. **Technical Support**
   - GitHub Issues
   - Discord Community
   - Documentation

2. **Contract Issues**
   - NEAR Explorer
   - Transaction logs
   - Support tickets
