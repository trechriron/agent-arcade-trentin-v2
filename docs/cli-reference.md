# Agent Arcade CLI Reference

The Agent Arcade CLI provides a comprehensive set of commands for training AI agents, participating in competitions, and managing stakes on the NEAR blockchain.

## Table of Contents

- [Installation](#installation)
- [Core Commands](#core-commands)
- [Training Workflow](#training-workflow)
- [Evaluation Workflow](#evaluation-workflow)
- [Staking Workflow](#staking-workflow)
- [Pool Management](#pool-management)
- [Leaderboard Commands](#leaderboard-commands)
- [Wallet Management](#wallet-management)

## Installation

```bash
# Install base package
pip install -e .

# Install with staking support
pip install -e ".[staking]"
```

## Core Commands

### List Available Games

```bash
agent-arcade list-games
```

Lists all available games with their descriptions, versions, and staking status.

**Output Format:**

```bash
Available Games:
--------------------------------------------------------------------------------
Name                Description                               Version   Staking   
--------------------------------------------------------------------------------
pong                Classic paddle vs paddle game            1.0       ✓        
space_invaders      Defend Earth from alien invasion        1.0       ✓        
river_raid          Control jet and manage fuel             1.0       ✓        
```

## Training Workflow

### Train an Agent

```bash
agent-arcade train <game> [OPTIONS]
```

**Options:**

- `--render/--no-render`: Show training visualization (default: False)
- `--config PATH`: Custom configuration file
- `--output-dir PATH`: Model save location (default: models/<game>)
- `--checkpoint-freq N`: Save frequency in steps (default: 100000)

**Example:**

```bash
# Train with default settings
agent-arcade train pong

# Train with custom config
agent-arcade train space_invaders --config models/space_invaders/custom_config.yaml
```

**Default Paths:**

- Configuration: `models/<game>/config.yaml`
- Output: `models/<game>/final_model.zip`
- Checkpoints: `models/<game>/checkpoints/`

## Evaluation Workflow

### Evaluate a Model

```bash
agent-arcade evaluate <game> <model-path> [OPTIONS]
```

**Options:**

- `--episodes N`: Number of evaluation episodes (default: 100)
- `--render/--no-render`: Show evaluation visualization (default: False)
- `--record/--no-record`: Save evaluation videos (default: False)

**Example:**

```bash
# Basic evaluation
agent-arcade evaluate pong models/pong/final_model.zip

# Detailed evaluation with video recording
agent-arcade evaluate space_invaders models/space_invaders/final_model.zip --episodes 200 --record
```

**Output Includes:**

- Mean score and standard deviation
- Success rate
- Staking thresholds
- Recommended stake targets
- Current leaderboard rank

## Staking Workflow

### Place a Stake

```bash
agent-arcade stake place <game> [OPTIONS]
```

**Options:**

- `--model PATH`: Path to trained model (required)
- `--amount FLOAT`: Amount to stake in NEAR (required)
- `--target-score FLOAT`: Target score to achieve (required)
- `--evaluate/--no-evaluate`: Run evaluation before staking (default: True)

**Example:**

```bash
agent-arcade stake place pong --model models/pong/final_model.zip --amount 1 --target-score 15
```

**Pre-stake Evaluation:**

- Quick performance check (20 episodes)
- Target score validation
- Potential reward calculation
- Risk assessment warnings

### View Current Stake

```bash
agent-arcade stake view
```

Shows details of your active stake:

- Game name
- Staked amount
- Target score
- Games played
- Stake placement time

### Submit Score

```bash
agent-arcade stake submit <game> <score>
```

**Example:**

```bash
agent-arcade stake submit pong 18
```

**Validations:**

- Active stake existence
- Game matching
- Score range validation

## Pool Management

### Fund Pool (Owner Only)

```bash
agent-arcade pool fund <amount>
```

**Example:**

```bash
agent-arcade pool fund 100  # Adds 100 NEAR to pool
```

### Check Pool Balance

```bash
agent-arcade pool balance
```

Shows current pool balance in NEAR.

## Leaderboard Commands

### View Top Scores

```bash
agent-arcade leaderboard top <game> [OPTIONS]
```

**Options:**

- `--limit N`: Number of entries to show (default: 10)

### View Recent Scores

```bash
agent-arcade leaderboard recent <game> [OPTIONS]
```

**Options:**

- `--limit N`: Number of entries to show (default: 10)

### View Player Stats

```bash
agent-arcade leaderboard player <game>
```

Shows your performance statistics:

- Best score
- Success rate
- Current rank
- Total games played

### View Global Stats

```bash
agent-arcade leaderboard stats
```

Shows global statistics:

- Total players
- Total entries
- Per-game statistics

## Wallet Management

### Login to NEAR Wallet

```bash
agent-arcade wallet-cmd login [OPTIONS]
```

**Options:**

- `--network`: NEAR network to use (default: testnet)
- `--account-id`: Specific account ID to use

### Check Wallet Status

```bash
agent-arcade wallet-cmd status
```

Shows:

- Login status
- Account ID
- Current balance
- Network

### Logout

```bash
agent-arcade wallet-cmd logout
```

## Complete Workflow Example

1. **Train an Agent**

   ```bash
   agent-arcade train pong
   ```

2. **Evaluate Performance**

   ```bash
   agent-arcade evaluate pong models/pong/final_model.zip --episodes 100
   ```

3. **Place a Stake**

   ```bash
   agent-arcade stake place pong --model models/pong/final_model.zip --amount 1 --target-score 15
   ```

4. **Monitor Progress**

   ```bash
   agent-arcade stake view
   agent-arcade leaderboard player pong
   ```

5. **Submit Score**

   ```bash
   agent-arcade stake submit pong 18
   ```

## Environment Variables

- `NEAR_ENV`: Network selection (testnet/mainnet)
- `NEAR_ACCOUNT_ID`: Default account ID
- `AGENT_ARCADE_LOG_LEVEL`: Logging level (INFO/DEBUG/WARNING)

## Error Handling

The CLI provides detailed error messages for common issues:

- Network connectivity problems
- Invalid stake parameters
- Insufficient funds
- Model loading errors
- Contract interaction failures

## Best Practices

1. **Training:**
   - Start with default configurations
   - Use checkpoints for long training sessions
   - Monitor with TensorBoard

2. **Evaluation:**
   - Run multiple evaluation episodes
   - Record videos for verification
   - Check performance consistency

3. **Staking:**
   - Start with small stakes while learning
   - Use the pre-stake evaluation
   - Set realistic target scores
   - Monitor pool balance and rewards
