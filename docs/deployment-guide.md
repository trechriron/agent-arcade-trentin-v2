# Agent Arcade Contract Deployment Guide

This guide documents the deployment process for the Agent Arcade smart contract, which manages staking and competitions for AI game agents.

## Contract Overview

The Agent Arcade contract implements:

- Game configuration management
- Staking system with rate limiting
- Reward pool management
- Global leaderboard tracking
- Security features and access control

## Prerequisites

- Rust toolchain with `wasm32-unknown-unknown` target
- NEAR CLI RS (v0.18.0 or later)
- NEAR testnet/mainnet account

## Build Process

1. **Build Contract**

```bash
# Navigate to contract directory
cd contract

# Build for WASM target
cargo build --target wasm32-unknown-unknown --release
```

The WASM file will be generated at `target/wasm32-unknown-unknown/release/agent_arcade.wasm`

## Testnet Deployment

### 1. Account Setup

```bash
# Login to NEAR testnet
near login

# Note: This will open a browser window for authorization
# Use your account ID when prompted
```

### 2. Deploy Contract

```bash
# Deploy the WASM file
near deploy near-agent-arcade.testnet ./target/wasm32-unknown-unknown/release/agent_arcade.wasm
```

### 3. Initialize Contract

```bash
# Initialize with owner account
near contract call-function as-transaction near-agent-arcade.testnet new \
  json-args '{"owner_id": "near-agent-arcade.testnet"}' \
  prepaid-gas '30.0 Tgas' \
  attached-deposit '0 NEAR' \
  sign-as near-agent-arcade.testnet \
  network-config testnet \
  sign-with-keychain send
```

## Game Configuration

### Register Games

Each game requires specific configuration. Here are our current game configurations:

1. **Pong**

```bash
near contract call-function as-transaction near-agent-arcade.testnet register_game \
  json-args '{
    "game": "pong",
    "min_score": 0,
    "max_score": 21,
    "min_stake": "100000000000000000000000",
    "max_multiplier": 3,
    "rate_limit": {
      "max_evaluations_per_hour": 5,
      "min_hours_between_stakes": 24
    }
  }' \
  prepaid-gas '30.0 Tgas' \
  attached-deposit '0.1 NEAR' \
  sign-as near-agent-arcade.testnet \
  network-config testnet \
  sign-with-keychain send
```

2. **Space Invaders**

```bash
near contract call-function as-transaction near-agent-arcade.testnet register_game \
  json-args '{
    "game": "space_invaders",
    "min_score": 0,
    "max_score": 1000,
    "min_stake": "100000000000000000000000",
    "max_multiplier": 3,
    "rate_limit": {
      "max_evaluations_per_hour": 5,
      "min_hours_between_stakes": 24
    }
  }' \
  prepaid-gas '30.0 Tgas' \
  attached-deposit '0.1 NEAR' \
  sign-as near-agent-arcade.testnet \
  network-config testnet \
  sign-with-keychain send
```

3. **River Raid**

```bash
near contract call-function as-transaction near-agent-arcade.testnet register_game \
  json-args '{
    "game": "river_raid",
    "min_score": 0,
    "max_score": 15000,
    "min_stake": "100000000000000000000000",
    "max_multiplier": 3,
    "rate_limit": {
      "max_evaluations_per_hour": 5,
      "min_hours_between_stakes": 24
    }
  }' \
  prepaid-gas '30.0 Tgas' \
  attached-deposit '0.1 NEAR' \
  sign-as near-agent-arcade.testnet \
  network-config testnet \
  sign-with-keychain send
```

## Reward Pool Management

### Fund Pool

```bash
# Add funds to reward pool (example: 1 NEAR)
near contract call-function as-transaction near-agent-arcade.testnet fund_pool \
  json-args '{}' \
  prepaid-gas '30.0 Tgas' \
  attached-deposit '1 NEAR' \
  sign-as near-agent-arcade.testnet \
  network-config testnet \
  sign-with-keychain send
```

### Check Pool Balance

```bash
near contract call-function as-read-only near-agent-arcade.testnet get_pool_balance \
  json-args '{}' \
  network-config testnet \
  now
```

## Staking Operations

### Place Stake

```bash
# Example: Place 0.1 NEAR stake on Pong with target score 15
near contract call-function as-transaction near-agent-arcade.testnet place_stake \
  json-args '{
    "game": "pong",
    "target_score": 15
  }' \
  prepaid-gas '30.0 Tgas' \
  attached-deposit '0.1 NEAR' \
  sign-as near-agent-arcade.testnet \
  network-config testnet \
  sign-with-keychain send
```

### View Stake

```bash
near contract call-function as-read-only near-agent-arcade.testnet get_stake \
  json-args '{
    "account_id": "near-agent-arcade.testnet"
  }' \
  network-config testnet \
  now
```

## Leaderboard System

The leaderboard system operates on two levels:

### 1. Contract Leaderboard

The contract maintains a global leaderboard with the following data per entry:

```rust
pub struct LeaderboardEntry {
    pub account_id: AccountId,    // Player's account
    pub game: String,             // Game identifier
    pub best_score: i32,          // Highest score achieved
    pub total_earned: Balance,    // Total NEAR earned
    pub games_played: u32,        // Total games played
    pub win_rate: f64,            // Success rate
    pub highest_reward_multiplier: u32,  // Best multiplier achieved
    pub last_played: u64,         // Last game timestamp
}
```

The leaderboard is automatically updated when:

- A stake is evaluated successfully
- A player achieves a new high score
- Rewards are distributed

### 2. CLI Leaderboard

The CLI maintains a local leaderboard database in `~/.agent-arcade/leaderboards/` with commands:

```bash
# View top scores for a game
agent-arcade leaderboard top pong --limit 10

# View recent scores
agent-arcade leaderboard recent pong --limit 10

# View player stats
agent-arcade leaderboard player pong

# View global statistics
agent-arcade leaderboard stats
```

### View Player Stats

```bash
near contract call-function as-read-only near-agent-arcade.testnet get_player_stats \
  json-args '{
    "account_id": "near-agent-arcade.testnet",
    "game": "pong"
  }' \
  network-config testnet \
  now
```

### View Top Players

```bash
near contract call-function as-read-only near-agent-arcade.testnet get_top_players \
  json-args '{
    "game": "pong",
    "limit": 10
  }' \
  network-config testnet \
  now
```

## Security Features

The contract implements several security measures:

1. **Rate Limiting**
   - Maximum 5 evaluations per hour per account
   - Minimum 24 hours between stakes
   - Configurable per game

2. **Fund Safety**
   - Maximum reward multiplier: 3x
   - Pool balance validation for rewards
   - Owner-only pool funding

3. **Access Control**
   - Owner-only administrative functions
   - Protected game configuration
   - Secure upgrade process

## Monitoring

### View Game Configuration

```bash
near contract call-function as-read-only near-agent-arcade.testnet get_game_config \
  json-args '{"game": "pong"}' \
  network-config testnet \
  now
```

## Current Status

- ✅ Contract deployed to testnet
- ✅ Games registered with configurations
- ✅ Staking system operational
- ✅ Pool funding implemented
- ✅ Rate limiting configured
- ✅ Leaderboard system active
- ⏳ Security audit pending
