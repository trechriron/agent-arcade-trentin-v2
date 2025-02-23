# Agent Arcade Smart Contract

Production-grade smart contract for the Agent Arcade staking and competition platform, deployed at `agent-arcade.near` (mainnet) and `agent-arcade.testnet` (testnet).

## Architecture

The contract implements a secure and upgradeable staking system with the following components:

### Core Components

1. **Game Configuration**

   ```rust
   pub struct GameConfig {
       pub min_score: i32,
       pub max_score: i32,
       pub min_stake: Balance,
       pub max_multiplier: u32,
       pub enabled: bool,
       pub rate_limit: RateLimit,
   }
   ```

   - Configurable game parameters
   - Rate limiting for stakes and evaluations
   - Dynamic score ranges and multipliers

2. **Staking System**

   ```rust
   pub struct StakeInfo {
       pub game: String,
       pub amount: Balance,
       pub target_score: i32,
       pub timestamp: u64,
       pub games_played: u32,
       pub last_evaluation: u64,
   }
   ```

   - Secure stake tracking
   - Rate-limited evaluations
   - Performance-based rewards

3. **Leaderboard**

   ```rust
   pub struct LeaderboardEntry {
       pub account_id: AccountId,
       pub game: String,
       pub best_score: i32,
       pub total_earned: Balance,
       pub games_played: u32,
       pub win_rate: f64,
       pub highest_reward_multiplier: u32,
       pub last_played: u64,
   }
   ```

   - Global performance tracking
   - Historical earnings
   - Player statistics

## Security Features

1. **Access Control**
   - Owner-only administrative functions
   - Protected game configuration
   - Secure upgrade process

2. **Rate Limiting**
   - Maximum evaluations per hour
   - Minimum time between stakes
   - Configurable per game

3. **Fund Safety**
   - Protected pool balance
   - Secure reward distribution
   - Balance validation

4. **Emergency Controls**
   - Contract pause functionality
   - Safe upgrade process
   - State migration support

## Contract Functions

### Administrative

1. **Initialize Contract**

   ```rust
   pub fn new(owner_id: AccountId) -> Self
   ```

2. **Emergency Controls**

   ```rust
   pub fn emergency_pause(&mut self)
   pub fn resume(&mut self)
   ```

3. **Game Management**

   ```rust
   pub fn register_game(
       &mut self,
       game: String,
       min_score: i32,
       max_score: i32,
       min_stake: Balance,
       max_multiplier: u32,
       rate_limit: RateLimit,
   )
   ```

### User Functions

1. **Staking**

   ```rust
   pub fn place_stake(&mut self, game: String, target_score: i32)
   pub fn evaluate_stake(&mut self, achieved_score: i32) -> Promise
   ```

2. **View Functions**

   ```rust
   pub fn get_stake(&self, account_id: AccountId) -> Option<StakeInfo>
   pub fn get_game_config(&self, game: String) -> Option<GameConfig>
   pub fn get_leaderboard(&self, game: String, from_index: u64, limit: u64)
   pub fn get_top_players(&self, game: String, limit: u64)
   pub fn get_player_stats(&self, account_id: AccountId, game: String)
   ```

### Upgrade Infrastructure

1. **Version Management**

   ```rust
   pub fn get_version(&self) -> String
   pub fn needs_upgrade(&self) -> bool
   ```

2. **Upgrade Process**

   ```rust
   pub fn upgrade(code: Vec<u8>) -> Promise
   pub fn migrate() -> Self
   ```

## Development

### Prerequisites

- Rust 1.70.0 or later
- `wasm32-unknown-unknown` target
- NEAR CLI

### Build Instructions

```bash
# Add WASM target
rustup target add wasm32-unknown-unknown

# Build for development
cargo build --target wasm32-unknown-unknown

# Build for production (optimized)
cargo build --target wasm32-unknown-unknown --release

# Generate reproducible build
cargo near build --release
```

### Testing

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --test integration
cargo test --test upgrade_tests
```

## Deployment Process

### 1. Build Contract

```bash
# Build optimized WASM
cargo near build --release
```

### 2. Deploy Contract

```bash
# Deploy to testnet
near deploy agent-arcade.testnet ./target/wasm32-unknown-unknown/release/agent_arcade.wasm

# After testing, deploy to mainnet
near deploy agent-arcade.near ./target/wasm32-unknown-unknown/release/agent_arcade.wasm
```

### 3. Initialize Contract

```bash
# Initialize on testnet
near call agent-arcade.testnet new '{"owner_id": "agent-arcade.testnet"}' --accountId agent-arcade.testnet

# Initialize on mainnet
near call agent-arcade.near new '{"owner_id": "agent-arcade.near"}' --accountId agent-arcade.near
```

## Integration Guide

Developers should use the official contract deployments:

```python
# In cli/core/wallet.py
CONTRACT_ID_MAINNET = "agent-arcade.near"
CONTRACT_ID_TESTNET = "agent-arcade.testnet"
```

## Security Considerations

1. **Rate Limiting**
   - Maximum 5 evaluations per hour
   - Minimum 24h between stakes
   - Configurable per game

2. **Fund Safety**
   - Maximum reward multiplier: 3x
   - Pool balance validation
   - Secure reward distribution

3. **Access Control**
   - Owner-only admin functions
   - Protected upgrade process
   - Secure state migration

## Maintenance

1. **Monitoring**
   - Daily balance checks
   - Weekly performance review
   - Monthly security audit

2. **Updates**
   - Semantic versioning
   - State migration support
   - Backward compatibility

## License

This contract is licensed under MIT. See the LICENSE file for details.
