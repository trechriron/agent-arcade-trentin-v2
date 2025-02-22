# Agent Arcade Smart Contract

This smart contract powers the staking and reward system for Agent Arcade, a platform for training and competing with AI agents in classic arcade games using Stable Baselines3, Gymnasium, and the Arcade Learning Environment.

## Contract Architecture

The contract implements a staking and reward system with the following key components:

### Core Components

1. **Game Configuration**

   ```rust
   pub struct GameConfig {
       pub min_score: i32,
       pub max_score: i32,
       pub min_stake: Balance,
       pub max_multiplier: u32,
       pub enabled: bool,
   }
   ```

   - Defines parameters for each supported game
   - Controls minimum stakes and score ranges
   - Allows games to be enabled/disabled

2. **Staking System**

   ```rust
   pub struct StakeInfo {
       pub game: String,
       pub amount: Balance,
       pub target_score: i32,
       pub timestamp: u64,
       pub games_played: u32,
   }
   ```

   - Tracks user stakes and performance targets
   - Manages stake timing and game attempts

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
   - Historical earnings and statistics

## Security Features

1. **Owner Controls**
   - Only contract owner can register/modify games
   - Protected game configuration updates
   - Controlled stake validation

2. **Stake Protection**
   - Minimum stake requirements
   - Score range validation
   - Timestamp tracking for attempts

3. **Fund Safety**
   - Protected pool balance
   - Secure reward distribution
   - Validated claim process

## Contract Functions

### Administrative Functions

1. **Initialize Contract**

   ```rust
   pub fn new(owner_id: AccountId) -> Self
   ```

   - Creates new contract instance
   - Sets contract owner
   - Initializes storage collections

2. **Register Game**

   ```rust
   pub fn register_game(
       &mut self, 
       game: String, 
       min_score: i32, 
       max_score: i32, 
       min_stake: Balance, 
       max_multiplier: u32
   )
   ```

   - Registers new game configuration
   - Sets score ranges and stake requirements
   - Owner-only function

### User Functions

1. **Place Stake**

   ```rust
   pub fn place_stake(&mut self, game: String, target_score: i32)
   ```

   - Users stake NEAR on performance target
   - Validates stake amount and target score
   - Records stake information

2. **Submit Score**

   ```rust
   pub fn submit_score(&mut self, game: String, score: i32)
   ```

   - Records game performance
   - Updates leaderboard
   - Triggers reward calculation

3. **Claim Rewards**

   ```rust
   pub fn claim_reward(&mut self)
   ```

   - Distributes earned rewards
   - Updates stake status
   - Manages pool balance

### View Functions

1. **Get Game Configuration**

   ```rust
   pub fn get_game_config(&self, game: String) -> GameConfig
   ```

   - Returns game parameters
   - Validates game existence

2. **View Leaderboard**

   ```rust
   pub fn get_leaderboard(&self, game: String, from_index: u64, limit: u64)
   ```

   - Returns paginated leaderboard
   - Filters by game

3. **Get Stake Information**

   ```rust
   pub fn get_stake(&self, account_id: AccountId) -> Option<StakeInfo>
   ```

   - Returns user stake details
   - Validates stake existence

## Reward System

### Multiplier Calculation

- Score ≥ Target: 3x stake
- Score ≥ 75% Target: 2x stake
- Score ≥ 50% Target: 1.5x stake
- Score < 50% Target: Stake goes to pool

### Pool Management

- Initial pool: 100 NEAR
- Minimum stake: 0.1 NEAR
- Maximum multiplier: 3x
- Weekly distribution

## Development and Testing

### Prerequisites

- Rust 1.64.0 or later
- NEAR CLI
- WASM target support

### Build Instructions

```bash
# Add WASM target
rustup target add wasm32-unknown-unknown

# Build contract
cargo build --target wasm32-unknown-unknown --release
```

### Test Commands

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --features integration-tests
```

## Deployment

See [NEAR Integration Guide](../docs/near-integration.md) for detailed deployment instructions.

## Security Considerations

1. **Stake Validation**
   - Always verify stake amounts
   - Validate score ranges
   - Check game enablement

2. **Access Control**
   - Use owner-only functions appropriately
   - Validate account permissions
   - Check function access

3. **Fund Management**
   - Verify reward calculations
   - Ensure pool balance
   - Validate claim conditions

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

This contract is licensed under MIT. See the LICENSE file for details.
