Based on the NEAR documentation and our current implementation, I'll outline a detailed plan for Phase 3B. Let's break it down into structured steps:

## Phase 3B Implementation Plan

### 1. Smart Contract Development (Rust)

First, we need to set up our contract development environment:

```bash
# Install Rust and WASM target
rustup target add wasm32-unknown-unknown

# Install NEAR CLI-RS and cargo-near
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/near/near-cli-rs/releases/latest/download/near-cli-rs-installer.sh | sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/near/cargo-near/releases/latest/download/cargo-near-installer.sh | sh
```

#### A. Contract Structure

```rust
// contract/src/lib.rs
use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::serde::{Deserialize, Serialize};
use near_sdk::{env, near_bindgen, AccountId, Balance, Promise};

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize)]
pub struct AgentArcadeContract {
    // Contract state
    owner_id: AccountId,
    total_stakes: Balance,
    stakes: UnorderedMap<AccountId, Stake>,
    leaderboard: UnorderedMap<String, Vec<LeaderboardEntry>>,
    game_configs: UnorderedMap<String, GameConfig>,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
pub struct Stake {
    amount: Balance,
    target_score: f64,
    timestamp: u64,
    game: String,
    status: StakeStatus,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    account_id: AccountId,
    game: String,
    score: f64,
    stake_amount: Balance,
    reward_multiplier: f64,
    timestamp: u64,
}
```

#### B. Core Contract Methods
```rust
#[near_bindgen]
impl AgentArcadeContract {
    // Staking
    pub fn place_stake(&mut self, game: String, target_score: f64) {
        // Validate stake and record it
    }
    
    pub fn evaluate_stake(&mut self, stake_id: String, achieved_score: f64) {
        // Calculate rewards and update leaderboard
    }
    
    // Leaderboard
    pub fn get_leaderboard(&self, game: String, limit: u64) -> Vec<LeaderboardEntry> {
        // Return top scores
    }
    
    pub fn get_player_stats(&self, account_id: AccountId) -> PlayerStats {
        // Return player statistics
    }
    
    // Pool Management
    pub fn get_pool_stats(&self) -> PoolStats {
        // Return pool statistics
    }
}
```

### 2. Contract Deployment & Testing

```bash
# Build contract
cd contract
cargo build --target wasm32-unknown-unknown --release

# Deploy to testnet
near deploy --accountId agent-arcade.testnet --wasmFile target/wasm32-unknown-unknown/release/agent_arcade.wasm
```

#### A. Test Cases
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_place_stake() {
        // Test stake placement
    }
    
    #[test]
    fn test_evaluate_stake() {
        // Test stake evaluation and rewards
    }
    
    #[test]
    fn test_leaderboard() {
        // Test leaderboard updates
    }
}
```

### 3. Pool Initialization

```typescript
// scripts/initialize_pool.js
const { connect, keyStores, utils } = require("near-api-js");

async function initializePool() {
    const near = await connect({
        networkId: "testnet",
        keyStore: new keyStores.UnencryptedFileSystemKeyStore(),
        nodeUrl: "https://rpc.testnet.near.org"
    });
    
    const account = await near.account("agent-arcade.testnet");
    await account.functionCall({
        contractId: "agent-arcade.testnet",
        methodName: "new",
        args: {
            owner_id: "agent-arcade.testnet",
            initial_pool: utils.format.parseNearAmount("1000")
        }
    });
}
```

### 4. Integration with Client-Side

#### A. Update NEAR Integration
```python
# cli/core/near.py
class NEARContract:
    async def initialize_contract(self):
        """Initialize contract connection"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
        
        try:
            # Connect to contract
            self.contract = await self.wallet.account.load_contract(
                self.contract_id,
                {
                    'viewMethods': ['get_leaderboard', 'get_pool_stats'],
                    'changeMethods': ['place_stake', 'evaluate_stake']
                }
            )
        except Exception as e:
            logger.error(f"Contract initialization failed: {e}")
            return False
```

### 5. Deployment Steps

1. **Contract Deployment**:
```bash
# Deploy contract
near deploy --accountId agent-arcade.testnet \
    --wasmFile contract/target/wasm32-unknown-unknown/release/agent_arcade.wasm \
    --initFunction 'new' \
    --initArgs '{"owner_id": "agent-arcade.testnet"}'
```

2. **Pool Initialization**:
```bash
# Initialize pool with funds
near call agent-arcade.testnet initialize_pool '{"initial_amount": "1000000000000000000000000"}' \
    --accountId agent-arcade.testnet --amount 1000
```

3. **Game Registration**:
```bash
# Register games
near call agent-arcade.testnet register_game '{"game": "pong", "min_score": -21, "max_score": 21}' \
    --accountId agent-arcade.testnet

near call agent-arcade.testnet register_game '{"game": "space-invaders", "min_score": 0, "max_score": 1000}' \
    --accountId agent-arcade.testnet
```

### 6. Testing & Verification

1. **Contract Testing**:
```bash
# Run contract tests
cd contract
cargo test

# Test on testnet
near view agent-arcade.testnet get_pool_stats
near view agent-arcade.testnet get_leaderboard '{"game": "pong", "limit": 10}'
```

2. **Integration Testing**:
```bash
# Test stake placement
agent-arcade stake pong --model models/pong_final.zip --amount 10 --target-score 15

# Test evaluation
agent-arcade evaluate pong <stake-id> --episodes 100
```

### 7. Security Considerations

1. **Access Control**:
   - Owner-only functions for critical operations
   - Stake validation and limits
   - Prevention of double-claiming rewards

2. **Economic Security**:
   - Maximum stake limits
   - Minimum stake requirements
   - Cooldown periods between stakes

3. **Data Integrity**:
   - Validation of scores and rewards
   - Prevention of stake manipulation
   - Secure storage of leaderboard data