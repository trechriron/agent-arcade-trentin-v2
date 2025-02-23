//! # Agent Arcade Smart Contract
//! 
//! This contract implements a staking and reward system for AI agents competing in classic arcade games.
//! It manages game configurations, user stakes, and a global leaderboard system.
//! 
//! ## Architecture
//! 
//! The contract uses NEAR's storage collections to maintain:
//! - Game configurations and parameters
//! - User stakes and performance tracking
//! - Global leaderboard and statistics
//! 
//! ## Security
//! 
//! The contract implements several security measures:
//! - Owner-only administrative functions
//! - Stake amount validation
//! - Score range verification
//! - Protected pool balance management

use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::serde::{Deserialize, Serialize};
use near_sdk::collections::{LookupMap, UnorderedMap};
use near_sdk::{env, near_bindgen, AccountId, PanicOnDefault, Promise, BorshStorageKey, NearToken, require, log, Gas};
use near_sdk::json_types::U128;

mod types;

#[cfg(test)]
mod tests {
    mod integration;
    mod upgrade_tests;
}

pub use types::*;

/// Balance type alias for U128 to represent NEAR token amounts
pub type Balance = U128;

/// Storage keys for contract collections
#[derive(BorshStorageKey, BorshSerialize)]
pub enum StorageKey {
    Stakes,
    Leaderboard,
    GameConfigs,
}

/// Configuration parameters for each supported game
#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct GameConfig {
    /// Minimum achievable score for the game
    pub min_score: i32,
    /// Maximum achievable score for the game
    pub max_score: i32,
    /// Minimum stake amount required (in yoctoNEAR)
    pub min_stake: Balance,
    /// Maximum reward multiplier for perfect performance
    pub max_multiplier: u32,
    /// Whether the game is currently enabled for staking
    pub enabled: bool,
    /// Rate limit for placing stakes
    pub rate_limit: RateLimit,
}

/// Information about a user's active stake
#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct StakeInfo {
    /// Game identifier (e.g., "pong", "space_invaders")
    pub game: String,
    /// Staked amount in yoctoNEAR
    pub amount: Balance,
    /// Target score the user aims to achieve
    pub target_score: i32,
    /// Timestamp when the stake was placed
    pub timestamp: u64,
    /// Number of game attempts made
    pub games_played: u32,
    /// Timestamp of last evaluation
    pub last_evaluation: u64,
}

/// Entry in the global leaderboard
#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct LeaderboardEntry {
    /// User's NEAR account ID
    pub account_id: AccountId,
    /// Game identifier
    pub game: String,
    /// Highest score achieved
    pub best_score: i32,
    /// Total NEAR tokens earned
    pub total_earned: Balance,
    /// Total number of games played
    pub games_played: u32,
    /// Percentage of successful stakes
    pub win_rate: f64,
    /// Highest reward multiplier achieved
    pub highest_reward_multiplier: u32,
    /// Timestamp of last game played
    pub last_played: u64,
}

/// Main contract structure
#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct AgentArcadeContract {
    /// Contract owner's account ID
    pub owner_id: AccountId,
    /// Total NEAR tokens in the reward pool
    pub pool_balance: Balance,
    /// Mapping of account IDs to their active stakes
    pub stakes: LookupMap<AccountId, StakeInfo>,
    /// Global leaderboard entries
    pub leaderboard: UnorderedMap<AccountId, LeaderboardEntry>,
    /// Game configurations
    pub game_configs: UnorderedMap<String, GameConfig>,
    /// Contract version
    pub version: String,
    /// Contract paused state
    pub paused: bool,
}

/// Old contract state for migration
#[derive(BorshDeserialize, BorshSerialize)]
pub struct OldAgentArcadeContract {
    pub owner_id: AccountId,
    pub pool_balance: U128,
    pub stakes: LookupMap<AccountId, StakeInfo>,
    pub leaderboard: UnorderedMap<AccountId, LeaderboardEntry>,
    pub game_configs: UnorderedMap<String, GameConfig>,
}

#[near_bindgen]
impl AgentArcadeContract {
    /// Initializes the contract with an owner account
    /// 
    /// # Arguments
    /// * `owner_id` - The account ID that will have administrative privileges
    #[init]
    pub fn new(owner_id: AccountId) -> Self {
        require!(!env::state_exists(), "Already initialized");
        Self {
            owner_id,
            paused: false,
            version: env!("CARGO_PKG_VERSION").to_string(),
            stakes: LookupMap::new(StorageKey::Stakes),
            game_configs: UnorderedMap::new(StorageKey::GameConfigs),
            leaderboard: UnorderedMap::new(StorageKey::Leaderboard),
            pool_balance: U128(0),
        }
    }

    /// Security check for owner-only functions
    fn assert_owner(&self) {
        require!(env::predecessor_account_id() == self.owner_id, "Unauthorized: owner only");
    }

    /// Emergency pause for contract functions
    pub fn emergency_pause(&mut self) {
        self.assert_owner();
        require!(!self.paused, "Contract is already paused");
        self.paused = true;
        log!("Contract paused by owner");
    }

    /// Resume contract functions
    pub fn resume(&mut self) {
        self.assert_owner();
        require!(self.paused, "Contract is not paused");
        self.paused = false;
        log!("Contract resumed by owner");
    }

    /// Register or update a game configuration
    #[payable]
    pub fn register_game(
        &mut self,
        game: String,
        min_score: i32,
        max_score: i32,
        min_stake: U128,
        max_multiplier: u32,
        rate_limit: RateLimit,
    ) {
        self.assert_owner();
        require!(!self.paused, "Contract is paused");
        
        self.game_configs.insert(&game, &GameConfig {
            min_score,
            max_score,
            min_stake,
            max_multiplier,
            enabled: true,
            rate_limit,
        });
        
        log!("Game {} registered with config: min_score={}, max_score={}, min_stake={}, max_multiplier={}", 
            game, min_score, max_score, min_stake.0, max_multiplier);
    }

    /// Place a stake on achieving a target score
    #[payable]
    pub fn place_stake(&mut self, game: String, target_score: i32) {
        require!(!self.paused, "Contract is paused");
        let account_id = env::predecessor_account_id();
        let stake_amount = U128(env::attached_deposit().as_yoctonear());
        
        // Validate game and stake
        let game_config = self.game_configs.get(&game)
            .expect("Game not registered");
        require!(game_config.enabled, "Game is disabled");
        require!(stake_amount.0 >= game_config.min_stake.0, "Stake amount too low");
        require!(
            target_score >= game_config.min_score && target_score <= game_config.max_score,
            "Invalid target score"
        );
        
        // Check rate limits
        let current_time = env::block_timestamp();
        if let Some(existing_stake) = self.stakes.get(&account_id) {
            require!(
                current_time - existing_stake.last_evaluation 
                >= game_config.rate_limit.min_hours_between_stakes as u64 * 3600_000_000_000,
                "Rate limit: Too soon to place new stake"
            );
        }

        // Record stake
        self.stakes.insert(&account_id, &StakeInfo {
            game: game.clone(),
            amount: stake_amount,
            target_score,
            timestamp: current_time,
            games_played: 0,
            last_evaluation: current_time,
        });
        
        self.pool_balance = U128(self.pool_balance.0 + stake_amount.0);
        log!("Stake placed: {} NEAR on {} with target score {}", 
            stake_amount.0 as f64 / 1e24, game, target_score);
    }

    /// Calculate reward multiplier based on achievement
    fn get_reward_multiplier(&self, game: &String, achieved_score: i32, target_score: i32) -> u32 {
        let config = self.game_configs.get(game).expect("Game not found");
        
        let score_ratio = (achieved_score as f64 / target_score as f64).abs();
        if score_ratio >= 1.0 {
            config.max_multiplier
        } else if score_ratio >= 0.8 {
            config.max_multiplier / 2
        } else if score_ratio >= 0.5 {
            config.max_multiplier / 4
        } else {
            0
        }
    }

    /// Submit and evaluate a stake result
    #[payable]
    pub fn evaluate_stake(&mut self, achieved_score: i32) -> Promise {
        require!(!self.paused, "Contract is paused");
        let account_id = env::predecessor_account_id();
        let stake_info = self.stakes.get(&account_id).expect("No active stake");
        
        // Check rate limits
        let current_time = env::block_timestamp();
        let game_config = self.game_configs.get(&stake_info.game).expect("Game config not found");
        require!(
            (current_time - stake_info.last_evaluation) >= 3600_000_000_000 / game_config.rate_limit.max_evaluations_per_hour as u64,
            "Rate limit: Too many evaluations per hour"
        );
        
        // Calculate reward
        let multiplier = self.get_reward_multiplier(
            &stake_info.game,
            achieved_score,
            stake_info.target_score
        );
        let reward = if multiplier > 0 {
            U128(stake_info.amount.0 * multiplier as u128)
        } else {
            U128(0)
        };

        // Update leaderboard
        self.update_leaderboard(
            &account_id, 
            &stake_info.game, 
            achieved_score, 
            stake_info.target_score, 
            reward
        );
        
        // Remove stake and update pool
        self.stakes.remove(&account_id);
        if reward.0 > 0 {
            require!(reward.0 <= self.pool_balance.0, "Insufficient pool balance");
            self.pool_balance = U128(self.pool_balance.0 - reward.0);
            log!("Stake evaluated: Score {} achieved, reward {} NEAR", 
                achieved_score, reward.0 as f64 / 1e24);
            Promise::new(account_id).transfer(NearToken::from_yoctonear(reward.0))
        } else {
            log!("Stake evaluated: Score {} achieved, no reward", achieved_score);
            Promise::new(account_id)
        }
    }

    /// Update leaderboard with evaluation results
    fn update_leaderboard(
        &mut self, 
        account_id: &AccountId, 
        game: &String, 
        score: i32, 
        target_score: i32, 
        earned: U128
    ) {
        let mut entry = self.leaderboard.get(account_id).unwrap_or(LeaderboardEntry {
            account_id: account_id.clone(),
            game: game.clone(),
            best_score: score,
            total_earned: U128(0),
            games_played: 0,
            win_rate: 0.0,
            highest_reward_multiplier: 0,
            last_played: env::block_timestamp(),
        });

        // Update stats
        entry.best_score = std::cmp::max(entry.best_score, score);
        entry.total_earned = U128(entry.total_earned.0 + earned.0);
        entry.games_played += 1;
        entry.last_played = env::block_timestamp();
        
        // Calculate win rate
        let game_config = self.game_configs.get(game).expect("Game config not found");
        let win_threshold = (game_config.max_score as f64 * 0.7) as i32;
        let wins = if score >= win_threshold { 1 } else { 0 };
        entry.win_rate = ((entry.win_rate * (entry.games_played - 1) as f64) + wins as f64) 
            / entry.games_played as f64;
        
        // Update highest reward multiplier
        let current_multiplier = self.get_reward_multiplier(game, score, target_score);
        entry.highest_reward_multiplier = std::cmp::max(
            entry.highest_reward_multiplier,
            current_multiplier
        );

        self.leaderboard.insert(account_id, &entry);
    }

    /// Funds the reward pool with additional NEAR
    /// 
    /// # Arguments
    /// None - amount is determined by attached deposit
    /// 
    /// # Returns
    /// * `U128` - New pool balance after funding
    /// 
    /// # Security
    /// * Only callable by contract owner
    /// * Requires attached deposit
    #[payable]
    pub fn fund_pool(&mut self) -> U128 {
        // Only owner can fund pool
        self.assert_owner();
        
        // Get attached deposit
        let deposit = U128(env::attached_deposit().as_yoctonear());
        require!(deposit.0 > 0, "Deposit required to fund pool");

        // Update pool balance
        self.pool_balance = U128(self.pool_balance.0 + deposit.0);
        
        // Log the funding event
        log!("Pool funded with {} NEAR. New balance: {} NEAR", 
            deposit.0 as f64 / 1e24,
            self.pool_balance.0 as f64 / 1e24
        );

        self.pool_balance
    }

    /// Returns the current pool balance
    /// 
    /// # Returns
    /// * `U128` - Current pool balance in yoctoNEAR
    pub fn get_pool_balance(&self) -> U128 {
        self.pool_balance
    }

    // View methods
    pub fn get_stake(&self, account_id: AccountId) -> Option<StakeInfo> {
        self.stakes.get(&account_id)
    }

    pub fn get_game_config(&self, game: String) -> Option<GameConfig> {
        self.game_configs.get(&game)
    }

    pub fn get_leaderboard(&self, game: String, from_index: u64, limit: u64) -> Vec<LeaderboardEntry> {
        self.leaderboard
            .values()
            .filter(|entry| entry.game == game)
            .skip(from_index as usize)
            .take(limit as usize)
            .collect()
    }

    pub fn get_top_players(&self, game: String, limit: u64) -> Vec<LeaderboardEntry> {
        let mut entries: Vec<_> = self.leaderboard
            .values()
            .filter(|entry| entry.game == game)
            .collect();
        entries.sort_by(|a, b| b.best_score.cmp(&a.best_score));
        entries.into_iter().take(limit as usize).collect()
    }

    pub fn get_player_stats(&self, account_id: AccountId, game: String) -> Option<LeaderboardEntry> {
        self.leaderboard
            .get(&account_id)
            .filter(|entry| entry.game == game)
    }

    /// Upgrades the contract code and migrates state if necessary
    /// Only callable by contract owner
    #[private]
    pub fn upgrade(&mut self, code: Vec<u8>) -> Promise {
        require!(env::predecessor_account_id() == self.owner_id, "Only owner can upgrade");
        require!(!self.paused, "Contract must not be paused during upgrade");
        
        // Store current state for migration
        let state_data = borsh::to_vec(self).unwrap();
        env::storage_write(b"STATE_BACKUP", &state_data);
        
        // Deploy new code
        Promise::new(env::current_account_id())
            .deploy_contract(code)
            .function_call(
                "migrate".to_string(),
                Vec::new(),
                NearToken::from_yoctonear(0),
                Gas::from_tgas(30) // Use 30 TGas for migration
            )
    }

    /// Migrates contract state after code upgrade
    /// Only callable by the contract itself during upgrade
    #[private]
    #[init(ignore_state)]
    pub fn migrate() -> Self {
        // Ensure called by the contract itself
        require!(
            env::predecessor_account_id() == env::current_account_id(),
            "Migration can only be called by the contract"
        );

        // Load backed up state
        let state_data = env::storage_read(b"STATE_BACKUP")
            .expect("No state backup found");
        
        // Try to deserialize as current version
        if let Ok(current_contract) = borsh::from_slice::<Self>(&state_data) {
            // No migration needed
            current_contract
        } else {
            // Attempt to deserialize as old version and migrate
            let old_contract = borsh::from_slice::<OldAgentArcadeContract>(&state_data)
                .expect("Failed to deserialize old contract state");
            
            // Migrate to new version
            let new_contract = Self {
                owner_id: old_contract.owner_id,
                pool_balance: old_contract.pool_balance,
                stakes: old_contract.stakes,
                leaderboard: old_contract.leaderboard,
                game_configs: old_contract.game_configs,
                version: env!("CARGO_PKG_VERSION").to_string(),
                paused: false,
            };

            // Clean up backup
            env::storage_remove(b"STATE_BACKUP");

            new_contract
        }
    }

    /// Returns the current contract version
    pub fn get_version(&self) -> String {
        self.version.clone()
    }

    /// Checks if a contract upgrade is needed
    pub fn needs_upgrade(&self) -> bool {
        self.version != env!("CARGO_PKG_VERSION")
    }
} 