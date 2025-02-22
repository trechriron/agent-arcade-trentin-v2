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
use near_sdk::{env, near_bindgen, AccountId, PanicOnDefault, Promise, BorshStorageKey, NearToken};
use near_sdk::json_types::U128;

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
}

#[near_bindgen]
impl AgentArcadeContract {
    /// Initializes the contract with an owner account
    /// 
    /// # Arguments
    /// * `owner_id` - The account ID that will have administrative privileges
    #[init]
    pub fn new(owner_id: AccountId) -> Self {
        Self {
            owner_id,
            pool_balance: U128(0),
            stakes: LookupMap::new(StorageKey::Stakes),
            leaderboard: UnorderedMap::new(StorageKey::Leaderboard),
            game_configs: UnorderedMap::new(StorageKey::GameConfigs),
        }
    }

    /// Registers a new game or updates existing game configuration
    /// 
    /// # Arguments
    /// * `game` - Game identifier (e.g., "pong")
    /// * `min_score` - Minimum achievable score
    /// * `max_score` - Maximum achievable score
    /// * `min_stake` - Minimum stake amount in yoctoNEAR
    /// * `max_multiplier` - Maximum reward multiplier
    /// 
    /// # Security
    /// * Only callable by contract owner
    #[payable]
    pub fn register_game(&mut self, game: String, min_score: i32, max_score: i32, min_stake: Balance, max_multiplier: u32) {
        assert_eq!(env::predecessor_account_id(), self.owner_id, "Only owner can register games");
        
        self.game_configs.insert(&game, &GameConfig {
            min_score,
            max_score,
            min_stake,
            max_multiplier,
            enabled: true,
        });
    }

    /// Places a stake on achieving a target score in a game
    /// 
    /// # Arguments
    /// * `game` - Game identifier
    /// * `target_score` - Score the user aims to achieve
    /// 
    /// # Panics
    /// * If game is not registered or disabled
    /// * If stake amount is below minimum
    /// * If target score is outside valid range
    #[payable]
    pub fn place_stake(&mut self, game: String, target_score: i32) {
        let account_id = env::predecessor_account_id();
        let stake_amount = U128(env::attached_deposit().as_yoctonear());
        
        // Get game config
        let game_config = self.game_configs.get(&game)
            .expect("Game not registered");
        assert!(game_config.enabled, "Game is disabled");
        
        // Validate stake
        assert!(stake_amount.0 >= game_config.min_stake.0, "Stake amount too low");
        assert!(
            target_score >= game_config.min_score && target_score <= game_config.max_score,
            "Invalid target score"
        );
        assert!(self.stakes.get(&account_id).is_none(), "Active stake exists");

        // Record stake
        self.stakes.insert(&account_id, &StakeInfo {
            game: game.clone(),
            amount: stake_amount,
            target_score,
            timestamp: env::block_timestamp(),
            games_played: 0,
        });
        self.pool_balance = U128(self.pool_balance.0 + stake_amount.0);
    }

    pub fn get_reward_multiplier(&self, game: &String, achieved_score: i32, target_score: i32) -> u32 {
        let config = self.game_configs.get(game).expect("Game not found");
        
        // Calculate multiplier based on achievement relative to target
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

    #[payable]
    pub fn evaluate_stake(&mut self, achieved_score: i32) -> Promise {
        let account_id = env::predecessor_account_id();
        let stake_info = self.stakes.get(&account_id).expect("No active stake");
        
        // Get game config
        let _game_config = self.game_configs.get(&stake_info.game)
            .expect("Game config not found");
        
        // Calculate reward using the stake's target_score
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

        // Update leaderboard with the correct target score
        self.update_leaderboard(
            &account_id, 
            &stake_info.game, 
            achieved_score, 
            stake_info.target_score, 
            reward
        );
        
        // Remove stake record
        self.stakes.remove(&account_id);

        if reward.0 > 0 {
            self.pool_balance = U128(self.pool_balance.0 - reward.0);
            Promise::new(account_id).transfer(NearToken::from_yoctonear(reward.0))
        } else {
            Promise::new(account_id)
        }
    }

    fn update_leaderboard(
        &mut self, 
        account_id: &AccountId, 
        game: &String, 
        score: i32, 
        target_score: i32, 
        earned: Balance
    ) {
        let mut entry = self.leaderboard.get(account_id).unwrap_or(LeaderboardEntry {
            account_id: account_id.clone(),
            game: game.to_string(),
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
        
        // Calculate win rate based on game-specific criteria
        let game_config = self.game_configs.get(game).expect("Game config not found");
        let win_threshold = (game_config.max_score as f64 * 0.7) as i32;
        let wins = if score >= win_threshold { 1 } else { 0 };
        entry.win_rate = ((entry.win_rate * (entry.games_played - 1) as f64) + wins as f64) 
            / entry.games_played as f64;
        
        // Update highest reward multiplier using the correct target_score
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
        assert_eq!(
            env::predecessor_account_id(),
            self.owner_id,
            "Only owner can fund pool"
        );
        
        // Get attached deposit
        let deposit = U128(env::attached_deposit().as_yoctonear());
        assert!(deposit.0 > 0, "Deposit required to fund pool");

        // Update pool balance
        self.pool_balance = U128(self.pool_balance.0 + deposit.0);
        
        // Log the funding event
        env::log_str(&format!(
            "Pool funded with {} yoctoNEAR. New balance: {}",
            deposit.0,
            self.pool_balance.0
        ));

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
} 