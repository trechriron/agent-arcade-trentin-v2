use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::serde::{Deserialize, Serialize};
use near_sdk::{AccountId, json_types::U128};

/// Information about a user's active stake
#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct StakeInfo {
    /// Game identifier (e.g., "pong", "space_invaders")
    pub game: String,
    /// Staked amount in yoctoNEAR
    pub amount: U128,
    /// Target score the user aims to achieve
    pub target_score: i32,
    /// Timestamp when the stake was placed
    pub timestamp: u64,
    /// Number of game attempts made
    pub games_played: u32,
    /// Last evaluation timestamp
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
    pub total_earned: U128,
    /// Total number of games played
    pub games_played: u32,
    /// Percentage of successful stakes
    pub win_rate: f64,
    /// Highest reward multiplier achieved
    pub highest_reward_multiplier: u32,
    /// Timestamp of last game played
    pub last_played: u64,
} 