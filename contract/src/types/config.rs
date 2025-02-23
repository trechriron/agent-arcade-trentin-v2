use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::serde::{Deserialize, Serialize};
use near_sdk::json_types::U128;

/// Configuration parameters for each supported game
#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct GameConfig {
    /// Minimum achievable score for the game
    pub min_score: i32,
    /// Maximum achievable score for the game
    pub max_score: i32,
    /// Minimum stake amount required (in yoctoNEAR)
    pub min_stake: U128,
    /// Maximum reward multiplier for perfect performance
    pub max_multiplier: u32,
    /// Whether the game is currently enabled for staking
    pub enabled: bool,
    /// Rate limiting configuration
    pub rate_limit: RateLimit,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct RateLimit {
    /// Maximum evaluations per hour
    pub max_evaluations_per_hour: u32,
    /// Minimum hours between stakes
    pub min_hours_between_stakes: u32,
} 