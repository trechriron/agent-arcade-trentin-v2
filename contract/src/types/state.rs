use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::collections::{LookupMap, UnorderedMap};
use near_sdk::{AccountId, BorshStorageKey, PanicOnDefault};
use near_sdk::json_types::U128;

use super::{GameConfig, StakeInfo, LeaderboardEntry};

/// Storage keys for contract collections
#[derive(BorshStorageKey, BorshSerialize)]
pub enum StorageKey {
    Stakes,
    GameConfigs,
    Leaderboard,
    Admins,
}

#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct AgentArcadeContract {
    // Core state
    pub owner_id: AccountId,
    pub paused: bool,
    pub version: String,

    // Storage
    pub stakes: LookupMap<AccountId, StakeInfo>,
    pub game_configs: UnorderedMap<String, GameConfig>,
    pub leaderboard: UnorderedMap<AccountId, LeaderboardEntry>,
    pub pool_balance: U128,
} 