use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::serde::{Deserialize, Serialize};
use near_sdk::collections::{LookupMap, UnorderedMap};
use near_sdk::{env, near_bindgen, AccountId, PanicOnDefault, Promise, BorshStorageKey};
use near_sdk::json_types::U128;
use near_sdk::schemars::{self, JsonSchema};

pub type Balance = U128;

#[derive(BorshStorageKey, BorshSerialize)]
pub enum StorageKey {
    Stakes,
    Leaderboard,
    GameConfigs,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, JsonSchema)]
#[serde(crate = "near_sdk::serde")]
pub struct GameConfig {
    pub min_score: i32,
    pub max_score: i32,
    pub min_stake: Balance,
    pub max_multiplier: u32,
    pub enabled: bool,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, JsonSchema)]
#[serde(crate = "near_sdk::serde")]
pub struct StakeInfo {
    pub game: String,
    pub amount: Balance,
    pub target_score: i32,
    pub timestamp: u64,
    pub games_played: u32,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize, JsonSchema)]
#[serde(crate = "near_sdk::serde")]
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

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct AgentArcadeContract {
    pub owner_id: AccountId,
    pub pool_balance: Balance,
    pub stakes: LookupMap<AccountId, StakeInfo>,
    pub leaderboard: UnorderedMap<AccountId, LeaderboardEntry>,
    pub game_configs: UnorderedMap<String, GameConfig>,
}

#[near_bindgen]
impl AgentArcadeContract {
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

    #[payable]
    pub fn place_stake(&mut self, game: String, target_score: i32) {
        let account_id = env::predecessor_account_id();
        let stake_amount = U128(env::attached_deposit());
        
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
        let game_config = self.game_configs.get(&stake_info.game)
            .expect("Game config not found");
        
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
        self.update_leaderboard(&account_id, &stake_info.game, achieved_score, reward);
        
        // Remove stake
        self.stakes.remove(&account_id);

        if reward.0 > 0 {
            self.pool_balance = U128(self.pool_balance.0 - reward.0);
            Promise::new(account_id).transfer(reward.0)
        } else {
            Promise::new(account_id)
        }
    }

    fn update_leaderboard(&mut self, account_id: &AccountId, game: &String, score: i32, earned: Balance) {
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
        
        // Update highest reward multiplier
        let current_multiplier = self.get_reward_multiplier(game, score, 0);
        entry.highest_reward_multiplier = std::cmp::max(
            entry.highest_reward_multiplier,
            current_multiplier
        );

        self.leaderboard.insert(account_id, &entry);
    }

    // View methods
    pub fn get_pool_balance(&self) -> Balance {
        self.pool_balance
    }

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