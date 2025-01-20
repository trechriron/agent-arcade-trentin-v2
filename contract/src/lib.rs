use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::serde::{Deserialize, Serialize};
use near_sdk::collections::{LookupMap, UnorderedMap};
use near_sdk::{env, near_bindgen, AccountId, Balance, PanicOnDefault, Promise, BorshStorageKey};

#[derive(BorshStorageKey, BorshSerialize)]
enum StorageKey {
    Stakes,
    Leaderboard,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct StakeInfo {
    pub amount: Balance,
    pub target_score: i32,
    pub timestamp: u64,
    pub games_played: u32,
}

#[derive(BorshDeserialize, BorshSerialize, Serialize, Deserialize)]
#[serde(crate = "near_sdk::serde")]
pub struct LeaderboardEntry {
    pub account_id: AccountId,
    pub best_score: i32,
    pub total_earned: Balance,
    pub games_played: u32,
    pub win_rate: f64,
    pub highest_reward_multiplier: u32,
    pub last_played: u64,
}

#[near_bindgen]
#[derive(BorshDeserialize, BorshSerialize, PanicOnDefault)]
pub struct PongStaking {
    pub owner_id: AccountId,
    pub pool_balance: Balance,
    pub stakes: LookupMap<AccountId, StakeInfo>,
    pub leaderboard: UnorderedMap<AccountId, LeaderboardEntry>,
    pub min_stake: Balance,
    pub max_multiplier: u32,
}

#[near_bindgen]
impl PongStaking {
    #[init]
    pub fn new(owner_id: AccountId, min_stake: Balance, max_multiplier: u32) -> Self {
        Self {
            owner_id,
            pool_balance: 0,
            stakes: LookupMap::new(StorageKey::Stakes),
            leaderboard: UnorderedMap::new(StorageKey::Leaderboard),
            min_stake,
            max_multiplier,
        }
    }

    #[payable]
    pub fn stake(&mut self, target_score: i32) {
        let account_id = env::predecessor_account_id();
        let stake_amount = env::attached_deposit();
        
        assert!(stake_amount >= self.min_stake, "Stake amount too low");
        assert!(target_score >= -21 && target_score <= 21, "Invalid target score");
        assert!(self.stakes.get(&account_id).is_none(), "Active stake exists");

        self.stakes.insert(&account_id, &StakeInfo {
            amount: stake_amount,
            target_score,
            timestamp: env::block_timestamp(),
            games_played: 0,
        });
        self.pool_balance += stake_amount;
    }

    pub fn get_reward_multiplier(&self, achieved_score: i32, target_score: i32) -> u32 {
        if achieved_score >= 15 {
            3
        } else if achieved_score >= 10 {
            2
        } else if achieved_score >= 5 {
            3/2
        } else {
            0
        }
    }

    #[payable]
    pub fn claim_reward(&mut self, achieved_score: i32) -> Promise {
        let account_id = env::predecessor_account_id();
        let stake_info = self.stakes.get(&account_id).expect("No active stake");
        
        // Calculate reward
        let multiplier = self.get_reward_multiplier(achieved_score, stake_info.target_score);
        let reward = if multiplier > 0 {
            stake_info.amount * multiplier as u128
        } else {
            0
        };

        // Update leaderboard
        self.update_leaderboard(&account_id, achieved_score, reward);
        
        // Remove stake
        self.stakes.remove(&account_id);

        if reward > 0 {
            self.pool_balance -= reward;
            Promise::new(account_id).transfer(reward)
        } else {
            Promise::new(account_id)
        }
    }

    fn update_leaderboard(&mut self, account_id: &AccountId, score: i32, earned: Balance) {
        let mut entry = self.leaderboard.get(account_id).unwrap_or(LeaderboardEntry {
            account_id: account_id.clone(),
            best_score: score,
            total_earned: 0,
            games_played: 0,
            win_rate: 0.0,
            highest_reward_multiplier: 0,
            last_played: env::block_timestamp(),
        });

        // Update stats
        entry.best_score = std::cmp::max(entry.best_score, score);
        entry.total_earned += earned;
        entry.games_played += 1;
        entry.last_played = env::block_timestamp();
        
        // Calculate win rate (score >= 0 is considered a win)
        let wins = if score >= 0 { 1 } else { 0 };
        entry.win_rate = ((entry.win_rate * (entry.games_played - 1) as f64) + wins as f64) 
            / entry.games_played as f64;
        
        // Update highest reward multiplier
        let current_multiplier = self.get_reward_multiplier(score, 0);
        entry.highest_reward_multiplier = std::cmp::max(entry.highest_reward_multiplier, current_multiplier);

        self.leaderboard.insert(account_id, &entry);
    }

    // View methods
    pub fn get_pool_balance(&self) -> Balance {
        self.pool_balance
    }

    pub fn get_stake(&self, account_id: AccountId) -> Option<StakeInfo> {
        self.stakes.get(&account_id)
    }

    pub fn get_leaderboard(&self, from_index: u64, limit: u64) -> Vec<LeaderboardEntry> {
        self.leaderboard
            .values()
            .skip(from_index as usize)
            .take(limit as usize)
            .collect()
    }

    // New view methods for leaderboard
    pub fn get_top_players(&self, limit: u64) -> Vec<LeaderboardEntry> {
        let mut entries: Vec<_> = self.leaderboard.values().collect();
        entries.sort_by(|a, b| b.best_score.cmp(&a.best_score));
        entries.into_iter().take(limit as usize).collect()
    }

    pub fn get_player_stats(&self, account_id: AccountId) -> Option<LeaderboardEntry> {
        self.leaderboard.get(&account_id)
    }

    pub fn get_recent_games(&self, limit: u64) -> Vec<LeaderboardEntry> {
        let mut entries: Vec<_> = self.leaderboard.values().collect();
        entries.sort_by(|a, b| b.last_played.cmp(&a.last_played));
        entries.into_iter().take(limit as usize).collect()
    }
} 