use near_sdk::borsh::{self, BorshDeserialize, BorshSerialize};
use near_sdk::{env, near_bindgen, require, AccountId, Promise, NearToken, Gas};
use near_sdk::json_types::U128;
use near_sdk::collections::{LookupMap, UnorderedMap};

use crate::{AgentArcadeContract, GameConfig, StakeInfo, LeaderboardEntry};

#[derive(BorshDeserialize, BorshSerialize)]
pub struct OldAgentArcadeContract {
    pub owner_id: AccountId,
    pub pool_balance: U128,
    pub stakes: LookupMap<AccountId, StakeInfo>,
    pub leaderboard: UnorderedMap<AccountId, LeaderboardEntry>,
    pub game_configs: UnorderedMap<String, GameConfig>,
}

pub trait Upgradable {
    fn upgrade(&mut self, code: Vec<u8>) -> Promise;
    fn migrate() -> Self;
    fn get_version(&self) -> String;
    fn needs_upgrade(&self) -> bool;
}

#[near_bindgen]
impl Upgradable for AgentArcadeContract {
    /// Upgrades the contract code and migrates state if necessary
    /// Only callable by contract owner
    #[private]
    fn upgrade(&mut self, code: Vec<u8>) -> Promise {
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
    fn migrate() -> Self {
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
    fn get_version(&self) -> String {
        self.version.clone()
    }

    /// Checks if a contract upgrade is needed
    fn needs_upgrade(&self) -> bool {
        self.version != env!("CARGO_PKG_VERSION")
    }
} 