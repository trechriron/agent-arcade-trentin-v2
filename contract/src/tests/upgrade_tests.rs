use near_sdk::json_types::U128;
use near_sdk::serde_json::json;
use near_workspaces::{Account, Contract, DevNetwork, Worker};
use near_workspaces::network::Sandbox;
use near_workspaces::types::NearToken;
use anyhow::Result;

const WASM_FILEPATH: &str = "../target/wasm32-unknown-unknown/release/agent_arcade.wasm";

#[tokio::test]
async fn test_upgrade_permissions() -> Result<()> {
    let worker = near_workspaces::sandbox().await?;
    let (contract, owner) = deploy_contract_and_get_owner(&worker).await?;
    let non_owner = create_test_account(&worker, "non_owner").await?;

    // Setup initial state
    register_test_game(&owner, &contract).await?;
    fund_pool(&owner, &contract, 10).await?;

    // Non-owner attempt should fail
    let wasm = std::fs::read(WASM_FILEPATH)?;
    let result = non_owner
        .batch(contract.id())
        .call(
            "upgrade",
            wasm.clone(),
            NearToken::from_yoctonear(0)
        )
        .transact()
        .await;
    assert!(result.is_err() || !result.unwrap().is_success());

    // Owner attempt should succeed
    let result = owner
        .batch(contract.id())
        .call(
            "upgrade",
            wasm,
            NearToken::from_yoctonear(0)
        )
        .transact()
        .await?;
    assert!(result.is_success());

    Ok(())
}

#[tokio::test]
async fn test_upgrade_with_state_preservation() -> Result<()> {
    let worker = near_workspaces::sandbox().await?;
    let (contract, owner) = deploy_contract_and_get_owner(&worker).await?;

    // Setup initial state
    register_test_game(&owner, &contract).await?;
    fund_pool(&owner, &contract, 10).await?;

    // Create a player and place stake
    let player = create_test_account(&worker, "player").await?;
    place_test_stake(&player, &contract, 1, 15).await?;

    // Get pre-upgrade state
    let pre_upgrade_state = get_contract_state(&contract, &player).await?;

    // Perform upgrade
    let wasm = std::fs::read(WASM_FILEPATH)?;
    owner
        .batch(contract.id())
        .call(
            "upgrade",
            wasm,
            NearToken::from_yoctonear(0)
        )
        .transact()
        .await?;

    // Verify state preservation
    let post_upgrade_state = get_contract_state(&contract, &player).await?;
    assert_eq!(
        pre_upgrade_state.pool_balance,
        post_upgrade_state.pool_balance,
        "Pool balance not preserved"
    );
    assert_eq!(
        pre_upgrade_state.game_config,
        post_upgrade_state.game_config,
        "Game config not preserved"
    );
    assert_eq!(
        pre_upgrade_state.stake_info,
        post_upgrade_state.stake_info,
        "Stake info not preserved"
    );

    // Verify contract functionality after upgrade
    let result = player
        .batch(contract.id())
        .call(
            "evaluate_stake",
            json!({"achieved_score": 18}).to_string().into_bytes(),
            NearToken::from_yoctonear(0)
        )
        .transact()
        .await?;
    assert!(result.is_success());

    Ok(())
}

#[tokio::test]
async fn test_upgrade_paused_contract() -> Result<()> {
    let worker = near_workspaces::sandbox().await?;
    let (contract, owner) = deploy_contract_and_get_owner(&worker).await?;

    // Pause contract
    owner
        .batch(contract.id())
        .call(
            "emergency_pause",
            Vec::new(),
            NearToken::from_yoctonear(0)
        )
        .transact()
        .await?;

    // Attempt upgrade while paused
    let wasm = std::fs::read(WASM_FILEPATH)?;
    let result = owner
        .batch(contract.id())
        .call(
            "upgrade",
            wasm,
            NearToken::from_yoctonear(0)
        )
        .transact()
        .await;
    assert!(result.is_err() || !result.unwrap().is_success());

    Ok(())
}

#[tokio::test]
async fn test_version_tracking() -> Result<()> {
    let worker = near_workspaces::sandbox().await?;
    let (contract, _) = deploy_contract_and_get_owner(&worker).await?;

    // Check initial version
    let version: String = contract
        .view("get_version")
        .await?
        .json()?;
    assert_eq!(version, env!("CARGO_PKG_VERSION"));

    // Check upgrade needed
    let needs_upgrade: bool = contract
        .view("needs_upgrade")
        .await?
        .json()?;
    assert!(!needs_upgrade);

    Ok(())
}

// Helper functions
async fn deploy_contract_and_get_owner(worker: &Worker<Sandbox>) -> Result<(Contract, Account)> {
    let wasm = std::fs::read(WASM_FILEPATH)?;
    let contract = worker.dev_deploy(&wasm).await?;
    
    let owner = worker.root_account()?;
    let args = json!({
        "owner_id": owner.id()
    });

    owner
        .batch(contract.id())
        .call(
            "new",
            args.to_string().into_bytes(),
            NearToken::from_yoctonear(0)
        )
        .transact()
        .await?;
    
    Ok((contract, owner))
}

async fn create_test_account(worker: &Worker<Sandbox>, name: &str) -> Result<Account> {
    let owner = worker.root_account()?;
    owner
        .create_subaccount(name)
        .initial_balance(NearToken::from_near(5))
        .transact()
        .await
}

async fn register_test_game(owner: &Account, contract: &Contract) -> Result<()> {
    let args = json!({
        "game": "pong",
        "min_score": 0_i32,
        "max_score": 21_i32,
        "min_stake": U128(100_000_000_000_000_000_000_000), // 0.1 NEAR
        "max_multiplier": 3_u32,
        "rate_limit": {
            "max_evaluations_per_hour": 10,
            "min_hours_between_stakes": 24
        }
    });

    let result = owner
        .batch(contract.id())
        .call(
            "register_game",
            args.to_string().into_bytes(),
            NearToken::from_yoctonear(1)
        )
        .transact()
        .await?;
    assert!(result.is_success());
    Ok(())
}

async fn fund_pool(owner: &Account, contract: &Contract, amount: u64) -> Result<()> {
    let result = owner
        .batch(contract.id())
        .call(
            "fund_pool",
            Vec::new(),
            NearToken::from_near(amount)
        )
        .transact()
        .await?;
    assert!(result.is_success());
    Ok(())
}

async fn place_test_stake(
    player: &Account,
    contract: &Contract,
    amount: u64,
    target_score: i32
) -> Result<()> {
    let args = json!({
        "game": "pong",
        "target_score": target_score
    });

    let result = player
        .batch(contract.id())
        .call(
            "place_stake",
            args.to_string().into_bytes(),
            NearToken::from_near(amount)
        )
        .transact()
        .await?;
    assert!(result.is_success());
    Ok(())
}

#[derive(Debug, PartialEq)]
struct ContractState {
    pool_balance: U128,
    game_config: Option<serde_json::Value>,
    stake_info: Option<serde_json::Value>,
}

async fn get_contract_state(contract: &Contract, player: &Account) -> Result<ContractState> {
    let pool_balance: U128 = contract
        .view("get_pool_balance")
        .await?
        .json()?;

    let game_config = contract
        .view("get_game_config")
        .args_json(json!({"game": "pong"}))
        .await?
        .json()?;

    let stake_info = contract
        .view("get_stake")
        .args_json(json!({"account_id": player.id()}))
        .await?
        .json()?;

    Ok(ContractState {
        pool_balance,
        game_config,
        stake_info,
    })
} 