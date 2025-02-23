use near_sdk::json_types::U128;
use near_sdk::serde_json::json;
use near_workspaces::{Account, Contract, DevNetwork, Worker};
use near_workspaces::network::Sandbox;
use near_workspaces::result::ExecutionResult;
use near_workspaces::types::NearToken;
use anyhow::Result;
use near_sdk::AccountId as NearAccountId;

const WASM_FILEPATH: &str = "../target/wasm32-unknown-unknown/release/agent_arcade.wasm";

#[tokio::test]
async fn test_contract_deployment() -> Result<()> {
    let worker = near_workspaces::sandbox().await?;
    let contract = deploy_contract(&worker).await?;

    // Get initial version
    let version: String = contract
        .view("get_version")
        .await?
        .json()?;
    assert_eq!(version, env!("CARGO_PKG_VERSION"));

    Ok(())
}

#[tokio::test]
async fn test_game_registration() -> Result<()> {
    let worker = near_workspaces::sandbox().await?;
    let (contract, owner) = deploy_contract_and_get_owner(&worker).await?;

    // Register a game
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

    // Verify game config
    let config = contract
        .view("get_game_config")
        .args_json(json!({"game": "pong"}))
        .await?
        .json()?;
    assert!(config.is_some());

    Ok(())
}

#[tokio::test]
async fn test_stake_placement_and_evaluation() -> Result<()> {
    let worker = near_workspaces::sandbox().await?;
    let (contract, owner) = deploy_contract_and_get_owner(&worker).await?;

    // Register game
    register_test_game(&owner, &contract).await?;

    // Create player account
    let player = create_player(&worker, &owner, "player").await?;

    // Place stake
    let stake_amount = NearToken::from_near(1); // 1 NEAR
    let args = json!({
        "game": "pong",
        "target_score": 15_i32
    });

    let result = player
        .batch(contract.id())
        .call(
            "place_stake",
            args.to_string().into_bytes(),
            stake_amount
        )
        .transact()
        .await?;
    assert!(result.is_success());

    // Verify stake
    let stake = contract
        .view("get_stake")
        .args_json(json!({"account_id": player.id()}))
        .await?
        .json()?;
    assert_eq!(stake.game, "pong");
    assert_eq!(stake.target_score, 15);

    // Submit score and evaluate
    let args = json!({"achieved_score": 18_i32});
    let result = player
        .batch(contract.id())
        .call(
            "evaluate_stake",
            args.to_string().into_bytes(),
            NearToken::from_yoctonear(0)
        )
        .transact()
        .await?;
    assert!(result.is_success());

    // Verify leaderboard entry
    let stats = contract
        .view("get_player_stats")
        .args_json(json!({
            "account_id": player.id(),
            "game": "pong"
        }))
        .await?
        .json()?;
    assert_eq!(stats.best_score, 18);
    assert!(stats.win_rate > 0.0);

    Ok(())
}

#[tokio::test]
async fn test_emergency_pause() -> Result<()> {
    let worker = near_workspaces::sandbox().await?;
    let (contract, owner) = deploy_contract_and_get_owner(&worker).await?;
    let player = create_player(&worker, &owner, "player1").await?;

    // Register game
    register_test_game(&owner, &contract).await?;

    // Pause contract
    let result = owner
        .batch(contract.id())
        .call(
            "emergency_pause",
            Vec::new(),
            NearToken::from_yoctonear(0)
        )
        .transact()
        .await?;
    assert!(result.is_success());

    // Attempt to place stake (should fail)
    let stake_amount = NearToken::from_near(1); // 1 NEAR
    let args = json!({
        "game": "pong",
        "target_score": 15_i32
    });

    let result = player
        .batch(contract.id())
        .call(
            "place_stake",
            args.to_string().into_bytes(),
            stake_amount
        )
        .transact()
        .await;
    assert!(result.is_err() || !result.unwrap().is_success());

    // Resume contract
    let result = owner
        .batch(contract.id())
        .call(
            "resume",
            Vec::new(),
            NearToken::from_yoctonear(0)
        )
        .transact()
        .await?;
    assert!(result.is_success());

    // Place stake (should succeed)
    let result = player
        .batch(contract.id())
        .call(
            "place_stake",
            args.to_string().into_bytes(),
            stake_amount
        )
        .transact()
        .await?;
    assert!(result.is_success());

    Ok(())
}

#[tokio::test]
async fn test_reward_pool() -> Result<()> {
    let worker = near_workspaces::sandbox().await?;
    let (contract, owner) = deploy_contract_and_get_owner(&worker).await?;

    // Fund pool
    let pool_amount = NearToken::from_near(10); // 10 NEAR
    let result = owner
        .batch(contract.id())
        .call(
            "fund_pool",
            Vec::new(),
            pool_amount
        )
        .transact()
        .await?;
    assert!(result.is_success());

    // Verify pool balance
    let balance: U128 = contract
        .view("get_pool_balance")
        .await?
        .json()?;
    assert_eq!(balance.0, pool_amount.as_yoctonear());

    Ok(())
}

// Helper functions
async fn deploy_contract(worker: &Worker<Sandbox>) -> Result<Contract> {
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
    
    Ok(contract)
}

async fn deploy_contract_and_get_owner(worker: &Worker<Sandbox>) -> Result<(Contract, Account)> {
    let contract = deploy_contract(worker).await?;
    let owner = worker.root_account()?;
    Ok((contract, owner))
}

async fn create_player(
    worker: &Worker<Sandbox>,
    owner: &Account,
    name: &str
) -> Result<Account> {
    owner
        .create_subaccount(name)
        .initial_balance(NearToken::from_near(5)) // 5 NEAR
        .transact()
        .await
}

async fn register_test_game(owner: &Account, contract: &Contract) -> Result<ExecutionResult> {
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

    owner
        .batch(contract.id())
        .call(
            "register_game",
            args.to_string().into_bytes(),
            NearToken::from_yoctonear(1)
        )
        .transact()
        .await
} 