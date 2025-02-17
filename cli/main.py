"""Agent Arcade CLI."""
import os
from pathlib import Path
import click
from loguru import logger
from typing import Optional

# Optional NEAR imports
try:
    from .core.wallet import NEARWallet
    from .core.leaderboard import LeaderboardManager
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False
    NEARWallet = None
    LeaderboardManager = None

from .core.evaluation import EvaluationConfig, EvaluationPipeline
from .games import get_registered_games, get_game_info

# Initialize global managers
wallet = NEARWallet() if NEAR_AVAILABLE else None
leaderboard_manager = LeaderboardManager() if NEAR_AVAILABLE else None

@click.group()
@click.version_option(package_name="agent-arcade")
def cli():
    """Agent Arcade CLI for training and evaluating RL agents."""
    pass

@cli.group()
def wallet_cmd():
    """Manage NEAR wallet."""
    if not NEAR_AVAILABLE:
        logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
        return
    pass

@wallet_cmd.command()
@click.option('--network', default='testnet', help='NEAR network to use')
@click.option('--account-id', help='Optional specific account ID')
def login(network: str, account_id: Optional[str]):
    """Log in to NEAR wallet using web browser."""
    try:
        wallet.config.network = network
        success = wallet.login(account_id)
        if not success:
            logger.error("Login failed. Please try again.")
    except Exception as e:
        logger.error(f"Login failed: {e}")

@wallet_cmd.command()
def logout():
    """Log out from NEAR wallet."""
    wallet.logout()
    logger.info("Successfully logged out")

@wallet_cmd.command()
def status():
    """Check wallet login status."""
    if wallet.is_logged_in():
        logger.info(f"Logged in as {wallet.config.account_id} on {wallet.config.network}")
        balance = wallet.get_balance()
        if balance is not None:
            logger.info(f"Balance: {balance} NEAR")
        else:
            logger.error("Failed to fetch balance")
    else:
        logger.info("Not logged in")

@cli.group()
def leaderboard():
    """View leaderboards."""
    if not NEAR_AVAILABLE:
        logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
        return
    pass

@leaderboard.command()
@click.argument('game')
@click.option('--limit', default=10, help='Number of entries to show')
def top(game: str, limit: int):
    """Show top scores for a game."""
    if not leaderboard_manager:
        logger.error("Leaderboard manager not initialized")
        return
        
    game_board = leaderboard_manager.get_leaderboard(game)
    entries = game_board.get_top_scores(limit)
    
    if not entries:
        logger.info(f"No entries found for {game}")
        return
    
    click.echo(f"\nTop {limit} scores for {game}:")
    click.echo("-" * 80)
    click.echo(f"{'Rank':<6}{'Player':<30}{'Score':<15}{'Success Rate':<15}{'Episodes':<10}")
    click.echo("-" * 80)
    
    for i, entry in enumerate(entries, 1):
        click.echo(
            f"{i:<6}{entry.account_id:<30}"
            f"{entry.score:<15.2f}"
            f"{entry.success_rate*100:<14.1f}%"
            f"{entry.episodes:<10}"
        )

@leaderboard.command()
@click.argument('game')
@click.option('--limit', default=10, help='Number of entries to show')
def recent(game: str, limit: int):
    """Show recent scores for a game."""
    if not leaderboard_manager:
        logger.error("Leaderboard manager not initialized")
        return
        
    game_board = leaderboard_manager.get_leaderboard(game)
    entries = game_board.get_recent_entries(limit)
    
    if not entries:
        logger.info(f"No entries found for {game}")
        return
    
    from datetime import datetime
    
    click.echo(f"\nRecent {limit} scores for {game}:")
    click.echo("-" * 80)
    click.echo(f"{'Player':<30}{'Score':<15}{'Success Rate':<15}{'Date':<20}")
    click.echo("-" * 80)
    
    for entry in entries:
        date = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d %H:%M')
        click.echo(
            f"{entry.account_id:<30}"
            f"{entry.score:<15.2f}"
            f"{entry.success_rate*100:<14.1f}%"
            f"{date:<20}"
        )

@leaderboard.command()
@click.argument('game')
def player(game: str):
    """Show player's best score and rank for a game."""
    if not wallet or not leaderboard_manager:
        logger.error("Wallet or leaderboard manager not initialized")
        return
        
    if not wallet.is_logged_in():
        logger.error("Must be logged in to view player stats")
        return
    
    game_board = leaderboard_manager.get_leaderboard(game)
    best_entry = game_board.get_player_best(wallet.account_id)
    rank = game_board.get_player_rank(wallet.account_id)
    
    if not best_entry:
        logger.info(f"No entries found for {wallet.account_id} in {game}")
        return
    
    from datetime import datetime
    
    click.echo(f"\nStats for {wallet.account_id} in {game}:")
    click.echo("-" * 80)
    click.echo(f"Best Score: {best_entry.score:.2f}")
    click.echo(f"Success Rate: {best_entry.success_rate*100:.1f}%")
    click.echo(f"Rank: {rank}")
    click.echo(f"Episodes Played: {best_entry.episodes}")
    click.echo(f"Last Played: {datetime.fromtimestamp(best_entry.timestamp).strftime('%Y-%m-%d %H:%M')}")

@leaderboard.command()
def stats():
    """Show global leaderboard statistics."""
    if not leaderboard_manager:
        logger.error("Leaderboard manager not initialized")
        return
        
    stats = leaderboard_manager.get_global_stats()
    
    click.echo("\nGlobal Leaderboard Statistics:")
    click.echo("-" * 80)
    click.echo(f"Total Players: {stats['total_players']}")
    click.echo(f"Total Entries: {stats['total_entries']}")
    click.echo("\nGame Statistics:")
    
    for game, game_stats in stats['games'].items():
        click.echo(f"\n{game}:")
        click.echo(f"  Players: {game_stats['players']}")
        click.echo(f"  Entries: {game_stats['entries']}")
        click.echo(f"  Best Score: {game_stats['best_score']:.2f}")
        click.echo(f"  Average Score: {game_stats['avg_score']:.2f}")

@cli.command()
@click.argument('game')
@click.argument('model-path', type=click.Path(exists=True))
@click.option('--episodes', default=20, help='Number of evaluation episodes')
@click.option('--render/--no-render', default=True, help='Render evaluation episodes')
@click.option('--verbose', default=1, help='Verbosity level')
def test_evaluate(game: str, model_path: str, episodes: int = 20, render: bool = True, verbose: int = 1):
    """Test evaluate a trained model without requiring login."""
    games = get_registered_games()
    if game not in games:
        logger.error(f"Game {game} not found")
        return
    
    game_interface = games[game]()
    
    try:
        result = game_interface.evaluate(
            model_path=Path(model_path),
            episodes=episodes,
            record=render
        )
        
        click.echo(f"\nEvaluation Results for {game}:")
        click.echo("-" * 80)
        click.echo(f"Average Score: {result.score:.2f}")
        click.echo(f"Best Score: {result.best_episode_score:.2f}")
        click.echo(f"Success Rate: {result.success_rate*100:.1f}%")
        click.echo(f"Average Episode Length: {result.avg_episode_length:.1f}")
        click.echo(f"Episodes: {result.episodes}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

@cli.command()
@click.argument('game')
@click.argument('model-path', type=click.Path(exists=True))
@click.option('--episodes', default=100, help='Number of evaluation episodes')
@click.option('--render/--no-render', default=False, help='Render evaluation episodes')
@click.option('--verbose', default=1, help='Verbosity level')
def evaluate(game: str, model_path: str, episodes: int, render: bool, verbose: int):
    """Evaluate a trained model and record to leaderboard."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to evaluate models")
        return
    
    games = get_registered_games()
    if game not in games:
        logger.error(f"Game {game} not found")
        return
    
    game_info = get_game_info(game)
    env = game_info.make_env()
    model = game_info.load_model(model_path)
    
    config = EvaluationConfig(
        n_eval_episodes=episodes,
        render=render,
        verbose=verbose
    )
    
    pipeline = EvaluationPipeline(
        game=game,
        env=env,
        model=model,
        wallet=wallet,
        leaderboard_manager=leaderboard_manager,
        config=config
    )
    
    try:
        result = pipeline.run_and_record(Path(model_path))
        
        click.echo(f"\nEvaluation Results for {game}:")
        click.echo("-" * 80)
        click.echo(f"Mean Reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
        click.echo(f"Success Rate: {result.success_rate*100:.1f}%")
        click.echo(f"Episodes: {result.n_episodes}")
        
        rank = leaderboard_manager.get_leaderboard(game).get_player_rank(wallet.account_id)
        if rank:
            click.echo(f"Current Rank: {rank}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
    finally:
        env.close()

@cli.group()
def stake():
    """Stake NEAR on game performance."""
    if not NEAR_AVAILABLE:
        logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
        return
    pass

@stake.command()
@click.argument('game')
@click.option('--model', required=True, type=click.Path(exists=True), help='Path to trained model')
@click.option('--amount', required=True, type=float, help='Amount to stake in NEAR')
@click.option('--target-score', required=True, type=float, help='Target score to achieve')
def place(game: str, model: str, amount: float, target_score: float):
    """Place a stake on game performance."""
    if not NEAR_AVAILABLE:
        logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
        return
        
    if not wallet:
        logger.error("NEAR wallet not initialized")
        return
        
    if not wallet.is_logged_in():
        logger.error("Please log in first with: agent-arcade wallet-cmd login")
        return
        
    game_info = get_game_info(game)
    if not game_info:
        logger.error(f"Unknown game: {game}")
        return
        
    try:
        import asyncio
        from cli.games.staking import stake_on_game
        
        # Run staking operation in event loop
        asyncio.run(stake_on_game(
            wallet=wallet,
            game_name=game,
            model_path=Path(model),
            amount=amount,
            target_score=target_score,
            score_range=game_info.score_range
        ))
    except Exception as e:
        logger.error(f"Failed to place stake: {e}")

@stake.command()
@click.argument('game')
@click.argument('stake-id')
@click.option('--episodes', default=100, help='Number of evaluation episodes')
@click.option('--render/--no-render', default=False, help='Render evaluation episodes')
def evaluate(game: str, stake_id: str, episodes: int, render: bool):
    """Evaluate a staked model."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to evaluate stakes")
        return
    
    try:
        # Get stake record
        stake = wallet.get_stake(stake_id)
        if not stake:
            logger.error(f"Stake {stake_id} not found")
            return
        
        # Run evaluation
        config = EvaluationConfig(
            n_eval_episodes=episodes,
            render=render,
            verbose=1
        )
        
        games = get_registered_games()
        if game not in games:
            logger.error(f"Game {game} not found")
            return
        
        game_info = get_game_info(game)
        env = game_info.make_env()
        model = game_info.load_model(stake.model_path)
        
        pipeline = EvaluationPipeline(
            game=game,
            env=env,
            model=model,
            wallet=wallet,
            leaderboard_manager=leaderboard_manager,
            config=config
        )
        
        result = pipeline.evaluate()
        
        # Update stake record
        stake.achieved_score = result.mean_reward
        stake.status = "completed"
        
        # Calculate reward
        if result.mean_reward >= stake.target_score:
            multiplier = min(3.0, 1.0 + (result.mean_reward - stake.target_score) / stake.target_score)
            reward = stake.amount * multiplier
            logger.info(f"🎉 Success! Earned {reward:.2f} NEAR (x{multiplier:.1f})")
        else:
            logger.info("❌ Target score not achieved")
        
        wallet.record_stake(stake)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
    finally:
        env.close()

@stake.command()
def list():
    """List all stakes."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to view stakes")
        return
    
    stakes = wallet.get_stakes()
    if not stakes:
        logger.info("No stakes found")
        return
    
    click.echo("\nYour Stakes:")
    click.echo("-" * 80)
    click.echo(f"{'Game':<15}{'Amount':<10}{'Target':<10}{'Status':<15}{'Score':<10}")
    click.echo("-" * 80)
    
    for stake in stakes:
        score = f"{stake.achieved_score:.1f}" if stake.achieved_score is not None else "-"
        click.echo(
            f"{stake.game:<15}"
            f"{stake.amount:<10.1f}"
            f"{stake.target_score:<10.1f}"
            f"{stake.status:<15}"
            f"{score:<10}"
        )

@cli.command()
@click.argument('game')
@click.option('--render/--no-render', default=False, help='Render training environment')
@click.option('--config', type=click.Path(exists=True), help='Path to configuration file')
def train(game: str, render: bool, config: Optional[str]):
    """Train an agent for a specific game."""
    games = get_registered_games()
    if game not in games:
        logger.error(f"Game {game} not found")
        return
    
    game_instance = games[game]()
    try:
        config_path = Path(config) if config else None
        model_path = game_instance.train(render=render, config_path=config_path)
        logger.info(f"Training complete! Model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    cli() 