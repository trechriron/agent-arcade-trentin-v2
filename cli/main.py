"""Agent Arcade CLI."""
import os
from pathlib import Path
import click
from loguru import logger
from typing import Optional
import subprocess
import json
from datetime import datetime

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
from .games import get_registered_games, get_game_info, list_games, get_game

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
            f"{entry.success_rate*100:>13.1f}%"
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
            f"{entry.success_rate*100:>13.1f}%"
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
    best_entry = game_board.get_player_best(wallet.config.account_id)
    rank = game_board.get_player_rank(wallet.config.account_id)
    
    if not best_entry:
        logger.info(f"No entries found for {wallet.config.account_id} in {game}")
        return
    
    from datetime import datetime
    
    click.echo(f"\nStats for {wallet.config.account_id} in {game}:")
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
    game_info = get_game_info(game)
    if not game_info:
        logger.error(f"Game {game} not found")
        return
    
    try:
        result = game_info.evaluate(
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
@click.option('--record/--no-record', default=False, help='Record videos of evaluation')
def evaluate(game: str, model_path: str, episodes: int, render: bool, record: bool):
    """Evaluate a trained model and record to leaderboard."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to evaluate models")
        return
    
    try:
        # Get game config to show target scores
        cmd = [
            'near', 'contract', 'call-function',
            'as-read-only',
            wallet.config.contract_id,
            'get_game_config',
            'json-args', f'{{"game": "{game}"}}',
            'network-config', wallet.config.network,
            'now'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        game_config = None
        if result.returncode == 0:
            game_config = json.loads(result.stdout.strip())
    
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
            verbose=verbose,
            record_video=record
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
            click.echo(f"Mean Score: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
            click.echo(f"Success Rate: {result.success_rate*100:.1f}%")
            click.echo(f"Episodes: {result.n_episodes}")
            
            if game_config:
                click.echo("\nStaking Thresholds:")
                click.echo(f"Min Score: {game_config['min_score']}")
                click.echo(f"Max Score: {game_config['max_score']}")
                click.echo(f"Max Multiplier: {game_config['max_multiplier']}x")
                
                # Calculate recommended stake target
                mean_score = result.mean_reward
                if mean_score >= game_config['min_score']:
                    recommended_target = min(
                        mean_score * 0.8,  # 80% of mean score
                        game_config['max_score']
                    )
                    click.echo(f"\nRecommended stake target: {recommended_target:.1f}")
                    click.echo("To place stake with this target:")
                    click.echo(f"agent-arcade stake place {game} --model {model_path} "
                             f"--amount <NEAR> --target-score {recommended_target:.1f}")
            
            rank = leaderboard_manager.get_leaderboard(game).get_player_rank(wallet.config.account_id)
            if rank:
                click.echo(f"\nCurrent Rank: {rank}")
        
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
        finally:
            env.close()
            
    except Exception as e:
        logger.error(f"Evaluation setup failed: {e}")
        if 'env' in locals():
            env.close()

@cli.group()
def pool():
    """Manage reward pool."""
    if not NEAR_AVAILABLE:
        logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
        return
    pass

@pool.command()
@click.argument('amount', type=float)
def fund(amount: float):
    """Fund the reward pool with NEAR."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to fund pool")
        return
    
    try:
        # Convert NEAR to yoctoNEAR for the contract
        yocto_amount = str(int(amount * 1e24))
        
        cmd = [
            'near', 'contract', 'call-function',
            'as-transaction',
            wallet.config.contract_id,
            'fund_pool',
            'json-args', '{}',
            'prepaid-gas', '100 TGas',
            'attached-deposit', f'{amount} NEAR',
            'sign-as', wallet.config.account_id,
            'network-config', wallet.config.network,
            'sign-with-keychain', 'send'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully funded pool with {amount} NEAR")
        else:
            logger.error(f"Failed to fund pool: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Failed to fund pool: {e}")

@pool.command()
def balance():
    """Get current pool balance."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to view pool balance")
        return
    
    try:
        cmd = [
            'near', 'contract', 'call-function',
            'as-read-only',
            wallet.config.contract_id,
            'get_pool_balance',
            'json-args', '{}',
            'network-config', wallet.config.network,
            'now'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse the yoctoNEAR amount from the result and convert to NEAR
            balance_yocto = json.loads(result.stdout.strip())
            balance_near = float(balance_yocto) / 1e24
            logger.info(f"Current pool balance: {balance_near:.2f} NEAR")
        else:
            logger.error(f"Failed to get pool balance: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Failed to get pool balance: {e}")

@cli.group()
def stake():
    """Manage stakes."""
    if not NEAR_AVAILABLE:
        logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
        return
    pass

@stake.command()
@click.argument('game')
@click.option('--model', required=True, type=click.Path(exists=True), help='Path to trained model')
@click.option('--amount', required=True, type=float, help='Amount to stake in NEAR')
@click.option('--target-score', required=True, type=float, help='Target score to achieve')
@click.option('--evaluate/--no-evaluate', default=True, help='Run evaluation before staking')
def place(game: str, model: str, amount: float, target_score: float, evaluate: bool):
    """Place a stake on game performance."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to place stake")
        return
    
    try:
        # Verify game exists and get config
        cmd = [
            'near', 'contract', 'call-function',
            'as-read-only',
            wallet.config.contract_id,
            'get_game_config',
            'json-args', f'{{"game": "{game}"}}',
            'network-config', wallet.config.network,
            'now'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to get game config: {result.stderr}")
            return
            
        game_config = json.loads(result.stdout.strip())
        if not game_config:
            logger.error(f"Game {game} not registered")
            return
            
        # Validate stake parameters
        if target_score < game_config['min_score'] or target_score > game_config['max_score']:
            logger.error(f"Target score must be between {game_config['min_score']} and {game_config['max_score']}")
            return
            
        min_stake_near = float(game_config['min_stake']) / 1e24
        if amount < min_stake_near:
            logger.error(f"Minimum stake is {min_stake_near} NEAR")
            return
        
        # Optional evaluation before staking
        if evaluate:
            logger.info("Evaluating model before staking...")
            game_info = get_game_info(game)
            if not game_info:
                logger.error(f"Game {game} not found")
                return
                
            result = game_info.evaluate(
                model_path=Path(model),
                episodes=20,
                record=False
            )
            
            mean_score = result.score
            logger.info(f"\nQuick Evaluation Results:")
            logger.info(f"Mean Score: {mean_score:.2f}")
            logger.info(f"Target Score: {target_score}")
            
            # Warn if target seems unrealistic
            if mean_score < target_score * 0.5:
                logger.warning(f"Warning: Your target score ({target_score}) is significantly higher than current performance ({mean_score:.2f})")
                if not click.confirm("Do you want to continue with this stake?"):
                    return
            
            # Calculate potential reward
            if mean_score >= target_score:
                potential_reward = amount * game_config['max_multiplier']
                logger.info(f"Potential reward if performance maintained: {potential_reward:.2f} NEAR (x{game_config['max_multiplier']})")
            elif mean_score >= target_score * 0.8:
                potential_reward = amount * (game_config['max_multiplier'] / 2)
                logger.info(f"Potential reward if performance maintained: {potential_reward:.2f} NEAR (x{game_config['max_multiplier']/2})")
        
        # Place stake
        cmd = [
            'near', 'contract', 'call-function',
            'as-transaction',
            wallet.config.contract_id,
            'place_stake',
            'json-args', f'{{"game": "{game}", "target_score": {target_score}}}',
            'prepaid-gas', '100 TGas',
            'attached-deposit', f'{amount} NEAR',
            'sign-as', wallet.config.account_id,
            'network-config', wallet.config.network,
            'sign-with-keychain', 'send'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"\nSuccessfully placed stake of {amount} NEAR on {game}")
            logger.info(f"Target score: {target_score}")
            logger.info("\nNext steps:")
            logger.info("1. Check your stake status:")
            logger.info("   agent-arcade stake view")
            logger.info("2. Submit your score when ready:")
            logger.info(f"   agent-arcade stake submit {game} <achieved_score>")
        else:
            logger.error(f"Failed to place stake: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Failed to place stake: {e}")

@stake.command()
@click.argument('game')
@click.argument('score', type=float)
def submit(game: str, score: float):
    """Submit a score for your current stake."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to submit score")
        return
    
    try:
        # Verify active stake exists
        cmd = [
            'near', 'contract', 'call-function',
            'as-read-only',
            wallet.config.contract_id,
            'get_stake',
            'json-args', f'{{"account_id": "{wallet.config.account_id}"}}',
            'network-config', wallet.config.network,
            'now'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to get stake info: {result.stderr}")
            return
            
        stake_info = json.loads(result.stdout.strip())
        if not stake_info:
            logger.error("No active stake found")
            return
            
        if stake_info['game'] != game:
            logger.error(f"Active stake is for {stake_info['game']}, not {game}")
            return
        
        # Submit score
        cmd = [
            'near', 'contract', 'call-function',
            'as-transaction',
            wallet.config.contract_id,
            'evaluate_stake',
            'json-args', f'{{"achieved_score": {score}}}',
            'prepaid-gas', '100 TGas',
            'attached-deposit', '0 NEAR',
            'sign-as', wallet.config.account_id,
            'network-config', wallet.config.network,
            'sign-with-keychain', 'send'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully submitted score of {score} for {game}")
            logger.info("Check the leaderboard to see your ranking!")
        else:
            logger.error(f"Failed to submit score: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Failed to submit score: {e}")

@stake.command()
def view():
    """View current stake details."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to view stake")
        return
    
    try:
        cmd = [
            'near', 'contract', 'call-function',
            'as-read-only',
            wallet.config.contract_id,
            'get_stake',
            'json-args', f'{{"account_id": "{wallet.config.account_id}"}}',
            'network-config', wallet.config.network,
            'now'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            stake_info = json.loads(result.stdout.strip())
            
            if stake_info:
                # Convert yoctoNEAR to NEAR for display
                amount_near = float(stake_info['amount']) / 1e24
                # Convert timestamp to readable format
                timestamp = datetime.fromtimestamp(int(stake_info['timestamp']) / 1_000_000_000)
                
                click.echo("\nCurrent Stake Details:")
                click.echo("-" * 80)
                click.echo(f"Game: {stake_info['game']}")
                click.echo(f"Amount: {amount_near:.2f} NEAR")
                click.echo(f"Target Score: {stake_info['target_score']}")
                click.echo(f"Games Played: {stake_info['games_played']}")
                click.echo(f"Placed: {timestamp}")
            else:
                logger.info("No active stake found")
        else:
            logger.error(f"Failed to get stake info: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Failed to get stake info: {e}")

@cli.command()
@click.argument('game')
@click.option('--render/--no-render', default=False, help='Render training environment')
@click.option('--config', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--output-dir', type=click.Path(), help='Directory to save model (default: models/<game>)')
@click.option('--checkpoint-freq', type=int, default=100000, help='Save checkpoint every N steps')
def train(game: str, render: bool, config: Optional[str], output_dir: Optional[str], checkpoint_freq: int):
    """Train an agent for a specific game."""
    try:
        # Set default config path if not provided
        if not config:
            default_config = Path(f"models/{game}/config.yaml")
            if default_config.exists():
                config = str(default_config)
                logger.info(f"Using default config: {config}")
            else:
                logger.warning(f"No config found at {default_config}, using base configuration")
        
        # Set default output directory
        if not output_dir:
            output_dir = f"models/{game}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get game instance and train
        game_instance = get_game(game)
        if not game_instance:
            logger.error(f"Game {game} not found")
            return
            
        config_path = Path(config) if config else None
        model_path = game_instance.train(
            render=render,
            config_path=config_path,
            output_dir=output_path,
            checkpoint_freq=checkpoint_freq
        )
        
        # Show next steps
        logger.info(f"Training complete! Model saved to: {model_path}")
        logger.info("\nNext steps:")
        logger.info("1. Evaluate your model:")
        logger.info(f"   agent-arcade evaluate {game} --model {model_path}")
        logger.info("2. Enter competition:")
        logger.info(f"   agent-arcade stake place {game} --model {model_path} --amount <NEAR> --target-score <SCORE>")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

@cli.command()
def list_games():
    """List all available games."""
    from cli.games import get_registered_games
    
    games = []
    for game_name in get_registered_games():
        game = get_game_info(game_name)
        if game:
            games.append({
                "name": game.name,
                "description": game.description,
                "version": game.version,
                "staking_enabled": NEAR_AVAILABLE
            })
    
    click.echo("\nAvailable Games:")
    click.echo("-" * 80)
    click.echo(f"{'Name':<20}{'Description':<40}{'Version':<10}{'Staking':<10}")
    click.echo("-" * 80)
    
    for game in games:
        staking = "✓" if game["staking_enabled"] else "-"
        click.echo(
            f"{game['name']:<20}"
            f"{game['description']:<40}"
            f"{game['version']:<10}"
            f"{staking:<10}"
        )
    click.echo()

if __name__ == "__main__":
    cli() 