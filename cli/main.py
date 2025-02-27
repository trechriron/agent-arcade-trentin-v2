"""Agent Arcade CLI."""
import os
from pathlib import Path
import click
from loguru import logger
from typing import Optional
import subprocess
import json
from datetime import datetime
import re

# Optional NEAR imports
try:
    from .core.wallet import NEARWallet
    from .core.leaderboard import LeaderboardManager
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False
    NEARWallet = None
    LeaderboardManager = None

from .core.evaluation import EvaluationConfig, EvaluationPipeline, analyze_staking
from .games import get_registered_games, get_game_info, list_games, get_game

# Initialize global managers
wallet = NEARWallet() if NEAR_AVAILABLE else None
leaderboard_manager = LeaderboardManager() if NEAR_AVAILABLE else None

# Helper functions for parsing NEAR CLI output
def parse_game_config_from_output(output_text):
    """Parse game configuration from NEAR CLI output."""
    try:
        # Look for min_score and max_score in the output
        min_score_match = re.search(r'min_score:\s*(\d+)', output_text)
        max_score_match = re.search(r'max_score:\s*(\d+)', output_text)
        min_stake_match = re.search(r'min_stake:\s*\'([^\']+)\'', output_text)
        max_multiplier_match = re.search(r'max_multiplier:\s*(\d+)', output_text)
        
        if min_score_match and max_score_match and min_stake_match and max_multiplier_match:
            return {
                'min_score': int(min_score_match.group(1)),
                'max_score': int(max_score_match.group(1)),
                'min_stake': min_stake_match.group(1),
                'max_multiplier': int(max_multiplier_match.group(1))
            }
        return None
    except Exception as e:
        logger.error(f"Failed to parse game config: {e}")
        return None

def parse_stake_info_from_output(output_text):
    """Parse stake information from NEAR CLI output."""
    try:
        # Search for pattern that looks like a JS object in the output
        js_obj_pattern = r'{\s*game:\s*\'([^\']+)\',\s*amount:\s*\'([^\']+)\',\s*target_score:\s*(\d+),\s*timestamp:\s*(\d+),\s*games_played:\s*(\d+),\s*last_evaluation:\s*(\d+)\s*}'
        match = re.search(js_obj_pattern, output_text)
        
        if match:
            return {
                'game': match.group(1),
                'amount': match.group(2),
                'target_score': int(match.group(3)),
                'timestamp': int(match.group(4)),
                'games_played': int(match.group(5)),
                'last_evaluation': int(match.group(6))
            }
            
        # If we can't parse the JS object with regex, try to find a JSON object
        json_pattern = r'({[\s\S]*?})'
        match = re.search(json_pattern, output_text)
        if match:
            try:
                stake_info = json.loads(match.group(1).replace("'", '"'))
                return stake_info
            except (json.JSONDecodeError, KeyError):
                return None
        return None
    except Exception as e:
        logger.error(f"Failed to parse stake info: {e}")
        return None

def parse_transaction_details_from_output(output_text):
    """Parse transaction details from NEAR CLI output for stake evaluation."""
    try:
        # Extract transaction hash
        tx_hash_match = re.search(r'Transaction Id\s+([a-zA-Z0-9]+)', output_text)
        tx_hash = tx_hash_match.group(1) if tx_hash_match else "Unknown"
        
        # Extract reward information
        reward_match = re.search(r'Stake evaluated: Score (\d+) achieved, reward ([\d\.]+) NEAR', output_text)
        no_reward_match = re.search(r'Stake evaluated: Score (\d+) achieved, no reward', output_text)
        
        if reward_match:
            score = int(reward_match.group(1))
            reward = float(reward_match.group(2))
            has_reward = True
        elif no_reward_match:
            score = int(no_reward_match.group(1))
            reward = 0.0
            has_reward = False
        else:
            score = None
            reward = None
            has_reward = False
        
        return {
            'transaction_hash': tx_hash,
            'score': score,
            'reward': reward,
            'has_reward': has_reward
        }
    except Exception as e:
        logger.error(f"Failed to parse transaction details: {e}")
        return None

def extract_near_amount_from_output(output_text):
    """Extract NEAR amount (in yoctoNEAR) from NEAR CLI output."""
    try:
        # Look for a quoted number pattern like '3100000000000000000000000'
        matches = re.search(r'\'(\d+)\'', output_text)
        if matches:
            return int(matches.group(1))
            
        # If we can't find a quoted number, try to find any number sequence
        matches = re.search(r'(\d+)', output_text)
        if matches:
            return int(matches.group(1))
            
        return None
    except ValueError as e:
        logger.error(f"Failed to parse NEAR amount: {e}")
        return None

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
@click.option('--account-id', help='Optional specific account ID to use')
def login(account_id: Optional[str]):
    """Log in to NEAR wallet using web browser.
    
    This command will:
    1. Open your browser to authenticate with NEAR wallet
    2. Create a new account if you don't have one
    3. Store your credentials locally for future use
    
    The web wallet will guide you through:
    - Creating a new account if needed
    - Granting access to your account
    - Generating and storing access keys
    """
    try:
        success = wallet.login(account_id)
        if not success:
            logger.error("\nLogin failed. Please ensure:")
            logger.error("1. You have NEAR CLI installed (npm install -g near-cli)")
            logger.error("2. You completed the authentication in your browser")
            logger.error("3. You have a valid NEAR account")
            logger.error("\nTo create a new account:")
            logger.error("1. Visit https://wallet.near.org")
            logger.error("2. Click 'Create Account'")
            logger.error("3. Follow the instructions to set up your account")
            logger.error("\nThen try logging in again with:")
            logger.error("  agent-arcade wallet-cmd login")
    except Exception as e:
        logger.error(f"Login failed: {e}")

@wallet_cmd.command()
def logout():
    """Log out from NEAR wallet."""
    try:
        wallet.logout()
    except Exception as e:
        logger.error(f"Logout failed: {e}")

@wallet_cmd.command()
def status():
    """Check wallet login status and balance."""
    try:
        if wallet.is_logged_in():
            logger.info(f"Logged in as {wallet.config.account_id} on {wallet.config.network}")
            balance = wallet.get_balance()
            if balance is not None:
                logger.info(f"Balance: {balance:.2f} NEAR")
            else:
                logger.info("Balance: Not available (use 'near state' to check)")
        else:
            logger.info("Not logged in")
            logger.info("\nTo log in, run:")
            logger.info("  agent-arcade wallet-cmd login")
    except Exception as e:
        logger.error(f"Failed to check status: {e}")

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
@click.option('--model', help='Path to model file')
@click.option('--episodes', default=10, help='Number of episodes to evaluate')
@click.option('--seed', default=42, help='Random seed')
@click.option('--render/--no-render', default=False, help='Render evaluation environment')
@click.option('--deterministic/--non-deterministic', default=True, help='Use deterministic actions for evaluation')
def evaluate(game: str, model: Optional[str], episodes: int, seed: int, render: bool, deterministic: bool):
    """Evaluate a trained model."""
    try:
        # Configure SDL video driver for rendering if needed
        if render:
            import platform
            import os
            system = platform.system()
            if system == "Darwin":  # macOS
                os.environ['SDL_VIDEODRIVER'] = 'cocoa'
                logger.debug("Set SDL_VIDEODRIVER to 'cocoa' for macOS rendering")
            elif system == "Linux":
                # X11 is typically used on Linux
                os.environ['SDL_VIDEODRIVER'] = 'x11'
                logger.debug("Set SDL_VIDEODRIVER to 'x11' for Linux rendering")
            # Windows typically uses directx by default, no need to set
            
            logger.info(f"Rendering enabled. SDL_VIDEODRIVER={os.environ.get('SDL_VIDEODRIVER', 'default')}")

        # Check if game exists
        games = get_registered_games()
        if game not in games:
            click.echo(f"Game {game} not found")
            return

        # Find the model file
        model_path = None
        if model:
            model_path = Path(model)
        else:
            # Try to find a model in the default locations
            possible_locations = [
                Path(f"models/{game}/baseline/{game}_fullTrain_latest.zip"),  # Latest full training model
                Path(f"models/{game}/baseline/final_model.zip"),              # Legacy path
                Path(f"models/{game}/demo/{game}_demo_latest.zip"),           # Latest demo model
            ]
            
            for loc in possible_locations:
                if loc.exists():
                    model_path = loc
                    logger.info(f"Using model found at: {model_path}")
                    break
                    
            if not model_path:
                click.echo(f"No model found for {game}. Please specify a model path.")
                return
        
        if not model_path.exists():
            click.echo(f"Model file {model_path} not found")
            return

        # Get game instance and evaluate
        game_instance = get_game(game)
        evaluation_results = game_instance.evaluate(
            model_path=str(model_path),
            episodes=episodes,
            seed=seed,
            render=render,
            deterministic=deterministic
        )

        # Display results
        click.echo("\nEvaluation Results:")
        click.echo(f"Game: {game}")
        click.echo(f"Model: {model_path}")
        click.echo(f"Episodes: {episodes}")
        click.echo(f"Mean Score: {evaluation_results.mean_reward:.2f}")
        click.echo(f"Success Rate: {evaluation_results.success_rate * 100:.1f}%")
        
        # Staking analysis
        staking_analysis = analyze_staking(
            success_rate=evaluation_results.success_rate,
            mean_score=evaluation_results.mean_reward,
            game_info=game_instance.get_game_info()
        )
        
        click.echo("\nStaking Analysis:")
        click.echo(f"Risk Level: {staking_analysis.risk_level}")
        click.echo(f"Recommendation: {staking_analysis.recommendation}")
        
        return evaluation_results
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

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
            'near', 'call',
            wallet.config.contract_id,
            'fund_pool',
            '{}',
            '--accountId', wallet.config.account_id,
            '--networkId', wallet.config.network,
            '--amount', f'{amount}',
            '--gas', '100000000000000'
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
        logger.info(f"Using NEAR {wallet.config.network} network with contract {wallet.config.contract_id}")
        cmd = [
            'near', 'call',
            wallet.config.contract_id,
            'get_pool_balance',
            '{}',
            '--accountId', wallet.config.account_id,
            '--networkId', wallet.config.network
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                # Use helper function to extract amount
                balance_yocto = extract_near_amount_from_output(result.stdout)
                if balance_yocto:
                    balance_near = float(balance_yocto) / 1e24
                    logger.info(f"Current pool balance: {balance_near:.2f} NEAR ({wallet.config.network} network)")
                    logger.info(f"Contract ID: {wallet.config.contract_id}")
                else:
                    logger.error("Failed to extract pool balance from response")
                    logger.error(f"Raw output: {result.stdout}")
            except ValueError as e:
                logger.error(f"Failed to parse pool balance: {e}")
                logger.error(f"Raw output: {result.stdout}")
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
            'near', 'call',
            wallet.config.contract_id,
            'get_game_config',
            f'{{"game": "{game}"}}',
            '--accountId', wallet.config.account_id,
            '--networkId', wallet.config.network
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to get game config: {result.stderr}")
            return
            
        # Use helper function to parse game config
        game_config = parse_game_config_from_output(result.stdout.strip())
        if not game_config:
            logger.error(f"Could not parse game config for {game}")
            logger.error(f"Raw output: {result.stdout}")
            return
            
        if not game_config:
            logger.error(f"Game {game} not registered")
            return
            
        # Continue with validation checks
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
            
            # Fix: Ensure score is a regular Python float, not a NumPy scalar
            mean_score = float(result.mean_reward)
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
            'near', 'call',
            wallet.config.contract_id,
            'place_stake',
            f'{{"game": "{game}", "target_score": {target_score}}}',
            '--accountId', wallet.config.account_id,
            '--networkId', wallet.config.network,
            '--amount', f'{amount}',
            '--gas', '100000000000000'
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
            # Enhanced error detection for common contract errors
            if "Rate limit: Too soon to place new stake" in result.stderr or "Rate limit" in result.stderr:
                logger.error("⚠️ Rate limit reached: The contract prevents placing stakes too frequently")
                logger.error("Wait at least 24 hours between stakes or try with a different account")
            elif "already has an active stake" in result.stderr:
                logger.error("⚠️ You already have an active stake. You must complete or cancel it before placing a new one.")
                logger.error("Check your current stake with: agent-arcade stake view")
            else:
                logger.error(f"Failed to place stake: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Failed to place stake: {e}")

@stake.command()
@click.argument('game')
def game_config(game: str):
    """View game configuration including min score, max score, and minimum stake."""
    if not wallet.is_logged_in():
        logger.error("Must be logged in to view game configuration")
        return
    
    logger.info(f"Using NEAR {wallet.config.network} network with contract {wallet.config.contract_id}")
    cmd = [
        'near', 'call',
        wallet.config.contract_id,
        'get_game_config',
        f'{{"game": "{game}"}}',
        '--accountId', wallet.config.account_id,
        '--networkId', wallet.config.network
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info(f"Game configuration for {game}:")
        logger.info("-" * 80)
        logger.info(result.stdout.strip())
        
        # Try to extract key values with our helper function
        game_config = parse_game_config_from_output(result.stdout)
        
        if game_config:
            min_stake_near = float(game_config['min_stake']) / 1e24
            logger.info(f"\nKey parameters:")
            logger.info(f"Min Score: {game_config['min_score']}")
            logger.info(f"Max Score: {game_config['max_score']}")
            logger.info(f"Min Stake: {min_stake_near:.2f} NEAR")
            logger.info(f"Max Multiplier: {game_config['max_multiplier']}x")
    else:
        logger.error(f"Failed to get game config: {result.stderr}")

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
        logger.info("Checking for active stake...")
        cmd = [
            'near', 'call',
            wallet.config.contract_id,
            'get_stake',
            f'{{"account_id": "{wallet.config.account_id}"}}',
            '--accountId', wallet.config.account_id,
            '--networkId', wallet.config.network
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to get stake info: {result.stderr}")
            return
            
        # Use helper function to parse stake info
        stake_info = parse_stake_info_from_output(result.stdout)
        if not stake_info:
            logger.error("No active stake found or could not parse stake info")
            logger.error(f"Raw output: {result.stdout}")
            return
            
        if stake_info['game'] != game:
            logger.error(f"Active stake is for {stake_info['game']}, not {game}")
            return
        
        # Display stake info before submission
        stake_amount = float(int(stake_info['amount']) / 1e24)
        target_score = stake_info['target_score']
        logger.info("-" * 60)
        logger.info(f"Submitting score for evaluation:")
        logger.info(f"Game: {game}")
        logger.info(f"Your staked amount: {stake_amount:.4f} NEAR")
        logger.info(f"Your target score: {target_score}")
        logger.info(f"Your achieved score: {score}")
        logger.info("-" * 60)
        logger.info("Processing transaction on NEAR blockchain...")
        logger.info("This may take a few moments...")
            
        # Submit score
        cmd = [
            'near', 'call',
            wallet.config.contract_id,
            'evaluate_stake',
            f'{{"achieved_score": {score}}}',
            '--accountId', wallet.config.account_id,
            '--networkId', wallet.config.network,
            '--gas', '100000000000000'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Record score in local leaderboard as well
            if leaderboard_manager:
                # Use default values for success_rate and episodes since we don't have that data
                # from the contract submission
                success_rate = 1.0  # Assume 100% success for stake submissions
                episodes = 50       # Use minimum recommended episodes
                model_path = ""     # We don't know which model was used
                
                leaderboard_manager.record_score(
                    game_name=game,
                    account_id=wallet.config.account_id,
                    score=score,
                    success_rate=success_rate,
                    episodes=episodes,
                    model_path=model_path
                )
                logger.info(f"Score recorded in local leaderboard")
            
            # Parse transaction details
            tx_details = parse_transaction_details_from_output(result.stdout)
            
            # Display transaction results
            logger.info("-" * 60)
            logger.info("✅ Transaction completed successfully!")
            logger.info("-" * 60)
            
            if tx_details:
                logger.info(f"Transaction Hash: {tx_details['transaction_hash']}")
                
                if tx_details['has_reward']:
                    logger.info(f"🎉 Congratulations! You earned a reward of {tx_details['reward']:.4f} NEAR")
                    logger.info(f"Reward has been sent to your wallet: {wallet.config.account_id}")
                else:
                    logger.info("Your score was recorded, but did not qualify for a reward.")
                    logger.info(f"Target score ({target_score}) was not reached with your score of {score}.")
            
            logger.info("-" * 60)
            logger.info("Score has been recorded in both blockchain and local leaderboard")
            logger.info("Check the leaderboard to see your ranking!")
            logger.info("Run: agent-arcade leaderboard player " + game)
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
        logger.info(f"Using NEAR {wallet.config.network} network with contract {wallet.config.contract_id}")
        cmd = [
            'near', 'call',
            wallet.config.contract_id,
            'get_stake',
            f'{{"account_id": "{wallet.config.account_id}"}}',
            '--accountId', wallet.config.account_id,
            '--networkId', wallet.config.network
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                # Use helper function to parse stake info
                stake_info = parse_stake_info_from_output(result.stdout)
                if stake_info:
                    # Convert yoctoNEAR to NEAR for display
                    amount_near = float(stake_info['amount']) / 1e24
                    # Convert timestamp to readable format
                    readable_time = datetime.fromtimestamp(int(stake_info['timestamp']) / 1_000_000_000)
                    
                    click.echo("\nCurrent Stake Details:")
                    click.echo("-" * 80)
                    click.echo(f"Game: {stake_info['game']}")
                    click.echo(f"Amount: {amount_near:.2f} NEAR")
                    click.echo(f"Target Score: {stake_info['target_score']}")
                    click.echo(f"Games Played: {stake_info['games_played']}")
                    click.echo(f"Placed: {readable_time}")
                else:
                    # Fallback if parsing fails
                    if "no stake found" in result.stdout.lower():
                        logger.info("No active stake found for your account")
                        logger.info(f"To place a stake, use: agent-arcade stake place <game> --model <path> --amount <near> --target-score <score>")
                    else:
                        logger.info("Received stake data (raw format):")
                        logger.info("-" * 80)
                        logger.info(result.stdout.strip())
            except Exception as e:
                logger.error(f"Failed to parse stake info: {e}")
                logger.error(f"Raw output: {result.stdout}")
        else:
            logger.error(f"Failed to get stake info: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Failed to get stake info: {e}")

@cli.command()
@click.argument('game')
@click.option('--render/--no-render', default=False, help='Render training environment')
@click.option('--config', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--output-dir', type=click.Path(), help='Custom output directory for models')
@click.option('--checkpoint-freq', default=250000, help='Frequency of saving checkpoints (in timesteps)')
def train(game: str, render: bool, config: Optional[str], output_dir: Optional[str], checkpoint_freq: int):
    """Train an agent for a specific game."""
    try:
        # Configure SDL video driver for rendering if needed
        if render:
            import platform
            import os
            system = platform.system()
            if system == "Darwin":  # macOS
                os.environ['SDL_VIDEODRIVER'] = 'cocoa'
                logger.debug("Set SDL_VIDEODRIVER to 'cocoa' for macOS rendering")
            elif system == "Linux":
                # X11 is typically used on Linux
                os.environ['SDL_VIDEODRIVER'] = 'x11'
                logger.debug("Set SDL_VIDEODRIVER to 'x11' for Linux rendering")
            # Windows typically uses directx by default, no need to set
            
            logger.info(f"Rendering enabled. SDL_VIDEODRIVER={os.environ.get('SDL_VIDEODRIVER', 'default')}")
        
        # Check if game exists
        games = get_registered_games()
        if game not in games:
            click.echo(f"Game {game} not found")
            return
        
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
            if render:
                output_dir = f"models/{game}/demo"
                logger.info(f"Saving demo model to {output_dir}")
            else:
                output_dir = f"models/{game}/baseline"
                logger.info(f"Saving training model to {output_dir}")
                
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
            config_path=config_path
        )
        
        # Show next steps
        logger.info(f"Training complete! Model saved to: {model_path}")
        logger.info("\nNext steps:")
        logger.info("1. Evaluate your model:")
        logger.info(f"   agent-arcade evaluate {game} --model {str(model_path)}")
        logger.info("2. Enter competition:")
        logger.info(f"   agent-arcade stake place {game} --model {str(model_path)} --amount <NEAR> --target-score <SCORE>")
        
        # Provide info about the latest link
        model_dir = Path(f"models/{game}/{'demo' if render else 'baseline'}")
        model_type = "demo" if render else "fullTrain"
        latest_link = model_dir / f"{game}_{model_type}_latest.zip"
        if latest_link.exists():
            logger.info("\nYou can also use the 'latest' link for convenience:")
            logger.info(f"   agent-arcade evaluate {game} --model {str(latest_link)}")
            
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