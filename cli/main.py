"""Agent Arcade CLI main entry point."""
import click
from pathlib import Path
from typing import Optional
from loguru import logger

from cli.core.near import NEARWallet
from cli.games import get_game, list_games, GAMES

VERSION = "0.1.0"

def setup_logging():
    """Configure logging for the CLI."""
    logger.add(
        "agent_arcade.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )

@click.group()
@click.version_option(version=VERSION, prog_name="Agent Arcade")
def cli():
    """Train and compete with AI agents using deep reinforcement learning."""
    setup_logging()

@cli.command()
def games():
    """List available games with descriptions."""
    available_games = list_games()
    if not available_games:
        click.echo("No games available.")
        return
    
    click.echo("\nAvailable Games:")
    click.echo("================")
    for game in available_games:
        click.echo(f"\n{game['name']} (v{game['version']})")
        click.echo(f"  {game['description']}")

@cli.command()
@click.option("--network", default="testnet", help="NEAR network to use")
@click.option("--account-id", help="Specific NEAR account ID")
def login(network: str, account_id: Optional[str]):
    """Login to NEAR wallet."""
    try:
        wallet = NEARWallet(network)
        success = wallet.login_with_cli(account_id)
        if success:
            click.echo("✅ Successfully logged in!")
        else:
            click.echo("❌ Login failed!")
    except Exception as e:
        logger.error(f"Login failed: {e}")
        click.echo("❌ Login failed! Check agent_arcade.log for details.")

@cli.command()
@click.argument("game")
@click.option("--render/--no-render", default=False, help="Enable visualization")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
def train(game: str, render: bool, config: Optional[str]):
    """Train an agent for a specific game."""
    try:
        game_instance = get_game(game)
        config_path = Path(config) if config else None
        model_path = game_instance.train(render=render, config_path=config_path)
        click.echo(f"✅ Training complete! Model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo("❌ Training failed! Check agent_arcade.log for details.")

@cli.command()
@click.argument("game")
@click.option("--model", required=True, type=click.Path(exists=True), help="Path to model file")
@click.option("--episodes", default=10, help="Number of evaluation episodes")
@click.option("--record/--no-record", default=False, help="Record evaluation videos")
def evaluate(game: str, model: str, episodes: int, record: bool):
    """Evaluate a trained model."""
    try:
        game_instance = get_game(game)
        model_path = Path(model)
        
        if not game_instance.validate_model(model_path):
            click.echo("❌ Invalid model for this game!")
            return
        
        results = game_instance.evaluate(
            model_path=model_path,
            episodes=episodes,
            record=record
        )
        
        click.echo("\n📊 Evaluation Results")
        click.echo("===================")
        click.echo(f"Score: {results.score:.2f}")
        click.echo(f"Success Rate: {results.success_rate:.1%}")
        click.echo(f"Best Episode: {results.best_episode_score:.2f}")
        click.echo(f"Avg Episode Length: {results.avg_episode_length:.1f}")
        
        reward_multiplier = game_instance.calculate_reward_multiplier(results.score)
        click.echo(f"\n🎯 Potential Reward Multiplier: {reward_multiplier}x")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        click.echo("❌ Evaluation failed! Check agent_arcade.log for details.")

@cli.command()
@click.argument("game")
@click.option("--model", required=True, type=click.Path(exists=True), help="Path to model file")
@click.option("--amount", required=True, type=float, help="Amount of NEAR to stake")
@click.option("--target-score", required=True, type=float, help="Target score to achieve")
def stake(game: str, model: str, amount: float, target_score: float):
    """Stake NEAR on agent performance."""
    try:
        game_instance = get_game(game)
        model_path = Path(model)
        
        if not game_instance.validate_model(model_path):
            click.echo("❌ Invalid model for this game!")
            return
        
        # Validate target score is reasonable
        min_score, max_score = game_instance.get_score_range()
        if target_score < min_score or target_score > max_score:
            click.echo(f"❌ Target score must be between {min_score} and {max_score}!")
            return
        
        wallet = NEARWallet()
        if not wallet.is_logged_in():
            click.echo("❌ Please login first using 'agent-arcade login'")
            return
        
        # Evaluate model to verify performance
        results = game_instance.evaluate(model_path=model_path, episodes=5)
        click.echo(f"\n📊 Model Verification Score: {results.score:.2f}")
        
        if results.score < target_score:
            click.echo(f"⚠️  Warning: Model's current performance ({results.score:.2f}) is below target ({target_score})")
            if not click.confirm("Do you want to proceed with staking?"):
                return
        
        game_instance.stake(wallet, model_path, amount, target_score)
        click.echo("✅ Stake placed successfully!")
        
    except Exception as e:
        logger.error(f"Staking failed: {e}")
        click.echo("❌ Staking failed! Check agent_arcade.log for details.")

if __name__ == "__main__":
    cli() 