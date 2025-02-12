"""Agent Arcade CLI main entry point."""
import click
from cli.core.near import NEARWallet
from cli.games import get_game, GAMES

VERSION = "0.1.0"

@click.group()
@click.version_option(version=VERSION, prog_name="Agent Arcade")
def cli():
    """Train and compete with AI agents using deep reinforcement learning."""
    pass

@cli.command()
def list_games():
    """List available games."""
    click.echo("Available games:")
    for name in sorted(GAMES.keys()):
        click.echo(f"  - {name}")

@cli.command()
@click.option("--network", default="testnet", help="NEAR network to use")
def login(network: str):
    """Login to NEAR wallet."""
    wallet = NEARWallet(network)
    success = wallet.login_with_cli()
    if success:
        click.echo("✅ Successfully logged in!")
    else:
        click.echo("❌ Login failed!")

@cli.command()
@click.argument("game")
@click.option("--render/--no-render", default=False, help="Enable visualization")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
def train(game: str, render: bool, config: str):
    """Train an agent for a specific game."""
    try:
        game_instance = get_game(game)
        game_instance.train(render=render, config_path=config)
    except Exception as e:
        click.echo(f"❌ Training failed: {e}", err=True)

@cli.command()
@click.argument("game")
@click.option("--model", required=True, type=click.Path(exists=True), help="Path to model file")
@click.option("--episodes", default=10, help="Number of evaluation episodes")
def evaluate(game: str, model: str, episodes: int):
    """Evaluate a trained model."""
    try:
        game_instance = get_game(game)
        results = game_instance.evaluate(model_path=model, episodes=episodes)
        click.echo("📊 Evaluation Results:")
        for key, value in results.items():
            click.echo(f"  {key}: {value}")
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}", err=True)

if __name__ == "__main__":
    cli() 