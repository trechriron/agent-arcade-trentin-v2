import click
import asyncio
from ...core import cli, agent_arcade
from stable_baselines3 import DQN
import gymnasium as gym

@agent_arcade.group()
def pong():
    """Pong game commands"""
    pass

@pong.command()
@click.option('--amount', type=float, required=True, help='Amount of NEAR to stake')
@click.option('--target-score', type=int, required=True, help='Target score to achieve')
def stake(amount: float, target_score: int):
    """Stake NEAR tokens on achieving a target Pong score"""
    contract = cli.get_contract("pong")
    try:
        if asyncio.run(contract.stake(amount, target_score)):
            click.echo(f"Successfully staked {amount} NEAR on achieving score {target_score}")
        else:
            click.echo("Staking failed. Please try again.")
    except ValueError as e:
        click.echo(f"Error: {e}")

@pong.command()
@click.option('--score', type=int, required=True, help='Achieved score to claim reward for')
def claim(score: int):
    """Claim rewards for achieved score"""
    contract = cli.get_contract("pong")
    try:
        reward = asyncio.run(contract.claim_reward(score))
        if reward > 0:
            click.echo(f"Successfully claimed {reward} NEAR!")
        else:
            click.echo("No rewards to claim.")
    except Exception as e:
        click.echo(f"Failed to claim reward: {e}")

@pong.command()
def evaluate():
    """Evaluate trained Pong agent"""
    try:
        # Use existing evaluation logic
        from scripts.evaluate_pong import evaluate_agent
        score = evaluate_agent()
        
        # Auto-claim rewards if score is good
        if score > 0:
            contract = cli.get_contract("pong")
            reward = asyncio.run(contract.claim_reward(score))
            if reward > 0:
                click.echo(f"\nClaimed {reward} NEAR for score {score}!")
    except Exception as e:
        click.echo(f"Evaluation failed: {e}")

@pong.command()
@click.option('--limit', type=int, default=10, help='Number of entries to show')
def leaderboard(limit: int):
    """Show Pong leaderboard"""
    contract = cli.get_contract("pong")
    try:
        entries = asyncio.run(contract.get_top_players(limit))
        if not entries:
            click.echo("No entries found in leaderboard")
            return
            
        click.echo("\nPong Leaderboard")
        click.echo("-" * 50)
        for entry in entries:
            click.echo(f"Player: {entry.account_id}")
            click.echo(f"Best Score: {entry.best_score}")
            click.echo(f"Total Earned: {entry.total_earned} NEAR")
            click.echo(f"Games Played: {entry.games_played}")
            click.echo(f"Win Rate: {entry.win_rate * 100:.1f}%")
            click.echo("-" * 50)
    except Exception as e:
        click.echo(f"Error fetching leaderboard: {e}")

@pong.command()
@click.option('--account-id', help='Account to check (defaults to current user)')
def stats(account_id: str = None):
    """Show player statistics"""
    contract = cli.get_contract("pong")
    try:
        stats = asyncio.run(contract.get_player_stats(account_id or cli.wallet.account_id))
        if not stats:
            click.echo("No stats found")
            return
            
        click.echo("\nPlayer Statistics")
        click.echo("-" * 50)
        click.echo(f"Account: {stats.account_id}")
        click.echo(f"Best Score: {stats.best_score}")
        click.echo(f"Total Earned: {stats.total_earned} NEAR")
        click.echo(f"Games Played: {stats.games_played}")
        click.echo(f"Win Rate: {stats.win_rate * 100:.1f}%")
        click.echo(f"Highest Reward Multiplier: {stats.highest_reward_multiplier}x")
    except Exception as e:
        click.echo(f"Error fetching stats: {e}")

@pong.command()
def recent():
    """Show recent games"""
    contract = cli.get_contract("pong")
    try:
        games = asyncio.run(contract.get_recent_games(5))
        if not games:
            click.echo("No recent games found")
            return
            
        click.echo("\nRecent Games")
        click.echo("-" * 50)
        for game in games:
            click.echo(f"Player: {game.account_id}")
            click.echo(f"Score: {game.best_score}")
            click.echo(f"Earned: {game.total_earned} NEAR")
            click.echo("-" * 50)
    except Exception as e:
        click.echo(f"Error fetching recent games: {e}") 