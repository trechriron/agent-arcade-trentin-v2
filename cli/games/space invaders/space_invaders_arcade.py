import click
import os
import sys
from pathlib import Path
from stable_baselines3 import DQN

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.train_space_invaders import train as train_agent
from scripts.evaluate_space_invaders import evaluate as evaluate_agent

@click.group()
def space_invaders():
    """Space Invaders game commands"""
    pass

@space_invaders.command()
@click.option('--render', is_flag=True, help='Render training environment')
@click.option('--wandb', is_flag=True, help='Use Weights & Biases for logging')
@click.option('--demo', is_flag=True, help='Run in demo mode')
@click.option('--record', is_flag=True, help='Record training videos')
@click.option('--config', default='configs/space_invaders_sb3_config.yaml', help='Path to config file')
def train(render, wandb, demo, record, config):
    """Train a Space Invaders agent"""
    if not os.path.exists(config):
        click.echo(f"Config file not found: {config}")
        return
    
    click.echo("Starting Space Invaders training...")
    click.echo(f"Config: {config}")
    click.echo(f"Render: {'Enabled' if render else 'Disabled'}")
    click.echo(f"W&B: {'Enabled' if wandb else 'Disabled'}")
    click.echo(f"Demo: {'Enabled' if demo else 'Disabled'}")
    click.echo(f"Record: {'Enabled' if record else 'Disabled'}")
    
    train_agent(config, render, wandb, demo, record)

@space_invaders.command()
@click.option('--model', required=True, help='Path to trained model')
@click.option('--episodes', default=10, help='Number of episodes to evaluate')
@click.option('--no-render', is_flag=True, help='Disable rendering')
@click.option('--difficulty', default=0, type=click.IntRange(0, 1), help='Game difficulty (0-1)')
@click.option('--mode', default=0, type=click.IntRange(0, 15), help='Game mode (0-15)')
def evaluate(model, episodes, no_render, difficulty, mode):
    """Evaluate a trained Space Invaders agent"""
    if not os.path.exists(model):
        click.echo(f"Model file not found: {model}")
        return
    
    click.echo("Starting Space Invaders evaluation...")
    click.echo(f"Model: {model}")
    click.echo(f"Episodes: {episodes}")
    click.echo(f"Render: {'Disabled' if no_render else 'Enabled'}")
    click.echo(f"Difficulty: {difficulty}")
    click.echo(f"Mode: {mode}")
    
    results = evaluate_agent(model, episodes, not no_render, difficulty, mode)
    
    click.echo("\nEvaluation Results:")
    click.echo(f"Mean reward: {results['mean_reward']:.2f}")
    click.echo(f"Max reward: {results['max_reward']:.2f}")
    click.echo(f"Mean episode length: {results['mean_length']:.2f}")

@space_invaders.command()
@click.option('--amount', required=True, type=float, help='Amount of NEAR to stake')
@click.option('--model', required=True, help='Path to model to stake on')
@click.option('--target-score', required=True, type=float, help='Target score to achieve')
def stake(amount, model, target_score):
    """Stake NEAR on a Space Invaders agent's performance"""
    if not os.path.exists(model):
        click.echo(f"Model file not found: {model}")
        return
    
    click.echo(f"Staking {amount} NEAR on Space Invaders model")
    click.echo(f"Model: {model}")
    click.echo(f"Target score: {target_score}")
    # TODO: Implement staking logic once NEAR contract is ready
    click.echo("Staking functionality coming soon!") 