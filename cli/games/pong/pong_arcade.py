#!/usr/bin/env python3
import click
import json
import os
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import DQN
from rich.console import Console
from rich.table import Table
import asyncio
from near_api_py import Account, connect, Contract, KeyPair
from typing import List, Optional

console = Console()

class PongArcade:
    def __init__(self, network_id: str = "testnet"):
        self.config = self.load_config()
        self.network_id = network_id
        self.account = None
        self.contract = None

    def load_config(self) -> dict:
        config_path = Path.home() / ".near-config" / "pong-arcade.json"
        if not config_path.exists():
            raise FileNotFoundError(
                "NEAR configuration not found. Please run 'pong-arcade login' first."
            )
        return json.loads(config_path.read_text())

    async def connect(self):
        near = connect(self.network_id)
        self.account = Account(
            near,
            self.config["account_id"],
            KeyPair.from_string(self.config["private_key"])
        )
        self.contract = Contract(
            self.account,
            self.config["contract_id"],
            {
                "stake": self.stake,
                "claim_reward": self.claim_reward,
                "get_pool_balance": self.get_pool_balance,
                "get_leaderboard": self.get_leaderboard,
            }
        )

    async def evaluate_agent(self, model_path: str, num_games: int = 5) -> List[float]:
        """Evaluate a Pong agent over multiple games."""
        model = DQN.load(model_path)
        env = gym.make("ALE/Pong-v5", render_mode="human")
        scores = []

        with console.status("[bold green]Evaluating agent...") as status:
            for game in range(num_games):
                obs, _ = env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated

                scores.append(episode_reward)
                status.update(f"[bold green]Game {game + 1}/{num_games}: Score {episode_reward}")

        env.close()
        return scores

    def display_leaderboard(self, entries: List[dict], title: str = "🏆 Pong Arcade Leaderboard"):
        """Display the leaderboard in a rich table format."""
        table = Table(title=title)
        table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
        table.add_column("Player", style="magenta")
        table.add_column("Best Score", justify="right", style="green")
        table.add_column("Win Rate", justify="right", style="yellow")
        table.add_column("Total Earned", justify="right", style="blue")
        table.add_column("Games", justify="right", style="cyan")
        table.add_column("Best Multiplier", justify="right", style="green")

        for i, entry in enumerate(entries, 1):
            table.add_row(
                str(i),
                entry["account_id"],
                str(entry["best_score"]),
                f"{entry['win_rate']*100:.1f}%",
                f"{entry['total_earned']/1e24:.2f} Ⓝ",
                str(entry["games_played"]),
                f"{entry['highest_reward_multiplier']}x"
            )

        console.print(table)

    async def stake(self, amount: float, target_score: int):
        """Stake NEAR on achieving a target score."""
        amount_yocto = int(amount * 1e24)
        return await self.contract.stake(
            args={"target_score": target_score},
            amount=amount_yocto
        )

    async def claim_reward(self, achieved_score: int):
        """Claim reward for achieving score."""
        return await self.contract.claim_reward(
            args={"achieved_score": achieved_score}
        )

    async def get_pool_balance(self) -> float:
        """Get current pool balance in NEAR."""
        balance = await self.contract.get_pool_balance()
        return float(balance) / 1e24

    async def get_leaderboard(self, limit: int = 10) -> List[dict]:
        """Get current leaderboard entries."""
        return await self.contract.get_leaderboard(
            args={"from_index": 0, "limit": limit}
        )

    async def get_top_players(self, limit: int = 10) -> List[dict]:
        """Get top players by score."""
        return await self.contract.get_top_players(
            args={"limit": limit}
        )

    async def get_player_stats(self, account_id: str) -> dict:
        """Get detailed stats for a specific player."""
        return await self.contract.get_player_stats(
            args={"account_id": account_id}
        )

    async def get_recent_games(self, limit: int = 5) -> List[dict]:
        """Get recently played games."""
        return await self.contract.get_recent_games(
            args={"limit": limit}
        )

@click.group()
def cli():
    """Pong Arcade - Stake NEAR on your Pong agent's performance!"""
    pass

@cli.command()
@click.option("--model-path", type=click.Path(exists=True), required=True,
              help="Path to trained model file")
@click.option("--amount", type=float, required=True,
              help="Amount of NEAR to stake")
@click.option("--target-score", type=int, required=True,
              help="Target score to achieve")
def stake(model_path: str, amount: float, target_score: int):
    """Stake NEAR on your agent achieving a target score."""
    async def run():
        arcade = PongArcade()
        await arcade.connect()

        # Show current pool stats
        pool_balance = await arcade.get_pool_balance()
        console.print(f"\n💰 Current Pool Balance: {pool_balance:.2f} Ⓝ")

        # Show leaderboard
        leaderboard = await arcade.get_leaderboard()
        arcade.display_leaderboard(leaderboard)

        # Confirm stake
        if not click.confirm(f"\nStake {amount} Ⓝ on achieving score {target_score}?"):
            return

        # Place stake
        console.print("\n[bold yellow]Placing stake...[/]")
        await arcade.stake(amount, target_score)

        # Evaluate agent
        console.print("\n[bold green]Starting evaluation...[/]")
        scores = await arcade.evaluate_agent(model_path)
        best_score = max(scores)

        # Process reward
        console.print(f"\n🎯 Best Score: {best_score}")
        if best_score >= target_score:
            console.print("[bold green]Claiming reward...[/]")
            result = await arcade.claim_reward(best_score)
            console.print(f"💰 Reward claimed: {result/1e24:.2f} Ⓝ")
        else:
            console.print("[bold red]Target score not achieved. Stake lost.[/]")

        # Show updated leaderboard
        console.print("\n[bold]Updated Leaderboard:[/]")
        leaderboard = await arcade.get_leaderboard()
        arcade.display_leaderboard(leaderboard)

    asyncio.run(run())

@cli.command()
def login():
    """Login to NEAR wallet and save credentials."""
    # Implementation for NEAR wallet login
    pass

@cli.group()
def leaderboard():
    """Leaderboard commands."""
    pass

@leaderboard.command(name="top")
@click.option("--limit", default=10, help="Number of entries to show")
def show_top_players(limit: int):
    """Show top players by score."""
    async def run():
        arcade = PongArcade()
        await arcade.connect()
        entries = await arcade.get_top_players(limit)
        arcade.display_leaderboard(entries, "🏆 Top Players")

    asyncio.run(run())

@leaderboard.command(name="recent")
@click.option("--limit", default=5, help="Number of recent games to show")
def show_recent_games(limit: int):
    """Show recently played games."""
    async def run():
        arcade = PongArcade()
        await arcade.connect()
        entries = await arcade.get_recent_games(limit)
        arcade.display_leaderboard(entries, "🎮 Recent Games")

    asyncio.run(run())

@leaderboard.command(name="player")
@click.argument("account_id")
def show_player_stats(account_id: str):
    """Show detailed stats for a specific player."""
    async def run():
        arcade = PongArcade()
        await arcade.connect()
        stats = await arcade.get_player_stats(account_id)
        
        if not stats:
            console.print(f"[red]No stats found for {account_id}[/]")
            return

        console.print(f"\n[bold]Player Stats: {account_id}[/]")
        console.print(f"Best Score: {stats['best_score']}")
        console.print(f"Win Rate: {stats['win_rate']*100:.1f}%")
        console.print(f"Total Earned: {stats['total_earned']/1e24:.2f} Ⓝ")
        console.print(f"Games Played: {stats['games_played']}")
        console.print(f"Best Reward Multiplier: {stats['highest_reward_multiplier']}x")

    asyncio.run(run())

if __name__ == "__main__":
    cli() 