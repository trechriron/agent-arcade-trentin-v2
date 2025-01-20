import click
import asyncio
from typing import Optional
from pathlib import Path
import json
import os

class AgentArcadeCLI:
    def __init__(self):
        self.wallet = None
        self.contract = None
        self.config_dir = Path.home() / ".agent-arcade"
        self.config_dir.mkdir(exist_ok=True)
        self.credentials_file = self.config_dir / "credentials.json"
        
    def get_wallet(self):
        """Get or initialize wallet"""
        from .near import NEARWallet
        if not self.wallet:
            self.wallet = NEARWallet()
        return self.wallet
        
    def get_contract(self, game: str):
        """Get or initialize contract for specific game"""
        from .near import NEARContract
        if not self.contract:
            wallet = self.get_wallet()
            contract_id = f"{game}.agent-arcade.testnet"  # Example contract ID
            self.contract = NEARContract(contract_id, wallet)
        return self.contract

    def save_account(self, account_id: str):
        """Save NEAR account ID"""
        self.credentials_file.write_text(json.dumps({
            "account_id": account_id
        }))
        self.credentials_file.chmod(0o600)  # Secure file permissions

    def load_account(self) -> Optional[str]:
        """Load saved NEAR account ID"""
        if self.credentials_file.exists():
            data = json.loads(self.credentials_file.read_text())
            return data.get("account_id")
        return None

# Global CLI instance
cli = AgentArcadeCLI()

@click.group()
def agent_arcade():
    """Agent Arcade CLI - Train, Evaluate, and Compete with AI Agents"""
    pass

@agent_arcade.command()
@click.option('--account-id', help='Your NEAR account ID (e.g. alice.near)')
def login(account_id: Optional[str]):
    """Login to NEAR wallet using NEAR CLI"""
    wallet = cli.get_wallet()
    
    try:
        # Check for existing account
        saved_account = cli.load_account()
        if saved_account and not account_id:
            if click.confirm(f"Continue with saved account {saved_account}?", default=True):
                account_id = saved_account
        
        # Try to use existing credentials if account provided
        if account_id:
            click.echo(f"Checking for existing credentials for {account_id}...")
            success, msg = asyncio.run(wallet.login_with_cli(account_id))
            if success:
                cli.save_account(account_id)
                click.echo(msg)
                return
            elif "NEAR CLI not found" in msg:
                click.echo(msg)
                return
                
        # Start fresh NEAR CLI login
        click.echo("\nStarting NEAR web wallet login...")
        click.echo("1. Your browser will open to authenticate with NEAR")
        click.echo("2. Please complete the authentication in your browser")
        click.echo("3. Return here once you've finished\n")
        
        success, msg = asyncio.run(wallet.login_with_cli())
        
        if not success and "Please provide your NEAR account ID" in msg:
            if not account_id:
                account_id = click.prompt("Enter your NEAR account ID", type=str)
            success, msg = asyncio.run(wallet.login_with_cli(account_id))
            
        if success:
            cli.save_account(account_id)
            click.echo(f"\n{msg}")
            click.echo("\nYou can now use the following commands:")
            click.echo("  agent-arcade balance     - Check your NEAR balance")
            click.echo("  agent-arcade pong stake  - Stake NEAR on achieving a target score")
        else:
            click.echo(f"\nLogin failed: {msg}")
            if "NEAR CLI not found" in msg:
                click.echo("\nTo install NEAR CLI:")
                click.echo("1. Install Node.js from https://nodejs.org")
                click.echo("2. Run: npm install -g near-cli")
            
    except Exception as e:
        click.echo(f"Login failed: {e}")
        click.echo("\nPlease ensure you have NEAR CLI installed:")
        click.echo("npm install -g near-cli")

@agent_arcade.command()
def balance():
    """Check your NEAR wallet balance"""
    wallet = cli.get_wallet()
    try:
        balance = asyncio.run(wallet.get_balance())
        click.echo(f"Balance: {balance} NEAR")
    except Exception as e:
        click.echo(f"Failed to get balance: {e}")

@agent_arcade.command()
def pool():
    """Check the total pool balance"""
    contract = cli.get_contract("pong")  # Default to pong for now
    try:
        balance = asyncio.run(contract.get_pool_balance())
        click.echo(f"Pool Balance: {balance} NEAR")
    except Exception as e:
        click.echo(f"Failed to get pool balance: {e}") 