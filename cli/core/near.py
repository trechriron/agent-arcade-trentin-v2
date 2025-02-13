import os
import json
import subprocess
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from decimal import Decimal
from py_near.account import Account
from py_near.providers import JsonProvider
from py_near.dapps.core import NEAR
from pathlib import Path
from loguru import logger
from pydantic import BaseModel

@dataclass
class StakeInfo:
    amount: int
    target_score: int
    timestamp: int
    games_played: int

@dataclass
class LeaderboardEntry:
    account_id: str
    best_score: int
    total_earned: int
    games_played: int
    win_rate: float
    highest_reward_multiplier: int
    last_played: int

class WalletConfig(BaseModel):
    """NEAR wallet configuration."""
    network: str = "testnet"
    account_id: Optional[str] = None
    node_url: str = "https://rpc.testnet.near.org"

class StakeRecord(BaseModel):
    """Record of a stake placed on an agent."""
    game: str
    model_path: str
    amount: float
    target_score: float
    status: str = "pending"  # pending, completed, claimed
    transaction_hash: Optional[str] = None
    achieved_score: Optional[float] = None
    reward_multiplier: Optional[float] = None

class NEARWallet:
    """NEAR wallet integration with local state management."""
    
    def __init__(self, network: str = "testnet"):
        """Initialize NEAR wallet integration.
        
        Args:
            network: NEAR network to use (testnet/mainnet)
        """
        self.config = WalletConfig(network=network)
        self.config_dir = self._get_config_dir()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._load_config()
    
    def _get_config_dir(self) -> Path:
        """Get platform-specific config directory."""
        if os.name == 'nt':  # Windows
            return Path(os.getenv('APPDATA')) / 'agent-arcade'
        elif os.name == 'darwin':  # macOS
            return Path.home() / 'Library' / 'Application Support' / 'agent-arcade'
        else:  # Linux and others
            return Path.home() / '.config' / 'agent-arcade'
    
    def _load_config(self) -> None:
        """Load wallet configuration from disk."""
        config_path = self.config_dir / 'wallet_config.json'
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = json.load(f)
                self.config = WalletConfig(**data)
            except Exception as e:
                logger.error(f"Failed to load wallet config: {e}")
    
    def _save_config(self) -> None:
        """Save wallet configuration to disk."""
        config_path = self.config_dir / 'wallet_config.json'
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config.dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save wallet config: {e}")
    
    def login_with_cli(self, account_id: Optional[str] = None) -> bool:
        """Login to NEAR wallet using NEAR CLI.
        
        Args:
            account_id: Optional specific account ID to use
            
        Returns:
            True if login successful
        """
        try:
            cmd = ['near', 'login']
            if account_id:
                cmd.extend(['--accountId', account_id])
            if self.config.network != "mainnet":
                cmd.extend(['--networkId', self.config.network])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Update config with account ID if provided
                if account_id:
                    self.config.account_id = account_id
                    self._save_config()
                return True
            else:
                logger.error(f"Login failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def is_logged_in(self) -> bool:
        """Check if wallet is logged in."""
        try:
            cmd = ['near', 'state', self.config.account_id or '']
            if self.config.network != "mainnet":
                cmd.extend(['--networkId', self.config.network])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def get_balance(self) -> Optional[float]:
        """Get NEAR balance for current account."""
        if not self.config.account_id:
            return None
        
        try:
            cmd = ['near', 'view-account', self.config.account_id]
            if self.config.network != "mainnet":
                cmd.extend(['--networkId', self.config.network])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Parse balance from output
                for line in result.stdout.split('\n'):
                    if 'amount' in line:
                        amount = float(line.split(':')[1].strip().replace("'", ""))
                        return amount / 1e24  # Convert from yoctoNEAR to NEAR
            return None
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None
    
    def record_stake(self, stake: StakeRecord) -> None:
        """Record a stake in local storage.
        
        Args:
            stake: Stake record to save
        """
        stakes_dir = self.config_dir / 'stakes'
        stakes_dir.mkdir(exist_ok=True)
        
        # Save stake record
        stake_path = stakes_dir / f"{stake.transaction_hash or 'pending'}.json"
        with open(stake_path, 'w') as f:
            json.dump(stake.dict(), f, indent=2)
    
    def get_stakes(self, game: Optional[str] = None) -> list[StakeRecord]:
        """Get all recorded stakes.
        
        Args:
            game: Optional game to filter by
            
        Returns:
            List of stake records
        """
        stakes_dir = self.config_dir / 'stakes'
        if not stakes_dir.exists():
            return []
        
        stakes = []
        for stake_file in stakes_dir.glob('*.json'):
            try:
                with open(stake_file) as f:
                    data = json.load(f)
                    stake = StakeRecord(**data)
                    if game is None or stake.game == game:
                        stakes.append(stake)
            except Exception as e:
                logger.error(f"Failed to load stake {stake_file}: {e}")
        
        return sorted(stakes, key=lambda s: s.transaction_hash or '')

class NEARContract:
    def __init__(self, contract_id: str, wallet: NEARWallet):
        self.contract_id = contract_id
        self.wallet = wallet
        
    async def initialize_contract(self):
        """Initialize contract connection"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
        
        try:
            # Connect to contract
            self.contract = await self.wallet.account.load_contract(
                self.contract_id,
                {
                    'viewMethods': ['get_leaderboard', 'get_pool_stats'],
                    'changeMethods': ['place_stake', 'evaluate_stake']
                }
            )
            return True
        except Exception as e:
            logger.error(f"Contract initialization failed: {e}")
            return False
            
    async def place_stake(self, game: str, model_path: str, amount: float, target_score: float) -> bool:
        """Place a stake on agent performance"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
            
        try:
            # Convert NEAR to yoctoNEAR
            amount_yocto = int(amount * 1e24)
            
            # Call contract method
            result = await self.wallet.account.function_call(
                self.contract_id,
                "place_stake",
                {
                    "game": game,
                    "target_score": target_score,
                    "model_path": str(model_path)
                },
                amount_yocto
            )
            return True
        except Exception as e:
            logger.error(f"Stake placement failed: {e}")
            return False
            
    async def evaluate_stake(self, stake_id: str, achieved_score: float) -> Optional[float]:
        """Evaluate a stake and claim rewards if successful"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
            
        try:
            result = await self.wallet.account.function_call(
                self.contract_id,
                "evaluate_stake",
                {
                    "stake_id": stake_id,
                    "achieved_score": achieved_score
                }
            )
            
            # Parse reward amount from result
            reward = float(result) / 1e24  # Convert from yoctoNEAR to NEAR
            return reward
        except Exception as e:
            logger.error(f"Stake evaluation failed: {e}")
            return None

    async def stake(self, amount: Decimal, target_score: int) -> bool:
        """Stake NEAR tokens with a target score"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
        assert -21 <= target_score <= 21, "Invalid target score"
        
        try:
            # Convert NEAR to yoctoNEAR
            amount_yocto = int(amount * NEAR)
            
            # Call the stake method on the contract
            await self.wallet.account.function_call(
                self.contract_id,
                "stake",
                {"target_score": target_score},
                amount_yocto
            )
            return True
        except Exception as e:
            print(f"Staking failed: {e}")
            return False
        
    async def get_leaderboard(self, from_index: int = 0, limit: int = 10) -> List[LeaderboardEntry]:
        """Get current leaderboard"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
            
        try:
            result = await self.wallet.account.view_function(
                self.contract_id,
                "get_leaderboard",
                {"from_index": from_index, "limit": limit}
            )
            return [LeaderboardEntry(**entry) for entry in result]
        except Exception as e:
            print(f"Failed to fetch leaderboard: {e}")
            return []

    async def get_top_players(self, limit: int = 10) -> List[LeaderboardEntry]:
        """Get top players by score"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
            
        try:
            result = await self.wallet.account.view_function(
                self.contract_id,
                "get_top_players",
                {"limit": limit}
            )
            return [LeaderboardEntry(**entry) for entry in result]
        except Exception as e:
            print(f"Failed to fetch top players: {e}")
            return []

    async def get_player_stats(self, account_id: str) -> Optional[LeaderboardEntry]:
        """Get stats for a specific player"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
            
        try:
            result = await self.wallet.account.view_function(
                self.contract_id,
                "get_player_stats",
                {"account_id": account_id}
            )
            return LeaderboardEntry(**result) if result else None
        except Exception as e:
            print(f"Failed to fetch player stats: {e}")
            return None

    async def get_recent_games(self, limit: int = 10) -> List[LeaderboardEntry]:
        """Get recently played games"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
            
        try:
            result = await self.wallet.account.view_function(
                self.contract_id,
                "get_recent_games",
                {"limit": limit}
            )
            return [LeaderboardEntry(**entry) for entry in result]
        except Exception as e:
            print(f"Failed to fetch recent games: {e}")
            return []

    async def claim_reward(self, achieved_score: int) -> Decimal:
        """Claim earned rewards"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
            
        try:
            result = await self.wallet.account.function_call(
                self.contract_id,
                "claim_reward",
                {"achieved_score": achieved_score}
            )
            # Convert yoctoNEAR to NEAR
            return Decimal(str(result)) / NEAR
        except Exception as e:
            print(f"Failed to claim reward: {e}")
            return Decimal("0.0")

    async def get_pool_balance(self) -> Decimal:
        """Get total pool balance"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
            
        try:
            result = await self.wallet.account.view_function(
                self.contract_id,
                "get_pool_balance",
                {}
            )
            # Convert yoctoNEAR to NEAR
            return Decimal(str(result)) / NEAR
        except Exception as e:
            print(f"Failed to fetch pool balance: {e}")
            return Decimal("0.0")

    async def get_stake(self, account_id: Optional[str] = None) -> Optional[StakeInfo]:
        """Get stake information for an account"""
        if not self.wallet.account:
            raise ValueError("Wallet not logged in")
            
        account_id = account_id or self.wallet.account_id
        if not account_id:
            raise ValueError("No account specified")
            
        try:
            result = await self.wallet.account.view_function(
                self.contract_id,
                "get_stake",
                {"account_id": account_id}
            )
            return StakeInfo(**result) if result else None
        except Exception as e:
            print(f"Failed to fetch stake info: {e}")
            return None 