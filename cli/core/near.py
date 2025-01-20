import os
import json
import subprocess
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from decimal import Decimal
from py_near.account import Account
from py_near.providers import JsonProvider
from py_near.dapps.core import NEAR

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

class NEARWallet:
    def __init__(self, network: str = "testnet"):
        self.network = network
        self.account_id: Optional[str] = None
        self.account: Optional[Account] = None
        self.rpc_addr = "https://rpc.testnet.near.org" if network == "testnet" else "https://rpc.mainnet.near.org"
        self.near_credentials_dir = os.path.expanduser('~/.near-credentials')
        
    def _get_credentials_from_near_cli(self, account_id: str) -> Optional[str]:
        """Get credentials from NEAR CLI credentials directory"""
        creds_path = os.path.join(self.near_credentials_dir, self.network, f"{account_id}.json")
        if os.path.exists(creds_path):
            with open(creds_path, 'r') as f:
                creds = json.load(f)
                return creds.get('private_key')
        return None
        
    async def login_with_cli(self, account_id: Optional[str] = None) -> Tuple[bool, str]:
        """Login using NEAR CLI"""
        try:
            # Check if NEAR CLI is installed
            try:
                subprocess.run(['near', '--version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False, "NEAR CLI not found. Please install it with: npm install -g near-cli"

            # If account_id provided, try to use existing credentials
            if account_id:
                private_key = self._get_credentials_from_near_cli(account_id)
                if private_key:
                    success = await self.login(private_key)
                    if success:
                        self.account_id = account_id
                        return True, f"Successfully logged in as {account_id}!"
            
            # Start web wallet login process
            print("\nStarting NEAR web wallet login...")
            print("Please complete the authentication in your browser.")
            print("After authenticating, the credentials will be saved automatically.\n")
            
            cmd = ['near', 'login']
            if self.network != "testnet":
                cmd.extend(['--networkId', self.network])
                
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                error_msg = process.stderr or process.stdout
                return False, f"NEAR CLI login failed: {error_msg}"
                
            # After web login, user needs to provide their account ID if not already provided
            if not account_id:
                return False, "Please provide your NEAR account ID to complete login"
                
            # Get credentials for the account
            private_key = self._get_credentials_from_near_cli(account_id)
            if not private_key:
                return False, "Could not find credentials after login. Please try again or check your account ID."
                
            success = await self.login(private_key)
            if success:
                self.account_id = account_id
                return True, f"Successfully logged in as {account_id}!"
            return False, "Failed to initialize account with credentials"
            
        except Exception as e:
            return False, f"Login failed: {str(e)}"
            
    async def login(self, private_key: str) -> bool:
        """Handle NEAR wallet login using private key"""
        try:
            self.account = Account(
                account_id=self.account_id,
                private_key=private_key,
                rpc_addr=self.rpc_addr
            )
            await self.account.startup()
            self.account_id = self.account.account_id
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False
            
    async def get_balance(self) -> Decimal:
        """Get wallet balance in NEAR"""
        if not self.account:
            raise ValueError("Not logged in")
        balance = await self.account.get_balance()
        return Decimal(str(balance)) / NEAR

class NEARContract:
    def __init__(self, contract_id: str, wallet: NEARWallet):
        self.contract_id = contract_id
        self.wallet = wallet
        
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