"""NEAR wallet integration."""
import os
import json
import platform
import subprocess
from pathlib import Path
from typing import Optional
from loguru import logger
from pydantic import BaseModel

class WalletConfig(BaseModel):
    """NEAR wallet configuration."""
    network: str = "testnet"
    account_id: Optional[str] = None
    node_url: str = "https://rpc.testnet.near.org"
    contract_id: str = "agent-arcade.testnet"  # Default testnet contract

    def __init__(self, **data):
        super().__init__(**data)
        # Set contract ID based on network
        if self.network == "mainnet":
            self.contract_id = "agent-arcade.near"
        else:
            self.contract_id = "agent-arcade.testnet"

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
        self._check_near_credentials()
    
    def _get_config_dir(self) -> Path:
        """Get platform-specific config directory."""
        if os.name == 'nt':  # Windows
            return Path(os.getenv('APPDATA')) / 'agent-arcade'
        elif platform.system() == 'Darwin':  # macOS
            return Path.home() / 'Library' / 'Application Support' / 'agent-arcade'
        else:  # Linux and others
            return Path.home() / '.config' / 'agent-arcade'
    
    def _check_near_credentials(self) -> None:
        """Check NEAR credentials using CLI."""
        try:
            # Try to get account ID from NEAR CLI
            cmd = ['near', 'account', 'list', 'network-config', self.config.network]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse account ID from output
                for line in result.stdout.split('\n'):
                    if line.strip():
                        account_id = line.strip()
                        self.config.account_id = account_id
                        self._save_config()
                        logger.debug(f"Found NEAR credentials for {account_id}")
                        break
        except Exception as e:
            logger.debug(f"Failed to check NEAR credentials: {e}")
    
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
    
    def login(self, account_id: Optional[str] = None) -> bool:
        """Login to NEAR wallet using web-based flow.
        
        Args:
            account_id: Optional specific account ID to use
            
        Returns:
            True if login successful
        """
        try:
            # If account_id is provided, use it directly
            if account_id:
                self.config.account_id = account_id
                self._save_config()
                logger.info(f"Using account {account_id}")
                return True
                
            # Otherwise, try to get account from CLI
            cmd = ['near', 'account', 'list', 'network-config', self.config.network]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse account ID from output
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.config.account_id = line.strip()
                        self._save_config()
                        logger.info(f"Using account {self.config.account_id}")
                        return True
            
            logger.error("No account found. Please provide an account ID.")
            return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def logout(self) -> None:
        """Log out from NEAR wallet."""
        try:
            if self.config.account_id:
                cmd = ['near', 'account', 'delete-key', self.config.account_id, 'network-config', self.config.network]
                subprocess.run(cmd, capture_output=True, text=True)
            
            self.config.account_id = None
            self._save_config()
            logger.info("Successfully logged out")
        except Exception as e:
            logger.error(f"Logout failed: {e}")
    
    def is_logged_in(self) -> bool:
        """Check if wallet is logged in."""
        if not self.config.account_id:
            return False
            
        try:
            cmd = ['near', 'state', self.config.account_id]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def get_balance(self) -> Optional[float]:
        """Get NEAR balance for current account."""
        if not self.config.account_id:
            return None
        
        try:
            cmd = ['near', 'account', 'view-account-summary', self.config.account_id, 'network-config', self.config.network, 'now']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Parse balance from output
                for line in result.stdout.split('\n'):
                    if 'Native account balance' in line:
                        # Extract balance value (e.g., " Native account balance           10.00 NEAR ")
                        parts = line.split()
                        for part in parts:
                            try:
                                return float(part)
                            except ValueError:
                                continue
            return None
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None 