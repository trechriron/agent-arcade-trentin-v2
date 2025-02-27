"""NEAR wallet integration."""
import os
import json
import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
from pydantic import BaseModel

class WalletConfig(BaseModel):
    """NEAR wallet configuration."""
    network: str = "testnet"
    account_id: Optional[str] = None
    node_url: str = "https://rpc.testnet.near.org"
    contract_id: str = "near-agent-arcade.testnet"  # Default testnet contract
    secret_key: Optional[str] = None  # Secret key for verification tokens

    def __init__(self, **data):
        super().__init__(**data)
        # Set contract ID based on network
        if self.network == "mainnet":
            self.contract_id = "near-agent-arcade.near"
            self.node_url = "https://rpc.mainnet.near.org"
        else:
            self.contract_id = "near-agent-arcade.testnet"
            self.node_url = "https://rpc.testnet.near.org"

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
        elif platform.system() == 'Darwin':  # macOS
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

    def _check_near_cli(self) -> bool:
        """Check if NEAR CLI is installed and accessible."""
        try:
            result = subprocess.run(['near', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(
                "\nNEAR CLI not found. Please install it with one of these commands:"
                "\n  npm install -g near-cli"
                "\n  npm install -g near-cli-rs@latest"
                "\n  cargo install near-cli-rs"
            )
            return False

    def _get_credentials_path(self) -> Path:
        """Get path to NEAR credentials directory."""
        return Path.home() / '.near-credentials' / self.config.network

    def _find_account_from_credentials(self) -> Optional[str]:
        """Find first account ID from NEAR credentials."""
        creds_dir = self._get_credentials_path()
        if not creds_dir.exists():
            return None
            
        # Look for .json credential files
        for file in creds_dir.glob('*.json'):
            return file.stem  # Return first account found
        return None

    def login(self, account_id: Optional[str] = None) -> bool:
        """Login to NEAR wallet using web-based flow.
        
        Args:
            account_id: Optional specific account ID to use
            
        Returns:
            True if login successful
        """
        try:
            # First check if NEAR CLI is installed
            if not self._check_near_cli():
                return False

            # Start the NEAR CLI login flow
            logger.info("\nStarting NEAR web wallet login...")
            logger.info("1. Your browser will open to authenticate with NEAR")
            logger.info("2. Please complete the authentication in your browser")
            logger.info("3. Return here once you've finished\n")
            
            # Run the login command - it will open browser automatically
            cmd = ['near', 'login']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Login failed: {result.stderr}")
                return False
            
            # After login, check for credentials
            if account_id:
                # If account ID provided, verify it exists in credentials
                creds_path = self._get_credentials_path() / f"{account_id}.json"
                if not creds_path.exists():
                    logger.error(f"No credentials found for account {account_id}")
                    return False
                self.config.account_id = account_id
            else:
                # Otherwise use the first account found in credentials
                account_id = self._find_account_from_credentials()
                if not account_id:
                    logger.error("No account credentials found after login")
                    return False
                self.config.account_id = account_id
            
            self._save_config()
            
            # Verify we can access the account
            if not self.is_logged_in():
                logger.error("Login verification failed. Please try logging in again.")
                return False
                
            logger.info(f"Successfully logged in as {self.config.account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def logout(self) -> None:
        """Log out from NEAR wallet."""
        try:
            if self.config.account_id:
                # Remove local credentials
                creds_path = self._get_credentials_path() / f"{self.config.account_id}.json"
                if creds_path.exists():
                    creds_path.unlink()
            
            self.config.account_id = None
            self._save_config()
            logger.info("Successfully logged out")
        except Exception as e:
            logger.error(f"Logout failed: {e}")
    
    def is_logged_in(self) -> bool:
        """Check if wallet is logged in and we have valid credentials."""
        if not self.config.account_id:
            return False
            
        try:
            # Check if we have credentials
            creds_path = self._get_credentials_path() / f"{self.config.account_id}.json"
            if not creds_path.exists():
                return False
                
            # Verify we can access the account
            cmd = ['near', 'state', self.config.account_id]
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
            # Use near state with proper network flag
            cmd = ['near', 'state', self.config.account_id, '--networkId', self.config.network]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                try:
                    # Try to find the formattedAmount line in the output
                    for line in result.stdout.split('\n'):
                        if 'formattedAmount' in line:
                            # Extract the formatted amount value
                            # Format: "  formattedAmount: '123.45'"
                            amount_str = line.split("'")[1]  # Get value between quotes
                            return float(amount_str)
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse balance: {e}")
                    logger.debug(f"State output: {result.stdout}")
            return None
        except Exception as e:
            logger.debug(f"Failed to get balance: {e}")
            return None

    def get_secret_key(self) -> Optional[str]:
        """Get the secret key used for verification tokens."""
        return self.config.secret_key
    
    def save_secret_key(self, secret_key: str) -> None:
        """Save a new secret key for verification tokens."""
        self.config.secret_key = secret_key
        self._save_config() 