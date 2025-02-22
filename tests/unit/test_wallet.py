"""Unit tests for the NEAR wallet integration."""
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from cli.core.wallet import NEARWallet, WalletConfig

@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory."""
    with patch('cli.core.wallet.NEARWallet._get_config_dir') as mock_dir:
        mock_dir.return_value = tmp_path
        yield tmp_path

@pytest.fixture
def wallet(temp_config_dir):
    """Create a test wallet instance."""
    return NEARWallet()

def test_wallet_initialization(wallet):
    """Test wallet initialization."""
    assert isinstance(wallet.config, WalletConfig)
    assert wallet.config.network == "testnet"
    assert wallet.config.account_id is None
    assert wallet.config.node_url == "https://rpc.testnet.near.org"

def test_config_persistence(temp_config_dir):
    """Test wallet configuration persistence."""
    # Create wallet and set config
    wallet = NEARWallet()
    wallet.config.account_id = "test.testnet"
    wallet.config.network = "testnet"
    wallet._save_config()
    
    # Create new wallet instance and verify config loaded
    new_wallet = NEARWallet()
    assert new_wallet.config.account_id == "test.testnet"
    assert new_wallet.config.network == "testnet"

def test_login_success(wallet):
    """Test successful wallet login."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = """
    Logged in as [ test.testnet ] with public key [ ed25519:AbCdEf123... ] in keychain [ /Users/username/.near-credentials ]
    Please see your seed phrase at: /Users/username/.near-credentials/testnet/test.testnet.json
    """
    
    with patch('subprocess.run', return_value=mock_result):
        success = wallet.login()
        assert success
        assert wallet.config.account_id == "test.testnet"

def test_login_failure(wallet):
    """Test failed wallet login."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "Login failed"
    
    with patch('subprocess.run', return_value=mock_result):
        success = wallet.login()
        assert not success
        assert wallet.config.account_id is None

def test_login_with_account(wallet):
    """Test login with specific account ID."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Logged in successfully"
    
    with patch('subprocess.run', return_value=mock_result):
        success = wallet.login("specific.testnet")
        assert success
        assert wallet.config.account_id == "specific.testnet"

def test_logout(wallet):
    """Test wallet logout."""
    # Setup initial logged in state
    wallet.config.account_id = "test.testnet"
    wallet._save_config()
    
    with patch('subprocess.run') as mock_run:
        wallet.logout()
        assert wallet.config.account_id is None
        mock_run.assert_called_once()

def test_is_logged_in(wallet):
    """Test login status check."""
    wallet.config.account_id = "test.testnet"
    
    # Test successful state check
    mock_result = MagicMock()
    mock_result.returncode = 0
    with patch('subprocess.run', return_value=mock_result):
        assert wallet.is_logged_in()
    
    # Test failed state check
    mock_result.returncode = 1
    with patch('subprocess.run', return_value=mock_result):
        assert not wallet.is_logged_in()
    
    # Test without account ID
    wallet.config.account_id = None
    assert not wallet.is_logged_in()

def test_get_balance(wallet):
    """Test balance retrieval."""
    wallet.config.account_id = "test.testnet"
    
    # Test successful balance check with actual NEAR CLI format
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = """
    Account test.testnet
    amount: 1000000000000000000000000
    locked: 0
    code_hash: 11111111111111111111111111111111
    storage_usage: 642
    storage_paid_at: 0
    block_height: 12345
    block_hash: 11111111111111111111111111111111
    """
    
    with patch('subprocess.run', return_value=mock_result):
        balance = wallet.get_balance()
        assert balance == 1.0  # 1 NEAR
    
    # Test failed balance check
    mock_result.returncode = 1
    mock_result.stderr = "Account test.testnet does not exist"
    with patch('subprocess.run', return_value=mock_result):
        balance = wallet.get_balance()
        assert balance is None
    
    # Test without account ID
    wallet.config.account_id = None
    assert wallet.get_balance() is None

def test_config_dir_paths(tmp_path):
    """Test config directory paths for different platforms."""
    # Test macOS path
    with patch('platform.system', return_value='Darwin'), \
         patch('pathlib.Path.home') as mock_home:
        mock_home.return_value = tmp_path
        wallet = NEARWallet()
        expected_path = tmp_path / 'Library' / 'Application Support' / 'agent-arcade'
        assert str(wallet.config_dir) == str(expected_path)
    
    # Test Linux path
    with patch('platform.system', return_value='Linux'), \
         patch('pathlib.Path.home') as mock_home:
        mock_home.return_value = tmp_path
        wallet = NEARWallet()
        expected_path = tmp_path / '.config' / 'agent-arcade'
        assert str(wallet.config_dir) == str(expected_path) 