"""Test configuration and fixtures for Agent Arcade."""
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from cli.core.wallet import NEARWallet
from cli.core.leaderboard import LeaderboardManager

@pytest.fixture
def test_data_dir():
    """Create and return a temporary test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def mock_wallet():
    """Create a mock NEAR wallet for testing."""
    wallet = MagicMock(spec=NEARWallet)
    wallet.is_logged_in.return_value = True
    wallet.config.account_id = "test.testnet"
    wallet.config.network = "testnet"
    wallet.config.contract_id = "test-contract.testnet"
    return wallet

@pytest.fixture
def mock_leaderboard():
    """Create a mock leaderboard manager for testing."""
    return MagicMock(spec=LeaderboardManager)

@pytest.fixture
def test_model_path(test_data_dir):
    """Return a test model path."""
    return test_data_dir / "test_model.zip"

@pytest.fixture
def test_config_path(test_data_dir):
    """Return a test configuration path."""
    return test_data_dir / "test_config.yaml"

@pytest.fixture
def env_setup():
    """Set up environment variables for testing."""
    os.environ["NEAR_ENV"] = "testnet"
    os.environ["AGENT_ARCADE_LOG_LEVEL"] = "DEBUG"
    yield
    del os.environ["NEAR_ENV"]
    del os.environ["AGENT_ARCADE_LOG_LEVEL"] 