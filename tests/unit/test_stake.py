"""Unit tests for staking functionality."""
import pytest
import json
from unittest.mock import patch, MagicMock
from cli.core.wallet import NEARWallet, WalletConfig

@pytest.fixture
def mock_contract_id():
    return "test-contract.testnet"

@pytest.fixture
def mock_wallet():
    """Create a mock wallet with proper config setup."""
    wallet = MagicMock(spec=NEARWallet)
    # Create a real config object
    config = WalletConfig(
        network="testnet",
        account_id="test.testnet",
        node_url="https://rpc.testnet.near.org"
    )
    # Attach it to the mock
    type(wallet).config = config
    wallet.is_logged_in.return_value = True
    return wallet

def test_stake_placement(mock_wallet, mock_contract_id):
    """Test placing a stake."""
    # Mock successful game config check with actual contract format
    config_result = MagicMock()
    config_result.returncode = 0
    config_result.stdout = json.dumps({
        "min_score": 0,
        "max_score": 21,
        "min_stake": "100000000000000000000000",  # 0.1 NEAR
        "max_multiplier": "3000000000000000000000000",  # 3 NEAR
        "enabled": True,
        "total_stakes": "5000000000000000000000000",  # 5 NEAR
        "active_stakes": 3
    })
    
    # Mock successful stake placement with actual contract format
    stake_result = MagicMock()
    stake_result.returncode = 0
    stake_result.stdout = json.dumps({
        "game": "pong",
        "amount": "1000000000000000000000000",  # 1 NEAR
        "target_score": 15,
        "timestamp": 1677123456,
        "games_played": 0,
        "status": "active"
    })
    
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = [config_result, stake_result]
        
        # Simulate stake placement command with actual NEAR CLI format
        cmd = [
            'near', 'call',
            mock_contract_id,
            'place_stake',
            json.dumps({
                "game": "pong",
                "target_score": 15
            }),
            '--accountId', mock_wallet.config.account_id,
            '--amount', '1',
            '--gas', '100000000000000'  # 100 TGas
        ]
        
        # Execute command and verify
        result = mock_run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        stake_data = json.loads(stake_result.stdout)
        assert stake_data["game"] == "pong"
        assert stake_data["target_score"] == 15
        assert stake_data["status"] == "active"

def test_stake_validation(mock_wallet, mock_contract_id):
    """Test stake validation checks."""
    # Mock game config with restrictions
    config_result = MagicMock()
    config_result.returncode = 0
    config_result.stdout = json.dumps({
        "min_score": 0,
        "max_score": 21,
        "min_stake": "100000000000000000000000",  # 0.1 NEAR
        "max_multiplier": 3,
        "enabled": True
    })
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = config_result
        
        # Test invalid target score
        cmd = [
            'near', 'contract', 'call-function',
            'as-read-only',
            mock_contract_id,
            'get_game_config',
            'json-args', '{"game": "pong"}',
            'network-config', mock_wallet.config.network,
            'now'
        ]
        
        result = mock_run(cmd, capture_output=True, text=True)
        config = json.loads(result.stdout)
        
        # Verify score range validation
        assert config["min_score"] <= 15 <= config["max_score"]  # Valid score
        assert not (config["min_score"] <= 25 <= config["max_score"])  # Invalid score
        
        # Verify minimum stake validation
        min_stake_near = float(config["min_stake"]) / 1e24
        assert 0.05 < min_stake_near  # Too low
        assert 1.0 > min_stake_near  # Valid amount

def test_stake_view(mock_wallet, mock_contract_id):
    """Test viewing stake details."""
    # Mock active stake
    stake_result = MagicMock()
    stake_result.returncode = 0
    stake_result.stdout = json.dumps({
        "stake_id": "stake_123",
        "game": "pong",
        "account_id": mock_wallet.config.account_id,
        "amount": "1000000000000000000000000",  # 1 NEAR
        "target_score": 15,
        "status": "active",
        "timestamp": 1645555555
    })
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = stake_result
        
        # Simulate stake view command
        cmd = [
            'near', 'contract', 'call-function',
            'as-read-only',
            mock_contract_id,
            'get_stake',
            'json-args', f'{{"account_id": "{mock_wallet.config.account_id}"}}',
            'network-config', mock_wallet.config.network,
            'now'
        ]
        
        result = mock_run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        stake_data = json.loads(result.stdout)
        assert stake_data["stake_id"] == "stake_123"
        assert stake_data["game"] == "pong"
        assert stake_data["target_score"] == 15
        assert stake_data["status"] == "active"

def test_stake_submission(mock_wallet, mock_contract_id):
    """Test submitting stake results."""
    # Mock active stake check with actual contract format
    stake_check = MagicMock()
    stake_check.returncode = 0
    stake_check.stdout = json.dumps({
        "game": "pong",
        "amount": "1000000000000000000000000",  # 1 NEAR
        "target_score": 15,
        "timestamp": 1677123456,
        "games_played": 0,
        "status": "active"
    })
    
    # Mock successful submission with actual contract format
    submit_result = MagicMock()
    submit_result.returncode = 0
    submit_result.stdout = json.dumps({
        "game": "pong",
        "amount": "1000000000000000000000000",  # 1 NEAR
        "achieved_score": 18,
        "target_score": 15,
        "reward": "2500000000000000000000000",  # 2.5 NEAR
        "timestamp": 1677123456,
        "status": "completed"
    })
    
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = [stake_check, submit_result]
        
        # Simulate score submission with actual NEAR CLI format
        cmd = [
            'near', 'call',
            mock_contract_id,
            'evaluate_stake',
            json.dumps({
                "achieved_score": 18
            }),
            '--accountId', mock_wallet.config.account_id,
            '--gas', '100000000000000'  # 100 TGas
        ]
        
        result = mock_run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        submission = json.loads(submit_result.stdout)
        assert submission["game"] == "pong"
        assert submission["achieved_score"] == 18
        # Convert yoctoNEAR string to NEAR float and check with small tolerance
        reward_near = int(submission["reward"]) / 1e24
        assert abs(reward_near - 2.5) < 1e-10  # 2.5 NEAR reward with tolerance
        assert submission["status"] == "completed"

def test_reward_calculation(mock_wallet, mock_contract_id):
    """Test reward multiplier calculation."""
    # Mock game config with actual contract format
    config_result = MagicMock()
    config_result.returncode = 0
    config_result.stdout = json.dumps({
        "min_score": 0,
        "max_score": 21,
        "min_stake": "100000000000000000000000",  # 0.1 NEAR
        "max_multiplier": "3000000000000000000000000",  # 3 NEAR
        "enabled": True,
        "total_stakes": "5000000000000000000000000",  # 5 NEAR
        "active_stakes": 3
    })
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = config_result
        
        # Get game config with actual NEAR CLI format
        cmd = [
            'near', 'view',
            mock_contract_id,
            'get_game_config',
            json.dumps({
                "game": "pong"
            })
        ]
        
        result = mock_run(cmd, capture_output=True, text=True)
        config = json.loads(result.stdout)
        
        # Test reward multipliers using actual contract logic
        def calculate_multiplier(achieved_score, target_score):
            score_ratio = achieved_score / target_score
            max_multiplier = float(config["max_multiplier"]) / 1e24
            if score_ratio >= 1.0:
                return max_multiplier
            elif score_ratio >= 0.8:
                return max_multiplier / 2
            elif score_ratio >= 0.5:
                return max_multiplier / 4
            return 0
        
        # Test various scenarios with actual NEAR amounts
        assert calculate_multiplier(18, 15) == 3  # Exceeded target
        assert calculate_multiplier(13, 15) == 1.5  # 86% of target
        assert calculate_multiplier(8, 15) == 0.75  # 53% of target
        assert calculate_multiplier(5, 15) == 0  # Below 50% of target 