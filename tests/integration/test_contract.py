"""Integration tests for NEAR contract interaction."""
import pytest
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_contract_view_game_config(mock_wallet):
    """Test viewing game configuration from contract."""
    with patch('subprocess.run') as mock_run:
        # Mock successful contract call
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = json.dumps({
            "min_score": 0,
            "max_score": 21,
            "min_stake": "100000000000000000000000",  # 0.1 NEAR
            "max_stake": "10000000000000000000000000",  # 10 NEAR
            "max_multiplier": 3
        })
        mock_run.return_value = mock_process
        
        # Execute contract view command
        cmd = [
            'near', 'contract', 'call-function',
            'as-read-only',
            mock_wallet.config.contract_id,
            'get_game_config',
            'json-args', '{"game": "pong"}',
            'network-config', mock_wallet.config.network,
            'now'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Verify command execution
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == cmd
        
        # Parse and verify response
        config = json.loads(mock_process.stdout)
        assert config["min_score"] == 0
        assert config["max_score"] == 21
        assert config["min_stake"] == "100000000000000000000000"
        assert config["max_multiplier"] == 3

def test_contract_stake_placement(mock_wallet):
    """Test placing a stake through contract."""
    with patch('subprocess.run') as mock_run:
        # Mock successful stake placement
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = json.dumps({
            "success": True,
            "stake_id": "stake_123",
            "amount": "1000000000000000000000000",  # 1 NEAR
            "target_score": 15
        })
        mock_run.return_value = mock_process
        
        # Execute stake placement command
        cmd = [
            'near', 'contract', 'call-function',
            mock_wallet.config.contract_id,
            'place_stake',
            'json-args', json.dumps({
                "game": "pong",
                "target_score": 15
            }),
            '--amount', '1',
            'network-config', mock_wallet.config.network
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Verify command execution
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == cmd
        
        # Parse and verify response
        response = json.loads(mock_process.stdout)
        assert response["success"]
        assert response["stake_id"] == "stake_123"
        assert response["target_score"] == 15

def test_contract_stake_view(mock_wallet):
    """Test viewing stake status from contract."""
    with patch('subprocess.run') as mock_run:
        # Mock successful stake view
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = json.dumps({
            "stake_id": "stake_123",
            "game": "pong",
            "account_id": mock_wallet.config.account_id,
            "amount": "1000000000000000000000000",  # 1 NEAR
            "target_score": 15,
            "status": "active",
            "created_at": 1645555555,
            "expires_at": 1645641955
        })
        mock_run.return_value = mock_process
        
        # Execute stake view command
        cmd = [
            'near', 'contract', 'call-function',
            'as-read-only',
            mock_wallet.config.contract_id,
            'get_stake',
            'json-args', '{"stake_id": "stake_123"}',
            'network-config', mock_wallet.config.network,
            'now'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Verify command execution
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == cmd
        
        # Parse and verify response
        stake = json.loads(mock_process.stdout)
        assert stake["stake_id"] == "stake_123"
        assert stake["game"] == "pong"
        assert stake["account_id"] == mock_wallet.config.account_id
        assert stake["target_score"] == 15
        assert stake["status"] == "active"

def test_contract_submit_score(mock_wallet):
    """Test submitting score to contract."""
    with patch('subprocess.run') as mock_run:
        # Mock successful score submission
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = json.dumps({
            "success": True,
            "stake_id": "stake_123",
            "score": 18,
            "reward": "2500000000000000000000000"  # 2.5 NEAR (2.5x multiplier)
        })
        mock_run.return_value = mock_process
        
        # Execute score submission command
        cmd = [
            'near', 'contract', 'call-function',
            mock_wallet.config.contract_id,
            'submit_score',
            'json-args', json.dumps({
                "stake_id": "stake_123",
                "score": 18
            }),
            'network-config', mock_wallet.config.network
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Verify command execution
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == cmd
        
        # Parse and verify response
        response = json.loads(mock_process.stdout)
        assert response["success"]
        assert response["stake_id"] == "stake_123"
        assert response["score"] == 18
        assert response["reward"] == "2500000000000000000000000"

def test_contract_error_handling(mock_wallet):
    """Test contract error handling."""
    with patch('subprocess.run') as mock_run:
        # Mock contract error
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = "Smart contract panicked: Game not found"
        mock_run.return_value = mock_process
        
        # Execute invalid game config request
        cmd = [
            'near', 'contract', 'call-function',
            'as-read-only',
            mock_wallet.config.contract_id,
            'get_game_config',
            'json-args', '{"game": "invalid_game"}',
            'network-config', mock_wallet.config.network,
            'now'
        ]
        
        with pytest.raises(subprocess.CalledProcessError):
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Verify error handling
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == cmd
        assert "Game not found" in mock_process.stderr 