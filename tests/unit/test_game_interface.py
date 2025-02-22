"""Unit tests for the game interface."""
import pytest
from pathlib import Path
from cli.games.base import GameInterface, GameConfig
from cli.games.pong.game import PongGame

def test_game_initialization():
    """Test game interface initialization."""
    game = PongGame()
    assert game.name == "pong"
    assert game.env_id == "ALE/Pong-v5"
    assert isinstance(game.score_range, tuple)
    assert len(game.score_range) == 2

def test_default_config():
    """Test default configuration loading."""
    game = PongGame()
    config = game.get_default_config()
    assert isinstance(config, GameConfig)
    assert config.total_timesteps > 0
    assert 0 < config.learning_rate < 1
    assert config.buffer_size > 0
    assert config.frame_stack in [4, 16]

def test_reward_multiplier_calculation():
    """Test reward multiplier calculation."""
    game = PongGame()
    
    # Test exceptional performance (3.0x)
    # For score range (-21, 21), normalized score of 0.8 would be:
    # 0.8 * (21 - (-21)) + (-21) = 0.8 * 42 - 21 = 12.6
    high_score = 12.6
    assert game.calculate_reward_multiplier(high_score) == 3.0
    
    # Test great performance (2.0x)
    # For normalized score of 0.6:
    # 0.6 * 42 - 21 = 4.2
    good_score = 4.2
    assert game.calculate_reward_multiplier(good_score) == 2.0
    
    # Test good performance (1.5x)
    # For normalized score of 0.4:
    # 0.4 * 42 - 21 = -4.2
    ok_score = -4.2
    assert game.calculate_reward_multiplier(ok_score) == 1.5
    
    # Test base performance (1.0x)
    # For normalized score of 0.2:
    # 0.2 * 42 - 21 = -12.6
    low_score = -12.6
    assert game.calculate_reward_multiplier(low_score) == 1.0

def test_custom_config_loading(test_config_path):
    """Test loading custom configuration."""
    game = PongGame()
    
    # Create test config file
    test_config_path.parent.mkdir(parents=True, exist_ok=True)
    test_config_path.write_text("""
total_timesteps: 500000
learning_rate: 0.0001
buffer_size: 100000
learning_starts: 25000
batch_size: 128
exploration_fraction: 0.1
target_update_interval: 1000
frame_stack: 4
    """.strip())
    
    config = game.load_config(test_config_path)
    assert config.total_timesteps == 500000
    assert config.learning_rate == 0.0001
    assert config.buffer_size == 100000
    assert config.frame_stack == 4

def test_invalid_config_loading(test_config_path):
    """Test loading invalid configuration falls back to defaults."""
    game = PongGame()
    
    # Create invalid config file
    test_config_path.parent.mkdir(parents=True, exist_ok=True)
    test_config_path.write_text("invalid: yaml: content")
    
    config = game.load_config(test_config_path)
    assert isinstance(config, GameConfig)
    assert config == game.get_default_config()

def test_model_validation(test_model_path):
    """Test model validation."""
    game = PongGame()
    
    # Test with non-existent model
    assert not game.validate_model(test_model_path)
    
    # TODO: Add test with valid model file once we have test data 