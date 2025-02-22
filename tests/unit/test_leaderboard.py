"""Unit tests for the leaderboard system."""
import pytest
import time
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from cli.core.leaderboard import (
    LeaderboardManager, 
    GameLeaderboard, 
    LeaderboardEntry,
    get_leaderboard_dir
)

@pytest.fixture(autouse=True)
def temp_leaderboard_dir():
    """Create a temporary directory for leaderboard storage."""
    temp_dir = tempfile.mkdtemp()
    os.environ["AGENT_ARCADE_LEADERBOARD_DIR"] = temp_dir
    
    # Clean up any existing files
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    
    yield temp_dir
    
    # Clean up after tests
    shutil.rmtree(temp_dir)
    os.environ.pop("AGENT_ARCADE_LEADERBOARD_DIR", None)

@pytest.fixture
def test_entry():
    return LeaderboardEntry(
        account_id="test_account",
        score=100.0,
        success_rate=0.8,
        episodes=10,
        timestamp=datetime.now().timestamp()
    )

def test_leaderboard_entry_creation(test_entry):
    """Test creating a leaderboard entry."""
    assert test_entry.account_id == "test_account"
    assert test_entry.score == 100.0
    assert test_entry.success_rate == 0.8
    assert test_entry.episodes == 10
    assert isinstance(test_entry.timestamp, float)

def test_game_leaderboard_initialization(temp_leaderboard_dir):
    """Test initializing a game leaderboard."""
    leaderboard = GameLeaderboard("test_game")
    assert len(leaderboard.entries) == 0
    assert leaderboard.game_name == "test_game"
    
    # File should not exist until entries are added
    path = os.path.join(get_leaderboard_dir(), "test_game.json")
    assert not os.path.exists(path)
    
    # Add an entry and verify file is created
    entry = LeaderboardEntry(
        account_id="test_account",
        score=100.0,
        success_rate=0.8,
        episodes=10,
        timestamp=datetime.now().timestamp()
    )
    leaderboard.add_entry(entry)
    assert os.path.exists(path)

def test_game_leaderboard_add_entry(temp_leaderboard_dir, test_entry):
    """Test adding an entry to the game leaderboard."""
    leaderboard = GameLeaderboard("test_game")
    leaderboard.add_entry(test_entry)
    assert len(leaderboard.entries) == 1
    assert leaderboard.entries[0].account_id == test_entry.account_id

def test_game_leaderboard_sorting(temp_leaderboard_dir):
    """Test leaderboard entry sorting."""
    leaderboard = GameLeaderboard("test_game")
    entries = [
        LeaderboardEntry("player1", 10.0, 0.6, 10, datetime.now().timestamp()),
        LeaderboardEntry("player2", 20.0, 0.9, 10, datetime.now().timestamp()),
        LeaderboardEntry("player3", 15.0, 0.7, 10, datetime.now().timestamp())
    ]
    
    # Add entries in random order
    leaderboard.add_entry(entries[2])  # 15.0
    leaderboard.add_entry(entries[0])  # 10.0
    leaderboard.add_entry(entries[1])  # 20.0
    
    # Verify sorting (highest score first)
    assert len(leaderboard.entries) == 3
    assert leaderboard.entries[0].score == 20.0
    assert leaderboard.entries[1].score == 15.0
    assert leaderboard.entries[2].score == 10.0

def test_game_leaderboard_top_scores(temp_leaderboard_dir):
    """Test retrieving top scores."""
    leaderboard = GameLeaderboard("test_game")
    entries = [
        LeaderboardEntry("player1", 100.0, 0.8, 10, datetime.now().timestamp()),
        LeaderboardEntry("player2", 200.0, 0.9, 10, datetime.now().timestamp()),
        LeaderboardEntry("player3", 150.0, 0.7, 10, datetime.now().timestamp())
    ]
    for entry in entries:
        leaderboard.add_entry(entry)
    
    top_scores = leaderboard.get_top_scores(2)
    assert len(top_scores) == 2
    assert top_scores[0].score == 200.0
    assert top_scores[1].score == 150.0

def test_leaderboard_manager(temp_leaderboard_dir):
    """Test leaderboard manager functionality."""
    manager = LeaderboardManager()
    leaderboard = manager.get_leaderboard("test_game")
    assert isinstance(leaderboard, GameLeaderboard)
    assert leaderboard.game_name == "test_game"
    assert len(leaderboard.entries) == 0
    
    # Test getting the same leaderboard again
    same_leaderboard = manager.get_leaderboard("test_game")
    assert leaderboard is same_leaderboard  # Should return cached instance

def test_game_leaderboard_recent_entries(temp_leaderboard_dir):
    """Test retrieving recent entries."""
    leaderboard = GameLeaderboard("test_game")
    
    # Add entries with different timestamps
    entries = []
    for i in range(5):
        entry = LeaderboardEntry(
            account_id=f"player{i}",
            score=float(i),
            success_rate=0.5,
            episodes=10,
            timestamp=datetime.now().timestamp() + i,
            model_path=""
        )
        entries.append(entry)
        leaderboard.add_entry(entry)
    
    recent = leaderboard.get_recent_entries(3)
    assert len(recent) == 3
    # Most recent (highest timestamp) should be first
    assert recent[0].account_id == "player4"
    assert recent[1].account_id == "player3"
    assert recent[2].account_id == "player2" 