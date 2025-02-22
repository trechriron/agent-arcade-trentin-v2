"""Leaderboard management for Agent Arcade."""
import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
from pydantic import BaseModel
from loguru import logger
from dataclasses import dataclass, asdict
from datetime import datetime

def get_leaderboard_dir() -> str:
    """Get the leaderboard directory path."""
    return os.getenv(
        "AGENT_ARCADE_LEADERBOARD_DIR",
        os.path.join(os.path.expanduser("~"), ".agent-arcade", "leaderboards")
    )

@dataclass
class LeaderboardEntry:
    """Entry in the leaderboard."""
    account_id: str
    score: float
    success_rate: float
    episodes: int
    timestamp: float
    model_path: str = ""

class GameLeaderboard:
    """Leaderboard for a specific game."""
    
    def __init__(self, game_name: str):
        """Initialize game leaderboard.
        
        Args:
            game_name: Game identifier
        """
        self.game_name = game_name
        self.entries: List[LeaderboardEntry] = []
        self._load_entries()
    
    def _load_entries(self):
        """Load entries from disk."""
        leaderboard_dir = get_leaderboard_dir()
        os.makedirs(leaderboard_dir, exist_ok=True)
        path = os.path.join(leaderboard_dir, f"{self.game_name}.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.entries = [LeaderboardEntry(**entry) for entry in data]
            except (json.JSONDecodeError, FileNotFoundError):
                # If file is corrupted or missing, start with empty entries
                self.entries = []
        else:
            self.entries = []
    
    def _save_entries(self):
        """Save entries to disk."""
        leaderboard_dir = get_leaderboard_dir()
        os.makedirs(leaderboard_dir, exist_ok=True)
        path = os.path.join(leaderboard_dir, f"{self.game_name}.json")
        with open(path, 'w') as f:
            json.dump([asdict(entry) for entry in self.entries], f)
    
    def add_entry(self, entry: LeaderboardEntry):
        """Add a new entry to the leaderboard."""
        self.entries.append(entry)
        self.entries.sort(key=lambda x: (-x.score, -x.success_rate))
        self._save_entries()
    
    def get_top_scores(self, limit: int = 10) -> List[LeaderboardEntry]:
        """Get top N scores."""
        return self.entries[:limit]
    
    def get_recent_entries(self, limit: int = 10) -> List[LeaderboardEntry]:
        """Get most recent N entries."""
        sorted_entries = sorted(self.entries, key=lambda x: -x.timestamp)
        return sorted_entries[:limit]
    
    def get_player_best(self, account_id: str) -> Optional[LeaderboardEntry]:
        """Get player's best entry."""
        player_entries = [e for e in self.entries if e.account_id == account_id]
        return max(player_entries, key=lambda x: x.score) if player_entries else None
    
    def get_player_rank(self, account_id: str) -> Optional[int]:
        """Get player's rank based on their best score."""
        best_entry = self.get_player_best(account_id)
        if not best_entry:
            return None
        return next(i for i, e in enumerate(self.entries, 1) 
                   if e.score == best_entry.score and e.account_id == account_id)

class LeaderboardManager:
    """Manages leaderboards for all games."""
    
    def __init__(self):
        """Initialize leaderboard manager."""
        self._leaderboards: Dict[str, GameLeaderboard] = {}
    
    def get_leaderboard(self, game_name: str) -> GameLeaderboard:
        """Get leaderboard for a game.
        
        Args:
            game_name: Game identifier
            
        Returns:
            Game leaderboard
        """
        if game_name not in self._leaderboards:
            self._leaderboards[game_name] = GameLeaderboard(game_name)
        return self._leaderboards[game_name]
    
    def record_score(self, game_name: str, account_id: str, score: float, 
                    success_rate: float, episodes: int, model_path: str = ""):
        """Record a new score in the leaderboard."""
        entry = LeaderboardEntry(
            account_id=account_id,
            score=score,
            success_rate=success_rate,
            episodes=episodes,
            timestamp=time.time(),
            model_path=model_path
        )
        self.get_leaderboard(game_name).add_entry(entry)
    
    def get_global_stats(self) -> Dict:
        """Get global statistics across all leaderboards."""
        stats = {
            'total_players': set(),
            'total_entries': 0,
            'games': {}
        }
        
        for game_name, board in self._leaderboards.items():
            game_players = {e.account_id for e in board.entries}
            game_scores = [e.score for e in board.entries]
            
            stats['total_players'].update(game_players)
            stats['total_entries'] += len(board.entries)
            
            stats['games'][game_name] = {
                'players': len(game_players),
                'entries': len(board.entries),
                'best_score': max(game_scores) if game_scores else 0,
                'avg_score': sum(game_scores) / len(game_scores) if game_scores else 0
            }
        
        stats['total_players'] = len(stats['total_players'])
        return stats

# Initialize global leaderboard manager
leaderboard_manager = LeaderboardManager() 