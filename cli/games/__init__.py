"""Game loading and management for Agent Arcade."""
from pathlib import Path
from importlib import import_module
from typing import Dict, Type

from .base import GameInterface

# Game registry
GAMES: Dict[str, Type[GameInterface]] = {}

def register_game(name: str, game_class: Type[GameInterface]):
    """Register a game with the arcade."""
    GAMES[name] = game_class

def get_game(name: str) -> GameInterface:
    """Get a game instance by name."""
    if name not in GAMES:
        raise ValueError(f"Game '{name}' not found. Available games: {list(GAMES.keys())}")
    return GAMES[name]()

# Auto-load games from subdirectories
games_dir = Path(__file__).parent
for game_dir in games_dir.iterdir():
    if game_dir.is_dir() and not game_dir.name.startswith('_'):
        try:
            module = import_module(f".{game_dir.name}.game", package="cli.games")
            if hasattr(module, 'register'):
                module.register()
        except ImportError as e:
            print(f"Warning: Could not load game {game_dir.name}: {e}") 