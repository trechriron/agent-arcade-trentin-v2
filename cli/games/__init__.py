"""Game loading and management for Agent Arcade."""
from pathlib import Path
from importlib import import_module
from typing import Dict, Type, List
from loguru import logger

from .base import GameInterface, GameConfig

# Game registry
GAMES: Dict[str, Type[GameInterface]] = {}

def register_game(name: str, game_class: Type[GameInterface]) -> None:
    """Register a game with the arcade.
    
    Args:
        name: Unique identifier for the game
        game_class: Game implementation class
    """
    if name in GAMES:
        logger.warning(f"Game '{name}' already registered. Overwriting...")
    GAMES[name] = game_class
    logger.info(f"Registered game: {name}")

def get_game(name: str) -> GameInterface:
    """Get a game instance by name.
    
    Args:
        name: Name of the game to load
        
    Returns:
        Instantiated game interface
        
    Raises:
        ValueError: If game not found
    """
    if name not in GAMES:
        available = ", ".join(sorted(GAMES.keys()))
        raise ValueError(
            f"Game '{name}' not found. Available games: {available}"
        )
    return GAMES[name]()

def list_games() -> List[Dict[str, str]]:
    """List all available games with metadata.
    
    Returns:
        List of game info dictionaries
    """
    games = []
    for name, game_class in sorted(GAMES.items()):
        try:
            instance = game_class()
            games.append({
                "name": name,
                "description": instance.description,
                "version": instance.version
            })
        except Exception as e:
            logger.error(f"Failed to load game {name}: {e}")
    return games

def validate_game_implementation(game_class: Type[GameInterface]) -> bool:
    """Validate that a game implementation meets requirements.
    
    Args:
        game_class: Game class to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        instance = game_class()
        # Verify required properties
        assert isinstance(instance.name, str)
        assert isinstance(instance.description, str)
        assert isinstance(instance.version, str)
        # Verify config handling
        config = instance.get_default_config()
        assert isinstance(config, GameConfig)
        # Verify score range
        min_score, max_score = instance.get_score_range()
        assert isinstance(min_score, (int, float))
        assert isinstance(max_score, (int, float))
        assert min_score < max_score
        return True
    except Exception as e:
        logger.error(f"Game validation failed: {e}")
        return False

# Auto-load games from subdirectories
games_dir = Path(__file__).parent
for game_dir in games_dir.iterdir():
    if game_dir.is_dir() and not game_dir.name.startswith('_'):
        try:
            module = import_module(f".{game_dir.name}.game", package="cli.games")
            if hasattr(module, 'register'):
                module.register()
                logger.info(f"Loaded game module: {game_dir.name}")
        except ImportError as e:
            logger.warning(f"Could not load game {game_dir.name}: {e}") 