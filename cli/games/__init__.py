"""Game loading and management for Agent Arcade."""
from pathlib import Path
from importlib import import_module
from typing import Dict, Type, List, Optional, Tuple, Callable
from loguru import logger
from dataclasses import dataclass

from .base import GameInterface, GameConfig
from .pong.game import PongGame
from .space_invaders.game import SpaceInvadersGame
from .riverraid.game import RiverraidGame

# Register Atari environments
try:
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    logger.debug("Atari environments registered successfully")
except ImportError:
    logger.warning("Failed to register Atari environments. Some games may not be available.")

# Check for NEAR availability
try:
    from cli.core.near import NEARWallet
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False
    logger.debug("NEAR integration not available")

# Dictionary of registered games
GAMES: Dict[str, Type[GameInterface]] = {}

@dataclass
class GameInfo:
    """Information about a registered game."""
    name: str
    env_id: str
    description: str
    score_range: Tuple[float, float]  # (min_score, max_score)
    make_env: Callable
    load_model: Callable

def register_game(game_class: Type[GameInterface]) -> None:
    """Register a game with the system."""
    game = game_class()
    logger.debug(f"Registered game: {game.name}")
    GAMES[game.name] = game_class

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

def get_registered_games() -> List[str]:
    """Get list of registered game names."""
    return list(GAMES.keys())

def get_game_info(name: str) -> Optional[GameInterface]:
    """Get information about a registered game."""
    if name not in GAMES:
        return None
    return GAMES[name]()

def list_games() -> List[Dict[str, str]]:
    """List all available games with their information."""
    games = []
    for name, game_class in sorted(GAMES.items()):
        try:
            game = game_class()
            games.append({
                "name": game.name,
                "description": game.description,
                "version": game.version,
                "staking_enabled": NEAR_AVAILABLE
            })
        except Exception as e:
            logger.warning(f"Could not load game {name}: {e}")
    return games

def validate_game_implementation(game_class: Type[GameInterface]) -> bool:
    """Validate that a game implementation has all required methods."""
    try:
        # Create instance to test
        game = game_class()
        
        # Test required properties
        _ = game.name
        _ = game.description
        _ = game.version
        
        # Test required methods with minimal arguments
        _ = game.get_default_config()
        _ = game.get_score_range()
        
        # If NEAR is not available, we don't validate staking
        if not NEAR_AVAILABLE:
            return True
            
        # Test staking only if NEAR is available
        _ = game.stake
        
        return True
    except Exception as e:
        logger.error(f"Game validation failed: {e}")
        return False

# Load games from the games directory
games_dir = Path(__file__).parent

# Recursively find all Python files
for game_dir in games_dir.iterdir():
    if not game_dir.is_dir() or game_dir.name.startswith("__"):
        continue
        
    game_file = game_dir / "game.py"
    if not game_file.exists():
        continue
        
    try:
        # Import the game module
        module = import_module(f".{game_dir.name}.game", package="cli.games")
        
        # Look for game classes
        for item in dir(module):
            if item.endswith("Game"):
                game_class = getattr(module, item)
                if validate_game_implementation(game_class):
                    # Use the game's name property as the registry key
                    game_instance = game_class()
                    register_game(game_instance)
    except Exception as e:
        logger.warning(f"Could not load game from {game_file}: {e}")

def make_atari_env(env_id: str):
    """Create an Atari environment with standard wrappers."""
    env = gym.make(env_id, render_mode='rgb_array')
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

def load_model(model_path: str):
    """Load a trained model."""
    try:
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None 