"""Agent Arcade CLI."""
import gymnasium as gym
import ale_py

# Register Atari environments early
try:
    gym.register_envs(ale_py)
except ImportError:
    pass  # Will be handled by more specific error messages later

__version__ = "0.1.0" 