"""Base interface for Agent Arcade games."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from loguru import logger
from pydantic import BaseModel
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv
)
from datetime import datetime, timedelta
import time
import gymnasium as gym
from dataclasses import dataclass

# Optional NEAR imports
try:
    from cli.core.near import NEARWallet
    from .staking import stake_on_game
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False

@dataclass
class GameConfig:
    """Game training configuration."""
    total_timesteps: int = 1_000_000
    learning_rate: float = 0.00025
    buffer_size: int = 250_000
    learning_starts: int = 50_000
    batch_size: int = 256
    exploration_fraction: float = 0.2
    target_update_interval: int = 2000
    frame_stack: int = 4

class EvaluationResult(BaseModel):
    """Evaluation results for a game."""
    score: float
    episodes: int
    success_rate: float
    best_episode_score: float
    avg_episode_length: float
    metadata: Dict[str, Any] = {}

class ProgressCallback(BaseCallback):
    """Custom callback for monitoring training progress."""
    
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        
    def _on_training_start(self):
        self.start_time = time.time()
        logger.info("Starting training...")
        
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:  # Update every 1000 steps
            progress = self.num_timesteps / self.total_timesteps
            elapsed_time = time.time() - self.start_time
            estimated_total = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total - elapsed_time
            
            # Format time as HH:MM:SS
            remaining = str(timedelta(seconds=int(remaining_time)))
            
            logger.info(
                f"Progress: {progress*100:.1f}% "
                f"({self.num_timesteps}/{self.total_timesteps} steps) | "
                f"Estimated time remaining: {remaining}"
            )
        return True

class GameInterface(ABC):
    """Base interface for all games."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get game name."""
        pass
        
    @property
    @abstractmethod
    def env_id(self) -> str:
        """Get environment ID."""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Get game description."""
        pass
        
    @property
    @abstractmethod
    def score_range(self) -> Tuple[float, float]:
        """Get valid score range."""
        pass
        
    @abstractmethod
    def make_env(self) -> Any:
        """Create game environment."""
        pass
        
    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """Load trained model."""
        pass
        
    @abstractmethod
    def get_default_config(self) -> GameConfig:
        """Get default game configuration."""
        pass
    
    @abstractmethod
    def train(self, render: bool = False, config_path: Optional[Path] = None) -> Path:
        """Train an agent for this game.
        
        Args:
            render: Whether to render the game during training
            config_path: Path to custom configuration file
            
        Returns:
            Path to the saved model
        """
        config = self.load_config(config_path)
        
        # Create vectorized environment
        env = DummyVecEnv([lambda: self._make_env(render)])
        env = VecFrameStack(env, config.frame_stack)
        
        # Create and train the model
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            exploration_fraction=config.exploration_fraction,
            target_update_interval=config.target_update_interval,
            tensorboard_log=f"./tensorboard/{self.name}",
            verbose=1
        )
        
        # Add progress callback
        callback = ProgressCallback(config.total_timesteps)
        
        logger.info(f"Training {self.name} agent for {config.total_timesteps} timesteps...")
        logger.info("Monitor progress in TensorBoard: tensorboard --logdir ./tensorboard")
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback
        )
        
        # Save the model
        model_path = Path(f"models/{self.name}_final.zip")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    @abstractmethod
    def evaluate(self, model_path: Path, episodes: int = 10, record: bool = False) -> EvaluationResult:
        """Evaluate a trained model.
        
        Args:
            model_path: Path to the model to evaluate
            episodes: Number of episodes to evaluate
            record: Whether to record videos of evaluation
            
        Returns:
            Evaluation results
        """
        pass
    
    @abstractmethod
    def validate_model(self, model_path: Path) -> bool:
        """Validate that a model file is valid for this game."""
        pass
    
    async def stake(self, wallet: Optional['NEARWallet'], model_path: Path, amount: float, target_score: float) -> None:
        """Stake on the agent's performance.
        
        Args:
            wallet: NEAR wallet instance
            model_path: Path to the model to stake on
            amount: Amount to stake in NEAR
            target_score: Target score to achieve
        """
        if not NEAR_AVAILABLE:
            logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
            return
            
        # Validate model first
        if not self.validate_model(model_path):
            logger.error("Invalid model for this game")
            return
            
        # Use the staking module
        await stake_on_game(
            wallet=wallet,
            game_name=self.name,
            model_path=model_path,
            amount=amount,
            target_score=target_score,
            score_range=self.score_range
        )
    
    def load_config(self, config_path: Optional[Path] = None) -> GameConfig:
        """Load and validate configuration.
        
        Args:
            config_path: Path to custom configuration file
            
        Returns:
            Validated configuration
        """
        try:
            if config_path is None:
                return self.get_default_config()
            
            import yaml
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            return GameConfig(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            logger.info("Using default configuration")
            return self.get_default_config()
    
    def calculate_reward_multiplier(self, score: float) -> float:
        """Calculate reward multiplier based on score.
        
        Args:
            score: Achieved score
            
        Returns:
            Reward multiplier (1.0-3.0)
        """
        min_score, max_score = self.score_range
        normalized_score = (score - min_score) / (max_score - min_score)
        
        if normalized_score >= 0.8:  # Exceptional performance
            return 3.0
        elif normalized_score >= 0.6:  # Great performance
            return 2.0
        elif normalized_score >= 0.4:  # Good performance
            return 1.5
        else:
            return 1.0

    def _make_env(self, render: bool = False) -> gym.Env:
        """Create the game environment with proper wrappers."""
        render_mode = "human" if render else "rgb_array"
        env = gym.make(self.env_id, render_mode=render_mode, frameskip=1)
        
        # Standard Atari wrappers
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        
        # Observation preprocessing
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return 1.0 