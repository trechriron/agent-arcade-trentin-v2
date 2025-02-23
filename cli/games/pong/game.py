"""Pong game implementation using ALE."""
import gymnasium as gym
from pathlib import Path
from typing import Optional, Tuple, Any
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv
)
from loguru import logger
import numpy as np
import torch

from cli.games.base import GameInterface, GameConfig, EvaluationResult, ProgressCallback

# Optional NEAR imports
try:
    from cli.core.near import NEARWallet
    from cli.core.stake import StakeRecord
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False
    NEARWallet = Any  # Type alias for type hints

class ScaleObservation(gym.ObservationWrapper):
    """Scale observations to [0, 1]."""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=self.observation_space.shape,
            dtype=np.float32
        )
    
    def observation(self, obs):
        return obs.astype(np.float32) / 255.0

class PongGame(GameInterface):
    """Pong game implementation."""
    
    def __call__(self):
        """Make the class callable to return itself."""
        return self
    
    @property
    def name(self) -> str:
        return "pong"
        
    @property
    def env_id(self) -> str:
        return "ALE/Pong-v5"
        
    @property
    def description(self) -> str:
        return "Classic paddle vs paddle game"
        
    @property
    def version(self) -> str:
        return "2.0.0"
        
    @property
    def score_range(self) -> tuple[float, float]:
        return (-21, 21)  # Pong scores range from -21 to 21
        
    def get_score_range(self) -> tuple[float, float]:
        """Get valid score range for the game."""
        return self.score_range
        
    def make_env(self):
        """Create the game environment."""
        return self._make_env()
        
    def load_model(self, model_path: str):
        """Load a trained model."""
        return DQN.load(model_path)
        
    def get_default_config(self) -> GameConfig:
        """Get default configuration for the game."""
        return GameConfig(
            total_timesteps=2_000_000,    # Increased for better convergence
            learning_rate=0.00025,        # Standard DQN learning rate
            buffer_size=500_000,          # Larger buffer for better sampling
            learning_starts=50_000,       # More exploration before learning
            batch_size=1024,              # Large batches for GPU efficiency
            exploration_fraction=0.1,      # Standard exploration
            target_update_interval=1000,   # Standard update interval
            frame_stack=16,               # Increased for better temporal info
            policy="CnnPolicy",
            tensorboard_log=True,
            log_interval=100              # Frequent logging
        )
    
    def _make_env(self, render: bool = False, config: Optional[GameConfig] = None) -> gym.Env:
        """Create the game environment."""
        if config is None:
            config = self.get_default_config()
        
        render_mode = 'human' if render else 'rgb_array'
        env = gym.make(
            self.env_id,
            render_mode=render_mode
        )
        
        # Add standard Atari wrappers in correct order
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        env = ClipRewardEnv(env)
        
        # Standard observation preprocessing
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, config.frame_stack)
        
        return env
    
    def train(self, render: bool = False, config_path: Optional[Path] = None) -> Path:
        """Train an agent for this game."""
        config = self.load_config(config_path)
        
        # Create vectorized environment with parallel envs
        env = DummyVecEnv([lambda: self._make_env(render, config) for _ in range(8)])
        
        # Create and train the model with optimized policy network
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            exploration_fraction=config.exploration_fraction,
            target_update_interval=config.target_update_interval,
            tensorboard_log=f"./tensorboard/{self.name}" if config.tensorboard_log else None,
            policy_kwargs={
                "net_arch": [512, 512],   # Larger network for better learning
                "normalize_images": True,  # Input normalization
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "eps": 1e-5,
                    "weight_decay": 1e-6
                }
            },
            train_freq=(8, "step"),      # Update every 8 steps
            gradient_steps=2,            # Two gradient steps per update
            verbose=1,
            device="cuda"               # Use GPU
        )
        
        # Add progress callback
        callback = ProgressCallback(config.total_timesteps)
        
        logger.info(f"Training {self.name} agent for {config.total_timesteps} timesteps...")
        if config.tensorboard_log:
            logger.info("Monitor progress in TensorBoard: tensorboard --logdir ./tensorboard")
        
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
            log_interval=config.log_interval
        )
        
        # Save the model
        model_path = Path(f"models/{self.name}/baseline/final_model.zip")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def evaluate(self, model_path: Path, episodes: int = 10, record: bool = False) -> EvaluationResult:
        """Evaluate a trained Pong model."""
        env = DummyVecEnv([lambda: self._make_env(record)])
        model = DQN.load(model_path, env=env)
        
        total_score = 0
        episode_lengths = []
        best_score = float('-inf')
        successes = 0
        
        logger.info(f"Evaluating model for {episodes} episodes...")
        if record:
            logger.info("Recording evaluation videos to videos/evaluation/")
        
        for episode in range(episodes):
            obs = env.reset()[0]
            done = False
            episode_score = 0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_score += reward[0]
                episode_length += 1
                done = terminated[0] or truncated[0]
            
            total_score += episode_score
            episode_lengths.append(episode_length)
            best_score = max(best_score, episode_score)
            if episode_score > 0:  # Consider winning as success
                successes += 1
            
            logger.info(f"Episode {episode + 1}/{episodes} - Score: {episode_score:.2f}")
        
        env.close()
        
        result = EvaluationResult(
            score=total_score / episodes,
            episodes=episodes,
            success_rate=successes / episodes,
            best_episode_score=best_score,
            avg_episode_length=sum(episode_lengths) / len(episode_lengths)
        )
        
        logger.info(f"Evaluation complete - Average score: {result.score:.2f}")
        return result
    
    def validate_model(self, model_path: Path) -> bool:
        """Validate Pong model file."""
        try:
            env = DummyVecEnv([lambda: self._make_env()])
            DQN.load(model_path, env=env)
            return True
        except Exception as e:
            logger.error(f"Invalid model file: {e}")
            return False
    
    async def stake(self, wallet: NEARWallet, model_path: Path, amount: float, target_score: float) -> None:
        """Stake NEAR on Pong performance."""
        if not NEAR_AVAILABLE:
            raise RuntimeError("NEAR integration not available")
            
        if not wallet.is_logged_in():
            raise ValueError("Must be logged in to stake")
        
        if not self.validate_model(model_path):
            raise ValueError("Invalid model file")
        
        # Verify target score is within range
        min_score, max_score = self.score_range
        if not min_score <= target_score <= max_score:
            raise ValueError(f"Target score must be between {min_score} and {max_score}")
        
        # Use staking module
        await stake_on_game(
            wallet=wallet,
            game_name=self.name,
            model_path=model_path,
            amount=amount,
            target_score=target_score,
            score_range=self.score_range
        )

def register():
    """Register the Pong game."""
    from cli.games import register_game
    register_game(PongGame) 