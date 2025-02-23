"""Space Invaders game implementation using ALE."""
import gymnasium as gym
from pathlib import Path
from typing import Optional, Tuple, Any
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
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

class TransposeObservation(gym.ObservationWrapper):
    """Transpose observation for PyTorch CNN input."""
    
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=np.float32
        )
    
    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))

class SpaceInvadersGame(GameInterface):
    """Space Invaders game implementation."""
    
    def __call__(self):
        """Make the class callable to return itself."""
        return self
    
    @property
    def name(self) -> str:
        return "space_invaders"
        
    @property
    def env_id(self) -> str:
        return "ALE/SpaceInvaders-v5"
        
    @property
    def description(self) -> str:
        return "Defend Earth from alien invasion"
        
    @property
    def version(self) -> str:
        return "1.0.0"
        
    @property
    def score_range(self) -> tuple[float, float]:
        return (0, 1000)  # Space Invaders typical score range
        
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
            total_timesteps=5_000_000,    # Extended training for better strategies
            learning_rate=0.00025,        # Standard DQN learning rate
            buffer_size=1_000_000,        # Increased for H100 memory capacity
            learning_starts=50_000,       # More initial exploration
            batch_size=1024,              # Larger batches for H100
            exploration_fraction=0.2,      # More exploration for complex strategies
            target_update_interval=2000,   # Less frequent updates for stability
            frame_stack=4,                # Standard Atari frame stack
            policy="CnnPolicy",
            tensorboard_log=True,
            log_interval=100              # Frequent logging
        )
    
    def _make_env(self, render: bool = False, config: Optional[GameConfig] = None) -> gym.Env:
        """Create the Space Invaders environment with proper wrappers."""
        if config is None:
            config = self.get_default_config()
        
        render_mode = "human" if render else "rgb_array"
        env = gym.make(
            self.env_id,
            render_mode=render_mode
        )
        
        # Add standard Atari wrappers in correct order
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)  # Space Invaders requires FIRE to start
        env = ClipRewardEnv(env)
        
        # Standard observation preprocessing
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = ScaleObservation(env)  # Scale to [0, 1]
        
        # Transpose for PyTorch: (H, W, C) -> (C, H, W)
        env = TransposeObservation(env)
        
        # Debug observation space
        logger.debug(f"Single env observation space before vectorization: {env.observation_space}")
        return env
    
    def train(self, render: bool = False, config_path: Optional[Path] = None) -> Path:
        """Train an agent for this game."""
        config = self.load_config(config_path)
        
        def make_env():
            env = self._make_env(render, config)
            return env
        
        # Create vectorized environment with more parallel envs for H100
        env = DummyVecEnv([make_env for _ in range(16)])  # Increased from 8 to 16
        
        # Stack frames in the correct order for SB3 (n_envs, n_stack, h, w)
        env = VecFrameStack(env, n_stack=4, channels_order='first')
        
        # Debug final observation space
        logger.debug(f"Final observation space: {env.observation_space}")
        
        # Create and train the model with optimized policy network for H100
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
                "net_arch": [1024, 1024],  # Larger network for H100
                "normalize_images": False,  # Images are already normalized
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "eps": 1e-5,
                    "weight_decay": 1e-6
                }
            },
            train_freq=(4, "step"),       # Update every 4 steps
            gradient_steps=4,             # Multiple gradient steps per update
            verbose=1,
            device="cuda"
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
        """Evaluate a trained Space Invaders model."""
        # Load model metadata to get frame stack configuration
        config = self.get_default_config()
        metadata_path = model_path.parent / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
                if "hyperparameters" in metadata:
                    config.frame_stack = metadata["hyperparameters"].get("frame_stack", 16)
                    logger.debug(f"Using frame_stack={config.frame_stack} from metadata")
        
        # Create environment with correct frame stack size
        env = DummyVecEnv([lambda: self._make_env(record, config)])
        model = DQN.load(model_path, env=env)
        
        total_score = 0
        episode_lengths = []
        best_score = float('-inf')
        successes = 0
        
        logger.info(f"Evaluating model for {episodes} episodes...")
        logger.debug(f"Using frame stack size: {config.frame_stack}")
        if record:
            logger.info("Recording evaluation videos to videos/evaluation/")
        
        for episode in range(episodes):
            obs = env.reset()[0]
            done = False
            episode_score = 0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_score += reward[0]
                episode_length += 1
                done = terminated[0] or truncated[0]
                
                # Track success using info dict for consistency with evaluation.py
                if isinstance(info, (list, tuple)):
                    info = info[0]  # Get first env's info
                if isinstance(info, dict) and info.get("is_success", False):
                    successes += 1
                elif episode_score > 100:  # Fallback success criteria
                    successes += 1
            
            total_score += episode_score
            episode_lengths.append(episode_length)
            best_score = max(best_score, episode_score)
            
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
        """Validate Space Invaders model file."""
        try:
            env = DummyVecEnv([lambda: self._make_env()])
            DQN.load(model_path, env=env)
            return True
        except Exception as e:
            logger.error(f"Invalid model file: {e}")
            return False
    
    async def stake(self, wallet: NEARWallet, model_path: Path, amount: float, target_score: float) -> None:
        """Stake NEAR on Space Invaders performance."""
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
    """Register the Space Invaders game."""
    from cli.games import register_game
    register_game(SpaceInvadersGame) 