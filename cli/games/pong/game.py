"""Pong game implementation using ALE."""
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

from cli.games.base import GameInterface, GameConfig, ProgressCallback
from cli.core.evaluation import EvaluationResult, EvaluationConfig, EvaluationPipeline

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
        
    def get_game_info(self) -> dict:
        """Get game information for analysis."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'score_range': self.score_range
        }
        
    def make_env(self, render: bool = False, config: Optional[GameConfig] = None) -> gym.Env:
        """Create the game environment."""
        return self._make_env(render, config)
        
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
        """Create the game environment with proper wrappers."""
        import platform
        import os
        
        # Configure SDL video driver for rendering if needed
        if render:
            system = platform.system()
            if system == "Darwin":  # macOS
                os.environ['SDL_VIDEODRIVER'] = 'cocoa'
                logger.debug("Using 'cocoa' SDL driver for macOS rendering")
            elif system == "Linux":
                os.environ['SDL_VIDEODRIVER'] = 'x11'
                logger.debug("Using 'x11' SDL driver for Linux rendering")
            # Windows typically uses directx by default, no need to set
            
            logger.info(f"Environment rendering enabled. SDL_VIDEODRIVER={os.environ.get('SDL_VIDEODRIVER', 'default')}")
        
        if config is None:
            config = self.get_default_config()
        
        render_mode = "human" if render else "rgb_array"
        env = gym.make(
            self.env_id,
            render_mode=render_mode,
            frameskip=4,  # Fixed frameskip for deterministic behavior
            repeat_action_probability=0.25,  # Standard sticky actions
            full_action_space=False  # Use minimal action space
        )
        
        # Add standard Atari wrappers in correct order
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        try:
            env = FireResetEnv(env)  # Required for Pong
        except Exception as e:
            logger.warning(f"Could not apply FireResetEnv: {e}")
        env = ClipRewardEnv(env)
        
        # Standard observation preprocessing
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = ScaleObservation(env)
        env = TransposeObservation(env)
        
        # Debug observation space
        logger.debug(f"Final observation space: {env.observation_space}")
        return env
    
    def train(self, render: bool = False, config_path: Optional[Path] = None) -> Path:
        """Train an agent for this game."""
        config = self.load_config(config_path)
        
        # For demonstration purposes with rendering, use a single environment
        if render:
            # Create a single environment for better rendering support
            env = self._make_env(render, config)
            
            # Create a special wrapper class to retain rendering capability
            class RenderableDummyVecEnv(DummyVecEnv):
                def __init__(self, env_fns):
                    super().__init__(env_fns)
                    self.original_env = env_fns[0]()
                
                def render(self, *args, **kwargs):
                    return self.original_env.render(*args, **kwargs)
            
            # Wrap in custom DummyVecEnv for rendering
            env = RenderableDummyVecEnv([lambda: env])
            env = VecFrameStack(env, n_stack=4, channels_order='first')
            
            # Use CPU for rendering compatibility
            device = "cpu"
            # Reduce buffer size for demo purposes to avoid memory issues
            buffer_size = min(config.buffer_size, 50000)
            logger.info("Using reduced settings for rendering demo")
            
            # For demo, use much smaller number of steps
            total_timesteps = min(config.total_timesteps, 50000)
        else:
            # Create multiple environments for faster training when not rendering
            def make_env():
                return self._make_env(render, config)
            
            # Use fewer environments to save memory
            num_envs = 8  # Reduced from 16 to save memory
            env = DummyVecEnv([make_env for _ in range(num_envs)])
            env = VecFrameStack(env, n_stack=4, channels_order='first')
            
            # Use GPU for training when not rendering
            device = "cuda" if torch.cuda.is_available() else "cpu"
            buffer_size = config.buffer_size
            total_timesteps = config.total_timesteps
        
        logger.debug(f"Training observation space: {env.observation_space}")
        logger.info(f"Training on device: {device} with {total_timesteps} timesteps")
        
        # Create and train the model with adjusted parameters
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=buffer_size,
            learning_starts=min(config.learning_starts, 1000) if render else config.learning_starts,
            batch_size=min(config.batch_size, 256) if render else config.batch_size,  # Smaller batch for demo
            exploration_fraction=config.exploration_fraction,
            target_update_interval=config.target_update_interval,
            tensorboard_log=f"./tensorboard/{self.name}" if config.tensorboard_log else None,
            policy_kwargs={
                "net_arch": [256, 256] if render else [1024, 1024],  # Much smaller network for demo
                "normalize_images": False,  # Images already normalized
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {
                    "eps": 1e-5,
                    "weight_decay": 1e-6
                }
            },
            train_freq=(4, "step"),
            gradient_steps=1 if render else 4,  # Reduced for demo
            verbose=1,
            device=device
        )
        
        # Add progress callback
        callback = ProgressCallback(total_timesteps)
        
        logger.info(f"Training {self.name} agent for {total_timesteps} timesteps...")
        if config.tensorboard_log:
            logger.info("Monitor progress in TensorBoard: tensorboard --logdir ./tensorboard")
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=config.log_interval
            )
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Saving current model...")
        finally:
            # Save the model even if interrupted
            model_path = Path(f"models/{self.name}/baseline/final_model.zip")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
            logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def evaluate(self, model_path, episodes=10, seed=42, render=False, deterministic=True, record=False):
        """Evaluate a trained model."""
        
        try:
            # Create environment with proper settings
            render_mode = "human" if render else None
            env = self._make_env(render=render)
            
            # Load model first to get its observation space
            temp_model = DQN.load(str(model_path))
            model_obs_space = temp_model.observation_space
            n_stack = model_obs_space.shape[0]  # Get frame stack from model's observation space
            logger.debug(f"Model observation space: {model_obs_space}")
            logger.debug(f"Using frame_stack={n_stack} from model")
            
            # Create evaluation config
            eval_config = EvaluationConfig(
                game_id=self.name,
                n_eval_episodes=episodes,
                deterministic=deterministic,
                render=render,
                verbose=1,
                frame_stack=n_stack,  # Use the frame stack from the model
                record=record  # Add record parameter
            )
            
            # Create dummy wallet and leaderboard since we're not using blockchain features
            from unittest.mock import MagicMock
            dummy_wallet = MagicMock()
            dummy_leaderboard = MagicMock()
            
            # Create evaluation pipeline
            pipeline = EvaluationPipeline(
                game=self.name,
                env=env,
                model=DQN,
                wallet=dummy_wallet,
                leaderboard_manager=dummy_leaderboard,
                config=eval_config
            )
            
            # Use the existing model observation space to configure the environment correctly
            pipeline.env = pipeline._prepare_environment(model_obs_space)
            
            # Load the model with the properly configured environment
            pipeline.model = DQN.load(str(model_path), env=pipeline.env)
            
            # Run evaluation
            result = pipeline.evaluate()
            
            # No need to transform the result - return the EvaluationResult directly
            # The core EvaluationResult already has mean_reward which is what the CLI uses as score
            logger.info(f"Evaluation completed with mean score: {result.mean_reward}")
            
            # Return the result directly - it already has the correct structure
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def validate_model(self, model_path: Path) -> bool:
        """Validate model file."""
        try:
            env = DummyVecEnv([lambda: self._make_env()])
            env = VecFrameStack(env, n_stack=4, channels_order='first')
            DQN.load(model_path, env=env)
            return True
        except Exception as e:
            logger.error(f"Invalid model file: {e}")
            return False
    
    async def stake(self, wallet: NEARWallet, model_path: Path, amount: float, target_score: float) -> None:
        """Stake NEAR on performance."""
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
    """Register the game."""
    from cli.games import register_game
    register_game(PongGame) 