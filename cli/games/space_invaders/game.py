"""Space Invaders game implementation using ALE."""
import gymnasium as gym
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
    WarpFrame
)
from loguru import logger
import numpy as np
import torch
import os
import datetime
from stable_baselines3.common.monitor import Monitor

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
        render_mode = "human" if render else None
        return self._make_env(render_mode=render_mode)
        
    def load_model(self, model_path: Path) -> DQN:
        """Load a trained model with correct environment configuration."""
        env = DummyVecEnv([lambda: self._make_env()])
        env = VecFrameStack(env, n_stack=4, channels_order='first')  # Use 'first' for PyTorch compatibility
        return DQN.load(model_path, env=env)
        
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
    
    def _make_env(self, seed=None, idx=0, capture_video=False, render_mode=None):
        """Create the Space Invaders environment with the specified settings."""
        import platform
        import os
        
        # Configure SDL video driver for rendering if needed
        if render_mode == "human":
            system = platform.system()
            if system == "Darwin":  # macOS
                os.environ['SDL_VIDEODRIVER'] = 'cocoa'
                logger.debug("Using 'cocoa' SDL driver for macOS rendering")
            elif system == "Linux":
                os.environ['SDL_VIDEODRIVER'] = 'x11'
                logger.debug("Using 'x11' SDL driver for Linux rendering")
            # Windows typically uses directx by default, no need to set
            
            logger.info(f"Environment rendering enabled. SDL_VIDEODRIVER={os.environ.get('SDL_VIDEODRIVER', 'default')}")
            
        # Create environment
        env = gym.make(
            "ALE/SpaceInvaders-v5",
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
            env = FireResetEnv(env)  # Required for some Atari games
        except Exception as e:
            logger.warning(f"Could not apply FireResetEnv: {e}")
        env = ClipRewardEnv(env)
        
        # Standard observation preprocessing - matches ALE standard
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = ScaleObservation(env)
        env = TransposeObservation(env)  # Transposes to channel-first format (C, H, W)
        
        # Debug observation space
        logger.debug(f"Final observation space: {env.observation_space}")
        return env
    
    def train(self, **kwargs):
        """Train a reinforcement learning model for Space Invaders."""
        render = kwargs.get('render', False)
        config_path = kwargs.get('config_path', None)
        
        # Set model type based on render mode
        model_type = "demo" if render else "fullTrain"
        
        # Generate timestamp for unique model filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory with appropriate subfolder
        model_dir = Path(f"models/{self.name}/{'demo' if render else 'baseline'}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create descriptive model filename
        model_filename = f"{self.name}_{model_type}_{timestamp}.zip"
        model_path = model_dir / model_filename
        
        # Default training setting based on render mode
        total_timesteps = 50_000 if render else 10_000_000
        
        try:
            # Load configuration if provided
            config = {}
            if config_path:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded configuration from {config_path}")
                    
                    # Override total_timesteps if in config
                    if 'total_timesteps' in config:
                        total_timesteps = config['total_timesteps']
            
            # Create environment
            if render:
                # For rendering, we use a single environment
                env = self._make_env(seed=0, idx=0, capture_video=False, render_mode="human")
                vec_env = DummyVecEnv([lambda: env])
                # Stack frames in channel-first format for PyTorch compatibility
                vec_env = VecFrameStack(vec_env, n_stack=4, channels_order='first')
                logger.info("Created single environment for rendering")
                
                # Use smaller buffer and network for demo mode to reduce memory usage
                buffer_size = 5000
                policy_kwargs = dict(
                    net_arch=[256, 256]
                )
            else:
                # For full training without rendering, use vectorized environments
                vec_env = make_vec_env(
                    self._make_env, 
                    n_envs=12, 
                    seed=0,
                    env_kwargs=dict(capture_video=False, render_mode=None)
                )
                # Stack frames in channel-first format for PyTorch compatibility
                vec_env = VecFrameStack(vec_env, n_stack=4, channels_order='first')
                logger.info("Created vectorized environment for training")
                
                # Use standard buffer and network for full training
                buffer_size = 1_000_000
                policy_kwargs = dict(
                    net_arch=[1024, 1024]
                )
            
            # Create model
            model = DQN(
                "CnnPolicy",
                vec_env,
                buffer_size=buffer_size,
                learning_rate=2.5e-4,
                batch_size=128,
                learning_starts=10000,
                target_update_interval=1000,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.1,
                exploration_final_eps=0.01,
                verbose=1,
                policy_kwargs=policy_kwargs
            )
            
            try:
                # Train model
                model.learn(total_timesteps=total_timesteps, progress_bar=True)
                
                # Save model
                model.save(str(model_path))
                logger.info(f"Model saved to {model_path}")
                
                # Create a "latest" link for convenience
                latest_link = model_dir / f"{self.name}_{model_type}_latest.zip"
                if os.path.exists(latest_link):
                    os.remove(latest_link)
                os.symlink(model_filename, latest_link)
                logger.info(f"Created latest link at {latest_link}")
                
                return model_path
                
            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                # Save model on interrupt
                model.save(str(model_path))
                logger.info(f"Interrupted model saved to {model_path}")
                
                # Create a "latest" link for convenience
                latest_link = model_dir / f"{self.name}_{model_type}_latest.zip"
                if os.path.exists(latest_link):
                    os.remove(latest_link)
                os.symlink(model_filename, latest_link)
                logger.info(f"Created latest link at {latest_link}")
                
                return model_path
                
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def evaluate(self, model_path, episodes=10, seed=42, render=False, deterministic=True, record=False):
        """
        Evaluate a trained model's performance in the game.
        
        Args:
            model_path: Path to the trained model file
            episodes: Number of episodes to evaluate
            seed: Random seed for evaluation
            render: Whether to render the environment (bool)
            deterministic: Whether to use deterministic actions
            record: Whether to record videos of evaluation
            
        Returns:
            EvaluationResult containing metrics from the evaluation
        """
        try:
            # Create environment with proper settings
            render_mode = "human" if render else None
            env = self._make_env(render_mode=render_mode)
            
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
            
            # Log evaluation results
            logger.info(f"Evaluation completed with mean score: {result.mean_reward}")
            
            # Return the result directly - it already has the correct structure
            return result
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def validate_model(self, model_path: Path) -> bool:
        """Validate Space Invaders model file."""
        try:
            env = DummyVecEnv([lambda: self._make_env()])
            env = VecFrameStack(env, n_stack=4, channels_order='first')
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

    def create_training_env(self, n_envs=12, seed=None):
        """
        Create a standardized training environment with the correct settings.
        
        This method ensures that the environment is properly set up for training
        with consistent observation spaces and channel ordering.
        
        Args:
            n_envs: Number of parallel environments to create
            seed: Random seed for reproducibility
            
        Returns:
            A vectorized environment ready for training
        """
        from stable_baselines3.common.env_util import make_vec_env
        
        # Create vectorized environments
        vec_env = make_vec_env(
            self._make_env, 
            n_envs=n_envs, 
            seed=seed,
            env_kwargs=dict(capture_video=False, render_mode=None)
        )
        
        # Stack frames in channel-first format for PyTorch compatibility
        vec_env = VecFrameStack(vec_env, n_stack=4, channels_order='first')
        
        logger.info(f"Created standardized training environment with shape: {vec_env.observation_space.shape}")
        return vec_env

    def create_evaluation_env(self, n_stack=4, render_mode=None):
        """
        Create a standardized environment for model evaluation.
        
        This ensures the environment has the correct observation space format,
        particularly for loading and evaluating models.
        
        Args:
            n_stack: Number of frames to stack
            render_mode: Render mode (None, 'human', 'rgb_array')
            
        Returns:
            A vectorized environment ready for evaluation
        """
        # Create base environment
        env = self._make_env(render_mode=render_mode)
        
        # Add monitor for logging
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)
        
        # Vectorize and stack frames
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=n_stack, channels_order='first')
        
        logger.info(f"Created standardized evaluation environment with shape: {env.observation_space.shape}")
        return env

def register():
    """Register the Space Invaders game."""
    from cli.games import register_game
    register_game(SpaceInvadersGame) 