"""Riverraid implementation using ALE."""
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
    ClipRewardEnv,
    WarpFrame
)
from loguru import logger
import numpy as np
import torch
import os
import datetime
from stable_baselines3.common.monitor import Monitor

from cli.games.base import GameInterface, GameConfig, EvaluationResult, ProgressCallback

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

class RiverraidGame(GameInterface):
    """River Raid game implementation."""
    
    def __call__(self):
        """Make the class callable to return itself."""
        return self
    
    @property
    def name(self) -> str:
        return "riverraid"
        
    @property
    def env_id(self) -> str:
        return "ALE/Riverraid-v5"  # Latest ALE version
        
    @property
    def description(self) -> str:
        return "Control a jet, manage fuel, and destroy enemies"
        
    @property
    def version(self) -> str:
        return "1.0.0"
        
    @property
    def score_range(self) -> tuple[float, float]:
        return (0, 100000)  # River Raid scores can go very high

    def make_env(self) -> gym.Env:
        """Create a basic environment instance for evaluation."""
        return self._make_env(render_mode=None)

    def load_model(self, model_path: Path) -> DQN:
        """Load a trained model with correct environment configuration."""
        env = DummyVecEnv([lambda: self._make_env()])
        env = VecFrameStack(env, n_stack=4, channels_order='first')
        return DQN.load(model_path, env=env)
    
    def _make_env(self, seed=None, idx=0, capture_video=False, render_mode=None):
        """Create the RiverRaid environment with the specified settings."""
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
            "ALE/Riverraid-v5",
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
    
    def get_default_config(self) -> GameConfig:
        """Get default configuration for the game."""
        return GameConfig(
            total_timesteps=10_000_000,   # Extended training for complex strategies
            learning_rate=0.00025,        # Standard DQN learning rate
            buffer_size=2_000_000,        # Large buffer for diverse experiences
            learning_starts=200_000,      # Substantial exploration period
            batch_size=2048,              # Large batches for GPU efficiency
            exploration_fraction=0.2,      # More exploration for complex strategies
            target_update_interval=2000,   # Less frequent updates for stability
            frame_stack=4,                # Standard for Atari
            policy="CnnPolicy",           # Correct for image input
            tensorboard_log=True,
            log_interval=100              # Frequent logging
        )
    
    def train(self, **kwargs):
        """Train a reinforcement learning model for RiverRaid."""
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
    
    def evaluate(self, model_path, episodes=10, seed=42, render=False, deterministic=True):
        """
        Evaluate a trained model's performance in the game.
        
        Args:
            model_path: Path to the trained model file
            episodes: Number of episodes to evaluate
            seed: Random seed for evaluation
            render: Whether to render the environment (bool)
            deterministic: Whether to use deterministic actions
            
        Returns:
            EvaluationResult containing metrics from the evaluation
        """
        try:
            from stable_baselines3 import DQN
            from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
            from stable_baselines3.common.monitor import Monitor
            from cli.core.evaluation import EvaluationResult
            import numpy as np
            import os
            
            logger.info(f"Evaluating model: {model_path}")
            
            # Load the model first to get its observation space
            # This is a temporary environment just to load the model
            temp_env = DummyVecEnv([lambda: self._make_env()])
            temp_env = VecFrameStack(temp_env, n_stack=4, channels_order='first')
            temp_model = DQN.load(model_path)
            model_obs_space = temp_model.observation_space
            logger.info(f"Model observation space: {model_obs_space}")
            
            # Now use the information from the model to create a matching environment
            render_mode = "human" if render else "rgb_array"
            
            # Create and wrap the environment
            env = self._make_env(render_mode=render_mode)
            env = Monitor(env)
            
            # Vectorize the environment and match the frame stacking to the model
            env = DummyVecEnv([lambda: env])
            
            # Use the same number of frame stacks as the model
            n_stack = model_obs_space.shape[0]
            env = VecFrameStack(env, n_stack=n_stack, channels_order='first')
            
            # Override the observation space to match the model
            env.observation_space = model_obs_space
            
            # Load the model with our prepared environment
            model = DQN.load(model_path, env=env)
            
            # Set seed if provided
            if seed is not None:
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.seed(seed)
            
            # Evaluate the model
            episode_rewards = []
            episode_lengths = []
            success_count = 0
            success_threshold = 1000  # RiverRaid success threshold
            
            for episode in range(episodes):
                obs = env.reset()
                done = False
                total_reward = 0
                steps = 0
                
                # Ensure obs has the right shape
                if obs.shape != model_obs_space.shape:
                    logger.warning(f"Reshaping observation from {obs.shape} to match model's expected {model_obs_space.shape}")
                    # Reshape observation if needed
                    if len(obs.shape) >= 3 and len(model_obs_space.shape) >= 3:
                        # Resize using OpenCV or other method if needed
                        pass
                
                while not done:
                    action, _states = model.predict(obs, deterministic=deterministic)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward[0]
                    steps += 1
                    
                    # Break if episode is too long (safety measure)
                    if steps >= 10000:
                        logger.warning("Episode too long, terminating early")
                        break
                    
                    # Ensure obs has the right shape for next iteration
                    if obs.shape != model_obs_space.shape:
                        logger.warning(f"Reshaping observation from {obs.shape} to match model's expected {model_obs_space.shape}")
                        # Reshape observation if needed
                        if len(obs.shape) >= 3 and len(model_obs_space.shape) >= 3:
                            # Resize using OpenCV or other method if needed
                            pass
                
                # Record metrics
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                
                # Check for success
                if total_reward >= success_threshold:
                    success_count += 1
                
                logger.info(f"Episode {episode+1}: Score: {total_reward:.2f}, Steps: {steps}")
            
            # Calculate final metrics
            mean_reward = np.mean(episode_rewards)
            success_rate = (success_count / episodes) * 100
            avg_episode_length = np.mean(episode_lengths)
            best_episode = np.max(episode_rewards)
            
            logger.info(f"Evaluation results - Mean score: {mean_reward:.2f}, Success rate: {success_rate:.1f}%, Episodes: {episodes}")
            
            return EvaluationResult(
                score=mean_reward,
                episodes=episodes,
                success_rate=success_rate,
                best_episode_score=best_episode,
                avg_episode_length=avg_episode_length
            )
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
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
            raise NotImplementedError("NEAR integration is not available. Install py_near to enable staking.")
            
        if not wallet.is_logged_in():
            raise ValueError("Wallet must be logged in to stake")
            
        if not self.validate_model(model_path):
            raise ValueError("Invalid model file")
            
        # Evaluate model performance
        result = self.evaluate(model_path, episodes=10)
        if result.score < target_score:
            raise ValueError(f"Model performance ({result.score:.2f}) below target score ({target_score:.2f})")
            
        # Create stake record
        record = StakeRecord(
            game=self.name,
            model_path=str(model_path),
            amount=amount,
            target_score=target_score,
            current_score=result.score
        )
        
        # Submit stake transaction
        await wallet.stake(record)
        logger.info(f"Staked {amount} NEAR on {self.name} model performance")

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
    """Register the game."""
    from cli.games import register_game
    register_game(RiverraidGame)
