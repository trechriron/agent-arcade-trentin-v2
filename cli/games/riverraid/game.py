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
    ClipRewardEnv
)
from loguru import logger

from cli.games.base import GameInterface, GameConfig, EvaluationResult, ProgressCallback

try:
    from cli.core.near import NEARWallet
    from cli.core.stake import StakeRecord
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False
    NEARWallet = Any  # Type alias for type hints

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
        return "ALE/RiverRaid-v5"
        
    @property
    def description(self) -> str:
        return "Control a jet, manage fuel, and destroy enemies"
        
    @property
    def version(self) -> str:
        return "1.0.0"
        
    @property
    def score_range(self) -> tuple[float, float]:
        return (0, 100000)  # River Raid scores can go very high
        
    def make_env(self):
        """Create the game environment."""
        env = gym.make(self.env_id, render_mode='rgb_array')
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env
        
    def load_model(self, model_path: str):
        """Load a trained model."""
        return DQN.load(model_path)
        
    def get_default_config(self) -> GameConfig:
        """Get default configuration for the game."""
        return GameConfig(
            total_timesteps=1_000_000,
            learning_rate=0.00025,
            buffer_size=250_000,
            learning_starts=50_000,
            batch_size=256,
            exploration_fraction=0.2,
            target_update_interval=2000,
            frame_stack=4
        )
    
    def _make_env(self, render: bool = False, record: bool = False) -> gym.Env:
        """Create the game environment with proper wrappers."""
        render_mode = "human" if render else "rgb_array"
        env = gym.make("ALE/Riverraid-v5", render_mode=render_mode, frameskip=4)
        
        # Add standard Atari wrappers
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        # Observation preprocessing
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
        env = gym.wrappers.FrameStackObservation(env, 4)
        
        # Add video recording wrapper if using rgb_array rendering
        if not render:
            env = gym.wrappers.RecordVideo(
                env,
                "videos/training",
                episode_trigger=lambda x: x % 100 == 0  # Record every 100th episode
            )
        
        return env
    
    def train(self, render: bool = False, config_path: Optional[Path] = None) -> Path:
        """Train a River Raid agent."""
        config = self.load_config(config_path)
        
        # Create vectorized environment without recording
        env = DummyVecEnv([lambda: self._make_env(render, record=False) for _ in range(4)])  # Use 4 parallel environments
        env = VecFrameStack(env, config.frame_stack)
        
        # Create and train the model with optimized parameters
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            exploration_fraction=config.exploration_fraction,
            target_update_interval=config.target_update_interval,
            tensorboard_log="./tensorboard/riverraid",
            verbose=1,
            train_freq=2,           # More frequent updates
            gradient_steps=1,       # One gradient step per update
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,  # Higher final exploration
            max_grad_norm=10,       # Clip gradients for stability
            device='auto'           # Automatically use GPU if available
        )
        
        logger.info(f"Training River Raid agent for {config.total_timesteps} timesteps...")
        logger.info("This initial training will take about 10-15 minutes.")
        logger.info("For better performance, you can increase total_timesteps to 250000 after this initial run.")
        logger.info("Monitor progress in TensorBoard: tensorboard --logdir ./tensorboard")
        
        # Use our custom ProgressCallback for consistent progress tracking
        callback = ProgressCallback(config.total_timesteps)
        
        try:
            model.learn(
                total_timesteps=config.total_timesteps,
                callback=callback,
                progress_bar=True,
                log_interval=100    # More frequent logging
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        # Save the model
        model_path = Path("models/riverraid_final.zip")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def evaluate(self, model_path: Path, episodes: int = 10, record: bool = False) -> EvaluationResult:
        """Evaluate a trained model."""
        env = DummyVecEnv([lambda: self._make_env(record)])
        env = VecFrameStack(env, 4)
        
        model = DQN.load(model_path, env=env)
        
        total_score = 0
        episode_lengths = []
        best_score = float('-inf')
        successes = 0
        
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
            if episode_score >= 1000.0:  # Success threshold
                successes += 1
        
        return EvaluationResult(
            score=total_score / episodes,
            episodes=episodes,
            success_rate=successes / episodes,
            best_episode_score=best_score,
            avg_episode_length=sum(episode_lengths) / len(episode_lengths)
        )
    
    def get_score_range(self) -> tuple[float, float]:
        """Get valid score range for the game."""
        return self.score_range
    
    def validate_model(self, model_path: Path) -> bool:
        """Validate model file."""
        try:
            env = DummyVecEnv([lambda: self._make_env()])
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
    register_game(RiverraidGame)
