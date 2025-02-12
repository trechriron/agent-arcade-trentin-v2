"""Space Invaders game implementation using ALE."""
import gymnasium as gym
from pathlib import Path
from typing import Optional, Tuple
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

from cli.games.base import GameInterface, GameConfig, EvaluationResult
from cli.core.near import NEARWallet

class SpaceInvadersGame(GameInterface):
    """Space Invaders game implementation."""
    
    @property
    def name(self) -> str:
        return "space-invaders"
    
    @property
    def description(self) -> str:
        return "Classic Space Invaders - defend Earth from alien invasion"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def _make_env(self, render: bool = False) -> gym.Env:
        """Create the Space Invaders environment with proper wrappers."""
        render_mode = "human" if render else "rgb_array"
        env = gym.make("ALE/SpaceInvaders-v5", render_mode=render_mode, frameskip=1)
        
        # Add standard Atari wrappers
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        # Observation preprocessing
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        
        return env
    
    def train(self, render: bool = False, config_path: Optional[Path] = None) -> Path:
        """Train a Space Invaders agent."""
        config = self.load_config(config_path)
        
        # Create vectorized environment
        env = DummyVecEnv([lambda: self._make_env(render)])
        env = VecFrameStack(env, config.frame_stack)
        
        # Create and train the model with optimized hyperparameters
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            exploration_fraction=config.exploration_fraction,
            target_update_interval=config.target_update_interval,
            tensorboard_log="./tensorboard/space_invaders"
        )
        
        logger.info(f"Training Space Invaders agent for {config.total_timesteps} timesteps...")
        model.learn(total_timesteps=config.total_timesteps)
        
        # Save the model
        model_path = Path("models/space_invaders_final.zip")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    def evaluate(self, model_path: Path, episodes: int = 10, record: bool = False) -> EvaluationResult:
        """Evaluate a trained Space Invaders model."""
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
            if episode_score >= 300:  # Consider clearing first wave as success
                successes += 1
        
        return EvaluationResult(
            score=total_score / episodes,
            episodes=episodes,
            success_rate=successes / episodes,
            best_episode_score=best_score,
            avg_episode_length=sum(episode_lengths) / len(episode_lengths)
        )
    
    def get_default_config(self) -> GameConfig:
        """Get default Space Invaders configuration."""
        return GameConfig(
            total_timesteps=2000000,  # Space Invaders needs more training
            learning_rate=0.0001,     # Lower learning rate for stability
            buffer_size=500000,       # Larger buffer for more experience
            learning_starts=100000,
            batch_size=128,
            exploration_fraction=0.1,
            target_update_interval=1000,
            frame_stack=4
        )
    
    def get_score_range(self) -> Tuple[float, float]:
        """Get Space Invaders score range."""
        return (0.0, 1000.0)  # Typical score range for Space Invaders
    
    def validate_model(self, model_path: Path) -> bool:
        """Validate Space Invaders model file."""
        try:
            env = DummyVecEnv([lambda: self._make_env()])
            DQN.load(model_path, env=env)
            return True
        except Exception as e:
            logger.error(f"Invalid model file: {e}")
            return False
    
    def stake(self, wallet: NEARWallet, model_path: Path, amount: float, target_score: float) -> None:
        """Stake NEAR on Space Invaders performance."""
        # This will be implemented in Phase 3
        pass

def register():
    """Register the Space Invaders game."""
    from cli.games import register_game
    register_game("space-invaders", SpaceInvadersGame) 