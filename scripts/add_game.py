#!/usr/bin/env python3
"""Script to add a new ALE game to Agent Arcade."""
import os
import sys
import argparse
from pathlib import Path
import gymnasium as gym
from loguru import logger

TEMPLATE_GAME_PY = '''"""[GAME_NAME] implementation using ALE."""
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
from cli.core.stake import StakeRecord

class [CLASS_NAME](GameInterface):
    """[GAME_NAME] implementation."""
    
    @property
    def name(self) -> str:
        return "[GAME_ID]"
    
    @property
    def description(self) -> str:
        return "[GAME_DESCRIPTION]"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def _make_env(self, render: bool = False) -> gym.Env:
        """Create the game environment with proper wrappers."""
        render_mode = "human" if render else "rgb_array"
        env = gym.make("[ENV_ID]", render_mode=render_mode, frameskip=4)
        
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
        """Train agent."""
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
            tensorboard_log=f"./tensorboard/{self.name}"
        )
        
        logger.info(f"Training {self.name} agent for {config.total_timesteps} timesteps...")
        model.learn(total_timesteps=config.total_timesteps)
        
        # Save the model
        model_path = Path(f"models/{self.name}_final.zip")
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
            if episode_score >= [SUCCESS_THRESHOLD]:  # Success threshold
                successes += 1
        
        return EvaluationResult(
            score=total_score / episodes,
            episodes=episodes,
            success_rate=successes / episodes,
            best_episode_score=best_score,
            avg_episode_length=sum(episode_lengths) / len(episode_lengths)
        )
    
    def get_default_config(self) -> GameConfig:
        """Get default configuration."""
        return GameConfig(
            total_timesteps=1000000,
            learning_rate=0.00025,
            buffer_size=250000,
            learning_starts=50000,
            batch_size=256,
            exploration_fraction=0.2,
            target_update_interval=2000,
            frame_stack=4
        )
    
    def get_score_range(self) -> Tuple[float, float]:
        """Get score range."""
        return ([MIN_SCORE], [MAX_SCORE])
    
    def validate_model(self, model_path: Path) -> bool:
        """Validate model file."""
        try:
            env = DummyVecEnv([lambda: self._make_env()])
            DQN.load(model_path, env=env)
            return True
        except Exception as e:
            logger.error(f"Invalid model file: {e}")
            return False
    
    def stake(self, wallet: NEARWallet, model_path: Path, amount: float, target_score: float) -> None:
        """Stake NEAR on performance."""
        if not wallet.is_logged_in():
            raise ValueError("Must be logged in to stake")
        
        if not self.validate_model(model_path):
            raise ValueError("Invalid model file")
        
        # Verify target score is within range
        min_score, max_score = self.get_score_range()
        if not min_score <= target_score <= max_score:
            raise ValueError(f"Target score must be between {min_score} and {max_score}")
        
        # Create stake record
        stake_record = StakeRecord(
            game=self.name,
            model_path=str(model_path),
            amount=amount,
            target_score=target_score
        )
        
        # Record stake
        wallet.record_stake(stake_record)
        logger.info(f"Successfully staked {amount} NEAR on achieving score {target_score}")

def register():
    """Register the game."""
    from cli.games import register_game
    register_game("[GAME_ID]", [CLASS_NAME])
'''

TEMPLATE_INIT_PY = '''"""[GAME_NAME] package."""
from .game import register

__all__ = ["register"]
'''

TEMPLATE_CONFIG_YAML = '''# [GAME_NAME] Training Configuration

# Training parameters
total_timesteps: 1000000  # Total training steps
learning_rate: 0.00025    # Learning rate for optimization
buffer_size: 250000       # Size of replay buffer
learning_starts: 50000    # Steps before starting learning
batch_size: 256          # Batch size for training
exploration_fraction: 0.2 # Fraction of training for exploration
target_update_interval: 2000  # Steps between target network updates
frame_stack: 4           # Number of frames to stack

# Environment settings
env_id: "[ENV_ID]"
frame_skip: 4            # Number of frames to skip
noop_max: 30            # Max random no-ops at start

# Model architecture
policy: "CnnPolicy"      # Policy network type
features_extractor: "NatureCNN"  # CNN architecture
features_dim: 512       # Feature dimension

# Preprocessing
normalize_images: true   # Normalize pixel values
grayscale: true         # Convert to grayscale
resize_shape: [84, 84]  # Input image size

# Evaluation settings
eval_episodes: 100      # Episodes for evaluation
eval_deterministic: true # Use deterministic actions
render_eval: false      # Render during evaluation

# Logging
tensorboard_log: true   # Enable TensorBoard logging
save_freq: 100000       # Save frequency in timesteps
log_interval: 1000      # Logging interval in timesteps
'''

def validate_env_id(env_id: str) -> bool:
    """Validate that the environment ID exists in Gymnasium."""
    try:
        env = gym.make(env_id)
        env.close()
        return True
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

def create_game_files(game_name: str, env_id: str, description: str, 
                     min_score: float, max_score: float, success_threshold: float):
    """Create all necessary files for a new game."""
    # Convert game name to different formats
    game_id = game_name.lower().replace(" ", "-")
    class_name = "".join(word.capitalize() for word in game_name.split()) + "Game"
    
    # Create directories
    game_dir = Path(f"cli/games/{game_id}")
    game_dir.mkdir(parents=True, exist_ok=True)
    
    # Create game.py
    game_content = TEMPLATE_GAME_PY.replace("[GAME_NAME]", game_name)
    game_content = game_content.replace("[CLASS_NAME]", class_name)
    game_content = game_content.replace("[GAME_ID]", game_id)
    game_content = game_content.replace("[ENV_ID]", env_id)
    game_content = game_content.replace("[GAME_DESCRIPTION]", description)
    game_content = game_content.replace("[MIN_SCORE]", str(min_score))
    game_content = game_content.replace("[MAX_SCORE]", str(max_score))
    game_content = game_content.replace("[SUCCESS_THRESHOLD]", str(success_threshold))
    
    with open(game_dir / "game.py", "w") as f:
        f.write(game_content)
    
    # Create __init__.py
    init_content = TEMPLATE_INIT_PY.replace("[GAME_NAME]", game_name)
    with open(game_dir / "__init__.py", "w") as f:
        f.write(init_content)
    
    # Create config file
    config_content = TEMPLATE_CONFIG_YAML.replace("[GAME_NAME]", game_name)
    config_content = config_content.replace("[ENV_ID]", env_id)
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    with open(config_dir / f"{game_id}.yaml", "w") as f:
        f.write(config_content)
    
    logger.info(f"Successfully created game files for {game_name}")
    logger.info(f"Game implementation: {game_dir}")
    logger.info(f"Configuration: configs/{game_id}.yaml")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Add a new ALE game to Agent Arcade")
    parser.add_argument("game_name", help="Name of the game (e.g., 'Breakout')")
    parser.add_argument("env_id", help="Gymnasium environment ID (e.g., 'ALE/Breakout-v5')")
    parser.add_argument("description", help="Short description of the game")
    parser.add_argument("--min-score", type=float, required=True, help="Minimum possible score")
    parser.add_argument("--max-score", type=float, required=True, help="Maximum possible score")
    parser.add_argument("--success-threshold", type=float, required=True, 
                      help="Score threshold for considering an episode successful")
    
    args = parser.parse_args()
    
    # Validate environment ID
    if not validate_env_id(args.env_id):
        logger.error(f"Invalid environment ID: {args.env_id}")
        sys.exit(1)
    
    # Create game files
    try:
        create_game_files(
            args.game_name,
            args.env_id,
            args.description,
            args.min_score,
            args.max_score,
            args.success_threshold
        )
    except Exception as e:
        logger.error(f"Failed to create game files: {e}")
        sys.exit(1)
    
    logger.info("\nNext steps:")
    logger.info("1. Review and customize the game implementation")
    logger.info("2. Adjust hyperparameters in the config file")
    logger.info("3. Test the implementation:")
    logger.info("   agent-arcade list-games  # Verify registration")
    logger.info("   agent-arcade train your-game-name --render")
    logger.info("   agent-arcade evaluate your-game-name --model models/your_game_name_final.zip")

if __name__ == "__main__":
    main() 