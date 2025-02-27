#!/usr/bin/env python3
"""Script to add a new ALE game to Agent Arcade."""
import os
import sys
import argparse
from pathlib import Path
import gymnasium as gym
from loguru import logger
import numpy as np
import torch
import re

TEMPLATE_GAME_PY = (
    '"""[GAME_NAME] implementation using ALE."""\n'
    'import gymnasium as gym\n'
    'from pathlib import Path\n'
    'from typing import Optional, Tuple, Any\n'
    'from stable_baselines3 import DQN\n'
    'from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack\n'
    'from stable_baselines3.common.atari_wrappers import (\n'
    '    NoopResetEnv,\n'
    '    MaxAndSkipEnv,\n'
    '    EpisodicLifeEnv,\n'
    '    FireResetEnv,\n'
    '    ClipRewardEnv\n'
    ')\n'
    'from loguru import logger\n'
    'import numpy as np\n'
    'import torch\n\n'
    'from cli.games.base import GameInterface, GameConfig, ProgressCallback\n'
    'from cli.core.evaluation import EvaluationResult, EvaluationConfig, EvaluationPipeline, GameSpecificConfig\n\n'
    'try:\n'
    '    from cli.core.near import NEARWallet\n'
    '    from cli.core.stake import StakeRecord\n'
    '    NEAR_AVAILABLE = True\n'
    'except ImportError:\n'
    '    NEAR_AVAILABLE = False\n'
    '    NEARWallet = Any  # Type alias for type hints\n\n'
    'class ScaleObservation(gym.ObservationWrapper):\n'
    '    """Scale observations to [0, 1]."""\n'
    '    \n'
    '    def __init__(self, env):\n'
    '        super().__init__(env)\n'
    '        self.observation_space = gym.spaces.Box(\n'
    '            low=0, high=1,\n'
    '            shape=self.observation_space.shape,\n'
    '            dtype=np.float32\n'
    '        )\n'
    '    \n'
    '    def observation(self, obs):\n'
    '        return obs.astype(np.float32) / 255.0\n\n'
    'class TransposeObservation(gym.ObservationWrapper):\n'
    '    """Transpose observation for PyTorch CNN input."""\n'
    '    \n'
    '    def __init__(self, env):\n'
    '        super().__init__(env)\n'
    '        obs_shape = self.observation_space.shape\n'
    '        self.observation_space = gym.spaces.Box(\n'
    '            low=0, high=1,\n'
    '            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),\n'
    '            dtype=np.float32\n'
    '        )\n'
    '    \n'
    '    def observation(self, obs):\n'
    '        return np.transpose(obs, (2, 0, 1))\n\n'
    'class [CLASS_NAME](GameInterface):\n'
    '    """[GAME_NAME] implementation."""\n'
    '    \n'
    '    def __call__(self):\n'
    '        """Make the class callable to return itself."""\n'
    '        return self\n'
    '    \n'
    '    @property\n'
    '    def name(self) -> str:\n'
    '        return "[GAME_ID]"\n'
    '        \n'
    '    @property\n'
    '    def env_id(self) -> str:\n'
    '        return "[ENV_ID]"\n'
    '        \n'
    '    @property\n'
    '    def description(self) -> str:\n'
    '        return "[GAME_DESCRIPTION]"\n'
    '        \n'
    '    @property\n'
    '    def version(self) -> str:\n'
    '        return "1.0.0"\n'
    '        \n'
    '    @property\n'
    '    def score_range(self) -> tuple[float, float]:\n'
    '        return ([MIN_SCORE], [MAX_SCORE])\n'
    '        \n'
    '    def get_score_range(self) -> tuple[float, float]:\n'
    '        """Get valid score range for the game."""\n'
    '        return self.score_range\n'
    '        \n'
    '    def make_env(self):\n'
    '        """Create the game environment."""\n'
    '        return self._make_env()\n'
    '        \n'
    '    def load_model(self, model_path: str):\n'
    '        """Load a trained model."""\n'
    '        return DQN.load(model_path)\n'
    '        \n'
    '    def get_default_config(self) -> GameConfig:\n'
    '        """Get default configuration for the game."""\n'
    '        return GameConfig(\n'
    '            total_timesteps=5_000_000,    # Extended training for better strategies\n'
    '            learning_rate=0.00025,        # Standard DQN learning rate\n'
    '            buffer_size=1_000_000,        # Increased for H100 memory capacity\n'
    '            learning_starts=50_000,       # More initial exploration\n'
    '            batch_size=1024,              # Larger batches for H100\n'
    '            exploration_fraction=0.2,      # More exploration for complex strategies\n'
    '            target_update_interval=2000,   # Less frequent updates for stability\n'
    '            frame_stack=4,                # Standard Atari frame stack\n'
    '            policy="CnnPolicy",\n'
    '            tensorboard_log=True,\n'
    '            log_interval=100              # Frequent logging\n'
    '        )\n'
    '    \n'
    '    def _make_env(self, render: bool = False, config: Optional[GameConfig] = None) -> gym.Env:\n'
    '        """Create the game environment with proper wrappers."""\n'
    '        if config is None:\n'
    '            config = self.get_default_config()\n'
    '        \n'
    '        render_mode = "human" if render else "rgb_array"\n'
    '        env = gym.make(\n'
    '            self.env_id,\n'
    '            render_mode=render_mode\n'
    '        )\n'
    '        \n'
    '        # Add standard Atari wrappers in correct order\n'
    '        env = NoopResetEnv(env, noop_max=30)\n'
    '        env = MaxAndSkipEnv(env, skip=4)\n'
    '        env = EpisodicLifeEnv(env)\n'
    '        if "FIRE" in env.unwrapped.get_action_meanings():\n'
    '            env = FireResetEnv(env)\n'
    '        env = ClipRewardEnv(env)\n'
    '        \n'
    '        # Standard observation preprocessing\n'
    '        env = gym.wrappers.ResizeObservation(env, (84, 84))\n'
    '        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)\n'
    '        env = ScaleObservation(env)  # Scale to [0, 1]\n'
    '        \n'
    '        # Transpose for PyTorch: (H, W, C) -> (C, H, W)\n'
    '        env = TransposeObservation(env)\n'
    '        \n'
    '        # Debug observation space\n'
    '        logger.debug(f"Single env observation space before vectorization: {env.observation_space}")\n'
    '        return env\n'
    '    \n'
    '    def train(self, render: bool = False, config_path: Optional[Path] = None) -> Path:\n'
    '        """Train an agent for this game."""\n'
    '        config = self.load_config(config_path)\n'
    '        \n'
    '        def make_env():\n'
    '            env = self._make_env(render, config)\n'
    '            return env\n'
    '        \n'
    '        # Create vectorized environment with more parallel envs for H100\n'
    '        env = DummyVecEnv([make_env for _ in range(16)])  # Increased from 8 to 16\n'
    '        \n'
    '        # Stack frames in the correct order for SB3 (n_envs, n_stack, h, w)\n'
    '        env = VecFrameStack(env, n_stack=4, channels_order=\'first\')\n'
    '        \n'
    '        # Debug final observation space\n'
    '        logger.debug(f"Final observation space: {env.observation_space}")\n'
    '        \n'
    '        # Create and train the model with optimized policy network for H100\n'
    '        model = DQN(\n'
    '            "CnnPolicy",\n'
    '            env,\n'
    '            learning_rate=config.learning_rate,\n'
    '            buffer_size=config.buffer_size,\n'
    '            learning_starts=config.learning_starts,\n'
    '            batch_size=config.batch_size,\n'
    '            exploration_fraction=config.exploration_fraction,\n'
    '            target_update_interval=config.target_update_interval,\n'
    '            tensorboard_log=f"./tensorboard/{self.name}" if config.tensorboard_log else None,\n'
    '            policy_kwargs={\n'
    '                "net_arch": [1024, 1024],  # Larger network for H100\n'
    '                "normalize_images": False,  # Images are already normalized\n'
    '                "optimizer_class": torch.optim.Adam,\n'
    '                "optimizer_kwargs": {\n'
    '                    "eps": 1e-5,\n'
    '                    "weight_decay": 1e-6\n'
    '                }\n'
    '            },\n'
    '            train_freq=(4, "step"),       # Update every 4 steps\n'
    '            gradient_steps=4,             # Multiple gradient steps per update\n'
    '            verbose=1,\n'
    '            device="cuda",\n'
    '            optimize_memory_usage=True     # Memory optimization for H100\n'
    '        )\n'
    '        \n'
    '        # Add progress callback\n'
    '        callback = ProgressCallback(config.total_timesteps)\n'
    '        \n'
    '        logger.info(f"Training {self.name} agent for {config.total_timesteps} timesteps...")\n'
    '        if config.tensorboard_log:\n'
    '            logger.info("Monitor progress in TensorBoard: tensorboard --logdir ./tensorboard")\n'
    '        \n'
    '        model.learn(\n'
    '            total_timesteps=config.total_timesteps,\n'
    '            callback=callback,\n'
    '            log_interval=config.log_interval\n'
    '        )\n'
    '        \n'
    '        # Save the model\n'
    '        model_path = Path(f"models/{self.name}/baseline/final_model.zip")\n'
    '        model_path.parent.mkdir(parents=True, exist_ok=True)\n'
    '        model.save(str(model_path))\n'
    '        logger.info(f"Model saved to {model_path}")\n'
    '        \n'
    '        return model_path\n'
    '    \n'
    '    def evaluate(self, model_path: Path, episodes: int = 10, record: bool = False) -> EvaluationResult:\n'
    '        """Evaluate a trained model."""\n'
    '        # Load model metadata to get frame stack configuration\n'
    '        config = self.get_default_config()\n'
    '        metadata_path = model_path.parent / "metadata.json"\n'
    '        if metadata_path.exists():\n'
    '            import json\n'
    '            with open(metadata_path) as f:\n'
    '                metadata = json.load(f)\n'
    '                if "hyperparameters" in metadata:\n'
    '                    config.frame_stack = metadata["hyperparameters"].get("frame_stack", 16)\n'
    '                    logger.debug(f"Using frame_stack={config.frame_stack} from metadata")\n'
    '        \n'
    '        # Create environment with correct frame stack size\n'
    '        env = DummyVecEnv([lambda: self._make_env(record, config)])\n'
    '        env = VecFrameStack(env, n_stack=4, channels_order=\'first\')\n'
    '        model = DQN.load(model_path, env=env)\n'
    '        \n'
    '        total_score = 0\n'
    '        episode_lengths = []\n'
    '        episode_rewards = []\n'
    '        best_score = float(\'-inf\')\n'
    '        successes = 0\n'
    '        \n'
    '        logger.info(f"Evaluating model for {episodes} episodes...")\n'
    '        logger.debug(f"Using frame stack size: {config.frame_stack}")\n'
    '        if record:\n'
    '            logger.info("Recording evaluation videos to videos/evaluation/")\n'
    '        \n'
    '        for episode in range(episodes):\n'
    '            obs = env.reset()[0]\n'
    '            done = False\n'
    '            episode_score = 0\n'
    '            episode_length = 0\n'
    '            \n'
    '            while not done:\n'
    '                action, _ = model.predict(obs, deterministic=True)\n'
    '                obs, reward, terminated, truncated, info = env.step(action)\n'
    '                episode_score += reward[0]\n'
    '                episode_length += 1\n'
    '                done = terminated[0] or truncated[0]\n'
    '                \n'
    '                # Track success using info dict for consistency with evaluation.py\n'
    '                if isinstance(info, (list, tuple)):\n'
    '                    info = info[0]  # Get first env\'s info\n'
    '                if isinstance(info, dict) and info.get("is_success", False):\n'
    '                    successes += 1\n'
    '                elif episode_score > 100:  # Fallback success criteria\n'
    '                    successes += 1\n'
    '            \n'
    '            total_score += episode_score\n'
    '            episode_lengths.append(episode_length)\n'
    '            episode_rewards.append(episode_score)\n'
    '            best_score = max(best_score, episode_score)\n'
    '            \n'
    '            logger.info(f"Episode {episode + 1}/{episodes} - Score: {episode_score:.2f}")\n'
    '        \n'
    '        env.close()\n'
    '        \n'
    '        result = EvaluationResult(\n'
    '            mean_reward=total_score / episodes,\n'
    '            std_reward=np.std(episode_rewards),\n'
    '            n_episodes=episodes,\n'
    '            success_rate=successes / episodes,\n'
    '            episode_lengths=episode_lengths,\n'
    '            episode_rewards=episode_rewards,\n'
    '            metadata={},\n'
    '            game_config=GameSpecificConfig(game_id=self.name, score_range=self.score_range)\n'
    '        )\n'
    '        \n'
    '        logger.info(f"Evaluation complete - Average score: {result.mean_reward:.2f}")\n'
    '        return result\n'
    '    \n'
    '    def validate_model(self, model_path: Path) -> bool:\n'
    '        """Validate model file."""\n'
    '        try:\n'
    '            env = DummyVecEnv([lambda: self._make_env()])\n'
    '            env = VecFrameStack(env, n_stack=4, channels_order=\'first\')\n'
    '            DQN.load(model_path, env=env)\n'
    '            return True\n'
    '        except Exception as e:\n'
    '            logger.error(f"Invalid model file: {e}")\n'
    '            return False\n'
    '    \n'
    '    async def stake(self, wallet: NEARWallet, model_path: Path, amount: float, target_score: float) -> None:\n'
    '        """Stake NEAR on performance."""\n'
    '        if not NEAR_AVAILABLE:\n'
    '            raise RuntimeError("NEAR integration not available")\n'
    '            \n'
    '        if not wallet.is_logged_in():\n'
    '            raise ValueError("Must be logged in to stake")\n'
    '        \n'
    '        if not self.validate_model(model_path):\n'
    '            raise ValueError("Invalid model file")\n'
    '        \n'
    '        # Verify target score is within range\n'
    '        min_score, max_score = self.score_range\n'
    '        if not min_score <= target_score <= max_score:\n'
    '            raise ValueError(f"Target score must be between {min_score} and {max_score}")\n'
    '        \n'
    '        # Use staking module\n'
    '        await stake_on_game(\n'
    '            wallet=wallet,\n'
    '            game_name=self.name,\n'
    '            model_path=model_path,\n'
    '            amount=amount,\n'
    '            target_score=target_score,\n'
    '            score_range=self.score_range\n'
    '        )\n'
    '\n'
    'def register():\n'
    '    """Register the game."""\n'
    '    from cli.games import register_game\n'
    '    register_game([CLASS_NAME])'
)

def register():
    """Register the game."""
    from cli.games import register_game
    register_game([CLASS_NAME])

def validate_game_id(game_id: str) -> bool:
    """Validate game ID format."""
    return bool(re.match(r'^[a-z0-9_]+$', game_id))

def validate_class_name(class_name: str) -> bool:
    """Validate class name format."""
    return bool(re.match(r'^[A-Z][a-zA-Z0-9]*Game$', class_name))

def create_game_files(game_id: str, class_name: str, env_id: str, description: str, min_score: float, max_score: float) -> None:
    """Create game implementation files."""
    # Create game directory
    game_dir = Path('cli/games') / game_id
    game_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py
    init_path = game_dir / '__init__.py'
    init_path.write_text(f'"""Game module for {game_id}."""\nfrom .game import {class_name}\n')
    
    # Create game.py with template
    game_path = game_dir / 'game.py'
    game_content = TEMPLATE_GAME_PY.replace('[GAME_NAME]', game_id.replace('_', ' ').title())
    game_content = game_content.replace('[CLASS_NAME]', class_name)
    game_content = game_content.replace('[GAME_ID]', game_id)
    game_content = game_content.replace('[ENV_ID]', env_id)
    game_content = game_content.replace('[GAME_DESCRIPTION]', description)
    game_content = game_content.replace('[MIN_SCORE]', str(min_score))
    game_content = game_content.replace('[MAX_SCORE]', str(max_score))
    game_path.write_text(game_content)
    
    # Create model directories
    model_dir = Path('models') / game_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'baseline').mkdir(exist_ok=True)
    (model_dir / 'checkpoints').mkdir(exist_ok=True)
    
    # Create tensorboard directory
    tensorboard_dir = Path('tensorboard') / game_id
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Create video directories
    video_dir = Path('videos') / game_id
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / 'training').mkdir(exist_ok=True)
    (video_dir / 'evaluation').mkdir(exist_ok=True)
    
    logger.info(f"Created game files for {game_id} in {game_dir}")
    logger.info(f"Created model directories in {model_dir}")
    logger.info(f"Created tensorboard directory in {tensorboard_dir}")
    logger.info(f"Created video directories in {video_dir}")

def main():
    """Main entry point."""
    if len(sys.argv) != 7:
        print("Usage: python add_game.py <game_id> <class_name> <env_id> <description> <min_score> <max_score>")
        print("Example: python add_game.py space_invaders SpaceInvadersGame ALE/SpaceInvaders-v5 'Space Invaders game' -100 1000")
        sys.exit(1)
    
    game_id = sys.argv[1]
    class_name = sys.argv[2]
    env_id = sys.argv[3]
    description = sys.argv[4]
    
    try:
        min_score = float(sys.argv[5])
        max_score = float(sys.argv[6])
    except ValueError:
        logger.error("Min and max scores must be numbers")
        sys.exit(1)
    
    # Validate inputs
    if not validate_game_id(game_id):
        logger.error("Invalid game ID. Must be lowercase with underscores only.")
        sys.exit(1)
    
    if not validate_class_name(class_name):
        logger.error("Invalid class name. Must be PascalCase and end with 'Game'.")
        sys.exit(1)
    
    if not env_id.startswith('ALE/'):
        logger.error("Invalid env ID. Must start with 'ALE/'.")
        sys.exit(1)
    
    if min_score >= max_score:
        logger.error("Min score must be less than max score.")
        sys.exit(1)
    
    # Create game files
    try:
        create_game_files(game_id, class_name, env_id, description, min_score, max_score)
        logger.info(f"Successfully added {game_id} to Agent Arcade!")
        logger.info(f"Next steps:")
        logger.info(f"1. Review the generated files in cli/games/{game_id}/")
        logger.info(f"2. Test the game with: python -m cli.games.{game_id}.game")
        logger.info(f"3. Train an agent with: python -m cli train {game_id}")
        logger.info(f"4. Evaluate the agent with: python -m cli evaluate {game_id}")
    except Exception as e:
        logger.error(f"Failed to create game files: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 