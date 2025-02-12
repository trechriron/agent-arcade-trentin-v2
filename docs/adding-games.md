# Adding New Games to Agent Arcade

This guide walks you through the process of adding a new Atari Learning Environment (ALE) game to Agent Arcade.

## Quick Start Template

Use this template to quickly add a new game:

```bash
# 1. Create new game directory
mkdir -p cli/games/your_game_name

# 2. Create game implementation files
touch cli/games/your_game_name/__init__.py
touch cli/games/your_game_name/game.py

# 3. Create configuration file
touch configs/your_game_name.yaml
```

## Step-by-Step Guide

### 1. Game Implementation

Create `cli/games/your_game_name/game.py`:

```python
"""[Game Name] implementation using ALE."""
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

class YourGameNameGame(GameInterface):
    """[Game Name] implementation."""
    
    @property
    def name(self) -> str:
        return "your-game-name"
    
    @property
    def description(self) -> str:
        return "Description of your game"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def _make_env(self, render: bool = False) -> gym.Env:
        """Create the game environment with proper wrappers."""
        render_mode = "human" if render else "rgb_array"
        env = gym.make("ALE/YourGame-v5", render_mode=render_mode, frameskip=1)
        
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
            if episode_score >= YOUR_SUCCESS_THRESHOLD:  # Define success threshold
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
        return (MIN_SCORE, MAX_SCORE)  # Define your game's score range
    
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
    register_game("your-game-name", YourGameNameGame)
```

### 2. Game Registration

Create `cli/games/your_game_name/__init__.py`:

```python
"""[Game Name] package."""
from .game import register

__all__ = ["register"]
```

### 3. Configuration File

Create `configs/your_game_name.yaml`:

```yaml
# [Game Name] Training Configuration

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
env_id: "ALE/YourGame-v5"
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
```

## Key Considerations

1. **Environment ID**: Find the correct ALE environment ID from [Gymnasium ALE](https://gymnasium.farama.org/environments/atari/).

2. **Score Range**: Define appropriate `MIN_SCORE` and `MAX_SCORE` for your game.

3. **Success Threshold**: Set `YOUR_SUCCESS_THRESHOLD` based on what constitutes good performance.

4. **Hyperparameters**: Adjust training parameters in the config file based on game complexity.

5. **Environment Wrappers**: Add game-specific wrappers if needed (e.g., `FireResetEnv` for games requiring FIRE to start).

## Testing Your Implementation

1. **Validate Registration**:

```bash
agent-arcade list-games  # Your game should appear
```

2. **Test Training**:

```bash
agent-arcade train your-game-name --render
```

3. **Test Evaluation**:

```bash
agent-arcade evaluate your-game-name --model models/your_game_name_final.zip
```

4. **Test Staking**:

```bash
agent-arcade stake your-game-name --model models/your_game_name_final.zip --amount 10 --target-score TARGET
```

## Common Issues & Solutions

1. **Environment Not Found**:
   - Ensure the game is available in ALE
   - Check environment ID spelling
   - Verify Gymnasium installation

2. **Training Issues**:
   - Adjust learning rate if training is unstable
   - Increase buffer size for complex games
   - Modify frame stack for temporal dependencies

3. **Performance Issues**:
   - Add game-specific wrappers
   - Tune reward scaling
   - Adjust success threshold

## Best Practices

1. **Documentation**:
   - Add clear game description
   - Document any game-specific parameters
   - Include expected score ranges

2. **Testing**:
   - Test all CLI commands
   - Verify model saving/loading
   - Check staking functionality

3. **Configuration**:
   - Start with default parameters
   - Document any deviations
   - Include game-specific settings

## Example Games

See implementations of:

- `cli/games/pong/`
- `cli/games/space_invaders/`

For reference on structure and best practices.
