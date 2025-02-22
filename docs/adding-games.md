# Adding New Games to Agent Arcade

This guide walks you through the process of adding a new Atari Learning Environment (ALE) game to Agent Arcade. **See all of the game environments at the link here: [ALE Game Environments](https://ale.farama.org/environments/).**

## Prerequisites

1. **Install Agent Arcade**

```bash
# Clone the repository
git clone https://github.com/jbarnes850/agent-arcade.git
cd agent-arcade

# Run installation script
./install.sh
```

The installation script will:

- Set up a Python virtual environment
- Install all required dependencies
- Configure ALE and Atari ROMs
- Set up NEAR CLI integration (optional)

2. **Verify ALE Installation**

```bash
# Activate virtual environment (if not already active)
source drl-env/bin/activate

# Verify required packages
pip list | grep -E "gymnasium|ale-py|shimmy|autorom"

# Install dependencies if needed
pip install "gymnasium[atari]>=0.29.1" "ale-py==0.10.1" "shimmy[atari]>=2.0.0" "autorom>=0.6.1"

# Test ALE environment registration and creation
python3 -c "
import gymnasium as gym
import ale_py
from pathlib import Path

# Register ALE environments
gym.register_envs(ale_py)

# Print ALE version
print(f'ALE version: {ale_py.__version__}')

# Verify ROM installation
rom_dir = Path(ale_py.__file__).parent / 'roms'
print(f'ROM directory: {rom_dir}')
print('Available ROMs:')
for rom in sorted(rom_dir.glob('*.bin')):
    print(f'  - {rom.name}')

# Test environment creation
env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
print('✅ Environment creation successful')

# Test observation preprocessing
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
env = gym.wrappers.FrameStackObservation(env, 4)
obs, _ = env.reset()
print(f'Observation shape: {obs.shape}')  # Should be (4, 84, 84)
print('✅ Observation preprocessing verified')
"

# Verify CLI works
agent-arcade --version
```

3. **Common Installation Issues**

If you encounter any issues:

a) **Missing ROMs**:

```bash
# Install AutoROM and download ROMs
pip install "autorom>=0.6.1"
python -m AutoROM --accept-license
```

b) **ALE Namespace Not Found**:

```bash
# Ensure ALE environments are registered
python3 -c "
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)  # This line is crucial
env = gym.make('ALE/Pong-v5')
"
```

c) **Incorrect Observation Shape**:

```python
# Ensure correct wrapper order and parameters
env = gym.make('ALE/YourGame-v5', render_mode='rgb_array')
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)  # keep_dim=False is important
env = gym.wrappers.FrameStackObservation(env, 4)  # Use FrameStackObservation, not FrameStack
```

## Quick Start Template

We provide two ways to add a new game:

### Option 1: Using the Automation Script (Recommended)

```bash
# Activate virtual environment if not already active
source drl-env/bin/activate

# Add a new game (example: Breakout)
python scripts/add_game.py "Breakout" "ALE/Breakout-v5" "Classic brick-breaking arcade game" \
    --min-score 0 \
    --max-score 864 \
    --success-threshold 100
```

### Option 2: Manual Setup

```bash
# 1. Create new game directory
mkdir -p cli/games/your_game_name

# 2. Create game implementation files
touch cli/games/your_game_name/__init__.py
touch cli/games/your_game_name/game.py

# 3. Create configuration file
touch configs/your_game_name.yaml
```

## Environment Setup

The environment setup is crucial for proper training. Here's the standard wrapper stack used in Agent Arcade:

```python
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
```

### Wrapper Explanation

1. **Base Environment**:
   - `frameskip=4`: Process every 4th frame for efficiency
   - `render_mode`: "human" for visualization, "rgb_array" for training

2. **Atari-specific Wrappers**:
   - `NoopResetEnv`: Random number of no-ops at start
   - `MaxAndSkipEnv`: Frame skipping and max pooling
   - `EpisodicLifeEnv`: End episode on life loss
   - `FireResetEnv`: Press FIRE to start games that require it

3. **Observation Processing**:
   - `ResizeObservation`: Resize to 84x84 pixels
   - `GrayscaleObservation`: Convert to grayscale (keep_dim=False)
   - `FrameStackObservation`: Stack 4 frames for temporal information

4. **Recording** (during training):
   - `RecordVideo`: Save episode videos for visualization

The final observation shape will be `(4, 84, 84)` representing 4 stacked grayscale frames.

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

1. **Verify Environment**:

```bash
# Activate virtual environment if not already active
source drl-env/bin/activate

# Test environment creation
python3 -c "import gymnasium; env = gymnasium.make('ALE/YourGame-v5')"
```

2. **Verify Game Registration**:

```bash
agent-arcade list-games  # Your game should appear
```

3. **Test Training**:

```bash
agent-arcade train your-game-name --render
```

4. **Test Evaluation**:

```bash
agent-arcade evaluate your-game-name --model models/your_game_name_final.zip
```

## Troubleshooting

### Common Issues

1. **Environment Not Found**

```bash
# Verify ALE installation
python3 -c "import ale_py; print(ale_py.__version__)"

# Check ROM installation
python3 -c "import ale_py; from pathlib import Path; print(Path(ale_py.__file__).parent / 'roms')"
```

2. **Observation Shape Issues**

```python
# Debug observation shape
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")  # Should be (4, 84, 84)

# Common fixes:
# 1. Ensure correct wrapper order
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)  # keep_dim=False is important
env = gym.wrappers.FrameStackObservation(env, 4)  # Use FrameStackObservation, not FrameStack

# 2. Check vectorized environment
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4)
```

3. **Package Version Issues**

```bash
# Required versions
pip install "gymnasium[atari]>=0.29.1"
pip install "ale-py==0.10.1"
pip install "shimmy[atari]>=2.0.0"
pip install "stable-baselines3[extra]>=2.5.0"
pip install "autorom>=0.6.1"
```

4. **ROM Installation Issues**

```bash
# Verify ROM installation
python3 -c "
import ale_py
from pathlib import Path
rom_dir = Path(ale_py.__file__).parent / 'roms'
print(f'ROM directory: {rom_dir}')
print('Available ROMs:')
for rom in sorted(rom_dir.glob('*.bin')):
    print(f'  - {rom.name}')
"

# Reinstall ROMs if needed
python3 -m AutoROM --accept-license
```

5. **Training Issues**

```bash
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor training progress
tensorboard --logdir ./tensorboard

# Check video recordings
ls -l videos/training/
```

## Best Practices

1. **Environment Configuration**:
   - Always use the standard wrapper stack
   - Maintain the correct wrapper order
   - Verify observation shapes before training
   - Test environment with both render modes

2. **Training Configuration**:
   - Start with default hyperparameters
   - Adjust based on game complexity
   - Monitor training progress with TensorBoard
   - Save checkpoints regularly

3. **Testing**:
   - Verify environment creation
   - Test with and without rendering
   - Check model saving/loading
   - Validate observation shapes
   - Test video recording

4. **Documentation**:
   - Document game-specific parameters
   - Include expected score ranges
   - Note any special requirements
   - Add example configurations

## Resources

- [Gymnasium Atari Documentation](https://gymnasium.farama.org/environments/atari/)
- [ALE Documentation](https://ale.farama.org/environments/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Agent Arcade GitHub](https://github.com/jbarnes850/agent-arcade)

## Example Games

See our reference implementations:

- `cli/games/pong/` - Simple game with basic dynamics
- `cli/games/space_invaders/` - Complex game with multiple objects

For reference on structure and best practices.

## Advanced Topics

### 1. Custom Environment Wrappers

For games requiring special handling:

```python
class CustomRewardWrapper(gym.Wrapper):
    """Example custom reward wrapper."""
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Modify reward based on game-specific logic
        modified_reward = self._calculate_reward(reward, info)
        return obs, modified_reward, terminated, truncated, info
    
    def _calculate_reward(self, reward, info):
        # Implement custom reward logic
        return reward

# Use in _make_env
env = CustomRewardWrapper(env)
```

### 2. Advanced Training Configuration

For complex games needing special treatment:

```yaml
# Advanced configuration options
advanced_training:
  # Prioritized Experience Replay
  prioritized_replay: true
  alpha: 0.6
  beta0: 0.4
  
  # N-step Learning
  n_step: 3
  
  # Dueling Network
  dueling: true
  
  # Double Q-Learning
  double_q: true
  
  # Gradient Clipping
  max_grad_norm: 10
```

### 3. Custom Feature Extraction

For games with unique visual patterns:

```python
import torch.nn as nn

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# Use in training
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512)
)
```

### 4. Performance Optimization

Tips for improving training efficiency:

1. **Memory Management**:

```python
# Clear GPU memory between training runs
import torch
torch.cuda.empty_cache()

# Monitor memory usage
from pynvml import *
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print(f"Free memory: {info.free/1024**2:.2f}MB")
```

2. **Vectorized Environments**:

```python
# Use multiple environments for parallel training
n_envs = 4  # Number of parallel environments
env = SubprocVecEnv([lambda: make_env() for _ in range(n_envs)])
```

3. **Frame Skipping**:

```python
# Implement efficient frame skipping
env = MaxAndSkipEnv(env, skip=4)  # Process every 4th frame
```

### 5. Testing Framework

Example test suite structure:

```python
# tests/games/test_your_game.py
import pytest
from cli.games.your_game_name.game import YourGameNameGame

def test_environment_creation():
    game = YourGameNameGame()
    env = game._make_env(render=False)
    assert env is not None
    env.close()

def test_model_training():
    game = YourGameNameGame()
    model_path = game.train(render=False, total_timesteps=1000)
    assert model_path.exists()

def test_evaluation():
    game = YourGameNameGame()
    model_path = Path("models/test_model.zip")
    result = game.evaluate(model_path, episodes=2)
    assert result.episodes == 2
```

## Contributing Guidelines

1. **Code Style**:
   - Follow PEP 8 guidelines
   - Use type hints
   - Document all public methods
   - Add meaningful comments

2. **Testing Requirements**:
   - Unit tests for core functionality
   - Integration tests for training/evaluation
   - Performance benchmarks
   - Documentation updates

3. **Pull Request Process**:
   - Create feature branch
   - Add tests
   - Update documentation
   - Submit PR with description

Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Guide](https://stable-baselines3.readthedocs.io/)
- [NEAR Protocol Docs](https://docs.near.org/)
- [Atari Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
