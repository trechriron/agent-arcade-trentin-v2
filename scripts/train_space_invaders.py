import yaml
import gymnasium as gym
import torch
import sys
import os
import time
import signal
import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Space Invaders.
    Similar to Pong but with adjusted architecture for Space Invaders' specific visual patterns.
    """
    def __init__(self, observation_space, features_dim=512):
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
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

class SpaceInvadersRewardScaler(gym.Wrapper):
    """Custom reward shaping for Space Invaders"""
    def __init__(self, env):
        super().__init__(env)
        self._current_score = 0
        self._current_lives = 3

    def reset(self, **kwargs):
        self._current_score = 0
        self._current_lives = 3
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Base reward from score difference
        score_diff = info.get('score', 0) - self._current_score
        self._current_score = info.get('score', 0)
        
        # Life loss penalty
        lives = info.get('lives', 3)
        life_loss = self._current_lives - lives
        self._current_lives = lives
        
        # Reward shaping
        shaped_reward = 0.0
        
        # Score-based reward
        if score_diff > 0:
            shaped_reward += score_diff * 0.1  # Base scoring reward
            
        # Survival bonus (small positive reward for staying alive)
        if not terminated and not truncated:
            shaped_reward += 0.1
            
        # Life loss penalty
        if life_loss > 0:
            shaped_reward -= 1.0
            
        # Terminal state penalties/rewards
        if terminated:
            if lives == 0:  # Game over
                shaped_reward -= 2.0
            else:  # Level complete
                shaped_reward += 5.0
                
        return obs, shaped_reward, terminated, truncated, info

def make_env(rank, seed=0):
    def _init():
        # Create environment with specific seed
        env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
        
        # Basic Atari wrappers
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        # Custom reward shaping
        env = SpaceInvadersRewardScaler(env)
        
        # Observation preprocessing
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        
        # Set seeds
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        
        return env
    return _init

class WorkshopCallback(BaseCallback):
    def __init__(self, viz_interval, video_interval, video_length, checkpoint_interval, 
                 demo_mode=False, record_video=False):
        super().__init__()
        self.viz_interval = viz_interval
        self.video_interval = video_interval
        self.video_length = video_length
        self.checkpoint_interval = checkpoint_interval
        self.demo_mode = demo_mode
        self.record_video = record_video
        self.viz_enabled = False
        self.best_reward = float('-inf')
        self.episode_rewards = []
        self.last_viz_step = 0
        
    def _on_step(self):
        # Record episode rewards
        if self.locals.get('done'):
            reward = self.locals.get('rewards')[0]
            self.episode_rewards.append(reward)
            
            # Record video for significant improvements
            if self.record_video and reward > self.best_reward:
                self.best_reward = reward
                self._record_video(f"milestone_reward_{reward}")
        
        return True
    
    def _record_video(self, name):
        video_path = f"videos/{self.num_timesteps}_{name}"
        Path("videos").mkdir(exist_ok=True)
        
        # Create a separate environment for recording using the same make_env function
        video_env = DummyVecEnv([
            make_env(
                rank=0,
                seed=42  # Fixed seed for consistent visualization
            )
        ])
        video_env = VecFrameStack(video_env, n_stack=4)  # Match the training environment's frame stacking
        
        video_env = VecVideoRecorder(
            video_env,
            video_path,
            record_video_trigger=lambda x: x < self.video_length,
            video_length=self.video_length
        )
        
        # Record episode using current policy
        obs = video_env.reset()
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, infos = video_env.step(action)
            if dones.any():
                obs = video_env.reset()
        
        video_env.close()

def get_device():
    """
    Auto-detect and return the best available device for training.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        torch.backends.mps.enable_mps = True
        return "mps"
    return "cpu"

def train(config_path, render_training=False, use_wandb=False, demo_mode=False, record_video=False):
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}\nPlease ensure the config file exists.")
    
    # Auto-detect device
    device = get_device()
    print(f"\nUsing device: {device.upper()}")
    
    # Set performance optimizations
    if device == "mps":
        torch.set_num_threads(6)  # Optimize thread usage for M1/M2
    
    # Load YAML config
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error reading config file: {e}\nPlease check the YAML syntax.")

    # Extract configuration
    algo_name = config.get("algo", "DQN")
    env_id = config.get("env", "ALE/SpaceInvaders-v5")
    n_envs = config.get("n_envs", 1)  # Number of parallel environments
    
    # Verify ROM availability
    try:
        test_env = gym.make(env_id)
        test_env.close()
    except gym.error.Error as e:
        raise RuntimeError(
            f"Error creating environment: {e}\n"
            "This might be because the Atari ROM is not installed.\n"
            "Please run: pip install 'AutoROM[accept-rom-license]' and try again."
        )
    
    # Rest of the configuration
    total_timesteps = config.get("total_timesteps", 1000000)
    learning_rate = config.get("learning_rate", 0.00025)
    buffer_size = config.get("buffer_size", 250000)
    learning_starts = config.get("learning_starts", 50000)
    batch_size = config.get("batch_size", 256)
    exploration_fraction = config.get("exploration_fraction", 0.2)
    exploration_final_eps = config.get("exploration_final_eps", 0.01)
    train_log_interval = config.get("train_log_interval", 100)
    
    # Network optimization parameters
    gamma = config.get("gamma", 0.99)
    target_update_interval = config.get("target_update_interval", 2000)
    gradient_steps = config.get("gradient_steps", 2)
    train_freq = config.get("train_freq", 4)
    frame_stack = config.get("frame_stack", 4)
    
    # Preprocessing settings
    scale_rewards = config.get("scale_rewards", True)
    normalize_frames = config.get("normalize_frames", True)
    terminal_on_life_loss = config.get("terminal_on_life_loss", True)
    
    # Game specific settings
    difficulty = config.get("difficulty", 0)
    mode = config.get("mode", 0)
    
    # Workshop settings
    viz_interval = config.get("viz_interval", 25000)
    video_interval = config.get("video_interval", 100000)
    video_length = config.get("video_length", 400)
    checkpoint_interval = config.get("checkpoint_interval", 100000)

    # Initialize wandb if requested
    if use_wandb:
        import wandb
        wandb.init(project="ai-arcade-space-invaders", config=config)

    # Create directories and ensure clean tensorboard logs
    for dir_name in ["logs", "models", "tensorboard", "videos", "checkpoints"]:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Create a unique run directory for tensorboard
    current_time = time.strftime("%Y%m%d-%H%M%S")
    tensorboard_log_dir = f"./tensorboard/DQN_space_invaders_{current_time}"
    Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)

    # Create vectorized environment
    env = DummyVecEnv([
        make_env(
            rank=i,  # Use rank for each environment
            seed=i
        ) for i in range(n_envs)  # Create n_envs parallel environments
    ])
    env = VecFrameStack(env, n_stack=frame_stack)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Training device: {device.upper()}")
    print(f"Demo mode: {'Enabled' if demo_mode else 'Disabled'}")
    print(f"Video recording: {'Enabled' if record_video else 'Disabled'}")
    print("\nTraining parameters:")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer size: {buffer_size}")
    print(f"Learning starts: {learning_starts}")
    print(f"Exploration fraction: {exploration_fraction}")
    print(f"Target update interval: {target_update_interval}")
    print(f"Frame stack: {frame_stack}")
    print(f"Reward scaling: {'Enabled' if scale_rewards else 'Disabled'}")
    print(f"Frame normalization: {'Enabled' if normalize_frames else 'Disabled'}")
    print(f"Difficulty: {difficulty}")
    print(f"Mode: {mode}")

    # Initialize DQN model with optimized parameters
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        gamma=gamma,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        device=device,
        policy_kwargs={
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {"eps": 1e-5},
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [512, 512]
        }
    )

    # Set up callbacks
    callbacks = [
        WorkshopCallback(
            viz_interval=viz_interval,
            video_interval=video_interval,
            video_length=video_length,
            checkpoint_interval=checkpoint_interval,
            demo_mode=demo_mode,
            record_video=record_video
        ),
        CheckpointCallback(
            save_freq=checkpoint_interval,
            save_path="./checkpoints/",
            name_prefix="space_invaders_model"
        )
    ]

    try:
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # Save the final model
        model.save(f"models/space_invaders_dqn_{total_timesteps}_steps.zip")
        print(f"\nTraining completed! Model saved to models/space_invaders_dqn_{total_timesteps}_steps.zip")

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")
        model.save(f"models/space_invaders_dqn_interrupted_{model.num_timesteps}_steps.zip")
        print(f"Model saved to models/space_invaders_dqn_interrupted_{model.num_timesteps}_steps.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Space Invaders")
    parser.add_argument("--config", type=str, default="configs/space_invaders_sb3_config.yaml",
                      help="Path to the config file")
    parser.add_argument("--render", action="store_true",
                      help="Render the environment during training")
    parser.add_argument("--wandb", action="store_true",
                      help="Use Weights & Biases for logging")
    parser.add_argument("--demo", action="store_true",
                      help="Run in demo mode (renders environment)")
    parser.add_argument("--record", action="store_true",
                      help="Record videos of training milestones")
    
    args = parser.parse_args()
    train(args.config, args.render, args.wandb, args.demo, args.record) 