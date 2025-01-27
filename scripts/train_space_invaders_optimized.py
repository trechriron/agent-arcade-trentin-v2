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
from tqdm.auto import tqdm
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from Nature paper, adapted for Space Invaders.
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

class OptimizedRewardScaler(gym.Wrapper):
    """Optimized reward structure for Space Invaders"""
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
        
        # Score-based reward
        score_diff = info.get('score', 0) - self._current_score
        self._current_score = info.get('score', 0)
        
        # Life tracking
        lives = info.get('lives', 3)
        life_loss = self._current_lives - lives
        self._current_lives = lives
        
        # Simplified reward shaping
        shaped_reward = 0.0
        
        # Score reward (main component)
        if score_diff > 0:
            shaped_reward += score_diff * 0.1
            
        # Life loss penalty (reduced)
        if life_loss > 0:
            shaped_reward -= 0.5
            
        # Level completion bonus
        if terminated and lives > 0:
            shaped_reward += 10.0
                
        return obs, shaped_reward, terminated, truncated, info

def make_env(rank, seed=0):
    def _init():
        env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
        
        # Standard Atari wrappers
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        # Optimized reward structure
        env = OptimizedRewardScaler(env)
        
        # Observation preprocessing
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
        
        video_env = DummyVecEnv([make_env(rank=0, seed=42)])
        video_env = VecFrameStack(video_env, n_stack=4)
        
        video_env = VecVideoRecorder(
            video_env,
            video_path,
            record_video_trigger=lambda x: x < self.video_length,
            video_length=self.video_length
        )
        
        obs = video_env.reset()
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, _ = video_env.step(action)
            if dones.any():
                obs = video_env.reset()
        
        video_env.close()

class ProgressBarCallback(BaseCallback):
    """
    Display progress bar during training.
    """
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)
        return True
    
    def _on_training_end(self):
        self.pbar.close()

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        torch.backends.mps.enable_mps = True
        return "mps"
    return "cpu"

def train(config_path, render_training=False, record_video=False):
    # Load and validate config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    device = get_device()
    print(f"\nUsing device: {device.upper()}")
    
    if device == "mps":
        torch.set_num_threads(6)
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error reading config file: {e}")

    # Environment setup
    env_id = config.get("env", "ALE/SpaceInvaders-v5")
    n_envs = config.get("n_envs", 8)
    
    try:
        test_env = gym.make(env_id)
        test_env.close()
    except gym.error.Error as e:
        raise RuntimeError(f"Error creating environment: {e}\nPlease install ROMs: pip install 'AutoROM[accept-rom-license]'")
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)

    # Set up logging
    log_path = f"./tensorboard/DQN_space_invaders_optimized_{int(time.time())}"
    logger = configure(log_path, ["tensorboard", "stdout"])
    print(f"TensorBoard logging directory: {log_path}")
    
    # Initialize model with Nature CNN
    policy_kwargs = {
        "features_extractor_class": NatureCNN,
        "features_extractor_kwargs": dict(features_dim=512),
        "normalize_images": True
    }
    
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=config.get("learning_rate", 0.00025),
        buffer_size=config.get("buffer_size", 200000),
        learning_starts=config.get("learning_starts", 50000),
        batch_size=config.get("batch_size", 64),
        tau=config.get("tau", 1.0),
        gamma=config.get("gamma", 0.99),
        train_freq=config.get("train_freq", 4),
        gradient_steps=config.get("gradient_steps", 1),
        target_update_interval=config.get("target_update_interval", 10000),
        exploration_fraction=config.get("exploration_fraction", 0.1),
        exploration_initial_eps=config.get("exploration_initial_eps", 1.0),
        exploration_final_eps=config.get("exploration_final_eps", 0.05),
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log=log_path
    )
    
    # Set up callbacks
    total_timesteps = config.get("total_timesteps", 2000000)
    callbacks = [
        ProgressBarCallback(total_timesteps),
        WorkshopCallback(
            viz_interval=config.get("viz_interval", 25000),
            video_interval=config.get("video_interval", 25000),
            video_length=config.get("video_length", 400),
            checkpoint_interval=config.get("checkpoint_interval", 25000),
            record_video=record_video
        ),
        CheckpointCallback(
            save_freq=config.get("checkpoint_freq", 50000),
            save_path=config.get("save_path", "models/space_invaders_optimized"),
            name_prefix="space_invaders_model"
        )
    ]
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=config.get("train_log_interval", 100),
            tb_log_name="DQN"
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        model.save(f"{config.get('save_path', 'models/space_invaders_optimized')}/final_model")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--render", action="store_true", help="Enable environment rendering")
    parser.add_argument("--record", action="store_true", help="Enable video recording")
    args = parser.parse_args()
    
    train(args.config, args.render, args.record) 