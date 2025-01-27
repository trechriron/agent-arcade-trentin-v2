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
    ClipRewardEnv
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # Should be 4 for frame stack
        
        # Nature DQN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float() / 255.0
            n_flatten = self.cnn(sample).shape[1]
            
        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations):
        observations = observations.float() / 255.0
        return self.linear(self.cnn(observations))

def make_env(env_id, rank, seed=None, video_folder=None, video_length=400):
    """Create Pong-specific environment with proper wrappers."""
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Pong-specific wrappers
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
            
        # No reward clipping or scaling for Pong
        
        # Ensure proper observation processing
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        
        if video_folder is not None:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda x: x < video_length
            )
        
        if seed is not None:
            env.action_space.seed(seed + rank)
            
        return env
    return _init()

class WorkshopCallback(BaseCallback):
    def __init__(self, viz_interval, video_interval, video_length, checkpoint_interval, demo_mode=False, record_video=False):
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
        if len(self.locals.get('dones', [])) > 0 and self.locals.get('dones')[0]:
            reward = sum(self.locals.get('rewards', [0]))
            self.episode_rewards.append(reward)
            
            # Record video for significant improvements
            if self.record_video and reward > self.best_reward:
                self.best_reward = reward
                self._record_video(f"milestone_reward_{reward}")
        
        return True
    
    def _record_video(self, milestone):
        """Record a video of the agent's performance."""
        video_path = os.path.join("videos", str(self.n_calls), f"{milestone}")
        os.makedirs(video_path, exist_ok=True)

        # Create a single environment for recording
        env = make_env(
            env_id=self.training_env.envs[0].spec.id,
            rank=0,
            seed=None,
            video_folder=video_path,
            video_length=self.video_length
        )
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=4)
        
        try:
            # Handle both old and new Gymnasium reset API
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
            
            episode_reward = 0
            for _ in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                step_result = env.step(action)
                
                # Handle both old and new Gymnasium step API
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, _ = step_result
                    
                episode_reward += reward[0]
                if done[0]:
                    break
            
            return episode_reward
        finally:
            env.close()

def train(config_path, render_training=False, use_wandb=False, demo_mode=False, record_video=False):
    # Load and validate config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML config: {e}")

    # Extract configuration with Pong-specific defaults
    algo_name = config.get("algo", "DQN")
    env_id = config.get("env", "ALE/Pong-v5")
    total_timesteps = config.get("total_timesteps", 2000000)  # Longer training
    learning_rate = config.get("learning_rate", 0.0001)  # More stable learning
    buffer_size = config.get("buffer_size", 100000)  # Smaller buffer for recent experiences
    learning_starts = config.get("learning_starts", 50000)  # More initial exploration
    batch_size = config.get("batch_size", 32)  # Smaller batches
    train_freq = config.get("train_freq", 4)
    
    # Create vectorized environment
    n_envs = config.get("n_envs", 4)
    env_fns = [lambda i=i: make_env(env_id, i, seed=42+i) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=4)

    # Initialize model with Pong-specific parameters
    policy_kwargs = {
        "net_arch": [64, 64],  # Simpler architecture
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "normalize_images": True
    }

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=1.0,  # Full target net update
        gamma=0.99,
        train_freq=train_freq,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=config.get("exploration_fraction", 0.3),
        exploration_initial_eps=1.0,
        exploration_final_eps=config.get("exploration_final_eps", 0.05),
        tensorboard_log=f"./tensorboard/DQN_pong_{time.strftime('%Y%m%d-%H%M%S')}",
        policy_kwargs=policy_kwargs,
        device="auto"
    )

    # Configure logging with SB3's monitor
    new_logger = configure(
        f"./tensorboard/DQN_pong_{time.strftime('%Y%m%d-%H%M%S')}/logs",  # Put logs in tensorboard directory
        ["stdout", "csv", "tensorboard"]
    )
    model.set_logger(new_logger)

    print(f"\nTensorBoard logging directory: ./tensorboard/DQN_pong_{time.strftime('%Y%m%d-%H%M%S')}")
    print("To visualize training, run:")
    print(f"tensorboard --logdir ./tensorboard/DQN_pong_{time.strftime('%Y%m%d-%H%M%S')}")

    # Create callback
    callbacks = [
        WorkshopCallback(
            viz_interval=config.get("viz_interval", 25000),
            video_interval=config.get("video_interval", 100000),
            video_length=config.get("video_length", 400),
            checkpoint_interval=config.get("checkpoint_interval", 100000),
            demo_mode=demo_mode,
            record_video=record_video
        ),
        CheckpointCallback(
            save_freq=config.get("checkpoint_interval", 100000),
            save_path="checkpoints",
            name_prefix="pong_model"
        )
    ]

    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=config.get("train_log_interval", 100),
            tb_log_name="DQN",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        # Save the final model
        model.save("models/pong_final")
        env.close()

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN agent on Pong')
    parser.add_argument('--config', type=str, default='configs/pong_sb3_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--render', action='store_true',
                      help='Enable rendering during training')
    parser.add_argument('--wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    parser.add_argument('--demo', action='store_true',
                      help='Enable workshop demonstration mode')
    parser.add_argument('--record-video', action='store_true',
                      help='Enable video recording of milestones')
    args = parser.parse_args()
    
    train(args.config, render_training=args.render, use_wandb=args.wandb, 
          demo_mode=args.demo, record_video=args.record_video) 