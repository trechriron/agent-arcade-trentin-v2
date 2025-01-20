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
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomCNN(BaseFeaturesExtractor):
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

class PongRewardScaler(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._running_mean = 0
        self._running_std = 1
        self._alpha = 0.99  # Exponential moving average factor

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update running statistics
        self._running_mean = self._alpha * self._running_mean + (1 - self._alpha) * reward
        self._running_std = self._alpha * self._running_std + (1 - self._alpha) * abs(reward - self._running_mean)
        
        # Clip and scale reward
        scaled_reward = np.clip(reward / (self._running_std + 1e-8), -1, 1)
        
        return obs, scaled_reward, terminated, truncated, info

def make_env(env_id, rank, render_mode=None, scale_rewards=True, normalize_frames=True, terminal_on_life_loss=True):
    def _init():
        # Create environment with basic Atari wrapper
        env = gym.make(env_id, render_mode=render_mode, frameskip=4)
        env = AtariWrapper(env, terminal_on_life_loss=terminal_on_life_loss)
        
        # Add observation normalization if requested
        if normalize_frames:
            env = gym.wrappers.TransformObservation(
                env, lambda obs: obs / 255.0
            )
        
        # Add reward scaling if requested
        if scale_rewards:
            env = PongRewardScaler(env)
            
        return env
    return _init

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
        
        # Create a separate environment for recording
        def make_env():
            env = gym.make(self.training_env.envs[0].spec.id, render_mode="rgb_array")
            env = AtariWrapper(env, terminal_on_life_loss=True)
            return env
        
        video_env = VecVideoRecorder(
            DummyVecEnv([make_env]),
            video_path,
            record_video_trigger=lambda x: x < self.video_length,
            video_length=self.video_length
        )
        
        # Record episode using current policy
        obs = video_env.reset()[0]
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = video_env.step(action)
            if done.any():
                obs = video_env.reset()[0]
        
        video_env.close()

def train(config_path, render_training=False, use_wandb=False, demo_mode=False, record_video=False):
    # Enable MPS fallback and optimization
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    torch.backends.mps.enable_mps = True
    
    # Set torch to use high performance mode
    if torch.backends.mps.is_available():
        torch.set_num_threads(6)  # Optimize thread usage for M1/M2

    # Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration
    algo_name = config.get("algo", "DQN")
    env_id = config.get("env", "ALE/Pong-v5")
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
    
    # Workshop settings
    viz_interval = config.get("viz_interval", 25000)
    video_interval = config.get("video_interval", 100000)
    video_length = config.get("video_length", 400)
    checkpoint_interval = config.get("checkpoint_interval", 100000)

    # Initialize wandb if requested
    if use_wandb:
        import wandb
        wandb.init(project="ai-arcade-pong", config=config)

    # Create directories and ensure clean tensorboard logs
    for dir_name in ["logs", "models", "tensorboard", "videos", "checkpoints"]:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Create a unique run directory for tensorboard
    current_time = time.strftime("%Y%m%d-%H%M%S")
    tensorboard_log_dir = f"./tensorboard/DQN_pong_{current_time}"
    Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)

    # Create vectorized environment
    env = DummyVecEnv([
        make_env(
            env_id, 
            rank=0,
            render_mode="human" if (render_training or demo_mode) else None,
            scale_rewards=scale_rewards,
            normalize_frames=normalize_frames,
            terminal_on_life_loss=terminal_on_life_loss
        )
    ])
    env = VecFrameStack(env, n_stack=frame_stack)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Training device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
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
        tensorboard_log=tensorboard_log_dir,  # Use the unique directory
        device="mps",
        policy_kwargs={
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {"eps": 1e-5},
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [512, 512]
        }
    )

    # Configure logging with SB3's monitor
    new_logger = configure(
        f"{tensorboard_log_dir}/logs",  # Put logs in tensorboard directory
        ["stdout", "csv", "tensorboard"]
    )
    model.set_logger(new_logger)

    print(f"\nTensorBoard logging directory: {tensorboard_log_dir}")
    print("To visualize training, run:")
    print(f"tensorboard --logdir {tensorboard_log_dir}")

    # Setup callbacks
    workshop_callback = WorkshopCallback(
        viz_interval=viz_interval,
        video_interval=video_interval,
        video_length=video_length,
        checkpoint_interval=checkpoint_interval,
        demo_mode=demo_mode,
        record_video=record_video
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path="./checkpoints/",
        name_prefix="pong_model"
    )

    try:
        # Train the model with more frequent logging
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,  # More frequent logging
            progress_bar=True,
            callback=[workshop_callback, checkpoint_callback]
        )
        
        # Save final model
        model_path = f"models/pong_dqn_{total_timesteps}_steps.zip"
        model.save(model_path)
        print(f"Training complete. Model saved as {model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        model.save("models/pong_dqn_interrupted.zip")
        print("Model saved as models/pong_dqn_interrupted.zip")
    
    finally:
        env.close()
        if use_wandb:
            wandb.finish()

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