import sys
import gymnasium as gym
import torch
import os
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import numpy as np

def evaluate(model_path, episodes=5, render=True):
    # Enable MPS if needed
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    torch.backends.mps.enable_mps = True

    # Validate model path
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create environment with rendering if specified
    def make_env():
        render_mode = "human" if render else None
        env = gym.make("ALE/Pong-v5", render_mode=render_mode, frameskip=4)
        return AtariWrapper(env, terminal_on_life_loss=True)

    # Create vectorized environment
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Load the trained model
    try:
        model = DQN.load(model_path, env=env, device="mps")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return

    episode_rewards = []
    try:
        for ep in range(episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                done = dones[0]
                total_reward += rewards[0]  # Get scalar reward from vectorized env
                steps += 1

            episode_rewards.append(total_reward)
            print(f"Episode {ep + 1}: Steps = {steps}, Reward = {total_reward}")

        # Print summary statistics
        print("\nEvaluation Summary:")
        print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Max Reward: {np.max(episode_rewards):.2f}")
        print(f"Min Reward: {np.min(episode_rewards):.2f}")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    
    finally:
        env.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_pong.py <model_path> [num_episodes]")
        sys.exit(1)

    model_path = sys.argv[1]
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    evaluate(model_path, episodes) 