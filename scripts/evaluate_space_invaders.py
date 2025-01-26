import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
)
from stable_baselines3.common.monitor import Monitor
from pathlib import Path

def make_atari_env(env_id, rank=0, render_mode=None, difficulty=0, mode=0):
    def _init():
        env = gym.make(env_id, render_mode=render_mode, frameskip=1,
                      difficulty=difficulty, mode=mode)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env = Monitor(env)
        return env
    return _init

def evaluate(model_path, episodes=10, render=True, difficulty=0, mode=0):
    # Load the trained model
    env = DummyVecEnv([
        make_atari_env(
            "ALE/SpaceInvaders-v5",
            rank=0,
            render_mode="human" if render else None,
            difficulty=difficulty,
            mode=mode
        )
    ])
    
    # Load the model with the correct environment
    model = DQN.load(model_path, env=env)
    
    # Run evaluation
    rewards, lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=episodes,
        render=render,
        deterministic=True,
        return_episode_rewards=True
    )
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Number of episodes: {episodes}")
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print(f"Std reward: {np.std(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    print(f"Mean episode length: {np.mean(lengths):.2f}")
    
    # Return results for potential further analysis
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_length': np.mean(lengths),
        'rewards': rewards,
        'lengths': lengths
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Space Invaders agent")
    parser.add_argument("--model", type=str, required=True,
                      help="Path to the trained model file")
    parser.add_argument("--episodes", type=int, default=10,
                      help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                      help="Disable rendering")
    parser.add_argument("--difficulty", type=int, default=0,
                      help="Game difficulty (0 or 1)")
    parser.add_argument("--mode", type=int, default=0,
                      help="Game mode (0-15)")
    
    args = parser.parse_args()
    evaluate(args.model, args.episodes, not args.no_render, args.difficulty, args.mode) 