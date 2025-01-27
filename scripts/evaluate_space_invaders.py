import yaml
import gymnasium as gym
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from train_space_invaders_optimized import make_env, NatureCNN

def evaluate(model_path, config_path, n_eval_episodes=100, render=True, record_video=True):
    """
    Evaluate a trained Space Invaders model with real-time visualization and competition metrics.
    """
    print("\n🎮 Space Invaders Agent Evaluation 🚀")
    print("======================================")
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create eval environment with rendering
    eval_env = gym.make("ALE/SpaceInvaders-v5", render_mode="human" if render else "rgb_array")
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Load model
    print("\n📱 Loading model...")
    custom_objects = {
        "policy_kwargs": {
            "features_extractor_class": NatureCNN,
            "features_extractor_kwargs": dict(features_dim=512),
            "normalize_images": True
        }
    }
    model = DQN.load(model_path, env=eval_env, custom_objects=custom_objects)
    
    # Competition metrics
    scores = []
    max_score = float('-inf')
    best_episodes = []
    
    # Set up video recording for highlights
    if record_video:
        video_folder = "videos/evaluation"
        Path(video_folder).mkdir(parents=True, exist_ok=True)
        
        # Create a separate environment for recording
        record_env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
        record_env = DummyVecEnv([lambda: record_env])
        record_env = VecVideoRecorder(
            record_env,
            video_folder=video_folder,
            record_video_trigger=lambda x: x % 25 == 0,  # Record every 25th episode
            video_length=3000,
            name_prefix="space_invaders_eval"
        )
    
    # Run evaluation with progress bar
    print("\n🎯 Running evaluation...")
    print("💡 Watch the agent play in real-time!")
    pbar = tqdm(total=n_eval_episodes, desc="Evaluating episodes", position=0, leave=True)
    
    try:
        for episode in range(n_eval_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            episode_frames = []
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
                
                if record_video:
                    episode_frames.append(obs[0].copy())
                
                if done[0]:
                    scores.append(episode_reward)
                    if episode_reward > max_score:
                        max_score = episode_reward
                        if record_video:
                            # Record the best episode
                            obs_record = record_env.reset()
                            for action_frame in episode_frames:
                                record_env.step(action_frame)
                            best_episodes = episode_frames
                    break
            
            pbar.update(1)
            if episode % 10 == 0:
                pbar.set_postfix({
                    'Max Score': f'{max(scores):.1f}',
                    'Avg Score': f'{np.mean(scores):.1f}',
                    'Current': f'{episode_reward:.1f}'
                })
        
        pbar.close()
        
        # Print competition results
        print("\n🏆 Competition Results:")
        print(f"{'='*40}")
        print(f"🥇 Max Score: {max(scores):.1f}")
        print(f"📊 Average Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
        print(f"📈 Score Distribution:")
        print(f"  - Top 10%: {np.percentile(scores, 90):.1f}")
        print(f"  - Median: {np.median(scores):.1f}")
        print(f"  - Bottom 10%: {np.percentile(scores, 10):.1f}")
        print(f"🎯 Consistency: {len([s for s in scores if s > np.mean(scores)])/len(scores)*100:.1f}% above average")
        
        # Save highlight reel
        if record_video and best_episodes:
            highlight_path = f"{video_folder}/highlight_reel_score_{max_score:.1f}.mp4"
            record_env.env_method("save_video", best_episodes, highlight_path)
            print(f"\n🎥 Highlight reel saved: {highlight_path}")
            print(f"   Best performance score: {max_score:.1f}")
    
    finally:
        eval_env.close()
        if record_video:
            record_env.close()
    
    return np.mean(scores), np.std(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Space Invaders agent")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable real-time rendering")
    parser.add_argument("--record", action="store_true", help="Record evaluation videos")
    args = parser.parse_args()
    
    evaluate(args.model, args.config, args.episodes, not args.no_render, args.record) 