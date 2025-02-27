"""Evaluation pipeline for Agent Arcade."""
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecEnvWrapper, VecEnv
from loguru import logger
import numpy as np
import json
import hmac
import hashlib
import secrets

from .leaderboard import LeaderboardManager
from .wallet import NEARWallet

# Custom wrapper to ensure channel-first format for observations
class ChannelFirstWrapper(gym.ObservationWrapper):
    """Ensures observations are in channel-first format (C, H, W) and properly normalized for PyTorch CNN models."""
    
    def __init__(self, env, model_obs_space=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.needs_transpose = False
        self.needs_scaling = False
        self.model_obs_space = model_obs_space
        
        # Check if we need to transpose
        if len(obs_shape) == 3 and obs_shape[-1] in [1, 3]:  # Channel-last format (H, W, C)
            self.needs_transpose = True
            transposed_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        else:
            transposed_shape = obs_shape
            
        # Check if we need to scale values (convert from uint8 to float32)
        if self.observation_space.dtype == np.uint8:
            self.needs_scaling = True
            dtype = np.float32
            low, high = 0.0, 1.0
        else:
            dtype = self.observation_space.dtype
            low = self.observation_space.low.min()
            high = self.observation_space.high.max()
        
        # Update observation space to reflect the transformed shape and range
        if self.needs_transpose or self.needs_scaling:
            self.observation_space = gym.spaces.Box(
                low=low,
                high=high,
                shape=transposed_shape, 
                dtype=dtype
            )
            
        # If model observation space is provided, ensure we match it
        if model_obs_space is not None:
            if model_obs_space.dtype != self.observation_space.dtype:
                self.needs_scaling = True
                
            # Use model's low/high values for proper normalization
            self.target_low = model_obs_space.low.min()
            self.target_high = model_obs_space.high.max()
        else:
            self.target_low = 0.0
            self.target_high = 1.0
    
    def observation(self, obs):
        # Apply transformations as needed
        if self.needs_transpose and len(obs.shape) == 3:
            obs = np.transpose(obs, (2, 0, 1))
            
        if self.needs_scaling:
            # Scale from [0, 255] to [0.0, 1.0] or whatever the model expects
            if obs.dtype == np.uint8:
                obs = obs.astype(np.float32) / 255.0
                
                # Rescale to target range if not [0.0, 1.0]
                if self.target_low != 0.0 or self.target_high != 1.0:
                    obs = obs * (self.target_high - self.target_low) + self.target_low
                    
        return obs

# Custom wrapper to skip grayscale conversion if already grayscale
class SkipGrayscaleConversionWrapper(gym.ObservationWrapper):
    """Wrapper to avoid redundant grayscale conversions in Atari environments."""
    
    def __init__(self, env):
        super().__init__(env)
        # No need to change observation space
        
    def reset(self, **kwargs):
        # Patch reset to avoid cv2.cvtColor for grayscale images
        obs_tuple = self.env.reset(**kwargs)
        if isinstance(obs_tuple, tuple):
            obs, info = obs_tuple
            return obs, info
        return obs_tuple
    
    def step(self, action):
        # Patch step to avoid cv2.cvtColor for grayscale images
        result = self.env.step(action)
        
        # Handle both gym API versions
        if len(result) == 4:  # Old API: obs, reward, done, info
            obs, reward, done, info = result
            return obs, reward, done, info
        else:  # New API: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated, truncated, info
            
    def observation(self, observation):
        # Don't modify the observation
        return observation

class GameSpecificConfig:
    """Game-specific configuration for evaluation."""
    
    def __init__(
        self,
        game_id: str,
        score_range: Tuple[float, float],
        success_threshold: float,
        default_frame_stack: int = 4,
        observation_type: str = "grayscale",
        action_space: Optional[List[int]] = None,
        difficulty: int = 0,
        mode: int = 0
    ):
        """Initialize game-specific configuration.
        
        Args:
            game_id: ALE game identifier
            score_range: (min_score, max_score) for the game
            success_threshold: Score threshold for considering an episode successful
            default_frame_stack: Default number of frames to stack
            observation_type: Type of observation ("rgb", "grayscale", "ram")
            action_space: List of valid actions for the game
            difficulty: Game difficulty level
            mode: Game mode
        """
        self.game_id = game_id
        self.score_range = score_range
        self.success_threshold = success_threshold
        self.default_frame_stack = default_frame_stack
        self.observation_type = observation_type
        self.action_space = action_space
        self.difficulty = difficulty
        self.mode = mode
        
        # Game-specific staking thresholds
        self.staking_thresholds = {
            "min_confidence": 0.7,
            "min_success_rate": 0.6,
            "min_stability": 0.5,
            "target_score_percentile": 75,
            "risk_levels": {
                "low": 0.8,
                "medium": 0.5,
                "high": 0.3
            }
        }

class EvaluationConfig:
    """Configuration for model evaluation."""
    
    def __init__(
        self,
        game_id: str,
        n_eval_episodes: int = 100,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        frame_stack: Optional[int] = None,  # Will use game-specific default if None
        obs_type: Optional[str] = None,  # Will use game-specific default if None
        frameskip: Optional[Union[int, Tuple[int, int]]] = None,
        repeat_action_probability: float = 0.25,  # ALE v5 default
        full_action_space: bool = False,
        mode: str = "staking",  # "staking", "training", or "competition"
        **kwargs
    ):
        """Initialize evaluation configuration."""
        self.game_id = game_id
        self.game_config = GAME_CONFIGS.get(game_id)
        if not self.game_config:
            raise ValueError(f"No configuration found for game: {game_id}")
        
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.verbose = verbose
        self.frame_stack = frame_stack or self.game_config.default_frame_stack
        self.obs_type = obs_type or self.game_config.observation_type
        self.frameskip = frameskip
        self.repeat_action_probability = repeat_action_probability
        self.full_action_space = full_action_space
        self.mode = mode
        self.additional_config = kwargs
        
        # Use game-specific staking thresholds
        self.staking_config = {
            "min_confidence": self.game_config.staking_thresholds["min_confidence"],
            "min_success_rate": self.game_config.staking_thresholds["min_success_rate"],
            "min_stability": self.game_config.staking_thresholds["min_stability"],
            "target_score_percentile": self.game_config.staking_thresholds["target_score_percentile"],
            "risk_levels": self.game_config.staking_thresholds["risk_levels"].copy(),
            "stability_window": 10  # Window size for stability calculation
        }
        
        # Resource management settings
        self.resource_config = {
            "max_memory_gb": 16,  # Maximum memory usage in GB
            "target_fps": 250,  # Target evaluation FPS
            "batch_size": 1,  # Evaluation batch size
            "use_gpu": True,  # Whether to use GPU for evaluation
        }
        
        # Observation preprocessing
        self.preprocessing_config = {
            "normalize": True,  # Normalize observations
            "scale_range": (0, 1),  # Observation scaling range
            "resize_shape": (84, 84),  # Standard ALE preprocessing
            "grayscale": self.obs_type == "grayscale",
        }
    
    def get_env_kwargs(self) -> Dict[str, Any]:
        """Get environment keyword arguments for ALE."""
        return {
            "obs_type": self.obs_type,
            "frameskip": self.frameskip,
            "repeat_action_probability": self.repeat_action_probability,
            "full_action_space": self.full_action_space,
            "difficulty_ramping": False,  # Consistent difficulty for evaluation
            "mode": self.game_config.mode,
            "difficulty": self.game_config.difficulty
        }
    
    def validate_for_staking(self) -> Tuple[bool, str]:
        """Validate configuration for staking evaluation."""
        if self.n_eval_episodes < 50:  # Minimum episodes for reliable evaluation
            return False, "Need at least 50 episodes for reliable staking evaluation"
        if self.mode != "staking":
            return False, "Configuration not in staking mode"
        return True, "Configuration valid for staking"

class EvaluationResult:
    """Results from model evaluation with staking-focused metrics."""
    
    def __init__(
        self,
        mean_reward: float,
        std_reward: float,
        n_episodes: int,
        success_rate: float,
        episode_lengths: List[int],
        episode_rewards: List[float],
        metadata: Dict[str, Any],
        game_config: GameSpecificConfig,
        staking_metrics: Optional[Dict[str, Any]] = None,
        verification_token: Optional[Dict[str, Any]] = None
    ):
        """Initialize evaluation result with staking-focused metrics.
        
        Args:
            mean_reward: Mean reward across episodes
            std_reward: Standard deviation of rewards
            n_episodes: Number of episodes evaluated
            success_rate: Success rate (if applicable)
            episode_lengths: List of episode lengths
            episode_rewards: List of episode rewards
            metadata: Additional metadata about the evaluation
            game_config: Game-specific configuration
            staking_metrics: Game-specific staking metrics including:
                - confidence_score: How consistent is the performance (0-1)
                - risk_level: Assessed risk ("low", "medium", "high")
                - recommended_stake: Suggested stake based on performance
                - expected_multiplier: Predicted reward multiplier
                - stability_score: Measure of performance stability (0-1)
                - min_target_score: Minimum score for successful stake
                - optimal_target_score: Recommended target score for staking
            verification_token: Verification token for data integrity
        """
        self.mean_reward = mean_reward
        self.std_reward = std_reward
        self.n_episodes = n_episodes
        self.success_rate = success_rate
        self.episode_lengths = episode_lengths
        self.episode_rewards = episode_rewards
        self.metadata = metadata
        self.game_config = game_config
        
        # Initialize staking metrics with defaults if not provided
        self.staking_metrics = staking_metrics or {
            "confidence_score": self._calculate_confidence_score(),
            "risk_level": self._assess_risk_level(),
            "recommended_stake": self._calculate_recommended_stake(),
            "expected_multiplier": self._calculate_expected_multiplier(),
            "stability_score": self._calculate_stability_score(),
            "min_target_score": self._calculate_min_target_score(),
            "optimal_target_score": self._calculate_optimal_target_score()
        }
        
        self.verification_token = verification_token
    
    def _calculate_confidence_score(self) -> float:
        """Calculate confidence score based on performance consistency and absolute performance."""
        if not self.episode_rewards:
            return 0.0
        
        reward_std = np.std(self.episode_rewards)
        reward_mean = np.mean(self.episode_rewards)
        
        # Get game-specific score range
        min_score, max_score = self.game_config.score_range
        score_range = max_score - min_score
        
        # Calculate consistency component (0 to 1)
        cv = reward_std / (score_range + 1e-8)  # Use score range instead of mean
        consistency = max(0.0, min(1.0, 1.0 - cv))
        
        # Calculate performance component (0 to 1)
        # Map mean score to [0, 1] range relative to game's score range
        performance = (reward_mean - min_score) / (score_range + 1e-8)
        performance = max(0.0, min(1.0, performance))
        
        # Combine consistency and performance
        # High confidence requires both consistency and good performance
        return consistency * performance
    
    def _assess_risk_level(self) -> str:
        """Assess risk level based on performance metrics."""
        confidence = self._calculate_confidence_score()
        thresholds = self.game_config.staking_thresholds["risk_levels"]
        min_success = self.game_config.staking_thresholds["min_success_rate"]
        
        if confidence >= thresholds["low"] and self.success_rate >= min_success:
            return "low"
        elif confidence >= thresholds["medium"] and self.success_rate >= min_success * 0.5:
            return "medium"
        else:
            return "high"
    
    def _calculate_recommended_stake(self) -> float:
        """Calculate recommended stake based on performance metrics."""
        confidence = self._calculate_confidence_score()
        base_stake = 5.0  # Base stake in NEAR
        
        # No stake if performance is too poor
        if confidence < 0.3 or self.success_rate < self.game_config.staking_thresholds["min_success_rate"] * 0.5:
            return 0.0
        
        # Calculate stake multiplier
        stake_multiplier = min(confidence * self.success_rate, 2.0)  # Cap at 2x
        return base_stake * stake_multiplier
    
    def _calculate_expected_multiplier(self) -> float:
        """Calculate expected reward multiplier based on performance."""
        min_success = self.game_config.staking_thresholds["min_success_rate"]
        
        if self.success_rate >= min_success:
            return 2.0
        elif self.success_rate >= min_success * 0.75:
            return 1.5
        else:
            return 1.0
    
    def _calculate_stability_score(self) -> float:
        """Calculate stability score based on reward consistency and performance level."""
        if not self.episode_rewards or len(self.episode_rewards) < 2:
            return 0.0
        
        # Get game-specific score range
        min_score, max_score = self.game_config.score_range
        score_range = max_score - min_score
        
        # Calculate trend stability
        window = min(10, len(self.episode_rewards))  # Use fixed window size
        rolling_means = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
        
        # Calculate stability component
        stability = 1.0 - np.std(rolling_means) / (score_range + 1e-8)
        stability = max(0.0, min(1.0, stability))
        
        # Calculate performance level component
        mean_score = np.mean(self.episode_rewards)
        performance = (mean_score - min_score) / (score_range + 1e-8)
        performance = max(0.0, min(1.0, performance))
        
        # Combine stability and performance
        return stability * performance
    
    def _calculate_min_target_score(self) -> float:
        """Calculate minimum viable target score for staking."""
        if not self.episode_rewards:
            return 0.0
        
        # Use lower percentile of achieved scores
        return float(np.percentile(self.episode_rewards, 25))
    
    def _calculate_optimal_target_score(self) -> float:
        """Calculate optimal target score for maximum expected return."""
        if not self.episode_rewards:
            return 0.0
        
        # Use a score that's consistently achievable but challenging
        # 75th percentile balances ambition with achievability
        return float(np.percentile(self.episode_rewards, 75))
    
    def get_staking_recommendation(self) -> Dict[str, Any]:
        """Get comprehensive staking recommendation."""
        confidence = self.staking_metrics["confidence_score"]
        min_success = self.game_config.staking_thresholds["min_success_rate"]
        
        if confidence >= 0.8 and self.success_rate >= min_success:
            action = "stake_high"
            reasoning = (f"High confidence ({confidence:.2f}) and strong success rate ({self.success_rate:.2%}) "
                       "indicate reliable performance.")
        elif confidence >= 0.5 and self.success_rate >= min_success * 0.75:
            action = "stake_moderate"
            reasoning = (f"Good confidence ({confidence:.2f}) and decent success rate ({self.success_rate:.2%}). "
                       "Consider moderate stakes.")
        elif confidence >= 0.3 and self.success_rate >= min_success * 0.5:
            action = "stake_minimal"
            reasoning = (f"Moderate confidence ({confidence:.2f}) with room for improvement. "
                       "Start with minimal stakes.")
        else:
            action = "train_more"
            if self.success_rate < min_success * 0.5:
                reasoning = f"Success rate ({self.success_rate:.2%}) too low. Additional training needed."
            else:
                reasoning = f"Insufficient confidence ({confidence:.2f}). More training recommended."
        
        return {
            "recommended_action": action,
            "confidence_score": confidence,
            "risk_level": self.staking_metrics["risk_level"],
            "recommended_stake": self.staking_metrics["recommended_stake"],
            "expected_multiplier": self.staking_metrics["expected_multiplier"],
            "optimal_target": self.staking_metrics["optimal_target_score"],
            "min_target": self.staking_metrics["min_target_score"],
            "reasoning": reasoning
        }

class ObservationProcessor:
    """Flexible observation processing for different games."""
    
    def __init__(self, obs_type: str = "grayscale", resize_shape: Tuple[int, int] = (84, 84)):
        """Initialize observation processor.
        
        Args:
            obs_type: Type of observation ("rgb", "grayscale", "ram")
            resize_shape: Shape to resize observations to
        """
        self.obs_type = obs_type
        self.resize_shape = resize_shape
    
    def process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process observation based on configuration."""
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # Handle different observation types
        if self.obs_type == "grayscale":
            if len(obs.shape) == 4:  # Vectorized observation
                if obs.shape[-1] == 3:  # RGB
                    obs = np.mean(obs, axis=-1, keepdims=True)
            elif len(obs.shape) == 3 and obs.shape[-1] == 3:  # Single RGB
                obs = np.mean(obs, axis=-1, keepdims=True)
        
        # Ensure correct shape and normalization
        if self.obs_type != "ram":
            obs = obs.astype(np.float32) / 255.0
        
        return obs

class VecObservationWrapper(VecEnvWrapper):
    """Enhanced wrapper for handling different observation types."""
    
    def __init__(self, venv, target_obs_space, processor: ObservationProcessor):
        """Initialize wrapper with observation processor."""
        super().__init__(venv)
        self.target_obs_space = target_obs_space
        self.observation_space = target_obs_space
        self.processor = processor
    
    def reset(self):
        """Reset environment with processed observation."""
        obs = self.venv.reset()
        if isinstance(obs, tuple):
            obs, info = obs
            return self.processor.process_observation(obs), info
        return self.processor.process_observation(obs)
    
    def step_wait(self):
        """Step environment with processed observation."""
        obs, reward, done, info = self.venv.step_wait()
        return self.processor.process_observation(obs), reward, done, info

# Game-specific configurations
GAME_CONFIGS = {
    "pong": GameSpecificConfig(
        game_id="ALE/Pong-v5",
        score_range=(-21, 21),
        success_threshold=5.0,  # Updated from 0 to 5.0 - a moderate positive score indicates decent play
        default_frame_stack=4,
        observation_type="grayscale",
        action_space=[0, 1, 2, 3, 4, 5]  # Pong uses a subset of actions
    ),
    "space_invaders": GameSpecificConfig(
        game_id="ALE/SpaceInvaders-v5",
        score_range=(0, 1000),
        success_threshold=100,
        default_frame_stack=4,
        observation_type="grayscale",
        action_space=[0, 1, 2, 3, 4]  # Space Invaders actions
    ),
    "riverraid": GameSpecificConfig(
        game_id="ALE/Riverraid-v5",
        score_range=(0, 100000),
        success_threshold=1000,
        default_frame_stack=4,
        observation_type="grayscale",
        action_space=[0, 1, 2, 3, 4, 5]  # River Raid actions
    )
}

def register_game_config(game_id: str, config: GameSpecificConfig) -> None:
    """Register a new game configuration."""
    GAME_CONFIGS[game_id] = config

class EvaluationPipeline:
    """Pipeline for evaluating trained models with focus on staking decisions."""
    
    def __init__(
        self,
        game: str,
        env: gym.Env,
        model: BaseAlgorithm,
        wallet: NEARWallet,
        leaderboard_manager: LeaderboardManager,
        config: Optional[EvaluationConfig] = None
    ):
        """Initialize evaluation pipeline."""
        self.game = game
        self.env = env
        self.model = model
        self.wallet = wallet
        self.leaderboard_manager = leaderboard_manager
        self.config = config or EvaluationConfig(game_id=game)
        
        # Create observation processor
        self.obs_processor = ObservationProcessor(
            obs_type=self.config.obs_type,
            resize_shape=self.config.preprocessing_config["resize_shape"]
        )
        
        # Validate configuration for staking
        if self.config.mode == "staking":
            valid, message = self.config.validate_for_staking()
            if not valid:
                logger.warning(f"Staking validation failed: {message}")
    
    def _prepare_environment(self, model_obs_space: gym.spaces.Box) -> gym.Env:
        """Prepare the environment for evaluation."""
        logger.debug(f"Original observation space: {self.env.observation_space}")
        
        # Apply a patch for environments that use WarpFrame and would cause
        # the "Invalid number of channels in input image" error
        env_id = None
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'spec'):
            env_id = self.env.unwrapped.spec.id
            
        # Check if this is an Atari environment with grayscale observation that needs special handling
        is_atari_grayscale = (
            env_id is not None and 
            ("ALE" in env_id or "Atari" in env_id) and 
            len(self.env.observation_space.shape) == 3 and
            self.env.observation_space.shape[-1] == 1
        )
        
        if is_atari_grayscale:
            logger.debug(f"Detected Atari grayscale environment: {env_id}")
            # Apply wrapper to prevent redundant grayscale conversion
            self.env = SkipGrayscaleConversionWrapper(self.env)
        
        # Apply channel-first transformations and normalization as needed
        obs_shape = self.env.observation_space.shape
        if len(obs_shape) == 3:
            # First check if we need to convert channel format
            channel_conversion_needed = False
            if obs_shape[-1] in [1, 3]:  # Last dimension is channels (channel-last format)
                channel_conversion_needed = True
                
            # Check if we need to normalize data types/ranges
            normalization_needed = False
            if self.env.observation_space.dtype != model_obs_space.dtype:
                normalization_needed = True
                
            # Apply wrapper if needed
            if channel_conversion_needed or normalization_needed:
                logger.debug(f"Applying observation space transformations: channel_format={channel_conversion_needed}, normalization={normalization_needed}")
                self.env = ChannelFirstWrapper(self.env, model_obs_space)
                logger.debug(f"Transformed observation space: {self.env.observation_space}")

        # If the environment is already vectorized, use it directly
        if isinstance(self.env, VecEnv):
            env = self.env
        else:
            env = DummyVecEnv([lambda: self.env])
        
        logger.debug(f"Observation space after vectorization: {env.observation_space}")

        # Stack frames if needed and not already stacked
        if self.config.frame_stack > 1 and not isinstance(env, VecFrameStack):
            # Use first channel order for PyTorch compatibility
            env = VecFrameStack(env, n_stack=self.config.frame_stack, channels_order='first')
            logger.debug(f"Observation space after stacking: {env.observation_space}")

        # Ensure observation space matches model's expectations
        if env.observation_space.shape != model_obs_space.shape:
            logger.debug(f"Adjusting environment observation space from {env.observation_space.shape} to {model_obs_space.shape}")
            env.observation_space = model_obs_space

        return env
    
    def _is_success(self, score: float, info: Dict[str, Any]) -> bool:
        """Determine if episode was successful based on game-specific criteria."""
        # First check if environment provides explicit success signal
        if isinstance(info, (list, tuple)):
            info = info[0]
        
        if isinstance(info, dict) and info.get("is_success", False):
            return True
        
        # For Atari games, use the score threshold from game configuration
        # Compare the raw episode score with the success threshold
        threshold = self.config.game_config.success_threshold
        
        # Debug logging to help diagnose issues
        if self.config.verbose > 1:
            logger.debug(f"Score: {score}, Threshold: {threshold}, Success: {score >= threshold}")
            
        # An episode is successful if the score is at least the threshold
        return score >= threshold
    
    def _calculate_confidence_score(self, rewards: List[float]) -> float:
        """Calculate confidence score based on performance consistency and absolute performance."""
        if not rewards:
            return 0.0
        
        reward_std = np.std(rewards)
        reward_mean = np.mean(rewards)
        
        # Get game-specific score range
        min_score, max_score = self.config.game_config.score_range
        score_range = max_score - min_score
        
        # Calculate consistency component (0 to 1)
        cv = reward_std / (score_range + 1e-8)  # Use score range instead of mean
        consistency = max(0.0, min(1.0, 1.0 - cv))
        
        # Calculate performance component (0 to 1)
        # Map mean score to [0, 1] range relative to game's score range
        performance = (reward_mean - min_score) / (score_range + 1e-8)
        performance = max(0.0, min(1.0, performance))
        
        # Combine consistency and performance
        # High confidence requires both consistency and good performance
        return consistency * performance
    
    def _assess_risk_level(self, rewards: List[float], success_rate: float) -> str:
        """Assess risk level based on performance metrics."""
        confidence = self._calculate_confidence_score(rewards)
        thresholds = self.config.game_config.staking_thresholds["risk_levels"]
        min_success = self.config.game_config.staking_thresholds["min_success_rate"]
        
        if confidence >= thresholds["low"] and success_rate >= min_success:
            return "low"
        elif confidence >= thresholds["medium"] and success_rate >= min_success * 0.5:
            return "medium"
        else:
            return "high"
    
    def _calculate_recommended_stake(self, rewards: List[float], success_rate: float) -> float:
        """Calculate recommended stake based on performance metrics."""
        confidence = self._calculate_confidence_score(rewards)
        base_stake = 5.0  # Base stake in NEAR
        
        # No stake if performance is too poor
        if confidence < 0.3 or success_rate < self.config.game_config.staking_thresholds["min_success_rate"] * 0.5:
            return 0.0
        
        # Calculate stake multiplier
        stake_multiplier = min(confidence * success_rate, 2.0)  # Cap at 2x
        return base_stake * stake_multiplier
    
    def _calculate_expected_multiplier(self, success_rate: float) -> float:
        """Calculate expected reward multiplier based on performance."""
        min_success = self.config.game_config.staking_thresholds["min_success_rate"]
        
        if success_rate >= min_success:
            return 2.0
        elif success_rate >= min_success * 0.75:
            return 1.5
        else:
            return 1.0
    
    def _calculate_stability_score(self, rewards: List[float]) -> float:
        """Calculate stability score based on reward consistency and performance level."""
        if not rewards or len(rewards) < 2:
            return 0.0
        
        # Get game-specific score range
        min_score, max_score = self.config.game_config.score_range
        score_range = max_score - min_score
        
        # Calculate trend stability
        window = min(10, len(rewards))  # Use fixed window size
        rolling_means = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        # Calculate stability component
        stability = 1.0 - np.std(rolling_means) / (score_range + 1e-8)
        stability = max(0.0, min(1.0, stability))
        
        # Calculate performance level component
        mean_score = np.mean(rewards)
        performance = (mean_score - min_score) / (score_range + 1e-8)
        performance = max(0.0, min(1.0, performance))
        
        # Combine stability and performance
        return stability * performance
    
    def _sign_verification_data(self, data: Dict[str, Any]) -> str:
        """Generate a signature for verification data using HMAC-SHA256.
        
        Args:
            data: Dictionary containing verification data
            
        Returns:
            Hexadecimal signature string
        """
        # Get or create a secret key from wallet config
        secret_key = self.wallet.config.get_secret_key()
        if not secret_key:
            # Generate and save a new secret key if not exists
            secret_key = secrets.token_hex(32)
            self.wallet.config.save_secret_key(secret_key)
        
        # Create a deterministic string representation of the data
        message = f"{data['game']}:{data['account_id']}:{data['score']:.4f}:{data['timestamp']}:{data['nonce']}"
        
        # Generate HMAC using SHA-256
        signature = hmac.new(
            secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _store_verification_token(self, data: Dict[str, Any], signature: str) -> Path:
        """Store verification token in local JSON file.
        
        Args:
            data: Dictionary containing verification data
            signature: Cryptographic signature for the data
            
        Returns:
            Path to the stored token file
        """
        verification_token = {
            'data': data,
            'signature': signature
        }
        
        # Use a consistent file path
        token_dir = Path.home() / '.agent-arcade' / 'verification_tokens'
        token_dir.mkdir(parents=True, exist_ok=True)
        
        # Use game and score to create a unique filename
        token_file = token_dir / f"{data['game']}_{data['score']:.4f}_{data['timestamp']}.json"
        
        with open(token_file, 'w') as f:
            json.dump(verification_token, f)
            
        return token_file
    
    def evaluate(self) -> EvaluationResult:
        """Run evaluation episodes with focus on staking metrics."""
        episode_rewards = []
        episode_lengths = []
        episode_metrics = []
        successes = 0
        
        # Track resource usage
        start_time = time.time()
        
        for i in range(self.config.n_eval_episodes):
            if self.config.verbose > 0:
                logger.info(f"Starting evaluation episode {i+1}/{self.config.n_eval_episodes}")
            
            # Handle both old and new gym reset API
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            
            # Ensure obs is a numpy array
            if isinstance(obs, tuple):
                obs = obs[0]  # Take first element if it's a tuple
            
            done = False
            episode_reward = 0
            episode_length = 0
            episode_specific_metrics = []
            
            while not done:
                # Model expects (n_stack, h, w) or (n_env, n_stack, h, w)
                action, _ = self.model.predict(obs, deterministic=self.config.deterministic)
                
                # Handle both old and new gym step API
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    # New gym API: obs, reward, terminated, truncated, info
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    # Old gym API: obs, reward, done, info
                    obs, reward, done, info = step_result
                    terminated = truncated = done
                
                # Ensure obs is a numpy array
                if isinstance(obs, tuple):
                    obs = obs[0]
                
                if self.config.render:
                    self.env.render()
                
                # Track metrics
                episode_reward += reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward
                episode_length += 1
                episode_specific_metrics.append(self._get_game_specific_metrics(info))
                
            
            # Check if the episode was successful based on final score
            if self._is_success(episode_reward, info):
                successes += 1
                if self.config.verbose > 0:
                    logger.debug(f"Episode {i+1} was successful with score {episode_reward}")
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_metrics.append(episode_specific_metrics)
            
            if self.config.verbose > 0:
                logger.info(f"Episode {i+1} finished with reward {episode_reward}")
        
        # Calculate evaluation time and FPS
        total_time = time.time() - start_time
        total_steps = sum(episode_lengths)
        fps = total_steps / total_time
        
        # Calculate core metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        success_rate = successes / self.config.n_eval_episodes
        
        # Get environment ID and metadata
        env_id = None
        if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
            if hasattr(self.env.envs[0].unwrapped, 'spec'):
                env_id = self.env.envs[0].unwrapped.spec.id
        
        # Prepare comprehensive metadata
        metadata = {
            "env_id": env_id,
            "model_class": self.model.__class__.__name__,
            "frame_stack": self.config.frame_stack,
            "frame_skip": self.config.frameskip or 4,  # ALE v5 default
            "sticky_actions": self.config.repeat_action_probability,
            "observation_size": self.config.preprocessing_config["resize_shape"],
            "fps": fps,
            "total_time": total_time,
            "mode": self.config.mode,
            **self.config.additional_config
        }
        
        # Calculate staking-focused metrics
        if self.config.mode == "staking":
            staking_metrics = {
                "confidence_score": self._calculate_confidence_score(episode_rewards),
                "risk_level": self._assess_risk_level(episode_rewards, success_rate),
                "recommended_stake": self._calculate_recommended_stake(episode_rewards, success_rate),
                "expected_multiplier": self._calculate_expected_multiplier(success_rate),
                "stability_score": self._calculate_stability_score(episode_rewards),
                "min_target_score": np.percentile(episode_rewards, 25),
                "optimal_target_score": np.percentile(episode_rewards, 75)
            }
        else:
            staking_metrics = None
        
        # Generate verification token if wallet is available
        verification_token = None
        if self.wallet and self.wallet.is_logged_in():
            # Create verification data
            verification_data = {
                'game': self.game,
                'account_id': self.wallet.config.account_id,
                'score': float(mean_reward),  # Convert numpy float to Python float
                'timestamp': int(time.time()),
                'nonce': secrets.token_hex(8),  # Random value for uniqueness
                'episodes': self.config.n_eval_episodes
            }
            
            # Generate signature
            signature = self._sign_verification_data(verification_data)
            
            # Store token
            token_path = self._store_verification_token(verification_data, signature)
            logger.debug(f"Verification token stored at: {token_path}")
            
            verification_token = {
                'data': verification_data,
                'signature': signature
            }
        
        return EvaluationResult(
            mean_reward=mean_reward,
            std_reward=std_reward,
            n_episodes=self.config.n_eval_episodes,
            success_rate=success_rate,
            episode_lengths=episode_lengths,
            episode_rewards=episode_rewards,
            metadata=metadata,
            game_config=self.config.game_config,
            staking_metrics=staking_metrics,
            verification_token=verification_token
        )
    
    def run_and_record(self, model_path: Path) -> EvaluationResult:
        """Run evaluation and record results with staking focus."""
        if not self.wallet.is_logged_in():
            raise ValueError("Must be logged in to record evaluation results")
        
        # Load model first to get its observation space
        temp_model = self.model.__class__.load(model_path)
        model_obs_space = temp_model.observation_space
        n_stack = model_obs_space.shape[0]  # Get frame stack from model's observation space
        logger.debug(f"Model observation space: {model_obs_space}")
        logger.debug(f"Using frame_stack={n_stack} from model")
        
        # Update config with model's frame stack
        self.config.frame_stack = n_stack
        
        # Now prepare environment with correct frame stack
        self.env = self._prepare_environment(model_obs_space)
        
        # Ensure observation space matches model's expectations
        if hasattr(self.env, 'observation_space') and self.env.observation_space != model_obs_space:
            logger.debug(f"Adjusting environment observation space to match model")
            from gymnasium import spaces
            self.env.observation_space = model_obs_space
        
        # Load model with properly configured environment
        self.model = self.model.__class__.load(model_path, env=self.env)
        
        # Run evaluation
        result = self.evaluate()
        
        # Record results if in staking mode
        if self.config.mode == "staking":
            # Convert numpy types to Python types for JSON serialization
            self.leaderboard_manager.record_score(
                game_name=self.game,
                account_id=self.wallet.config.account_id,
                score=float(result.mean_reward),  # Convert np.float32 to float
                success_rate=float(result.success_rate),  # Convert np.float32 to float
                episodes=int(result.n_episodes),  # Convert np.int64 to int if needed
                model_path=str(model_path)
            )
        
        return result
    
    def _get_game_specific_metrics(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Get game-specific metrics from environment info."""
        metrics = {}
        
        # Common ALE metrics
        if isinstance(info, (list, tuple)):
            info = info[0]  # Get first env's info
            
        if isinstance(info, dict):
            # Standard metrics
            metrics["lives"] = info.get("lives", None)
            metrics["score"] = info.get("score", None)
            
            # Game-specific success criteria
            metrics["is_success"] = info.get("is_success", False)
            
            # Additional game metrics
            for key in ["level", "time", "fuel", "health"]:
                if key in info:
                    metrics[key] = info[key]
        
        return metrics 

def analyze_staking(success_rate, mean_score, game_info):
    """Analyze staking suitability based on evaluation results.
    
    Args:
        success_rate: Success rate from evaluation (0-1)
        mean_score: Mean score from evaluation
        game_info: Game information object with score_range
        
    Returns:
        An object with risk_level and recommendation attributes
    """
    class StakingAnalysis:
        def __init__(self, risk_level, recommendation):
            self.risk_level = risk_level
            self.recommendation = recommendation
    
    min_score, max_score = game_info.get('score_range', (0, 100))
    normalized_score = (mean_score - min_score) / (max_score - min_score) if max_score > min_score else 0
    
    # Determine risk level
    if success_rate >= 0.8 and normalized_score >= 0.7:
        risk_level = "Low"
        recommendation = f"Safe to stake. Consider a target score of {mean_score * 0.9:.1f}."
    elif success_rate >= 0.5 and normalized_score >= 0.4:
        risk_level = "Medium"
        recommendation = f"Moderate risk. Consider a smaller stake with target score of {mean_score * 0.8:.1f}."
    else:
        risk_level = "High"
        recommendation = "High risk. Improve performance before staking."
    
    return StakingAnalysis(risk_level, recommendation) 