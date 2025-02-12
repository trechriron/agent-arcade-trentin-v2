"""Base interface for Agent Arcade games."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path

class GameInterface(ABC):
    """Base interface that all games must implement."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Game name."""
        pass
    
    @abstractmethod
    def train(self, render: bool = False, config_path: Optional[Path] = None) -> None:
        """Train an agent for this game."""
        pass
    
    @abstractmethod
    def evaluate(self, model_path: Path, episodes: int = 10) -> Dict[str, Any]:
        """Evaluate a trained model."""
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        pass
    
    @abstractmethod
    def get_score_range(self) -> tuple[float, float]:
        """Get the possible score range for this game."""
        pass
    
    @abstractmethod
    def validate_model(self, model_path: Path) -> bool:
        """Validate that a model file is valid for this game."""
        pass 