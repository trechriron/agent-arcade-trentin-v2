"""Staking functionality for Agent Arcade."""
from typing import Optional
from pydantic import BaseModel

class StakeRecord(BaseModel):
    """Record of a stake placed on an agent."""
    game: str
    model_path: str
    amount: float
    target_score: float
    status: str = "pending"  # pending, completed, claimed
    transaction_hash: Optional[str] = None
    achieved_score: Optional[float] = None
    reward_multiplier: Optional[float] = None 