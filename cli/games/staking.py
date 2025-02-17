"""NEAR staking functionality for games."""
import json
import base64
from typing import Optional, Tuple
from pathlib import Path
from loguru import logger
import aiohttp

try:
    from cli.core.near import NEARWallet
    from cli.core.stake import StakeRecord
    NEAR_AVAILABLE = True
except ImportError:
    NEAR_AVAILABLE = False

class StakeManager:
    """Manages staking operations using NEAR RPC."""
    
    def __init__(self, network: str = "testnet"):
        self.network = network
        self.rpc_url = f"https://rpc.{network}.near.org"
        
    async def place_stake(self, 
                         account_id: str,
                         game_name: str,
                         model_path: Path, 
                         amount: float, 
                         target_score: float) -> bool:
        """Place a stake on game performance.
        
        Args:
            account_id: NEAR account ID
            game_name: Name of the game
            model_path: Path to the model to stake on
            amount: Amount to stake in NEAR
            target_score: Target score to achieve
            
        Returns:
            True if stake was placed successfully
        """
        try:
            # Convert NEAR to yoctoNEAR
            amount_yocto = int(amount * 10**24)
            
            # Prepare function call
            args = {
                "game": game_name,
                "target_score": target_score,
                "model_path": str(model_path)
            }
            
            # Encode arguments
            args_base64 = base64.b64encode(json.dumps(args).encode()).decode()
            
            # Make RPC call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.rpc_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": "dontcare",
                        "method": "query",
                        "params": {
                            "request_type": "call_function",
                            "finality": "final",
                            "account_id": "agent-arcade.testnet",
                            "method_name": "place_stake",
                            "args_base64": args_base64
                        }
                    }
                ) as response:
                    result = await response.json()
                    
                    if "error" in result:
                        logger.error(f"Failed to place stake: {result['error']}")
                        return False
                        
                    logger.info(f"Successfully placed stake of {amount} NEAR on {game_name}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to place stake: {e}")
            return False
            
async def stake_on_game(wallet: Optional['NEARWallet'], 
                       game_name: str,
                       model_path: Path, 
                       amount: float, 
                       target_score: float,
                       score_range: Tuple[float, float]) -> None:
    """Stake on a game's performance.
    
    Args:
        wallet: NEAR wallet instance
        game_name: Name of the game
        model_path: Path to the model to stake on
        amount: Amount to stake in NEAR
        target_score: Target score to achieve
        score_range: Valid score range for the game
    """
    if not NEAR_AVAILABLE:
        logger.error("NEAR integration is not available. Install with: pip install -e '.[staking]'")
        return
        
    if not wallet:
        logger.error("Wallet not initialized")
        return
        
    min_score, max_score = score_range
    if not min_score <= target_score <= max_score:
        logger.error(f"Target score {target_score} is outside valid range [{min_score}, {max_score}]")
        return
        
    try:
        # Create stake manager
        stake_manager = StakeManager(network=wallet.config.network)
        
        # Place stake
        success = await stake_manager.place_stake(
            account_id=wallet.config.account_id,
            game_name=game_name,
            model_path=model_path,
            amount=amount,
            target_score=target_score
        )
        
        if success:
            # Record stake locally
            stake_record = StakeRecord(
                game=game_name,
                model_path=str(model_path),
                amount=amount,
                target_score=target_score,
                status="pending"
            )
            wallet.record_stake(stake_record)
            logger.info(f"Successfully placed stake of {amount} NEAR on {game_name}")
        else:
            logger.error("Failed to place stake")
            
    except Exception as e:
        logger.error(f"Failed to place stake: {e}") 