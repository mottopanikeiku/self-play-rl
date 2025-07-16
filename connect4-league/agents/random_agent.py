"""random agent

a baseline agent that selects random valid moves.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from .base import Agent


class RandomAgent(Agent):
    """agent that selects random valid moves."""
    
    def __init__(self, name: str = "Random", seed: Optional[int] = None) -> None:
        super().__init__(name)
        self.rng = np.random.RandomState(seed)
    
    def act(self, obs: np.ndarray) -> int:
        """select a random valid column."""
        # find valid columns (top row is empty)
        valid_cols = []
        for col in range(obs.shape[1]):
            if obs[0, col] == 0:  # top row is empty
                valid_cols.append(col)
        
        if not valid_cols:
            raise ValueError("No valid moves available")
        
        return self.rng.choice(valid_cols)
    
    def reset(self) -> None:
        """reset agent state."""
        pass


def build_agent(weights_path: Optional[Path] = None) -> Agent:
    """build a random agent.
    
    args:
        weights_path: ignored for random agent
        
    returns:
        randomagent instance
    """
    return RandomAgent() 