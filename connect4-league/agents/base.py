"""base agent interface

defines the abstract interface that all connect-4 agents must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np


class Agent(ABC):
    """abstract base class for all connect-4 agents.
    
    all agents must implement the act() method to select actions given observations.
    stateful agents can override reset() for initialization between games.
    """
    
    def __init__(self, name: str) -> None:
        """initialize agent with a name.
        
        args:
            name: human-readable name for the agent
        """
        self.name = name
    
    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        """select an action given the current board observation.
        
        args:
            obs: board state as numpy array (6, 7) with values:
                 0 = empty, 1 = current player, 2 = opponent
        
        returns:
            column index (0-6) to drop piece
        """
        pass
    
    def reset(self) -> None:
        """reset agent state between games.
        
        called before each new game starts. stateful agents should
        override this to reset internal state.
        """
        pass
    
    def __str__(self) -> str:
        """string representation of the agent."""
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self) -> str:
        """string representation of the agent."""
        return self.__str__() 