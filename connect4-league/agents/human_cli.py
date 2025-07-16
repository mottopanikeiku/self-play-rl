"""human cli agent

allows human players to interact with the connect-4 system via command line.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from .base import Agent


class HumanCLIAgent(Agent):
    """human agent that gets moves from command line input."""
    
    def __init__(self, name: str = "Human") -> None:
        super().__init__(name)
    
    def act(self, obs: np.ndarray) -> int:
        """get move from human player via cli."""
        self._display_board(obs)
        
        # get valid moves
        valid_moves = [col for col in range(7) if obs[0, col] == 0]
        
        if not valid_moves:
            print("No valid moves available!")
            return 0
        
        print(f"Valid moves: {valid_moves}")
        
        while True:
            try:
                move_input = input("Enter your move (column 0-6): ").strip()
                
                if not move_input:
                    continue
                
                move = int(move_input)
                
                if move in valid_moves:
                    return move
                else:
                    print(f"Invalid move {move}. Valid moves are: {valid_moves}")
                    
            except ValueError:
                print("Please enter a valid number (0-6)")
            except KeyboardInterrupt:
                print("\nExiting...")
                return valid_moves[0] if valid_moves else 0
            except EOFError:
                print("\nNo input available, using random move")
                return np.random.choice(valid_moves) if valid_moves else 0
    
    def _display_board(self, obs: np.ndarray) -> None:
        """display the current board state."""
        print("\nCurrent board:")
        print(" " + " ".join(str(i) for i in range(7)))
        print("+" + "-" * 13 + "+")
        
        for row in range(6):
            print("|", end="")
            for col in range(7):
                piece = obs[row, col]
                if piece == 0:
                    print(" ", end="")
                elif piece == 1:
                    print("X", end="")  # current player
                else:
                    print("O", end="")  # opponent
                if col < 6:
                    print(" ", end="")
            print("|")
        
        print("+" + "-" * 13 + "+")
    
    def reset(self) -> None:
        """reset agent state."""
        pass


def build_agent(weights_path: Optional[Path] = None) -> Agent:
    """build a human cli agent.
    
    args:
        weights_path: ignored for human agent
        
    returns:
        humancliagent instance
    """
    return HumanCLIAgent() 